# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import math
from dataclasses import dataclass
from typing import Any, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
import torch.types
from torch import nn
from torch.nn.utils import parametrizations
from tqdm import tqdm
from transformers import LlamaConfig, LlamaModel, PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.models.whisper.modeling_whisper import (
    WhisperAttention,
    WhisperConfig,
    WhisperEncoder,
)

from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternTokenPairs,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    MultimodalDataItem,
    MultimodalInputFormat,
    MultimodalInputs,
    flatten_nested_list,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.idefics2 import Idefics2VisionTransformer
from sglang.srt.models.minicpmv import MiniCPMBaseModel, Resampler2_5
from sglang.srt.models.qwen2 import Qwen2ForCausalLM
from sglang.srt.utils import get_device, logger

try:


    from transformers.generation import LogitsProcessor as LogitsWarper
    from vector_quantize_pytorch import GroupedResidualFSQ

    _tts_deps = True
except:
    LogitsWarper = None
    _tts_deps = False


def apply_spk_emb(
    input_ids: torch.Tensor = None,
    spk_emb: torch.Tensor = None,
    input_embeds: torch.Tensor = None,
    spk_emb_token_id: int = 0,
    num_spk_embs: int = 1,
):


    batch_size = input_ids.shape[0]

    for idx in range(batch_size):
        input_ids_ = input_ids[idx]
        spk_emb_ = spk_emb[idx]
        mask_ = input_ids_ == spk_emb_token_id
        nonzero_position_idx = mask_.nonzero(as_tuple=False)
        assert nonzero_position_idx.shape[0] == num_spk_embs
        begin_idx = nonzero_position_idx.min()
        end_idx = nonzero_position_idx.max()
        input_embeds[idx, begin_idx : end_idx + 1, :] = spk_emb_

    return


@dataclass
class ConditionalChatTTSGenerationOutput(ModelOutput):


    new_ids: torch.LongTensor = None
    audio_input_ids: torch.LongTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    finished: bool = None


def make_streaming_chunk_mask_generation(
    inputs_embeds: torch.Tensor,
    past_seen_tokens: int,
    streaming_tts_text_mask: torch.Tensor,
    streaming_reserved_length: int = 300,
    streaming_audio_chunk_size: int = 50,
    streaming_text_chunk_size: int = 10,
    num_spk_emb: int = 1,
    use_spk_emb: bool = True,
) -> torch.Tensor:

    assert inputs_embeds.shape[0] == 1

    dtype = inputs_embeds.dtype
    device = inputs_embeds.device
    min_dtype = torch.finfo(dtype).min


    causal_mask = torch.full(
        (1, past_seen_tokens + inputs_embeds.shape[1]),
        fill_value=0,
        dtype=dtype,
        device=device,
    )


    invisible_text_tokens_start = (
        min(
            math.ceil(
                (past_seen_tokens - streaming_reserved_length)
                / streaming_audio_chunk_size
            )
            * streaming_text_chunk_size,
            streaming_reserved_length,
        )
        + 1
        + num_spk_emb * use_spk_emb
    )

    invisible_text_tokens_end = (
        streaming_reserved_length + 1 + num_spk_emb * use_spk_emb + 1
    )


    causal_mask[0, invisible_text_tokens_start:invisible_text_tokens_end] = min_dtype


    causal_mask[
        0, 0 : 1 + num_spk_emb * use_spk_emb + streaming_reserved_length + 1
    ].masked_fill_(streaming_tts_text_mask == 0, min_dtype)


    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    return causal_mask


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        kernel: int,
        dilation: int,
        layer_scale_init_value: float = 1e-6,
    ):

        super().__init__()
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel,
            padding=dilation * (kernel // 2),
            dilation=dilation,
            groups=dim,
        )

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.coef = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor, cond=None) -> torch.Tensor:
        residual = x

        y = self.dwconv(x)
        y.transpose_(1, 2)
        x = self.norm(y)
        del y
        y = self.pwconv1(x)
        del x
        x = self.act(y)
        del y
        y = self.pwconv2(x)
        del x
        if self.coef is not None:
            y *= self.coef
        y.transpose_(1, 2)

        x = y + residual
        del y

        return x


class DVAEDecoder(nn.Module):
    def __init__(
        self,
        idim: int,
        odim: int,
        n_layer=12,
        bn_dim=64,
        hidden=256,
        kernel=7,
        dilation=2,
        up=False,
    ):
        super().__init__()
        self.up = up
        self.conv_in = nn.Sequential(
            nn.Conv1d(idim, bn_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv1d(bn_dim, hidden, 3, 1, 1),
        )
        self.decoder_block = nn.ModuleList(
            [
                ConvNeXtBlock(
                    hidden,
                    hidden * 4,
                    kernel,
                    dilation,
                )
                for _ in range(n_layer)
            ]
        )
        self.conv_out = nn.Conv1d(hidden, odim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, conditioning=None) -> torch.Tensor:

        y = self.conv_in(x)
        del x
        for f in self.decoder_block:
            y = f(y, conditioning)

        x = self.conv_out(y)
        del y
        return x


class GFSQ(nn.Module):
    def __init__(
        self,
        dim: int,
        levels: List[int],
        G: int,
        R: int,
        eps=1e-5,
        transpose=True,
    ):
        super(GFSQ, self).__init__()
        self.quantizer = GroupedResidualFSQ(
            dim=dim,
            levels=list(levels),
            num_quantizers=R,
            groups=G,
        )
        self.n_ind = math.prod(levels)
        self.eps = eps
        self.transpose = transpose
        self.G = G
        self.R = R

    def _embed(self, x: torch.Tensor):
        if self.transpose:
            x = x.transpose(1, 2)
        x = x.view(x.size(0), x.size(1), self.G, self.R).permute(2, 0, 1, 3)
        feat = self.quantizer.get_output_from_indices(x)
        return feat.transpose_(1, 2) if self.transpose else feat

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.transpose:
            x.transpose_(1, 2)
        _, ind = self.quantizer(x)
        ind = ind.permute(1, 2, 0, 3).contiguous()
        ind = ind.view(ind.size(0), ind.size(1), -1)
        return ind.transpose_(1, 2) if self.transpose else ind


class DVAE(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        coef = torch.rand(100)
        self.coef = nn.Parameter(coef.unsqueeze(0).unsqueeze_(2))

        self.downsample_conv = nn.Sequential(
            nn.Conv1d(100, 512, 3, 1, 1),
            nn.GELU(),
            nn.Conv1d(512, 512, 4, 2, 1),
            nn.GELU(),
        )

        self.encoder = DVAEDecoder(
            idim=512,
            odim=1024,
            hidden=256,
            n_layer=12,
            bn_dim=128,
        )

        self.decoder = DVAEDecoder(
            idim=512,
            odim=512,
            hidden=256,
            n_layer=12,
            bn_dim=128,
        )

        self.out_conv = nn.Conv1d(512, 100, 3, 1, 1, bias=False)

        self.vq_layer = GFSQ(
            dim=1024,
            levels=(5, 5, 5, 5),
            G=2,
            R=2,
        )

    @torch.inference_mode()
    def forward(
        self, inp: torch.Tensor, mode: Literal["encode", "decode"] = "decode"
    ) -> torch.Tensor:
        if mode == "encode" and hasattr(self, "encoder") and self.vq_layer is not None:
            mel = inp.clone()
            x: torch.Tensor = self.downsample_conv(
                torch.div(mel, self.coef.view(100, 1).expand(mel.shape), out=mel),
            ).unsqueeze_(0)
            del mel
            x = self.encoder(x)
            ind = self.vq_layer(x)
            del x
            return ind

        if self.vq_layer is not None:
            vq_feats = self.vq_layer._embed(inp)
        else:
            vq_feats = inp

        vq_feats = (
            vq_feats.view(
                (vq_feats.size(0), 2, vq_feats.size(1) // 2, vq_feats.size(2)),
            )
            .permute(0, 2, 3, 1)
            .flatten(2)
        )

        dec_out = self.out_conv(
            self.decoder(
                x=vq_feats,
            ),
        )

        del vq_feats

        return torch.mul(dec_out, self.coef, out=dec_out)


class CustomRepetitionPenaltyLogitsProcessorRepeat:
    def __init__(self, penalty: float, max_input_ids: int, past_window: int):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(
                f"`penalty` has to be a strictly positive float, but is {penalty}"
            )

        self.penalty = penalty
        self.max_input_ids = max_input_ids
        self.past_window = past_window

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if input_ids.size(1) > self.past_window:
            input_ids = input_ids.narrow(1, -self.past_window, self.past_window)
        freq = F.one_hot(input_ids, scores.size(1)).sum(1)
        if freq.size(0) > self.max_input_ids:
            freq.narrow(
                0, self.max_input_ids, freq.size(0) - self.max_input_ids
            ).zero_()
        alpha = torch.pow(self.penalty, freq)
        scores = scores.contiguous()
        inp = scores.multiply(alpha)
        oth = scores.divide(alpha)
        con = scores < 0
        out = torch.where(con, inp, oth)
        del inp, oth, scores, con, alpha
        return out


class ConditionalChatTTS(PreTrainedModel):


    config_class = PretrainedConfig
    _no_split_modules = []

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

        self.use_speaker_embedding = config.use_speaker_embedding
        self.use_llm_hidden_state = config.use_llm_hidden_state
        self.num_spk_embs = config.num_spk_embs
        self.spk_emb_token_id = config.spk_emb_token_id

        self.use_text = config.use_text
        self.streaming = config.streaming
        self.streaming_text_chunk_size = config.streaming_text_chunk_size
        self.streaming_audio_chunk_size = config.streaming_audio_chunk_size
        self.streaming_text_reserved_len = config.streaming_text_reserved_len
        self.audio_bos_token_id = config.audio_bos_token_id
        self.num_mel_bins = config.num_mel_bins
        self.num_vq = config.num_vq
        self.num_audio_tokens = config.num_audio_tokens

        self.top_p = config.top_p
        self.top_k = config.top_k
        self.repetition_penalty = config.repetition_penalty

        if self.config.use_mlp:
            self.projector = MultiModalProjector(config.llm_dim, config.hidden_size)
        else:
            self.projector = nn.Linear(config.llm_dim, config.hidden_size, bias=False)
        self.emb_code = nn.ModuleList(
            [
                nn.Embedding(config.num_audio_tokens, config.hidden_size)
                for _ in range(config.num_vq)
            ]
        )
        self.emb_text = nn.Embedding(config.num_text_tokens, config.hidden_size)
        self.head_code = nn.ModuleList(
            [
                parametrizations.weight_norm(
                    nn.Linear(config.hidden_size, config.num_audio_tokens, bias=False),
                    name="weight",
                )
                for _ in range(config.num_vq)
            ]
        )

        dvae = DVAE()
        self.dvae = dvae

        model_config = LlamaConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            num_hidden_layers=config.num_hidden_layers,
            max_position_embeddings=config.max_position_embeddings,
            attn_implementation=config.attn_implementation,
        )

        model = LlamaModel(model_config)
        self.model = model

    @torch.inference_mode()
    def merge_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        lm_spk_emb_last_hidden_states: Optional[torch.Tensor] = None,
    ):

        assert input_ids.shape[0] == 1


        inputs_embeds = self.emb_text(input_ids)


        if self.use_speaker_embedding:
            spk_emb_mask = input_ids == self.spk_emb_token_id
            if spk_emb_mask.any():
                assert lm_spk_emb_last_hidden_states is not None

                lm_spk_emb_last_hidden_states = lm_spk_emb_last_hidden_states.to(
                    self.projector.linear1.weight.dtype
                )
                projected_spk_emb = self.projector(lm_spk_emb_last_hidden_states)
                projected_spk_emb = F.normalize(projected_spk_emb, p=2, dim=-1)
                apply_spk_emb(
                    input_ids=input_ids,
                    spk_emb=projected_spk_emb,
                    input_embeds=inputs_embeds,
                    spk_emb_token_id=self.spk_emb_token_id,
                    num_spk_embs=self.num_spk_embs,
                )
        else:
            raise NotImplementedError

        return inputs_embeds

    @torch.inference_mode()
    def prefill_text(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        lm_spk_emb_last_hidden_states: Optional[torch.Tensor] = None,
    ):

        assert input_ids.shape[0] == 1
        assert past_key_values is not None


        inputs_embeds = self.merge_inputs_embeds(
            input_ids=input_ids,
            lm_spk_emb_last_hidden_states=lm_spk_emb_last_hidden_states,
        )


        past_key_values_for_prefill = []
        for i in range(len(past_key_values)):
            past_key_values_for_prefill.append(
                (
                    past_key_values[i][0][:, :, : position_ids[:, 0], :].clone(),
                    past_key_values[i][1][:, :, : position_ids[:, 0], :].clone(),
                )
            )


        outputs_prefill: BaseModelOutputWithPast = self.model(
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=past_key_values_for_prefill,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            output_attentions=False,
            cache_position=position_ids,
        )


        past_key_values_for_prefill_updated = outputs_prefill.past_key_values


        for layer_idx in range(len(past_key_values)):

            past_key_values[layer_idx][0][
                :, :, position_ids[:, 0] : position_ids[:, -1] + 1, :
            ] = past_key_values_for_prefill_updated[layer_idx][0][
                :, :, position_ids[:, 0] : position_ids[:, -1] + 1
            ].clone()

            past_key_values[layer_idx][1][
                :, :, position_ids[:, 0] : position_ids[:, -1] + 1, :
            ] = past_key_values_for_prefill_updated[layer_idx][1][
                :, :, position_ids[:, 0] : position_ids[:, -1] + 1
            ].clone()


        return past_key_values

    @torch.inference_mode()
    def prefill_audio_ids(
        self,
        input_ids: torch.Tensor,
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        streaming_tts_text_mask=None,
        add_audio_bos: bool = True,
    ):

        assert input_ids.shape[0] == 1
        assert past_key_values is not None

        code_emb = [self.emb_code[i](input_ids[:, :, i]) for i in range(self.num_vq)]
        inputs_embeds = torch.stack(code_emb, 3).sum(3)
        input_len = input_ids.shape[1]

        if add_audio_bos:
            narrowed_input_ids = torch.tensor(
                [[self.audio_bos_token_id]], dtype=torch.long, device=self.device
            )
            bos_inputs_embeds = self.emb_text(narrowed_input_ids)
            inputs_embeds = torch.cat([bos_inputs_embeds, inputs_embeds], dim=1)
            input_len += 1

        past_key_values_length = past_key_values[0][0].shape[2]
        position_ids = torch.arange(
            past_key_values_length,
            past_key_values_length + input_len,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)

        cache_position = position_ids.clone()
        causal_mask = make_streaming_chunk_mask_generation(
            inputs_embeds=inputs_embeds,
            past_seen_tokens=past_key_values[0][0].shape[2],
            streaming_tts_text_mask=streaming_tts_text_mask,
            streaming_reserved_length=self.streaming_text_reserved_len,
            streaming_text_chunk_size=self.streaming_text_chunk_size,
        )


        outputs: BaseModelOutputWithPast = self.model(
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            output_attentions=False,
            cache_position=cache_position,
        )
        past_key_values = outputs.past_key_values
        return past_key_values

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        temperature: torch.Tensor,
        eos_token: Union[int, torch.Tensor],
        streaming_tts_text_mask=None,
        force_no_stop=False,
        min_new_token=10,
        max_new_token=50,
        logits_warpers: Optional[List[LogitsWarper]] = None,
        logits_processors: Optional[
            List[CustomRepetitionPenaltyLogitsProcessorRepeat]
        ] = None,
        show_tqdm=False,
    ):


        assert input_ids.shape[0] == 1
        assert past_key_values is not None

        logits_warpers = logits_warpers or []
        logits_processors = logits_processors or []


        start_idx = (
            1
            + self.num_spk_embs * self.use_speaker_embedding
            + self.streaming_text_reserved_len
            + 1
        )

        finish = torch.zeros(input_ids.shape[0], device=input_ids.device).bool()

        temperature = (
            temperature.unsqueeze(0)
            .expand(input_ids.shape[0], -1)
            .contiguous()
            .view(-1, 1)
        )

        progress = input_ids.shape[1]


        input_ids_buf = torch.zeros(
            input_ids.shape[0],
            progress
            + max_new_token,
            input_ids.shape[2],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )


        input_ids_buf.narrow(1, 0, progress).copy_(input_ids)

        del input_ids
        input_ids = input_ids_buf.narrow(1, 0, progress)

        pbar: Optional[tqdm] = None
        if show_tqdm:
            pbar = tqdm(
                total=max_new_token,
                desc="code",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}(max) [{elapsed}, {rate_fmt}{postfix}]",
            )

        condition_length = (
            1
            + self.num_spk_embs * self.use_speaker_embedding
            + self.streaming_text_reserved_len
            + 1
        )

        for i in range(max_new_token):

            audio_bos = False


            if progress == condition_length:
                audio_bos = True

            assert progress == (
                past_key_values[0][0].shape[2] + 1
            )

            if audio_bos:


                narrowed_input_ids = torch.tensor(
                    [[self.audio_bos_token_id]], dtype=torch.long, device=self.device
                )
                inputs_embeds = self.emb_text(narrowed_input_ids)
                del narrowed_input_ids
            else:


                narrowed_input_ids = input_ids.narrow(
                    dim=1, start=input_ids.shape[1] - 1, length=1
                )
                code_emb = [
                    self.emb_code[i](narrowed_input_ids[:, :, i])
                    for i in range(self.num_vq)
                ]
                inputs_embeds = torch.stack(code_emb, 3).sum(3)

            position_ids = torch.tensor(
                [past_key_values[0][0].shape[2]], dtype=torch.long, device=self.device
            ).unsqueeze(0)

            cache_position = position_ids.clone()


            causal_mask = make_streaming_chunk_mask_generation(
                inputs_embeds=inputs_embeds,
                past_seen_tokens=past_key_values[0][0].shape[2],
                streaming_tts_text_mask=streaming_tts_text_mask,
                streaming_reserved_length=self.streaming_text_reserved_len,
                streaming_text_chunk_size=self.streaming_text_chunk_size,
            )


            outputs: BaseModelOutputWithPast = self.model(
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=True,
                output_attentions=False,
                cache_position=cache_position,
            )

            del position_ids
            del inputs_embeds
            del cache_position
            del causal_mask

            hidden_states = outputs.last_hidden_state
            past_key_values = outputs.past_key_values

            with P.cached():
                logits = torch.empty(
                    hidden_states.size(0),
                    hidden_states.size(1),
                    self.num_audio_tokens,
                    self.num_vq,
                    dtype=torch.float,
                    device=self.device,
                )
                for num_vq_iter in range(self.num_vq):
                    x: torch.Tensor = self.head_code[num_vq_iter](hidden_states)
                    logits[..., num_vq_iter] = x
                    del x

            del hidden_states


            logits = logits.narrow(1, -1, 1).squeeze_(1).float()


            logits = logits.permute(0, 2, 1)
            logits = logits.reshape(-1, logits.size(2))

            input_ids_sliced = input_ids.narrow(
                1,
                start_idx,
                input_ids.size(1) - start_idx,
            ).permute(0, 2, 1)
            logits_token = input_ids_sliced.reshape(
                input_ids_sliced.size(0) * input_ids_sliced.size(1),
                -1,
            ).to(self.device)
            del input_ids_sliced

            logits /= temperature

            if not audio_bos:
                for logitsProcessors in logits_processors:
                    logits = logitsProcessors(logits_token, logits)
            if not audio_bos:
                for logitsWarpers in logits_warpers:
                    logits = logitsWarpers(logits_token, logits)

            del logits_token

            if i < min_new_token:
                logits[:, eos_token] = -torch.inf

            if force_no_stop:
                logits[:, eos_token] = -torch.inf

            scores = F.softmax(logits, dim=-1)

            del logits
            idx_next = torch.multinomial(scores, num_samples=1)

            del scores


            idx_next = idx_next.view(-1, self.num_vq)
            finish_or = idx_next.eq(eos_token).any(1)
            finish.logical_or_(finish_or)

            del finish_or

            input_ids_buf.narrow(1, progress, 1).copy_(idx_next.unsqueeze_(1))

            if i == 0 and finish.any():

                break

            del idx_next
            progress += 1
            input_ids = input_ids_buf.narrow(1, 0, progress)

            if finish.all():
                break

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        if not finish.all():
            if show_tqdm:
                logger.info(f"incomplete result. hit max_new_token: {max_new_token}")

        del input_ids_buf

        if finish.all():

            genrated_input_ids = input_ids[:, condition_length:-1, :]
        else:

            genrated_input_ids = input_ids[:, condition_length:, :]

        return ConditionalChatTTSGenerationOutput(
            new_ids=genrated_input_ids,
            audio_input_ids=input_ids,
            past_key_values=past_key_values,
            finished=finish.all(),
        )

    @torch.inference_mode()
    def decode_to_mel_specs(
        self,
        result_list: List[torch.Tensor],
    ):


        decoder = self.dvae
        max_x_len = -1
        if len(result_list) == 0:
            return np.array([], dtype=np.float32)
        for result in result_list:
            if result.size(0) > max_x_len:
                max_x_len = result.size(0)
        batch_result = torch.zeros(
            (len(result_list), result_list[0].size(1), max_x_len),
            dtype=result_list[0].dtype,
            device=result_list[0].device,
        )
        for i in range(len(result_list)):
            src = result_list[i]
            batch_result[i].narrow(1, 0, src.size(0)).copy_(src.permute(1, 0))
            del src

        mel_specs = decoder(batch_result)
        del batch_result
        return mel_specs


class MiniCPMWhisperEncoderLayer(nn.Module):
    def __init__(self, config: WhisperConfig, layer_idx: int = None):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
            layer_idx=layer_idx,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
        past_key_values: Optional[EncoderDecoderCache] = None,
        use_cache: Optional[bool] = False,
    ) -> torch.Tensor:

        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)


        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            past_key_value=past_key_values,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=False
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=False
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=False
        )
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        if use_cache:
            outputs += (past_key_values,)

        return outputs


class MiniCPMWhisperEncoder(WhisperEncoder):

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [
                MiniCPMWhisperEncoderLayer(config, layer_idx=i)
                for i in range(config.encoder_layers)
            ]
        )

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        use_cache: Optional[bool] = None,
    ):

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )


        input_features = input_features.to(
            dtype=self.conv1.weight.dtype, device=self.conv1.weight.device
        )

        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)

        embed_pos = self.embed_positions.weight
        past_key_values_length = 0
        if use_cache:
            if past_key_values is None:
                past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
            elif isinstance(past_key_values, list):
                past_key_values = EncoderDecoderCache(
                    DynamicCache.from_legacy_cache(past_key_values), DynamicCache()
                )
            elif isinstance(past_key_values, DynamicCache):
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache())
            else:
                pass
            past_key_values_length = (
                past_key_values.self_attention_cache.get_usable_length(
                    inputs_embeds.shape[1]
                )
            )
            if inputs_embeds.shape[1] + past_key_values_length > embed_pos.shape[0]:
                logger.warning(
                    "seems the audio is longer than 30s. repeating the last part of the audio"
                )
                embed_pos_front = embed_pos[past_key_values_length:, :]
                embed_pos = torch.cat(
                    (
                        embed_pos_front,
                        torch.repeat_interleave(
                            embed_pos[-1, :].unsqueeze(0),
                            inputs_embeds.shape[1]
                            - embed_pos.shape[0]
                            + past_key_values_length,
                            dim=0,
                        ),
                    )
                )
            else:
                embed_pos = embed_pos[
                    past_key_values_length : inputs_embeds.shape[1]
                    + past_key_values_length,
                    :,
                ]
        else:
            embed_pos = embed_pos[: inputs_embeds.shape[1], :]

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=False
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None


        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            to_drop = False


            if to_drop:
                layer_outputs = (None, None)
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    output_attentions=output_attentions,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                )

                hidden_states = layer_outputs[0]

            if use_cache:
                next_encoder_cache = layer_outputs[2 if output_attentions else 1]
            else:
                next_encoder_cache = None

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, encoder_states, all_attentions]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
            past_key_values=next_encoder_cache,
        )


class MultiModalProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_dim, out_features=out_dim, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=out_dim, out_features=out_dim, bias=True)

    def forward(self, audio_features):
        hidden_states = self.relu(self.linear1(audio_features))
        hidden_states = self.linear2(hidden_states)
        return hidden_states


class MiniCPMO(MiniCPMBaseModel):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__(config=config, quant_config=quant_config)

        self.llm = self.init_llm(config=config, quant_config=quant_config)

        self.embed_dim = self.llm.config.hidden_size


        if self.config.init_vision:

            self.vpm = self.init_vision_module(config=config, quant_config=quant_config)
            self.vision_dim = self.vpm.embed_dim
            self.resampler = self.init_resampler(self.embed_dim, self.vision_dim)


        self.config.init_audio = True
        if self.config.init_audio:

            self.apm = self.init_audio_module()
            audio_output_dim = int(self.apm.config.encoder_ffn_dim // 4)
            self.audio_avg_pooler = nn.AvgPool1d(
                self.config.audio_pool_step, stride=self.config.audio_pool_step
            )
            self.audio_projection_layer = MultiModalProjector(
                in_dim=audio_output_dim, out_dim=self.embed_dim
            )
            self.audio_encoder_layer = -1


        self.config.init_tts = False
        logger.info("TTS is disabled for now")
        if self.config.init_tts:

            assert (
                _tts_deps
            ), "please make sure vector_quantize_pytorch and vocos are installed."
            self.tts = self.init_tts_module()

    def init_tts_module(self):
        model = ConditionalChatTTS(self.config.tts_config)
        return model

    def init_audio_module(self):
        model = MiniCPMWhisperEncoder(self.config.audio_config)
        return model

    def init_llm(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> nn.Module:
        return Qwen2ForCausalLM(config=config, quant_config=quant_config, prefix=prefix)

    def init_vision_module(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
    ):
        if self.config._attn_implementation == "flash_attention_2":
            self.config.vision_config._attn_implementation = "flash_attention_2"
        else:
            self.config.vision_config._attn_implementation = "eager"
        model = Idefics2VisionTransformer(
            config=config.vision_config, quant_config=quant_config, prefix=prefix
        )
        if self.config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]

        setattr(model, "embed_dim", model.embeddings.embed_dim)
        setattr(model, "patch_size", model.embeddings.patch_size)

        return model

    def init_resampler(
        self,
        embed_dim: int,
        vision_dim: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> nn.Module:
        with set_default_torch_dtype(torch.float16):

            resampler = Resampler2_5(
                num_queries=self.config.query_num,
                embed_dim=embed_dim,
                num_heads=embed_dim // 128,
                kv_dim=vision_dim,
                quant_config=quant_config,
                prefix=prefix,
            )

        return resampler.to(device=get_device(), dtype=torch.get_default_dtype())

    def pad_input_ids(self, input_ids: List[int], mm_input: MultimodalInputs):

        im_start_id: int = mm_input.im_start_id
        im_end_id: int = mm_input.im_end_id
        slice_start_id: int = mm_input.slice_start_id
        slice_end_id: int = mm_input.slice_end_id

        data_token_pairs = [
            (im_start_id, im_end_id),
            (slice_start_id, slice_end_id),
            (mm_input.audio_start_id, mm_input.audio_end_id),
        ]
        data_start_token_ids = [im_start_id, mm_input.audio_start_id]
        pattern = MultiModalityDataPaddingPatternTokenPairs(
            data_token_pairs=data_token_pairs, data_start_token_ids=data_start_token_ids
        )

        return pattern.pad_input_tokens(input_ids, mm_input)

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):

        input_lengths_after_cnn = (input_lengths - 1) // 2 + 1
        input_lengths_after_pooling = (
            input_lengths_after_cnn - self.config.audio_pool_step
        ) // self.config.audio_pool_step + 1
        input_lengths_after_pooling = input_lengths_after_pooling.to(dtype=torch.int32)

        return input_lengths_after_cnn, input_lengths_after_pooling

    def get_audio_embedding_streaming(self, items: List[MultimodalDataItem]):

        wavforms = flatten_nested_list([item.feature for item in items if item.feature])

        audio_feature_lens_raw = flatten_nested_list(
            [item.audio_feature_lens for item in items if item.audio_feature_lens]
        )


        if len(wavforms) > 0:
            audio_feature_lens = torch.hstack(audio_feature_lens_raw)
            batch_size, _, max_mel_seq_len = wavforms.shape
            assert batch_size == 1
            max_seq_len = (max_mel_seq_len - 1) // 2 + 1

            if self.audio_past_key_values is not None:
                cache_length = self.audio_past_key_values[0][0].shape[2]
                apm_max_len = self.apm.embed_positions.weight.shape[0]
                if cache_length + max_seq_len >= apm_max_len:
                    logger.warning(
                        f"audio_past_key_values length {cache_length + max_seq_len} exceed {apm_max_len}, reset."
                    )
                    self.audio_past_key_values = None

            audio_outputs = self.apm(
                wavforms, past_key_values=self.audio_past_key_values, use_cache=True
            )
            audio_states = (
                audio_outputs.last_hidden_state
            )
            self.audio_past_key_values = audio_outputs.past_key_values

            audio_embeds = self.audio_projection_layer(audio_states)

            audio_embeds = audio_embeds.transpose(1, 2)
            audio_embeds = self.audio_avg_pooler(audio_embeds)
            audio_embeds = audio_embeds.transpose(1, 2)

            _, feature_lens_after_pooling = self._get_feat_extract_output_lengths(
                audio_feature_lens
            )

            num_audio_tokens = feature_lens_after_pooling

            final_audio_embeds = []
            idx = 0
            for i in range(len(audio_feature_lens_raw)):
                target_audio_embeds = []
                for _ in range(len(audio_feature_lens_raw[i])):
                    target_audio_embeds.append(
                        audio_embeds[idx, : num_audio_tokens[idx], :]
                    )
                    idx += 1
                final_audio_embeds.append(target_audio_embeds)
            return final_audio_embeds
        else:
            return []

    def subsequent_chunk_mask(
        self,
        size: int,
        chunk_size: int,
        num_left_chunks: int = -1,
        device: torch.device = torch.device("cpu"),
        num_lookhead: int = 0,
    ) -> torch.Tensor:

        ret = torch.zeros(size, size, device=device, dtype=torch.bool)
        for i in range(size):
            if num_left_chunks < 0:
                start = 0
            else:
                start = max((i // chunk_size - num_left_chunks) * chunk_size, 0)
            ending = min((i // chunk_size + 1) * chunk_size + num_lookhead, size)
            ret[i, start:ending] = True
        return ret

    def get_audio_embedding(self, items: List[MultimodalDataItem], chunk_length=-1):


        wavforms = flatten_nested_list([item.feature for item in items if item.feature])

        audio_feature_lens_raw = flatten_nested_list(
            [item.audio_feature_lens for item in items if item.audio_feature_lens]
        )


        if audio_feature_lens_raw:
            if isinstance(audio_feature_lens_raw[0], torch.Tensor):

                audio_feature_lens_raw = [[lens] for lens in audio_feature_lens_raw]
            elif isinstance(audio_feature_lens_raw[0], list):


                flattened = []
                for item in audio_feature_lens_raw:
                    if isinstance(item, list):
                        flattened.extend(item)
                    else:
                        flattened.append(item)
                audio_feature_lens_raw = [
                    [item] if not isinstance(item, list) else item for item in flattened
                ]

        final_audio_embeds = []

        assert isinstance(wavforms, list)
        assert isinstance(wavforms[0], torch.Tensor)

        for wavform in wavforms:
            if len(wavform) > 0:

                flattened_lens = []
                for item in audio_feature_lens_raw:
                    if isinstance(item, list):
                        flattened_lens.extend(item)
                    else:
                        flattened_lens.append(item)
                audio_feature_lens = torch.hstack(flattened_lens)
                batch_size, _, max_mel_seq_len = wavform.shape
                max_seq_len = (max_mel_seq_len - 1) // 2 + 1


                seq_range = (
                    torch.arange(
                        0,
                        max_seq_len,
                        dtype=audio_feature_lens.dtype,
                        device=audio_feature_lens.device,
                    )
                    .unsqueeze(0)
                    .expand(batch_size, max_seq_len)
                )
                lengths_expand = audio_feature_lens.unsqueeze(1).expand(
                    batch_size, max_seq_len
                )

                padding_mask = seq_range >= lengths_expand

                audio_attention_mask_ = padding_mask.view(
                    batch_size, 1, 1, max_seq_len
                ).expand(batch_size, 1, max_seq_len, max_seq_len)
                audio_attention_mask = audio_attention_mask_.to(
                    dtype=self.apm.conv1.weight.dtype,
                    device=self.apm.conv1.weight.device,
                )

                if chunk_length > 0:
                    chunk_num_frame = int(chunk_length * 50)
                    chunk_mask = self.subsequent_chunk_mask(
                        size=max_seq_len,
                        chunk_size=chunk_num_frame,
                        num_left_chunks=-1,
                        device=audio_attention_mask_.device,
                    )
                    audio_attention_mask_ = torch.logical_or(
                        audio_attention_mask_, torch.logical_not(chunk_mask)
                    )

                audio_attention_mask[audio_attention_mask_] = float("-inf")
                audio_states = self.apm(
                    wavform,
                    output_hidden_states=True,
                    attention_mask=audio_attention_mask,
                ).hidden_states[self.audio_encoder_layer]
                audio_embeds = self.audio_projection_layer(audio_states)

                audio_embeds = audio_embeds.transpose(1, 2)
                audio_embeds = self.audio_avg_pooler(audio_embeds)
                audio_embeds = audio_embeds.transpose(1, 2)

                _, feature_lens_after_pooling = self._get_feat_extract_output_lengths(
                    audio_feature_lens
                )

                num_audio_tokens = feature_lens_after_pooling

                idx = 0
                for i in range(len(audio_feature_lens_raw)):
                    target_audio_embeds = []
                    for _ in range(len(audio_feature_lens_raw[i])):
                        target_audio_embeds.append(
                            audio_embeds[idx, : num_audio_tokens[idx], :]
                        )
                        idx += 1
                    final_audio_embeds.append(target_audio_embeds)
            return final_audio_embeds

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        embedding = self.get_omni_embedding(
            items=items,
            chunk_length=self.config.audio_chunk_length,
            stream_input=False,
        )
        return embedding

    def get_omni_embedding(
        self,
        items: List[MultimodalDataItem],
        chunk_length=-1,
        stream_input=False,
    ):


        if stream_input:
            audio_embeddings = self.get_audio_embedding_streaming(items)
        else:
            audio_embeddings = self.get_audio_embedding(items, chunk_length)
        bs = len(audio_embeddings)

        audio_embs = torch.cat(flatten_nested_list(audio_embeddings), dim=0)

        return audio_embs

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        if items and items[0].format == MultimodalInputFormat.PRECOMPUTED_EMBEDDING:
            result = torch.cat([item.feature for item in items])
            return result.reshape(-1, result.shape[-1])


        pixel_values = flatten_nested_list([item.feature for item in items])
        tgt_sizes = torch.stack(
            flatten_nested_list([item.tgt_size for item in items]), dim=0
        )
        assert len(pixel_values) == tgt_sizes.shape[0]

        device = self.vpm.embeddings.position_embedding.weight.device
        dtype = self.vpm.embeddings.position_embedding.weight.dtype
        all_pixel_values_lst = [
            i.flatten(end_dim=1).permute(1, 0) for i in pixel_values
        ]

        max_patches = (tgt_sizes[:, 0] * tgt_sizes[:, 1]).max().item()
        assert isinstance(max_patches, int)
        all_pixel_values = torch.nn.utils.rnn.pad_sequence(
            all_pixel_values_lst, batch_first=True, padding_value=0.0
        )

        B, L, _ = all_pixel_values.shape
        all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)
        patch_attn_mask = torch.zeros(
            (B, 1, max_patches), dtype=torch.bool, device=device
        )

        tgt_sizes_tensor = tgt_sizes.clone().to(device=patch_attn_mask.device)
        mask_shapes = tgt_sizes_tensor[:, 0] * tgt_sizes_tensor[:, 1]
        patch_attn_mask[:, 0, :] = torch.arange(
            patch_attn_mask.size(2), device=patch_attn_mask.device
        ).unsqueeze(0) < mask_shapes.unsqueeze(1)

        vision_embedding = self.vpm(
            all_pixel_values.type(dtype),
            patch_attention_mask=patch_attn_mask,
            tgt_sizes=tgt_sizes,
        )
        return self.resampler(vision_embedding, tgt_sizes)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ) -> torch.Tensor:

        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.llm,
            multimodal_model=self,
            positions=positions,
        )
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [

            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:

            if "rotary_emb.inv_freq~" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:


                continue


            if self.config.init_tts and "tts" in name:

                if ".weight_g" in name or ".weight_v" in name:
                    name = name.replace(
                        ".weight_g", ".parametrizations.weight.original0"
                    )
                    name = name.replace(
                        ".weight_v", ".parametrizations.weight.original1"
                    )
                elif ".weight" in name and name not in params_dict:
                    param_name = name.replace(
                        ".weight", ".parametrizations.weight.original0"
                    )
                    if param_name in params_dict:
                        name = param_name


            if "vpm" in name:
                name = name.replace(r"self_attn.out_proj", r"self_attn.proj")

            if not self.config.init_tts and "tts" in name:
                continue
            if not self.config.init_audio and ("apm" in name or "audio" in name):
                continue
            if not self.config.init_vision and "vpm" in name:
                continue

            if (
                "sampler" in name
                or "apm" in name
                or ("tts" in name and "self_attn" in name)
                or ("tts.model.layers" in name and ".mlp" in name)
            ):
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:

                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:

                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = [MiniCPMO]
