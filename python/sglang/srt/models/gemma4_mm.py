# Copyright 2025 SGLang Team
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


import logging
import re
from functools import lru_cache
from typing import Iterable, List, Optional, Set, Tuple, TypedDict, Union

import torch
from torch import nn
from transformers import (
    Gemma4AudioConfig,
    Gemma4Config,
    Gemma4TextConfig,
    Gemma4VisionConfig,
    PreTrainedModel,
)

from sglang.srt.distributed import get_pp_group
from sglang.srt.environ import envs
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.layernorm import Gemma4RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    flatten_nested_list,
)
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.gemma4_audio import Gemma4AudioEncoder
from sglang.srt.models.gemma4_causal import (
    Gemma4TextModel,
    load_tied_lm_head,
    pp_filter_load_weight,
)
from sglang.srt.models.gemma4_vision import Gemma4VisionEncoder
from sglang.srt.utils import add_prefix, cpu_has_amx_support, is_cpu
from sglang.srt.utils.hf_transformers_utils import get_processor

logger = logging.getLogger(__name__)

_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()

cached_get_processor = lru_cache(get_processor)


class Gemma4ImagePixelInputs(TypedDict):
    pixel_values: torch.Tensor
    """Shape: `(batch_size * num_images, num_channels, height, width)`"""


class Gemma4AudioInputs(TypedDict):
    input_features_padded: torch.Tensor
    """Shape: `(batch_size * num_audio, seq_length, num_features)`"""
    input_features_mask: torch.Tensor
    """Shape: `(batch_size * num_audio, seq_length)`"""


class Gemma4MultimodalEmbedder(nn.Module):


    def __init__(
        self,
        multimodal_config: Union[Gemma4AudioConfig, Gemma4VisionConfig],
        text_config: Gemma4TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        self.eps = multimodal_config.rms_norm_eps
        self.text_hidden_size = text_config.hidden_size


        embedding_dim = (
            getattr(multimodal_config, "output_proj_dims", None)
            or multimodal_config.hidden_size
        )

        self.embedding_projection = ReplicatedLinear(
            embedding_dim,
            self.text_hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("embedding_projection", prefix),
        )

        self.embedding_pre_projection_norm = Gemma4RMSNorm(
            embedding_dim,
            eps=self.eps,
            with_scale=False,
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:

        embs_normed = self.embedding_pre_projection_norm(inputs_embeds)
        embs_proj, _ = self.embedding_projection(embs_normed)
        return embs_proj


class Gemma4ForConditionalGeneration(PreTrainedModel):
    config_class = Gemma4Config
    """Gemma4 multimodal model for conditional generation."""


    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }


    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]

    embedding_modules = {}
    embedding_padding_modules = []
    supports_lora = True

    def __init__(
        self,
        config: Gemma4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config)
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config

        text_config = config.text_config

        prefix = add_prefix("model", prefix)


        if self.pp_group.is_first_rank:
            self.vision_tower = Gemma4VisionEncoder(
                config=config.vision_config,
                quant_config=quant_config,
                prefix=add_prefix("vision_tower", prefix),
            )
            self.embed_vision = Gemma4MultimodalEmbedder(
                config.vision_config,
                config.text_config,
                quant_config=quant_config,
                prefix=add_prefix("embed_vision", prefix),
            )
            if getattr(config, "audio_config", None) is not None:
                self.audio_tower = Gemma4AudioEncoder(
                    config=config.audio_config,
                    quant_config=quant_config,
                    prefix=add_prefix("audio_tower", prefix),
                )
                self.embed_audio = Gemma4MultimodalEmbedder(
                    config.audio_config,
                    config.text_config,
                    quant_config=quant_config,
                    prefix=add_prefix("embed_audio", prefix),
                )
            else:
                self.audio_tower = None
                self.embed_audio = None
        else:
            self.vision_tower = PPMissingLayer()
            self.embed_vision = PPMissingLayer()
            self.audio_tower = None
            self.embed_audio = None

        self.vocab_size = config.text_config.vocab_size
        self.vocab_size_per_layer_input = getattr(
            config.text_config,
            "vocab_size_per_layer_input",
            config.text_config.vocab_size,
        )


        self.language_model = Gemma4TextModel(
            config.text_config,
            quant_config,
            prefix=add_prefix("language_model", prefix),
        )


        text_tie = getattr(text_config, "tie_word_embeddings", True)
        self.lm_head_is_tied = (
            self.pp_group.world_size == 1
            and text_tie
            and not (_is_cpu and _is_cpu_amx_available)
        )
        if self.lm_head_is_tied:
            self.lm_head = self.language_model.embed_tokens
        elif self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(
                text_config.vocab_size,
                text_config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
        else:
            self.lm_head = PPMissingLayer()


        self.logits_processor = LogitsProcessor(config.text_config)
        self.capture_aux_hidden_states = False

        self.post_init()

    @property
    def model(self):


        return self.language_model

    def __setattr__(self, name, value):


        if name == "model":
            return
        super().__setattr__(name, value)

    def pad_input_ids(
        self,
        input_ids: List[int],
        mm_inputs: MultimodalInputs,
    ) -> List[int]:

        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.get_input_embeddings()

    def get_embed_and_head(self) -> Tuple[torch.Tensor, torch.Tensor]:

        embed = self.language_model.embed_tokens.weight
        return embed, embed

    def get_attention_sliding_window_size(self):
        return getattr(self.config.text_config, "sliding_window", -1) - 1

    def set_dflash_layers_to_capture(self, layer_ids: List[int]):
        if layer_ids is None:
            raise ValueError(
                "DFLASH requires explicit layer_ids for aux hidden capture."
            )
        self.capture_aux_hidden_states = True
        self.language_model.layers_to_capture = [val + 1 for val in layer_ids]

    def prepare_attn_masks(
        self,
        forward_batch: ForwardBatch,
        input_ids: torch.Tensor,
        mask_dtype: torch.dtype,
    ):

        if not isinstance(get_attn_backend(), TritonAttnBackend):
            logger.warning_once(
                "Bidirectional attention for image tokens requires TritonAttnBackend. "
                "Falling back to causal attention, which may degrade image quality."
            )
            return
        assert forward_batch.forward_mode == ForwardMode.EXTEND

        bidirectional_attn_masks_list = []
        bidirectional_attn_mask_indptr = torch.zeros(
            forward_batch.batch_size + 1, dtype=torch.int64, device=input_ids.device
        )

        split_images = []

        for i in range(forward_batch.batch_size):
            extend_seq_len = forward_batch.extend_seq_lens[i]
            prefix_len = forward_batch.extend_prefix_lens[i]
            bidirectional_attn_mask = torch.zeros(
                extend_seq_len,
                extend_seq_len + prefix_len,
                dtype=mask_dtype,
                device=input_ids.device,
            )

            bidirectional_attn_mask.fill_(1)
            bidirectional_attn_mask = bidirectional_attn_mask.tril(diagonal=prefix_len)


            mm_inputs = forward_batch.mm_inputs[i]
            if mm_inputs is not None:
                for mm_item in mm_inputs.mm_items:
                    if mm_item.is_image():
                        for im_begin, im_end in mm_item.offsets:


                            if (
                                im_begin >= prefix_len
                                and im_end < prefix_len + extend_seq_len
                            ):
                                bidirectional_attn_mask[
                                    im_begin - prefix_len : im_end + 1 - prefix_len,
                                    im_begin : im_end + 1,
                                ] = 1
                            elif (
                                im_end >= prefix_len
                                and im_begin < prefix_len + extend_seq_len
                            ):
                                split_images.append((i, im_begin, im_end))

            bidirectional_attn_masks_list.append(bidirectional_attn_mask.flatten())
            bidirectional_attn_mask_indptr[i + 1] = (
                bidirectional_attn_mask_indptr[i] + bidirectional_attn_mask.nelement()
            )
        if split_images:
            num_split_images = len(split_images)
            logger.warning_once(
                f"{num_split_images} images are split across chunk boundaries. "
                "Below are the first 5 images that are split across chunk boundaries: "
            )
            for i, im_begin, im_end in split_images[:5]:
                logger.warning_once(
                    f"Image {i}:{im_begin}-{im_end} is split across chunk boundaries.\n",
                )
            logger.warning_once(
                "Those images will receive causal attention. Disable chunked prefill (--chunked-prefill-size=-1) for full bidirectional attention.",
            )
        if bidirectional_attn_masks_list:
            bidirectional_attn_masks = torch.cat(bidirectional_attn_masks_list, dim=0)
            get_attn_backend().forward_metadata.mask_indptr = (
                bidirectional_attn_mask_indptr
            )
            get_attn_backend().forward_metadata.custom_mask = bidirectional_attn_masks

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        vt = self.vision_tower

        all_embeds = []
        for item in items:
            all_pixel_values = flatten_nested_list([item.feature])
            all_position_ids = flatten_nested_list(
                [getattr(item, "image_position_ids", None)]
            )

            for pv_idx, pv in enumerate(all_pixel_values):
                if (
                    pv.dim() in (2, 3)
                    and pv.shape[-1] == self.config.text_config.hidden_size
                ):
                    all_embeds.append(pv.to(self.language_model.device))
                    continue

                if pv_idx >= len(all_position_ids) or all_position_ids[pv_idx] is None:
                    raise ValueError(
                        f"pixel_values[{pv_idx}] has no matching image_position_ids. "
                        "The HF image processor likely renamed this output — "
                        "update ATTR_NAME_TO_MODALITY in the Gemma4 processor."
                    )
                pp = all_position_ids[pv_idx]


                if pv.dim() == 2:
                    pv = pv.unsqueeze(0)
                if pp.dim() == 2:
                    pp = pp.unsqueeze(0)

                pv = pv.to(device=vt.device, dtype=self.language_model.dtype())
                pp = pp.to(device=vt.device)

                pooled, pooler_mask = vt(pv, pp)

                for hs, mask in zip(pooled, pooler_mask):
                    real_tokens = hs[mask]
                    all_embeds.append(
                        self.embed_vision(
                            inputs_embeds=real_tokens.unsqueeze(0)
                        ).squeeze(0)
                    )

        if all_embeds:
            return torch.cat(all_embeds, dim=0)
        else:
            return torch.empty(
                0,
                self.language_model.config.hidden_size,
                device=next(self.parameters()).device,
                dtype=self.language_model.dtype(),
            )

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:

        vt = self.vision_tower

        all_embeds = []
        for item in items:
            all_pixel_values = flatten_nested_list([item.feature])
            all_position_ids = flatten_nested_list(
                [getattr(item, "video_position_ids", None)]
            )

            for pv_idx, pv in enumerate(all_pixel_values):
                if (
                    pv.dim() in (2, 3)
                    and pv.shape[-1] == self.config.text_config.hidden_size
                ):
                    all_embeds.append(pv.to(self.language_model.device))
                    continue

                if pv_idx >= len(all_position_ids) or all_position_ids[pv_idx] is None:
                    raise ValueError(
                        f"pixel_values_videos[{pv_idx}] has no matching video_position_ids."
                    )
                pp = all_position_ids[pv_idx]


                if pv.dim() == 4:
                    pv = pv.reshape(-1, pv.shape[-2], pv.shape[-1])
                if pp.dim() == 4:
                    pp = pp.reshape(-1, pp.shape[-2], pp.shape[-1])

                pv = pv.to(device=vt.device, dtype=self.language_model.dtype())
                pp = pp.to(device=vt.device)

                pooled, pooler_mask = vt(pv, pp)

                for hs, mask in zip(pooled, pooler_mask):
                    real_tokens = hs[mask]
                    all_embeds.append(
                        self.embed_vision(
                            inputs_embeds=real_tokens.unsqueeze(0)
                        ).squeeze(0)
                    )

        if all_embeds:
            return torch.cat(all_embeds, dim=0)
        else:
            return torch.empty(
                0,
                self.language_model.config.hidden_size,
                device=next(self.parameters()).device,
                dtype=self.language_model.dtype(),
            )

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        if self.audio_tower is None:
            raise ValueError(
                "Audio inputs provided but the model does not have an audio tower."
            )

        all_input_features = flatten_nested_list([item.feature for item in items])
        all_input_features_mask = flatten_nested_list(
            [~item.input_features_mask for item in items]
        )

        all_embeds = []
        for input_features, input_features_mask in zip(
            all_input_features, all_input_features_mask
        ):
            if input_features.dim() == 2:
                input_features = input_features.unsqueeze(0)
            if input_features_mask.dim() == 1:
                input_features_mask = input_features_mask.unsqueeze(0)

            input_features = input_features.to(
                device=self.audio_tower.device,
                dtype=self.language_model.dtype(),
            )
            input_features_mask = input_features_mask.to(device=input_features.device)


            audio_encodings, audio_mask = self.audio_tower(
                input_features, input_features_mask
            )

            audio_features = self.embed_audio(inputs_embeds=audio_encodings)

            for enc, mask in zip(audio_features, audio_mask):
                all_embeds.append(enc[~mask])

        if all_embeds:
            return torch.cat(all_embeds, dim=0)
        else:
            return torch.empty(
                0,
                self.language_model.config.hidden_size,
                device=next(self.parameters()).device,
                dtype=self.language_model.dtype(),
            )

    def get_per_layer_inputs(
        self, input_ids: torch.LongTensor
    ) -> Optional[torch.Tensor]:
        return self.language_model.get_per_layer_inputs(input_ids)

    def project_per_layer_inputs(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.language_model.project_per_layer_inputs(
            inputs_embeds, per_layer_inputs
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        **kwargs: object,
    ) -> Union[LogitsProcessor, PPProxyTensors]:

        is_first_rank = self.pp_group.is_first_rank
        is_last_rank = self.pp_group.is_last_rank


        if is_first_rank and (input_ids is None) ^ (input_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if envs.SGLANG_GEMMA_OUT_OF_PLACE_POSITION_MUTATION.get():
            positions = positions + 1
        else:
            positions += 1

        per_layer_inputs = None


        if is_first_rank and input_ids is not None:
            ple_ids = input_ids.clone()
            pad_id = self.config.text_config.pad_token_id

            ple_ids = torch.where(
                input_ids == self.config.image_token_id, pad_id, ple_ids
            )
            ple_ids = torch.where(
                input_ids == self.config.video_token_id, pad_id, ple_ids
            )
            ple_ids = torch.where(
                input_ids == self.config.audio_token_id, pad_id, ple_ids
            )
            per_layer_inputs = self.get_per_layer_inputs(ple_ids)


        if (
            forward_batch.forward_mode == ForwardMode.EXTEND
            and forward_batch.contains_image_inputs()
        ):
            self.prepare_attn_masks(
                forward_batch,
                input_ids,
                mask_dtype=torch.bool,
            )


        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={
                Modality.IMAGE: self.get_image_feature,
                Modality.VIDEO: self.get_video_feature,
                Modality.AUDIO: self.get_audio_feature,
            },
            positions=positions,
            per_layer_inputs=per_layer_inputs,
            pp_proxy_tensors=pp_proxy_tensors,
            **kwargs,
        )

        if not is_last_rank:


            return hidden_states


        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        head = (
            self.language_model.embed_tokens if self.lm_head_is_tied else self.lm_head
        )
        return self.logits_processor(
            input_ids,
            hidden_states,
            head,
            forward_batch,
            aux_hidden_states,
        )

    def tie_weights(self, recompute_mapping=False):


        if self.pp_group.world_size > 1:
            return
        return self.language_model.tie_weights()


    stacked_params_mapping = [

        (".qkv_proj", ".q_proj", "q"),
        (".qkv_proj", ".k_proj", "k"),
        (".qkv_proj", ".v_proj", "v"),
        (".gate_up_proj", ".up_proj", 1),
        (".gate_up_proj", ".gate_proj", 0),
    ]


    _RE_TOWER_QKV = re.compile(
        r"(.+\.(?:self_attn|attn))\.(q_proj|k_proj|v_proj)\.(.*)"
    )

    _RE_TOWER_GATE_UP = re.compile(r"(.+\.mlp)\.(gate_proj|up_proj)\.(.*)")

    _RE_AUDIO_LAYER = re.compile(r"(audio_tower)\.layers\.(\d+)\.(.*)")

    @staticmethod
    def _remap_audio_tower_name(name: str) -> str:

        if "audio_tower." not in name:
            return name


        name = name.replace(
            "subsample_conv_projection.layer0.",
            "subsample_conv_projection.conv_0.",
        )
        name = name.replace(
            "subsample_conv_projection.layer1.",
            "subsample_conv_projection.conv_1.",
        )


        m = Gemma4ForConditionalGeneration._RE_AUDIO_LAYER.match(name)
        if m:
            tower, layer_idx, suffix = m.groups()


            suffix = suffix.replace(
                "self_attn.relative_k_proj.",
                "attention.attn.relative_position_embedding.pos_proj.",
            )

            suffix = suffix.replace("self_attn.post.", "attention.post.")

            suffix = suffix.replace("self_attn.", "attention.attn.")

            suffix = suffix.replace("norm_pre_attn.", "attention.pre_attn_norm.")
            suffix = suffix.replace("norm_post_attn.", "attention.post_norm.")
            suffix = suffix.replace("norm_out.", "norm.")

            suffix = suffix.replace("feed_forward1.", "ffw_layer_start.")
            suffix = suffix.replace("feed_forward2.", "ffw_layer_end.")

            name = f"{tower}.conformer.{layer_idx}.{suffix}"

        return name

    @staticmethod
    def _remap_tower_name(name: str, params_dict: dict) -> str:


        m = Gemma4ForConditionalGeneration._RE_TOWER_QKV.match(name)
        if m:
            pfx, proj, attr = m.groups()
            if attr in ("weight", "bias", "linear.weight", "linear.bias"):
                bare_attr = attr.rsplit(".", 1)[-1]
                return f"{pfx}.qkv.{proj}.{bare_attr}"
            if attr.startswith("output_"):
                return f"{pfx}.qkv.{proj[0]}_{attr}"
            if attr.startswith("input_"):
                return f"{pfx}.qkv.{attr}"


        m = Gemma4ForConditionalGeneration._RE_TOWER_GATE_UP.match(name)
        if m:
            pfx, proj, attr = m.groups()
            short = proj.split("_")[0]
            if attr in ("weight", "bias", "linear.weight", "linear.bias"):
                bare_attr = attr.rsplit(".", 1)[-1]
                return f"{pfx}.gate_up.{proj}.{bare_attr}"
            if attr.startswith("output_"):
                return f"{pfx}.gate_up.{short}_{attr}"
            if attr.startswith("input_"):
                return f"{pfx}.gate_up.{attr}"


        if name.endswith(".weight") or name.endswith(".bias"):
            base, attr = name.rsplit(".", 1)
            alt = f"{base}.linear.{attr}"
            if alt in params_dict:
                return alt

        return name

    def _get_k_eq_v_layers(self) -> set:

        text_config = self.config.text_config
        if not getattr(text_config, "attention_k_eq_v", False):
            return set()
        return {
            i for i, lt in enumerate(text_config.layer_types) if lt == "full_attention"
        }

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        k_eq_v_layers = self._get_k_eq_v_layers()

        num_experts = getattr(self.config.text_config, "num_experts", 0) or 0
        expert_params_mapping = [


            ("experts.w13_weight", "experts.gate_up_proj", ("w1", "w3")),
            ("experts.w2_weight", "experts.down_proj", ("w2",)),
        ]


        per_expert_params_mapping = (
            FusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=num_experts,
            )
            if num_experts
            else []
        )

        params_dict = dict(self.named_parameters())
        params_dict.update(dict(self.named_buffers()))
        non_persistent_buffers: Set[str] = set()
        for mod_name, mod in self.named_modules():
            for buf_name in getattr(mod, "_non_persistent_buffers_set", set()):
                full = f"{mod_name}.{buf_name}" if mod_name else buf_name
                non_persistent_buffers.add(full)

        text_tie = getattr(self.config.text_config, "tie_word_embeddings", True)
        start_layer = self.language_model.start_layer
        end_layer = self.language_model.end_layer

        loaded_params: Set[str] = set()

        for name, loaded_weight in weights:
            if "embed_vision.embedding." in name or "embed_audio.embedding." in name:
                continue
            if self.audio_tower is None and (
                "audio_tower." in name or "embed_audio." in name
            ):
                continue

            name = re.sub(r"^model\.", "", name)

            if pp_filter_load_weight(
                name,
                loaded_weight,
                pp_group=self.pp_group,
                start_layer=start_layer,
                end_layer=end_layer,
                params_dict=params_dict,
                loaded_params=loaded_params,
                tie_word_embeddings=text_tie,
                embed_weight_name="language_model.embed_tokens.weight",
                first_rank_only_patterns=(
                    "language_model.embed_tokens",
                    "language_model.per_layer_model_projection",
                    "language_model.per_layer_projection_norm",
                    "vision_tower.",
                    "embed_vision.",
                    "audio_tower.",
                    "embed_audio.",
                ),
                last_rank_only_prefixes=("language_model.norm.", "lm_head."),
            ):
                continue


            name = name.replace(".router.per_expert_scale", ".moe.per_expert_scale")
            if ".experts." in name and ".moe.experts." not in name:
                name = name.replace(".experts.", ".moe.experts.")


            if "audio_tower." in name:
                name = self._remap_audio_tower_name(name)


            if "vision_tower." in name or "audio_tower." in name:
                name = self._remap_tower_name(name, params_dict)


            should_dup_k_to_v = (
                ".k_proj." in name
                and k_eq_v_layers
                and "language_model." in name
                and (m := re.search(r"layers\.(\d+)\.", name)) is not None
                and int(m.group(1)) in k_eq_v_layers
            )


            orig_name = name


            for (
                param_name,
                weight_name,
                expert_id,
                shard_id,
            ) in per_expert_params_mapping:
                if weight_name not in orig_name:
                    continue
                name = orig_name.replace(weight_name, param_name)
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(
                    param,
                    loaded_weight,
                    name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                )
                loaded_params.add(name)
                break
            else:


                for param_name, weight_name, shard_ids in expert_params_mapping:
                    name = orig_name
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    for i in range(num_experts):
                        chunks = loaded_weight[i].chunk(len(shard_ids), dim=0)
                        for chunk, sid in zip(chunks, shard_ids):
                            weight_loader(param, chunk, name, sid, i)
                    loaded_params.add(name)
                    break
                else:
                    for (
                        param_name,
                        weight_name,
                        shard_id,
                    ) in self.stacked_params_mapping:
                        name = orig_name
                        if weight_name not in name:
                            continue
                        name = name.replace(weight_name, param_name)
                        if name not in params_dict:
                            continue
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        weight_loader(param, loaded_weight, shard_id)
                        if should_dup_k_to_v:
                            weight_loader(param, loaded_weight, "v")
                        loaded_params.add(name)
                        break
                    else:
                        name = orig_name
                        if name.endswith(".bias") and name not in params_dict:
                            continue
                        name = maybe_remap_kv_scale_name(name, params_dict)
                        if name is None:
                            continue
                        if name not in params_dict:
                            continue
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                        if (
                            text_tie
                            and not self.lm_head_is_tied
                            and name == "language_model.embed_tokens.weight"
                        ):
                            load_tied_lm_head(
                                loaded_weight,
                                params_dict=params_dict,
                                loaded_params=loaded_params,
                            )
                        loaded_params.add(name)
        unloaded_params = params_dict.keys() - loaded_params
        if unloaded_params:
            param_names = set(dict(self.named_parameters()).keys())
            buckets = {
                logging.WARNING: (
                    "Some weights are not initialized from checkpoints",
                    lambda p: p in param_names,
                ),
                logging.INFO: (
                    "Persistent buffers not in checkpoint (using default init)",
                    lambda p: p not in param_names and p not in non_persistent_buffers,
                ),
                logging.DEBUG: (
                    "Non-persistent buffers not in checkpoint (expected)",
                    lambda p: p in non_persistent_buffers,
                ),
            }
            for level, (msg, pred) in buckets.items():
                names = sorted(p for p in unloaded_params if pred(p))
                if names:
                    logger.log(level, "%s: %s", msg, names)
        return loaded_params

    lora_pattern = re.compile(
        r"^language_model\.layers\.(\d+)\.(?:self_attn|mlp)\.(?:qkv_proj|o_proj|down_proj|gate_up_proj)"
    )

    def should_apply_lora(self, module_name: str) -> bool:
        return bool(self.lora_pattern.match(module_name))

    def get_hidden_dim(self, module_name, layer_idx):

        if module_name == "qkv_proj":
            return (
                self.config.hidden_size,
                self.config.head_dim
                * (
                    self.config.num_attention_heads
                    + self.config.num_key_value_heads * 2
                ),
            )
        elif module_name == "o_proj":
            return (
                self.config.head_dim * self.config.num_attention_heads,
                self.config.hidden_size,
            )
        elif module_name == "gate_up_proj":
            assert len(set(self.config.intermediate_size)) == 1, (
                "Currently SGLang requires uniform intermediate size for all layers. "
                "Please file an issue if you need support for non-uniform intermediate sizes."
            )
            return self.config.hidden_size, self.config.intermediate_size[0] * 2
        elif module_name == "down_proj":
            assert len(set(self.config.intermediate_size)) == 1, (
                "Currently SGLang requires uniform intermediate size for all layers. "
                "Please file an issue if you need support for non-uniform intermediate sizes."
            )
            return self.config.intermediate_size[0], self.config.hidden_size
        else:
            raise NotImplementedError()

    def get_embed(self):
        return self.language_model.embed_tokens.weight

    def get_embed_and_head(self):
        if self.pp_group.world_size > 1:


            raise NotImplementedError(
                "get_embed_and_head() is not implemented for Gemma4 "
                "multimodal under pipeline parallelism. embed_tokens lives "
                "on the first PP rank and lm_head on the last; use "
                "--pp-size 1 if you need this API."
            )
        embed = self.language_model.embed_tokens.weight

        return embed, embed

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        self.capture_aux_hidden_states = True
        text_config = self.config.text_config
        if layer_ids is None:
            num_layers = text_config.num_hidden_layers
            self.language_model.layers_to_capture = [
                2,
                num_layers // 2,
                num_layers - 3,
            ]
        else:


            self.language_model.layers_to_capture = [val + 1 for val in layer_ids]


EntryClass = Gemma4ForConditionalGeneration
