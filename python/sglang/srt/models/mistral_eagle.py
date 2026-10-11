# Copyright 2023-2026 SGLang Team
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
from collections.abc import Iterable
from typing import Optional, Tuple

import regex as re
import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import RowParallelLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.llama import LlamaDecoderLayer, LlamaForCausalLM
from sglang.srt.models.llama_eagle import LlamaForCausalLMEagle
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class MistralEagleModel(nn.Module):


    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        assert (
            get_pp_group().world_size == 1
        ), "MistralForCausalLMEagle currently does not support pipeline parallelism"
        self.pp_group = get_pp_group()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(
                    config=config,
                    layer_id=i,
                    prefix=add_prefix(f"layers.{i}", prefix),
                    quant_config=quant_config,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.start_layer = 0
        self.end_layer = config.num_hidden_layers
        self.fc = RowParallelLinear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("fc", prefix),
            input_is_parallel=False,
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds


        hidden_states, _ = self.fc(
            torch.cat(
                (hidden_states, forward_batch.spec_info.hidden_states),
                dim=-1,
            )
        )

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual
            )
        return hidden_states + residual


class MistralForCausalLMEagle(LlamaForCausalLMEagle):


    remapping = {
        r"layers\.(\d+)\.attention_norm\.weight": r"model.layers.\1.input_layernorm.weight",
        r"layers\.(\d+)\.attention\.wq\.(\w+)": r"model.layers.\1.self_attn.q_proj.\2",
        r"layers\.(\d+)\.attention\.wk\.(\w+)": r"model.layers.\1.self_attn.k_proj.\2",
        r"layers\.(\d+)\.attention\.wv\.(\w+)": r"model.layers.\1.self_attn.v_proj.\2",
        r"layers\.(\d+)\.attention\.wo\.(\w+)": r"model.layers.\1.self_attn.o_proj.\2",
        r"layers\.(\d+)\.ffn_norm\.weight": r"model.layers.\1.post_attention_layernorm.weight",
        r"layers\.(\d+)\.feed_forward\.w1\.(\w+)": r"model.layers.\1.mlp.gate_proj.\2",
        r"layers\.(\d+)\.feed_forward\.w2\.(\w+)": r"model.layers.\1.mlp.down_proj.\2",
        r"layers\.(\d+)\.feed_forward\.w3\.(\w+)": r"model.layers.\1.mlp.up_proj.\2",
        r"norm\.weight": "model.norm.weight",


        r"eagle_linear\.weight": r"model.fc.weight",
        r"eagle_linear\.qscale_act": r"model.fc.input_scale",
        r"eagle_linear\.qscale_weight": r"model.fc.weight_scale",


    }


    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:


        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        self.model = MistralEagleModel(
            config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):


        return LlamaForCausalLM.load_weights(
            self, self._remap_mistral_to_llama(weights)
        )

    def _remap_mistral_to_llama(
        self, weights: Iterable[Tuple[str, torch.Tensor]]
    ) -> Iterable[Tuple[str, torch.Tensor]]:
        for name, loaded_weight in weights:
            if name.startswith("model.") or name.startswith("lm_head."):
                yield name, loaded_weight
                continue
            for k, v in self.remapping.items():
                match = re.fullmatch(k, name)
                if match:
                    name = match.expand(v)
                    break
            else:
                logger.warning(f"Unrecognized weight: {name}. Skipping.")
                continue
            if name.endswith(".qscale_act"):
                name = re.sub(r"\.qscale_act$", ".input_scale", name)
            elif name.endswith(".qscale_weight"):
                name = re.sub(r"\.qscale_weight$", ".weight_scale", name)
            yield name, loaded_weight


EntryClass = [MistralForCausalLMEagle]
