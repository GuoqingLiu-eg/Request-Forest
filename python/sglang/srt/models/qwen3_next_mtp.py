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


import copy
import logging
from contextlib import ExitStack
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_pp_group
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen3_next import Qwen3NextForCausalLM, Qwen3NextModel
from sglang.srt.runtime_context import get_model, get_parallel, get_spec
from sglang.srt.utils import add_prefix, is_npu

logger = logging.getLogger(__name__)


class Qwen3NextForCausalLMMTP(Qwen3NextForCausalLM):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)

        config = copy.deepcopy(config)
        self.config = config
        self.tp_size = get_parallel().tp_size
        if is_npu() and get_spec().speculative_draft_model_quantization is None:
            quant_config = None
        self.quant_config = quant_config

        self.pp_group = get_pp_group()


        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        RMSNorm_cls = GemmaRMSNorm
        self.pre_fc_norm_embedding = RMSNorm_cls(
            config.hidden_size, config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = RMSNorm_cls(config.hidden_size, config.rms_norm_eps)
        mtp_config = copy.deepcopy(config)
        mtp_config.num_hidden_layers = 1
        mtp_config.full_attention_interval = 1
        self.model = Qwen3NextModel(
            mtp_config,
            quant_config,
            prefix=add_prefix("model", prefix),
            is_nextn=True,
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("model.shared_head.head", prefix),
            use_attn_tp_group=get_parallel().config.enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)


        self.num_fused_shared_experts = self._get_num_fused_shared_experts()
        if self.num_fused_shared_experts > 1:
            raise ValueError(
                "Qwen3-Next MTP shared expert fusion currently supports exactly one "
                "shared expert because checkpoint weight remapping maps it into "
                "a single fused MoE expert slot."
            )
        self.enable_shared_expert_fusion = self.num_fused_shared_experts > 0

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        exit_stack = ExitStack()
        if (
            is_npu()
            and self.quant_config is None
            and get_model().quantization is not None
        ):

            exit_stack.enter_context(envs.SGLANG_DEEPEP_BF16_DISPATCH.override(True))
            exit_stack.enter_context(
                envs.DEEP_NORMAL_MODE_USE_INT8_QUANT.override(False)
            )

        try:
            if input_embeds is None:
                input_embeds = self.model.embed_tokens(input_ids)

            hidden_states = forward_batch.spec_info.hidden_states

            if not forward_batch.forward_mode.is_idle():
                input_embeds = self.pre_fc_norm_embedding(input_embeds)
                hidden_states = self.pre_fc_norm_hidden(hidden_states)
            hidden_states = self.fc(torch.cat((input_embeds, hidden_states), dim=-1))

            with get_global_expert_distribution_recorder().disable_this_region():
                hidden_states = self.model(
                    input_ids,
                    positions,
                    forward_batch,
                    hidden_states,
                )
        finally:
            exit_stack.close()

        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]], is_mtp: bool = False
    ):
        super().load_weights(weights, is_mtp=True)


EntryClass = [Qwen3NextForCausalLMMTP]
