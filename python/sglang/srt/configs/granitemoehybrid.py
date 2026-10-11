# coding=utf-8
# Copyright 2025 IBM and the HuggingFace Inc. team. All rights reserved.
#
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


from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.runtime_context import get_parallel

logger = logging.get_logger(__name__)

MAMBA = "mamba"
ATTENTION = "attention"


def _to_sglang_layer_types(layer_types: list[str]) -> list[str]:

    aliases = {"linear_attention": MAMBA, "full_attention": ATTENTION}
    return [aliases.get(t, t) for t in layer_types]


class GraniteMoeHybridConfig(PretrainedConfig):


    model_type = "granitemoehybrid"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=100352,
        tie_word_embeddings=True,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=40,
        layer_types=None,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        initializer_range=0.1,
        rms_norm_eps=1e-5,
        normalization_function="rmsnorm",
        use_cache=True,
        pad_token_id=100256,
        bos_token_id=100257,
        eos_token_id=100257,
        max_position_embeddings=131072,
        attention_dropout=0.0,
        attention_bias=False,
        position_embedding_type="nope",
        rope_theta=10000.0,
        rope_scaling=None,
        mamba_d_state=128,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_d_head=64,
        mamba_n_heads=64,
        mamba_n_groups=1,
        mamba_chunk_size=256,
        mamba_conv_bias=True,
        mamba_proj_bias=False,
        embedding_multiplier=12.0,
        logits_scaling=8.0,
        attention_multiplier=0.015625,
        residual_multiplier=0.22,
        num_local_experts=0,
        num_experts_per_tok=0,
        shared_intermediate_size=8192,
        output_router_logits=False,
        router_aux_loss_coef=0.01,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers


        if layer_types is None:

            self.layer_types = []
            for i in range(num_hidden_layers):
                if (i + 1) % 6 == 0:
                    self.layer_types.append(ATTENTION)
                else:
                    self.layer_types.append(MAMBA)
        else:
            self.layer_types = _to_sglang_layer_types(layer_types)


        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types must have length equal to num_hidden_layers ({num_hidden_layers}), "
                f"but got {len(self.layer_types)}"
            )

        for layer_type in self.layer_types:
            if layer_type not in [MAMBA, ATTENTION]:
                raise ValueError(
                    f"Each element in layer_types must be either '{MAMBA}' or '{ATTENTION}', "
                    f"but got '{layer_type}'"
                )

        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.normalization_function = normalization_function

        self.use_cache = use_cache
        self.max_position_embeddings = max_position_embeddings
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias

        self.position_embedding_type = position_embedding_type
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling


        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_d_head = mamba_d_head
        self.mamba_n_heads = mamba_n_heads
        self.mamba_n_groups = mamba_n_groups
        self.mamba_chunk_size = mamba_chunk_size
        self.mamba_conv_bias = mamba_conv_bias
        self.mamba_proj_bias = mamba_proj_bias


        self.mamba_intermediate_size = mamba_expand * hidden_size


        if self.mamba_intermediate_size % mamba_n_heads != 0:
            raise ValueError(
                f"mamba_intermediate_size ({self.mamba_intermediate_size}) must be divisible by "
                f"mamba_n_heads ({mamba_n_heads})"
            )

        if mamba_d_head * mamba_n_heads != self.mamba_intermediate_size:
            raise ValueError(
                f"mamba_d_head ({mamba_d_head}) * mamba_n_heads ({mamba_n_heads}) must equal "
                f"mamba_intermediate_size ({self.mamba_intermediate_size})"
            )


        self.embedding_multiplier = embedding_multiplier
        self.logits_scaling = logits_scaling
        self.attention_multiplier = attention_multiplier
        self.residual_multiplier = residual_multiplier


        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.shared_intermediate_size = shared_intermediate_size
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


        self.layer_types = _to_sglang_layer_types(self.layer_types)

    @property
    def mamba_layer_ids(self):

        return [
            i for i in range(self.num_hidden_layers) if self.layer_types[i] == MAMBA
        ]

    @property
    def attention_layer_ids(self):

        return [
            i for i in range(self.num_hidden_layers) if self.layer_types[i] == ATTENTION
        ]

    @property
    def full_attention_layer_ids(self):

        return self.attention_layer_ids

    @property
    def mamba2_cache_params(self):


        shape = Mamba2StateShape.create(
            tp_world_size=get_parallel().attn_tp_size,
            intermediate_size=self.mamba_intermediate_size,
            n_groups=self.mamba_n_groups,
            num_heads=self.mamba_n_heads,
            head_dim=self.mamba_d_head,
            state_size=self.mamba_d_state,
            conv_kernel=self.mamba_d_conv,
        )
        return Mamba2CacheParams(shape=shape, layers=self.mamba_layer_ids)
