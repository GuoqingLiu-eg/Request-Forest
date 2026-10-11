# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/transformers_utils/configs/nemotron_h.py


import copy
from typing import Any

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateShape,
    mamba2_state_dtype,
)
from sglang.srt.runtime_context import get_parallel

logger = logging.get_logger(__name__)

MAMBA = "M"
ATTENTION = "*"
MLP = "-"
MOE = "E"
DEFAULT_LAYERS_BLOCK_TYPE = ["mamba", "moe", "attention", "moe"]
DEFAULT_MTP_LAYERS_BLOCK_TYPE = ["attention", "moe"]
DEFAULT_MAMBA_CHUNK_SIZE = 256


class NemotronHConfig(PretrainedConfig):


    model_type = "nemotron_h"
    keys_to_ignore_at_inference = ["past_key_values"]

    @staticmethod
    def _validate_layers_block_type(
        layers_block_type, expected_length=None, param_name="layers_block_type"
    ):

        if not isinstance(layers_block_type, list):
            raise ValueError(
                f"{param_name} must be a list of strings. Got type: {type(layers_block_type)}"
            )
        if expected_length is not None and len(layers_block_type) != expected_length:
            raise ValueError(
                f"{param_name} must have length {expected_length}. Got length {len(layers_block_type)}."
            )
        valid_types = {"mamba", "attention", "mlp", "moe"}
        if not all(block_type in valid_types for block_type in layers_block_type):
            invalid = set(layers_block_type) - valid_types
            raise ValueError(
                f"{param_name} contains invalid types: {invalid}. Must be one of: {valid_types}"
            )

    @staticmethod
    def _resolve_layers_block_type(
        layers_block_type, hybrid_override_pattern, kwargs
    ) -> list[str]:


        pattern = kwargs.pop("hybrid_override_pattern", hybrid_override_pattern)
        if layers_block_type is None:
            if pattern is not None:
                layers_block_type = NemotronHConfig._pattern_to_list(pattern)
            else:


                layers_block_type = DEFAULT_LAYERS_BLOCK_TYPE
        return layers_block_type

    @staticmethod
    def _resolve_mtp_layers_block_type(mtp_layers_block_type, kwargs) -> list[str]:

        if "mtp_hybrid_override_pattern" in kwargs:
            pattern = kwargs.pop("mtp_hybrid_override_pattern")
            if mtp_layers_block_type is None or mtp_layers_block_type == [
                "attention",
                "moe",
            ]:
                mtp_layers_block_type = NemotronHConfig._pattern_to_list(pattern)
        return mtp_layers_block_type

    @staticmethod
    def _resolve_mamba_chunk_size(mamba_chunk_size, kwargs) -> int:

        chunk_size = kwargs.pop("chunk_size", None)
        if (
            mamba_chunk_size is not None
            and chunk_size is not None
            and mamba_chunk_size != chunk_size
        ):
            logger.warning(
                "Both chunk_size=%s and mamba_chunk_size=%s were provided. "
                "Using mamba_chunk_size.",
                chunk_size,
                mamba_chunk_size,
            )

        if mamba_chunk_size is None:
            mamba_chunk_size = chunk_size
        if mamba_chunk_size is None:
            mamba_chunk_size = DEFAULT_MAMBA_CHUNK_SIZE
        return mamba_chunk_size

    def __init__(
        self,
        *,
        vocab_size=131072,
        tie_word_embeddings=False,
        hidden_size=4096,
        intermediate_size=21504,
        num_hidden_layers=None,
        hybrid_override_pattern="M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-",
        layers_block_type=None,
        num_attention_heads=32,
        head_dim=128,
        num_key_value_heads=8,
        mlp_hidden_act="relu2",
        attention_bias=False,
        mlp_bias=False,
        use_bias=False,
        initializer_range=0.02,
        layer_norm_epsilon=1e-5,
        residual_in_fp32=False,
        use_cache=True,
        num_logits_to_keep=1,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        sliding_window=None,
        max_position_embeddings=4096,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        use_mamba_kernels=True,
        ssm_state_size=128,
        mamba_num_heads=128,
        mamba_n_groups=8,
        mamba_head_dim=64,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_hidden_act="silu",
        mamba_dt_min=0.001,
        mamba_dt_max=0.1,
        mamba_dt_limit=(0.0, float("inf")),
        mamba_dt_init_floor=1e-4,
        mamba_conv_bias=True,
        mamba_proj_bias=False,
        mamba_chunk_size=None,
        rescale_prenorm_residual=True,
        n_routed_experts=8,
        n_shared_experts=1,
        moe_intermediate_size=7688,
        moe_shared_expert_intermediate_size=7688,
        moe_latent_size=None,
        num_experts_per_tok=2,
        routed_scaling_factor=1.0,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        num_nextn_predict_layers=0,
        mtp_layers_block_type=DEFAULT_MTP_LAYERS_BLOCK_TYPE,
        **kwargs,
    ):
        mamba_chunk_size = self._resolve_mamba_chunk_size(mamba_chunk_size, kwargs)


        layers_block_type = self._resolve_layers_block_type(
            layers_block_type, hybrid_override_pattern, kwargs
        )
        mtp_layers_block_type = self._resolve_mtp_layers_block_type(
            mtp_layers_block_type, kwargs
        )


        if (
            num_hidden_layers is not None
            and len(layers_block_type) != num_hidden_layers
        ):
            logger.warning(
                f"num_hidden_layers ({num_hidden_layers}) is deprecated and doesn't match "
                f"layers_block_type length ({len(layers_block_type)}). Using layers_block_type length."
            )


        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.max_position_embeddings = max_position_embeddings
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout

        self._validate_layers_block_type(
            layers_block_type, expected_length=None, param_name="layers_block_type"
        )
        self.layers_block_type = layers_block_type


        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.mlp_hidden_act = mlp_hidden_act
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.use_bias = use_bias
        self.initializer_range = initializer_range
        self.layer_norm_epsilon = layer_norm_epsilon
        self.residual_in_fp32 = residual_in_fp32

        self.use_cache = use_cache
        self.num_logits_to_keep = num_logits_to_keep


        self.use_mamba_kernels = use_mamba_kernels
        self.mamba_n_groups = mamba_n_groups
        self.mamba_head_dim = mamba_head_dim
        self.ssm_state_size = ssm_state_size
        self.mamba_num_heads = mamba_num_heads
        self.conv_kernel = mamba_d_conv
        self.expand = mamba_expand
        self.mamba_hidden_act = mamba_hidden_act
        self.time_step_min = mamba_dt_min
        self.time_step_max = mamba_dt_max
        self.time_step_limit = mamba_dt_limit
        self.time_step_floor = mamba_dt_init_floor
        self.use_conv_bias = mamba_conv_bias
        self.mamba_proj_bias = mamba_proj_bias
        self.mamba_chunk_size = mamba_chunk_size
        self.rescale_prenorm_residual = rescale_prenorm_residual

        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_shared_expert_intermediate_size = moe_shared_expert_intermediate_size
        self.moe_latent_size = moe_latent_size
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob

        self.num_nextn_predict_layers = num_nextn_predict_layers

        if self.num_nextn_predict_layers > 0:
            if mtp_layers_block_type is None:
                raise ValueError(
                    "mtp_layers_block_type is required when num_nextn_predict_layers > 0. "
                    "Please provide an explicit list of layer types for MTP layers. "
                    "Example: mtp_layers_block_type=['attention', 'moe']"
                )
            self._validate_layers_block_type(
                mtp_layers_block_type, None, "mtp_layers_block_type"
            )
        self.mtp_layers_block_type = mtp_layers_block_type

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def mamba_layer_ids(self):
        return [
            i
            for i in range(self.num_hidden_layers)
            if self.hybrid_override_pattern[i] == MAMBA
        ]

    @property
    def full_attention_layer_ids(self):
        return [
            i
            for i in range(self.num_hidden_layers)
            if self.hybrid_override_pattern[i] == ATTENTION
        ]

    @property
    def mamba2_cache_params(self) -> Mamba2CacheParams:

        shape = Mamba2StateShape.create(
            tp_world_size=get_parallel().attn_tp_size,
            intermediate_size=self.mamba_num_heads * self.mamba_head_dim,
            n_groups=self.n_groups,
            num_heads=self.mamba_num_heads,
            head_dim=self.mamba_head_dim,
            state_size=self.ssm_state_size,
            conv_kernel=self.conv_kernel,
        )

        return Mamba2CacheParams(
            shape=shape, layers=self.mamba_layer_ids, dtype=mamba2_state_dtype(self)
        )

    @property
    def num_hidden_layers(self) -> int:

        return len(self.layers_block_type)

    @num_hidden_layers.setter
    def num_hidden_layers(self, value):

        pass

    @property
    def hybrid_override_pattern(self) -> str:

        return self._list_to_pattern(self.layers_block_type)

    @hybrid_override_pattern.setter
    def hybrid_override_pattern(self, value):

        self.layers_block_type = self._pattern_to_list(value)

    @property
    def mtp_hybrid_override_pattern(self) -> str:

        return self._list_to_pattern(self.mtp_layers_block_type)

    @mtp_hybrid_override_pattern.setter
    def mtp_hybrid_override_pattern(self, value):

        self.mtp_layers_block_type = self._pattern_to_list(value)

    @staticmethod
    def _list_to_pattern(layers_list: list[str]) -> str:

        reverse_mapping = {
            "mamba": MAMBA,
            "moe": MOE,
            "attention": ATTENTION,
            "mlp": MLP,
        }
        return "".join(reverse_mapping[layer_type] for layer_type in layers_list)

    @staticmethod
    def _pattern_to_list(pattern: str) -> list[str]:

        if any(char not in {MAMBA, MOE, ATTENTION, MLP} for char in pattern):
            raise ValueError(
                "Pattern must only contain characters 'M', '*', '-' or 'E'. "
                f"Got: {pattern}"
            )
        pattern_mapping = {
            MAMBA: "mamba",
            MOE: "moe",
            ATTENTION: "attention",
            MLP: "mlp",
        }
        return [pattern_mapping[char] for char in pattern]

    def get_nemotron_h_config_for_layer(self, layer_idx: int) -> "NemotronHConfig":
        return self

    def get_mtp_config(self) -> "NemotronHConfig":
        return self

    @property
    def max_n_routed_experts(self) -> int:
        return self.n_routed_experts


class NemotronHPuzzleConfig(NemotronHConfig):
    model_type = "nemotron_h_puzzle"
    has_no_defaults_at_init = True

    def __init__(
        self,
        *,
        block_configs: list[dict[str, Any]],
        mtp_block_configs: list[dict[str, Any]] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.block_configs = block_configs
        self.mtp_block_configs = mtp_block_configs

    def get_nemotron_h_config_for_layer(self, layer_idx: int) -> NemotronHConfig:
        layer_config = copy.copy(self)
        for key, value in self.block_configs[layer_idx].items():
            setattr(layer_config, key, value)
        return layer_config

    def get_mtp_config(self) -> NemotronHConfig:
        assert self.mtp_block_configs
        mtp_config = copy.copy(self)
        mtp_config.block_configs = self.mtp_block_configs
        return mtp_config

    @property
    def max_n_routed_experts(self) -> int:
        block_n_routed_experts = [
            block["n_routed_experts"]
            for block in self.block_configs
            if block["block_type"] == "moe"
        ]
        max_experts = max(block_n_routed_experts)
        assert max_experts > 0
        return max_experts
