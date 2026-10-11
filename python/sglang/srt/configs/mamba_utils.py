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


import logging
from abc import ABC
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from sglang.srt.distributed.utils import divide
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def extra_groups_for_head_shards(ngroups: int, tp_size: int):


    if ngroups % tp_size == 0:
        return 0


    return tp_size - ngroups


@dataclass(kw_only=True, frozen=True)
class Mamba2StateDType:
    conv: torch.dtype
    temporal: torch.dtype


def mamba2_state_dtype(config=None) -> Mamba2StateDType:

    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    conv_dtype = dtype_map.get(envs.SGLANG_MAMBA_CONV_DTYPE.get(), torch.bfloat16)


    ssm_dtype = torch.float32


    if config is not None:
        config_dtype = None
        if hasattr(config, "text_config") and hasattr(
            config.text_config, "mamba_ssm_dtype"
        ):

            config_dtype = config.text_config.mamba_ssm_dtype
        elif hasattr(config, "mamba_ssm_dtype"):

            config_dtype = config.mamba_ssm_dtype

        if config_dtype is not None:
            if config_dtype not in dtype_map:
                logger.warning(
                    f"Invalid mamba_ssm_dtype '{config_dtype}' in config. "
                    f"Must be one of {list(dtype_map.keys())}. Using default 'float32'."
                )
            else:
                ssm_dtype = dtype_map[config_dtype]


    env_ssm_dtype = envs.SGLANG_MAMBA_SSM_DTYPE.get()
    if env_ssm_dtype is not None:
        if env_ssm_dtype not in dtype_map:
            logger.warning(
                f"Invalid mamba_ssm_dtype '{env_ssm_dtype}' from environment variable. "
                f"Must be one of {list(dtype_map.keys())}. Using default 'float32'."
            )
        else:
            ssm_dtype = dtype_map[env_ssm_dtype]

    logger.debug(f"Mamba2 state dtype: conv_dtype={conv_dtype}, ssm_dtype={ssm_dtype}")

    return Mamba2StateDType(conv=conv_dtype, temporal=ssm_dtype)


@dataclass(kw_only=True, frozen=True)
class BaseLinearStateParams(ABC):
    dtype: Mamba2StateDType = field(default_factory=lambda: mamba2_state_dtype(None))
    layers: list[int]

    @property
    def mamba_cache_per_req(self) -> int:
        conv_numel = int(
            np.sum([np.prod(conv_shape) for conv_shape in self.shape.conv])
        )

        ssm_numel = int(np.prod(self.shape.temporal))
        return (
            conv_numel * self.dtype.conv.itemsize
            + ssm_numel * self.dtype.temporal.itemsize
        ) * len(self.layers)

    def replayssm_ring_bytes_per_req(self, record_len: int) -> int:

        hv, v_dim, k_dim = self.shape.temporal
        h_k = self.shape.num_k_heads_per_tp
        conv_b = self.dtype.conv.itemsize
        fp32_b = 4
        per_layer = (
            hv * record_len * v_dim * conv_b
            + h_k * record_len * k_dim * conv_b
            + hv * record_len * fp32_b

            + hv * record_len * (k_dim if self.is_kda else 1) * fp32_b
        )
        if self.is_kda:
            per_layer += (
                hv * record_len * v_dim * conv_b
                + h_k * record_len * k_dim * conv_b
            )
        return per_layer * len(self.layers)

    @property
    def is_kda(self) -> bool:

        return False


@dataclass(kw_only=True, frozen=True)
class Mamba2StateShape:
    conv: list[tuple[int, int]]
    temporal: tuple[int, int, int]


    disable_conv_window_dedup: bool = False

    intermediate_size: int
    conv_dim: int
    ssm_state_size: int
    num_heads: int
    head_dim: int
    state_size: int
    conv_kernel: int


    num_k_heads_per_tp: int = 1


    conv_shard_groups: Optional[List[int]] = None

    @staticmethod
    def create(
        *,
        tp_world_size: int,
        intermediate_size: int,
        n_groups: int,
        num_heads: int,
        head_dim: int,
        state_size: int,
        conv_kernel: int,
        conv_shard_groups: Optional[List[int]] = None,
    ) -> "Mamba2StateShape":


        num_k_heads_per_tp = (
            divide(n_groups, tp_world_size)
            if n_groups % tp_world_size == 0
            else -(-n_groups // tp_world_size)
        )


        if n_groups % tp_world_size != 0:
            extra_groups = extra_groups_for_head_shards(n_groups, tp_world_size)
            n_groups += extra_groups

        conv_dim = intermediate_size + 2 * n_groups * state_size


        conv_state_shape = divide(conv_dim, tp_world_size), conv_kernel - 1


        temporal_state_shape = (divide(num_heads, tp_world_size), head_dim, state_size)
        return Mamba2StateShape(
            conv=[conv_state_shape],
            temporal=temporal_state_shape,
            intermediate_size=intermediate_size,
            conv_dim=conv_dim,
            ssm_state_size=state_size,
            num_heads=num_heads,
            head_dim=head_dim,
            state_size=state_size,
            conv_kernel=conv_kernel,
            num_k_heads_per_tp=num_k_heads_per_tp,
            conv_shard_groups=conv_shard_groups,
        )


@dataclass(kw_only=True, frozen=True)
class Mamba2CacheParams(BaseLinearStateParams):
    shape: Mamba2StateShape


@dataclass(kw_only=True, frozen=True)
class KimiLinearStateShape:
    conv: List[tuple[int, int]]
    temporal: tuple[int, int, int]


    disable_conv_window_dedup: bool = True


    conv_slice_axis: int = 1

    num_heads: int
    head_dim: int
    num_k_heads: int
    head_k_dim: int
    conv_kernel: int
    num_spec: int

    conv_shard_groups: Optional[List[int]] = None


    num_k_heads_per_tp: int = 1

    @staticmethod
    def create(
        *,
        tp_world_size: int,
        num_heads: int,
        head_dim: int,
        num_k_heads: Optional[int] = None,
        head_k_dim: Optional[int] = None,
        conv_kernel_size: int = 4,
        num_spec: int = 0,
    ) -> "KimiLinearStateShape":
        if num_k_heads is None:
            num_k_heads = num_heads
        if head_k_dim is None:
            head_k_dim = head_dim
        num_k_heads_per_tp = (
            divide(num_k_heads, tp_world_size)
            if num_k_heads % tp_world_size == 0
            else -(-num_k_heads // tp_world_size)
        )

        proj_size = num_heads * head_dim
        proj_k_size = num_k_heads * head_k_dim

        conv_state_shape = (divide(proj_size, tp_world_size), conv_kernel_size - 1)
        conv_state_k_shape = (divide(proj_k_size, tp_world_size), conv_kernel_size - 1)
        temporal_state_shape = (divide(num_heads, tp_world_size), head_dim, head_dim)

        conv_state_shape = (
            conv_state_shape[1],
            conv_state_shape[0] + conv_state_k_shape[0] * 2,
        )

        return KimiLinearStateShape(
            conv=[conv_state_shape],
            temporal=temporal_state_shape,
            num_heads=num_heads,
            head_dim=head_dim,
            num_k_heads=num_k_heads,
            head_k_dim=head_k_dim,
            conv_kernel=conv_kernel_size,
            num_spec=num_spec,
            conv_shard_groups=[proj_size, proj_k_size, proj_k_size],
            num_k_heads_per_tp=num_k_heads_per_tp,
        )


@dataclass(kw_only=True, frozen=True)
class KimiLinearCacheParams(BaseLinearStateParams):
    shape: KimiLinearStateShape

    @property
    def is_kda(self) -> bool:
        return True
