from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import torch

from sglang.kernels.ops.kv_canary.verify import RealKvSource


class PoolKind(IntEnum):


    FULL = 0
    SWA = 1


@dataclass(frozen=True, slots=True, kw_only=True)
class CanaryBufferGroup:


    kind: PoolKind
    k_head: torch.Tensor
    k_tail: torch.Tensor
    v_head: Optional[torch.Tensor]
    v_tail: Optional[torch.Tensor]
    real_kv_sources_k: tuple[RealKvSource, ...]
    real_kv_sources_v: tuple[RealKvSource, ...]
    swa_index_lut: Optional[torch.Tensor]
    kv_token_id_vs_position_offset: int

    @property
    def has_v_half(self) -> bool:
        return self.v_head is not None
