from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from sglang.kernels.ops.kv_canary.consts import VIOLATION_FIELDS
from sglang.srt.kv_canary.config import CanaryConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class ViolationLog:


    violation_ring: torch.Tensor
    violation_write_index: torch.Tensor

    @classmethod
    def allocate(cls, *, ring_capacity: int, device: torch.device) -> ViolationLog:
        if ring_capacity <= 0:
            raise ValueError(
                f"kv-canary: ViolationLog ring_capacity must be positive, got {ring_capacity}"
            )
        return cls(
            violation_ring=torch.zeros(
                ring_capacity, VIOLATION_FIELDS, dtype=torch.int64, device=device
            ),
            violation_write_index=torch.zeros(1, dtype=torch.int32, device=device),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CanaryDeviceState:


    violation_log: ViolationLog
    kernel_run_counters: torch.Tensor
    slot_run_counters: torch.Tensor
    enable_chain_position_assert: torch.Tensor
    req_to_verify_expected_tokens: Optional[torch.Tensor]

    @classmethod
    def allocate(
        cls,
        *,
        config: CanaryConfig,
        device: torch.device,
        num_tags: int,
        req_to_token_alloc_size: Optional[int] = None,
        max_context_len: Optional[int] = None,
    ) -> CanaryDeviceState:
        if num_tags <= 0:
            raise ValueError(
                f"kv-canary: CanaryDeviceState num_tags must be positive, got {num_tags}"
            )
        violation_log = ViolationLog.allocate(
            ring_capacity=config.ring_capacity, device=device
        )
        kernel_run_counters = torch.zeros(num_tags, dtype=torch.int64, device=device)
        slot_run_counters = torch.zeros(num_tags, dtype=torch.int64, device=device)
        enable_chain_position_assert = torch.ones(1, dtype=torch.int32, device=device)
        if config.enable_verify_token_assert:
            if req_to_token_alloc_size is None or max_context_len is None:
                raise ValueError(
                    "kv-canary: CanaryDeviceState.allocate requires req_to_token_alloc_size "
                    "and max_context_len when CanaryConfig.enable_verify_token_assert is on"
                )
            req_to_verify_expected_tokens = torch.empty(
                (req_to_token_alloc_size, max_context_len),
                dtype=torch.int32,
                device=device,
            )
        else:
            req_to_verify_expected_tokens = None
        return cls(
            violation_log=violation_log,
            kernel_run_counters=kernel_run_counters,
            slot_run_counters=slot_run_counters,
            enable_chain_position_assert=enable_chain_position_assert,
            req_to_verify_expected_tokens=req_to_verify_expected_tokens,
        )
