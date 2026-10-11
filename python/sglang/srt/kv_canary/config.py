from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from sglang.kernels.ops.kv_canary.consts import (
    RealKvHashMode,
)
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs


class CanaryMode(str, Enum):
    NONE = "none"
    LOG = "log"
    RAISE = "raise"


@dataclass(frozen=True, slots=True, kw_only=True)
class CanaryConfig:


    mode: CanaryMode
    ring_capacity: int
    sweep_interval: int
    real_kv_hash_mode: RealKvHashMode
    enable_write_input_assert: bool
    enable_verify_token_assert: bool
    stats_print_every_n_steps: int

    @classmethod
    def from_env(cls, server_args: ServerArgs) -> CanaryConfig:
        mode_raw = server_args.kv_canary.strip().lower()
        if mode_raw not in ("none", "log", "raise"):
            raise ValueError(
                f"kv-canary: kv_canary must be one of none/log/raise, got {mode_raw!r}"
            )

        real_kv_raw = server_args.kv_canary_real_data.strip().upper()

        return cls(
            mode=CanaryMode(mode_raw),
            ring_capacity=envs.SGLANG_KV_CANARY_RING_CAPACITY.get(),
            sweep_interval=server_args.kv_canary_sweep_interval,
            real_kv_hash_mode=RealKvHashMode[real_kv_raw],
            enable_write_input_assert=envs.SGLANG_KV_CANARY_ENABLE_WRITE_INPUT_ASSERT.get(),
            enable_verify_token_assert=envs.SGLANG_KV_CANARY_ENABLE_VERIFY_TOKEN_ASSERT.get(),
            stats_print_every_n_steps=envs.SGLANG_KV_CANARY_STATS_PRINT_EVERY_N_STEPS.get(),
        )
