from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Final

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.ops.kv_canary import consts

if TYPE_CHECKING:
    from tvm_ffi.module import Module


CANARY_SLOT_BYTES: Final[int] = consts.CANARY_FIELDS_PER_SLOT * 8


class CanaryLaunchTag(IntEnum):


    HEAD_K_FULL = 0
    HEAD_V_FULL = 1
    TAIL_K_FULL = 2
    TAIL_V_FULL = 3
    SWEEP_K_FULL = 4
    SWEEP_V_FULL = 5
    HEAD_K_SWA = 6
    HEAD_V_SWA = 7
    TAIL_K_SWA = 8
    TAIL_V_SWA = 9
    SWEEP_K_SWA = 10
    SWEEP_V_SWA = 11


def _assert_contiguous(tensor: torch.Tensor, name: str) -> None:
    if not tensor.is_contiguous():
        raise ValueError(f"kv-canary: {name} must be contiguous")


@dataclass(frozen=True, slots=True, kw_only=True)
class RealKvSource:


    tensor: torch.Tensor
    page_size: int
    num_bytes_per_token: int
    read_bytes: int

    def __post_init__(self) -> None:
        if self.page_size < 1:
            raise ValueError(
                f"kv-canary: RealKvSource.page_size must be >= 1, got {self.page_size}"
            )
        if self.num_bytes_per_token <= 0 or self.num_bytes_per_token % 16 != 0:
            raise ValueError(
                f"kv-canary: RealKvSource.num_bytes_per_token must be a positive multiple of 16, "
                f"got {self.num_bytes_per_token}"
            )
        if (
            self.read_bytes <= 0
            or self.read_bytes > self.num_bytes_per_token
            or self.read_bytes % 16 != 0
        ):
            raise ValueError(
                f"kv-canary: RealKvSource.read_bytes must be a positive multiple of 16 in "
                f"(0, num_bytes_per_token={self.num_bytes_per_token}], got {self.read_bytes}"
            )
        if self.tensor.ndim < 2:
            raise ValueError(
                f"kv-canary: RealKvSource.tensor must be at least 2-D, got shape {tuple(self.tensor.shape)}"
            )
        row_stride_bytes = int(self.tensor.shape[1]) * self.tensor.element_size()
        if row_stride_bytes % 16 != 0:
            raise ValueError(
                f"kv-canary: RealKvSource.tensor dim-1 byte width must be a multiple of 16, "
                f"got {row_stride_bytes} bytes (shape={tuple(self.tensor.shape)}, "
                f"dtype={self.tensor.dtype})"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class VerifyOrWriteContext:


    canary_buf: torch.Tensor
    kernel_kind: CanaryLaunchTag
    violation_ring: torch.Tensor
    violation_write_index: torch.Tensor
    slot_run_counter: torch.Tensor
    kernel_run_counter: torch.Tensor
    real_kv_sources: tuple[RealKvSource, ...]
    real_kv_hash_mode: consts.RealKvHashMode
    enable_chain_position_assert: torch.Tensor


@dataclass(frozen=True, slots=True, kw_only=True)
class VerifyPlan:


    verify_slot_indices: torch.Tensor
    verify_expected_tokens: torch.Tensor
    verify_expected_positions: torch.Tensor
    verify_prev_slot_indices: torch.Tensor
    verify_num_valid: torch.Tensor
    enable: torch.Tensor

    @classmethod
    def allocate(cls, *, verify_capacity: int, device: torch.device) -> VerifyPlan:
        if verify_capacity <= 0:
            raise ValueError(
                f"kv-canary: VerifyPlan verify_capacity must be positive, got {verify_capacity}"
            )
        return cls(
            verify_slot_indices=torch.empty(
                verify_capacity, dtype=torch.int64, device=device
            ),
            verify_expected_tokens=torch.empty(
                verify_capacity, dtype=torch.int64, device=device
            ),
            verify_expected_positions=torch.empty(
                verify_capacity, dtype=torch.int64, device=device
            ),
            verify_prev_slot_indices=torch.empty(
                verify_capacity, dtype=torch.int64, device=device
            ),
            verify_num_valid=torch.empty(1, dtype=torch.int32, device=device),


            enable=torch.ones(1, dtype=torch.int32, device=device),
        )

    def zero_for_testing_(self) -> VerifyPlan:

        self.verify_slot_indices.zero_()


        self.verify_expected_tokens.fill_(-1)
        self.verify_expected_positions.zero_()
        self.verify_prev_slot_indices.zero_()
        self.verify_num_valid.zero_()
        self.enable.zero_()
        return self


def launch_canary_verify_kernel(
    *,
    context: VerifyOrWriteContext,
    plan: VerifyPlan,
    check_verify_expected_token: bool,
) -> None:

    canary_buf = context.canary_buf
    real_kv_sources = context.real_kv_sources
    if len(real_kv_sources) > consts.MAX_REAL_KV_SOURCES:
        raise ValueError(
            f"kv-canary: at most {consts.MAX_REAL_KV_SOURCES} RealKvSource entries supported by the CUDA ABI, "
            f"got {len(real_kv_sources)}"
        )

    _assert_contiguous(canary_buf, "canary_buf")
    _assert_contiguous(plan.verify_slot_indices, "plan.verify_slot_indices")
    _assert_contiguous(plan.verify_expected_tokens, "plan.verify_expected_tokens")
    _assert_contiguous(plan.verify_expected_positions, "plan.verify_expected_positions")
    _assert_contiguous(plan.verify_prev_slot_indices, "plan.verify_prev_slot_indices")
    _assert_contiguous(plan.verify_num_valid, "plan.verify_num_valid")
    _assert_contiguous(plan.enable, "plan.enable")
    _assert_contiguous(context.violation_ring, "violation_ring")
    _assert_contiguous(context.violation_write_index, "violation_write_index")
    _assert_contiguous(context.slot_run_counter, "slot_run_counter")
    _assert_contiguous(context.kernel_run_counter, "kernel_run_counter")

    padded_bufs, source_params = _build_real_kv_source_abi(
        real_kv_sources=real_kv_sources, device=canary_buf.device
    )

    module = _jit_canary_verify_module(check_verify_expected_token)
    module.canary_verify_step_cuda(
        canary_buf,
        plan.verify_slot_indices,
        plan.verify_expected_tokens,
        plan.verify_expected_positions,
        plan.verify_prev_slot_indices,
        plan.verify_num_valid,
        plan.enable,
        int(context.kernel_kind),
        context.violation_ring,
        context.violation_write_index,
        context.slot_run_counter,
        context.kernel_run_counter,
        padded_bufs[0],
        padded_bufs[1],
        padded_bufs[2],
        padded_bufs[3],
        source_params,
        len(real_kv_sources),
        int(context.real_kv_hash_mode),
    )


@cache_once
def _jit_canary_verify_module(check_verify_expected_token: bool) -> Module:
    args = make_cpp_args(check_verify_expected_token)
    return load_jit(
        "kv_canary_verify",
        *args,
        cuda_files=["kv_canary/canary_verify.cuh"],
        cuda_wrappers=[
            (
                "canary_verify_step_cuda",
                f"canary::CanaryVerifyKernel<{args}>::run",
            ),
        ],
    )


def _build_real_kv_source_abi(
    *,
    real_kv_sources: tuple[RealKvSource, ...],
    device: torch.device,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    padded_bufs: list[torch.Tensor] = []
    params = torch.zeros(
        (consts.MAX_REAL_KV_SOURCES, consts.REAL_KV_SOURCE_FIELDS_PER_ENTRY),
        dtype=torch.int32,
        device="cpu",
    )

    for i, source in enumerate(real_kv_sources):
        _assert_contiguous(source.tensor, f"real_kv_sources[{i}].tensor")
        source_u8 = source.tensor.view(torch.uint8)
        if source_u8.dim() != 2:
            raise ValueError(
                f"kv-canary: real_kv_sources[{i}].tensor (viewed as uint8) must be 2-D, "
                f"got {source_u8.dim()}-D"
            )
        padded_bufs.append(source_u8)
        params[i, consts.REAL_KV_SOURCE_FIELD_PAGE_SIZE] = source.page_size
        params[i, consts.REAL_KV_SOURCE_FIELD_NUM_BYTES_PER_TOKEN] = (
            source.num_bytes_per_token
        )
        params[i, consts.REAL_KV_SOURCE_FIELD_READ_BYTES] = source.read_bytes


    dummy = torch.empty((1, 1), dtype=torch.uint8, device=device)
    for _ in range(len(real_kv_sources), consts.MAX_REAL_KV_SOURCES):
        padded_bufs.append(dummy)

    return padded_bufs, params
