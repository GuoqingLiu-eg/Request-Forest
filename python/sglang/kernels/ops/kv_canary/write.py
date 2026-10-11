from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.ops.kv_canary import consts
from sglang.kernels.ops.kv_canary.verify import (
    VerifyOrWriteContext,
    _assert_contiguous,
    _build_real_kv_source_abi,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@dataclass(frozen=True, slots=True, kw_only=True)
class WritePlan:


    write_offsets: torch.Tensor
    write_seed_slot_indices: torch.Tensor
    write_num_valid_reqs: torch.Tensor

    @classmethod
    def allocate(
        cls,
        *,
        write_req_capacity: int,
        device: torch.device,
    ) -> WritePlan:
        if write_req_capacity <= 0:
            raise ValueError(
                f"kv-canary: WritePlan write_req_capacity must be positive, got {write_req_capacity}"
            )
        return cls(
            write_offsets=torch.empty(
                write_req_capacity + 1, dtype=torch.int64, device=device
            ),
            write_seed_slot_indices=torch.empty(
                write_req_capacity, dtype=torch.int64, device=device
            ),
            write_num_valid_reqs=torch.empty(1, dtype=torch.int32, device=device),
        )

    def zero_for_testing_(self) -> WritePlan:

        self.write_offsets.zero_()
        self.write_seed_slot_indices.zero_()
        self.write_num_valid_reqs.zero_()
        return self


def launch_canary_write_kernel(
    *,
    context: VerifyOrWriteContext,
    plan: WritePlan,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    out_cache_loc: torch.Tensor,
    enable_write_input_assert: bool,
    expected_input_tokens: torch.Tensor | None,
    expected_input_positions: torch.Tensor | None,
) -> None:

    canary_buf = context.canary_buf
    real_kv_sources = context.real_kv_sources
    if len(real_kv_sources) > consts.MAX_REAL_KV_SOURCES:
        raise ValueError(
            f"kv-canary: at most {consts.MAX_REAL_KV_SOURCES} RealKvSource entries supported by the CUDA ABI, "
            f"got {len(real_kv_sources)}"
        )

    _assert_contiguous(canary_buf, "canary_buf")
    _assert_contiguous(plan.write_offsets, "plan.write_offsets")
    _assert_contiguous(plan.write_seed_slot_indices, "plan.write_seed_slot_indices")
    _assert_contiguous(plan.write_num_valid_reqs, "plan.write_num_valid_reqs")
    _assert_contiguous(input_ids, "input_ids")
    _assert_contiguous(positions, "positions")
    _assert_contiguous(out_cache_loc, "out_cache_loc")
    if enable_write_input_assert:
        if expected_input_tokens is None or expected_input_positions is None:
            raise ValueError(
                "kv-canary: expected input tensors are required when enable_write_input_assert=True"
            )
        _assert_contiguous(expected_input_tokens, "expected_input_tokens")
        _assert_contiguous(expected_input_positions, "expected_input_positions")
    else:
        if expected_input_tokens is not None or expected_input_positions is not None:
            raise ValueError(
                "kv-canary: expected input tensors must be None when enable_write_input_assert=False"
            )
    _assert_contiguous(context.violation_ring, "violation_ring")
    _assert_contiguous(context.violation_write_index, "violation_write_index")
    _assert_contiguous(context.slot_run_counter, "slot_run_counter")
    _assert_contiguous(context.kernel_run_counter, "kernel_run_counter")
    _assert_contiguous(
        context.enable_chain_position_assert, "enable_chain_position_assert"
    )

    padded_bufs, source_params = _build_real_kv_source_abi(
        real_kv_sources=real_kv_sources, device=canary_buf.device
    )

    module = _jit_canary_write_module()
    module.canary_write_step_cuda(
        canary_buf,
        plan.write_offsets,
        plan.write_seed_slot_indices,
        plan.write_num_valid_reqs,
        input_ids,
        positions,
        out_cache_loc,
        int(context.kernel_kind),
        int(enable_write_input_assert),
        expected_input_tokens,
        expected_input_positions,
        context.violation_ring,
        context.violation_write_index,
        context.slot_run_counter,
        context.kernel_run_counter,
        context.enable_chain_position_assert,
        padded_bufs[0],
        padded_bufs[1],
        padded_bufs[2],
        padded_bufs[3],
        source_params,
        len(real_kv_sources),
        int(context.real_kv_hash_mode),
    )


@cache_once
def _jit_canary_write_module() -> Module:
    return load_jit(
        "kv_canary_write",
        cuda_files=["kv_canary/canary_write.cuh"],
        cuda_wrappers=[
            ("canary_write_step_cuda", "canary::canary_write_step_cuda"),
        ],
    )
