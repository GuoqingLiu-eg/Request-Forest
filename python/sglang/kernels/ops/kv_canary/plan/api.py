from __future__ import annotations

from typing import Optional

import torch

from sglang.kernels.ops.kv_canary.plan.entries_kernel import (
    launch_plan_entries_kernel,
)
from sglang.kernels.ops.kv_canary.plan.offsets_kernel import (
    _PLAN_BS_BLOCK_SIZE,
    launch_plan_offsets_kernel,
)
from sglang.kernels.ops.kv_canary.verify import VerifyPlan
from sglang.kernels.ops.kv_canary.write import WritePlan


def launch_canary_plan_kernels(
    *,
    verify_plan_out: VerifyPlan,
    write_plan_out: WritePlan,
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    swa_window_size: int,
    full_to_swa_index_mapping: Optional[torch.Tensor],
    verify_capacity: int,
    req_to_verify_expected_tokens: Optional[torch.Tensor],
    req_to_verify_expected_tokens_valid_lens: Optional[torch.Tensor],
    kv_token_id_vs_position_offset: int,
) -> None:

    bs = int(req_pool_indices.shape[0])
    if bs > _PLAN_BS_BLOCK_SIZE:
        raise ValueError(
            f"kv-canary: launch_canary_plan_kernels supports at most bs={_PLAN_BS_BLOCK_SIZE} reqs per launch, "
            f"got bs={bs}. Bump _PLAN_BS_BLOCK_SIZE if real workloads need this."
        )
    if swa_window_size > 0 and full_to_swa_index_mapping is None:
        raise ValueError(
            "kv-canary: launch_canary_plan_kernels requires full_to_swa_index_mapping when swa_window_size > 0"
        )

    device = verify_plan_out.verify_slot_indices.device
    verify_offsets_scratch = torch.empty(
        _PLAN_BS_BLOCK_SIZE + 1, dtype=torch.int64, device=device
    )

    plan_verify_capacity = int(verify_plan_out.verify_slot_indices.shape[0])
    if verify_capacity != plan_verify_capacity:
        raise ValueError(
            f"kv-canary: launch_canary_plan_kernels verify_capacity={verify_capacity} does not match "
            f"verify_plan_out.verify_slot_indices.shape[0]={plan_verify_capacity}"
        )

    write_plan_out.write_offsets.zero_()

    launch_plan_offsets_kernel(
        req_pool_indices=req_pool_indices,
        prefix_lens=prefix_lens,
        extend_seq_lens=extend_seq_lens,
        req_to_token=req_to_token,
        full_to_swa_index_mapping=full_to_swa_index_mapping,
        out_verify_offsets_scratch=verify_offsets_scratch,
        out_write_offsets=write_plan_out.write_offsets,
        out_write_seed_slot_indices=write_plan_out.write_seed_slot_indices,
        out_verify_num_valid=verify_plan_out.verify_num_valid,
        out_verify_enable=verify_plan_out.enable,
        out_write_num_valid_reqs=write_plan_out.write_num_valid_reqs,
        swa_window_size=int(swa_window_size),
        verify_capacity=verify_capacity,
    )

    launch_plan_entries_kernel(
        req_pool_indices=req_pool_indices,
        prefix_lens=prefix_lens,
        req_to_token=req_to_token,
        full_to_swa_index_mapping=full_to_swa_index_mapping,
        verify_offsets_scratch=verify_offsets_scratch,
        verify_enable=verify_plan_out.enable,
        req_to_verify_expected_tokens=req_to_verify_expected_tokens,
        req_to_verify_expected_tokens_valid_lens=req_to_verify_expected_tokens_valid_lens,
        out_verify_slot_indices=verify_plan_out.verify_slot_indices,
        out_verify_expected_tokens=verify_plan_out.verify_expected_tokens,
        out_verify_expected_positions=verify_plan_out.verify_expected_positions,
        out_verify_prev_slot_indices=verify_plan_out.verify_prev_slot_indices,
        kv_token_id_vs_position_offset=int(kv_token_id_vs_position_offset),
        swa_window_size=int(swa_window_size),
    )
