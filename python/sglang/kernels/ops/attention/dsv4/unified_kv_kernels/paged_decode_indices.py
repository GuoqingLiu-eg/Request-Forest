# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


import torch
import triton
import triton.language as tl


@triton.jit
def _v4_paged_decode_indices_kernel(
    state_slot_per_seq_ptr,
    batch_id_per_token_ptr,
    positions_ptr,
    swa_indptr_ptr,
    csa_indptr_ptr,
    hca_indptr_ptr,
    swa_indices_ptr,
    csa_indices_ptr,
    hca_indices_ptr,
    ring_stride,
    win: tl.constexpr,
    BLOCK_N: tl.constexpr,
):

    t = tl.program_id(0)
    bid = tl.load(batch_id_per_token_ptr + t)
    if bid < 0:
        return

    slot = tl.load(state_slot_per_seq_ptr + bid)
    pos = tl.load(positions_ptr + t)


    n = tl.minimum(pos + 1, win)
    swa_base = tl.load(swa_indptr_ptr + t)
    csa_base = tl.load(csa_indptr_ptr + t)
    hca_base = tl.load(hca_indptr_ptr + t)

    i = tl.arange(0, BLOCK_N)
    mask = i < n
    abs_pos = pos - n + 1 + i
    ring_idx = abs_pos % ring_stride
    paged = slot * ring_stride + ring_idx

    tl.store(swa_indices_ptr + swa_base + i, paged, mask=mask)
    tl.store(csa_indices_ptr + csa_base + i, paged, mask=mask)
    tl.store(hca_indices_ptr + hca_base + i, paged, mask=mask)


def write_v4_paged_decode_indices(
    *,
    state_slot_per_seq: torch.Tensor,
    batch_id_per_token: torch.Tensor,
    positions: torch.Tensor,
    swa_indptr: torch.Tensor,
    csa_indptr: torch.Tensor,
    hca_indptr: torch.Tensor,
    swa_indices: torch.Tensor,
    csa_indices: torch.Tensor,
    hca_indices: torch.Tensor,
    T: int,
    win: int,
    ring_stride: int,
) -> None:

    if T == 0:
        return
    assert state_slot_per_seq.dim() == 1
    assert batch_id_per_token.dim() == 1 and batch_id_per_token.shape[0] >= T
    assert positions.dim() == 1 and positions.shape[0] >= T
    assert swa_indptr.dim() == 1 and swa_indptr.shape[0] >= T + 1
    assert csa_indptr.dim() == 1 and csa_indptr.shape[0] >= T + 1
    assert hca_indptr.dim() == 1 and hca_indptr.shape[0] >= T + 1
    assert swa_indices.dim() == 1
    assert csa_indices.dim() == 1
    assert hca_indices.dim() == 1

    BLOCK_N = triton.next_power_of_2(win)
    _v4_paged_decode_indices_kernel[(T,)](
        state_slot_per_seq,
        batch_id_per_token,
        positions,
        swa_indptr,
        csa_indptr,
        hca_indptr,
        swa_indices,
        csa_indices,
        hca_indices,
        ring_stride,
        win=win,
        BLOCK_N=BLOCK_N,
    )
