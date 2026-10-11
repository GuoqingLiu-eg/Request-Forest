from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.gemm.kernel_utils import _resolve_token_positions
from sglang.srt.lora.utils import LoRABatchInfo


_BLOCK_S = 16
_STEP_A_BLOCK_K = 64
_STEP_A_BLOCK_N = 16
_STEP_B_BLOCK_K = 16
_STEP_B_BLOCK_N = 64


def _num_segments(batch_info: LoRABatchInfo) -> int:
    return batch_info.num_segments or batch_info.bs


def _max_segment_len(batch_info: LoRABatchInfo) -> int:
    if batch_info.max_len is not None:
        return batch_info.max_len
    if batch_info.seg_lens is not None:
        return int(batch_info.seg_lens.max().item())
    raise ValueError("LoRA batch_info must provide max_len or seg_lens.")


def _segment_grid_size(batch_info: LoRABatchInfo, num_segments: int) -> int:
    return (
        batch_info.weight_indices.shape[0]
        if batch_info.use_cuda_graph
        else num_segments
    )


@triton.jit(do_not_specialize=["num_segments"])
def _step_a_q_kernel(
    x,
    w,
    out,

    S,
    H_FULL_K,
    K,
    N,

    x_stride_s,
    x_stride_h,
    x_stride_k,
    w_stride_l,
    w_stride_n,
    w_stride_k,
    out_stride_s,
    out_stride_h,
    out_stride_n,

    seg_indptr,
    weight_indices,
    lora_ranks,
    sorted_token_ids,
    num_segments,

    FULL_K: tl.constexpr,
    SORTED_BY_ADAPTER: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch_id = tl.program_id(axis=2)
    head_id = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)

    if batch_id >= num_segments:
        return

    w_index = tl.load(weight_indices + batch_id)
    cur_rank = tl.load(lora_ranks + w_index)
    if cur_rank == 0:
        return

    seg_start = tl.load(seg_indptr + batch_id)
    seg_end = tl.load(seg_indptr + batch_id + 1)
    seg_len = seg_end - seg_start
    if seg_len == 0:
        return


    N_eff = tl.minimum(N, cur_rank)

    num_pid_n = tl.cdiv(N_eff, BLOCK_N)
    pid_s = pid // num_pid_n
    pid_n = pid % num_pid_n
    if pid_s * BLOCK_S >= seg_len:
        return

    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K)

    s_physical = _resolve_token_positions(
        sorted_token_ids, seg_start, s_offset, seg_len, SORTED_BY_ADAPTER
    )


    row_mask = s_offset < seg_len
    safe_row = tl.minimum(s_physical, S - 1)
    safe_n = tl.minimum(n_offset, N_eff - 1)

    head_row_base = (
        head_id * FULL_K
    )

    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        cur_k = k_block * BLOCK_K + k_offset
        k_mask = cur_k < K
        safe_k = tl.minimum(cur_k, K - 1)


        x_tile = tl.load(
            x
            + safe_row[:, None] * x_stride_s
            + head_id * x_stride_h
            + safe_k[None, :] * x_stride_k,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )


        w_tile = tl.load(
            w
            + w_index * w_stride_l
            + (head_row_base + safe_k[:, None]) * w_stride_n
            + safe_n[None, :] * w_stride_k,
            mask=k_mask[:, None] & (n_offset[None, :] < N_eff),
            other=0.0,
        )

        partial_sum += tl.dot(x_tile, w_tile)

    partial_sum = partial_sum.to(x.dtype.element_ty)
    out_offs = (
        safe_row[:, None] * out_stride_s
        + head_id * out_stride_h
        + safe_n[None, :] * out_stride_n
    )
    out_mask = row_mask[:, None] & (n_offset[None, :] < N_eff)
    tl.store(out + out_offs, partial_sum, mask=out_mask)


def step_a_q_fwd(
    q_nope: torch.Tensor,
    B_buf: torch.Tensor,
    batch_info: LoRABatchInfo,
    full_K_per_head: int,
) -> torch.Tensor:

    S, H, qk_nope_dim = q_nope.shape
    rank = B_buf.shape[-1]
    out = torch.empty((S, H, rank), device=q_nope.device, dtype=q_nope.dtype)
    num_segments = _num_segments(batch_info)
    max_segment_len = _max_segment_len(batch_info)
    segment_grid = _segment_grid_size(batch_info, num_segments)

    grid = (
        triton.cdiv(max_segment_len, _BLOCK_S) * triton.cdiv(rank, _STEP_A_BLOCK_N),
        H,
        segment_grid,
    )
    sorted_by_adapter = batch_info.permutation is not None

    _step_a_q_kernel[grid](
        q_nope,
        B_buf,
        out,
        S,
        H * full_K_per_head,
        qk_nope_dim,
        rank,
        q_nope.stride(0),
        q_nope.stride(1),
        q_nope.stride(2),
        B_buf.stride(0),
        B_buf.stride(1),
        B_buf.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        batch_info.permutation,
        segment_grid,
        FULL_K=full_K_per_head,
        SORTED_BY_ADAPTER=sorted_by_adapter,
        BLOCK_S=_BLOCK_S,
        BLOCK_N=_STEP_A_BLOCK_N,
        BLOCK_K=_STEP_A_BLOCK_K,
    )
    return out


@triton.jit(do_not_specialize=["num_segments"])
def _step_b_q_kernel(
    x,
    w,
    base,

    S,
    K,
    N,

    x_stride_s,
    x_stride_h,
    x_stride_k,
    w_stride_l,
    w_stride_k,
    w_stride_n,
    b_stride_s,
    b_stride_h,
    b_stride_n,

    seg_indptr,
    weight_indices,
    lora_ranks,
    sorted_token_ids,
    scalings,
    num_segments,

    SORTED_BY_ADAPTER: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch_id = tl.program_id(axis=2)
    head_id = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)

    if batch_id >= num_segments:
        return

    w_index = tl.load(weight_indices + batch_id)
    cur_rank = tl.load(lora_ranks + w_index)
    if cur_rank == 0:
        return

    seg_start = tl.load(seg_indptr + batch_id)
    seg_end = tl.load(seg_indptr + batch_id + 1)
    seg_len = seg_end - seg_start
    if seg_len == 0:
        return
    scaling = tl.load(scalings + w_index)


    K_eff = tl.minimum(K, cur_rank)

    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_s = pid // num_pid_n
    pid_n = pid % num_pid_n
    if pid_s * BLOCK_S >= seg_len:
        return

    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K)

    s_physical = _resolve_token_positions(
        sorted_token_ids, seg_start, s_offset, seg_len, SORTED_BY_ADAPTER
    )

    row_mask = s_offset < seg_len
    safe_row = tl.minimum(s_physical, S - 1)
    n_mask = n_offset[None, :] < N
    safe_n = tl.minimum(n_offset, N - 1)

    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    for k_block in range(0, tl.cdiv(K_eff, BLOCK_K)):
        cur_k = k_block * BLOCK_K + k_offset
        k_mask = cur_k < K_eff
        safe_k = tl.minimum(cur_k, K_eff - 1)


        x_tile = tl.load(
            x
            + safe_row[:, None] * x_stride_s
            + head_id * x_stride_h
            + safe_k[None, :] * x_stride_k,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )


        w_tile = tl.load(
            w
            + w_index * w_stride_l
            + safe_k[:, None] * w_stride_k
            + safe_n[None, :] * w_stride_n,
            mask=k_mask[:, None] & n_mask,
            other=0.0,
        )

        partial_sum += tl.dot(x_tile, w_tile)

    partial_sum *= scaling
    partial_sum = partial_sum.to(x.dtype.element_ty)


    base_offs = (
        safe_row[:, None] * b_stride_s
        + head_id * b_stride_h
        + safe_n[None, :] * b_stride_n
    )
    out_mask = row_mask[:, None] & n_mask
    partial_sum += tl.load(base + base_offs, mask=out_mask, other=0.0)
    tl.store(base + base_offs, partial_sum, mask=out_mask)


def step_b_q_fwd(
    q_lora_a: torch.Tensor,
    A_buf: torch.Tensor,
    batch_info: LoRABatchInfo,
    base_output: torch.Tensor,
) -> torch.Tensor:

    S, H, rank = q_lora_a.shape
    kv_lora_rank = A_buf.shape[-1]
    num_segments = _num_segments(batch_info)
    max_segment_len = _max_segment_len(batch_info)
    segment_grid = _segment_grid_size(batch_info, num_segments)

    grid = (
        triton.cdiv(max_segment_len, _BLOCK_S)
        * triton.cdiv(kv_lora_rank, _STEP_B_BLOCK_N),
        H,
        segment_grid,
    )
    sorted_by_adapter = batch_info.permutation is not None

    _step_b_q_kernel[grid](
        q_lora_a,
        A_buf,
        base_output,
        S,
        rank,
        kv_lora_rank,
        q_lora_a.stride(0),
        q_lora_a.stride(1),
        q_lora_a.stride(2),
        A_buf.stride(0),
        A_buf.stride(1),
        A_buf.stride(2),
        base_output.stride(0),
        base_output.stride(1),
        base_output.stride(2),
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        batch_info.permutation,
        batch_info.scalings,
        segment_grid,
        SORTED_BY_ADAPTER=sorted_by_adapter,
        BLOCK_S=_BLOCK_S,
        BLOCK_N=_STEP_B_BLOCK_N,
        BLOCK_K=_STEP_B_BLOCK_K,
    )
    return base_output


@triton.jit(do_not_specialize=["num_segments"])
def _step_a_v_kernel(
    x,
    w,
    out,

    S,
    K,
    N,

    x_stride_s,
    x_stride_h,
    x_stride_k,
    w_stride_l,
    w_stride_n,
    w_stride_k,
    out_stride_s,
    out_stride_h,
    out_stride_n,

    seg_indptr,
    weight_indices,
    lora_ranks,
    sorted_token_ids,
    num_segments,

    SORTED_BY_ADAPTER: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch_id = tl.program_id(axis=2)
    head_id = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)

    if batch_id >= num_segments:
        return

    w_index = tl.load(weight_indices + batch_id)
    cur_rank = tl.load(lora_ranks + w_index)
    if cur_rank == 0:
        return

    seg_start = tl.load(seg_indptr + batch_id)
    seg_end = tl.load(seg_indptr + batch_id + 1)
    seg_len = seg_end - seg_start
    if seg_len == 0:
        return


    N_eff = tl.minimum(N, cur_rank)

    num_pid_n = tl.cdiv(N_eff, BLOCK_N)
    pid_s = pid // num_pid_n
    pid_n = pid % num_pid_n
    if pid_s * BLOCK_S >= seg_len:
        return

    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K)

    s_physical = _resolve_token_positions(
        sorted_token_ids, seg_start, s_offset, seg_len, SORTED_BY_ADAPTER
    )

    row_mask = s_offset < seg_len
    safe_row = tl.minimum(s_physical, S - 1)
    safe_n = tl.minimum(n_offset, N_eff - 1)

    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        cur_k = k_block * BLOCK_K + k_offset
        k_mask = cur_k < K
        safe_k = tl.minimum(cur_k, K - 1)


        x_tile = tl.load(
            x
            + safe_row[:, None] * x_stride_s
            + head_id * x_stride_h
            + safe_k[None, :] * x_stride_k,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )


        w_tile = tl.load(
            w
            + w_index * w_stride_l
            + safe_k[:, None] * w_stride_k
            + safe_n[None, :] * w_stride_n,
            mask=k_mask[:, None] & (n_offset[None, :] < N_eff),
            other=0.0,
        )

        partial_sum += tl.dot(x_tile, w_tile)

    partial_sum = partial_sum.to(x.dtype.element_ty)
    out_offs = (
        safe_row[:, None] * out_stride_s
        + head_id * out_stride_h
        + safe_n[None, :] * out_stride_n
    )
    out_mask = row_mask[:, None] & (n_offset[None, :] < N_eff)
    tl.store(out + out_offs, partial_sum, mask=out_mask)


def step_a_v_fwd(
    attn_output: torch.Tensor,
    A_buf: torch.Tensor,
    batch_info: LoRABatchInfo,
) -> torch.Tensor:

    S, H, kv_lora_rank = attn_output.shape
    rank = A_buf.shape[1]
    out = torch.empty((S, H, rank), device=attn_output.device, dtype=attn_output.dtype)
    num_segments = _num_segments(batch_info)
    max_segment_len = _max_segment_len(batch_info)
    segment_grid = _segment_grid_size(batch_info, num_segments)

    grid = (
        triton.cdiv(max_segment_len, _BLOCK_S) * triton.cdiv(rank, _STEP_A_BLOCK_N),
        H,
        segment_grid,
    )
    sorted_by_adapter = batch_info.permutation is not None

    _step_a_v_kernel[grid](
        attn_output,
        A_buf,
        out,
        S,
        kv_lora_rank,
        rank,
        attn_output.stride(0),
        attn_output.stride(1),
        attn_output.stride(2),
        A_buf.stride(0),
        A_buf.stride(1),
        A_buf.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        batch_info.permutation,
        segment_grid,
        SORTED_BY_ADAPTER=sorted_by_adapter,
        BLOCK_S=_BLOCK_S,
        BLOCK_N=_STEP_A_BLOCK_N,
        BLOCK_K=_STEP_A_BLOCK_K,
    )
    return out


@triton.jit(do_not_specialize=["num_segments"])
def _step_b_v_kernel(
    x,
    w,
    base,

    S,
    K,
    N,

    x_stride_s,
    x_stride_h,
    x_stride_k,
    w_stride_l,
    w_stride_n,
    w_stride_k,
    b_stride_s,
    b_stride_h,
    b_stride_n,

    seg_indptr,
    weight_indices,
    lora_ranks,
    sorted_token_ids,
    scalings,
    num_segments,

    FULL_K: tl.constexpr,
    QK_NOPE_OFFSET: tl.constexpr,
    SORTED_BY_ADAPTER: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch_id = tl.program_id(axis=2)
    head_id = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)

    if batch_id >= num_segments:
        return

    w_index = tl.load(weight_indices + batch_id)
    cur_rank = tl.load(lora_ranks + w_index)
    if cur_rank == 0:
        return

    seg_start = tl.load(seg_indptr + batch_id)
    seg_end = tl.load(seg_indptr + batch_id + 1)
    seg_len = seg_end - seg_start
    if seg_len == 0:
        return
    scaling = tl.load(scalings + w_index)

    K_eff = tl.minimum(K, cur_rank)

    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_s = pid // num_pid_n
    pid_n = pid % num_pid_n
    if pid_s * BLOCK_S >= seg_len:
        return

    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K)

    s_physical = _resolve_token_positions(
        sorted_token_ids, seg_start, s_offset, seg_len, SORTED_BY_ADAPTER
    )

    row_mask = s_offset < seg_len
    safe_row = tl.minimum(s_physical, S - 1)
    n_mask = n_offset[None, :] < N
    safe_n = tl.minimum(n_offset, N - 1)


    head_row_base = head_id * FULL_K + QK_NOPE_OFFSET

    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    for k_block in range(0, tl.cdiv(K_eff, BLOCK_K)):
        cur_k = k_block * BLOCK_K + k_offset
        k_mask = cur_k < K_eff
        safe_k = tl.minimum(cur_k, K_eff - 1)


        x_tile = tl.load(
            x
            + safe_row[:, None] * x_stride_s
            + head_id * x_stride_h
            + safe_k[None, :] * x_stride_k,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )


        w_tile = tl.load(
            w
            + w_index * w_stride_l
            + safe_k[:, None] * w_stride_k
            + (head_row_base + safe_n[None, :]) * w_stride_n,
            mask=k_mask[:, None] & n_mask,
            other=0.0,
        )

        partial_sum += tl.dot(x_tile, w_tile)

    partial_sum *= scaling
    partial_sum = partial_sum.to(x.dtype.element_ty)

    base_offs = (
        safe_row[:, None] * b_stride_s
        + head_id * b_stride_h
        + safe_n[None, :] * b_stride_n
    )
    out_mask = row_mask[:, None] & n_mask
    partial_sum += tl.load(base + base_offs, mask=out_mask, other=0.0)
    tl.store(base + base_offs, partial_sum, mask=out_mask)


def step_b_v_fwd(
    attn_lora_a: torch.Tensor,
    B_buf: torch.Tensor,
    batch_info: LoRABatchInfo,
    base_output: torch.Tensor,
    qk_nope_head_dim: int,
    v_head_dim: int,
) -> torch.Tensor:

    S, H, rank = attn_lora_a.shape
    full_K_per_head = qk_nope_head_dim + v_head_dim
    num_segments = _num_segments(batch_info)
    max_segment_len = _max_segment_len(batch_info)
    segment_grid = _segment_grid_size(batch_info, num_segments)

    grid = (
        triton.cdiv(max_segment_len, _BLOCK_S)
        * triton.cdiv(v_head_dim, _STEP_B_BLOCK_N),
        H,
        segment_grid,
    )
    sorted_by_adapter = batch_info.permutation is not None

    _step_b_v_kernel[grid](
        attn_lora_a,
        B_buf,
        base_output,
        S,
        rank,
        v_head_dim,
        attn_lora_a.stride(0),
        attn_lora_a.stride(1),
        attn_lora_a.stride(2),
        B_buf.stride(0),
        B_buf.stride(1),
        B_buf.stride(2),
        base_output.stride(0),
        base_output.stride(1),
        base_output.stride(2),
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        batch_info.permutation,
        batch_info.scalings,
        segment_grid,
        FULL_K=full_K_per_head,
        QK_NOPE_OFFSET=qk_nope_head_dim,
        SORTED_BY_ADAPTER=sorted_by_adapter,
        BLOCK_S=_BLOCK_S,
        BLOCK_N=_STEP_B_BLOCK_N,
        BLOCK_K=_STEP_B_BLOCK_K,
    )
    return base_output
