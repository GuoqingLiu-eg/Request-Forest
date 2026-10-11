# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


import torch
import triton
import triton.language as tl

from sglang.srt.utils.common import is_gfx95_supported


try:
    from aiter.ops.pa_sparse_prefill_opus import pa_sparse_prefill_opus

    _HAS_OPUS = is_gfx95_supported()
except ImportError:
    pa_sparse_prefill_opus = None
    _HAS_OPUS = False


@triton.jit
def _sparse_attn_v4_paged_prefill_kernel(
    q_ptr,
    unified_kv_ptr,
    kv_indices_prefix_ptr,
    kv_indptr_prefix_ptr,
    kv_ptr,
    kv_indices_extend_ptr,
    kv_indptr_extend_ptr,
    attn_sink_ptr,
    out_ptr,
    q_stride_t: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    pkv_stride_n: tl.constexpr,
    pkv_stride_d: tl.constexpr,
    ekv_stride_n: tl.constexpr,
    ekv_stride_d: tl.constexpr,
    out_stride_t: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_d: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    softmax_scale: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    t = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    d_offs = tl.arange(0, BLOCK_D)
    h_mask = h_offs < H
    d_mask = d_offs < D

    q = tl.load(
        q_ptr
        + t * q_stride_t
        + h_offs[:, None] * q_stride_h
        + d_offs[None, :] * q_stride_d,
        mask=h_mask[:, None] & d_mask[None, :],
        other=0.0,
    )

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    k_offs = tl.arange(0, BLOCK_K)


    p_start = tl.load(kv_indptr_prefix_ptr + t)
    p_end = tl.load(kv_indptr_prefix_ptr + t + 1)
    p_len = p_end - p_start

    for k_start in tl.range(0, p_len, BLOCK_K):
        k_pos = k_start + k_offs
        in_range = k_pos < p_len
        slot = tl.load(
            kv_indices_prefix_ptr + p_start + k_pos,
            mask=in_range,
            other=-1,
        )
        valid = in_range & (slot >= 0)
        slot_clamped = tl.maximum(slot, 0)

        kv = tl.load(
            unified_kv_ptr
            + slot_clamped[:, None] * pkv_stride_n
            + d_offs[None, :] * pkv_stride_d,
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(kv)) * softmax_scale
        scores = tl.where(h_mask[:, None] & valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(h_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new
        l_i = l_new


    e_start = tl.load(kv_indptr_extend_ptr + t)
    e_end = tl.load(kv_indptr_extend_ptr + t + 1)
    e_len = e_end - e_start

    for k_start in tl.range(0, e_len, BLOCK_K):
        k_pos = k_start + k_offs
        in_range = k_pos < e_len
        slot = tl.load(
            kv_indices_extend_ptr + e_start + k_pos,
            mask=in_range,
            other=-1,
        )
        valid = in_range & (slot >= 0)
        slot_clamped = tl.maximum(slot, 0)

        kv = tl.load(
            kv_ptr
            + slot_clamped[:, None] * ekv_stride_n
            + d_offs[None, :] * ekv_stride_d,
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(kv)) * softmax_scale
        scores = tl.where(h_mask[:, None] & valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(h_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new
        l_i = l_new


    sink = tl.load(attn_sink_ptr + h_offs, mask=h_mask, other=neg_large).to(tl.float32)
    m_final = tl.maximum(m_i, sink)
    alpha = tl.exp(m_i - m_final)
    l_final = l_i * alpha + tl.exp(sink - m_final)

    denom = tl.maximum(l_final, 1.0e-30)
    out = tl.where(l_final[:, None] > 0.0, (acc * alpha[:, None]) / denom[:, None], 0.0)
    tl.store(
        out_ptr
        + t * out_stride_t
        + h_offs[:, None] * out_stride_h
        + d_offs[None, :] * out_stride_d,
        out,
        mask=h_mask[:, None] & d_mask[None, :],
    )


def _sparse_attn_v4_paged_prefill_triton(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    if not q.is_cuda:
        raise RuntimeError(
            "Triton sparse_attn_v4_paged_prefill requires CUDA/HIP tensors"
        )
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(
            f"sparse_attn_v4_paged_prefill expects fp16/bf16 q, got {q.dtype}"
        )
    if unified_kv.dtype != q.dtype:
        raise RuntimeError(
            f"unified_kv dtype mismatch: kv={unified_kv.dtype}, q={q.dtype}"
        )
    if kv.dtype != q.dtype:
        raise RuntimeError(f"kv dtype mismatch: kv={kv.dtype}, q={q.dtype}")
    if unified_kv.size(-1) != kv.size(-1):
        raise RuntimeError(
            f"head_dim mismatch: unified_kv={unified_kv.size(-1)}, kv={kv.size(-1)}"
        )

    T, H, D = q.shape
    out = torch.empty_like(q)
    kv_indices_prefix = kv_indices_prefix.to(torch.int32).contiguous()
    kv_indptr_prefix = kv_indptr_prefix.to(torch.int32).contiguous()
    kv_indices_extend = kv_indices_extend.to(torch.int32).contiguous()
    kv_indptr_extend = kv_indptr_extend.to(torch.int32).contiguous()

    block_h = 16
    block_d = triton.next_power_of_2(D)
    block_k = 16 if D >= 256 else 32
    _sparse_attn_v4_paged_prefill_kernel[(T, triton.cdiv(H, block_h))](
        q,
        unified_kv,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        unified_kv.stride(0),
        unified_kv.stride(1),
        kv.stride(0),
        kv.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        H,
        D,
        float(softmax_scale),
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        num_warps=8,
    )
    return out


def sparse_attn_v4_paged_prefill(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:

    if _HAS_OPUS:


        q = q.contiguous()
        H = q.shape[1]
        if attn_sink.shape[0] != H:
            attn_sink = attn_sink[:H].contiguous()
        if (
            kv.stride(0) != unified_kv.stride(0)
            and kv.shape[0] == 1
            and kv.stride(1) == 1
        ):
            kv = kv.as_strided(kv.shape, (kv.shape[1], 1))
        return pa_sparse_prefill_opus(
            q,
            unified_kv,
            kv_indices_prefix,
            kv_indptr_prefix,
            kv,
            kv_indices_extend,
            kv_indptr_extend,
            attn_sink,
            softmax_scale,
        )
    return _sparse_attn_v4_paged_prefill_triton(
        q,
        unified_kv,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        softmax_scale,
    )
