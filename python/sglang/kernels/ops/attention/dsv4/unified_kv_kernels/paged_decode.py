# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl
from aiter.ops.triton.utils.device_info import get_num_sms

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz

LOG2E = 1.4426950408889634
_MAX_KV_SPLITS = 64


_FP8_GROUP_SIZE = 64


_FP8_DTYPE = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn


@functools.lru_cache(maxsize=1)
def _cu_count() -> int:

    return get_num_sms()


def _kernel_config(block_h: int) -> tuple[int, int, int]:

    block_k = 16
    num_warps = 4 if block_h <= 32 else 8
    num_stages = 2
    return block_k, num_warps, num_stages


def _prev_pow2(n: int) -> int:

    if n < 1:
        return 1
    return 1 << (n.bit_length() - 1)


def _kv_splits_heuristic(
    T: int,
    H: int,
    block_h: int,
    num_cu: int | None = None,
    target_wg_per_cu: float = 2.0,
    max_kv_splits: int = _MAX_KV_SPLITS,
) -> int:

    if num_cu is None:
        num_cu = _cu_count()
    target_wg = max(1, int(target_wg_per_cu * num_cu))
    head_blocks = max(1, (H + block_h - 1) // block_h)
    base_ctas = max(1, T * head_blocks)
    if base_ctas >= target_wg:
        return 1

    splits_to_fill = max(1, target_wg // base_ctas)
    return _prev_pow2(min(splits_to_fill, max_kv_splits))


@triton.jit
def _paged_decode_fused_kernel(
    q_ptr,
    unified_kv_ptr,
    kv_scales_ptr,
    kv_indices_ptr,
    kv_indptr_ptr,
    attn_sink_ptr,
    out_ptr,
    q_stride_t,
    q_stride_h,
    q_stride_d,
    kv_stride_n,
    kv_stride_d,
    ks_stride_n,
    out_stride_t,
    out_stride_h,
    out_stride_d,
    qk_scale,
    log2e,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    QUANT_KV: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
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

    kv_start = tl.load(kv_indptr_ptr + t)
    kv_end = tl.load(kv_indptr_ptr + t + 1)
    kv_len = kv_end - kv_start
    num_tiles = tl.cdiv(kv_len, BLOCK_K)

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    k_offs = tl.arange(0, BLOCK_K)
    if QUANT_KV:


        g_idx_per_d = d_offs // GROUP_SIZE


    for j in tl.range(0, num_tiles, num_stages=3):
        k_start = j * BLOCK_K
        k_pos = k_start + k_offs
        valid = k_pos < kv_len
        slot = tl.load(
            kv_indices_ptr + kv_start + k_pos,
            mask=valid,
            other=0,
        )

        kv_raw = tl.load(
            unified_kv_ptr
            + slot[:, None] * kv_stride_n
            + d_offs[None, :] * kv_stride_d,
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        )
        if QUANT_KV:


            scales_full = tl.load(
                kv_scales_ptr + slot[:, None] * ks_stride_n + g_idx_per_d[None, :],
                mask=valid[:, None] & d_mask[None, :],
                other=0.0,
            ).to(q.dtype)
            kv = kv_raw.to(q.dtype) * scales_full
        else:
            kv = kv_raw

        scores = tl.dot(q, tl.trans(kv)) * qk_scale


        scores = tl.where(valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new
        l_i = l_new


    sink_raw = tl.load(attn_sink_ptr + h_offs, mask=h_mask, other=neg_large).to(
        tl.float32
    )
    sink = sink_raw * log2e
    m_final = tl.maximum(m_i, sink)
    alpha_kv = tl.exp2(m_i - m_final)
    alpha_sink = tl.exp2(sink - m_final)
    l_final = l_i * alpha_kv + alpha_sink

    denom = tl.maximum(l_final, 1.0e-30)
    out = tl.where(
        l_final[:, None] > 0.0, (acc * alpha_kv[:, None]) / denom[:, None], 0.0
    )
    tl.store(
        out_ptr
        + t * out_stride_t
        + h_offs[:, None] * out_stride_h
        + d_offs[None, :] * out_stride_d,
        out.to(out_ptr.dtype.element_ty),
        mask=h_mask[:, None] & d_mask[None, :],
    )


@triton.jit
def _paged_decode_split_kernel(
    q_ptr,
    unified_kv_ptr,
    kv_scales_ptr,
    kv_indices_ptr,
    kv_indptr_ptr,
    m_partial_ptr,
    l_partial_ptr,
    acc_partial_ptr,
    q_stride_t,
    q_stride_h,
    q_stride_d,
    kv_stride_n,
    kv_stride_d,
    ks_stride_n,
    mp_stride_t,
    mp_stride_k,
    mp_stride_h,
    lp_stride_t,
    lp_stride_k,
    lp_stride_h,
    ap_stride_t,
    ap_stride_k,
    ap_stride_h,
    ap_stride_d,
    H: tl.constexpr,
    D: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    qk_scale,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    QUANT_KV: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
):

    t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_k = tl.program_id(2)

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

    kv_start = tl.load(kv_indptr_ptr + t)
    kv_end = tl.load(kv_indptr_ptr + t + 1)
    kv_len = kv_end - kv_start


    tiles_per_segment = tl.cdiv(kv_len, KV_SPLITS * BLOCK_K)
    if pid_k * tiles_per_segment * BLOCK_K >= kv_len:
        return
    num_tiles = tl.cdiv(kv_len, BLOCK_K)
    tile_start = pid_k * tiles_per_segment
    tile_end = tl.minimum((pid_k + 1) * tiles_per_segment, num_tiles)

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    k_offs = tl.arange(0, BLOCK_K)
    if QUANT_KV:
        g_idx_per_d = d_offs // GROUP_SIZE

    for j in tl.range(tile_start, tile_end, num_stages=3):
        k_start = j * BLOCK_K
        k_pos = k_start + k_offs
        valid = k_pos < kv_len
        slot = tl.load(
            kv_indices_ptr + kv_start + k_pos,
            mask=valid,
            other=0,
        )

        kv_raw = tl.load(
            unified_kv_ptr
            + slot[:, None] * kv_stride_n
            + d_offs[None, :] * kv_stride_d,
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        )
        if QUANT_KV:
            scales_full = tl.load(
                kv_scales_ptr + slot[:, None] * ks_stride_n + g_idx_per_d[None, :],
                mask=valid[:, None] & d_mask[None, :],
                other=0.0,
            ).to(q.dtype)
            kv = kv_raw.to(q.dtype) * scales_full
        else:
            kv = kv_raw

        scores = tl.dot(q, tl.trans(kv)) * qk_scale

        scores = tl.where(valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new
        l_i = l_new

    m_base = t * mp_stride_t + pid_k * mp_stride_k
    tl.store(m_partial_ptr + m_base + h_offs * mp_stride_h, m_i, mask=h_mask)
    l_base = t * lp_stride_t + pid_k * lp_stride_k
    tl.store(l_partial_ptr + l_base + h_offs * lp_stride_h, l_i, mask=h_mask)
    a_base = t * ap_stride_t + pid_k * ap_stride_k
    tl.store(
        acc_partial_ptr
        + a_base
        + h_offs[:, None] * ap_stride_h
        + d_offs[None, :] * ap_stride_d,
        acc,
        mask=h_mask[:, None] & d_mask[None, :],
    )


@triton.jit
def _paged_decode_reduce_kernel(
    m_partial_ptr,
    l_partial_ptr,
    acc_partial_ptr,
    attn_sink_ptr,
    kv_indptr_ptr,
    out_ptr,
    mp_stride_t,
    mp_stride_k,
    mp_stride_h,
    lp_stride_t,
    lp_stride_k,
    lp_stride_h,
    ap_stride_t,
    ap_stride_k,
    ap_stride_h,
    ap_stride_d,
    out_stride_t,
    out_stride_h,
    out_stride_d,
    log2e,
    H: tl.constexpr,
    D: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D_CHUNK: tl.constexpr,
    BLOCK_K: tl.constexpr,
):

    t = tl.program_id(0)
    h = tl.program_id(1)
    dc = tl.program_id(2)

    d_offs = dc * D_CHUNK + tl.arange(0, D_CHUNK)
    k_offs = tl.arange(0, KV_SPLITS)
    d_mask = d_offs < D

    neg_large = -3.4028234663852886e38

    kv_start = tl.load(kv_indptr_ptr + t)
    kv_end = tl.load(kv_indptr_ptr + t + 1)
    kv_len = kv_end - kv_start


    if kv_len == 0:
        out_off = t * out_stride_t + h * out_stride_h + d_offs * out_stride_d
        tl.store(
            out_ptr + out_off,
            tl.zeros([D_CHUNK], dtype=out_ptr.dtype.element_ty),
            mask=d_mask,
        )
        return
    tiles_per_segment = tl.cdiv(kv_len, KV_SPLITS * BLOCK_K)
    act_num_segments = tl.cdiv(kv_len, tl.maximum(tiles_per_segment, 1) * BLOCK_K)
    segm_mask = k_offs < act_num_segments


    m_p = tl.load(
        m_partial_ptr + t * mp_stride_t + k_offs * mp_stride_k + h * mp_stride_h,
        mask=segm_mask,
        other=neg_large,
    )
    l_p = tl.load(
        l_partial_ptr + t * lp_stride_t + k_offs * lp_stride_k + h * lp_stride_h,
        mask=segm_mask,
        other=0.0,
    )


    a_p = tl.load(
        acc_partial_ptr
        + t * ap_stride_t
        + k_offs[:, None] * ap_stride_k
        + h * ap_stride_h
        + d_offs[None, :] * ap_stride_d,
        mask=segm_mask[:, None] & d_mask[None, :],
        other=0.0,
    )


    m_max = tl.max(m_p, axis=0)
    alpha_split = tl.exp2(m_p - m_max)
    l_combined = tl.sum(l_p * alpha_split, axis=0)
    acc_combined = tl.sum(a_p * alpha_split[:, None], axis=0)


    sink_raw = tl.load(attn_sink_ptr + h).to(tl.float32)
    sink = sink_raw * log2e
    m_final = tl.maximum(m_max, sink)
    alpha_kv = tl.exp2(m_max - m_final)
    alpha_sink = tl.exp2(sink - m_final)
    l_final = l_combined * alpha_kv + alpha_sink

    denom = tl.maximum(l_final, 1.0e-30)


    acc_final = acc_combined * alpha_kv
    out = tl.where(l_final > 0.0, acc_final / denom, 0.0)

    tl.store(
        out_ptr + t * out_stride_t + h * out_stride_h + d_offs * out_stride_d,
        out.to(out_ptr.dtype.element_ty),
        mask=d_mask,
    )


def _sparse_attn_v4_paged_decode_triton(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    kv_scales: torch.Tensor | None = None,
    block_h: int | None = None,
    kv_splits: int | None = None,
    block_k: int | None = None,
) -> torch.Tensor:

    if not q.is_cuda:
        raise RuntimeError(
            "Triton sparse_attn_v4_paged_decode requires CUDA/HIP tensors"
        )
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(
            f"sparse_attn_v4_paged_decode expects fp16/bf16 q, got {q.dtype}"
        )

    quant_kv = kv_scales is not None
    if quant_kv:
        if unified_kv.dtype != _FP8_DTYPE:
            raise RuntimeError(
                f"kv_scales supplied but unified_kv is {unified_kv.dtype}, "
                f"expected {_FP8_DTYPE}"
            )
        if kv_scales.dtype != torch.float32:
            raise RuntimeError(f"kv_scales must be fp32, got {kv_scales.dtype}")
        D_check = unified_kv.shape[-1]
        if D_check % _FP8_GROUP_SIZE != 0:
            raise RuntimeError(
                f"D={D_check} must be divisible by GROUP_SIZE={_FP8_GROUP_SIZE}"
            )
        expected_g = D_check // _FP8_GROUP_SIZE
        if kv_scales.shape != (unified_kv.shape[0], expected_g):
            raise RuntimeError(
                f"kv_scales shape {tuple(kv_scales.shape)} does not match "
                f"expected ({unified_kv.shape[0]}, {expected_g})"
            )
        if kv_scales.stride(-1) != 1:
            kv_scales = kv_scales.contiguous()
    else:
        if unified_kv.dtype != q.dtype:
            raise RuntimeError(
                f"unified_kv dtype mismatch: kv={unified_kv.dtype}, q={q.dtype}"
            )

    T, H, D = q.shape
    out = torch.empty_like(q)

    if block_h is None:
        block_h = triton.next_power_of_2(min(H, 64))
    else:
        block_h = triton.next_power_of_2(block_h)
    block_h = max(block_h, 16)

    n_head_blocks = (H + block_h - 1) // block_h
    h_padded = n_head_blocks * block_h
    block_d = triton.next_power_of_2(D)

    if kv_splits is None:
        kv_splits = _kv_splits_heuristic(T, H, block_h)

    qk_scale = float(softmax_scale) * LOG2E
    _bk, num_warps, num_stages = _kernel_config(block_h)
    if block_k is None:


        block_k = 32 if quant_kv else _bk


    if quant_kv:
        kv_scales_arg = kv_scales
        ks_stride_n_arg = kv_scales.stride(0)
        num_groups_arg = D // _FP8_GROUP_SIZE
    else:
        kv_scales_arg = q.new_empty(1, dtype=torch.float32)
        ks_stride_n_arg = 1


        num_groups_arg = 1


    if kv_splits == 1:
        grid_fused = (T, n_head_blocks)
        _paged_decode_fused_kernel[grid_fused](
            q,
            unified_kv,
            kv_scales_arg,
            kv_indices,
            kv_indptr,
            attn_sink,
            out,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            unified_kv.stride(0),
            unified_kv.stride(1),
            ks_stride_n_arg,
            out.stride(0),
            out.stride(1),
            out.stride(2),
            qk_scale,
            LOG2E,
            H,
            D,
            BLOCK_H=block_h,
            BLOCK_D=block_d,
            BLOCK_K=block_k,
            QUANT_KV=quant_kv,
            GROUP_SIZE=_FP8_GROUP_SIZE,
            NUM_GROUPS=num_groups_arg,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return out


    m_partial = torch.empty(
        (T, kv_splits, h_padded), dtype=torch.float32, device=q.device
    )
    l_partial = torch.empty_like(m_partial)
    acc_partial = torch.empty(
        (T, kv_splits, h_padded, D), dtype=torch.float32, device=q.device
    )

    grid_split = (T, n_head_blocks, kv_splits)
    _paged_decode_split_kernel[grid_split](
        q,
        unified_kv,
        kv_scales_arg,
        kv_indices,
        kv_indptr,
        m_partial,
        l_partial,
        acc_partial,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        unified_kv.stride(0),
        unified_kv.stride(1),
        ks_stride_n_arg,
        m_partial.stride(0),
        m_partial.stride(1),
        m_partial.stride(2),
        l_partial.stride(0),
        l_partial.stride(1),
        l_partial.stride(2),
        acc_partial.stride(0),
        acc_partial.stride(1),
        acc_partial.stride(2),
        acc_partial.stride(3),
        H,
        D,
        kv_splits,
        qk_scale,
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        QUANT_KV=quant_kv,
        GROUP_SIZE=_FP8_GROUP_SIZE,
        NUM_GROUPS=num_groups_arg,
        num_warps=num_warps,
        num_stages=num_stages,
    )


    base_grid_t_h = T * H
    target_reduce_wg = 2 * _cu_count()
    if base_grid_t_h >= target_reduce_wg:
        d_chunk = block_d
    else:
        d_chunks_needed = max(1, target_reduce_wg // base_grid_t_h)
        d_chunks_needed = min(d_chunks_needed, block_d // 32)
        d_chunk = max(32, triton.next_power_of_2(block_d // d_chunks_needed))
    grid_reduce = (T, H, (D + d_chunk - 1) // d_chunk)
    _paged_decode_reduce_kernel[grid_reduce](
        m_partial,
        l_partial,
        acc_partial,
        attn_sink,
        kv_indptr,
        out,
        m_partial.stride(0),
        m_partial.stride(1),
        m_partial.stride(2),
        l_partial.stride(0),
        l_partial.stride(1),
        l_partial.stride(2),
        acc_partial.stride(0),
        acc_partial.stride(1),
        acc_partial.stride(2),
        acc_partial.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        LOG2E,
        H,
        D,
        kv_splits,
        BLOCK_D=block_d,
        D_CHUNK=d_chunk,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return out


def sparse_attn_v4_paged_decode(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    kv_scales: torch.Tensor | None = None,
) -> torch.Tensor:

    return _sparse_attn_v4_paged_decode_triton(
        q,
        unified_kv,
        kv_indices,
        kv_indptr,
        attn_sink,
        softmax_scale,
        kv_scales=kv_scales,
    )
