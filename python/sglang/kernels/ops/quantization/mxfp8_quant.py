from __future__ import annotations

from typing import NamedTuple

import torch
import triton
import triton.language as tl

MXFP8_BLOCK_SIZE = 32


class MXFP8Tensor(NamedTuple):
    data: torch.Tensor
    scale: torch.Tensor


@triton.jit
def _mxfp8_quant_kernel(
    x_ptr,
    xq_ptr,
    s_ptr,
    M,
    K,
    sxm,
    sxk,
    sqm,
    sqk,
    ssm,
    ssk,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_b * 32 + tl.arange(0, 32)
    m_mask = offs_m < M

    x = tl.load(
        x_ptr + offs_m[:, None] * sxm + offs_k[None, :] * sxk,
        mask=m_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-30)
    scale_biased = tl.ceil(tl.log2(amax / 448.0)) + 127.0
    scale_biased = tl.minimum(tl.maximum(scale_biased, 0.0), 254.0)
    descale = tl.exp2(scale_biased - 127.0)
    xq = tl.clamp(x / descale[:, None], -448.0, 448.0).to(xq_ptr.dtype.element_ty)

    tl.store(
        xq_ptr + offs_m[:, None] * sqm + offs_k[None, :] * sqk,
        xq,
        mask=m_mask[:, None],
    )
    tl.store(
        s_ptr + offs_m * ssm + pid_b * ssk,
        scale_biased.to(tl.uint8),
        mask=m_mask,
    )


def to_mxfp8(x: torch.Tensor) -> MXFP8Tensor:

    if x.shape[-1] % MXFP8_BLOCK_SIZE != 0:
        raise ValueError(
            f"MXFP8 quantization requires last dim divisible by {MXFP8_BLOCK_SIZE}, "
            f"got {x.shape[-1]}."
        )
    orig_shape = x.shape
    x2d = x.contiguous().view(-1, orig_shape[-1])
    M, K = x2d.shape
    xq = torch.empty_like(x2d, dtype=torch.float8_e4m3fn)
    scales = torch.empty((M, K // MXFP8_BLOCK_SIZE), dtype=torch.uint8, device=x.device)
    block_m = 64
    grid = (triton.cdiv(M, block_m), K // MXFP8_BLOCK_SIZE)
    _mxfp8_quant_kernel[grid](
        x2d,
        xq,
        scales,
        M,
        K,
        x2d.stride(0),
        x2d.stride(1),
        xq.stride(0),
        xq.stride(1),
        scales.stride(0),
        scales.stride(1),
        BLOCK_M=block_m,
    )
    return MXFP8Tensor(
        data=xq.view(orig_shape),
        scale=scales.view(*orig_shape[:-1], K // MXFP8_BLOCK_SIZE),
    )


@triton.jit
def _mxfp8_quant_store_qkv_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    loc_ptr,
    q8_ptr,
    sfq_ptr,
    kc_ptr,
    vc_ptr,
    sfk_ptr,
    sfv_ptr,
    sqt,
    sqh,
    skt,
    skh,
    svt,
    svh,
    NQ: tl.constexpr,
    NKV: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
):
    t = tl.program_id(0)
    r = tl.program_id(1)
    SF: tl.constexpr = D // 32
    blk = tl.arange(0, SF)
    off2 = blk[:, None] * 32 + tl.arange(0, 32)[None, :]

    if r < NQ:
        src = q_ptr + t * sqt + r * sqh
    elif r < NQ + NKV:
        src = k_ptr + t * skt + (r - NQ) * skh
    else:
        src = v_ptr + t * svt + (r - NQ - NKV) * svh

    x = tl.load(src + off2).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-30)
    scale_biased = tl.ceil(tl.log2(amax / 448.0)) + 127.0
    scale_biased = tl.minimum(tl.maximum(scale_biased, 0.0), 254.0)
    descale = tl.exp2(scale_biased - 127.0)
    xq = tl.clamp(x / descale[:, None], -448.0, 448.0).to(q8_ptr.dtype.element_ty)
    sf = scale_biased.to(tl.uint8)

    if r < NQ:
        tl.store(q8_ptr + (t * NQ + r) * D + off2, xq)
        tl.store(sfq_ptr + (t * NQ + r) * SF + blk, sf)
    else:
        myloc = tl.load(loc_ptr + t).to(tl.int64)
        if r < NQ + NKV:
            h = r - NQ
            cache = kc_ptr
            sfb = sfk_ptr
        else:
            h = r - NQ - NKV
            cache = vc_ptr
            sfb = sfv_ptr
        tl.store(cache + (myloc * NKV + h) * D + off2, xq)


        po = myloc % PAGE
        base = (
            ((myloc // PAGE) * NKV + h) * (32 * (PAGE // 32) * SF)
            + (po % 32) * ((PAGE // 32) * SF)
            + (po // 32) * SF
        )
        tl.store(sfb + base + blk, sf)


def quant_store_kv_mxfp8(
    k: torch.Tensor,
    v: torch.Tensor,
    loc: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    sfk: torch.Tensor,
    sfv: torch.Tensor,
    page_size: int = 128,
) -> None:

    T, NKV, D = k.shape
    assert D % 32 == 0 and page_size % 32 == 0
    assert k.stride(2) == 1 and v.stride(2) == 1
    assert k_cache.is_contiguous() and v_cache.is_contiguous()
    if T == 0:
        return
    _mxfp8_quant_store_qkv_kernel[(T, 2 * NKV)](
        k,
        k,
        v,
        loc,
        k_cache,
        sfk.view(torch.uint8),
        k_cache,
        v_cache,
        sfk.view(torch.uint8),
        sfv.view(torch.uint8),
        0,
        0,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        NQ=0,
        NKV=NKV,
        D=D,
        PAGE=page_size,
    )


@triton.jit
def _mxfp8_v_cache_update_kernel(
    v_ptr,
    loc_ptr,
    cache_ptr,
    sf_ptr,
    N,
    svn,
    svh,
    svd,
    scp,
    scs,
    sch,
    scd,
    sfp,
    sfh,
    sfd,
    sfb,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SCAN_WINDOW: tl.constexpr,
):
    n = tl.program_id(0)
    h = tl.program_id(1)
    myloc = tl.load(loc_ptr + n).to(tl.int64)
    blk = myloc // 32


    is_leader = myloc >= 0
    j_lo = tl.maximum(n - SCAN_WINDOW, 0)
    for j in range(j_lo, n):
        lj = tl.load(loc_ptr + j).to(tl.int64)
        is_leader = is_leader & ~((lj >= 0) & (lj // 32 == blk))

    if is_leader:
        page = blk * 32 // PAGE_SIZE
        blk_in_page = (blk * 32) % PAGE_SIZE // 32
        d = tl.arange(0, HEAD_DIM)
        sf_addr = sf_ptr + page * sfp + h * sfh + d * sfd + blk_in_page * sfb
        e_old = tl.load(sf_addr).to(tl.int32)


        e_new = tl.zeros((HEAD_DIM,), dtype=tl.int32)
        has_start = False
        j_hi = tl.minimum(n + SCAN_WINDOW, N)
        for j in range(n, j_hi):
            lj = tl.load(loc_ptr + j).to(tl.int64)
            hit = (lj >= 0) & (lj // 32 == blk)
            v = tl.load(v_ptr + j * svn + h * svh + d * svd, mask=hit, other=0.0).to(
                tl.float32
            )
            amax = tl.maximum(tl.abs(v), 1e-30)
            e_tok = (tl.ceil(tl.log2(amax / 448.0)) + 127.0).to(tl.int32)
            e_tok = tl.minimum(tl.maximum(e_tok, 0), 254)
            e_new = tl.maximum(e_new, tl.where(hit, e_tok, 0))
            has_start = has_start | (hit & (lj % 32 == 0))
        e_old = tl.where(has_start, 0, e_old)
        e_blk = tl.maximum(e_old, e_new)


        s = tl.arange(0, 32)
        cache_addr = (
            cache_ptr
            + page * scp
            + (blk_in_page * 32 + s)[:, None] * scs
            + h * sch
            + d[None, :] * scd
        )
        old = tl.load(cache_addr).to(tl.float32)
        old = old * tl.exp2((e_old - e_blk).to(tl.float32))[None, :]
        tl.store(cache_addr, old.to(cache_ptr.dtype.element_ty))


        descale = tl.exp2((e_blk - 127).to(tl.float32))
        for j in range(n, j_hi):
            lj = tl.load(loc_ptr + j).to(tl.int64)
            hit = (lj >= 0) & (lj // 32 == blk)
            v = tl.load(v_ptr + j * svn + h * svh + d * svd, mask=hit, other=0.0).to(
                tl.float32
            )
            q = tl.clamp(v / descale, -448.0, 448.0)
            slot_in_page = lj % PAGE_SIZE
            tl.store(
                cache_ptr + page * scp + slot_in_page * scs + h * sch + d * scd,
                q.to(cache_ptr.dtype.element_ty),
                mask=hit,
            )

        tl.store(sf_addr, e_blk.to(tl.uint8))


def update_mxfp8_v_cache_seqblocked(
    v: torch.Tensor,
    loc: torch.Tensor,
    v_cache: torch.Tensor,
    sfv: torch.Tensor,
) -> None:

    N, h_kv, head_dim = v.shape
    num_pages, page_size = v_cache.shape[0], v_cache.shape[1]
    assert page_size % MXFP8_BLOCK_SIZE == 0
    assert v_cache.shape[2:] == (h_kv, head_dim)
    assert sfv.shape == (num_pages, h_kv, head_dim, page_size // MXFP8_BLOCK_SIZE)
    assert sfv.dtype == torch.uint8 and v_cache.dtype == torch.float8_e4m3fn
    if N == 0:
        return
    _mxfp8_v_cache_update_kernel[(N, h_kv)](
        v,
        loc,
        v_cache,
        sfv,
        N,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        sfv.stride(0),
        sfv.stride(1),
        sfv.stride(2),
        sfv.stride(3),
        PAGE_SIZE=page_size,
        HEAD_DIM=head_dim,
        SCAN_WINDOW=MXFP8_BLOCK_SIZE,
    )


def from_mxfp8(x: MXFP8Tensor, out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    num_blocks = x.data.shape[-1] // MXFP8_BLOCK_SIZE
    data = x.data.to(torch.float32).view(
        *x.data.shape[:-1], num_blocks, MXFP8_BLOCK_SIZE
    )
    scale_biased = x.scale.view(torch.uint8).to(torch.float32)
    descale = torch.exp2(scale_biased - 127.0)
    return (data * descale.unsqueeze(-1)).view_as(x.data).to(out_dtype)
