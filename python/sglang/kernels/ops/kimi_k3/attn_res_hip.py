from __future__ import annotations

from functools import cache
from typing import Optional

import torch
import triton
import triton.language as tl


MAX_REGISTER_TILE: int = 8 * 8192


@cache
def supports_attn_res_hip(hidden_size: int, nvb: int) -> bool:

    return _tile_size(hidden_size, nvb) <= MAX_REGISTER_TILE


def _tile_size(hidden_size: int, nvb: int) -> int:
    return triton.next_power_of_2(max(nvb, 1)) * triton.next_power_of_2(hidden_size)


@triton.jit
def _agg_kernel(
    prefix_ptr,
    addend_ptr,
    prefix_out_ptr,
    bank_ptr,
    cw_ptr,
    ow_ptr,
    out_ptr,
    score_eps,
    out_eps,
    stride_pm,
    stride_am,
    stride_om,
    stride_bm,
    stride_bb,
    stride_o,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NVB: tl.constexpr,
    R_PAD: tl.constexpr,
    HAS_ADD: tl.constexpr,
    WRITE_BANK: tl.constexpr,
    APPLY_OUT_NORM: tl.constexpr,
):

    t = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H


    row = tl.load(prefix_ptr + t * stride_pm + offs, mask=mask, other=0.0)
    if HAS_ADD:


        row = (
            row.to(tl.float32)
            + tl.load(addend_ptr + t * stride_am + offs, mask=mask, other=0.0).to(
                tl.float32
            )
        ).to(prefix_out_ptr.dtype.element_ty)
        tl.store(prefix_out_ptr + t * stride_om + offs, row, mask=mask)
    if WRITE_BANK:
        tl.store(bank_ptr + t * stride_bm + NVB * stride_bb + offs, row, mask=mask)
    pv = row.to(tl.float32)

    cw = tl.load(cw_ptr + offs, mask=mask, other=0.0)
    p_score = tl.sum(pv * cw) / tl.sqrt(tl.sum(pv * pv) / H + score_eps)


    offs_r = tl.arange(0, R_PAD)
    mask_r = offs_r < NVB
    tile = tl.load(
        bank_ptr + t * stride_bm + offs_r[:, None] * stride_bb + offs[None, :],
        mask=mask_r[:, None] & mask[None, :],
        other=0.0,
    ).to(tl.float32)
    b_score = tl.sum(tile * cw[None, :], axis=1) / tl.sqrt(
        tl.sum(tile * tile, axis=1) / H + score_eps
    )

    m = tl.maximum(tl.max(tl.where(mask_r, b_score, -float("inf"))), p_score)
    b_w = tl.where(mask_r, tl.exp(b_score - m), 0.0)
    p_w = tl.exp(p_score - m)
    inv = 1.0 / (tl.sum(b_w) + p_w)

    acc = pv * (p_w * inv) + tl.sum(b_w[:, None] * inv * tile, axis=0)

    if APPLY_OUT_NORM:
        scale = 1.0 / tl.sqrt(tl.sum(acc * acc) / H + out_eps)
        ow = tl.load(ow_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        acc = acc * scale * ow
    tl.store(out_ptr + t * stride_o + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def attn_res_hip(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    cw: torch.Tensor,
    ow: Optional[torch.Tensor],
    out: torch.Tensor,
    nvb: int,
    score_eps: float,
    out_eps: float,
    *,
    addend: Optional[torch.Tensor] = None,
    prefix_out: Optional[torch.Tensor] = None,
    write_prefix: bool = False,
) -> None:

    T, H = prefix_sum.shape
    assert nvb >= 1, "nvb == 0 has nothing to aggregate; the caller must handle it"
    assert supports_attn_res_hip(H, nvb), (
        f"attn_res_hip: register tile {_tile_size(H, nvb)} exceeds "
        f"{MAX_REGISTER_TILE} (H={H}, nvb={nvb})"
    )
    has_add = addend is not None
    assert not has_add or prefix_out is not None, "addend requires prefix_out"


    addend_arg = addend if has_add else prefix_sum
    prefix_out_arg = prefix_out if has_add else prefix_sum
    ow_arg = ow if ow is not None else cw

    _agg_kernel[(T,)](
        prefix_sum,
        addend_arg,
        prefix_out_arg,
        bank,
        cw,
        ow_arg,
        out,
        score_eps,
        out_eps,
        prefix_sum.stride(0),
        addend_arg.stride(0),
        prefix_out_arg.stride(0),
        bank.stride(0),
        bank.stride(1),
        out.stride(0),
        H=H,
        BLOCK_H=triton.next_power_of_2(H),
        NVB=nvb,
        R_PAD=triton.next_power_of_2(nvb),
        HAS_ADD=has_add,
        WRITE_BANK=write_prefix,
        APPLY_OUT_NORM=ow is not None,
        num_warps=4,
    )
