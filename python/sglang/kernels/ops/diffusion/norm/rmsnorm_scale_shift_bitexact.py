# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from typing import Tuple

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

from sglang.kernels.ops.diffusion.common.numerics import (
    mul_rn_f32,
    round_bf16_to_fp32,
    rsqrt_approx_f32,
)
from sglang.kernels.ops.diffusion.common.platform import is_cuda
from sglang.srt.utils.custom_op import register_custom_op


@triton.jit
def _fold_adjacent(p, rows: tl.constexpr, width: tl.constexpr):


    a, b = tl.split(tl.reshape(p, (rows, width, 2)))
    return a + b


@triton.jit
def _rmsnorm_scale_shift_kernel(
    out_ptr,
    res_out_ptr,
    x_ptr,
    residual_ptr,
    gate_ptr,
    weight_ptr,
    scale_ptr,
    shift_ptr,
    seq_len,
    eps,
    D: tl.constexpr,
    TPR: tl.constexpr,
    WPR: tl.constexpr,
    HAS_GATE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    batch = row // seq_len
    row_base = row * D
    vec_base = batch * D


    tx = tl.arange(0, TPR) * 8
    acc = tl.zeros((TPR,), dtype=tl.float32)
    for b in tl.static_range(8):
        for v in tl.static_range(8):
            col = tx + (b * 8 * TPR + v)
            if HAS_GATE:
                rj = tl.load(residual_ptr + row_base + col).to(tl.float32)
                uj = tl.load(x_ptr + row_base + col).to(tl.float32)
                gj = tl.load(gate_ptr + vec_base + col).to(tl.float32)

                xj = round_bf16_to_fp32(rj + round_bf16_to_fp32(gj * uj))
            else:
                xj = tl.load(x_ptr + row_base + col).to(tl.float32)
            acc = acc + mul_rn_f32(xj, xj)


    p = tl.reshape(acc, (WPR, 32))
    p = _fold_adjacent(p, WPR, 16)
    p = _fold_adjacent(p, WPR, 8)
    p = _fold_adjacent(p, WPR, 4)
    p = _fold_adjacent(p, WPR, 2)
    p = _fold_adjacent(p, WPR, 1)
    s = tl.reshape(p, (1, WPR))
    if WPR == 2:
        s = _fold_adjacent(s, 1, 1)
    rcp = tl.sum(rsqrt_approx_f32(s / D + eps))


    for i in tl.static_range(D // 1024):
        cols = i * 1024 + tl.arange(0, 1024)
        if HAS_GATE:
            r = tl.load(residual_ptr + row_base + cols).to(tl.float32)
            u = tl.load(x_ptr + row_base + cols).to(tl.float32)
            g = tl.load(gate_ptr + vec_base + cols).to(tl.float32)
            xin = round_bf16_to_fp32(r + round_bf16_to_fp32(g * u))
            tl.store(res_out_ptr + row_base + cols, xin)
        else:
            xin = tl.load(x_ptr + row_base + cols).to(tl.float32)
        w = tl.load(weight_ptr + cols).to(tl.float32)
        sc = tl.load(scale_ptr + vec_base + cols).to(tl.float32)
        sh = tl.load(shift_ptr + vec_base + cols).to(tl.float32)
        y = round_bf16_to_fp32(xin * rcp * w)
        one_plus = round_bf16_to_fp32(1.0 + sc)
        prod = round_bf16_to_fp32(y * one_plus)
        tl.store(out_ptr + row_base + cols, prod + sh)


def _threads_per_row(hidden: int) -> int | None:


    tpr = 32 if hidden <= 3072 else 64 if hidden <= 6144 else None
    if tpr is None or hidden != 64 * tpr:
        return None
    return tpr


def _is_row_broadcast(t: torch.Tensor, x: torch.Tensor) -> bool:
    return (
        t.dtype is torch.bfloat16
        and t.is_cuda
        and t.device == x.device
        and t.shape == (x.shape[0], 1, x.shape[-1])
        and t.is_contiguous()
    )


def can_use_fused_rmsnorm_scale_shift(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> bool:
    return (


        is_cuda()
        and x.dtype is torch.bfloat16
        and x.is_cuda
        and x.dim() == 3
        and x.is_contiguous()
        and _threads_per_row(x.shape[-1]) is not None
        and weight.dtype is torch.bfloat16
        and weight.is_cuda
        and weight.device == x.device
        and weight.shape == (x.shape[-1],)
        and weight.is_contiguous()
        and _is_row_broadcast(scale, x)
        and _is_row_broadcast(shift, x)
    )


def can_use_fused_scale_residual_rmsnorm_scale_shift(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> bool:
    return (
        can_use_fused_rmsnorm_scale_shift(residual, weight, scale, shift)
        and update.dtype is torch.bfloat16
        and update.is_cuda
        and update.device == residual.device
        and update.shape == residual.shape
        and update.is_contiguous()
        and _is_row_broadcast(gate, residual)
    )


def _fake_norm_scale_shift(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.empty_like(x)


@register_custom_op(
    op_name="triton_fused_rmsnorm_scale_shift_bitexact",
    mutates_args=[],
    fake_impl=_fake_norm_scale_shift,
)
def fused_rmsnorm_scale_shift_bitexact(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> torch.Tensor:

    batch, seq_len, hidden = x.shape
    tpr = _threads_per_row(hidden)
    out = torch.empty_like(x)
    with torch.cuda.device(x.device):
        _rmsnorm_scale_shift_kernel[(batch * seq_len,)](
            out,
            out,
            x,
            x,
            x,
            weight,
            scale,
            shift,
            seq_len,
            eps,
            D=hidden,
            TPR=tpr,
            WPR=tpr // 32,
            HAS_GATE=False,


            num_warps=max(tpr // 32, 1),
        )
    return out


def _fake_scale_residual_norm_scale_shift(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(residual), torch.empty_like(residual)


@register_custom_op(
    op_name="triton_fused_scale_residual_rmsnorm_scale_shift_bitexact",
    mutates_args=[],
    fake_impl=_fake_scale_residual_norm_scale_shift,
)
def fused_scale_residual_rmsnorm_scale_shift_bitexact(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:

    batch, seq_len, hidden = residual.shape
    tpr = _threads_per_row(hidden)
    out = torch.empty_like(residual)
    res_out = torch.empty_like(residual)
    with torch.cuda.device(residual.device):
        _rmsnorm_scale_shift_kernel[(batch * seq_len,)](
            out,
            res_out,
            update,
            residual,
            gate,
            weight,
            scale,
            shift,
            seq_len,
            eps,
            D=hidden,
            TPR=tpr,
            WPR=tpr // 32,
            HAS_GATE=True,
            num_warps=max(tpr // 32, 1),
        )
    return out, res_out
