# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

from sglang.kernels.ops.diffusion.common.numerics import round_bf16_to_fp32
from sglang.srt.utils.custom_op import register_custom_op


@triton.jit
def _silu_mul_kernel(
    out_ptr,
    a_ptr,
    b_ptr,
    numel,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = round_bf16_to_fp32(a * tl.sigmoid(a))
    tl.store(out_ptr + offs, s * b, mask=mask)


@triton.jit
def _packed_silu_mul_kernel(
    out_ptr,
    x_ptr,
    num_rows,
    row_stride,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1).to(tl.int64)
    cols = block * BLOCK + tl.arange(0, BLOCK)
    mask = (row < num_rows) & (cols < D)
    row_base = row * row_stride
    a = tl.load(x_ptr + row_base + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(x_ptr + row_base + D + cols, mask=mask, other=0.0).to(tl.float32)
    s = round_bf16_to_fp32(a * tl.sigmoid(a))
    tl.store(out_ptr + row * D + cols, s * b, mask=mask)


def can_use_fused_silu_mul(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.dtype is torch.bfloat16
        and b.dtype is torch.bfloat16
        and a.is_cuda
        and b.is_cuda
        and a.device == b.device
        and a.shape == b.shape
        and a.is_contiguous()
        and b.is_contiguous()
        and a.numel() > 0
    )


def _fake_silu_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(a)


@register_custom_op(
    op_name="triton_fused_silu_mul_bitexact",
    mutates_args=[],
    fake_impl=_fake_silu_mul,
)
def fused_silu_mul_bitexact(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:

    out = torch.empty_like(a)
    numel = a.numel()
    with torch.cuda.device(a.device):
        _silu_mul_kernel[(triton.cdiv(numel, 1024),)](
            out,
            a,
            b,
            numel,
            BLOCK=1024,
        )
    return out


def fused_packed_silu_mul_bitexact(x: torch.Tensor) -> torch.Tensor:

    if not (
        x.is_cuda
        and x.dtype is torch.bfloat16
        and x.dim() == 3
        and x.stride(-1) == 1
        and x.stride(-2) >= x.shape[-1]
        and x.stride(0) == x.shape[1] * x.stride(1)
        and x.shape[-1] % 2 == 0
        and x.numel() > 0
    ):
        raise RuntimeError("unsupported input for packed fused SiLU-mul")
    hidden = x.shape[-1] // 2
    rows = x.numel() // x.shape[-1]
    row_stride = x.stride(-2)
    out = torch.empty((*x.shape[:-1], hidden), dtype=x.dtype, device=x.device)
    with torch.cuda.device(x.device):
        _packed_silu_mul_kernel[(rows, triton.cdiv(hidden, 1024))](
            out,
            x,
            rows,
            row_stride,
            D=hidden,
            BLOCK=1024,
        )
    return out
