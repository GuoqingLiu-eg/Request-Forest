# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

from sglang.kernels.jit.utils import get_jit_cuda_arch


@triton.jit
def _qk_rmsnorm_native_kernel(
    y_ptr,
    x_ptr,
    weight_ptr,
    token_stride,
    nheads,
    n_rows,
    head_dim: tl.constexpr,
    eps: tl.constexpr,
    rows_per_prog: tl.constexpr,
):

    prog = tl.program_id(0)
    row_offs = tl.arange(0, rows_per_prog)
    rows = prog * rows_per_prog + row_offs
    row_mask = rows < n_rows
    tokens = rows // nheads
    heads = rows % nheads
    base = x_ptr + tokens * token_stride + heads * head_dim

    offs = tl.arange(0, head_dim)
    x = tl.load(base[:, None] + offs[None, :], mask=row_mask[:, None], other=0.0)
    sq = (x * x).to(tl.bfloat16)


    g = tl.reshape(sq, (rows_per_prog, head_dim // 8, 4, 2), can_reorder=False)
    p0, p1 = tl.split(g)
    p00, p01 = tl.split(
        tl.reshape(p0, (rows_per_prog, head_dim // 8, 2, 2), can_reorder=False)
    )
    p10, p11 = tl.split(
        tl.reshape(p1, (rows_per_prog, head_dim // 8, 2, 2), can_reorder=False)
    )
    s0, s4 = tl.split(
        tl.reshape(p00, (rows_per_prog, head_dim // 8, 1, 2), can_reorder=False)
    )
    s2, s6 = tl.split(
        tl.reshape(p01, (rows_per_prog, head_dim // 8, 1, 2), can_reorder=False)
    )
    s1, s5 = tl.split(
        tl.reshape(p10, (rows_per_prog, head_dim // 8, 1, 2), can_reorder=False)
    )
    s3, s7 = tl.split(
        tl.reshape(p11, (rows_per_prog, head_dim // 8, 1, 2), can_reorder=False)
    )
    acc = tl.reshape(s0, (rows_per_prog, head_dim // 8)).to(tl.float32)
    acc = acc + tl.reshape(s1, (rows_per_prog, head_dim // 8)).to(tl.float32)
    acc = acc + tl.reshape(s2, (rows_per_prog, head_dim // 8)).to(tl.float32)
    acc = acc + tl.reshape(s3, (rows_per_prog, head_dim // 8)).to(tl.float32)
    acc = acc + tl.reshape(s4, (rows_per_prog, head_dim // 8)).to(tl.float32)
    acc = acc + tl.reshape(s5, (rows_per_prog, head_dim // 8)).to(tl.float32)
    acc = acc + tl.reshape(s6, (rows_per_prog, head_dim // 8)).to(tl.float32)
    acc = acc + tl.reshape(s7, (rows_per_prog, head_dim // 8)).to(tl.float32)


    acc = tl.sum(tl.reshape(acc, (rows_per_prog, 2, 8), can_reorder=False), axis=1)
    acc = tl.sum(tl.reshape(acc, (rows_per_prog, 2, 4), can_reorder=False), axis=1)
    acc = tl.sum(tl.reshape(acc, (rows_per_prog, 2, 2), can_reorder=False), axis=1)
    ssum = tl.sum(acc, axis=1)
    ms = (ssum / head_dim).to(tl.bfloat16)

    rstd = tl.rsqrt((ms.to(tl.float32) + eps).to(tl.bfloat16).to(tl.float32)).to(
        tl.bfloat16
    )

    weight = tl.load(weight_ptr + offs)
    y = ((x.to(tl.float32) * rstd.to(tl.float32)[:, None]).to(tl.bfloat16) * weight).to(
        tl.bfloat16
    )
    tl.store(
        y_ptr + rows[:, None] * head_dim + offs[None, :], y, mask=row_mask[:, None]
    )


def _qk_head_token_stride(x: torch.Tensor, head_dim: int) -> int | None:

    if x.dim() != 4 or x.shape[-1] != head_dim:
        return None
    if x.stride(-1) != 1 or x.stride(-2) != head_dim:
        return None
    token_stride = x.stride(1)
    if x.shape[0] > 1 and x.stride(0) != x.shape[1] * token_stride:
        return None
    return token_stride


def can_use_qk_rmsnorm_native(
    x: torch.Tensor, weight: torch.Tensor, head_dim: int
) -> bool:
    return (
        x.is_cuda
        and weight.is_cuda
        and x.device == weight.device
        and x.dtype == weight.dtype == torch.bfloat16
        and x.numel() > 0
        and head_dim == 128
        and weight.shape == (head_dim,)
        and weight.is_contiguous()
        and _qk_head_token_stride(x, head_dim) is not None
    )


def zimage_qk_rmsnorm_native(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor | None:

    head_dim = x.shape[-1]
    if not can_use_qk_rmsnorm_native(x, weight, head_dim):
        return None
    token_stride = _qk_head_token_stride(x, head_dim)
    if token_stride is None:
        return None
    nheads = x.shape[2]
    n_rows = x.shape[0] * x.shape[1] * nheads
    arch = get_jit_cuda_arch()
    is_sm103 = arch.major == 10 and arch.minor == 3


    rows_per_prog = 16 if is_sm103 else 8
    num_warps = 4 if is_sm103 else 8
    y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(n_rows, rows_per_prog),)
    with torch.get_device_module().device(x.device):
        _qk_rmsnorm_native_kernel[grid](
            y,
            x,
            weight,
            token_stride,
            nheads,
            n_rows,
            head_dim,
            eps,
            rows_per_prog=rows_per_prog,
            num_warps=num_warps,
        )
    return y


__all__ = ["can_use_qk_rmsnorm_native", "zimage_qk_rmsnorm_native"]
