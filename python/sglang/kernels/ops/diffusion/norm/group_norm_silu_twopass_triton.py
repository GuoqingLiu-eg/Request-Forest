# SPDX-License-Identifier: Apache-2.0


import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
_MAX_CHANNELS = 2048


@triton.jit
def _gn_partial_rows_kernel(
    x_ptr,
    psum_ptr,
    psq_ptr,
    rows,
    rows_per_prog,
    C: tl.constexpr,
    BLOCK_R: tl.constexpr,
):

    chunk = tl.program_id(0).to(tl.int64)
    n = tl.program_id(1).to(tl.int64)
    nchunks = tl.num_programs(0)
    row0 = chunk * rows_per_prog
    cols = tl.arange(0, C)
    acc_s = tl.zeros((C,), tl.float32)
    acc_q = tl.zeros((C,), tl.float32)
    x_base = x_ptr + n * rows * C
    for r_off in range(0, rows_per_prog, BLOCK_R):
        rs = row0 + r_off + tl.arange(0, BLOCK_R)
        m = rs < rows
        x = tl.load(
            x_base + rs[:, None] * C + cols[None, :],
            mask=m[:, None],
            other=0.0,
        ).to(tl.float32)
        acc_s += tl.sum(x, 0)
        acc_q += tl.sum(x * x, 0)
    out_off = (n * nchunks + chunk) * C + cols
    tl.store(psum_ptr + out_off, acc_s)
    tl.store(psq_ptr + out_off, acc_q)


@triton.jit
def _gn_finalize_kernel(
    psum_ptr,
    psq_ptr,
    w_ptr,
    b_ptr,
    ss_ptr,
    nchunks,
    group_numel,
    eps,
    C,
    CPG: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    g = tl.program_id(0).to(tl.int64)
    n = tl.program_id(1).to(tl.int64)
    cols = g * CPG + tl.arange(0, CPG)
    s = tl.zeros((), tl.float32)
    q = tl.zeros((), tl.float32)
    for k0 in range(0, nchunks, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        m = ks < nchunks
        offs = (n * nchunks + ks)[:, None] * C + cols[None, :]
        s += tl.sum(tl.load(psum_ptr + offs, mask=m[:, None], other=0.0))
        q += tl.sum(tl.load(psq_ptr + offs, mask=m[:, None], other=0.0))
    mean = s / group_numel
    var = q / group_numel - mean * mean
    var = tl.maximum(var, 0.0)
    rstd = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + cols).to(tl.float32)
    b = tl.load(b_ptr + cols).to(tl.float32)
    scale = w * rstd
    shift = b - mean * scale
    tl.store(ss_ptr + n * 2 * C + cols, scale)
    tl.store(ss_ptr + (n * 2 + 1) * C + cols, shift)


@triton.jit
def _gn_apply_rows_kernel(
    x_ptr,
    ss_ptr,
    y_ptr,
    rows,
    C: tl.constexpr,
    BLOCK_R: tl.constexpr,
    SILU: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    n = tl.program_id(1).to(tl.int64)
    cols = tl.arange(0, C)
    scale = tl.load(ss_ptr + n * 2 * C + cols)
    shift = tl.load(ss_ptr + (n * 2 + 1) * C + cols)
    rs = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    m = rs < rows
    offs = n * rows * C + rs[:, None] * C + cols[None, :]
    x = tl.load(x_ptr + offs, mask=m[:, None], other=0.0).to(tl.float32)
    y = x * scale[None, :] + shift[None, :]
    if SILU:
        y = y * tl.sigmoid(y)
    tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=m[:, None])


def _gn_silu_rows(x3, weight, bias, num_groups, eps, apply_silu):

    n_batch, rows, c = x3.shape
    cpg = c // num_groups
    block_r = max(1, 8192 // c)
    rows_per_prog = block_r * 32
    nchunks = triton.cdiv(rows, rows_per_prog)
    psum = torch.empty((n_batch, nchunks, c), device=x3.device, dtype=torch.float32)
    psq = torch.empty_like(psum)
    with torch.get_device_module().device(x3.device):
        _gn_partial_rows_kernel[(nchunks, n_batch)](
            x3, psum, psq, rows, rows_per_prog, C=c, BLOCK_R=block_r, num_warps=4
        )
        ss = torch.empty((n_batch, 2, c), device=x3.device, dtype=torch.float32)
        block_k = max(1, min(4096 // cpg, triton.next_power_of_2(nchunks)))
        _gn_finalize_kernel[(num_groups, n_batch)](
            psum,
            psq,
            weight,
            bias,
            ss,
            nchunks,
            rows * cpg,
            eps,
            c,
            CPG=cpg,
            BLOCK_K=block_k,
            num_warps=4,
        )
        y3 = torch.empty_like(x3)
        _gn_apply_rows_kernel[(triton.cdiv(rows, block_r), n_batch)](
            x3, ss, y3, rows, C=c, BLOCK_R=block_r, SILU=apply_silu, num_warps=4
        )
    return y3


def _twopass_supported(x, weight, bias, num_groups) -> bool:

    if not (x.is_cuda and x.numel() > 0 and not torch.is_grad_enabled()):
        return False
    if x.requires_grad or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if weight is None or bias is None:
        return False
    c = x.shape[1] if x.dim() == 4 else x.shape[-1]
    if weight.shape != (c,) or bias.shape != (c,):
        return False
    if not (
        weight.device == bias.device == x.device
        and weight.dtype == bias.dtype == x.dtype
        and weight.is_contiguous()
        and bias.is_contiguous()
    ):
        return False
    if num_groups < 1 or c % num_groups != 0:
        return False


    return triton.next_power_of_2(c) == c and c <= _MAX_CHANNELS


def can_use_group_norm_silu_4d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
) -> bool:

    if x.dim() != 4 or not _twopass_supported(x, weight, bias, num_groups):
        return False
    _, c, h, w = x.shape


    return (
        c > 1
        and (h > 1 or w > 1)
        and x.is_contiguous(memory_format=torch.channels_last)
    )


def can_use_group_norm_silu_rows(
    x3: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
) -> bool:

    return (
        x3.dim() == 3
        and x3.is_contiguous()
        and _twopass_supported(x3, weight, bias, num_groups)
    )


def group_norm_silu_4d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float,
    apply_silu: bool = True,
) -> torch.Tensor:

    if not can_use_group_norm_silu_4d(x, weight, bias, num_groups):
        raise ValueError("unsupported input for group_norm_silu_4d")
    n_batch, c, h, w = x.shape
    x3 = x.permute(0, 2, 3, 1).reshape(n_batch, h * w, c)
    y3 = _gn_silu_rows(x3, weight, bias, num_groups, eps, apply_silu)
    return y3.reshape(n_batch, h, w, c).permute(0, 3, 1, 2)


def group_norm_silu_rows(
    x3: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float,
    apply_silu: bool = True,
) -> torch.Tensor:

    if not can_use_group_norm_silu_rows(x3, weight, bias, num_groups):
        raise ValueError("unsupported input for group_norm_silu_rows")
    return _gn_silu_rows(x3, weight, bias, num_groups, eps, apply_silu)


__all__ = [
    "can_use_group_norm_silu_4d",
    "can_use_group_norm_silu_rows",
    "group_norm_silu_4d",
    "group_norm_silu_rows",
]
