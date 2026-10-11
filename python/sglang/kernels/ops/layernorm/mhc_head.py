from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _hc_head_kernel(
    x_ptr,
    fn_ptr,
    scale_ptr,
    base_ptr,
    y_ptr,
    hidden_size: tl.constexpr,
    HC_MULT: tl.constexpr,
    K_TOTAL: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    norm_eps: tl.constexpr,
    hc_eps: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)


    sumsq = tl.zeros((), dtype=tl.float32)
    mix = tl.zeros((HC_MULT,), dtype=tl.float32)

    x_row = x_ptr + pid * K_TOTAL
    m_idx = tl.arange(0, HC_MULT)

    for k_off in tl.range(0, K_TOTAL, BLOCK_K):
        k_offs = k_off + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K_TOTAL
        x_tile = tl.load(x_row + k_offs, mask=k_mask, other=0.0).to(tl.float32)

        sumsq += tl.sum(x_tile * x_tile, axis=0)

        fn_offs = m_idx[:, None] * K_TOTAL + k_offs[None, :]
        fn_mask = (m_idx[:, None] < HC_MULT) & k_mask[None, :]
        fn_tile = tl.load(fn_ptr + fn_offs, mask=fn_mask, other=0.0)
        mix += tl.sum(fn_tile * x_tile[None, :], axis=1)

    rsqrt = tl.rsqrt(sumsq / K_TOTAL + norm_eps)
    scale_v = tl.load(scale_ptr).to(tl.float32)
    base_v = tl.load(base_ptr + m_idx).to(tl.float32)


    pre = tl.sigmoid(mix * rsqrt * scale_v + base_v) + hc_eps


    y_row = y_ptr + pid * hidden_size

    for d_off in tl.range(0, hidden_size, BLOCK_D):
        d_offs = d_off + tl.arange(0, BLOCK_D)
        d_mask = d_offs < hidden_size

        x_offs = m_idx[:, None] * hidden_size + d_offs[None, :]
        x_mask = (m_idx[:, None] < HC_MULT) & d_mask[None, :]
        x_block = tl.load(x_row + x_offs, mask=x_mask, other=0.0).to(tl.float32)

        y_block = tl.sum(pre[:, None] * x_block, axis=0)

        tl.store(y_row + d_offs, y_block.to(y_ptr.dtype.element_ty), mask=d_mask)


def fused_hc_head(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:

    assert x.is_contiguous(), "x must be contiguous"
    assert hc_fn.is_contiguous(), "hc_fn must be contiguous"
    assert hc_scale.dtype == torch.float32 and hc_base.dtype == torch.float32
    assert hc_fn.dtype == torch.float32
    assert x.dim() == 3, f"x must be 3D (T, hc_mult, hidden_size), got {x.shape}"

    T, hc_mult, hidden_size = x.shape
    assert hc_fn.shape == (hc_mult, hc_mult * hidden_size), (
        f"hc_fn shape {hc_fn.shape} does not match (hc_mult={hc_mult}, "
        f"hc_mult*hidden_size={hc_mult * hidden_size})"
    )
    assert hc_base.shape == (hc_mult,)
    assert hc_scale.numel() == 1

    y = torch.empty((T, hidden_size), dtype=x.dtype, device=x.device)

    if T == 0:
        return y

    BLOCK_K = 512
    BLOCK_D = 512

    hc_mult_pow2 = max(1, triton.next_power_of_2(hc_mult))

    grid = (T,)
    _hc_head_kernel[grid](
        x,
        hc_fn,
        hc_scale,
        hc_base,
        y,
        hidden_size=hidden_size,
        HC_MULT=hc_mult_pow2,
        K_TOTAL=hc_mult * hidden_size,
        BLOCK_K=BLOCK_K,
        BLOCK_D=BLOCK_D,
        norm_eps=norm_eps,
        hc_eps=hc_eps,
        num_warps=4,
    )
    return y
