from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_mul_add_delta_kernel(
    x_ptr,
    d_ptr,
    out_ptr,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    cols = block * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N

    x_row = x_ptr + row * 2 * N
    d_row = d_ptr + row * 2 * N
    gate = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_row + N + cols, mask=mask, other=0.0).to(tl.float32)
    gate += tl.load(d_row + cols, mask=mask, other=0.0).to(tl.float32)
    up += tl.load(d_row + N + cols, mask=mask, other=0.0).to(tl.float32)

    silu = gate * tl.sigmoid(gate)
    out = silu * up
    tl.store(out_ptr + row * N + cols, out.to(out_ptr.dtype.element_ty), mask=mask)


def silu_and_mul_add_delta(
    gemm1_out: torch.Tensor,
    gate_up_delta: torch.Tensor,
    out: torch.Tensor,
) -> None:

    T, two_n = gemm1_out.shape
    N = two_n // 2
    assert gate_up_delta.shape == (T, two_n) and out.shape == (T, N)
    assert gemm1_out.is_contiguous() and gate_up_delta.is_contiguous()
    assert out.is_contiguous()
    if T == 0:
        return
    BLOCK = 512
    _silu_mul_add_delta_kernel[(T, triton.cdiv(N, BLOCK))](
        gemm1_out,
        gate_up_delta,
        out,
        N,
        BLOCK=BLOCK,
    )
