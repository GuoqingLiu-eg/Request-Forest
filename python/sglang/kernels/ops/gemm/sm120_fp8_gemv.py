from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_MAX_K = 32768
_MAX_N = 32768


def _config(n: int) -> tuple[int, int, int]:

    if n >= 8192:
        return (2, 1, 8)
    return (1, 1, 8)


@cache_once
def _jit_sm120_fp8_gemv_module(n: int, k: int) -> Module:
    rows, unroll, warps = _config(n)
    args = make_cpp_args(n, k, rows, unroll, warps)
    return load_jit(
        "sm120_fp8_gemv",
        *args,
        cuda_files=["gemm/sm120_fp8_gemv.cuh"],
        cuda_wrappers=[("run", f"sglang::Sm120Fp8GemvKernel<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def use_sm120_fp8_gemv(m: int, n: int, k: int) -> bool:
    return (
        m == 1
        and k % 512 == 0
        and 512 <= k <= _MAX_K
        and n % 16 == 0
        and 256 <= n <= _MAX_N
    )


def sm120_fp8_gemv(
    x_fp8: torch.Tensor, w_fp8: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:

    out = torch.empty((1, w_fp8.shape[0]), dtype=torch.bfloat16, device=x_fp8.device)
    module = _jit_sm120_fp8_gemv_module(w_fp8.shape[0], w_fp8.shape[1])
    module.run(x_fp8, w_fp8, alpha, out)
    return out
