from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_MAX_K = 17408
_MAX_N = 65536


def _config(n: int) -> tuple[int, int, int]:

    if n >= 8192:
        return (2, 2, 8)
    return (1, 2, 8)


@cache_once
def _jit_hopper_bf16_gemv_module(n: int, k: int) -> Module:
    rows, unroll, warps = _config(n)
    args = make_cpp_args(n, k, rows, unroll, warps)
    return load_jit(
        "hopper_bf16_gemv",
        *args,
        cuda_files=["gemm/hopper_bf16_gemv.cuh"],
        cuda_wrappers=[("run", f"sglang::HopperBf16GemvKernel<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def use_hopper_bf16_gemv(m: int, n: int, k: int) -> bool:
    if not (
        m == 1
        and k % 512 == 0
        and 512 <= k <= _MAX_K
        and n % 8 == 0
        and 64 <= n <= _MAX_N
    ):
        return False


    return n < 12288 or n >= 32768


def hopper_bf16_gemv(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:

    out = torch.empty((1, w.shape[0]), dtype=x.dtype, device=x.device)
    module = _jit_hopper_bf16_gemv_module(w.shape[0], w.shape[1])
    module.run(x, w, out)
    return out
