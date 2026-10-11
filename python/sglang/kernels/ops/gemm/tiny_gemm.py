from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_MAX_M_DEFAULT: int = 16


@cache_once
def _jit_tiny_gemm_module(
    n: int, k: int, max_m: int, split_n: int, out_dtype: torch.dtype
) -> Module:
    args = make_cpp_args(n, k, max_m, split_n, out_dtype, is_arch_support_pdl())
    return load_jit(
        "tiny_gemm",
        *args,
        cuda_files=["gemm/tiny_gemm.cuh"],
        cuda_wrappers=[("run", f"TinyNGemmKernel<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def _jit_tiny_k_gemm_module(
    n: int, k: int, max_m: int, n_unroll: int, out_dtype: torch.dtype
) -> Module:
    args = make_cpp_args(n, k, max_m, n_unroll, out_dtype, is_arch_support_pdl())
    return load_jit(
        "tiny_k_gemm",
        *args,
        cuda_files=["gemm/tiny_gemm.cuh"],
        cuda_wrappers=[("run", f"TinyKGemmKernel<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def _vec_elems() -> int:

    from sglang.kernels.jit.utils import get_jit_cuda_arch

    cuda = tuple(int(v) for v in (torch.version.cuda or "0.0").split(".")[:2])
    return 16 if get_jit_cuda_arch().major >= 10 and cuda >= (12, 9) else 8


def _default_split_n(n: int, k: int, max_m: int, device: torch.device) -> int:

    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    split_cap = (k // _vec_elems()) // max_m
    divisors = [d for d in range(1, min(n, split_cap) + 1) if n % d == 0]
    if not divisors:
        raise RuntimeError(
            f"tiny_gemm: no valid split_n for N={n}, K={k}, max_m={max_m};"
            " lower max_m"
        )
    for split in divisors:
        if n // split <= sm_count:
            return split
    return divisors[-1]


def tiny_n_gemm_bf16(
    x: torch.Tensor,
    w: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    out_dtype: Optional[torch.dtype] = None,
    split_n: Optional[int] = None,
    max_m: int = _MAX_M_DEFAULT,
) -> torch.Tensor:
    n = w.shape[0]
    k = x.shape[1]
    if out is None:
        out_dtype = out_dtype or torch.bfloat16
        out = torch.empty((x.shape[0], n), dtype=out_dtype, device=x.device)
    else:
        assert out_dtype is None or out_dtype == out.dtype
    if split_n is None:
        split_n = _default_split_n(n, k, max_m, x.device)
    module = _jit_tiny_gemm_module(n, k, max_m, split_n, out.dtype)
    module.run(x, w, out)
    return out


def _default_k_split_n(n: int, k: int) -> int:

    lanes = k // 8
    candidates = [
        d
        for d in range(1, n + 1)
        if n % d == 0 and d * lanes % 32 == 0 and d * lanes <= 1024
    ]
    if not candidates:
        raise RuntimeError(f"tiny_k_gemm: no valid split_n for N={n}, K={k}")
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    for d in candidates:
        if n // d <= sm_count:
            return d
    return candidates[-1]


def tiny_k_gemm_bf16(
    x: torch.Tensor,
    w: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    out_dtype: Optional[torch.dtype] = None,
    split_n: Optional[int] = None,
    max_m: int = _MAX_M_DEFAULT,
) -> torch.Tensor:

    n = w.shape[0]
    k = x.shape[1]
    if out is None:
        out_dtype = out_dtype or torch.bfloat16
        out = torch.empty((x.shape[0], n), dtype=out_dtype, device=x.device)
    else:
        assert out_dtype is None or out_dtype == out.dtype
    if split_n is None:
        split_n = _default_k_split_n(n, k)
    module = _jit_tiny_k_gemm_module(n, k, max_m, split_n, out.dtype)
    module.run(x, w, out)
    return out
