from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    get_jit_cuda_arch,
    is_arch_support_pdl,
    is_hip_runtime,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


def _make_name(*args):
    return "kimi_k3_" + "_".join(str(a) for a in args)


def _fast_math_flags() -> list[str]:


    if is_hip_runtime():
        return []
    if get_jit_cuda_arch().major >= 10:
        return []
    return ["--use_fast_math"]


@cache_once
def _jit_situ_and_mul_module(in_dtype: torch.dtype, out_dtype: torch.dtype) -> Module:

    args = make_cpp_args(in_dtype, out_dtype, is_arch_support_pdl())
    return load_jit(
        _make_name("situ_and_mul"),
        *args,
        cuda_files=["kimi_k3/situ_and_mul.cuh"],
        cuda_wrappers=[("run", f"SituAndMulKernel<{args}>::run")],
        extra_cuda_cflags=_fast_math_flags(),
    )


def situ_and_mul(
    input: torch.Tensor,
    out: Optional[torch.Tensor],
    beta: float,
    linear_beta: Optional[float],
) -> torch.Tensor:

    hidden_size = input.shape[-1] // 2
    if out is None:
        out_dtype = torch.bfloat16 if input.dtype == torch.float32 else input.dtype
        out = input.new_empty(*input.shape[:-1], hidden_size, dtype=out_dtype)


    if input.dim() == 2 and input.stride(1) == 1:
        input_2d = input
    else:
        input_2d = input.contiguous().view(-1, hidden_size * 2)
    out_2d = out.view(-1, hidden_size)

    has_linear_beta = linear_beta is not None
    module = _jit_situ_and_mul_module(input_2d.dtype, out_2d.dtype)
    module.run(
        input_2d,
        out_2d,
        float(beta),
        float(linear_beta) if has_linear_beta else 0.0,
        has_linear_beta,
    )
    return out
