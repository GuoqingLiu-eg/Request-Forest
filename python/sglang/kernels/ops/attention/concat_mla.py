from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_concat_mla_k_module() -> Module:
    return load_jit(
        "concat_mla_k",
        cuda_files=["elementwise/concat_mla.cuh"],
        cuda_wrappers=[("concat_mla_k", "ConcatMlaKKernel::run")],
    )


@cache_once
def _jit_concat_mla_absorb_q_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "concat_mla_absorb_q",
        *args,
        cuda_files=["elementwise/concat_mla.cuh"],
        cuda_wrappers=[("concat_mla_absorb_q", f"ConcatMlaAbsorbQKernel<{args}>::run")],
    )


def concat_mla_k(k: torch.Tensor, k_nope: torch.Tensor, k_rope: torch.Tensor) -> None:

    module = _jit_concat_mla_k_module()
    module.concat_mla_k(k, k_nope, k_rope)


def concat_mla_absorb_q(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:

    out = torch.empty(
        (*a.shape[:-1], a.shape[-1] + b.shape[-1]),
        dtype=a.dtype,
        device=a.device,
    )
    module = _jit_concat_mla_absorb_q_module()
    module.concat_mla_absorb_q(a, b, out)
    return out
