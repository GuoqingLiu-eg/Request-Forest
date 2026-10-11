from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.ops.kimi_k3.all_reduce import _COMM_MAP
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


K = 3584
N = 7168


MAX_TOKENS = 12


@cache_once
def _jit_module() -> Module:
    args = make_cpp_args(K, N, MAX_TOKENS, is_arch_support_pdl())
    cls = f"GEMMAGKernel<{args}>"
    return load_jit(
        "kimi_k3_gemm_ag",
        *args,
        cuda_files=["kimi_k3/comm/gemm_ag.cuh"],
        cuda_wrappers=[("run", f"{cls}::run")],
        extra_cuda_cflags=["-O3"],
    )


@register_custom_op(mutates_args=["out"])
def _gemm_ag_op(
    world_size: int,
    x: torch.Tensor,
    weight: torch.Tensor,
    b: torch.Tensor,
    c: Optional[torch.Tensor],
    out: torch.Tensor,
) -> None:
    _jit_module().run(_COMM_MAP[world_size], x, weight, b, c, out)


def gemm_ag_up_proj(
    world_size: int,
    x: torch.Tensor,
    weight: torch.Tensor,
    b: torch.Tensor,
    c: Optional[torch.Tensor],
    out: torch.Tensor,
) -> torch.Tensor:

    _gemm_ag_op(world_size, x, weight, b, c, out)
    return out
