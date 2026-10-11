from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def dsv3_router_gemm_module(
    num_experts: int,
    hidden_dim: int,
    use_pdl: bool,
    out_float: bool,
) -> Module:
    args = make_cpp_args(num_experts, hidden_dim, use_pdl, out_float)
    return load_jit(
        "dsv3_router_gemm",
        *args,
        cuda_files=["gemm/dsv3_router_gemm.cuh"],
        cuda_wrappers=[
            ("dsv3_router_gemm", f"DSV3RouterGemmKernel<{args}>::run"),
        ],
    )


@register_custom_op(
    op_name="dsv3_router_gemm",
    mutates_args=["output"],
)
def _dsv3_router_gemm_custom_op(
    hidden_states: torch.Tensor,
    router_weights: torch.Tensor,
    output: torch.Tensor,
) -> None:
    num_experts = router_weights.shape[0]
    hidden_dim = hidden_states.shape[1]
    out_float = output.dtype == torch.float32
    module = dsv3_router_gemm_module(
        num_experts, hidden_dim, is_arch_support_pdl(), out_float
    )
    module.dsv3_router_gemm(hidden_states, router_weights, output)
    return None


@debug_kernel_api
def dsv3_router_gemm(
    hidden_states: torch.Tensor,
    router_weights: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    if output is None:
        output = torch.empty(
            hidden_states.shape[0],
            router_weights.shape[0],
            device=hidden_states.device,
            dtype=out_dtype,
        )
    _dsv3_router_gemm_custom_op(hidden_states, router_weights, output)
    return output
