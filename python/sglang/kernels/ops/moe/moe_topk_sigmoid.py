from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_moe_topk_sigmoid_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "moe_topk_sigmoid",
        *args,
        cuda_files=["moe/moe_topk_sigmoid.cuh"],
        cuda_wrappers=[("topk_sigmoid", f"topk_sigmoid<{args}>")],
        extra_cuda_cflags=["--use_fast_math"],
    )


@register_custom_op(
    op_name="moe_topk_sigmoid_out",
    mutates_args=["topk_weights", "topk_ids"],
)
def moe_topk_sigmoid_out(
    gating_output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    workspace: torch.Tensor,
    renormalize: bool,
    correction_bias: Optional[torch.Tensor],
    routed_scaling_factor: float,
    num_fused_shared_experts: int,
) -> None:

    module = _jit_moe_topk_sigmoid_module(gating_output.dtype)
    module.topk_sigmoid(
        gating_output,
        topk_weights,
        topk_ids,
        workspace,
        renormalize,
        correction_bias,
        routed_scaling_factor,
        num_fused_shared_experts,
    )


def topk_sigmoid(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    correction_bias: Optional[torch.Tensor] = None,
    routed_scaling_factor: float = 1.0,
    num_fused_shared_experts: int = 0,
) -> None:

    num_tokens = gating_output.shape[0]
    num_experts = gating_output.shape[1]

    is_pow2 = num_experts != 0 and (num_experts & (num_experts - 1)) == 0
    needs_workspace = not is_pow2 or num_experts > 256
    workspace_size = num_tokens * num_experts if needs_workspace else 1
    workspace = torch.empty(
        workspace_size, dtype=torch.float32, device=gating_output.device
    )

    moe_topk_sigmoid_out(
        gating_output,
        topk_weights,
        topk_ids,
        workspace,
        renormalize,
        correction_bias,
        routed_scaling_factor,
        num_fused_shared_experts,
    )
