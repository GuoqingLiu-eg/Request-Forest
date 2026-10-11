from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_moe_topk_softmax_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "moe_topk_softmax",
        *args,
        cuda_files=["moe/moe_topk_softmax.cuh"],
        cuda_wrappers=[("topk_softmax", f"topk_softmax<{args}>")],
        extra_cuda_cflags=["--use_fast_math"],
    )


@register_custom_op(
    op_name="moe_topk_softmax_out",
    mutates_args=["topk_weights", "topk_ids", "workspace"],
)
def moe_topk_softmax_out(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    workspace: torch.Tensor,
    renormalize: bool,
    moe_softcapping: float,
    correction_bias: Optional[torch.Tensor],
) -> None:

    module = _jit_moe_topk_softmax_module(gating_output.dtype)
    module.topk_softmax(
        topk_weights,
        topk_ids,
        gating_output,
        workspace,
        renormalize,
        moe_softcapping,
        correction_bias,
    )


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    moe_softcapping: float = 0.0,
    correction_bias: Optional[torch.Tensor] = None,
) -> None:

    num_tokens = gating_output.shape[0]
    num_experts = gating_output.shape[1]


    is_pow2 = num_experts != 0 and (num_experts & (num_experts - 1)) == 0
    needs_workspace = not is_pow2 or num_experts > 512
    workspace_size = num_tokens * num_experts if needs_workspace else 1
    workspace = torch.empty(
        workspace_size, dtype=torch.float32, device=gating_output.device
    )

    moe_topk_softmax_out(
        topk_weights,
        topk_ids,
        gating_output,
        workspace,
        renormalize,
        moe_softcapping,
        correction_bias,
    )
