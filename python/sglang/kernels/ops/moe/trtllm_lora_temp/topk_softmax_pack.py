from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_topk_softmax_pack_module() -> Module:
    return load_jit(
        "topk_softmax_pack",
        cuda_files=["trtllm_lora_temp/topk_softmax_pack.cuh"],
        cuda_wrappers=[("topk_softmax_pack", "topk_softmax_pack")],
    )


@register_custom_op(mutates_args=["topk_weights", "topk_indices", "packed"])
def _jit_topk_softmax_pack_op(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    packed: torch.Tensor,
    gating_output: torch.Tensor,
    num_token_non_padded: Optional[torch.Tensor],
    renormalize: bool,
) -> None:
    module = _jit_topk_softmax_pack_module()
    module.topk_softmax_pack(
        topk_weights,
        topk_indices,
        packed,
        gating_output,
        num_token_non_padded,
        renormalize,
    )


def topk_softmax_pack(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    packed: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
    num_token_non_padded: Optional[torch.Tensor] = None,
) -> None:

    assert gating_output.dim() == 2
    num_experts = gating_output.shape[-1]
    assert num_experts & (num_experts - 1) == 0 and num_experts <= 512, (
        "topk_softmax_pack supports power-of-2 num_experts in [1, 512] only; "
        "fall back to topk_softmax + fused_pack_topk"
    )
    if gating_output.shape[0] == 0:
        return
    _jit_topk_softmax_pack_op(
        topk_weights,
        topk_indices,
        packed,
        gating_output.contiguous(),
        num_token_non_padded,
        renormalize,
    )
