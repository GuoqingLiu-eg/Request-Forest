from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch

from .forest_verify_input import ForestVerifyInput


class ForestTerminationCode(IntEnum):


    TARGET_MISS = 0
    LEAF_EXHAUSTED = 1
    BUDGET_BOUNDARY = 2
    NO_CANDIDATES = 3
    EOS = 4


@dataclass(slots=True)
class ForestGreedyResult:


    predicts: torch.Tensor
    output_token_ids: torch.Tensor
    accept_index: torch.Tensor
    accepted_draft_lens: torch.Tensor
    accept_lens: torch.Tensor
    first_miss_positions: torch.Tensor
    termination_reasons: torch.Tensor


_NATIVE_VERIFY_GREEDY = None


def _native_verify_greedy():
    global _NATIVE_VERIFY_GREEDY
    if _NATIVE_VERIFY_GREEDY is not None:
        return _NATIVE_VERIFY_GREEDY
    try:
        operation = torch.ops.sgl_kernel.request_forest_verify_greedy
    except AttributeError:
        return None
    _NATIVE_VERIFY_GREEDY = operation
    return operation


def verify_forest_greedy(
    verify_input: ForestVerifyInput,
    target_predict: torch.Tensor,
    *,
    eos_token_ids: torch.Tensor | None = None,
    request_ignore_eos: torch.Tensor | None = None,
) -> ForestGreedyResult:


    if not isinstance(verify_input, ForestVerifyInput):
        raise TypeError("verify_input must be a ForestVerifyInput")
    if verify_input.draft_token.device.type != "cuda":
        raise ValueError("packed greedy verification requires a CUDA input")
    if not isinstance(target_predict, torch.Tensor):
        raise TypeError("target_predict must be a torch.Tensor")
    if target_predict.dim() != 1 or target_predict.numel() != verify_input.logical_rows:
        raise ValueError("target_predict needs one entry per logical row")
    if target_predict.dtype != torch.int64:
        raise TypeError("target_predict must have dtype torch.int64")
    if target_predict.device != verify_input.draft_token.device:
        raise ValueError("target_predict must share ForestVerifyInput's device")
    if not target_predict.is_contiguous():
        raise ValueError("target_predict must be contiguous")

    if eos_token_ids is None:
        eos_token_ids = torch.empty(
            0,
            dtype=torch.int64,
            device=target_predict.device,
        )
    elif not isinstance(eos_token_ids, torch.Tensor):
        raise TypeError("eos_token_ids must be a torch.Tensor or None")
    elif (
        eos_token_ids.dim() != 1
        or eos_token_ids.dtype != torch.int64
        or eos_token_ids.device != target_predict.device
        or not eos_token_ids.is_contiguous()
    ):
        raise ValueError(
            "eos_token_ids must be a contiguous CUDA int64 vector on the input device"
        )

    request_count = verify_input.request_count
    if request_ignore_eos is None:
        request_ignore_eos = torch.zeros(
            request_count,
            dtype=torch.bool,
            device=target_predict.device,
        )
    elif not isinstance(request_ignore_eos, torch.Tensor):
        raise TypeError("request_ignore_eos must be a torch.Tensor or None")
    elif (
        request_ignore_eos.dim() != 1
        or request_ignore_eos.dtype != torch.bool
        or request_ignore_eos.device != target_predict.device
        or not request_ignore_eos.is_contiguous()
    ):
        raise ValueError(
            "request_ignore_eos must be a contiguous CUDA bool vector on the "
            "input device"
        )
    if request_ignore_eos.numel() != request_count:
        raise ValueError(
            "request_ignore_eos needs one entry per Request Forest request"
        )

    operation = _native_verify_greedy()
    if operation is None:
        raise RuntimeError(
            "request_forest_verify_greedy is not loaded; build the SGLang CUDA "
            "extension or load scripts/request_forest/load_native_cuda_ops.py"
        )

    device = target_predict.device
    output_shape = (request_count, verify_input.max_tree_depth)
    predicts = torch.empty(
        verify_input.logical_rows,
        dtype=torch.int32,
        device=device,
    )
    output_token_ids = torch.empty(output_shape, dtype=torch.int32, device=device)
    accept_index = torch.empty(output_shape, dtype=torch.int32, device=device)
    accepted_draft_lens = torch.empty(
        request_count,
        dtype=torch.int32,
        device=device,
    )
    accept_lens = torch.empty_like(accepted_draft_lens)
    first_miss_positions = torch.empty(
        request_count,
        dtype=torch.int64,
        device=device,
    )
    termination_reasons = torch.empty(
        request_count,
        dtype=torch.int8,
        device=device,
    )

    operation(
        predicts,
        output_token_ids,
        accept_index,
        accepted_draft_lens,
        accept_lens,
        first_miss_positions,
        termination_reasons,
        verify_input.draft_token[: verify_input.logical_rows],
        verify_input.first_child[: verify_input.logical_rows],
        verify_input.next_sibling[: verify_input.logical_rows],
        verify_input.positions[: verify_input.logical_rows],
        verify_input.has_unverified_children[: verify_input.logical_rows],
        verify_input.request_offsets,
        verify_input.anchor_rows,
        target_predict,
        request_ignore_eos,
        eos_token_ids,
    )
    return ForestGreedyResult(
        predicts=predicts,
        output_token_ids=output_token_ids,
        accept_index=accept_index,
        accepted_draft_lens=accepted_draft_lens,
        accept_lens=accept_lens,
        first_miss_positions=first_miss_positions,
        termination_reasons=termination_reasons,
    )


__all__ = [
    "ForestGreedyResult",
    "ForestTerminationCode",
    "verify_forest_greedy",
]
