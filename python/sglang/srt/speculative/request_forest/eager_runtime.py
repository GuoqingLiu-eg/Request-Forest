from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import torch

from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.utils.nvtx_utils import profile_range

from .forest_verify_input import ForestVerifyInput
from .gpu_accept import ForestGreedyResult, verify_forest_greedy
from .gpu_runtime import (
    ForestKVCommit,
    ForestKVScratchTransaction,
    ensure_tp_publish_ready,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.tp_worker import TpModelWorker


logger = logging.getLogger(__name__)
_TARGET_TOP1_CHECK_ENABLED = (
    os.environ.get("SGLANG_REQUEST_FOREST_TARGET_TOP1_CHECK", "0") == "1"
)
_target_top1_check_rounds = 0
_target_top1_check_rows = 0


def _check_target_top1(top1, logits_output, verify_input) -> None:

    global _target_top1_check_rounds, _target_top1_check_rows
    logits = getattr(logits_output, "next_token_logits", None)
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 2
        or logits.shape[0] != verify_input.physical_rows
        or logits.shape[1] == 0
        or logits.device != top1.device
    ):
        raise RuntimeError(
            "Request Forest top1 CHECK requires full reference logits for every physical row"
        )
    reference = torch.argmax(logits, dim=-1)


    if not torch.equal(top1, reference):
        row = int(torch.nonzero(top1 != reference, as_tuple=False)[0, 0].item())
        raise RuntimeError(
            "Request Forest top1 CHECK mismatch: "
            f"physical_row={row}, logical_row={row < verify_input.logical_rows}, "
            f"distributed_token={int(top1[row].item())}, "
            f"full_logits_token={int(reference[row].item())}, "
            f"prior_matched_rounds={_target_top1_check_rounds}"
        )
    _target_top1_check_rounds += 1
    _target_top1_check_rows += verify_input.physical_rows
    if _target_top1_check_rounds == 1 or _target_top1_check_rounds % 32 == 0:
        logger.info(
            "Request Forest top1 CHECK matched_rounds=%d physical_rows_total=%d "
            "current_physical_rows=%d current_logical_rows=%d",
            _target_top1_check_rounds,
            _target_top1_check_rows,
            verify_input.physical_rows,
            verify_input.logical_rows,
        )


@dataclass(slots=True)
class ForestEagerVerifyResult:


    generation_result: GenerationBatchResult
    acceptance: ForestGreedyResult
    kv_commit: ForestKVCommit
    target_predict: torch.Tensor


def _validate_greedy_scope(batch: ScheduleBatch) -> None:
    sampling_info = getattr(batch, "sampling_info", None)
    if sampling_info is not None and not sampling_info.is_all_greedy:
        raise NotImplementedError(
            "Request Forest eager verification currently supports greedy "
            "sampling only"
        )
    if getattr(batch, "has_grammar", False):
        raise NotImplementedError(
            "Request Forest eager verification does not yet support grammar "
            "constraints"
        )
    if getattr(batch, "return_logprob", False):
        raise NotImplementedError(
            "Request Forest eager verification does not yet return accepted "
            "token logprobs"
        )
    hidden_mode = getattr(batch, "return_hidden_states_mode", None)
    if getattr(batch, "return_hidden_states", False) or (
        hidden_mode is not None
        and callable(getattr(hidden_mode, "need_capture", None))
        and hidden_mode.need_capture()
    ):
        raise NotImplementedError(
            "Request Forest eager verification does not yet compact returned "
            "hidden states along the accepted paths"
        )


def _target_top1(logits_output: Any, verify_input: ForestVerifyInput) -> torch.Tensor:
    top1 = getattr(logits_output, "next_token_top1", None)
    if top1 is not None:
        if (
            not isinstance(top1, torch.Tensor)
            or top1.ndim != 1
            or top1.shape[0] != verify_input.physical_rows
            or top1.dtype != torch.int64
            or top1.device != verify_input.draft_token.device
            or not top1.is_contiguous()
        ):
            raise ValueError(
                "Target top-1 needs one contiguous int64 ID per physical Forest row"
            )
        if _TARGET_TOP1_CHECK_ENABLED:
            _check_target_top1(top1, logits_output, verify_input)


        return top1[: verify_input.logical_rows].clone()
    logits = getattr(logits_output, "next_token_logits", None)
    if not isinstance(logits, torch.Tensor):
        raise RuntimeError("Target verification returned no next-token logits")
    if logits.dim() != 2 or logits.shape[0] != verify_input.physical_rows:
        raise ValueError(
            "Target next-token logits need one vocabulary row per physical "
            "Forest row"
        )
    if logits.device != verify_input.draft_token.device:
        raise ValueError("Target logits must share the Request Forest device")
    if logits.shape[1] < 1:
        raise ValueError("Target next-token logits have an empty vocabulary")
    return torch.argmax(logits[: verify_input.logical_rows], dim=-1)


def run_forest_target_verify_eager(
    batch: ScheduleBatch,
    verify_input: ForestVerifyInput,
    target_worker: TpModelWorker,
    *,
    eos_token_ids: torch.Tensor | None = None,
    request_ignore_eos: torch.Tensor | None = None,
    before_kv_publish: Callable[
        [GenerationBatchResult, ForestGreedyResult, ForestKVScratchTransaction],
        None,
    ]
    | None = None,
    before_target_kv_publish: Callable[
        [GenerationBatchResult, ForestGreedyResult, ForestKVScratchTransaction],
        None,
    ]
    | None = None,
    tp_group: Any | None = None,
) -> ForestEagerVerifyResult:


    if not isinstance(verify_input, ForestVerifyInput):
        raise TypeError("verify_input must be a ForestVerifyInput")
    _validate_greedy_scope(batch)

    previous = (
        batch.input_ids,
        batch.out_cache_loc,
        batch.spec_info,
        batch.forward_mode,
    )
    transaction: ForestKVScratchTransaction | None = None
    target_work_may_be_in_flight = False
    try:
        with profile_range("request_forest.prepare_verify"):
            forward_batch, transaction = (
                verify_input.prepare_for_verify(batch, target_worker)
            )

        target_work_may_be_in_flight = True
        with profile_range("request_forest.target_forward"):
            target_output = target_worker.forward_batch_generation(
                batch=None,
                forward_batch=forward_batch,
                is_verify=True,
            )
        logits_output = getattr(target_output, "logits_output", None)
        if logits_output is None:
            raise RuntimeError("Target verification returned no logits output")
        with profile_range("request_forest.target_top1"):
            target_predict = _target_top1(logits_output, verify_input)
        tp_world_size = 1 if tp_group is None else int(tp_group.world_size)
        if tp_world_size > 1:


            with profile_range("request_forest.tp_predict_broadcast"):
                tp_group.broadcast(target_predict, src=0)
        with profile_range("request_forest.accept"):
            acceptance = verify_forest_greedy(
                verify_input,
                target_predict,
                eos_token_ids=eos_token_ids,
                request_ignore_eos=request_ignore_eos,
            )
        transaction.stage_commit_lens(
            acceptance.accept_lens,
            accepted_width=acceptance.accept_index.shape[1],
        )


        planned_new_seq_lens = batch.seq_lens + acceptance.accept_lens
        proposed_draft_lens = (
            verify_input.request_offsets[1:]
            - verify_input.request_offsets[:-1]
            - 1
        )
        generation_result = GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=acceptance.output_token_ids.reshape(-1),
            can_run_cuda_graph=bool(
                getattr(target_output, "can_run_cuda_graph", False)
            ),
            speculative_num_draft_tokens=verify_input.max_tree_depth,
            accept_lens=acceptance.accept_lens,
            accepted_draft_lens=acceptance.accepted_draft_lens,
            proposed_draft_lens=proposed_draft_lens,
            new_seq_lens=planned_new_seq_lens,
            routed_experts_output=getattr(
                target_output, "routed_experts_output", None
            ),
            indexer_topk_output=getattr(
                target_output, "indexer_topk_output", None
            ),

            extra_keep_alive_refs=[forward_batch],
        )
        callback_error: Exception | None = None
        if before_kv_publish is not None:
            try:
                with profile_range("request_forest.prepare_draft_state"):
                    before_kv_publish(
                        generation_result,
                        acceptance,
                        transaction,
                    )
            except Exception as exc:


                callback_error = exc

        try:
            with profile_range("request_forest.tp_publish_ready"):
                ensure_tp_publish_ready(
                    callback_error,
                    tp_group=tp_group,
                    scratch_device=transaction.scratch_slots.device,
                )
        except Exception:


            target_work_may_be_in_flight = False
            raise
        if before_target_kv_publish is not None:
            with profile_range("request_forest.commit_target_state"):
                before_target_kv_publish(
                    generation_result,
                    acceptance,
                    transaction,
                )


        with profile_range("request_forest.commit"):
            kv_commit = transaction.commit(
                acceptance.accept_index,
                acceptance.accept_lens,
            )
        generation_result.request_forest_committed_slots = (
            kv_commit.committed_slots
        )
        return ForestEagerVerifyResult(
            generation_result=generation_result,
            acceptance=acceptance,
            kv_commit=kv_commit,
            target_predict=target_predict,
        )
    except Exception:
        if (
            transaction is not None
            and transaction.is_active
            and target_work_may_be_in_flight
            and transaction.scratch_slots.device.type == "cuda"
        ):
            torch.cuda.current_stream(
                transaction.scratch_slots.device
            ).synchronize()
        raise
    finally:
        if transaction is not None and transaction.is_active:
            transaction.rollback()
        (
            batch.input_ids,
            batch.out_cache_loc,
            batch.spec_info,
            batch.forward_mode,
        ) = previous


__all__ = [
    "ForestEagerVerifyResult",
    "run_forest_target_verify_eager",
]
