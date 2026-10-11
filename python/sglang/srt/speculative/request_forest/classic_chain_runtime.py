from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterator, Sequence

import torch

from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

from .eager_runtime import (
    ForestEagerVerifyResult,
    _target_top1,
    _validate_greedy_scope,
)
from .forest_verify_input import ForestVerifyInput
from .gpu_accept import ForestGreedyResult, verify_forest_greedy
from .gpu_runtime import ForestKVScratchTransaction, ensure_tp_publish_ready

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.tp_worker import TpModelWorker


@contextmanager
def _temporary_chain_mapping(
    batch: ScheduleBatch,
    *,
    request_slots_int64: torch.Tensor,
    depths_int64: torch.Tensor,
    logical_rows: int,
    scratch_slots: torch.Tensor,
) -> Iterator[None]:


    request_slots = request_slots_int64[:logical_rows]
    depths = depths_int64[:logical_rows]
    scratch_slots = scratch_slots[:logical_rows]
    request_rows = batch.req_pool_indices.index_select(0, request_slots)
    token_columns = batch.seq_lens.index_select(0, request_slots) + (
        depths
    )
    req_to_token = batch.req_to_token_pool.req_to_token
    old_locations = req_to_token[request_rows, token_columns]
    try:
        req_to_token[request_rows, token_columns] = scratch_slots.to(
            req_to_token.dtype
        )
        yield
    finally:
        req_to_token[request_rows, token_columns] = old_locations


def _validate_verify_lens(
    verify_input: ForestVerifyInput,
    verify_lens_cpu: Sequence[int],
) -> tuple[int, ...]:
    verify_lens = tuple(int(length) for length in verify_lens_cpu)
    if len(verify_lens) != verify_input.request_count:
        raise ValueError("verify_lens_cpu must contain one value per request")
    if not verify_lens or min(verify_lens) < 1:
        raise ValueError("every classic chain must verify its anchor")
    if sum(verify_lens) != verify_input.logical_rows:
        raise ValueError("verify_lens_cpu must cover every packed Forest row")
    if max(verify_lens) != verify_input.max_request_rows:
        raise ValueError("verify_lens_cpu disagrees with the compiled Forest width")
    return verify_lens


@dataclass(slots=True)
class ClassicChainVerifyTemplate:


    static_input: ForestVerifyInput
    verify_lens_cpu: tuple[int, ...]
    ragged_layout: RaggedVerifyLayout
    request_slots_int64: torch.Tensor
    depths_int64: torch.Tensor
    candidate_depths_int64: torch.Tensor
    anchor_rows_int64: torch.Tensor

    @classmethod
    def from_verify_input(
        cls,
        verify_input: ForestVerifyInput,
        *,
        verify_lens_cpu: Sequence[int],
    ) -> ClassicChainVerifyTemplate:
        verify_lens = _validate_verify_lens(verify_input, verify_lens_cpu)
        verify_lens_device = torch.tensor(
            verify_lens,
            dtype=torch.int32,
            device=verify_input.draft_token.device,
        )
        request_slots = [
            request_slot
            for request_slot, length in enumerate(verify_lens)
            for _ in range(length)
        ]
        depths = [depth for length in verify_lens for depth in range(length)]
        padding_rows = verify_input.physical_rows - verify_input.logical_rows
        request_slots.extend([0] * padding_rows)
        depths.extend([0] * padding_rows)
        request_slots_int64 = torch.tensor(
            request_slots,
            dtype=torch.int64,
            device=verify_input.draft_token.device,
        )
        depths_int64 = torch.tensor(
            depths,
            dtype=torch.int64,
            device=verify_input.draft_token.device,
        )
        return cls(
            static_input=verify_input,
            verify_lens_cpu=verify_lens,
            ragged_layout=RaggedVerifyLayout.from_verify_lens_device(
                verify_lens=verify_lens_device,
                graph_num_tokens=verify_input.logical_rows,
            ),
            request_slots_int64=request_slots_int64,
            depths_int64=depths_int64,
            candidate_depths_int64=(depths_int64 - 1).clamp_min(0),
            anchor_rows_int64=verify_input.anchor_rows.to(torch.int64),
        )

    def materialize(
        self,
        *,
        candidate_token_ids: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
    ) -> ForestVerifyInput:


        request_count = self.static_input.request_count
        device = self.static_input.draft_token.device
        for name, value in (
            ("candidate_token_ids", candidate_token_ids),
            ("anchor_token_ids", anchor_token_ids),
            ("anchor_positions", anchor_positions),
        ):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value.device != device:
                raise ValueError(f"{name} must share the template device")
            if value.dtype != torch.int64:
                raise TypeError(f"{name} must have dtype torch.int64")
        if candidate_token_ids.dim() != 2 or candidate_token_ids.shape[0] != request_count:
            raise ValueError("candidate_token_ids must have shape [request_count, E]")
        if candidate_token_ids.shape[1] < self.static_input.max_request_rows - 1:
            raise ValueError("candidate_token_ids does not cover the cached chain depth")
        if anchor_token_ids.shape != (request_count,):
            raise ValueError("anchor_token_ids must have shape [request_count]")
        if anchor_positions.shape != (request_count,):
            raise ValueError("anchor_positions must have shape [request_count]")

        if candidate_token_ids.shape[1] == 0:
            draft_token = anchor_token_ids.index_select(
                0, self.request_slots_int64
            )
        else:
            draft_token = candidate_token_ids[
                self.request_slots_int64,
                self.candidate_depths_int64,
            ]
            draft_token.index_copy_(
                0,
                self.anchor_rows_int64,
                anchor_token_ids,
            )
        positions = anchor_positions.index_select(
            0, self.request_slots_int64
        ) + self.depths_int64
        static = self.static_input
        return ForestVerifyInput(
            draft_token=draft_token.contiguous(),
            positions=positions.contiguous(),
            first_child=static.first_child,
            next_sibling=static.next_sibling,
            has_unverified_children=static.has_unverified_children,
            request_offsets=static.request_offsets,
            anchor_rows=static.anchor_rows,
            ancestor_row_ptr=static.ancestor_row_ptr,
            ancestor_columns=static.ancestor_columns,
            logical_rows=static.logical_rows,
            max_request_rows=static.max_request_rows,
            max_tree_depth=static.max_tree_depth,
            tree_attention_workspace=static.tree_attention_workspace,
        )


def run_classic_chain_target_verify_eager(
    batch: ScheduleBatch,
    verify_input: ForestVerifyInput,
    target_worker: TpModelWorker,
    *,
    template: ClassicChainVerifyTemplate,
    eos_token_ids: torch.Tensor | None = None,
    request_ignore_eos: torch.Tensor | None = None,
    before_kv_publish: Callable[
        [GenerationBatchResult, ForestGreedyResult, ForestKVScratchTransaction],
        None,
    ]
    | None = None,
    tp_group: Any | None = None,
) -> ForestEagerVerifyResult:


    if not isinstance(verify_input, ForestVerifyInput):
        raise TypeError("verify_input must be a ForestVerifyInput")
    if not isinstance(template, ClassicChainVerifyTemplate):
        raise TypeError("template must be a ClassicChainVerifyTemplate")
    _validate_greedy_scope(batch)
    if (
        template.static_input.request_count != verify_input.request_count
        or template.static_input.logical_rows != verify_input.logical_rows
        or template.static_input.physical_rows != verify_input.physical_rows
    ):
        raise ValueError("template disagrees with the classic Forest shape")
    verify_lens = template.verify_lens_cpu
    ragged_layout = template.ragged_layout
    chain_verify_input = DFlashVerifyInput(
        draft_token=verify_input.draft_token,
        positions=verify_input.positions,
        draft_token_num=verify_input.max_request_rows,
        custom_mask=None,


        num_tokens_per_req=1,
        ragged_verify_layout=ragged_layout,
        live_seq_lens_cpu=batch.seq_lens_cpu,
    )

    previous = (
        batch.input_ids,
        batch.out_cache_loc,
        batch.spec_info,
        batch.forward_mode,
        batch.seq_lens_cpu,
        batch.seq_lens_sum,
    )
    transaction: ForestKVScratchTransaction | None = None
    target_work_may_be_in_flight = False
    try:
        transaction = ForestKVScratchTransaction.reserve_for_batch(
            batch,
            verify_input,
        )
        batch.out_cache_loc = transaction.scratch_slots
        if previous[4] is not None:
            verify_lens_host = torch.tensor(
                verify_lens,
                dtype=previous[4].dtype,
                device="cpu",
            )
            batch.seq_lens_cpu = previous[4] + verify_lens_host
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())

        with _temporary_chain_mapping(
            batch,
            request_slots_int64=template.request_slots_int64,
            depths_int64=template.depths_int64,
            logical_rows=verify_input.logical_rows,
            scratch_slots=transaction.scratch_slots,
        ):
            forward_batch, can_run_cuda_graph = (
                chain_verify_input.prepare_for_verify(batch, target_worker)
            )
            if can_run_cuda_graph:
                raise RuntimeError(
                    "classic Request Forest chain verification has not enabled "
                    "CUDA Graph replay"
                )
            target_work_may_be_in_flight = True
            target_output = target_worker.forward_batch_generation(
                batch=None,
                forward_batch=forward_batch,
                is_verify=True,
                skip_attn_backend_init=True,
            )

        logits_output = getattr(target_output, "logits_output", None)
        if logits_output is None:
            raise RuntimeError("Target verification returned no logits output")
        if getattr(target_output, "can_run_cuda_graph", False):
            raise RuntimeError(
                "classic Request Forest chain received an unexpected graph result"
            )
        target_predict = _target_top1(logits_output, verify_input)
        tp_world_size = 1 if tp_group is None else int(tp_group.world_size)
        if tp_world_size > 1:
            tp_group.broadcast(target_predict, src=0)
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
            can_run_cuda_graph=False,
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
                before_kv_publish(generation_result, acceptance, transaction)
            except Exception as exc:
                callback_error = exc
        try:
            ensure_tp_publish_ready(
                callback_error,
                tp_group=tp_group,
                scratch_device=transaction.scratch_slots.device,
            )
        except Exception:
            target_work_may_be_in_flight = False
            raise

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
            batch.seq_lens_cpu,
            batch.seq_lens_sum,
        ) = previous


__all__ = [
    "ClassicChainVerifyTemplate",
    "run_classic_chain_target_verify_eager",
]
