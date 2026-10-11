from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Callable,
    List,
    Optional,
    Tuple,
    Union,
)

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_MATCHED_TOKEN,
    Req,
    ScheduleBatch,
    mamba_lazy_spec_in_window,
)
from sglang.srt.mem_cache.common import (
    maybe_cache_unfinished_req,
    release_kv_cache,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    get_required_capture_hidden_mode,
    get_server_return_hidden_states_mode,
)
from sglang.srt.runtime_context import (
    get_disagg,
    get_memory,
    get_observability,
    mamba_extra_buffer_lazy_enabled,
    mamba_track_grid,
    max_speculative_num_draft_tokens,
)
from sglang.srt.sampling.sampling_observer import CommittedTokens
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.state_capturer.indexer_topk import get_global_indexer_capturer
from sglang.srt.state_capturer.routed_experts import get_global_experts_capturer

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
        DecodeKVCacheOffloadManager,
    )
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
    from sglang.srt.managers.scheduler_components.logprob_result_processor import (
        SchedulerLogprobResultProcessor,
    )
    from sglang.srt.managers.scheduler_components.metrics_reporter import (
        SchedulerMetricsReporter,
    )
    from sglang.srt.managers.scheduler_components.output_streamer import (
        SchedulerOutputStreamer,
    )
    from sglang.srt.managers.tp_worker import BaseTpWorker
    from sglang.srt.managers.utils import (
        EmbeddingBatchResult,
        GenerationBatchResult,
    )
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector
    from sglang.srt.sampling.sampling_observer import HostAuxiliaryOutput

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerBatchResultProcessor:
    is_generation: bool
    disaggregation_mode: DisaggregationMode
    enable_overlap: bool
    enable_overlap_mlx: bool
    model_config: ModelConfig
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    tree_cache: BasePrefixCache
    hisparse_coordinator: Optional[HiSparseCoordinator]
    req_to_token_pool: ReqToTokenPool
    decode_offload_manager: Optional[DecodeKVCacheOffloadManager]
    metrics_collector: SchedulerMetricsCollector
    metrics_reporter: SchedulerMetricsReporter
    draft_worker: BaseTpWorker
    model_worker: BaseTpWorker
    logprob_result_processor: SchedulerLogprobResultProcessor
    output_streamer: SchedulerOutputStreamer
    abort_request: Callable

    def process_batch_result_prebuilt(self, batch: ScheduleBatch):
        assert self.disaggregation_mode == DisaggregationMode.DECODE
        use_free_group = get_disagg().disaggregation_decode_enable_radix_cache
        if use_free_group:
            self.token_to_kv_pool_allocator.free_group_begin()
        for req in batch.reqs:
            req.time_stats.set_decode_prebuilt_finish_time()
            req.update_finish_state()
            if req.finished():
                req.time_stats.set_quick_finish_time()
                if get_memory().enable_hisparse:
                    self.hisparse_coordinator.request_finished(req)
                release_kv_cache(req, self.tree_cache)


        self.output_streamer.stream_output(batch.reqs, batch.return_logprob)
        if use_free_group:
            self.token_to_kv_pool_allocator.free_group_end()

    def _maybe_collect_routed_experts(self, req: Req):

        if not req.return_routed_experts:
            return
        capturer = get_global_experts_capturer()
        if capturer is None:
            return
        start_len = req.routed_experts_start_len
        seqlen = len(req.origin_input_ids) + len(req.output_ids_through_stop)
        req.routed_experts = capturer.get_topk(
            req_pool_idx=req.req_pool_idx,
            seqlen=seqlen,
            req_to_token_pool=self.req_to_token_pool,
            start_len=start_len,
        )

        expected_rows = max(0, seqlen - 1 - start_len)
        if (
            req.routed_experts is not None
            and req.routed_experts.shape[0] != expected_rows
        ):
            logger.warning(
                "routed_experts row-count mismatch for req %s: got %d, expected %d "
                "(seqlen=%d, raw_seqlen=%d, cached_tokens=%d, start_len=%s). "
                "This indicates a silent bug.",
                req.rid,
                req.routed_experts.shape[0],
                expected_rows,
                seqlen,
                req.seqlen,
                req.cached_tokens,
                req.routed_experts_start_len,
            )

    def _maybe_collect_indexer_topk(self, req: Req):
        capturer = get_global_indexer_capturer()
        if capturer is None:
            return
        seqlen = len(req.origin_input_ids) + len(req.output_ids_through_stop)
        req.indexer_topk = capturer.get_topk(
            req_pool_idx=req.req_pool_idx,
            seqlen=seqlen,
            req_to_token_pool=self.req_to_token_pool,
        )

    def _maybe_collect_customized_info(
        self,
        i: int,
        req: Req,
        logits_output: LogitsProcessorOutput,
    ):
        if logits_output is not None and logits_output.customized_info is not None:
            if req.customized_info is None:
                req.customized_info = {}
            for k, v in logits_output.customized_info.items():
                if k not in req.customized_info:
                    req.customized_info[k] = []


                elem = v[i]
                if isinstance(elem, torch.Tensor):
                    elem = elem.clone()
                elif hasattr(elem, "copy") and callable(elem.copy):
                    elem = elem.copy()
                req.customized_info[k].append(elem)

    @staticmethod
    def _visible_output_len(req: Req) -> int:
        return req.finished_len if req.finished_len is not None else len(req.output_ids)

    @classmethod
    def snapshot_auxiliary_output_starts(
        cls,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> Optional[List[int]]:
        if result.auxiliary_host_output is None:
            return None
        return [cls._visible_output_len(req) for req in batch.reqs]

    @classmethod
    def _build_auxiliary_commits(
        cls,
        batch: ScheduleBatch,
        output_starts: List[int],
    ) -> List[Optional[CommittedTokens]]:
        commits: List[Optional[CommittedTokens]] = []
        for req, output_start in zip(batch.reqs, output_starts, strict=True):
            output_end = cls._visible_output_len(req)
            if output_end < output_start:
                raise RuntimeError("committed output length moved backwards")
            if output_end == output_start:
                commits.append(None)
                continue
            commits.append(
                CommittedTokens(
                    output_index=output_start,
                    token_ids=tuple(req.output_ids[output_start:output_end]),
                )
            )
        return commits

    @classmethod
    def consume_auxiliary_output(
        cls,
        batch: ScheduleBatch,
        output: HostAuxiliaryOutput,
        output_starts: List[int],
    ) -> None:
        output.consume(batch, cls._build_auxiliary_commits(batch, output_starts))

    def process_batch_result_prefill(
        self,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
    ):
        skip_stream_req = None
        self.token_to_kv_pool_allocator.free_group_begin()

        if self.is_generation:
            if result.copy_done is not None:
                result.copy_done.synchronize()
            auxiliary_output_starts = self.snapshot_auxiliary_output_starts(
                batch, result
            )
            auxiliary_output = result.auxiliary_host_output
            if result.routed_experts_output is not None:
                result.routed_experts_output.finalize()
                result.routed_experts_output = None
            if result.indexer_topk_output is not None:
                result.indexer_topk_output.finalize()
                result.indexer_topk_output = None

            (
                logits_output,
                next_token_ids,
                extend_input_len_per_req,
                extend_logprob_start_len_per_req,
            ) = (
                result.logits_output,
                result.next_token_ids,
                result.extend_input_len_per_req,
                result.extend_logprob_start_len_per_req,
            )


            next_token_ids = next_token_ids.tolist()
            self.move_logprobs_to_cpu(batch=batch, logits_output=logits_output)

            self._validate_pp_skip_output_comm(batch, result)

            hidden_state_offset = 0
            prefill_hidden_capture_mode = self._get_prefill_hidden_capture_mode(
                batch,
            )


            logprob_pt = 0

            for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
                if (
                    batch.return_hidden_states
                    and logits_output.hidden_states is not None
                ):
                    assert extend_input_len_per_req is not None
                    hidden_state_offset = self._append_prefill_hidden_states(
                        req=req,
                        logits_output=logits_output,
                        hidden_state_offset=hidden_state_offset,
                        capture_hidden_mode=prefill_hidden_capture_mode,
                        extend_input_len=extend_input_len_per_req[i],
                        store=(
                            not req.finished()
                            and not req.is_retracted
                            and req.inflight_middle_chunks <= 0
                        ),
                    )

                if (
                    req.finished() and req.inflight_middle_chunks <= 0
                ) or req.is_retracted:


                    continue

                if req.inflight_middle_chunks <= 0:
                    req.time_stats.set_prefill_finished_time()


                    req.output_ids.append(next_token_id)

                    self._maybe_update_reasoning_tokens(req, next_token_id)

                    req.update_finish_state()
                    if req.finished():
                        self._maybe_collect_routed_experts(req)
                        self._maybe_collect_indexer_topk(req)
                        release_kv_cache(req, self.tree_cache)
                        req.time_stats.set_completion_time()
                    elif not batch.decoding_reqs or req not in batch.decoding_reqs:
                        maybe_cache_unfinished_req(req, self.tree_cache)
                        if get_memory().enable_hisparse:
                            self.hisparse_coordinator.admit_request_into_staging(req)

                    self._maybe_collect_customized_info(i, req, logits_output)

                    if batch.return_logprob:
                        logprob_pt = self._apply_prefill_logprobs(
                            req=req,
                            i=i,
                            logits_output=logits_output,
                            extend_input_len_per_req=extend_input_len_per_req,
                            extend_logprob_start_len_per_req=extend_logprob_start_len_per_req,
                            next_token_ids=next_token_ids,
                            logprob_pt=logprob_pt,
                        )

                    if req.return_sampling_mask:
                        self.add_sampling_mask_return_values(i, req, logits_output)

                    if req.grammar is not None:
                        self._apply_prefill_grammar(
                            req=req,
                            next_token_id=next_token_id,
                            already_advanced=result.grammar_advanced,
                        )

                else:

                    req.inflight_middle_chunks -= 1


                    skip_stream_req = req


                    if batch.return_logprob:
                        logprob_pt = self._apply_chunked_prefill_logprobs(
                            req=req,
                            i=i,
                            logits_output=logits_output,
                            extend_input_len_per_req=extend_input_len_per_req,
                            extend_logprob_start_len_per_req=extend_logprob_start_len_per_req,
                            logprob_pt=logprob_pt,
                        )

                    req.time_stats.set_last_chunked_prefill_finish_time()

            if auxiliary_output is not None:
                self.consume_auxiliary_output(
                    batch,
                    auxiliary_output,
                    auxiliary_output_starts,
                )

        else:
            if result.copy_done is not None:
                result.copy_done.synchronize()

            embeddings = self._convert_embeddings(result=result)
            phs = result.pooled_hidden_states

            if phs is not None:
                if isinstance(phs, list):
                    phs = [t.cpu().detach() for t in phs]
                else:
                    phs = phs.cpu().detach()


            for i, req in enumerate(batch.reqs):
                if req.is_retracted:
                    continue

                req.embedding = embeddings[i]
                if req.return_pooled_hidden_states and phs is not None:
                    req.pooled_hidden_state = phs[i]
                if req.inflight_middle_chunks <= 0:
                    req.time_stats.set_prefill_finished_time()

                    req.output_ids.append(0)
                    req.update_finish_state()

                    if req.finished():
                        release_kv_cache(req, self.tree_cache)
                        req.time_stats.set_completion_time()
                    else:
                        maybe_cache_unfinished_req(req, self.tree_cache)
                else:

                    req.inflight_middle_chunks -= 1
                    req.time_stats.set_last_chunked_prefill_finish_time()

        self.token_to_kv_pool_allocator.free_group_end()
        self.output_streamer.stream_output(
            batch.reqs, batch.return_logprob, skip_stream_req
        )

        can_run_cuda_graph = result.can_run_cuda_graph
        self.metrics_reporter.report_prefill_stats(
            batch=batch,
            prefill_stats=batch.prefill_stats,
            can_run_cuda_graph=can_run_cuda_graph,
            dp_cooperation_info=batch.dp_cooperation_info,
        )

    def _convert_embeddings(self, *, result: EmbeddingBatchResult) -> list:
        is_sparse = envs.SGLANG_EMBEDDINGS_SPARSE_HEAD.is_set()

        embeddings = result.embeddings

        if is_sparse:
            batch_ids, token_ids = embeddings.indices()
            values = embeddings.values()

            embeddings = [{} for _ in range(embeddings.size(0))]
            for i in range(batch_ids.shape[0]):
                embeddings[batch_ids[i].item()][token_ids[i].item()] = values[i].item()
        else:
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.tolist()
            else:
                embeddings = [tensor.tolist() for tensor in embeddings]
        return embeddings

    def move_logprobs_to_cpu(
        self,
        *,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
    ) -> None:
        if batch.return_logprob:
            if logits_output.next_token_logprobs is not None:
                logits_output.next_token_logprobs = (
                    logits_output.next_token_logprobs.tolist()
                )
            if logits_output.input_token_logprobs is not None:
                logits_output.input_token_logprobs = tuple(
                    logits_output.input_token_logprobs.tolist()
                )
            if logits_output.next_token_top_logprobs_val:
                logits_output.next_token_top_logprobs_val = [
                    v.tolist() for v in logits_output.next_token_top_logprobs_val
                ]
                logits_output.next_token_top_logprobs_idx = [
                    x.tolist() for x in logits_output.next_token_top_logprobs_idx
                ]
            if logits_output.next_token_token_ids_logprobs_val:
                logits_output.next_token_token_ids_logprobs_val = [
                    v.tolist() for v in logits_output.next_token_token_ids_logprobs_val
                ]

    def _apply_prefill_logprobs(
        self,
        *,
        req: Req,
        i: int,
        logits_output: LogitsProcessorOutput,
        extend_input_len_per_req: Optional[List[int]],
        extend_logprob_start_len_per_req: Optional[List[int]],
        next_token_ids: List[int],
        logprob_pt: int,
    ) -> int:
        assert extend_logprob_start_len_per_req is not None
        assert extend_input_len_per_req is not None
        extend_logprob_start_len = extend_logprob_start_len_per_req[i]
        extend_input_len = extend_input_len_per_req[i]

        num_input_logprobs = self.logprob_result_processor.calculate_num_input_logprobs(
            req,
            extend_input_len,
            extend_logprob_start_len,
        )

        if req.return_logprob:
            self.logprob_result_processor.add_logprob_return_values(
                i,
                req,
                logprob_pt,
                next_token_ids,
                num_input_logprobs,
                logits_output,
            )
        logprob_pt += num_input_logprobs
        return logprob_pt

    @staticmethod
    def _validate_pp_skip_output_comm(
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
    ):

        if not envs.SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM.get():
            return

        if not getattr(result, "skipped_output_comm", False):
            if batch.forward_mode.is_extend() and not batch.forward_mode.is_prebuilt():
                has_consumed_output = any(
                    req.inflight_middle_chunks <= 0
                    for req in batch.reqs
                    if not req.finished() and not req.is_retracted
                )
                if not has_consumed_output and len(batch.reqs) > 0:
                    chunks = list([r.inflight_middle_chunks for r in batch.reqs])
                    logger.warning(
                        f"PP non-skip output comm: no req consumed next_token_ids. "
                        f"contains_last_prefill_chunk={batch.contains_last_prefill_chunk}, "
                        f"num_reqs={len(batch.reqs)}, all inflight_middle_chunks={chunks}"
                    )
            return

        for req in batch.reqs:
            if not req.finished() and not req.is_retracted:
                assert req.inflight_middle_chunks > 0, (
                    f"PP skip output comm invariant violated: req {req.rid} "
                    f"has inflight_middle_chunks={req.inflight_middle_chunks} "
                    f"but output was skipped (contains_last_prefill_chunk="
                    f"{batch.contains_last_prefill_chunk}). "
                    f"Placeholder zeros would be appended to output_ids."
                )

    def _append_prefill_hidden_states(
        self,
        *,
        req: Req,
        logits_output: LogitsProcessorOutput,
        hidden_state_offset: int,
        capture_hidden_mode: CaptureHiddenMode,
        extend_input_len: int,
        store: bool = True,
    ) -> int:
        if capture_hidden_mode.is_full():
            start = hidden_state_offset
            hidden_state_offset += extend_input_len
            if not store or not req.return_hidden_states:
                return hidden_state_offset

            req_hidden_states = logits_output.hidden_states[start:hidden_state_offset]
            if req.return_hidden_states is True:
                req.hidden_states.append(req_hidden_states.cpu().clone().tolist())
            elif req.return_hidden_states == "last":
                req.hidden_states.append(req_hidden_states[-1].cpu().tolist())
        elif capture_hidden_mode.is_last():
            index = hidden_state_offset
            hidden_state_offset += 1
            if store and req.return_hidden_states:
                req.hidden_states.append(
                    logits_output.hidden_states[index].cpu().tolist()
                )
        else:
            raise ValueError(
                f"Unexpected hidden states capture mode: {capture_hidden_mode}"
            )
        return hidden_state_offset

    @staticmethod
    def _append_decode_hidden_states(
        *,
        req: Req,
        hidden_states: torch.Tensor,
        start: int,
        accept_len: int,
    ) -> None:
        if accept_len <= 0:
            return

        if req.return_hidden_states == "last":
            valid_accept_len = accept_len
            if req.finished_len is not None:
                step_start = len(req.output_ids) - accept_len
                valid_accept_len = max(
                    0,
                    min(accept_len, req.finished_len - step_start),
                )
            if valid_accept_len > 0:
                req.hidden_states[:] = [
                    hidden_states[start + valid_accept_len - 1].cpu().tolist()
                ]
        else:
            req.hidden_states.extend(
                hidden_states[start : start + accept_len].cpu().tolist()
            )

    @staticmethod
    def _get_prefill_hidden_capture_mode(batch: ScheduleBatch) -> CaptureHiddenMode:
        return get_required_capture_hidden_mode(
            max(
                batch.return_hidden_states_mode,
                get_server_return_hidden_states_mode(),
            ),
            batch.spec_info,
        )

    def _apply_prefill_grammar(
        self, *, req: Req, next_token_id: int, already_advanced: bool = False
    ) -> None:


        if not already_advanced:

            try:
                req.grammar.accept_token(next_token_id)
            except ValueError as e:


                logger.error(
                    f"Grammar accept_token failed for req {req.rid} with token {next_token_id}: {e}"
                )
                req.to_finish = FINISH_ABORT()
        req.grammar.finished = req.finished()

    def _apply_chunked_prefill_logprobs(
        self,
        *,
        req: Req,
        i: int,
        logits_output: LogitsProcessorOutput,
        extend_input_len_per_req: Optional[List[int]],
        extend_logprob_start_len_per_req: Optional[List[int]],
        logprob_pt: int,
    ) -> int:
        extend_logprob_start_len = extend_logprob_start_len_per_req[i]
        extend_input_len = extend_input_len_per_req[i]
        if extend_logprob_start_len < extend_input_len:

            num_input_logprobs = (
                self.logprob_result_processor.calculate_num_input_logprobs(
                    req,
                    extend_input_len,
                    extend_logprob_start_len,
                )
            )
            if req.return_logprob:
                self.logprob_result_processor.add_input_logprob_return_values(
                    i,
                    req,
                    logits_output,
                    logprob_pt,
                    num_input_logprobs,
                    last_prefill_chunk=False,
                )
            logprob_pt += num_input_logprobs
        return logprob_pt

    def _resolve_spec_v2_tokens(
        self,
        result: GenerationBatchResult,
        batch: ScheduleBatch,
    ) -> List[List[int]]:

        assert result.next_token_ids.is_cpu
        assert result.accept_lens.is_cpu

        next_token_ids = result.next_token_ids.tolist()
        accept_lens = result.accept_lens.tolist()
        is_request_forest = batch.spec_algorithm.is_request_forest()
        adaptive_native = (
            is_request_forest and result.adaptive_rf_mode == "dflash2"
        )
        if adaptive_native:
            stride = result.speculative_num_draft_tokens
            assert stride is not None, "Adaptive native result missing draft width"
            result.num_correct_drafts_per_req_cpu = [x - 1 for x in accept_lens]
            result.num_proposed_drafts_per_req_cpu = [stride - 1] * len(accept_lens)
        elif is_request_forest:
            assert result.accepted_draft_lens is not None, (
                "Request Forest result missing exact accepted_draft_lens"
            )
            assert result.proposed_draft_lens is not None, (
                "Request Forest result missing exact proposed_draft_lens"
            )
            assert result.accepted_draft_lens.is_cpu
            assert result.proposed_draft_lens.is_cpu
            result.num_correct_drafts_per_req_cpu = (
                result.accepted_draft_lens.tolist()
            )
            result.num_proposed_drafts_per_req_cpu = (
                result.proposed_draft_lens.tolist()
            )
        else:
            result.num_correct_drafts_per_req_cpu = [x - 1 for x in accept_lens]
        result.num_correct_drafts = sum(result.num_correct_drafts_per_req_cpu)

        block_accept_lens = (
            result.block_accept_lens.tolist()
            if result.block_accept_lens is not None
            else None
        )
        result.num_block_accept_tokens = (
            sum(block_accept_lens) if block_accept_lens else 0
        )
        cap_lens = result.cap_lens.tolist() if result.cap_lens is not None else None
        result.num_cap_tokens = sum(cap_lens) if cap_lens else 0


        self.model_worker.on_verify_complete_cpu(
            result.num_correct_drafts_per_req_cpu, batch_size=len(batch.reqs)
        )


        self.advance_grammar_fsm(result, batch)

        predict_tokens = []


        stride = result.speculative_num_draft_tokens
        assert stride is not None, "spec-v2 result missing speculative_num_draft_tokens"

        for i, req in enumerate(batch.reqs):
            accept_tokens = next_token_ids[i * stride : i * stride + accept_lens[i]]

            if req.is_retracted or req.finished():


                pass
            else:
                if req.grammar is not None:


                    accept_tokens = result.grammar_retained_tokens[i]


                num_accept_tokens = len(accept_tokens)
                req.kv_committed_len += num_accept_tokens
                if is_request_forest and not adaptive_native:


                    req.kv.kv_allocated_len = req.kv_committed_len
                req.spec_verify_ct += 1

                num_correct_drafts = result.num_correct_drafts_per_req_cpu[i]
                req.spec_num_correct_drafts += num_correct_drafts
                req.update_spec_correct_drafts_histogram(num_correct_drafts)
                if is_request_forest:
                    assert result.num_proposed_drafts_per_req_cpu is not None
                    if req.spec_num_proposed_drafts is None:
                        req.spec_num_proposed_drafts = 0
                    req.spec_num_proposed_drafts += (
                        result.num_proposed_drafts_per_req_cpu[i]
                    )

                if block_accept_lens is not None:
                    req.spec_num_block_accept_tokens += block_accept_lens[i]
                if cap_lens is not None:
                    req.spec_num_cap_tokens += cap_lens[i]
                    req.update_spec_cap_lens_histogram(cap_lens[i])

            predict_tokens.append(accept_tokens)

        return predict_tokens

    def _accept_grammar_tokens(
        self, req: Req, tokens: Union[int, List[int]]
    ) -> List[int]:

        if isinstance(tokens, int):
            tokens = [tokens]
        retained = []
        try:
            for token_id in tokens:
                req.grammar.accept_token(token_id)
                retained.append(token_id)
                if req.grammar.is_terminated():
                    break
        except ValueError as e:


            logger.error(
                f"Grammar accept_token failed for req {req.rid} with token "
                f"{tokens}: {e}"
            )
            req.to_finish = FINISH_ABORT()
        return retained

    def advance_grammar_fsm(
        self, result: GenerationBatchResult, batch: ScheduleBatch
    ) -> None:

        if result.grammar_advanced or not batch.has_grammar:
            return
        is_decode = batch.forward_mode.is_decode()
        if not (is_decode or batch.forward_mode.is_extend()):
            return
        if result.copy_done is not None:
            result.copy_done.synchronize()
        next_token_ids = result.next_token_ids.tolist()

        if not is_decode:


            for i, req in enumerate(batch.reqs):
                if (
                    req.grammar is None
                    or req.is_retracted
                    or req.finished()
                    or req.inflight_middle_chunks > 0
                ):
                    continue
                self._accept_grammar_tokens(req, next_token_ids[i])
            result.grammar_advanced = True
            return


        if result.accept_lens is None:
            return
        accept_lens = result.accept_lens.tolist()
        stride = result.speculative_num_draft_tokens
        assert stride is not None, "spec-v2 result missing speculative_num_draft_tokens"
        retained = [None] * len(batch.reqs)
        for i, req in enumerate(batch.reqs):
            if req.grammar is None or req.is_retracted or req.finished():
                continue
            accept_tokens = next_token_ids[i * stride : i * stride + accept_lens[i]]


            retained[i] = self._accept_grammar_tokens(req, accept_tokens)
        result.grammar_retained_tokens = retained
        result.grammar_advanced = True

    def process_batch_result_idle(
        self,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        if result.copy_done is not None:
            result.copy_done.synchronize()

        self.output_streamer._stream_output_generation(
            batch.reqs, batch.return_logprob, is_idle_batch=True
        )

    def process_batch_result_decode(
        self,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        if result.copy_done is not None:
            result.copy_done.synchronize()
        auxiliary_output_starts = self.snapshot_auxiliary_output_starts(batch, result)
        auxiliary_output = result.auxiliary_host_output
        if result.routed_experts_output is not None:
            result.routed_experts_output.finalize()
            result.routed_experts_output = None
        if result.indexer_topk_output is not None:
            result.indexer_topk_output.finalize()
            result.indexer_topk_output = None

        logits_output, next_token_ids, can_run_cuda_graph = (
            result.logits_output,
            result.next_token_ids,
            result.can_run_cuda_graph,
        )

        next_token_ids, next_token_logprobs = self._normalize_decode_outputs(
            batch=batch,
            result=result,
            logits_output=logits_output,
            next_token_ids=next_token_ids,
        )

        self.metrics_reporter.num_generated_tokens += len(batch.reqs)
        if not batch.spec_algorithm.is_none():
            self.metrics_reporter.update_spec_metrics(
                batch.batch_size(),
                result.num_correct_drafts,
                num_block_accept_tokens=result.num_block_accept_tokens,
                num_cap_tokens=result.num_cap_tokens,
            )
        if get_observability().enable_metrics:
            self.metrics_collector.increment_decode_cuda_graph_pass(
                value=can_run_cuda_graph
            )

        self.token_to_kv_pool_allocator.free_group_begin()
        self._release_unsettled_request_forest_slots(batch, result)

        for i, req in enumerate(batch.reqs):
            req: Req

            if (self.enable_overlap or self.enable_overlap_mlx) and (
                req.finished() or req.is_retracted
            ):


                continue


            next_token_id = next_token_ids[i]
            is_spec = not batch.spec_algorithm.is_none()

            req.output_ids.extend(next_token_id)
            new_accept_len = len(next_token_id)

            self._maybe_update_reasoning_tokens(req, next_token_id)
            req.time_stats.set_last_decode_finish_time()
            req.update_finish_state(new_accept_len)

            self._handle_finish_state_updated_req(req, batch, result, i, logits_output)

            if req.return_logprob:
                self._apply_decode_logprobs(
                    req=req,
                    i=i,
                    batch=batch,
                    next_token_id=next_token_id,
                    next_token_logprobs=next_token_logprobs,
                    logits_output=logits_output,
                )

            if req.return_sampling_mask:


                self.add_sampling_mask_return_values(i, req, logits_output)

            if req.return_hidden_states and logits_output.hidden_states is not None:


                stride = result.speculative_num_draft_tokens or 1
                accept_len = len(next_token_id)
                start = i * stride
                self._append_decode_hidden_states(
                    req=req,
                    hidden_states=logits_output.hidden_states,
                    start=start,
                    accept_len=accept_len,
                )

            if req.grammar is not None:
                if not is_spec:


                    self._accept_grammar_tokens(req, next_token_id)
                req.grammar.finished = req.finished()

        if auxiliary_output is not None:
            self.consume_auxiliary_output(
                batch,
                auxiliary_output,
                auxiliary_output_starts,
            )

        self.output_streamer.stream_output(batch.reqs, batch.return_logprob)
        self.token_to_kv_pool_allocator.free_group_end()

        self.metrics_reporter.forward_ct_decode = (
            self.metrics_reporter.forward_ct_decode + 1
        ) % (1 << 30)
        self.metrics_reporter.report_decode_stats(
            can_run_cuda_graph,
            running_batch=batch,
            num_correct_drafts=result.num_correct_drafts,
        )

    def _release_unsettled_request_forest_slots(
        self,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> None:


        committed_slots = result.request_forest_committed_slots
        if (
            committed_slots is None
            or not batch.spec_algorithm.is_request_forest()
            or not (self.enable_overlap or self.enable_overlap_mlx)
        ):
            return

        accept_lens = result.accept_lens.tolist()
        offset = 0
        unsettled = []
        for req, accept_len in zip(batch.reqs, accept_lens, strict=True):
            next_offset = offset + int(accept_len)
            if req.finished() or req.is_retracted:
                unsettled.append(committed_slots[offset:next_offset])
            offset = next_offset
        if offset != committed_slots.numel():
            raise RuntimeError(
                "Request Forest committed-slot count disagrees with accept_lens"
            )
        if unsettled:
            self.token_to_kv_pool_allocator.free(torch.cat(unsettled))

    def _normalize_decode_outputs(
        self,
        *,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
        logits_output: LogitsProcessorOutput,
        next_token_ids: Union[torch.Tensor, List[int]],
    ) -> Tuple[Union[List[int], List[List[int]]], Optional[List[float]]]:
        next_token_logprobs = None


        if not batch.spec_algorithm.is_none():
            next_token_ids = self._resolve_spec_v2_tokens(result, batch)
        else:

            ids = (
                next_token_ids.tolist()
                if torch.is_tensor(next_token_ids)
                else next_token_ids
            )
            next_token_ids = [[t] for t in ids]

        if batch.return_logprob:
            next_token_logprobs = logits_output.next_token_logprobs.tolist()
            if logits_output.next_token_top_logprobs_val:
                logits_output.next_token_top_logprobs_val = [
                    v.tolist() for v in logits_output.next_token_top_logprobs_val
                ]
                logits_output.next_token_top_logprobs_idx = [
                    x.tolist() for x in logits_output.next_token_top_logprobs_idx
                ]

            if logits_output.next_token_token_ids_logprobs_val:
                logits_output.next_token_token_ids_logprobs_val = [
                    v.tolist() for v in logits_output.next_token_token_ids_logprobs_val
                ]
        return next_token_ids, next_token_logprobs

    def _apply_decode_logprobs(
        self,
        *,
        req: Req,
        i: int,
        batch: ScheduleBatch,
        next_token_id: Union[int, List[int]],
        next_token_logprobs: list,
        logits_output: LogitsProcessorOutput,
    ) -> None:


        if not batch.spec_algorithm.is_none():
            accepted_logprobs = next_token_logprobs[i]
            accepted_ids = next_token_id
            max_accept = len(accepted_logprobs)
        else:
            accepted_logprobs = [next_token_logprobs[i]]
            accepted_ids = next_token_id
            max_accept = 1

        for j, tok_id in enumerate(accepted_ids):
            req.logprob.output_token_logprobs_val.append(accepted_logprobs[j])
            req.logprob.output_token_logprobs_idx.append(tok_id)
            if req.logprob.top_logprobs_num > 0:
                flat_idx = i * max_accept + j
                req.logprob.output_top_logprobs_val.append(
                    logits_output.next_token_top_logprobs_val[flat_idx]
                )
                req.logprob.output_top_logprobs_idx.append(
                    logits_output.next_token_top_logprobs_idx[flat_idx]
                )
            if req.logprob.token_ids_logprob is not None:
                flat_idx = i * max_accept + j
                req.logprob.output_token_ids_logprobs_val.append(
                    logits_output.next_token_token_ids_logprobs_val[flat_idx]
                )
                req.logprob.output_token_ids_logprobs_idx.append(
                    logits_output.next_token_token_ids_logprobs_idx[flat_idx]
                )

    def add_sampling_mask_return_values(
        self,
        i: int,
        req: Req,
        output: LogitsProcessorOutput,
    ) -> None:

        mask = output.next_token_sampling_mask_idx
        logprobs = output.next_token_sampling_logprobs
        req.output_token_sampling_mask.append(None if mask is None else mask[i])
        req.output_token_sampling_logprobs.append(
            None if logprobs is None else logprobs[i]
        )

    def _handle_finish_state_updated_req(
        self,
        req: Req,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
        i: int,
        logits_output: LogitsProcessorOutput,
    ):
        lazy = mamba_extra_buffer_lazy_enabled()
        known_mamba_boundary = None
        completed_mamba_boundary = None
        lookahead = 0
        if batch.mamba_track_mask_cpu is not None:
            completed_mamba_boundary = bool(batch.mamba_track_mask_cpu[i])
            lookahead = req.decode_batch_idx - batch.mamba_decode_batch_idx_cpu[i]
            assert lookahead in (0, 1), (
                f"mamba result lookahead={lookahead} for req {req.rid}; "
                "overlap advanced more than one decode batch"
            )
            if lookahead == 0:
                known_mamba_boundary = bool(batch.mamba_track_mask_cpu[i])
            else:
                known_mamba_boundary = bool(batch.mamba_track_mask_next_cpu[i])

            if completed_mamba_boundary and not lazy:
                req.mamba_last_track_idx = batch.mamba_track_buffer_indices[i]
                req.mamba_last_track_seqlen = req.kv_committed_len - lookahead
            elif (
                req.finished()
                and lazy
                and lookahead == 1
                and known_mamba_boundary
                and req.mamba_next_track_idx == req.mamba_last_track_idx
            ):
                req.mamba_lazy_is_insert = False


        should_update = completed_mamba_boundary if lazy else known_mamba_boundary
        if should_update is None or should_update:
            self._mamba_prefix_cache_update(
                req,
                batch,
                result,
                i,
                known_boundary=not lazy and known_mamba_boundary is True,
            )

        if (
            get_disagg().disaggregation_decode_enable_offload_kvcache
            and not req.finished()
        ):
            self.decode_offload_manager.offload_kv_cache(req)

        if req.finished():


            if isinstance(self.draft_worker, BaseSpecWorker):
                self.draft_worker.note_request_finished(
                    rid=req.rid,
                    natural_stop=isinstance(req.finished_reason, FINISH_MATCHED_TOKEN),
                )


            if req.multimodal_inputs is not None and req.session is None:
                req.multimodal_inputs.release_features()
            self._maybe_collect_routed_experts(req)
            self._maybe_collect_indexer_topk(req)

            if get_disagg().disaggregation_decode_enable_offload_kvcache:

                if not self.decode_offload_manager.offload_kv_cache(req):
                    self.decode_offload_manager.finalize_release_on_finish(req)
            else:
                if get_memory().enable_hisparse:
                    self.hisparse_coordinator.request_finished(req)
                prepare_release = getattr(
                    self.model_worker, "prepare_for_kv_cache_release", None
                )
                if callable(prepare_release):
                    prepare_release(req)
                is_insert = (
                    req.mamba_lazy_is_insert
                    if mamba_extra_buffer_lazy_enabled()
                    else True
                )
                release_kv_cache(req, self.tree_cache, is_insert=is_insert)

            req.time_stats.set_completion_time()

        self._maybe_collect_customized_info(i, req, logits_output)

    def _maybe_update_reasoning_tokens(
        self,
        req: Req,
        next_token_id: Union[int, List[int]],
    ):
        think_end_ids = self.model_config.think_end_ids
        if req.require_reasoning and think_end_ids:
            req.update_reasoning_tokens(next_token_id, think_end_ids)

    def _mamba_prefix_cache_update(
        self,
        req: Req,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
        i: int,
        known_boundary: bool = False,
    ) -> None:

        if req.mamba_ping_pong_track_buffer is None:
            return

        lazy = mamba_extra_buffer_lazy_enabled()
        if known_boundary:
            self._mamba_assert_committed_len_lookahead(req)
            track_seqlen = req.kv_committed_len
            assert track_seqlen % mamba_track_grid(self.tree_cache.page_size) == 0
            at_boundary = True
        else:
            at_boundary, track_seqlen = self._mamba_check_track_boundary(
                req, batch, result, i
            )

        if lazy and not batch.spec_algorithm.is_none():

            self._mamba_lazy_spec_update(req, batch, i, at_boundary, track_seqlen)
            return

        if not at_boundary:
            return

        track_idx = req.mamba_next_track_idx
        if not known_boundary and batch.mamba_track_buffer_indices is not None:
            track_idx = batch.mamba_track_buffer_indices[i]
        if not known_boundary:
            req.mamba_last_track_seqlen = track_seqlen
        if lazy:
            self.mamba_lazy_post_decode_at_boundary(req, batch, track_idx)
        else:
            if not known_boundary:
                req.mamba_last_track_idx = track_idx
            req.mamba_next_track_idx = (
                batch.req_to_token_pool.get_mamba_ping_pong_other_idx(track_idx)
            )

    def _mamba_lazy_spec_update(
        self, req: Req, batch: ScheduleBatch, i: int, crossed: bool, track_seqlen: int
    ) -> None:

        positions = batch.mamba_lazy_spec_track_positions_cpu
        planned_pos = (
            positions[i]
            if positions is not None and i < len(positions)
            else None
        )

        if req.finished():

            keep_written_by_this_step = (
                crossed and planned_pos == req.mamba_next_track_idx
            )
            other_idx = 1 - req.mamba_next_track_idx


            keep_may_be_written_in_flight = req.mamba_ping_pong_track_buffer[
                other_idx
            ].item() == -1 and mamba_lazy_spec_in_window(
                req,
                mamba_track_grid(self.tree_cache.page_size),
                max_speculative_num_draft_tokens(),
            )
            if (
                planned_pos is None
                or keep_written_by_this_step
                or keep_may_be_written_in_flight
            ):
                req.mamba_lazy_is_insert = False
            return

        if not crossed or planned_pos is None:
            return
        if planned_pos != req.mamba_next_track_idx:

            pool = batch.req_to_token_pool
            keep_idx = req.mamba_next_track_idx
            keep_val = req.mamba_ping_pong_track_buffer[keep_idx]
            pool.mamba_allocator.free(keep_val.unsqueeze(0))
            pool.set_mamba_ping_pong_slot(req, keep_idx, -1)
            req.mamba_next_track_idx = planned_pos


        req.mamba_last_track_idx = planned_pos
        req.mamba_last_track_seqlen = track_seqlen

    @staticmethod
    def _mamba_assert_committed_len_lookahead(req: Req) -> None:

        assert req.output_ids, (
            "mamba track boundary reached before a decode token was appended "
            f"(req {req.rid}); output_ids is empty"
        )
        token_seq_len = len(req.origin_input_ids) + len(req.output_ids) - 1
        assert (req.kv_committed_len - token_seq_len) in (0, 1), (
            f"mamba track boundary: kv_committed_len={req.kv_committed_len} "
            f"leads seq_len={token_seq_len} by more than one (req {req.rid}); "
            "overlap lookahead wider than assumed"
        )

    def _mamba_check_track_boundary(self, req, batch, result, i):

        interval = mamba_track_grid(self.tree_cache.page_size)

        if batch.spec_algorithm.is_none():
            lookahead = req.decode_batch_idx - batch.mamba_decode_batch_idx_cpu[i]
            committed_len = req.kv_committed_len - lookahead
            if committed_len % interval == 0:
                return True, committed_len
        elif result.num_correct_drafts_per_req_cpu is not None:
            cur = req.seqlen - 1
            prev = cur - result.num_correct_drafts_per_req_cpu[i] - 1
            if cur // interval != prev // interval:
                return True, cur // interval * interval

        return False, 0

    def mamba_lazy_post_decode_at_boundary(
        self, req: Req, batch: ScheduleBatch, track_idx: int
    ):

        req.mamba_last_track_idx = track_idx
        req.mamba_next_track_idx = track_idx
        other_idx = 1 - track_idx
        other_val = req.mamba_ping_pong_track_buffer[other_idx].item()
        if other_val != -1:
            pool = batch.req_to_token_pool
            pool.mamba_allocator.free(
                req.mamba_ping_pong_track_buffer[other_idx].unsqueeze(0)
            )
            pool.set_mamba_ping_pong_slot(req, other_idx, -1)
