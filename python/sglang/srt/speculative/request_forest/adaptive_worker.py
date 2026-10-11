from __future__ import annotations

import logging
import os
import time

import torch

from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config
from sglang.srt.speculative.draft_worker_common import make_draft_input_v2

from .adaptive_graphs import activate_lane, capture_lane
from .adaptive_metrics import AdaptiveRFMetrics
from .adaptive_config import (
    validate_adaptive_draft_window,
    validate_adaptive_selector,
    validate_adaptive_target,
)
from .adaptive_policy import AdaptiveRFPolicy, PolicyConfig
from .adaptive_state import (
    AdaptiveRequestForestDraftInput,
    AdaptiveRound,
    release_native_reserve,
    release_native_reserve_at_boundary,
)
from .draft_info import RequestForestDraftInput
from .eager_runtime import _validate_greedy_scope
from .worker import RequestForestWorkerV2

logger = logging.getLogger(__name__)


class AdaptiveRequestForestWorker(RequestForestWorkerV2):
    def __init__(self, server_args, gpu_id, ps, nccl_port, target_worker):
        validate_adaptive_target(target_worker)
        self.adaptive_policy = AdaptiveRFPolicy(PolicyConfig(
            enter_rf=int(os.environ.get('SGLANG_REQUEST_FOREST_ADAPTIVE_ENTER', '64')),
            exit_rf=int(os.environ.get('SGLANG_REQUEST_FOREST_ADAPTIVE_EXIT', '48')),
        ))
        self.adaptive_metrics = AdaptiveRFMetrics()
        self.adaptive_round_counts = self.adaptive_metrics.round_counts
        super().__init__(server_args, gpu_id, ps, nccl_port, target_worker)
        validate_adaptive_selector(self.selector)


        validate_adaptive_draft_window(
            parse_dflash_draft_config(
                draft_hf_config=self.draft_model_runner.model_config.hf_config
            ),
            int(self.block_size),
            target_architecture=self.model_runner.model.__class__.__name__,
        )
        self.adaptive_debug = None
        if os.environ.get('SGLANG_REQUEST_FOREST_ADAPTIVE_DIAGNOSTICS', '0') == '1':
            from .adaptive_debug import AdaptiveRFDiagnostics
            self.adaptive_debug = AdaptiveRFDiagnostics(self)
        self._adaptive_current = None
        self._adaptive_lanes = {}
        self._adaptive_active_mode = 'rf'
        self._adaptive_trace_rounds = os.environ.get('SGLANG_REQUEST_FOREST_ADAPTIVE_TRACE_ROUNDS', '0') == '1'
        self.adaptive_graph_hits = self.adaptive_metrics.graph_hits
        self._last_draft_graph_hit = False
        self._adaptive_boundary_switch_us = 0.0
        self._adaptive_prefill_counts = {
            'calls': 0,
            'target_graph_hits': 0,
            'by_decode_mode': {
                mode: {'calls': 0, 'target_graph_hits': 0}
                for mode in ('dflash2', 'rf')
            },
        }

    def get_adaptive_rf_metrics(self):
        metrics = self.adaptive_metrics.snapshot()
        metrics['active_mode'] = self._adaptive_active_mode
        metrics['next_mode'] = self.adaptive_policy.mode
        counts = self._adaptive_prefill_counts
        metrics['shared_prefill'] = {
            'calls': counts['calls'],
            'target_graph_hits': counts['target_graph_hits'],
            'by_decode_mode': {
                mode: values.copy()
                for mode, values in counts['by_decode_mode'].items()
            },
        }
        return metrics

    def _maybe_build_draft_sampler(self):
        return RequestForestWorkerV2._maybe_build_draft_sampler(self)

    def init_cuda_graphs(self):
        from sglang.srt.model_executor.graph_memory_usage import merge_graph_memory_usage, merge_graph_time_usage
        from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import capture_decode_graph
        from sglang.srt.speculative.draft_worker_common import make_draft_sampler_capture_hook
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


        target, draft = self.model_runner, self.draft_model_runner
        base_hooks = tuple(draft.capture_tail_hooks)
        DFlashWorkerV2.init_cuda_graphs(self)
        self._adaptive_lanes['rf'] = (capture_lane(target), capture_lane(draft, self._draft_sampler))
        for runner in (target, draft):
            runner.spec_algorithm = SpeculativeAlgorithm.DFLASH
            runner.init_attention_backends()


            if runner is draft:
                runner.capture_tail_hooks = list(base_hooks)
                self._draft_sampler = None


                if not self._can_capture_draft_cuda_graph(lane_name='Adaptive RF native'):
                    runner.decode_cuda_graph_runner = runner.eager_runner
                    continue
                self._draft_sampler = DFlashWorkerV2._maybe_build_draft_sampler(self)
                if self._draft_sampler is not None:
                    runner.capture_tail_hooks.append(make_draft_sampler_capture_hook(self._draft_sampler))


            capture = capture_decode_graph(model_runner=runner)
            runner.decode_cuda_graph_runner = capture.runner
            runner.graph_memory_usage = merge_graph_memory_usage(runner.graph_memory_usage, capture.memory_usage)
            runner.graph_time_usage = merge_graph_time_usage(runner.graph_time_usage, capture.time_usage)
        self._adaptive_lanes['dflash2'] = (capture_lane(target), capture_lane(draft, self._draft_sampler))
        self._adaptive_active_mode = 'dflash2'
        logger.info('Adaptive RF graph lanes initialized: target_memory=%s draft_memory=%s',
                    target.graph_memory_usage, draft.graph_memory_usage)

    def _activate_adaptive_lane(self, mode):
        if mode == self._adaptive_active_mode:
            return 0.0
        started = time.perf_counter_ns()
        target, draft = self._adaptive_lanes[mode]
        activate_lane(self.model_runner, target)
        activate_lane(self.draft_model_runner, draft)
        self._draft_sampler = draft.sampler
        self._adaptive_active_mode = mode
        return (time.perf_counter_ns() - started) / 1000.0

    def prepare_adaptive_schedule(self, running_batch, last_batch):

        requests = set()
        for batch in (running_batch, last_batch):
            if batch is None:
                continue
            state = getattr(batch, "spec_info", None)
            if isinstance(state, AdaptiveRequestForestDraftInput):
                state._require_unprepared()
            requests.update(
                id(req) for req in batch.reqs
                if not req.finished() and not req.is_retracted
            )
        self.adaptive_policy.prepare_load(len(requests), exact=False)

    def prepare_adaptive_decode(self, batch):

        state = getattr(batch, 'spec_info', None)
        if isinstance(state, AdaptiveRequestForestDraftInput):
            state._require_unprepared()
        self.adaptive_policy.prepare_load(len(batch.reqs), exact=True)

    def adaptive_overlap_transition_pending(self):

        return self.adaptive_policy.mode != self._adaptive_active_mode

    def finish_adaptive_overlap_transition(self, running_batch, last_batch):

        if not self.adaptive_overlap_transition_pending():
            return
        if self._adaptive_current is not None:
            raise RuntimeError("Cannot switch Adaptive RF during a forward")
        for batch in (running_batch, last_batch):
            state = getattr(batch, "spec_info", None)
            if isinstance(state, AdaptiveRequestForestDraftInput):
                state._require_unprepared()
        if self._adaptive_active_mode == 'dflash2':
            release_native_reserve_at_boundary((running_batch, last_batch))
        self._adaptive_boundary_switch_us += self._activate_adaptive_lane(
            self.adaptive_policy.mode
        )

    def _proposal_from_draft_output(self, **kwargs):
        self._last_draft_graph_hit = bool(kwargs['draft_output'].can_run_graph)
        return super()._proposal_from_draft_output(**kwargs)

    def _make_next_draft_input_prefill(self, *, bonus_tokens, seq_lens):
        return self._make_next_draft_input_decode(bonus_tokens=bonus_tokens, new_seq_lens=seq_lens)

    def _make_next_draft_input_decode(self, *, bonus_tokens, new_seq_lens):
        return AdaptiveRequestForestDraftInput(
            bonus_tokens=bonus_tokens.to(dtype=torch.int64),
            new_seq_lens=new_seq_lens.to(dtype=torch.int64), owner=self,
        )

    def _prepare_adaptive_round(self, batch, state):
        state._require_unprepared()
        _validate_greedy_scope(batch)
        overlap = bool(getattr(batch, 'enable_overlap', False))
        requests = tuple(batch.reqs)
        if (
            overlap and len(requests) >= self.adaptive_policy.config.enter_rf
            and self._adaptive_active_mode != 'rf'
        ):
            raise RuntimeError('Adaptive RF high load requires scheduler drain before preparation')
        if overlap and self.adaptive_overlap_transition_pending():
            raise RuntimeError('Adaptive RF lane change requires scheduler drain before preparation')
        if overlap and len(requests) == 1 and self._adaptive_active_mode != 'dflash2':
            raise RuntimeError('Adaptive RF singleton requires scheduler drain before preparation')
        if overlap and self.adaptive_debug is not None:
            raise RuntimeError('Adaptive RF ownership diagnostics require synchronous scheduling')
        if not overlap and state.future_indices is not None:
            raise RuntimeError('Synchronous Adaptive RF cannot consume overlap futures')
        if not requests:
            return
        rows = tuple(int(x) for x in batch.req_pool_indices_cpu.tolist())
        prefixes = tuple(int(req.kv_committed_len) for req in requests)
        if len(rows) != len(requests) or len(set(rows)) != len(rows):
            raise RuntimeError('Adaptive RF request rows are not unique and aligned')
        if not overlap and any(int(req.kv.kv_allocated_len) != prefix for req, prefix in zip(requests, prefixes)):
            raise RuntimeError('Adaptive RF expected an exact committed allocation boundary')
        decision = self.adaptive_policy.propose(len(requests))
        self._adaptive_switch_us = (
            self._activate_adaptive_lane(decision.mode)
            + self._adaptive_boundary_switch_us
        )
        self._adaptive_boundary_switch_us = 0.0
        prepared = AdaptiveRound(decision, requests, rows, prefixes,
                                 tuple(req.decode_batch_idx for req in requests))
        state.pending_round = prepared
        try:
            if decision.mode == 'dflash2' and len(requests) >= self.adaptive_policy.config.enter_rf:
                raise RuntimeError('Adaptive RF cannot prepare native decode at high load')
            if self.adaptive_debug is not None:
                self.adaptive_debug.snapshot(batch, prepared)
            if decision.mode == 'dflash2':
                width = batch.req_to_token_pool.req_to_token.shape[1]
                if any(prefix + 2 * self.block_size > width for prefix in prefixes):
                    raise RuntimeError('Insufficient request-table headroom for the native 2W reserve')
                native = make_draft_input_v2(bonus_tokens=state.bonus_tokens, new_seq_lens=state.new_seq_lens)
                prepared.native_input = native
                native.prepare_for_decode(batch)
            else:
                RequestForestDraftInput.prepare_for_decode(state, batch)
        except BaseException:
            self.adaptive_policy.abort(decision)
            state.pending_round = None


            raise

    def _prepare_accepted_draft_state(self, verify_input, generation_result, acceptance, transaction):
        if self.adaptive_debug is not None:
            batch, prepared = self._adaptive_current
            self.adaptive_debug.trace_result(prepared, generation_result, acceptance.accept_index)
            from .adaptive_shadow import compare_native_on_forest_prefix
            compare_native_on_forest_prefix(self, batch, prepared, verify_input, generation_result, acceptance)
        return super()._prepare_accepted_draft_state(verify_input, generation_result, acceptance, transaction)

    def _forward_shared_prefill(self, batch, on_publish=None, grammar_barrier=None):

        decode_mode = self._adaptive_active_mode
        self._activate_adaptive_lane('rf')
        try:
            result = DFlashWorkerV2.forward_batch_generation(
                self, batch, on_publish=on_publish, grammar_barrier=grammar_barrier
            )
        finally:
            self._activate_adaptive_lane(decode_mode)
        graph_hit = int(bool(result.can_run_cuda_graph))
        counts = self._adaptive_prefill_counts
        counts['calls'] += 1
        counts['target_graph_hits'] += graph_hit
        by_mode = counts['by_decode_mode'][decode_mode]
        by_mode['calls'] += 1
        by_mode['target_graph_hits'] += graph_hit
        return result

    def forward_batch_generation(self, batch, on_publish=None, grammar_barrier=None):
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            return self._forward_shared_prefill(batch, on_publish, grammar_barrier)
        if batch.forward_mode.is_idle():
            return RequestForestWorkerV2.forward_batch_generation(
                self, batch, on_publish=on_publish
            )
        state = batch.spec_info
        if not isinstance(state, AdaptiveRequestForestDraftInput) or state.owner is not self:
            raise RuntimeError('Adaptive RF requires its own prepared state')
        prepared = state.pending_round
        if prepared is None:
            raise RuntimeError('Adaptive RF round was not prepared before dispatch')
        if len(batch.reqs) != len(prepared.requests) or any(a is not b for a, b in zip(batch.reqs, prepared.requests)):
            raise RuntimeError('Adaptive RF batch changed after preparation')
        decision = prepared.decision
        self._adaptive_current = (batch, prepared)


        self._last_draft_graph_hit = False
        try:
            if decision.mode == 'dflash2':
                if len(prepared.requests) >= self.adaptive_policy.config.enter_rf:
                    raise RuntimeError('Adaptive RF cannot execute native decode at high load')


                prepared.native_input.bonus_tokens = state.bonus_tokens
                prepared.native_input.new_seq_lens = batch.seq_lens
                batch.spec_info = prepared.native_input
                result = DFlashWorkerV2.forward_batch_generation(
                    self, batch, on_publish=on_publish, grammar_barrier=grammar_barrier
                )
                result.adaptive_rf_mode = 'dflash2'
                if not batch.enable_overlap:
                    host_result = torch.stack((result.new_seq_lens, result.accept_lens)).to('cpu')
                    ends, counts = (tuple(int(x) for x in row) for row in host_result.tolist())
                    if len(ends) != len(prepared.requests) or len(counts) != len(ends) or any(
                        not 1 <= count <= self.block_size or end != prefix + count
                        for prefix, end, count in zip(prepared.committed_lens, ends, counts)
                    ):
                        raise RuntimeError('Native DFlash2 result violates committed-length contract')
                    release_native_reserve(batch, prepared, ends)

                    result.new_seq_lens_cpu = host_result[0].to(dtype=result.new_seq_lens.dtype)
                    result.accept_lens_cpu = host_result[1].to(dtype=result.accept_lens.dtype)
                    result.accepted_draft_lens = result.accept_lens_cpu - 1
                    result.proposed_draft_lens = torch.full_like(result.accept_lens_cpu, self.block_size - 1)


                if self.adaptive_debug is not None:
                    self.adaptive_debug.trace_result(prepared, result)
            else:
                result = RequestForestWorkerV2._forward_decode(
                    self, batch, state, on_publish=on_publish
                )
                result.adaptive_rf_mode = 'rf'
            if not isinstance(result.next_draft_input, AdaptiveRequestForestDraftInput):
                raise RuntimeError('Execution path did not return adaptive relay state')
            if self.adaptive_debug is not None:
                self.adaptive_debug.finish(batch, prepared, result)
        except BaseException:
            self.adaptive_policy.abort(decision)
            raise
        finally:
            state.pending_round = None
            self._adaptive_current = None
        switched = self.adaptive_policy.commit(decision)
        self.adaptive_metrics.record_round(
            mode=decision.mode,
            active_batch=decision.active_batch,
            target_graph=result.can_run_cuda_graph,
            draft_graph=self._last_draft_graph_hit,
            lane_assignment_us=self._adaptive_switch_us,
        )
        if self._adaptive_trace_rounds or switched or self._adaptive_switch_us:
            logger.info('Adaptive RF mode=%s batch=%d switched=%s next=%s counts=%s graphs=%s switch_us=%.3f',
                        decision.mode, decision.active_batch, switched,
                        self.adaptive_policy.mode, self.adaptive_round_counts,
                        self.adaptive_graph_hits, self._adaptive_switch_us)
        return result
