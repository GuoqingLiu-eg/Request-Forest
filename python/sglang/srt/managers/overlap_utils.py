from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Sequence

import msgspec
import torch

from sglang.kernels.ops.speculative.gather_spec_extras import gather_spec_extras
from sglang.srt.environ import envs
from sglang.srt.runtime_context import (
    get_exec,
    get_spec,
)
from sglang.srt.utils import is_cuda, is_hip, is_npu

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.speculative.eagle_info import EagleDraftInput
    from sglang.srt.speculative.ngram_info import NgramVerifyInput
    from sglang.srt.speculative.request_forest.draft_info import (
        RequestForestDraftInput,
    )
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def decide_needs_cpu_seq_lens(
    attn_backends: Sequence[AttentionBackend],
) -> bool:


    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    if get_exec().overlap.enable_two_batch_overlap:

        return True
    algo = SpeculativeAlgorithm.from_string(get_spec().speculative_algorithm)
    if algo.is_ngram() or algo.is_request_forest():


        return True


    return any(
        getattr(b, "needs_cpu_seq_lens", True) for b in attn_backends if b is not None
    )


def decide_needs_confidence_relay() -> bool:
    from sglang.srt.speculative.ragged_verify import (
        RaggedVerifyMode,
        read_ragged_verify_mode,
    )
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    algo = SpeculativeAlgorithm.from_string(get_spec().speculative_algorithm)
    if not algo.is_dspark():
        return False
    return read_ragged_verify_mode() is not RaggedVerifyMode.STATIC


_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()


_DEBUG_ASSERT = envs.SGLANG_IS_IN_CI.get()


@torch.compile(dynamic=True, disable=_is_npu)
def _assert_nonneg_and_invalidate(
    values: torch.Tensor, buf: torch.Tensor, indices: torch.Tensor
) -> None:

    torch._assert_async((values >= 0).all())
    buf[indices] = -1


def resolve_forward_inputs(batch: ScheduleBatch, future_map: FutureMap) -> None:

    if batch.prefill_input_ids_cpu is not None:
        prefill_gpu = batch.prefill_input_ids_cpu.to(batch.device, non_blocking=True)
        if batch.mix_running_indices is not None:
            decode_gpu = future_map.output_tokens_buf[batch.mix_running_indices]
            if _DEBUG_ASSERT:
                _assert_nonneg_and_invalidate(
                    decode_gpu,
                    future_map.output_tokens_buf,
                    batch.mix_running_indices,
                )
            batch.input_ids = torch.cat([prefill_gpu, decode_gpu])
        else:
            batch.input_ids = prefill_gpu
        batch.prefill_input_ids_cpu = None
        batch.mix_running_indices = None
    elif batch.input_ids is None and future_map.spec_algo.is_none():
        batch.input_ids = future_map.output_tokens_buf[batch.req_pool_indices]
        if _DEBUG_ASSERT:
            _assert_nonneg_and_invalidate(
                batch.input_ids, future_map.output_tokens_buf, batch.req_pool_indices
            )


    if batch.enable_overlap and not batch.spec_algorithm.is_none():
        future_map._resolve_spec_extras(batch)


CONFIDENCE_RELAY_RING_LAG: int = 2
CONFIDENCE_RELAY_RING_DEPTH: int = CONFIDENCE_RELAY_RING_LAG + 1


class ResolvedConfidence(msgspec.Struct):

    confidence: torch.Tensor
    generation: torch.Tensor


@dataclass
class RelayPayload:


    bonus_tokens: Optional[torch.Tensor]
    topk_p: Optional[torch.Tensor] = None
    topk_index: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    draft_probs: Optional[torch.Tensor] = None
    dsa_topk_indices: Optional[torch.Tensor] = None

    accept_tokens: Optional[torch.Tensor] = None
    accept_lens: Optional[torch.Tensor] = None

    @classmethod
    def from_ngram(cls, draft_input: NgramVerifyInput) -> RelayPayload:
        return cls(
            bonus_tokens=None,
            accept_tokens=draft_input.accept_tokens.reshape(
                -1, draft_input.draft_token_num
            ),
            accept_lens=draft_input.accept_lens,
        )

    @classmethod
    def from_draft_input(cls, draft_input: EagleDraftInput) -> RelayPayload:
        return cls(
            bonus_tokens=draft_input.bonus_tokens,
            topk_p=draft_input.topk_p,
            topk_index=draft_input.topk_index,
            hidden_states=draft_input.hidden_states,
            draft_probs=getattr(draft_input, "draft_probs", None),
            dsa_topk_indices=draft_input.dsa_topk_indices,
        )

    @classmethod
    def from_request_forest(
        cls, draft_input: RequestForestDraftInput
    ) -> RelayPayload:


        return cls(bonus_tokens=draft_input.bonus_tokens)


class ConfidenceRelay(msgspec.Struct):

    device: torch.device
    req_pool_size: int
    pool: Any
    confidence_buf: Optional[torch.Tensor] = None
    conf_ring: Optional[torch.Tensor] = None
    gen_ring: Optional[torch.Tensor] = None
    copy_done: Optional[list] = None
    ring_pos: int = 0
    initialized: bool = False

    def _lazy_init(self, confidence: torch.Tensor) -> None:
        self.initialized = True
        gamma = confidence.shape[-1]
        self.confidence_buf = torch.empty(
            (self.req_pool_size, gamma), dtype=torch.float32, device=self.device
        )
        if _is_cuda:
            depth = CONFIDENCE_RELAY_RING_DEPTH
            self.conf_ring = torch.empty(
                (depth, self.req_pool_size, gamma),
                dtype=torch.float32,
                pin_memory=True,
            )
            self.gen_ring = torch.zeros((depth, self.req_pool_size), dtype=torch.int64)
            self.copy_done = [
                torch.get_device_module(self.device).Event() for _ in range(depth)
            ]

    def scatter(self, indices: torch.Tensor, confidence: torch.Tensor) -> None:
        if not self.initialized:
            self._lazy_init(confidence)
        self.confidence_buf[indices] = confidence.to(self.confidence_buf.dtype)

    def issue_ring_copy(self, *, stream, publish_ready) -> None:
        if not self.initialized or stream is None or publish_ready is None:
            return
        slot = self.ring_pos % CONFIDENCE_RELAY_RING_DEPTH
        stream.wait_event(publish_ready)
        with torch.get_device_module(self.device).stream(stream):
            self.conf_ring[slot].copy_(self.confidence_buf, non_blocking=True)
            self.copy_done[slot].record()
        self.gen_ring[slot].copy_(self.pool.req_generation)
        self.ring_pos += 1

    def resolve(
        self, batch: ScheduleBatch, *, stream, publish_ready
    ) -> Optional[ResolvedConfidence]:
        if not self.initialized:
            return None
        draft_input = batch.spec_info
        if draft_input is None:
            return None
        fi = draft_input.future_indices
        if fi is None or fi.shape[0] == 0:
            return None

        if stream is None or publish_ready is None:
            idx = batch.req_pool_indices
            idx_cpu = batch.req_pool_indices_cpu
            return ResolvedConfidence(
                confidence=self.confidence_buf[idx].cpu(),
                generation=self.pool.req_generation[idx_cpu].clone(),
            )

        if self.ring_pos < CONFIDENCE_RELAY_RING_LAG:
            return None
        slot = (self.ring_pos - CONFIDENCE_RELAY_RING_LAG) % CONFIDENCE_RELAY_RING_DEPTH
        if not self.copy_done[slot].query():
            return None

        idx_cpu = batch.req_pool_indices_cpu
        return ResolvedConfidence(
            confidence=self.conf_ring[slot][idx_cpu],
            generation=self.gen_ring[slot][idx_cpu],
        )


class FutureMap:


    def __init__(
        self,
        device: torch.device,
        spec_algo: SpeculativeAlgorithm,
        req_to_token_pool: ReqToTokenPool,
        needs_cpu_seq_lens: bool = True,
        needs_confidence_relay: bool = False,
    ):


        self.device = device
        self.spec_algo = spec_algo


        self.needs_cpu_seq_lens = needs_cpu_seq_lens
        self.needs_confidence_relay = needs_confidence_relay
        self.req_pool_size = req_to_token_pool.req_to_token.shape[0]

        if _DEBUG_ASSERT:

            self.output_tokens_buf = torch.full(
                (self.req_pool_size,), -1, dtype=torch.int64, device=self.device
            )
            self.new_seq_lens_buf = torch.full(
                (self.req_pool_size,), -1, dtype=torch.int64, device=self.device
            )
        else:
            self.output_tokens_buf = torch.empty(
                (self.req_pool_size,), dtype=torch.int64, device=self.device
            )
            self.new_seq_lens_buf = torch.empty(
                (self.req_pool_size,), dtype=torch.int64, device=self.device
            )


        if _is_cuda:
            self.new_seq_lens_cpu_pinned = torch.empty(
                (self.req_pool_size,), dtype=torch.int64, pin_memory=True
            )
            self.fwd_prepare_d2h_stream = torch.get_device_module(self.device).Stream()
        else:
            self.new_seq_lens_cpu_pinned = None
            self.fwd_prepare_d2h_stream = None

        self._forward_buf_initialized = False
        self.dsa_topk_indices_buf = None


        self.accept_tokens_buf: Optional[torch.Tensor] = None
        self.accept_lens_buf: Optional[torch.Tensor] = None

        self.publish_ready = None


        self._publish_fresh = False

        self.confidence_relay = ConfidenceRelay(
            device=self.device,
            req_pool_size=self.req_pool_size,
            pool=req_to_token_pool,
        )

    def _lazy_init_forward_buf(self, payload: RelayPayload):

        from sglang.srt.speculative.spec_utils import spec_need_hidden_states

        self._forward_buf_initialized = True


        self.need_topk = self.spec_algo.is_some() and self.spec_algo.need_topk()
        self.need_hidden_states = (
            self.spec_algo.is_some()
            and spec_need_hidden_states()
            and payload.hidden_states is not None
        )

        if self.need_topk:
            topk_p0 = payload.topk_p[0]
            topk_index0 = payload.topk_index[0]
            self.topk_p_buf = torch.empty(
                (self.req_pool_size, *topk_p0.shape),
                dtype=topk_p0.dtype,
                device=self.device,
            )
            self.topk_index_buf = torch.empty(
                (self.req_pool_size, *topk_index0.shape),
                dtype=topk_index0.dtype,
                device=self.device,
            )
        if self.need_hidden_states:
            hidden_states0 = payload.hidden_states[0]
            self.hidden_states_buf = torch.empty(
                (self.req_pool_size, *hidden_states0.shape),
                dtype=hidden_states0.dtype,
                device=self.device,
            )

        self.draft_probs_buf = None
        if payload.draft_probs is not None:
            draft_probs0 = payload.draft_probs[0]
            self.draft_probs_buf = torch.empty(
                (self.req_pool_size, *draft_probs0.shape),
                dtype=draft_probs0.dtype,
                device=self.device,
            )

    def _maybe_init_dsa_topk_indices_buf(self, payload: RelayPayload) -> None:
        if self.dsa_topk_indices_buf is not None or payload.dsa_topk_indices is None:
            return
        seed0 = payload.dsa_topk_indices[0]
        self.dsa_topk_indices_buf = torch.empty(
            (self.req_pool_size, *seed0.shape),
            dtype=payload.dsa_topk_indices.dtype,
            device=self.device,
        )

    def _maybe_init_ngram_bufs(self, payload: RelayPayload) -> None:
        if self.accept_tokens_buf is not None:
            return


        self.accept_tokens_buf = torch.zeros(
            (self.req_pool_size, payload.accept_tokens.shape[1]),
            dtype=payload.accept_tokens.dtype,
            device=self.device,
        )
        self.accept_lens_buf = torch.zeros(
            (self.req_pool_size,),
            dtype=payload.accept_lens.dtype,
            device=self.device,
        )

    def resolve_confidence_cpu(
        self, batch: ScheduleBatch
    ) -> Optional[ResolvedConfidence]:
        if not self.needs_confidence_relay:
            return None
        return self.confidence_relay.resolve(
            batch,
            stream=self.fwd_prepare_d2h_stream,
            publish_ready=self.publish_ready,
        )

    def _resolve_spec_extras(self, batch: ScheduleBatch) -> None:
        if self.spec_algo.is_ngram():
            draft_input = batch.spec_info
            if draft_input is None or draft_input.future_indices is None:
                return
            indices = draft_input.future_indices
            if indices.shape[0] == 0:
                return
            draft_input.accept_tokens = self.accept_tokens_buf[indices].flatten()
            draft_input.accept_lens = self.accept_lens_buf[indices]
            return
        draft_input: EagleDraftInput = batch.spec_info
        if draft_input is None:

            return
        indices = draft_input.future_indices
        if indices.shape[0] == 0:
            return


        indices.record_stream(torch.get_device_module(self.device).current_stream())
        if self.need_topk:
            hidden_states_buf = (
                self.hidden_states_buf if self.need_hidden_states else None
            )
            (
                draft_input.topk_p,
                draft_input.topk_index,
                bonus_tokens,
                hidden_states,
            ) = gather_spec_extras(
                indices,
                self.topk_p_buf,
                self.topk_index_buf,
                self.output_tokens_buf,
                hidden_states_buf,
            )
            draft_input.bonus_tokens = bonus_tokens
            if hidden_states is not None:
                draft_input.hidden_states = hidden_states
            if self.draft_probs_buf is not None and draft_input.draft_probs is not None:
                draft_input.draft_probs = self.draft_probs_buf[indices]
        else:
            draft_input.bonus_tokens = self.output_tokens_buf[indices]
        if self.need_hidden_states and not self.need_topk:
            draft_input.hidden_states = self.hidden_states_buf[indices]
        if draft_input.future_dsa_topk_indices_available:
            assert self.dsa_topk_indices_buf is not None
            draft_input.dsa_topk_indices = self.dsa_topk_indices_buf[indices]
        else:
            draft_input.dsa_topk_indices = None
        if _DEBUG_ASSERT:
            _assert_nonneg_and_invalidate(
                draft_input.bonus_tokens, self.output_tokens_buf, indices
            )

    def resolve_seq_lens_cpu(self, batch: ScheduleBatch) -> None:


        draft_input = batch.spec_info
        if draft_input is None:
            return

        fi = draft_input.future_indices
        if fi is None:
            return
        if self.publish_ready is not None:
            if _DEBUG_ASSERT:


                assert self._publish_fresh, "resolve without a fresh forward publish"
                self._publish_fresh = False
            if _is_hip:

                self.publish_ready.synchronize()
            else:
                self.publish_ready.wait()
        batch.seq_lens = self.new_seq_lens_buf[fi]

        if not self.needs_cpu_seq_lens:


            batch.seq_lens_cpu = None
            batch.seq_lens_sum = None
            if _DEBUG_ASSERT:


                _assert_nonneg_and_invalidate(batch.seq_lens, self.new_seq_lens_buf, fi)
            return

        if self.fwd_prepare_d2h_stream is None or self.publish_ready is None:
            batch.seq_lens_cpu = batch.seq_lens.cpu()
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
            if _DEBUG_ASSERT:
                _assert_nonneg_and_invalidate(batch.seq_lens, self.new_seq_lens_buf, fi)
            return


        self.fwd_prepare_d2h_stream.wait_event(self.publish_ready)
        with torch.get_device_module(self.device).stream(self.fwd_prepare_d2h_stream):
            self.new_seq_lens_cpu_pinned.copy_(self.new_seq_lens_buf, non_blocking=True)
        self.fwd_prepare_d2h_stream.synchronize()


        batch.seq_lens_cpu = self.new_seq_lens_cpu_pinned[batch.req_pool_indices_cpu]
        batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
        if _DEBUG_ASSERT:


            _assert_nonneg_and_invalidate(batch.seq_lens, self.new_seq_lens_buf, fi)

    def publish(
        self,
        future_indices: torch.Tensor,
        new_seq_lens: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
    ) -> None:
        indices = future_indices
        if indices.shape[0] == 0:
            return
        self.new_seq_lens_buf[indices] = new_seq_lens.to(self.new_seq_lens_buf.dtype)
        publish_confidence = self.needs_confidence_relay and confidence is not None
        if publish_confidence:
            self.confidence_relay.scatter(indices, confidence)

        if self.spec_algo.is_some():
            if self.publish_ready is None:
                self.publish_ready = torch.get_device_module(self.device).Event()
            self.publish_ready.record()
            self._publish_fresh = True
        if publish_confidence:
            self.confidence_relay.issue_ring_copy(
                stream=self.fwd_prepare_d2h_stream,
                publish_ready=self.publish_ready,
            )

    def stash(self, future_indices: torch.Tensor, payload: RelayPayload) -> None:
        indices = future_indices
        if indices.shape[0] == 0:

            return
        if self.spec_algo.is_ngram():
            self._maybe_init_ngram_bufs(payload)
            self.accept_tokens_buf[indices] = payload.accept_tokens
            self.accept_lens_buf[indices] = payload.accept_lens
            return
        if not self._forward_buf_initialized:
            self._lazy_init_forward_buf(payload)
        self._maybe_init_dsa_topk_indices_buf(payload)
        self.output_tokens_buf[indices] = payload.bonus_tokens.to(
            self.output_tokens_buf.dtype
        )

        if self.need_topk:
            self.topk_p_buf[indices] = payload.topk_p.to(self.topk_p_buf.dtype)
            self.topk_index_buf[indices] = payload.topk_index.to(
                self.topk_index_buf.dtype
            )
        if self.need_hidden_states:
            self.hidden_states_buf[indices] = payload.hidden_states.to(
                self.hidden_states_buf.dtype
            )
        if self.draft_probs_buf is not None and payload.draft_probs is not None:
            self.draft_probs_buf[indices] = payload.draft_probs
        if (
            self.dsa_topk_indices_buf is not None
            and payload.dsa_topk_indices is not None
        ):
            self.dsa_topk_indices_buf[indices] = payload.dsa_topk_indices.to(
                self.dsa_topk_indices_buf.dtype
            )
