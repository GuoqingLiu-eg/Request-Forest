from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InitLoadBackParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.utils.common import ceil_align, is_npu

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ReqKvInfo


logger = logging.getLogger(__name__)


class _VirtualNode:


    pass


@dataclass
class SessionSlot:


    virtual_node: _VirtualNode = field(default_factory=_VirtualNode)


    req_pool_idx: Optional[int] = None
    kv_committed_len: int = 0
    kv: Optional[ReqKvInfo] = None


    last_node: Any = None
    cache_protected_len: int = 0
    swa_uuid_for_lock: Optional[str] = None


    skip_lock_node_ids: dict = field(default_factory=dict)


    mamba_pool_idx: Any = None
    mamba_ping_pong_track_buffer: Any = None
    mamba_next_track_idx: Any = None
    mamba_last_track_idx: Any = None
    mamba_last_track_seqlen: Any = None
    mamba_branching_seqlen: Any = None

    @property
    def is_holding_kv(self) -> bool:

        return self.kv is not None

    def save_from_req(self, req: Req, is_first: bool):

        self.req_pool_idx = req.req_pool_idx
        self.kv_committed_len = req.kv_committed_len
        self.kv = copy.copy(req.kv)

        if is_first:
            self.last_node = req.last_node
            self.cache_protected_len = req.cache_protected_len
            self.swa_uuid_for_lock = req.swa_uuid_for_lock
            self.skip_lock_node_ids = req.skip_lock_node_ids

        self.mamba_pool_idx = req.mamba_pool_idx
        self.mamba_ping_pong_track_buffer = req.mamba_ping_pong_track_buffer
        self.mamba_next_track_idx = req.mamba_next_track_idx
        self.mamba_last_track_idx = req.mamba_last_track_idx
        self.mamba_last_track_seqlen = req.mamba_last_track_seqlen
        self.mamba_branching_seqlen = req.mamba_branching_seqlen


        req.req_pool_idx = None
        req.kv = None
        req.mamba_pool_idx = None
        req.mamba_ping_pong_track_buffer = None
        req.mamba_next_track_idx = None
        req.mamba_last_track_idx = None
        req.mamba_last_track_seqlen = None
        req.mamba_branching_seqlen = None

    def restore_to_req(self, req: Req):

        req.req_pool_idx = self.req_pool_idx
        req.kv_committed_len = self.kv_committed_len
        req.kv = copy.copy(self.kv)
        req.swa_uuid_for_lock = self.swa_uuid_for_lock
        req.skip_lock_node_ids = self.skip_lock_node_ids

        req.mamba_pool_idx = self.mamba_pool_idx
        req.mamba_ping_pong_track_buffer = self.mamba_ping_pong_track_buffer
        req.mamba_next_track_idx = self.mamba_next_track_idx
        req.mamba_last_track_idx = self.mamba_last_track_idx
        req.mamba_last_track_seqlen = self.mamba_last_track_seqlen
        req.mamba_branching_seqlen = self.mamba_branching_seqlen


def _is_streaming(req: Optional[Req]) -> bool:
    return req is not None and req.session is not None and req.session.streaming


class StreamingSession(BasePrefixCache):


    def __init__(self, inner: BasePrefixCache):
        self.inner = inner
        self.slots: Dict[str, SessionSlot] = {}


    @property
    def req_to_token_pool(self):
        return self.inner.req_to_token_pool

    @req_to_token_pool.setter
    def req_to_token_pool(self, value):
        self.inner.req_to_token_pool = value

    @property
    def token_to_kv_pool_allocator(self):
        return self.inner.token_to_kv_pool_allocator

    @token_to_kv_pool_allocator.setter
    def token_to_kv_pool_allocator(self, value):
        self.inner.token_to_kv_pool_allocator = value

    @property
    def page_size(self):
        return self.inner.page_size

    @page_size.setter
    def page_size(self, value):
        self.inner.page_size = value

    @property
    def disable(self):
        return self.inner.disable

    @disable.setter
    def disable(self, value):
        self.inner.disable = value

    @property
    def metrics_collector(self):
        return self.inner.metrics_collector

    @metrics_collector.setter
    def metrics_collector(self, value):
        self.inner.metrics_collector = value


    def has_slot(self, session_id: str) -> bool:
        return session_id in self.slots

    def any_holding_kv(self) -> bool:
        return any(s.is_holding_kv for s in self.slots.values())


    def try_inc_lock_ref(self, node: Any) -> Optional[IncLockRefResult]:

        if isinstance(node, _VirtualNode):
            return IncLockRefResult()
        return None

    def try_dec_lock_ref(
        self, node: Any, params: Optional[DecLockRefParams] = None
    ) -> Optional[DecLockRefResult]:
        if isinstance(node, _VirtualNode):
            return DecLockRefResult()
        return None

    def find_active_slot(self, req: Req) -> Optional[SessionSlot]:

        if not _is_streaming(req):
            return None
        slot = self.slots.get(req.session.session_id)
        if slot is None or slot.kv is None:
            return None
        if req.to_finish is not None:
            req.session.abort_req()
            req.session = None
            return None
        return slot


    def reset(self):
        self.slots.clear()
        self.inner.reset()


    def try_match_prefix(self, params: MatchPrefixParams) -> Optional[MatchResult]:

        slot = self.find_active_slot(params.req)
        if slot is None:
            return None

        req = params.req


        if is_npu() and self.page_size > 1:
            expected_prefix_len = min(slot.kv_committed_len, len(params.key))
            aligned_prefix_len = (
                expected_prefix_len // self.page_size
            ) * self.page_size
            if aligned_prefix_len < slot.cache_protected_len or aligned_prefix_len == 0:


                self.release_session(req.session.session_id)
                return None

        slot.restore_to_req(req)


        prefix_len = min(req.kv_committed_len, len(params.key))


        assert prefix_len >= slot.cache_protected_len, (
            f"streaming session prefix shrank: {prefix_len=} < "
            f"{slot.cache_protected_len=}"
        )


        if is_npu() and self.page_size > 1:
            prefix_len = (prefix_len // self.page_size) * self.page_size
            req.kv_committed_len = min(req.kv_committed_len, prefix_len)
            slot.kv_committed_len = min(slot.kv_committed_len, prefix_len)


        self._free_tail(slot, req, prefix_len)

        device_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :prefix_len
        ].to(dtype=torch.int64)

        return MatchResult(
            device_indices=device_indices,
            last_device_node=slot.virtual_node,
            last_host_node=slot.virtual_node,
            best_match_node=slot.virtual_node,
            cache_protected_len=slot.cache_protected_len,
        )

    def try_cache_finished_req(
        self, req: Req, is_insert: bool = True, **kwargs
    ) -> bool:

        if not _is_streaming(req):
            return False

        from sglang.srt.managers.schedule_batch import FINISH_ABORT

        session_id = req.session.session_id
        slot = self.slots.get(session_id)
        is_first = slot is None


        if isinstance(req.finished_reason, FINISH_ABORT):
            if slot is None:


                slot = SessionSlot(
                    req_pool_idx=req.req_pool_idx,
                    kv=copy.copy(req.kv),
                    last_node=req.last_node,
                    cache_protected_len=req.cache_protected_len,
                    swa_uuid_for_lock=req.swa_uuid_for_lock,
                    skip_lock_node_ids=req.skip_lock_node_ids,
                    mamba_pool_idx=req.mamba_pool_idx,
                    mamba_ping_pong_track_buffer=req.mamba_ping_pong_track_buffer,
                )
                self.slots[session_id] = slot


                req.mamba_pool_idx = None
                req.mamba_ping_pong_track_buffer = None
            slot.kv.kv_allocated_len = max(
                slot.kv.kv_allocated_len, req.kv.kv_allocated_len
            )
            self.release_session(session_id)
            req.req_pool_idx = None
            req.kv = None
            req.session.abort_req()
            return True

        if is_first:
            slot = SessionSlot()
            self.slots[session_id] = slot

        finished_len = (
            req.finished_len if req.finished_len is not None else len(req.output_ids)
        )
        target = len(req.origin_input_ids) + finished_len
        self._trim_overshoot(req, finished_len)

        slot.save_from_req(req, is_first=is_first)


        slot.kv_committed_len = min(target, slot.kv.kv_allocated_len)


        req.session.finish_req(req)

        return True

    def try_cache_unfinished_req(
        self, req: Req, chunked: bool = False, **kwargs
    ) -> bool:

        if not _is_streaming(req):
            return False
        if chunked:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : req.extend_range.end
            ]
            req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)
            return True
        if req.session.session_id in self.slots:
            return True
        return False


    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        result = self.try_match_prefix(params)
        if result is not None:
            return result
        return self.inner.match_prefix(params)

    def cache_finished_req(self, req: Req, is_insert: bool = True, **kwargs):
        if self.try_cache_finished_req(req, is_insert=is_insert, **kwargs):
            return
        self.inner.cache_finished_req(req, is_insert=is_insert, **kwargs)

    def cache_unfinished_req(self, req: Req, **kwargs):
        if self.try_cache_unfinished_req(req, **kwargs):
            return
        self.inner.cache_unfinished_req(req, **kwargs)

    def evict(self, params: EvictParams) -> EvictResult:
        return self.inner.evict(params)

    def evict_for_alloc(self, params: EvictParams) -> EvictResult:
        return self.inner.evict_for_alloc(params)

    def inc_lock_ref(self, node: Any) -> IncLockRefResult:
        result = self.try_inc_lock_ref(node)
        if result is not None:
            return result
        return self.inner.inc_lock_ref(node)

    def dec_lock_ref(
        self, node: Any, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        result = self.try_dec_lock_ref(node, params)
        if result is not None:
            return result
        return self.inner.dec_lock_ref(node, params)


    def release_session(self, session_id: str) -> None:
        slot = self.slots.pop(session_id, None)
        if slot is None:
            return
        protected_len = slot.cache_protected_len
        lock_node = slot.last_node
        tokens_freed = (
            max(0, slot.kv.kv_allocated_len - protected_len)
            if slot.is_holding_kv
            else 0
        )
        logger.info(
            "Session KV released: %s (%d tokens freed)", session_id, tokens_freed
        )

        if lock_node is not None:
            self.inner.dec_lock_ref(
                lock_node,
                DecLockRefParams(
                    swa_uuid_for_lock=slot.swa_uuid_for_lock,
                    skip_lock_node_ids=slot.skip_lock_node_ids,
                ),
            )

        if slot.is_holding_kv:
            start = protected_len
            end = slot.kv.kv_allocated_len
            if start < end:
                kv_indices = self.req_to_token_pool.req_to_token[
                    slot.req_pool_idx, start:end
                ]
                self.token_to_kv_pool_allocator.free(kv_indices)
            self.req_to_token_pool.free(slot)

        self._free_slot_mamba(slot)

    def release_radix_session(self, session_id: str) -> None:
        self.inner.release_radix_session(session_id)

    def session_held_tokens(self, active_pool_idxs: Optional[set] = None) -> int:

        total = 0
        for slot in self.slots.values():
            in_batch = (
                active_pool_idxs is not None and slot.req_pool_idx in active_pool_idxs
            )
            if slot.is_holding_kv and not in_batch:
                allocated = ceil_align(slot.kv.kv_allocated_len, self.page_size)
                total += allocated - slot.cache_protected_len
        return total

    def session_held_full_tokens(self, active_pool_idxs: Optional[set] = None) -> int:

        return self.session_held_tokens(active_pool_idxs)

    def session_held_swa_tokens(self, active_pool_idxs: Optional[set] = None) -> int:

        total = 0
        for slot in self.slots.values():
            in_batch = (
                active_pool_idxs is not None and slot.req_pool_idx in active_pool_idxs
            )
            if slot.is_holding_kv and not in_batch:
                allocated = ceil_align(slot.kv.kv_allocated_len, self.page_size)
                total += allocated - max(
                    slot.cache_protected_len, slot.kv.swa_evicted_seqlen
                )
        return total

    def session_held_req_count(self, active_pool_idxs: Optional[set] = None) -> int:


        def _owned(s):
            in_batch = (
                active_pool_idxs is not None and s.req_pool_idx in active_pool_idxs
            )
            return s.is_holding_kv and not in_batch

        return sum(_owned(s) for s in self.slots.values())

    def session_held_mamba_slots(self, active_pool_idxs: Optional[set] = None) -> int:

        total = 0
        for slot in self.slots.values():
            in_batch = (
                active_pool_idxs is not None and slot.req_pool_idx in active_pool_idxs
            )
            if in_batch:
                continue
            if slot.mamba_pool_idx is not None:
                total += slot.mamba_pool_idx.numel()
            if slot.mamba_ping_pong_track_buffer is not None:
                total += slot.mamba_ping_pong_track_buffer.numel()
        return total

    def _free_slot_mamba(self, slot: SessionSlot) -> None:

        mamba_allocator = getattr(self.req_to_token_pool, "mamba_allocator", None)
        if mamba_allocator is None:
            return
        if slot.mamba_pool_idx is not None:
            mamba_allocator.free(slot.mamba_pool_idx.unsqueeze(0))
            slot.mamba_pool_idx = None
        if slot.mamba_ping_pong_track_buffer is not None:
            mamba_allocator.free(slot.mamba_ping_pong_track_buffer)
            slot.mamba_ping_pong_track_buffer = None


    def _free_tail(self, slot: SessionSlot, req: Req, prefix_len: int) -> None:

        self._free_kv_aligned(slot.req_pool_idx, prefix_len, slot.kv.kv_allocated_len)
        slot.kv.kv_allocated_len = prefix_len
        slot.kv_committed_len = min(slot.kv_committed_len, prefix_len)
        slot.kv.swa_evicted_seqlen = min(slot.kv.swa_evicted_seqlen, prefix_len)
        req.kv.kv_allocated_len = prefix_len
        req.kv_committed_len = min(req.kv_committed_len, prefix_len)
        req.kv.swa_evicted_seqlen = min(req.kv.swa_evicted_seqlen, prefix_len)

    def _trim_overshoot(self, req: Req, finished_len: int) -> None:

        target = len(req.origin_input_ids) + finished_len
        self._free_kv_aligned(req.req_pool_idx, target, req.kv.kv_allocated_len)
        req.kv.kv_allocated_len = min(req.kv.kv_allocated_len, target)
        req.kv_committed_len = min(req.kv_committed_len, target)
        req.kv.swa_evicted_seqlen = min(req.kv.swa_evicted_seqlen, target)
        req.output_ids = req.output_ids[:finished_len]

    def _free_kv_aligned(self, pool_idx: int, target: int, end: int) -> None:

        if end <= target:
            return
        start = target
        if self.page_size > 1:
            start = ceil_align(start, self.page_size)
        if start < end:
            tail = self.req_to_token_pool.req_to_token[pool_idx, start:end]
            self.token_to_kv_pool_allocator.free(tail)


    def evictable_size(self):
        return self.inner.evictable_size()

    def full_evictable_size(self):
        return self.inner.full_evictable_size()

    def swa_evictable_size(self):
        return self.inner.swa_evictable_size()

    def protected_size(self):
        return self.inner.protected_size()

    def full_protected_size(self):
        return self.inner.full_protected_size()

    def swa_protected_size(self):
        return self.inner.swa_protected_size()

    def total_size(self):
        return self.inner.total_size()

    def pretty_print(self):
        return self.inner.pretty_print()

    def init_load_back(self, params: InitLoadBackParams):
        return self.inner.init_load_back(params)

    def ready_to_load_host_cache(self):
        return self.inner.ready_to_load_host_cache()

    def check_hicache_events(self):
        return self.inner.check_hicache_events()

    def take_events(self):
        return self.inner.take_events()

    def supports_swa(self):
        return self.inner.supports_swa()

    def supports_mamba(self):
        return self.inner.supports_mamba()

    def supports_streaming_session(self) -> bool:
        return True

    def is_chunk_cache(self):
        return self.inner.is_chunk_cache()

    def is_tree_cache(self):
        return self.inner.is_tree_cache()

    def available_and_evictable_str(self):
        return self.inner.available_and_evictable_str()

    def init_metrics_collector(self):
        return self.inner.init_metrics_collector()

    def sanity_check(self):


        if self.any_holding_kv():
            return
        self.inner.sanity_check()


    def __getattr__(self, name):
        return getattr(self.inner, name)
