from __future__ import annotations

import logging
from array import array
from collections import deque
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.managers.cache_controller import HICACHE_WRITE_STAGING_POOL_FRACTION
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    InitLoadBackParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.cache_action import RebuildFullToSWAMapping
from sglang.srt.mem_cache.unified_cache.components import (
    BASE_COMPONENT_TYPE,
    ComponentType,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core import (
    NodeId,
    UnifiedTreeNode,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.unified_cache.components import SWAComponent
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

logger = logging.getLogger(__name__)


class _UnifiedBackupIntent(msgspec.Struct):


    node: UnifiedTreeNode
    node_id: int
    hash_values: list[str]
    key: RadixKey
    prefix_keys: Optional[list[str]] = None


class _UnifiedBufferBackupEntry(msgspec.Struct):


    intent: _UnifiedBackupIntent
    host_indices: torch.Tensor
    aux_xfers: list[PoolTransfer]
    lock_params: DecLockRefParams


class _StagedPrefetch(msgspec.Struct):


    req_id: str
    key_tokens: list[int]
    extra_key: Optional[str]
    matched_len: int
    num_tokens: int
    occupied_tokens: int
    host_indices: torch.Tensor
    aux_xfers: list[PoolTransfer]
    hash_values: list[str]
    operation_id: int


class _OngoingBufferLoadBack(msgspec.Struct):


    req_id: str
    num_tokens: int
    occupied_tokens: int
    aux_xfers: list[PoolTransfer]
    host_indices: torch.Tensor
    hash_values: list[str]


class _AnchorLock(msgspec.Struct):


    node_id: NodeId
    lock_params: DecLockRefParams
    tokens: int


def _track_content_refs(refs: dict[str, int], hash_values: list[str]) -> None:

    for h in hash_values:
        refs[h] = refs.get(h, 0) + 1


def _untrack_content_refs(refs: dict[str, int], hash_values: list[str]) -> None:

    for h in hash_values:
        n = refs.get(h, 0) - 1
        if n <= 0:
            refs.pop(h, None)
        else:
            refs[h] = n


def validate_buffer_only_stack(
    sidecar_pool_specs: list, swa_component: Optional[SWAComponent]
) -> None:

    if sidecar_pool_specs:
        raise ValueError(
            "--hicache-host-memory-mode buffer_only does not support "
            "sidecar storage pools (DeepSeek-V4 compressed regions)."
        )
    swa = swa_component
    if swa is not None and swa._swa_kv_pool_host is None:


        raise ValueError(
            "--hicache-host-memory-mode buffer_only on SWA models "
            "requires an SWA host staging pool; the unified_kv layout "
            "keeps SWA as a device-only ring."
        )
    if swa is not None and swa._swa_kv_pool_host is not None:


        window_tokens = swa.full_window_pages * swa._swa_kv_pool_host.page_size
        if swa._swa_kv_pool_host.size < 2 * window_tokens:
            raise ValueError(
                "--hicache-host-memory-mode buffer_only requires an SWA "
                f"host pool of at least two trailing windows "
                f"({2 * window_tokens} tokens; got "
                f"{swa._swa_kv_pool_host.size}): one staging a write "
                "while one stays reserved for prefetch window allocs."
            )


class BufferModePipeline:


    def __init__(
        self,
        cache: UnifiedRadixCache,
        swa_window_pages: int,
        write_backlog_cap: int,
    ):
        self._cache = cache


        self._swa_window_pages = swa_window_pages


        self.write_backlog_cap = write_backlog_cap

        self.anchor_lock_enabled = envs.SGLANG_ENABLE_HICACHE_BUFFER_ANCHOR_LOCK.get()
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        kvcache = cache.token_to_kv_pool_allocator.get_kvcache()
        full_pool = kvcache.full_kv_pool if isinstance(kvcache, SWAKVPool) else kvcache
        self.anchor_lock_cap_tokens = int(
            envs.SGLANG_HICACHE_BUFFER_ANCHOR_LOCK_CAP.get() * full_pool.size
        )
        self.reset()

    def reset(self) -> None:


        self.pending_hit_allocs: deque = deque()
        self._prefetch_prefix_ctx: dict[str, list[int]] = {}
        self.staged_prefetches: dict[str, _StagedPrefetch] = {}
        self.ongoing_buffer_load_back: dict[int, _OngoingBufferLoadBack] = {}


        self.pending_write_queue: deque[_UnifiedBackupIntent] = deque()
        self.inflight_backup_node_ids: set[int] = set()
        self.inflight_backup_hashes: dict[str, int] = {}


        self.ongoing_write_through: dict[int, _UnifiedBufferBackupEntry] = {}
        self.ongoing_backup: dict[int, _UnifiedBufferBackupEntry] = {}
        self.write_staged_tokens_ = 0
        self.write_backlog_tokens_ = 0
        self._backlog_cap_hits = 0

        self.anchor_locks: dict[str, _AnchorLock] = {}
        self.anchor_locked_tokens_ = 0
        self._anchor_lock_cap_skips = 0

    def is_idle(self) -> bool:

        return not (
            self.pending_write_queue or self.staged_prefetches or self.ongoing_backup
        )


    def _backup_parent_covered(self, node: UnifiedTreeNode) -> bool:

        parent = node.parent
        if (
            parent is self._cache.root_node
            or parent.id in self.inflight_backup_node_ids
        ):
            return True
        last_hash = parent.get_last_hash_value()
        return last_hash is not None and self._cache.storage_existence_cache.contains(
            PoolName.KV, last_hash
        )

    def _log_backup_dropped(self, num_tokens: int) -> None:
        cache = self._cache
        if cache.enable_storage_metrics and cache.storage_metrics_collector is not None:
            cache.storage_metrics_collector.log_backup_dropped_tokens(num_tokens)

    def enqueue_backup_intent(self, node: UnifiedTreeNode) -> None:

        if not self._cache.enable_storage or not node.hash_value:
            return
        if node.id in self.inflight_backup_node_ids:
            return


        if self._cache.storage_existence_cache.covers_all(
            PoolName.KV, node.hash_value, extra_cover=self.inflight_backup_hashes
        ):
            return
        intent_tokens = len(node.hash_value) * self._cache.page_size
        if self.write_backlog_tokens_ >= self.write_backlog_cap:


            self._backlog_cap_hits += 1
            if self._backlog_cap_hits <= 3 or self._backlog_cap_hits % 1000 == 0:
                logger.error(
                    "HiCache write backlog cap hit (occurrence %d): "
                    "backlog=%d cap=%d queue=%d. Live backlog is bounded "
                    "by the device pool span, so this indicates a "
                    "stale-sweep or accounting leak.",
                    self._backlog_cap_hits,
                    self.write_backlog_tokens_,
                    self.write_backlog_cap,
                    len(self.pending_write_queue),
                )
            self._log_backup_dropped(intent_tokens)
            return


        if not self._backup_parent_covered(node) or self._backup_oversize(
            node, intent_tokens
        ):
            self._log_backup_dropped(intent_tokens)
            return

        prefix_keys = (
            node.get_prefix_hash_values(node.parent)
            if self._cache.hicache_storage_pass_prefix_keys
            else None
        )
        intent = _UnifiedBackupIntent(
            node=node,
            node_id=node.id,
            hash_values=list(node.hash_value),
            key=node.key,
            prefix_keys=prefix_keys,
        )
        self.pending_write_queue.append(intent)
        self.inflight_backup_node_ids.add(node.id)
        self.write_backlog_tokens_ += intent_tokens

    def _build_aux_staging_transfers(
        self, node: UnifiedTreeNode
    ) -> Optional[list[PoolTransfer]]:

        transfers: list[PoolTransfer] = []
        if ComponentType.SWA in self._cache.components:
            cd = node.component_data[ComponentType.SWA]
            if cd.value is not None:
                num_pages = len(cd.value) // self._cache.page_size
                if num_pages > 0:
                    transfers.append(
                        PoolTransfer(
                            name=PoolName.SWA,
                            keys=node.hash_value[-num_pages:],
                            hit_policy=PoolHitPolicy.TRAILING_PAGES,
                        )
                    )
        return transfers or None

    def _backup_oversize(
        self,
        node: UnifiedTreeNode,
        intent_tokens: int,
        aux_xfers: Optional[list[PoolTransfer]] = None,
    ) -> bool:

        cc = self._cache.cache_controller
        if intent_tokens > cc.mem_pool_host.size:
            return True
        if aux_xfers is None:
            aux_xfers = self._build_aux_staging_transfers(node)
        for t in aux_xfers or ():
            entry = cc.mem_pool_host.entry_map.get(t.name)
            if entry is not None and (
                len(t.keys) * entry.host_pool.page_size
                > entry.host_pool.size - self._aux_loads_margin(entry.host_pool)
            ):
                return True
        return False

    def _aux_loads_margin(self, host_pool) -> int:

        return max(
            self._swa_window_pages * host_pool.page_size,
            host_pool.size // 10,
        )

    def _backup_intent_stale(self, intent: _UnifiedBackupIntent) -> bool:


        node = intent.node
        try:
            self._cache.tree_core.node_by_id(intent.node_id)
        except KeyError:
            return True
        return (
            node.component_data[BASE_COMPONENT_TYPE].value is None
            or node.hash_value != intent.hash_values
        )

    def _sweep_stale_backup_intents(self) -> None:

        if not self.pending_write_queue:
            return
        page_size = self._cache.page_size
        survivors: deque[_UnifiedBackupIntent] = deque()
        for intent in self.pending_write_queue:
            if self._backup_intent_stale(intent):
                self.inflight_backup_node_ids.discard(intent.node_id)
                self.write_backlog_tokens_ -= len(intent.hash_values) * page_size
                continue
            survivors.append(intent)
        self.pending_write_queue = survivors

    def flush_pending_writes(self) -> None:

        if not self.pending_write_queue:
            return
        cc = self._cache.cache_controller
        self._sweep_stale_backup_intents()


        pool_tokens = cc.mem_pool_host.size
        live_cap = max(
            int(HICACHE_WRITE_STAGING_POOL_FRACTION * pool_tokens),
            pool_tokens - cc.prefetch_tokens_occupied - pool_tokens // 10,
        )
        while self.pending_write_queue:
            intent = self.pending_write_queue[0]
            intent_tokens = len(intent.hash_values) * self._cache.page_size
            if not self._backup_parent_covered(intent.node) or self._backup_oversize(
                intent.node, intent_tokens
            ):


                self.pending_write_queue.popleft()
                self.inflight_backup_node_ids.discard(intent.node_id)
                self.write_backlog_tokens_ -= intent_tokens
                self._log_backup_dropped(intent_tokens)
                continue
            if self.write_staged_tokens_ >= live_cap:

                break
            if self._aux_budget_blocked(intent):


                break
            if not self._launch_backup_intent(intent):


                break
            self.pending_write_queue.popleft()

    def _launch_backup_intent(self, intent: _UnifiedBackupIntent) -> bool:

        cache = self._cache
        cc = cache.cache_controller
        node = intent.node


        device_value, comp_xfers = cache.tree_core.build_backup_spec(node.id)
        aux_xfers = [x for xfers in comp_xfers.values() for x in xfers]
        host_indices = cc.write(
            device_value,
            node_id=node.id,
            extra_pools=aux_xfers or None,
        )
        if host_indices is None:
            return False
        _track_content_refs(self.inflight_backup_hashes, intent.hash_values)


        lock_params = cache.inc_lock_ref(node.id).to_dec_params()
        self.ongoing_write_through[node.id] = _UnifiedBufferBackupEntry(
            intent=intent,
            host_indices=host_indices,
            aux_xfers=aux_xfers,
            lock_params=lock_params,
        )
        self.write_staged_tokens_ += len(host_indices)
        self.write_backlog_tokens_ -= len(intent.hash_values) * cache.page_size
        return True

    def _aux_budget_blocked(self, intent: _UnifiedBackupIntent) -> bool:

        aux = self._build_aux_staging_transfers(intent.node)
        if not aux:
            return False
        cc = self._cache.cache_controller
        for t in aux:
            entry = cc.mem_pool_host.entry_map.get(t.name)
            if entry is None:
                continue
            need = len(t.keys) * entry.host_pool.page_size
            headroom = entry.host_pool.available_size() - self._aux_loads_margin(
                entry.host_pool
            )
            if need > headroom:
                return True
        return False

    def _aux_window_keys(
        self, hash_values: list[str], transfer: PoolTransfer
    ) -> Optional[list[str]]:

        if transfer.host_indices is None or transfer.host_indices.numel() == 0:
            return None
        if transfer.indices_from_pool is not None:
            return None
        entry = self._cache.cache_controller.mem_pool_host.entry_map.get(transfer.name)
        if entry is None:
            return None
        num_keys = len(transfer.host_indices) // entry.host_pool.page_size
        if num_keys == 0 or num_keys > len(hash_values):
            return None
        return hash_values[-num_keys:]

    def finish_backup_ack(self, ack_id: int) -> None:

        entry = self.ongoing_write_through.pop(ack_id)
        intent = entry.intent
        self._cache.dec_lock_ref(intent.node_id, entry.lock_params)


        storage_xfers: list[PoolTransfer] = []
        for staged in entry.aux_xfers:
            keys = self._aux_window_keys(intent.hash_values, staged)
            if keys is None:
                continue
            storage_xfers.append(
                PoolTransfer(
                    name=staged.name,
                    host_indices=staged.host_indices,
                    keys=keys,
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
            )
        operation_id = self._cache.cache_controller.write_storage(
            entry.host_indices,
            intent.key.token_ids,
            intent.hash_values,
            intent.prefix_keys,
            extra_pools=storage_xfers or None,
        )
        self.ongoing_backup[operation_id] = entry

    def finish_storage_write_ack(self, operation_id: int) -> None:

        entry = self.ongoing_backup.pop(operation_id, None)
        if entry is None:
            return
        intent = entry.intent
        self._cache.storage_existence_cache.add(PoolName.KV, intent.hash_values)
        self._free_staging_now(entry.host_indices, entry.aux_xfers)
        self.write_staged_tokens_ -= len(entry.host_indices)
        self.inflight_backup_node_ids.discard(entry.intent.node_id)
        _untrack_content_refs(self.inflight_backup_hashes, intent.hash_values)

    def _free_staging_now(
        self, host_indices: torch.Tensor, aux_xfers: list[PoolTransfer]
    ) -> None:

        cc = self._cache.cache_controller
        if host_indices is not None and host_indices.numel() > 0:
            cc.mem_pool_host.free(host_indices)
        for t in aux_xfers or ():
            if (
                t.host_indices is None
                or t.host_indices.numel() == 0
                or t.indices_from_pool is not None
            ):
                continue
            entry = cc.mem_pool_host.entry_map.get(t.name)
            if entry is not None:
                entry.host_pool.free(t.host_indices)


    def try_lock_anchor(self, req_id: str, anchor_node_id: NodeId) -> None:

        if not self.anchor_lock_enabled or req_id in self.anchor_locks:
            return
        prefix_tokens = self._prefetch_prefix_ctx.get(req_id)
        if not prefix_tokens:
            return
        matched_len = len(prefix_tokens)
        if self.anchor_locked_tokens_ + matched_len > self.anchor_lock_cap_tokens:
            self._anchor_lock_cap_skips += 1
            if (
                self._anchor_lock_cap_skips <= 3
                or self._anchor_lock_cap_skips % 1000 == 0
            ):
                logger.warning(
                    "HiCache anchor-lock cap reached (skip %d): locked=%d "
                    "want=%d cap=%d; launching unlocked.",
                    self._anchor_lock_cap_skips,
                    self.anchor_locked_tokens_,
                    matched_len,
                    self.anchor_lock_cap_tokens,
                )
            return
        cache = self._cache
        try:
            node = cache.tree_core.node_by_id(anchor_node_id)
        except KeyError:
            return
        if node.component_data[BASE_COMPONENT_TYPE].value is None:

            logger.warning("HiCache anchor evicted before IO commit req=%s", req_id)
            return
        lock_params = cache.inc_lock_ref(anchor_node_id).to_dec_params()
        self.anchor_locks[req_id] = _AnchorLock(
            node_id=anchor_node_id,
            lock_params=lock_params,
            tokens=matched_len,
        )
        self.anchor_locked_tokens_ += matched_len

    def release_anchor_lock(self, req_id: str) -> None:

        lock = self.anchor_locks.pop(req_id, None)
        if lock is None:
            return
        self._cache.dec_lock_ref(lock.node_id, lock.lock_params)
        self.anchor_locked_tokens_ -= lock.tokens
        assert self.anchor_locked_tokens_ >= 0, (
            f"anchor-lock accounting corrupted: locked={self.anchor_locked_tokens_} "
            f"after releasing {req_id}"
        )

    def set_prefix_ctx(self, req_id: str, matched_prefix_tokens) -> None:

        self._prefetch_prefix_ctx[req_id] = list(matched_prefix_tokens or [])

    def pop_prefix_ctx(self, req_id: str) -> None:
        self._prefetch_prefix_ctx.pop(req_id, None)

    def has_staged(self, req_id: str) -> bool:
        return req_id in self.staged_prefetches

    @staticmethod
    def _occupied_span(host_indices) -> int:

        return len(host_indices) if host_indices is not None else 0

    def stage_completed_prefetch(
        self,
        req_id: str,
        num_tokens: int,
        hash_value: list[str],
    ) -> bool:

        cache = self._cache
        (
            _anchor,
            prefetch_key,
            host_indices,
            operation,
            _lock_params,
            comp_xfers,
        ) = cache.ongoing_prefetch.pop(req_id)
        cc = cache.cache_controller
        prefix_tokens = self._prefetch_prefix_ctx.pop(req_id, None)
        aux_xfers = [x for xfers in comp_xfers.values() for x in xfers]

        if num_tokens == 0 or prefix_tokens is None:

            self.release_anchor_lock(req_id)
            cc.append_host_mem_release(
                host_indices[:num_tokens], extra_pools=aux_xfers or None
            )
            cc.prefetch_tokens_occupied -= self._occupied_span(host_indices)
            cache.prefetch_loaded_tokens_by_reqid[req_id] = 0
            return True

        staged_pages = num_tokens // cache.page_size
        staged_hashes = hash_value[:staged_pages]
        staged_kv = host_indices[:num_tokens]


        cache.storage_existence_cache.add(PoolName.KV, list(staged_hashes))
        occupied_tokens = self._occupied_span(host_indices)

        self.staged_prefetches[req_id] = _StagedPrefetch(
            req_id=req_id,
            key_tokens=prefix_tokens + list(prefetch_key[:num_tokens].token_ids),
            extra_key=prefetch_key.extra_key,
            matched_len=len(prefix_tokens),
            num_tokens=num_tokens,
            occupied_tokens=occupied_tokens,
            host_indices=staged_kv,
            aux_xfers=aux_xfers,
            hash_values=staged_hashes,
            operation_id=operation.id,
        )
        cache.prefetch_loaded_tokens_by_reqid[req_id] = num_tokens
        return True

    def staged_prefetch_tokens(self, req_id: str) -> int:

        f = self.staged_prefetches.get(req_id)
        return f.num_tokens if f is not None else 0

    def staged_prefetch_swa_tokens(self, req_id: str) -> int:

        f = self.staged_prefetches.get(req_id)
        if f is None:
            return 0
        return sum(
            len(t.host_indices)
            for t in f.aux_xfers
            if t.name == PoolName.SWA and t.host_indices is not None
        )

    def init_load_back(self, params: InitLoadBackParams) -> tuple[torch.Tensor, NodeId]:

        cache = self._cache
        req = params.req
        assert req is not None
        empty = cache.tree_core.empty_match_result.device_indices
        unchanged = (empty, req.last_node)
        f = self.staged_prefetches.pop(req.rid, None)
        if f is None:
            self.release_anchor_lock(req.rid)
            return unchanged
        cc = cache.cache_controller

        def _drop() -> tuple[torch.Tensor, NodeId]:
            self.release_anchor_lock(req.rid)
            self._free_staging_now(f.host_indices, f.aux_xfers)
            cc.prefetch_tokens_occupied -= f.occupied_tokens
            return unchanged


        if len(req.prefix_indices) != f.matched_len:
            logger.warning(
                "HiCache staged prefetch dropped req=%s reason=%s matched=%d "
                "now=%d tokens_wasted=%d locked=%s",
                req.rid,
                "growth" if len(req.prefix_indices) > f.matched_len else "shrink",
                f.matched_len,
                len(req.prefix_indices),
                f.num_tokens,
                req.rid in self.anchor_locks,
            )
            return _drop()

        key = RadixKey(
            array("q", f.key_tokens),
            extra_key=f.extra_key,
            is_bigram=cache.tree_core.is_eagle,
        ).page_aligned(cache.page_size)
        span_end = f.matched_len + f.num_tokens


        live = cache.match_prefix(MatchPrefixParams(key=key))
        if (
            len(live.device_indices) != f.matched_len
            or live.full_kv_hit_length != f.matched_len
        ):
            logger.warning(
                "HiCache staged prefetch dropped req=%s reason=overlap "
                "matched=%d live_unified=%d live_full=%d tokens_wasted=%d "
                "locked=%s",
                req.rid,
                f.matched_len,
                len(live.device_indices),
                live.full_kv_hit_length,
                f.num_tokens,
                req.rid in self.anchor_locks,
            )
            return _drop()


        if cache.supports_swa():
            avail = cache.token_to_kv_pool_allocator.full_available_size()
        else:
            avail = cache.token_to_kv_pool_allocator.available_size()
        if avail < f.num_tokens:
            needed = f.num_tokens - avail
            cache.evict_for_alloc(EvictParams(num_tokens=needed))
            if cache.supports_swa():
                avail = cache.token_to_kv_pool_allocator.full_available_size()
            else:
                avail = cache.token_to_kv_pool_allocator.available_size()
            if avail < f.num_tokens:

                return _drop()

        load_back_id = -(f.operation_id) - 1
        device_indices = cc.load(
            host_indices=f.host_indices,
            node_id=load_back_id,
            extra_pools=f.aux_xfers or None,
        )
        if device_indices is None:


            return _drop()

        swa_dev = next(
            (
                t.device_indices
                for t in f.aux_xfers
                if t.name == PoolName.SWA
                and t.device_indices is not None
                and t.device_indices.numel() > 0
            ),
            None,
        )
        if swa_dev is not None:


            cache._apply_cache_action(
                RebuildFullToSWAMapping([device_indices[-len(swa_dev) :]], [swa_dev])
            )


        insert_result = cache.insert(
            InsertParams(
                key=key,
                value=torch.cat([req.prefix_indices, device_indices]),
                prev_prefix_len=f.matched_len,
                swa_evicted_seqlen=(
                    max(0, span_end - len(swa_dev)) if swa_dev is not None else 0
                ),
            )
        )
        self.ongoing_buffer_load_back[load_back_id] = _OngoingBufferLoadBack(
            req_id=f.req_id,
            num_tokens=f.num_tokens,
            occupied_tokens=f.occupied_tokens,
            aux_xfers=f.aux_xfers,
            host_indices=f.host_indices,
            hash_values=f.hash_values,
        )
        m = cache.match_prefix(MatchPrefixParams(key=key))
        self.release_anchor_lock(req.rid)
        canonical = m.device_indices[f.matched_len : span_end]
        if len(m.device_indices) < span_end or not torch.equal(
            canonical, device_indices
        ):


            raise RuntimeError(
                f"HiCache buffer load-back ownership violation req={f.req_id}: "
                f"insert prefix_len={insert_result.prefix_len} "
                f"expected={f.matched_len}, adopted={len(m.device_indices)} "
                f"span_end={span_end}, canonical_matches_incoming="
                f"{len(m.device_indices) >= span_end and torch.equal(canonical, device_indices)}; "
                f"in-flight H2D targets freed slots"
            )


        return canonical, m.last_device_node

    def try_finish_load_back(self, ack_id: int) -> bool:

        f = self.ongoing_buffer_load_back.pop(ack_id, None)
        if f is None:
            return False
        cache = self._cache
        cc = cache.cache_controller


        self._free_staging_now(f.host_indices, f.aux_xfers)

        cc.prefetch_tokens_occupied -= f.occupied_tokens
        logger.info(
            "HiCache prefetch fill committed req=%s filled=%d occupied=%d locked=%d",
            f.req_id,
            f.num_tokens,
            cc.prefetch_tokens_occupied,
            self.anchor_locked_tokens_,
        )
        if cache.enable_storage_metrics and cache.storage_metrics_collector is not None:
            cache.storage_metrics_collector.log_prefetched_tokens(f.num_tokens)
        return True

    def release_aborted_staged(self, rid: str) -> bool:

        self.release_anchor_lock(rid)
        staged = self.staged_prefetches.pop(rid, None)
        if staged is None:
            return False
        self._free_staging_now(staged.host_indices, staged.aux_xfers)
        self._cache.cache_controller.prefetch_tokens_occupied -= staged.occupied_tokens
        return True
