from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Optional, Sequence

import torch

from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)


class _PageRef:


    __slots__ = ("slots", "first_slot", "refs", "retain")

    def __init__(
        self, slots: torch.Tensor, first_slot: int, refs: int = 1, retain: bool = True
    ):
        self.slots = slots
        self.first_slot = first_slot
        self.refs = refs
        self.retain = retain


class BufferPageCache:
    def __init__(self) -> None:


        self._entries: dict[tuple[str, str], _PageRef] = {}

        self._zero_ref: dict[str, OrderedDict[str, None]] = {}

        self._held_tokens: dict[str, int] = {}

        self._zero_ref_tokens: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def num_zero_ref_pages(self) -> int:
        return sum(len(lru) for lru in self._zero_ref.values())

    def held_tokens(self, pool: str) -> int:
        return self._held_tokens.get(pool, 0)

    def zero_ref_tokens(self, pool: str) -> int:

        return self._zero_ref_tokens.get(pool, 0)

    def register(
        self,
        pool: str,
        hashes: Sequence[str],
        host_indices: torch.Tensor,
        page_size: int,
        retain: bool = True,
    ) -> int:

        assert len(host_indices) == len(hashes) * page_size
        registered = 0
        entries = self._entries

        first_slots = host_indices[::page_size].tolist()
        for i, page_hash in enumerate(hashes):
            key = (pool, page_hash)
            existing = entries.get(key)
            if existing is not None:
                existing.retain = existing.retain or retain
                continue
            entries[key] = _PageRef(
                host_indices[i * page_size : (i + 1) * page_size],
                first_slots[i],
                retain=retain,
            )
            registered += 1
        if registered:
            self._held_tokens[pool] = (
                self._held_tokens.get(pool, 0) + registered * page_size
            )
        return registered

    def contains(self, pool: str, page_hash: str) -> bool:

        return (pool, page_hash) in self._entries

    def peek_run_len(self, pool: str, hashes: Sequence[str]) -> int:

        entries = self._entries
        run = 0
        for page_hash in hashes:
            if (pool, page_hash) not in entries:
                break
            run += 1
        return run

    def acquire(self, pool: str, hashes: Sequence[str]) -> Optional[torch.Tensor]:

        entries = self._entries
        refs = []
        for page_hash in hashes:
            entry = entries.get((pool, page_hash))
            if entry is None:
                return None
            refs.append(entry)
        zero_ref = self._zero_ref.get(pool)
        for page_hash, entry in zip(hashes, refs):
            if entry.refs == 0 and zero_ref is not None:
                if zero_ref.pop(page_hash, None) is not None:
                    self._zero_ref_tokens[pool] -= len(entry.slots)
            entry.refs += 1

            entry.retain = True
        return torch.cat([entry.slots for entry in refs])

    def release(
        self,
        pool: str,
        hashes: Sequence[str],
        host_indices: torch.Tensor,
        page_size: int,
    ) -> Optional[torch.Tensor]:

        assert len(host_indices) == len(hashes) * page_size
        leftover: list[torch.Tensor] = []
        entries = self._entries

        first_slots = host_indices[::page_size].tolist()
        for i, page_hash in enumerate(hashes):
            entry = entries.get((pool, page_hash))
            if entry is None or entry.first_slot != first_slots[i]:
                leftover.append(host_indices[i * page_size : (i + 1) * page_size])
                continue
            assert entry.refs > 0, "release without a matching acquire/register"
            entry.refs -= 1
            if entry.refs == 0:
                if entry.retain:
                    self._zero_ref.setdefault(pool, OrderedDict())[page_hash] = None
                    self._zero_ref_tokens[pool] = self._zero_ref_tokens.get(
                        pool, 0
                    ) + len(entry.slots)
                else:
                    del entries[(pool, page_hash)]
                    self._held_tokens[pool] -= len(entry.slots)
                    leftover.append(entry.slots)
        if not leftover:
            return None
        return torch.cat(leftover)

    def reclaim(
        self,
        pool: str,
        need_tokens: int,
        free: Callable[[torch.Tensor], int],
    ) -> int:

        zero_ref = self._zero_ref.get(pool)
        if not zero_ref or need_tokens <= 0:
            return 0
        freed = 0
        batch: list[torch.Tensor] = []
        while zero_ref and freed < need_tokens:
            page_hash, _ = zero_ref.popitem(last=False)
            entry = self._entries.pop((pool, page_hash))
            batch.append(entry.slots)
            freed += len(entry.slots)
        if batch:
            free(torch.cat(batch))
            self._held_tokens[pool] -= freed
            self._zero_ref_tokens[pool] -= freed
        return freed


class BufferPageCacheOps:


    def __init__(
        self,
        page_cache: BufferPageCache,
        mem_pool_host,
        sw_window_pages_fn: Callable[[], int],
    ):

        self.page_cache = page_cache
        self._mem_pool_host = mem_pool_host


        self._sw_window_pages_fn = sw_window_pages_fn

    def aux_window_keys(
        self, hash_values: list[str], transfer: PoolTransfer
    ) -> Optional[list[str]]:

        if transfer.host_indices is None or transfer.host_indices.numel() == 0:
            return None
        if transfer.indices_from_pool is not None:
            return None
        entry = self._mem_pool_host.entry_map.get(transfer.name)
        if entry is None:
            return None
        pool_page_size = entry.host_pool.page_size
        num_keys = len(transfer.host_indices) // pool_page_size
        if num_keys == 0 or num_keys > len(hash_values):
            return None
        return hash_values[-num_keys:]

    def register_span(
        self,
        pool: PoolName,
        hashes: list[str],
        host_indices: torch.Tensor,
        retain: bool = True,
    ) -> None:

        if not hashes:
            return
        entry = self._mem_pool_host.entry_map.get(pool)
        if entry is None:
            return
        self.page_cache.register(
            pool, hashes, host_indices, entry.host_pool.page_size, retain=retain
        )

    def release_span(
        self,
        pool: PoolName,
        hashes: list[str],
        host_indices: torch.Tensor,
    ) -> None:

        if host_indices is None or host_indices.numel() == 0:
            return
        entry = self._mem_pool_host.entry_map.get(pool)
        if entry is None:
            return
        if not hashes:
            entry.host_pool.free(host_indices)
            return
        leftover = self.page_cache.release(
            pool, hashes, host_indices, entry.host_pool.page_size
        )
        if leftover is not None and leftover.numel() > 0:
            entry.host_pool.free(leftover)

    def register_hold(
        self,
        hash_values: list[str],
        host_indices: torch.Tensor,
        aux_xfers: list[PoolTransfer],
        retain: bool = True,
    ) -> None:

        self.register_span(PoolName.KV, hash_values, host_indices, retain=retain)
        for transfer in aux_xfers:
            keys = self.aux_window_keys(hash_values, transfer)
            if keys is not None:
                self.register_span(
                    transfer.name, keys, transfer.host_indices, retain=retain
                )

    def release_hold(
        self,
        hash_values: list[str],
        host_indices: torch.Tensor,
        aux_xfers: list[PoolTransfer],
    ) -> None:

        self.release_span(PoolName.KV, hash_values, host_indices)
        for transfer in aux_xfers:
            if transfer.indices_from_pool is not None:
                continue
            keys = self.aux_window_keys(hash_values, transfer)
            self.release_span(transfer.name, keys or [], transfer.host_indices)

    def reclaim(self, pool: PoolName, num_tokens: int) -> int:

        entry = self._mem_pool_host.entry_map.get(pool)
        if entry is None:
            return 0
        shortfall = num_tokens - entry.host_pool.available_size()
        if shortfall <= 0:
            return 0
        return self.page_cache.reclaim(pool, shortfall, entry.host_pool.free)

    def continuation_run(self, chain: list[str], start_pages: int) -> int:

        page_cache = self.page_cache
        kv_run = page_cache.peek_run_len(PoolName.KV, chain[start_pages:])
        if kv_run == 0:
            return 0
        sw_pages = self._sw_window_pages_fn()
        if sw_pages == 0:
            return kv_run
        for cont in range(kv_run, 0, -1):
            joint = start_pages + cont
            window = min(sw_pages, joint)
            if cont < window:


                break
            if all(
                page_cache.contains(PoolName.SWA, chain[i])
                for i in range(joint - window, joint)
            ):
                return cont
        return 0

    def acquire_span(
        self, chain: list[str], start_pages: int, cont_pages: int
    ) -> Optional[tuple[torch.Tensor, list[PoolTransfer]]]:

        page_cache = self.page_cache
        cont_hashes = list(chain[start_pages : start_pages + cont_pages])
        kv_slots = page_cache.acquire(PoolName.KV, cont_hashes)
        if kv_slots is None:
            return None
        aux_xfers: list[PoolTransfer] = []
        sw_pages = self._sw_window_pages_fn()
        if sw_pages > 0:
            joint = start_pages + cont_pages
            window_hashes = list(chain[joint - min(sw_pages, joint) : joint])
            swa_slots = page_cache.acquire(PoolName.SWA, window_hashes)
            if swa_slots is None:
                self.release_span(PoolName.KV, cont_hashes, kv_slots)
                return None
            aux_xfers.append(
                PoolTransfer(
                    name=PoolName.SWA,
                    host_indices=swa_slots,
                    keys=window_hashes,
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
            )
        return kv_slots, aux_xfers
