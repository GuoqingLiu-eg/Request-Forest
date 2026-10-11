from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    EvictResult,
    InsertParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


class PureSWARadixCache(RadixCache):


    def __init__(self, params: CacheInitParams):
        super().__init__(params)
        self.sliding_window_size = params.sliding_window_size

    def supports_swa(self) -> bool:
        assert (
            self.sliding_window_size is not None
        ), "sliding_window_size must be set for PureSWARadixCache"
        return True

    def swa_evictable_size(self):
        return self.evictable_size_

    def swa_protected_size(self):
        return self.protected_size_

    def full_evictable_size(self):
        return 0

    def full_protected_size(self):
        return 0

    def sanity_check(self):

        pass

    def evict(self, params: EvictParams) -> EvictResult:

        num_tokens = max(params.num_tokens, params.swa_num_tokens)
        return super().evict(EvictParams(num_tokens=num_tokens))

    def cache_finished_req(
        self, req: Req, is_insert: bool = True, *, kv_len_to_handle: int
    ):

        if self.disable_finished_insert:
            is_insert = False

        kv_committed_len = kv_len_to_handle
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_committed_len
        ]

        radix_key = RadixKey(
            token_ids,
            req.extra_key,
            is_bigram=self.is_eagle,
            cache_salt=req.cache_salt,
        ).page_aligned(self.page_size)
        keys_len = len(radix_key)

        old_prefix_len = req.cache_protected_len
        swa_evict_floor = req.swa_evict_floor
        swa_evicted_seqlen = req.kv.swa_evicted_seqlen

        if self.page_size > 1 and swa_evict_floor > 0:
            swa_evict_floor = -(-swa_evict_floor // self.page_size) * self.page_size

        if swa_evict_floor > 0:
            insert_end = min(swa_evict_floor, keys_len)
        else:
            insert_end = keys_len

        if is_insert and insert_end > 0:
            insert_values = kv_indices[:insert_end].to(dtype=torch.int64, copy=True)
            result = self.insert(
                InsertParams(key=radix_key[:insert_end], value=insert_values)
            )
            new_prefix_len = result.prefix_len
            if new_prefix_len > old_prefix_len:
                self.token_to_kv_pool_allocator.free(
                    kv_indices[old_prefix_len:new_prefix_len]
                )
            alive_start = max(swa_evicted_seqlen, insert_end)
            if alive_start < keys_len:
                self.token_to_kv_pool_allocator.free(kv_indices[alive_start:keys_len])
        else:
            free_end = (
                min(swa_evict_floor, keys_len) if swa_evict_floor > 0 else keys_len
            )
            if free_end > old_prefix_len:
                self.token_to_kv_pool_allocator.free(
                    kv_indices[old_prefix_len:free_end]
                )
            alive_start = max(swa_evicted_seqlen, old_prefix_len)
            if swa_evicted_seqlen > 0 and alive_start < keys_len:
                self.token_to_kv_pool_allocator.free(kv_indices[alive_start:keys_len])

        self.token_to_kv_pool_allocator.free(kv_indices[keys_len:])

        if req.last_node is not None:
            self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req, chunked=False):

        super().cache_unfinished_req(req, chunked=chunked)

    def available_and_evictable_str(self) -> str:
        allocator = self.token_to_kv_pool_allocator
        swa_available = allocator.swa_available_size()
        swa_evictable = self.swa_evictable_size()
        return (
            f"SWA available tokens: {swa_available + swa_evictable} "
            f"({swa_available=} + {swa_evictable=})\n"
        )
