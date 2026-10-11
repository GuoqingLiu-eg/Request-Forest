from __future__ import annotations

"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
The radix tree data structure for managing the hybrid (full and SWA) KV cache.
"""

import heapq
import time
from collections import defaultdict
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
from numpy import float64

from sglang.srt.environ import envs
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.events import KVCacheEventRecorder
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.utils import split_node_hash_value

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

import logging

logger = logging.getLogger(__name__)


class TreeNode:

    counter = 0
    swa_uuid_counter = 1
    last_access_time_counter_float = float64(1.0)

    def __init__(self, id: Optional[int] = None):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: RadixKey = None
        self.value: Optional[torch.Tensor] = None

        self.swa_tombstone = False


        self.full_lock_ref = 0
        self.swa_lock_ref = 0

        self.last_access_time = get_last_access_time()

        self.hit_count = 0

        self.host_value = None

        self.hash_value: Optional[List[str]] = None

        self.event_hash_value: Optional[List[str]] = None


        self.prev = None
        self.next = None
        self.swa_prev = None
        self.swa_next = None

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1
        self.swa_uuid = None

    @property
    def evicted(self):
        return self.value is None

    @property
    def backuped(self):
        return self.host_value is not None

    def __lt__(self, other: TreeNode):
        return self.last_access_time < other.last_access_time


def gen_swa_uuid() -> int:
    TreeNode.swa_uuid_counter += 1
    return TreeNode.swa_uuid_counter


def get_last_access_time() -> float64:
    ret = TreeNode.last_access_time_counter_float
    TreeNode.last_access_time_counter_float += 1.0
    return ret


class LRUList:
    def __init__(self, is_swa_list: bool = False):
        self.is_swa_list = is_swa_list
        if self.is_swa_list:
            self.prv = "swa_prev"
            self.nxt = "swa_next"
            self.lock_ref = "swa_lock_ref"
        else:
            self.prv = "prev"
            self.nxt = "next"
            self.lock_ref = "full_lock_ref"

        self.head = TreeNode()
        self.tail = TreeNode()
        setattr(self.head, self.nxt, self.tail)
        setattr(self.tail, self.prv, self.head)
        self.cache = {}

    def _add_node(self, node):

        self._add_node_after(self.head, node)

    def _add_node_after(self, old_node, new_node):

        setattr(new_node, self.prv, old_node)
        setattr(
            new_node, self.nxt, getattr(old_node, self.nxt)
        )
        setattr(
            getattr(old_node, self.nxt), self.prv, new_node
        )
        setattr(old_node, self.nxt, new_node)

    def _remove_node(self, node):

        setattr(
            getattr(node, self.prv), self.nxt, getattr(node, self.nxt)
        )
        setattr(
            getattr(node, self.nxt), self.prv, getattr(node, self.prv)
        )

        setattr(node, self.prv, None)
        setattr(node, self.nxt, None)

    def _get_lru(self) -> Optional[TreeNode]:

        if len(self.cache) == 0:
            return None
        return getattr(self.tail, self.prv)

    def reset_node_mru(self, node):

        assert node.id in self.cache, f"Resetting node {node.id=} not in lru list"
        assert (
            not self.is_swa_list or not node.swa_tombstone
        ), f"Resetting swa tombstone node in swa lru list: {node.id=}"
        self._remove_node(node)
        self._add_node(node)

    def reset_node_and_parents_mru(self, node, root_node):

        prev_node = self.head
        while node != root_node:

            if not self.is_swa_list or not node.swa_tombstone:
                assert (
                    node.id in self.cache
                ), f"Resetting node {node.id=} not in lru list when resetting node and parents mru"
                self._remove_node(node)
                self._add_node_after(prev_node, node)
                prev_node = node
            node = node.parent

    def insert_mru(self, node):

        assert (
            not self.is_swa_list or not node.swa_tombstone
        ), f"Inserting swa tombstone node in swa lru list: {node.id=}"
        assert (
            node.id not in self.cache
        ), f"Inserting node {node.id=} already in lru list, existing node: {self.cache[node.id].id=}"
        self.cache[node.id] = node
        self._add_node(node)

    def remove_node(self, node: TreeNode):

        assert node.id in self.cache, f"Removing node {node.id=} not in lru list"
        assert (
            not self.is_swa_list or not node.swa_tombstone
        ), f"Removing swa tombstone node from swa lru list: {node.id=}"
        del self.cache[node.id]
        self._remove_node(node)

    def get_lru_no_lock(self) -> Optional[TreeNode]:

        return self.get_prev_no_lock(self.tail, check_id=False)

    def get_leaf_lru_no_lock(self) -> Optional[TreeNode]:

        return self.get_prev_leaf_no_lock(self.tail, check_id=False)

    def get_prev_no_lock(
        self, node: TreeNode, check_id: bool = True
    ) -> Optional[TreeNode]:

        if check_id:
            assert (
                node.id in self.cache
            ), f"Getting prev of node {node.id=} not in lru list"
        x = getattr(node, self.prv)
        while getattr(x, self.lock_ref) > 0:
            x = getattr(x, self.prv)

        if x == self.head:
            return None
        return x

    def get_prev_leaf_no_lock(self, node: TreeNode, check_id: bool = True):

        if check_id:
            assert (
                node.id in self.cache
            ), f"Getting prev of node {node.id=} not in lru list"
        x = getattr(node, self.prv)
        while getattr(x, self.lock_ref) > 0 or len(x.children) > 0:
            x = getattr(x, self.prv)

        if x == self.head:
            return None
        return x

    def in_list(self, node: Optional[TreeNode]):

        if not node:
            return False
        return node.id in self.cache


    def sanity_check_evictable_size(self):

        node = self.get_lru_no_lock()
        evictable_size = 0
        while self.in_list(node):
            evictable_size += len(node.value)
            node = self.get_prev_no_lock(node)
        return evictable_size


    def sanity_check(self, tree_cache: SWARadixCache):

        try:
            if self.is_swa_list:
                nodes = tree_cache._collect_nontombstone_nodes()
            else:
                nodes = tree_cache._collect_all_nodes()
            total_nodes = len(nodes)
            total_lru_plus_1 = len(self.cache) + 1

            heapq.heapify(nodes)

            assert (
                len(nodes) == len(self.cache) + 1
            ), f"len(nodes): {len(nodes)} != len(self.cache) + 1: {len(self.cache) + 1}"

            x_lru = self._get_lru()
            while len(nodes):
                x = heapq.heappop(nodes)
                if x == tree_cache.root_node:

                    continue
                assert (
                    x == x_lru
                ), f"Incorrect LRU list, {self.is_swa_list=}, x: {x.id=} != x_lru: {x_lru.id=}"
                assert (
                    x_lru.full_lock_ref == 0
                ), f"x_lru should not be locked when idle, {x_lru.full_lock_ref=}, {x_lru.swa_uuid=}, {x_lru.id=}"
                assert (
                    x_lru.swa_lock_ref == 0
                ), f"x_lru should not be locked when idle, {x_lru.swa_lock_ref=}, {x_lru.swa_uuid=}, {x_lru.id=}"
                x_lru = getattr(x, self.prv)

            if self.is_swa_list:
                evictable_size = tree_cache.swa_evictable_size()
                lru_list_evictable_size = self.sanity_check_evictable_size()
            else:
                evictable_size = tree_cache.full_evictable_size()
                lru_list_evictable_size = self.sanity_check_evictable_size()

            assert (
                evictable_size == lru_list_evictable_size
            ), f"{self.is_swa_list=}, total nodes: {total_nodes}, total lru plus 1: {total_lru_plus_1}, evictable size: {evictable_size} != lru list evictable size: {lru_list_evictable_size}"
        except Exception as e:
            msg = f"SWA Radix tree sanity check failed, ping @hanming-lu: {e}"
            logger.error(msg)
            raise Exception(msg)


class SWARadixCache(BasePrefixCache):
    def __init__(self, params: CacheInitParams):
        assert isinstance(params.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator)
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        self.disable = params.disable
        self.is_eagle = params.is_eagle
        self.kv_events = KVCacheEventRecorder(
            enabled=params.enable_kv_cache_events, page_size=self.page_size
        )

        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        if params.enable_metrics:
            self.init_metrics_collector()

        self.sliding_window_size = params.sliding_window_size
        self.reset()


    def supports_swa(self) -> bool:
        assert (
            self.sliding_window_size is not None
        ), "sliding_window_size must be set for SWARadixCache"
        return True

    def swa_reprefill_tail_tokens(self) -> int:

        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
            is_unified_kv_triton,
        )

        if self.sliding_window_size and is_unified_kv_triton():
            return self.sliding_window_size
        return 0

    def reset(self) -> None:
        self.root_node = TreeNode()
        self.root_node.key = []
        self.root_node.value = []
        self.root_node.hash_value = []
        self.root_node.full_lock_ref = 1
        self.root_node.swa_lock_ref = 1
        self.full_evictable_size_ = 0
        self.swa_evictable_size_ = 0
        self.full_protected_size_ = 0
        self.swa_protected_size_ = 0

        self.full_lru_list = LRUList(is_swa_list=False)
        self.swa_lru_list = LRUList(is_swa_list=True)
        self.kv_events.record_all_cleared()

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:


        key = self._match_pre_processor(params)
        if key is None:
            return MatchResult(
                device_indices=torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ),
                last_device_node=self.root_node,
                last_host_node=self.root_node,
                best_match_node=self.root_node,
            )

        value, last_node, best_value_len = self._match_prefix_helper(key)
        return self._match_post_processor(params, value, last_node, best_value_len)

    def insert(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0)

        key = params.key
        value = params.value
        prev_prefix_len = params.prev_prefix_len
        swa_evicted_seqlen = params.swa_evicted_seqlen

        key, value = key.maybe_to_bigram_view(self.is_eagle, value)
        key = key.page_aligned(self.page_size)
        if value is not None:
            value = value[: len(key)]
        else:
            value = torch.tensor(key.token_ids[: len(key)], dtype=torch.int64)

        prefix_len = self._insert_helper(
            self.root_node, key, value, prev_prefix_len, swa_evicted_seqlen
        )
        return InsertResult(prefix_len=prefix_len)

    def cache_finished_req(
        self, req: Req, is_insert: bool = True, *, kv_len_to_handle: int
    ) -> None:

        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_len_to_handle
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_len_to_handle]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_len_to_handle
        ]

        radix_key = RadixKey(
            token_ids,
            req.extra_key,
            is_bigram=self.is_eagle,
            cache_salt=req.cache_salt,
        ).page_aligned(self.page_size)
        page_aligned_len = len(radix_key)
        values = kv_indices[:page_aligned_len].to(dtype=torch.int64, copy=True)
        old_prefix_len = req.cache_protected_len


        if is_insert:
            self.insert(
                InsertParams(
                    key=radix_key,
                    value=values,
                    prev_prefix_len=old_prefix_len,
                    swa_evicted_seqlen=req.kv.swa_evicted_seqlen,
                )
            )
        else:
            self.token_to_kv_pool_allocator.free(
                kv_indices[old_prefix_len:page_aligned_len]
            )


        self.token_to_kv_pool_allocator.free(kv_indices[page_aligned_len:])


        self.dec_lock_ref(
            req.last_node,
            DecLockRefParams(swa_uuid_for_lock=req.swa_uuid_for_lock),
            skip_swa=req.swa_prefix_lock_released,
        )
        req.swa_prefix_lock_released = False

    def cache_unfinished_req(self, req: Req, chunked=False) -> None:

        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : req.extend_range.end
            ]


            req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)
            return

        token_ids = req.get_fill_ids()
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        radix_key = RadixKey(
            token_ids,
            req.extra_key,
            is_bigram=self.is_eagle,
            cache_salt=req.cache_salt,
        ).page_aligned(self.page_size)
        values = kv_indices[: len(radix_key)].to(dtype=torch.int64, copy=True)
        old_prefix_len = req.cache_protected_len


        result = self.insert(
            InsertParams(
                key=radix_key,
                value=values,
                prev_prefix_len=old_prefix_len,
            )
        )
        new_prefix_len = result.prefix_len


        match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
        new_indices, new_last_node = (
            match_result.device_indices,
            match_result.last_device_node,
        )

        assert old_prefix_len <= len(new_indices), f"{old_prefix_len=}, {new_indices=}"
        assert new_prefix_len <= len(new_indices), f"{new_prefix_len=}, {new_indices=}"
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(old_prefix_len, len(new_indices))),
            new_indices[old_prefix_len:],
        )

        req.cache_protected_len = len(new_indices)

        self.dec_lock_ref(
            req.last_node,
            DecLockRefParams(swa_uuid_for_lock=req.swa_uuid_for_lock),
            skip_swa=req.swa_prefix_lock_released,
        )
        req.swa_prefix_lock_released = False
        result = self.inc_lock_ref(new_last_node)
        swa_uuid_for_lock = result.swa_uuid_for_lock


        if len(new_indices) < len(kv_indices):
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices
        req.last_node = new_last_node
        req.swa_uuid_for_lock = swa_uuid_for_lock

    def pretty_print(self) -> None:
        self._print_helper(self.root_node, 0)
        total_size, total_swa_size = self._total_size_helper()
        print(f"#full_tokens: {total_size}, #swa_tokens: {total_swa_size}")

    def total_size(self) -> Tuple[int, int]:
        return self._total_size_helper()

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()
        start_time = time.perf_counter()
        full_num_tokens = params.num_tokens
        swa_num_tokens = params.swa_num_tokens
        full_num_evicted = 0
        swa_num_evicted = 0
        if full_num_tokens > 0:

            x = self.full_lru_list.get_leaf_lru_no_lock()

            while full_num_evicted < full_num_tokens and self.full_lru_list.in_list(x):
                assert (
                    x != self.root_node
                ), f"root node should not exist in full lru list, {x.id=}"
                assert x.full_lock_ref == 0, f"node is in use, {x.id=}"


                self.kv_events.record_remove(x)
                self.token_to_kv_pool_allocator.free(x.value)
                full_num_evicted += len(x.value)

                if not x.swa_tombstone:
                    swa_num_evicted += len(x.value)


                x_next = self.full_lru_list.get_prev_leaf_no_lock(x)
                self.full_lru_list.remove_node(x)
                if not x.swa_tombstone:
                    self.swa_lru_list.remove_node(x)


                self._delete_leaf(x)


                x, leaf_full_num_evicted = self._iteratively_delete_tombstone_leaf(x)
                full_num_evicted += leaf_full_num_evicted


                if len(x.parent.children) == 0:
                    x_next = self.full_lru_list.get_leaf_lru_no_lock()

                x = x_next

        if swa_num_evicted < swa_num_tokens:

            x = self.swa_lru_list.get_lru_no_lock()


            while swa_num_evicted < swa_num_tokens and (self.swa_lru_list.in_list(x)):
                assert not x.swa_tombstone, f"duplicate swa tombstone node, {x.id=}"
                assert x != self.root_node, f"root node is not evictable, {x.id=}"
                assert x.swa_lock_ref == 0, f"node is in use by swa kv indices, {x.id=}"

                if len(x.children) > 0:

                    self.token_to_kv_pool_allocator.free_swa(x.value)
                    swa_num_evicted += len(x.value)


                    x_next = self.swa_lru_list.get_prev_no_lock(x)
                    self.swa_lru_list.remove_node(x)


                    self._tombstone_internal_node(x)
                elif x.full_lock_ref > 0:


                    self.token_to_kv_pool_allocator.free_swa(x.value)
                    swa_num_evicted += len(x.value)

                    x_next = self.swa_lru_list.get_prev_no_lock(x)
                    self.swa_lru_list.remove_node(x)

                    self.swa_evictable_size_ -= len(x.value)
                    x.swa_tombstone = True
                else:
                    assert (
                        x.full_lock_ref == 0
                    ), f"leaf node with full lock must also have swa lock, {x.id=}"

                    self.kv_events.record_remove(x)
                    self.token_to_kv_pool_allocator.free(x.value)
                    full_num_evicted += len(x.value)
                    swa_num_evicted += len(x.value)


                    x_next = self.swa_lru_list.get_prev_no_lock(x)
                    self.full_lru_list.remove_node(x)
                    self.swa_lru_list.remove_node(x)


                    self._delete_leaf(x)


                    self._iteratively_delete_tombstone_leaf(x)

                x = x_next

        self.update_eviction_metrics(full_num_evicted + swa_num_evicted, start_time)
        return EvictResult(
            num_tokens_evicted=full_num_evicted, swa_num_tokens_evicted=swa_num_evicted
        )

    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:

        if self.disable:
            return IncLockRefResult()

        swa_lock_size = 0
        swa_uuid_for_lock = None
        while node != self.root_node:

            assert (
                node.full_lock_ref >= 0
            ), f"inc_lock_ref on node with {node.full_lock_ref=}, {node.id=}"
            if node.full_lock_ref == 0:
                self.full_evictable_size_ -= len(node.value)
                self.full_protected_size_ += len(node.value)
            node.full_lock_ref += 1


            if swa_lock_size < self.sliding_window_size:
                assert (
                    not node.swa_tombstone
                ), f"inc_lock_swa on swa_tombstone node, {node.id=}"
                if node.swa_lock_ref == 0:
                    self.swa_evictable_size_ -= len(node.value)
                    self.swa_protected_size_ += len(node.value)
                node.swa_lock_ref += 1
                swa_lock_size += len(node.value)
                if swa_lock_size >= self.sliding_window_size:
                    if node.swa_uuid is None:
                        node.swa_uuid = gen_swa_uuid()
                    swa_uuid_for_lock = node.swa_uuid
            node = node.parent
        return IncLockRefResult(swa_uuid_for_lock=swa_uuid_for_lock)

    def dec_lock_ref(
        self,
        node: TreeNode,
        params: Optional[DecLockRefParams] = None,
        skip_swa: bool = False,
    ) -> DecLockRefResult:

        swa_uuid_for_lock = params.swa_uuid_for_lock if params is not None else None

        if self.disable:
            return DecLockRefResult()

        dec_lock_swa = not skip_swa
        while node != self.root_node:
            assert (
                node.full_lock_ref > 0
            ), f"dec_lock_ref on node with {node.full_lock_ref=}, {node.id=}"
            if node.full_lock_ref == 1:
                self.full_evictable_size_ += len(node.value)
                self.full_protected_size_ -= len(node.value)
            node.full_lock_ref -= 1

            if dec_lock_swa:
                assert (
                    not node.swa_tombstone
                ), f"dec_lock_ref on swa_tombstone node, {node.id=}"
                assert (
                    node.swa_lock_ref > 0
                ), f"dec_lock_ref on node with {node.swa_lock_ref=}, {node.id=}"

                if node.swa_lock_ref == 1:
                    self.swa_evictable_size_ += len(node.value)
                    self.swa_protected_size_ -= len(node.value)
                node.swa_lock_ref -= 1
                if swa_uuid_for_lock and node.swa_uuid == swa_uuid_for_lock:
                    dec_lock_swa = False

            node = node.parent

        return DecLockRefResult()

    def dec_swa_lock_only(
        self,
        node: TreeNode,
        swa_uuid_for_lock: Optional[int] = None,
        skip_lock_node_ids: Optional[dict] = None,
    ):

        if self.disable:
            return

        while node != self.root_node:
            assert (
                not node.swa_tombstone
            ), f"dec_swa_lock_only on swa_tombstone node, {node.id=}"
            assert (
                node.swa_lock_ref > 0
            ), f"dec_swa_lock_only on node with {node.swa_lock_ref=}, {node.id=}"

            if node.swa_lock_ref == 1:
                self.swa_protected_size_ -= len(node.value)
                if len(node.children) == 0:


                    self.token_to_kv_pool_allocator.free_swa(node.value)
                    self.swa_lru_list.remove_node(node)
                    node.swa_tombstone = True
                else:

                    self.swa_evictable_size_ += len(node.value)
            node.swa_lock_ref -= 1

            if swa_uuid_for_lock and node.swa_uuid == swa_uuid_for_lock:
                break
            node = node.parent

    def sanity_check(self):
        self.full_lru_list.sanity_check(self)
        self.swa_lru_list.sanity_check(self)

    def evictable_size(self) -> Tuple[int, int]:

        raise NotImplementedError

    def full_evictable_size(self) -> int:
        return self.full_evictable_size_

    def swa_evictable_size(self) -> int:
        return self.swa_evictable_size_

    def protected_size(self) -> Tuple[int, int]:

        raise NotImplementedError

    def full_protected_size(self) -> int:

        return self.full_protected_size_

    def swa_protected_size(self) -> int:

        return self.swa_protected_size_

    def all_values_flatten(self) -> torch.Tensor:
        values = []

        def _dfs_helper(node: TreeNode):
            for _, child in node.children.items():
                values.append(child.value)
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values)

    def available_and_evictable_str(self) -> str:
        full_available_size = self.token_to_kv_pool_allocator.full_available_size()
        swa_available_size = self.token_to_kv_pool_allocator.swa_available_size()
        full_evictable_size = self.full_evictable_size()
        swa_evictable_size = self.swa_evictable_size()
        return (
            f"Available full tokens: {full_available_size + full_evictable_size} ({full_available_size=} + {full_evictable_size=})\n"
            f"Available swa tokens: {swa_available_size + swa_evictable_size} ({swa_available_size=} + {swa_evictable_size=})\n"
            f"Full LRU list evictable size: {self.full_lru_list.sanity_check_evictable_size()}\n"
            f"SWA LRU list evictable size: {self.swa_lru_list.sanity_check_evictable_size()}\n"
        )


    def _match_prefix_helper(
        self, key: RadixKey
    ) -> Tuple[List[torch.Tensor], TreeNode, int]:

        node = self.root_node
        child_key = key.child_key(self.page_size)

        value = []

        match_len_since_tombstone = float("inf")
        best_value_len = 0
        best_last_node = node
        enable_compact = envs.SGLANG_OPT_SWA_RADIX_CACHE_COMPACT.get()
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]

            if enable_compact:
                self._compact_single_child_chain(child)

            if child.swa_tombstone:

                if match_len_since_tombstone >= self.sliding_window_size:
                    best_value_len = len(value)
                    best_last_node = node

                match_len_since_tombstone = 0

            prefix_len = child.key.match(key, page_size=self.page_size)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                if not new_node.swa_tombstone:
                    match_len_since_tombstone += len(new_node.value)
                node = new_node
                break
            else:
                value.append(child.value)
                if not child.swa_tombstone:
                    match_len_since_tombstone += len(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = key.child_key(self.page_size)


        if match_len_since_tombstone >= self.sliding_window_size:
            best_value_len = len(value)
            best_last_node = node

        return value, best_last_node, best_value_len

    def _match_pre_processor(self, params: MatchPrefixParams) -> Optional[RadixKey]:

        key = params.key
        key, _ = key.maybe_to_bigram_view(self.is_eagle)
        if self.disable or len(key) == 0:
            return None
        key = key.page_aligned(self.page_size)
        if len(key) == 0:
            return None
        return key

    def _match_post_processor(
        self,
        params: MatchPrefixParams,
        value: List[torch.Tensor],
        last_node: TreeNode,
        best_value_len: int,
    ) -> MatchResult:

        node_update = last_node


        self.full_lru_list.reset_node_and_parents_mru(node_update, self.root_node)
        self.swa_lru_list.reset_node_and_parents_mru(node_update, self.root_node)


        cur_time = get_last_access_time()
        while node_update:
            node_update.last_access_time = cur_time
            cur_time -= (
                0.00001
            )
            node_update = node_update.parent

        value = value[:best_value_len]
        if value:
            value = torch.cat(value)
        else:
            value = torch.empty((0,), dtype=torch.int64, device=self.device)

        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
            best_match_node=last_node,
        )

    def _compact_single_child_chain(self, node: TreeNode) -> None:


        while len(node.children) == 1:
            child = next(iter(node.children.values()))
            if len(child.children) == 0:
                break
            sum_gc_full_lock_ref = sum(
                gc.full_lock_ref for gc in child.children.values()
            )
            if child.full_lock_ref > sum_gc_full_lock_ref:
                break
            if (
                child.swa_tombstone != node.swa_tombstone
                or child.full_lock_ref != node.full_lock_ref
                or child.swa_lock_ref != node.swa_lock_ref
            ):
                break


            node.key = RadixKey(
                node.key.token_ids + child.key.token_ids,
                node.key.extra_key,
                is_bigram=node.key.is_bigram,
                cache_salt=node.key.cache_salt,
            )
            node.value = torch.cat([node.value, child.value])
            node.children = child.children
            for grandchild in node.children.values():
                grandchild.parent = node

            if child.swa_uuid is not None:
                node.swa_uuid = child.swa_uuid

            if node.hash_value is not None and child.hash_value is not None:
                node.hash_value = list(node.hash_value) + list(child.hash_value)
            else:
                node.hash_value = None
            if node.event_hash_value is not None and child.event_hash_value is not None:
                node.event_hash_value = list(node.event_hash_value) + list(
                    child.event_hash_value
                )
            else:
                node.event_hash_value = None

            self.full_lru_list.remove_node(child)
            if not child.swa_tombstone:
                self.swa_lru_list.remove_node(child)

    def _maybe_split_leaf_for_swa_lock(self, leaf: TreeNode) -> TreeNode:

        if (
            leaf is self.root_node
            or leaf.swa_lock_ref > 0
            or leaf.swa_tombstone
            or len(leaf.value) == 0
        ):
            return leaf


        tail_size = (
            (self.sliding_window_size + self.page_size - 1)
            // self.page_size
            * self.page_size
        )
        if len(leaf.value) <= tail_size:
            return leaf

        split_at = len(leaf.value) - tail_size

        if split_at <= 0 or split_at >= len(leaf.value):
            return leaf
        if self.page_size > 1 and (
            split_at % self.page_size != 0 or len(leaf.value) % self.page_size != 0
        ):
            return leaf

        self._split_node(leaf.key, leaf, split_at)
        return leaf

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int) -> TreeNode:

        new_node = TreeNode()
        new_node.children = {key[split_len:].child_key(self.page_size): child}
        new_node.parent = child.parent
        new_node.swa_tombstone = child.swa_tombstone
        new_node.full_lock_ref = child.full_lock_ref
        new_node.swa_lock_ref = child.swa_lock_ref
        new_node.key = child.key[:split_len]
        assert len(new_node.key) > 0, f"new_node.key should not be empty"
        new_node.value = child.value[:split_len].clone()

        new_node.swa_uuid = child.swa_uuid
        child.swa_uuid = None

        child.last_access_time = get_last_access_time()


        self.full_lru_list.remove_node(child)
        if not new_node.swa_tombstone:
            self.swa_lru_list.remove_node(child)
        child.parent = new_node
        child.key = child.key[split_len:]
        assert len(child.key) > 0, f"child.key should not be empty"
        child.value = child.value[split_len:].clone()
        new_node.parent.children[key.child_key(self.page_size)] = new_node
        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )
        new_node.event_hash_value, child.event_hash_value = split_node_hash_value(
            child.event_hash_value, split_len, self.page_size
        )


        self.full_lru_list.insert_mru(new_node)
        self.full_lru_list.insert_mru(child)
        if not new_node.swa_tombstone:
            self.swa_lru_list.insert_mru(new_node)
            self.swa_lru_list.insert_mru(child)
        return new_node

    def _insert_helper(
        self,
        node: TreeNode,
        key: RadixKey,
        value,
        update_kv_after_len: int,
        swa_evicted_seqlen: int = 0,
    ) -> int:


        node.last_access_time = get_last_access_time()
        if node != self.root_node:
            self.full_lru_list.reset_node_mru(node)
            if not node.swa_tombstone:
                self.swa_lru_list.reset_node_mru(node)
        if len(key) == 0:
            return 0

        child_key = key.child_key(self.page_size)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = get_last_access_time()
            self.full_lru_list.reset_node_mru(node)
            if not node.swa_tombstone:
                self.swa_lru_list.reset_node_mru(node)
            prefix_len = node.key.match(key, page_size=self.page_size)

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node


            if update_kv_after_len < total_prefix_length + prefix_len:


                if node.swa_tombstone:
                    assert (
                        node.swa_lock_ref == 0
                    ), f"tombstone swa_lock_ref should always be 0, {node.full_lock_ref=}, {node.swa_lock_ref=}, {node.id=}"
                    assert (
                        swa_evicted_seqlen % self.page_size == 0
                    ), f"swa_evicted_seqlen must be page aligned, {swa_evicted_seqlen=}, {self.page_size=}"
                    if swa_evicted_seqlen <= total_prefix_length:

                        if node.full_lock_ref > 0:


                            self._recover_tombstone_keeping_locked_full(
                                node, value[:prefix_len]
                            )
                        else:

                            self.token_to_kv_pool_allocator.free(
                                node.value[:prefix_len]
                            )

                            node.value = value[:prefix_len].clone()
                            node.swa_tombstone = False
                            self.swa_lru_list.insert_mru(node)
                            self.swa_evictable_size_ += len(node.value)
                    elif swa_evicted_seqlen < total_prefix_length + prefix_len:

                        start_update_idx = swa_evicted_seqlen - total_prefix_length
                        if node.full_lock_ref > 0:


                            self._split_node(node.key, node, start_update_idx)
                            self._recover_tombstone_keeping_locked_full(
                                node, value[start_update_idx:prefix_len]
                            )
                            self.token_to_kv_pool_allocator.free(
                                value[:start_update_idx]
                            )
                        else:
                            self.token_to_kv_pool_allocator.free(
                                node.value[start_update_idx:prefix_len]
                            )
                            self._split_node(node.key, node, start_update_idx)


                            node.value = value[start_update_idx:prefix_len].clone()
                            self.token_to_kv_pool_allocator.free(
                                value[:start_update_idx]
                            )
                            node.swa_tombstone = False
                            self.swa_lru_list.insert_mru(node)
                            self.swa_evictable_size_ += len(node.value)
                    else:

                        self.token_to_kv_pool_allocator.free(value[:prefix_len])
                else:

                    self.token_to_kv_pool_allocator.free(value[:prefix_len])

            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if len(key):
                child_key = key.child_key(self.page_size)

        if len(key):


            if swa_evicted_seqlen == total_prefix_length + len(key):
                self.token_to_kv_pool_allocator.free(value)
                return total_prefix_length

            if (
                swa_evicted_seqlen > total_prefix_length
                and swa_evicted_seqlen < total_prefix_length + len(key)
            ):
                swa_tombstone_len = swa_evicted_seqlen - total_prefix_length
                node = self._add_new_node(
                    node,
                    key[:swa_tombstone_len],
                    value[:swa_tombstone_len],
                    swa_tombstone=True,
                )
                key = key[swa_tombstone_len:]
                value = value[swa_tombstone_len:]

            new_leaf = self._add_new_node(node, key, value, swa_tombstone=False)

            if envs.SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT.get():


                self._maybe_split_leaf_for_swa_lock(new_leaf)

        return total_prefix_length

    def _recover_tombstone_keeping_locked_full(
        self, node: TreeNode, incoming_full: torch.Tensor
    ) -> None:

        assert len(node.value) == len(incoming_full), (
            f"locked-full recover size mismatch: {len(node.value)=}, "
            f"{len(incoming_full)=}"
        )

        allocator = self.token_to_kv_pool_allocator
        swa_value = allocator.translate_loc_from_full_to_swa(incoming_full)
        allocator.set_full_to_swa_mapping(node.value, swa_value)
        allocator.clear_full_to_swa_mapping(incoming_full)
        allocator.full_attn_allocator.free(incoming_full)

        node.swa_tombstone = False
        self.swa_lru_list.insert_mru(node)
        self.swa_evictable_size_ += len(node.value)

    def _add_new_node(
        self,
        parent: TreeNode,
        key: RadixKey,
        value: torch.Tensor,
        swa_tombstone: bool = False,
    ) -> TreeNode:
        assert len(key) > 0, f"key should not be empty"
        new_node = TreeNode()
        new_node.parent = parent
        new_node.key = key
        new_node.value = value.clone()
        new_node.swa_tombstone = swa_tombstone
        parent.children[key.child_key(self.page_size)] = new_node
        self.full_lru_list.insert_mru(new_node)
        self.full_evictable_size_ += len(value)
        if not swa_tombstone:
            self.swa_lru_list.insert_mru(new_node)
            self.swa_evictable_size_ += len(value)
        self.kv_events.record_store(new_node)
        return new_node

    def _iteratively_delete_tombstone_leaf(
        self, node: TreeNode
    ) -> Tuple[TreeNode, int]:
        full_num_evicted = 0
        while node.parent.swa_tombstone and len(node.parent.children) == 0:

            if node.parent == self.root_node:
                break

            if node.parent.full_lock_ref > 0:
                break
            assert (
                node.parent.swa_lock_ref == 0
            ), f"tombstone swa_lock_ref should always be 0, {node.parent.full_lock_ref=}, {node.parent.swa_lock_ref=}, {node.parent.id=}"

            self.kv_events.record_remove(node.parent)
            self.token_to_kv_pool_allocator.free(node.parent.value)
            full_num_evicted += len(node.parent.value)
            self.full_lru_list.remove_node(node.parent)
            self._delete_tombstone_leaf(node.parent)
            node = node.parent

        return node, full_num_evicted

    def _delete_leaf(self, node: TreeNode) -> None:
        assert len(node.children) == 0, f"leaf node has children, {node.id=}"
        key = node.key.child_key(self.page_size)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"
        self.full_evictable_size_ -= len(node.key)


        if not node.swa_tombstone:
            self.swa_evictable_size_ -= len(node.key)

    def _tombstone_internal_node(self, node: TreeNode) -> None:
        assert len(node.children) != 0, f"Cannot tombstone a leaf node, {node.id=}"
        node.swa_tombstone = True
        self.swa_evictable_size_ -= len(node.key)

    def _delete_tombstone_leaf(self, node: TreeNode) -> None:
        assert (
            node.swa_tombstone
        ), f"Deleting a unexpected non-tombstone leaf node, {node.id=}"
        assert len(node.children) == 0, f"leaf node has children, {node.id=}"
        key = node.key.child_key(self.page_size)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"

        self.full_evictable_size_ -= len(node.key)

    def _collect_nontombstone_nodes(self) -> List[TreeNode]:
        ret_list = []
        stack = [self.root_node]

        while stack:
            cur_node = stack.pop()
            if not cur_node.swa_tombstone:
                ret_list.append(cur_node)
            stack.extend(cur_node.children.values())

        return ret_list

    def _collect_all_nodes(self) -> List[TreeNode]:
        ret_list = []
        stack = [self.root_node]
        while stack:
            cur_node = stack.pop()
            ret_list.append(cur_node)
            stack.extend(cur_node.children.values())
        return ret_list

    def _print_helper(self, node: TreeNode, indent: int) -> None:

        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                current_node.id,
                len(current_node.key),
                f"fr={current_node.full_lock_ref}",
                f"sr={current_node.swa_lock_ref}",
                f"fll={self.full_lru_list.in_list(current_node)}",
                f"sll={self.swa_lru_list.in_list(current_node)}",
                f"ts={current_node.swa_tombstone}",
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

                assert key == child.key.child_key(
                    self.page_size
                ), f"{key=}, {child.key.child_key(self.page_size)=}"

    def _total_size_helper(self) -> Tuple[int, int]:
        total_size = 0
        total_swa_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            if not current_node.swa_tombstone:
                total_swa_size += len(current_node.value)
            for child in current_node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
        return total_size, total_swa_size
