from __future__ import annotations

import enum
import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    EvictResult,
    InitLoadBackParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode
from sglang.srt.mem_cache.storage.flexkv.flexkv_connector import FlexKVConnector
from sglang.srt.runtime_context import get_spec

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class FlexKVMode(enum.Enum):
    MP = enum.auto()
    IP = enum.auto()


@dataclass
class _LoadBackMarker:


    key: RadixKey
    value_numel: int


class FlexKVRadixCache(RadixCache):


    def __init__(
        self,
        params: CacheInitParams,
        model_config: Optional[ModelConfig],
        server_args: ServerArgs,
        tp_rank: int,
        tp_size: int,
        dp_rank: Optional[int],
        pp_rank: int,
        attn_cp_rank: int,
        tp_group=None,
        pp_group=None,
        attn_tp_group=None,
        attn_cp_group=None,
    ) -> None:
        super().__init__(params)

        kvcache = self.token_to_kv_pool_allocator.get_kvcache()


        attn_tp_group_eff = attn_tp_group if attn_tp_group is not None else tp_group

        self.flexkv_connector = FlexKVConnector(
            sgl_model_config=model_config,
            server_args=server_args,
            page_size=params.page_size,
            kvcache=kvcache,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            pp_rank=pp_rank,
            attn_cp_rank=attn_cp_rank,
            pp_group=pp_group,
            attn_tp_group=attn_tp_group_eff,
            attn_cp_group=attn_cp_group,
        )

        self._mode = (
            FlexKVMode.IP if self.flexkv_connector.enable_layerwise else FlexKVMode.MP
        )
        if self._mode is FlexKVMode.IP:


            self.flexkv_connector.register_layer_transfer_counter(kvcache)


        self.load_stream = torch.cuda.Stream()
        self.store_stream = torch.cuda.Stream()


        self._load_markers: dict[str, _LoadBackMarker] = {}


        self._inflight_store_nodes: dict[str, TreeNode] = {}
        self._node_lock = threading.Lock()


    def reset(self) -> None:  # type: ignore[override]
        super().reset()
        if hasattr(self, "_load_markers"):
            self._load_markers.clear()
        if hasattr(self, "_inflight_store_nodes"):
            with self._node_lock:
                self._inflight_store_nodes.clear()
        if hasattr(self, "flexkv_connector"):
            self.flexkv_connector.reset()

    def shutdown(self) -> None:
        if hasattr(self, "flexkv_connector"):
            self.flexkv_connector.shutdown()


    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:  # type: ignore[override]

        key = params.key
        if self.disable or not key:
            return super().match_prefix(params)


        if self.page_size != 1:
            aligned_len = (len(key) // self.page_size) * self.page_size
            key = key[:aligned_len]

        base_res = super().match_prefix(params)
        if len(key) == 0:
            return base_res

        device_value: torch.Tensor = base_res.device_indices
        last_node: TreeNode = base_res.last_device_node

        if self._mode is FlexKVMode.MP:
            if params.req is None:
                return base_res
            return self._mp_match_prefix(
                key, base_res, device_value, last_node, params.req
            )
        return self._ip_match_prefix(key, base_res, device_value, last_node)

    def _mp_match_prefix(
        self,
        key: RadixKey,
        base_res: MatchResult,
        device_value: torch.Tensor,
        last_node: TreeNode,
        req: Req,
    ) -> MatchResult:

        token_ids = key.raw_token_ids()
        device_len = int(device_value.numel())
        if device_len >= len(token_ids):
            return base_res


        token_mask = torch.zeros(len(token_ids), dtype=torch.bool)
        token_mask[device_len:] = True

        fkv_task_id, hit = self.flexkv_connector.lookup_kv(
            token_ids=token_ids, token_mask=token_mask, rid=req.rid
        )
        if hit <= 0:
            return base_res


        if token_ids is key.token_ids:
            token_ids_snap = token_ids[:]
        else:
            token_ids_snap = token_ids
        self._load_markers[req.rid] = _LoadBackMarker(
            key=RadixKey(
                token_ids_snap,
                key.extra_key,
                key.is_bigram,
                cache_salt=key.cache_salt,
            ),
            value_numel=device_len,
        )
        return MatchResult(
            device_indices=device_value,
            last_device_node=last_node,
            last_host_node=last_node,
            best_match_node=last_node,
            host_hit_length=hit,
        )

    def _ip_match_prefix(
        self,
        key: RadixKey,
        base_res: MatchResult,
        device_value: torch.Tensor,
        last_node: TreeNode,
    ) -> MatchResult:

        token_ids = key.raw_token_ids()
        device_len = int(device_value.numel())
        if device_len >= len(token_ids):
            return base_res


        token_mask = torch.zeros(len(token_ids), dtype=torch.bool)
        token_mask[device_len:] = True

        synthetic_rid = f"_ip_{id(key)}"
        _, hit = self.flexkv_connector.lookup_kv(
            token_ids=token_ids, token_mask=token_mask, rid=synthetic_rid
        )
        if hit <= 0:
            return base_res

        result = self._allocate_and_load(
            key=key,
            value_numel=device_len,
            uncached_len=hit,
            last_node=last_node,
            load_fn=lambda slot_mapping: self.flexkv_connector.start_load_kv_layerwise(
                synthetic_rid, slot_mapping
            )[0],
        )
        if result is None:
            return base_res
        new_slots, new_node = result
        return MatchResult(
            device_indices=torch.cat([device_value, new_slots]),
            last_device_node=new_node,
            last_host_node=new_node,
            best_match_node=new_node,
        )


    def init_load_back(  # type: ignore[override]
        self,
        params: InitLoadBackParams,
    ) -> Tuple[torch.Tensor, Optional[TreeNode]]:

        req = params.req
        last_node: TreeNode = params.best_match_node
        marker = self._load_markers.pop(req.rid, None)
        if marker is None:


            self.flexkv_connector.release_pending(req.rid)
            return (
                torch.empty((0,), dtype=torch.int64, device=self.device),
                last_node,
            )

        result = self._allocate_and_load(
            key=marker.key,
            value_numel=marker.value_numel,
            uncached_len=params.host_hit_length,
            last_node=last_node,
            load_fn=lambda slot_mapping: self.flexkv_connector.retrieve_kv(
                req.rid, slot_mapping
            ),
        )
        if result is None:


            self.flexkv_connector.release_pending(req.rid)
            return (
                torch.empty((0,), dtype=torch.int64, device=self.device),
                last_node,
            )
        return result

    def _allocate_and_load(
        self,
        *,
        key: RadixKey,
        value_numel: int,
        uncached_len: int,
        last_node: TreeNode,
        load_fn,
    ) -> Optional[Tuple[torch.Tensor, TreeNode]]:

        if uncached_len <= 0:
            return None


        if self.token_to_kv_pool_allocator.available_size() < uncached_len:
            self.evict(EvictParams(num_tokens=uncached_len))
        token_slots = self.token_to_kv_pool_allocator.alloc(uncached_len)
        if token_slots is None:
            return None


        num_retrieved = load_fn(token_slots.to(torch.int64))

        if num_retrieved <= 0:
            self.token_to_kv_pool_allocator.free(token_slots)
            return None


        if num_retrieved < uncached_len:
            self.token_to_kv_pool_allocator.free(token_slots[num_retrieved:])
            fetched_slots = token_slots[:num_retrieved]
        else:
            fetched_slots = token_slots

        new_node = TreeNode(priority=last_node.priority)
        start = value_numel
        end = start + num_retrieved
        new_node.key = key[start:end]
        new_node.value = fetched_slots
        new_node.parent = last_node
        last_node.children[new_node.key.child_key(self.page_size)] = new_node
        self.evictable_size_ += num_retrieved
        self._update_leaf_status(last_node)
        self._update_leaf_status(new_node)

        self.kv_events.record_store(new_node.parent)
        self.kv_events.record_store(new_node)

        return fetched_slots, new_node


    def cache_finished_req(  # type: ignore[override]
        self, req: Req, is_insert: bool = True, *, kv_len_to_handle: int
    ) -> None:

        super().cache_finished_req(
            req, is_insert=is_insert, kv_len_to_handle=kv_len_to_handle
        )
        if not is_insert:
            self._load_markers.pop(req.rid, None)
            return


        topk = get_spec().speculative_eagle_topk
        enable_kv_committed_len = topk is None or topk == 1
        if enable_kv_committed_len:
            kv_committed_len = req.kv_committed_len
        else:
            kv_committed_len = len(req.origin_input_ids) + max(
                len(req.output_ids) - 1, 0
            )

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        if not token_ids:
            return
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_committed_len
        ]


        match_result = super().match_prefix(
            MatchPrefixParams(
                key=RadixKey(
                    token_ids,
                    req.extra_key,
                    cache_salt=req.cache_salt,
                )
            )
        )
        new_last_node = match_result.last_device_node
        if new_last_node is None:
            return

        self.inc_lock_ref(new_last_node)
        try:
            with torch.cuda.stream(self.store_stream):
                fkv_task_id = self.flexkv_connector.store_kv(
                    rid=req.rid,
                    token_ids=list(token_ids),
                    kv_indices=kv_indices,
                )
        except Exception:
            self.dec_lock_ref(new_last_node)
            raise

        if fkv_task_id < 0:


            self.dec_lock_ref(new_last_node)
            return

        with self._node_lock:
            self._inflight_store_nodes[req.rid] = new_last_node


    def evict(self, params: EvictParams) -> EvictResult:  # type: ignore[override]

        if self.disable:
            return EvictResult()
        self._drain_completed_stores()


        self.store_stream.synchronize()
        return super().evict(params)

    def check_hicache_events(self) -> None:  # type: ignore[override]

        self._drain_completed_stores()
        self.flexkv_connector.drain_launched_loads()

    def _drain_completed_stores(self) -> None:
        completed_rids = self.flexkv_connector.check_completed_stores()
        if not completed_rids:
            return
        with self._node_lock:
            for rid in completed_rids:
                node = self._inflight_store_nodes.pop(rid, None)
                if node is not None:
                    self.dec_lock_ref(node)


    def release_aborted_request(self, rid: str) -> None:

        self._load_markers.pop(rid, None)
        with self._node_lock:
            node = self._inflight_store_nodes.pop(rid, None)
        if node is not None:
            self.dec_lock_ref(node)
        self.flexkv_connector.release_pending(rid)
        self.flexkv_connector.cancel_prefetch(rid)

    def prefetch_from_storage(
        self, rid: str, last_host_node: TreeNode, token_ids
    ) -> None:

        try:
            self.flexkv_connector.prefetch_async(rid, list(token_ids))
        except Exception as exc:
            logger.debug("[FlexKV] prefetch_from_storage: %s", exc)

    def check_prefetch_progress(self, rid: str) -> bool:
        return self.flexkv_connector.check_prefetch_progress(rid)

    def terminate_prefetch(self, rid: str) -> None:
        self.flexkv_connector.cancel_prefetch(rid)

    def pop_prefetch_loaded_tokens(self, rid: str) -> int:

        return 0

    @property
    def hicache_storage_pass_prefix_keys(self) -> bool:


        return False
