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
from sglang.srt.runtime_context import get_memory, get_spec
from sglang.srt.utils import create_device_stream, device_stream_context

try:
    from lmcache.integration.sglang.multi_process_adapter import LMCacheMPConnector
    from lmcache.integration.sglang.sglang_adapter import (
        LMCacheLayerwiseConnector,
        LoadMetadata,
        StoreMetadata,
    )
    from lmcache.integration.sglang.utils import lmcache_get_config
except ImportError as e:
    raise RuntimeError(
        "LMCache is not installed. Please install it by running `pip install lmcache`"
    ) from e


if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams

logger = logging.getLogger(__name__)


@dataclass
class _LMCacheLoadBackMarker:


    key: RadixKey

    value_numel: int


class LMCacheMode(enum.Enum):
    MP = enum.auto()
    IP = enum.auto()


class LayerTransferCounter:


    def __init__(
        self,
        num_layers: int,
        load_stream: torch.Stream,
        lmc_connector: LMCacheLayerwiseConnector,
        printable: bool = False,
    ):
        self.num_layers = num_layers
        self.load_stream = load_stream
        self.lmc_connector = lmc_connector

    def wait_until(self, layer_id: int):

        self.load_stream.synchronize()
        with device_stream_context(self.load_stream):
            self.lmc_connector.load_kv_layerwise(layer_id)


class LMCRadixCache(RadixCache):


    def __init__(
        self,
        params: CacheInitParams,
        model_config: Optional[ModelConfig] = None,
        tp_size: int = 1,
        rank: int = 0,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(params)

        cli_lmc_cfg = get_memory().lmcache_config_file or ""

        kvcache = self.token_to_kv_pool_allocator.get_kvcache()
        connector_kwargs = dict(
            sgl_config=model_config,
            tp_size=tp_size,
            rank=rank,


            k_pool=getattr(
                kvcache,
                "k_buffer",
                getattr(self.token_to_kv_pool_allocator._kvcache, "k_buffer"),
            ),
            v_pool=getattr(
                kvcache,
                "v_buffer",
                getattr(self.token_to_kv_pool_allocator._kvcache, "v_buffer"),
            ),
            tp_group=tp_group.device_group if tp_group is not None else None,
        )

        self.load_stream = create_device_stream(self.device)
        self.store_stream = create_device_stream(self.device)


        self._mode = LMCacheMode.IP if self.device.type == "xpu" else LMCacheMode.MP
        if self._mode is LMCacheMode.MP:
            if not cli_lmc_cfg:
                raise ValueError(
                    "MP mode requires --lmcache-config-file (the YAML "
                    "supplies mp_host / mp_port)."
                )
            lm_cfg = lmcache_get_config(cli_lmc_cfg)
            self.lmcache_connector = LMCacheMPConnector(
                page_size=params.page_size,
                host=lm_cfg.mp_host,
                port=lm_cfg.mp_port,
                **connector_kwargs,
            )
        elif self._mode is LMCacheMode.IP:
            self.lmcache_connector = LMCacheLayerwiseConnector(
                config_file=cli_lmc_cfg, **connector_kwargs
            )

            self.layer_done_executor = LayerTransferCounter(
                num_layers=(
                    model_config.num_hidden_layers if model_config is not None else 0
                ),
                load_stream=self.load_stream,
                lmc_connector=self.lmcache_connector,
            )
            kvcache.register_layer_transfer_counter(self.layer_done_executor)

        self._in_flight_nodes: list[TreeNode] = []
        self._node_lock = threading.Lock()
        self._mp_load_back_markers: dict[str, _LMCacheLoadBackMarker] = {}

    def reset(self):
        super().reset()
        if hasattr(self, "_in_flight_nodes"):
            with self._node_lock:
                self._in_flight_nodes.clear()
        if hasattr(self, "_mp_load_back_markers"):
            self._mp_load_back_markers.clear()

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:

        key = params.key
        if self.disable or not key:
            return super().match_prefix(params)

        if self.page_size != 1:
            aligned_len = len(key) // self.page_size * self.page_size
            key = key[:aligned_len]

        base_res = super().match_prefix(params)
        value: torch.Tensor = base_res.device_indices
        last_node: TreeNode = base_res.last_device_node

        if self._mode is LMCacheMode.MP:
            if params.req is None:
                return base_res
            return self._mp_match_prefix(key, base_res, value, last_node, params.req)
        elif self._mode is LMCacheMode.IP:
            return self._ip_match_prefix(key, base_res, value, last_node)
        return base_res

    def _mp_match_prefix(
        self,
        key: RadixKey,
        base_res: MatchResult,
        value: torch.Tensor,
        last_node: TreeNode,
        req: Req,
    ) -> MatchResult:

        token_ids = key.raw_token_ids()
        matched = self.lmcache_connector.lookup_kv(token_ids, req.rid)
        if matched <= value.numel():

            self.lmcache_connector.release_pending(req.rid)
            return base_res

        if token_ids is key.token_ids:
            token_ids = token_ids[:]
        self._mp_load_back_markers[req.rid] = _LMCacheLoadBackMarker(
            key=RadixKey(
                token_ids,
                key.extra_key,
                key.is_bigram,
                cache_salt=key.cache_salt,
            ),
            value_numel=int(value.numel()),
        )
        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
            best_match_node=last_node,
            host_hit_length=matched - int(value.numel()),
        )

    def _ip_match_prefix(
        self,
        key: RadixKey,
        base_res: MatchResult,
        value: torch.Tensor,
        last_node: TreeNode,
    ) -> MatchResult:

        if value.numel() == len(key):
            return base_res

        uncached_len = len(key) - value.numel()
        if uncached_len == 0:
            return base_res

        token_ids = key.raw_token_ids()
        result = self._load_back(
            key=key,
            value_numel=int(value.numel()),
            uncached_len=uncached_len,
            last_node=last_node,
            load_fn=lambda sm, pp: self._ip_load_back(
                token_ids=token_ids,
                value_numel=int(value.numel()),
                slot_mapping=sm,
                prefix_pad=pp,
            ),
        )
        if result is None:
            return base_res
        new_slots, new_node = result
        return MatchResult(
            device_indices=torch.cat([value, new_slots]),
            last_device_node=new_node,
            last_host_node=new_node,
            best_match_node=new_node,
        )

    def init_load_back(
        self, params: InitLoadBackParams
    ) -> Tuple[torch.Tensor, Optional[TreeNode]]:

        req = params.req
        marker = self._mp_load_back_markers.pop(req.rid)
        last_node: TreeNode = params.best_match_node

        result = self._load_back(
            key=marker.key,
            value_numel=marker.value_numel,
            uncached_len=params.host_hit_length,
            last_node=last_node,
            load_fn=lambda sm, pp: self._mp_load_back(
                marker=marker,
                request_id=req.rid,
                slot_mapping=sm,
                prefix_pad=pp,
            ),
        )
        if result is None:


            self.lmcache_connector.release_pending(req.rid)
            return (
                torch.empty((0,), dtype=torch.int64, device=self.device),
                last_node,
            )
        return result

    def _load_back(
        self,
        *,
        key: RadixKey,
        value_numel: int,
        uncached_len: int,
        last_node: TreeNode,
        load_fn,
    ) -> Optional[Tuple[torch.Tensor, TreeNode]]:

        chunk_size = self.lmcache_connector.chunk_size()
        prefix_pad = value_numel % chunk_size

        if self.token_to_kv_pool_allocator.available_size() < uncached_len:
            self.evict(EvictParams(num_tokens=uncached_len))

        token_slots = self.token_to_kv_pool_allocator.alloc(uncached_len)
        if token_slots is None:
            return None

        slot_mapping = torch.empty(
            value_numel + token_slots.numel(),
            dtype=torch.int64,
            device=self.device,
        )
        slot_mapping[:value_numel].fill_(-1)
        slot_mapping[value_numel:].copy_(token_slots)


        num_retrieved = load_fn(slot_mapping, prefix_pad)
        logger.debug("num_retrieved_tokens: %s", num_retrieved)

        if num_retrieved > 0:
            self.token_to_kv_pool_allocator.free(
                token_slots[(num_retrieved - prefix_pad) :]
            )
        else:
            self.token_to_kv_pool_allocator.free(token_slots)

        if num_retrieved > 0:
            fetched = num_retrieved - prefix_pad
            new_node = TreeNode(priority=last_node.priority)
            start = value_numel
            end = start + fetched
            new_node.key = key[start:end]
            new_node.value = token_slots[:fetched]
            new_node.parent = last_node
            last_node.children[new_node.key.child_key(self.page_size)] = new_node
            self.evictable_size_ += fetched
            self._update_leaf_status(last_node)
            self._update_leaf_status(new_node)

            self.kv_events.record_store(new_node.parent)
            self.kv_events.record_store(new_node)

            return token_slots[:fetched], new_node

        return None

    def _mp_load_back(
        self,
        *,
        marker: _LMCacheLoadBackMarker,
        request_id: str,
        slot_mapping: torch.Tensor,
        prefix_pad: int,
    ) -> int:

        current_stream = torch.get_device_module(self.device).current_stream()
        self.load_stream.wait_stream(current_stream)
        with device_stream_context(self.load_stream):
            n = self.lmcache_connector.retrieve_kv(
                LoadMetadata(
                    token_ids=marker.key.token_ids,
                    slot_mapping=slot_mapping,
                    offset=marker.value_numel - prefix_pad,
                    prefix_pad=prefix_pad,
                    request_id=request_id,
                )
            )
        current_stream.wait_stream(self.load_stream)
        return n

    def _ip_load_back(
        self,
        *,
        token_ids: list[int],
        value_numel: int,
        slot_mapping: torch.Tensor,
        prefix_pad: int,
    ) -> int:

        with device_stream_context(self.load_stream):
            return self.lmcache_connector.start_load_kv(
                LoadMetadata(
                    token_ids=token_ids,
                    slot_mapping=slot_mapping,
                    offset=value_numel - prefix_pad,
                )
            )

    def cache_finished_req(
        self, req: Req, is_insert: bool = True, *, kv_len_to_handle: int
    ) -> None:


        super().cache_finished_req(
            req, is_insert=is_insert, kv_len_to_handle=kv_len_to_handle
        )
        if not is_insert:
            if self._mode is LMCacheMode.MP:
                self._mp_load_back_markers.pop(req.rid, None)
                self.lmcache_connector.end_session(req.rid)
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
        assert new_last_node is not None

        self.inc_lock_ref(new_last_node)
        store_md = StoreMetadata(
            last_node=new_last_node,
            token_ids=token_ids,
            kv_indices=kv_indices,
            offset=0,
            request_id=req.rid,
        )
        if self._mode is LMCacheMode.MP:
            self.lmcache_connector.store_kv(store_md)

            self._mp_load_back_markers.pop(req.rid, None)
            self.dec_lock_ref(new_last_node)
            self.lmcache_connector.end_session(req.rid)
        elif self._mode is LMCacheMode.IP:
            with device_stream_context(self.store_stream):
                self.lmcache_connector.store_kv(store_md)

            with self._node_lock:
                self._in_flight_nodes.append(new_last_node)

    def evict(self, params: EvictParams) -> EvictResult:

        if self.disable:
            return EvictResult()

        self.store_stream.synchronize()
        with self._node_lock:
            for node in self._in_flight_nodes:
                self.dec_lock_ref(node)
            self._in_flight_nodes.clear()

        return super().evict(params)

    def pretty_print(self):
        super().pretty_print()
        try:
            logger.debug(
                "evictable=%d protected=%d", self.evictable_size_, self.protected_size_
            )
        except Exception:
            pass
