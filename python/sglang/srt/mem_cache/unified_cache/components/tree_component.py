from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum, IntFlag
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

import torch
from numpy import float64

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import PoolTransfer, PoolTransferResult
from sglang.srt.mem_cache.unified_cache.component_type import (
    BASE_COMPONENT_TYPE,
    ComponentType,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_cache.cache_action import (
        CacheAction,
        ComponentAction,
    )
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
    from sglang.srt.mem_cache.unified_radix_cache import (
        NodeId,
        UnifiedRadixCache,
        UnifiedTreeNode,
    )


_NUM_COMPONENT_TYPES = len(ComponentType)

_LAST_ACCESS_TIME_COUNTER_FLOAT = float64(1.0)
_COMPONENT_UUID_COUNTER = 1


@dataclasses.dataclass
class ComponentData:
    value: Optional[torch.Tensor] = None
    lock_ref: int = 0
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    host_value: Optional[torch.Tensor] = None
    host_lock_ref: int = 0
    session_ref: int = 0
    session_ids: Optional[set[str]] = None


class EvictLayer(IntFlag):


    DEVICE = 1
    HOST = 2
    ALL = DEVICE | HOST


@dataclasses.dataclass(frozen=True)
class PrepareLoadBackResult:


    allocated_mamba_slot: Optional[torch.Tensor] = None


@dataclasses.dataclass(frozen=True)
class PreparePrefetchResult:


    alloc_failed: bool = False

    host_indices: Optional[torch.Tensor] = None


class CacheTransferPhase(str, Enum):

    BACKUP_HOST = "backup_host"
    LOAD_BACK = "load_back"
    BACKUP_STORAGE = "backup_storage"
    PREFETCH = "prefetch"


class LRURefreshPhase(str, Enum):

    WALKDOWN = "walkdown"
    MATCH_END = "match_end"
    INSERT_END = "insert_end"


def get_and_increase_time_counter() -> float64:
    global _LAST_ACCESS_TIME_COUNTER_FLOAT
    ret = _LAST_ACCESS_TIME_COUNTER_FLOAT
    _LAST_ACCESS_TIME_COUNTER_FLOAT += 1.0
    return ret


def next_component_uuid() -> int:
    global _COMPONENT_UUID_COUNTER
    _COMPONENT_UUID_COUNTER += 1
    return _COMPONENT_UUID_COUNTER


class TreeComponent(ABC):
    def __init__(self, cache: UnifiedRadixCache, params: CacheInitParams):
        self.cache = cache

        self.tree_core: Optional[UnifiedTreeCore] = None
        self.is_evict_device_ongoing = False


        self._session_leaves: dict[str, set[UnifiedTreeNode]] = defaultdict(set)


    component_type: ComponentType

    def reset_session_state(self) -> None:
        self._session_leaves = defaultdict(set)

    def session_ref(self, node: UnifiedTreeNode) -> int:
        return node.component_data[self.component_type].session_ref

    def _can_evict_leaf_atomically(self, node: UnifiedTreeNode) -> bool:


        if self.session_ref(node) > 0:
            return True
        priority = self.eviction_priority(is_leaf=False)
        return not any(
            comp.eviction_priority(is_leaf=False) >= priority
            and comp.session_ref(node) > 0
            for comp in self.tree_core.components
        )

    def _refresh_session_partition(self, node: UnifiedTreeNode) -> None:
        ct = self.component_type
        lru = self.tree_core.lru_lists[ct]
        if lru.in_list(node):
            lru.reset_node_mru(node)
        host_lru = self.tree_core.host_lru_lists[ct]
        if host_lru.in_list(node):
            host_lru.reset_node_mru(node)

    def _find_reusable_session_leaf(
        self, node: Optional[UnifiedTreeNode]
    ) -> Optional[UnifiedTreeNode]:
        root_node = self.tree_core.root_node
        if node is None or node is root_node:
            return None

        validator = self.create_match_validator(match_device_only=False)
        cur = node
        while cur is not None and cur is not root_node:
            if validator(cur):
                return cur
            cur = cur.parent
        return None

    def resolve_session_leaf(
        self, req: Req, last_node: Optional[UnifiedTreeNode]
    ) -> Optional[UnifiedTreeNode]:
        return self._find_reusable_session_leaf(last_node)

    def _mark_session_leaf(self, session_id: str, leaf: UnifiedTreeNode) -> None:
        self._session_leaves[session_id].add(leaf)
        cd = leaf.component_data[self.component_type]
        if cd.session_ids is None:
            cd.session_ids = set()
        cd.session_ids.add(session_id)

    def _unmark_session_leaf(self, session_id: str, leaf: UnifiedTreeNode) -> None:
        leaves = self._session_leaves.get(session_id)
        assert leaves is not None and leaf in leaves
        leaves.remove(leaf)
        cd = leaf.component_data[self.component_type]
        assert cd.session_ids is not None and session_id in cd.session_ids
        cd.session_ids.remove(session_id)
        if not cd.session_ids:
            cd.session_ids = None
        if not leaves:
            self._session_leaves.pop(session_id, None)

    def _nearest_session_ancestor(
        self, new_leaf: UnifiedTreeNode, current_leaves: set[UnifiedTreeNode]
    ) -> Optional[UnifiedTreeNode]:
        cur = new_leaf.parent
        while cur is not None and cur is not self.tree_core.root_node:
            if cur in current_leaves:
                return cur
            cur = cur.parent
        return None

    @abstractmethod
    def _dec_session_coverage(self, session_id: str, leaf: UnifiedTreeNode) -> None:

        ...

    @abstractmethod
    def _advance_session_coverage(
        self,
        session_id: str,
        leaf: UnifiedTreeNode,
        old_ancestor: Optional[UnifiedTreeNode],
    ) -> None:

        ...

    @abstractmethod
    def _recede_session_coverage(
        self,
        session_id: str,
        leaf: UnifiedTreeNode,
        fallback: Optional[UnifiedTreeNode],
    ) -> None:

        ...

    def register_session_leaf(
        self, session_id: str, leaf: Optional[UnifiedTreeNode]
    ) -> None:
        if (
            not self.tree_core.enable_session_radix_cache
            or leaf is None
            or leaf is self.tree_core.root_node
        ):
            return

        current_leaves = self._session_leaves[session_id]
        if leaf in current_leaves:
            return

        old_ancestor = self._nearest_session_ancestor(leaf, current_leaves)
        self._advance_session_coverage(session_id, leaf, old_ancestor)
        self._mark_session_leaf(session_id, leaf)
        if old_ancestor is not None:
            self._unmark_session_leaf(session_id, old_ancestor)

    def release_session(self, session_id: str) -> int:
        leaves = tuple(self._session_leaves.get(session_id, ()))
        for leaf in leaves:
            self._dec_session_coverage(session_id, leaf)
            self._unmark_session_leaf(session_id, leaf)
        return len(leaves)

    def discard_deleted_session_leaf(self, node: UnifiedTreeNode) -> None:
        cd = node.component_data[self.component_type]
        for session_id in tuple(cd.session_ids or ()):
            leaves = self._session_leaves.get(session_id)
            assert leaves is not None and node in leaves
            fallback = self._find_reusable_session_leaf(node.parent)
            if fallback is not None and fallback in self._session_leaves.get(
                session_id, ()
            ):
                fallback = None
            self._recede_session_coverage(session_id, node, fallback)
            self._unmark_session_leaf(session_id, node)
            if fallback is not None:
                self._mark_session_leaf(session_id, fallback)

    def validate_session_state(
        self,
        reachable_nodes: set[UnifiedTreeNode],
        report_error: Callable[[str], None],
    ) -> None:
        reachable_nodes = set(reachable_nodes)
        ct = self.component_type

        for session_id, leaves in self._session_leaves.items():
            if not leaves:
                report_error(f"{ct} session {session_id!r} has an empty leaf index")
            for leaf in leaves:
                if leaf not in reachable_nodes:
                    report_error(
                        f"{ct} session {session_id!r} indexes unreachable leaf {leaf.id}"
                    )
                if session_id not in (leaf.component_data[ct].session_ids or ()):
                    report_error(
                        f"{ct} session {session_id!r} leaf {leaf.id} is missing its marker"
                    )

        for node in reachable_nodes:
            cd = node.component_data[ct]
            if cd.session_ref < 0:
                report_error(f"node {node.id} {ct} session_ref={cd.session_ref}")
            if cd.session_ids is not None and not cd.session_ids:
                report_error(f"node {node.id} {ct} has an empty session_ids marker")
            for session_id in cd.session_ids or ():
                if node not in self._session_leaves.get(session_id, ()):
                    report_error(
                        f"node {node.id} {ct} session {session_id!r} marker is not indexed"
                    )

    def node_has_component_data(
        self, node: UnifiedTreeNode, target: EvictLayer = EvictLayer.DEVICE
    ) -> bool:
        cd = node.component_data[self.component_type]
        if target is EvictLayer.DEVICE:
            return cd.value is not None
        return cd.host_value is not None

    def value_len(self, node: UnifiedTreeNode) -> int:
        value = node.component_data[self.component_type].value
        return len(value) if value is not None else 0

    def has_host_value_only(self, node: UnifiedTreeNode) -> bool:

        cd = node.component_data[self.component_type]
        return cd.value is None and cd.host_value is not None

    def needs_incremental_backup(self, node: UnifiedTreeNode) -> bool:

        return False

    def refresh_lru(
        self,
        phase: LRURefreshPhase,
        node: UnifiedTreeNode,
        root_node: UnifiedTreeNode,
    ) -> None:
        ct = self.component_type
        match phase:
            case LRURefreshPhase.WALKDOWN:
                if node.component_data[ct].value is None:
                    return
                self.tree_core.lru_lists[ct].reset_node_mru(node)
            case LRURefreshPhase.MATCH_END:
                self.tree_core.lru_lists[ct].reset_node_and_parents_mru(
                    node, root_node, self.node_has_component_data
                )
            case LRURefreshPhase.INSERT_END:


                return
            case _:
                raise ValueError(f"Unknown LRURefreshPhase: {phase}")

    @abstractmethod
    def create_match_validator(
        self, match_device_only: bool = False
    ) -> Callable[[UnifiedTreeNode], bool]:

        ...

    def finalize_match_result_in_tree_core(
        self,
        result: MatchResult,
        params: MatchPrefixParams,
        value_chunks: list[torch.Tensor],
        best_value_len: int,
    ) -> MatchResult:

        return result

    def finalize_match_result_in_cache(
        self, params: MatchPrefixParams, result: MatchResult
    ) -> MatchResult:

        return result

    def update_component_on_insert_overlap(
        self,
        node: UnifiedTreeNode,
        prefix_len: int,
        total_prefix_len: int,
        value_slice: torch.Tensor,
        params: InsertParams,
        cache_actions: list[CacheAction | ComponentAction],
    ) -> int:

        return prefix_len

    def recover_after_unevict(
        self,
        node: UnifiedTreeNode,
        prefix_len: int,
        total_prefix_len: int,
        params: InsertParams,
        cache_actions: list[CacheAction | ComponentAction],
    ) -> None:

        return None

    def commit_insert_component_data(
        self,
        node: UnifiedTreeNode,
        is_new_leaf: bool,
        params: InsertParams,
        result: InsertResult,
        cache_actions: list[CacheAction | ComponentAction],
    ) -> None:

        pass

    @abstractmethod
    def redistribute_on_node_split(
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode
    ):

        ...

    @abstractmethod
    def evict_component(
        self,
        node: UnifiedTreeNode,
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
        target: EvictLayer = EvictLayer.DEVICE,
    ) -> tuple[int, int]:

        ...

    def eviction_priority(self, is_leaf: bool) -> int:

        return 0

    def evict_device_start(self, request_cnt: int) -> None:

        assert (
            not self.is_evict_device_ongoing
        ), f"{self.component_type} device eviction already in progress"
        self._evict_device_start(request_cnt)
        self.is_evict_device_ongoing = True

    def evict_device_next_node(
        self,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> Optional[NodeId]:

        assert (
            self.is_evict_device_ongoing
        ), f"{self.component_type} device eviction not started"
        return self._evict_device_next_node(tracker, device_frees, host_frees)

    def evict_device_end(self) -> None:

        assert (
            self.is_evict_device_ongoing
        ), f"{self.component_type} device eviction not started"
        self._evict_device_end()
        self.is_evict_device_ongoing = False

    @abstractmethod
    def _evict_device_start(self, request_cnt: int) -> None:

        ...

    @abstractmethod
    def _evict_device_next_node(
        self,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> Optional[NodeId]:

        ...

    @abstractmethod
    def _evict_device_end(self) -> None:

        ...

    @abstractmethod
    def acquire_component_lock(
        self,
        node: UnifiedTreeNode,
        result: IncLockRefResult,
        lock_host: bool = False,
    ) -> IncLockRefResult:

        ...

    @abstractmethod
    def release_component_lock(
        self,
        node: UnifiedTreeNode,
        params: Optional[DecLockRefParams],
        lock_host: bool = False,
    ) -> None:

        ...

    def prepare_for_caching_req(
        self,
        req: Req,
        insert_params: InsertParams,
        token_ids_len: int,
        is_finished: bool,
    ) -> Optional[int]:

        return None

    def cleanup_after_caching_req(
        self,
        req: Req,
        is_finished: bool,
        insert_result: Optional[InsertResult] = None,
        insert_params: Optional[InsertParams] = None,
    ) -> None:

        pass

    def free_out_of_window_slots(
        self, req: Req, pre_len: int, insert_params: InsertParams
    ) -> None:
        pass

    def free_host_values(self, host_values: list[torch.Tensor]) -> None:

        raise NotImplementedError(f"{self.component_type} must free its host values")


    def prepare_load_back(
        self,
        node_id: NodeId,
        *,
        req: Optional[Req] = None,
    ) -> PrepareLoadBackResult:

        return PrepareLoadBackResult()

    def finalize_load_back(
        self, req: Optional[Req], prep: PrepareLoadBackResult, success: bool
    ) -> None:

        pass

    def prepare_prefetch(
        self,
        node_id: NodeId,
        *,
        prefetch_tokens: int = 0,
    ) -> PreparePrefetchResult:

        return PreparePrefetchResult()

    def build_hicache_transfers(
        self,
        node: UnifiedTreeNode,
        phase: CacheTransferPhase,
        *,
        mamba_pool_idx: Optional[torch.Tensor] = None,
        host_indices: Optional[torch.Tensor] = None,
        token_ids: Optional[Sequence[int]] = None,
        prefetch_tokens: int = 0,
        last_hash: Optional[str] = None,
    ) -> Optional[list[PoolTransfer]]:

        return None

    def commit_hicache_transfer(
        self,
        node: UnifiedTreeNode,
        phase: CacheTransferPhase,
        transfers: list[PoolTransfer] = (),
        *,
        cache_actions: list[CacheAction | ComponentAction],
        insert_result: Optional[InsertResult] = None,
        pool_storage_result: Optional[PoolTransferResult] = None,
    ) -> None:

        pass

    def drive_host_eviction(
        self,
        num_tokens: int,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> None:

        pass

    def apply_component_action(self, action: ComponentAction) -> None:

        raise NotImplementedError(
            f"{self.component_type} cannot apply {type(action).__name__}"
        )
