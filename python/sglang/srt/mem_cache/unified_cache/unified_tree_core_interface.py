from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING, Optional, Sequence

import msgspec


NodeId = int


class BaseEvictionResult(msgspec.Struct):


    device_frees: dict[ComponentType, list[torch.Tensor]] = msgspec.field(
        default_factory=lambda: defaultdict(list)
    )
    host_frees: dict[ComponentType, list[torch.Tensor]] = msgspec.field(
        default_factory=lambda: defaultdict(list)
    )
    tracker: dict[ComponentType, int] = msgspec.field(
        default_factory=lambda: defaultdict(int)
    )

    def __del__(self) -> None:

        assert (
            not self.device_frees and not self.host_frees
        ), "BaseEvictionResult dropped with undrained values"


class EvictDeviceNextNodeResult(BaseEvictionResult):


    node_id: Optional[NodeId] = None
    made_progress: bool = False


class EvictDeviceLeafResult(BaseEvictionResult):
    backup_kv: Optional[BackupKV] = None


class DemoteResult(BaseEvictionResult):
    pass


class DropSubtreeNoHostResult(BaseEvictionResult):
    is_dropped: bool = False


class DriveHostEvictionResult(BaseEvictionResult):
    pass


class DecSwaLockOnlyResult(BaseEvictionResult):
    pass


class RadixCacheWalkResult(msgspec.Struct, frozen=True, kw_only=True):


    slot_indices: torch.Tensor
    positions: torch.Tensor
    prev_slot_indices: torch.Tensor


class InsertStepResult(msgspec.Struct, frozen=True):


    actions: list[CacheAction | ComponentAction]
    result: Optional[InsertResult] = None


if TYPE_CHECKING:
    import torch

    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.base_prefix_cache import (
        DecLockRefParams,
        DecLockRefResult,
        IncLockRefResult,
        InsertParams,
        InsertResult,
        MatchPrefixParams,
        MatchResult,
    )
    from sglang.srt.mem_cache.events import KVCacheEventRecorder
    from sglang.srt.mem_cache.hicache_storage import PoolTransfer, PoolTransferResult
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache.cache_action import (
        BackupKV,
        CacheAction,
        ComponentAction,
    )
    from sglang.srt.mem_cache.unified_cache.components import (
        CacheTransferPhase,
        ComponentType,
    )
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import (
        StorageBackupSpec,
        UnifiedTreeNode,
    )


class UnifiedTreeCoreInterface(ABC):


    page_size: int
    is_eagle: bool
    device: torch.device
    enable_hicache: bool
    enable_storage: bool
    write_through_threshold: int
    is_write_back: bool
    has_swa_host_pool: bool
    kv_events: KVCacheEventRecorder


    def take_events(self) -> list:

        return self.kv_events.take()

    @abstractmethod
    def reset(self) -> None:

        ...

    @abstractmethod
    def node_by_id(self, node_id: NodeId) -> UnifiedTreeNode:

        ...

    @abstractmethod
    def is_backuped(self, node_id: NodeId) -> bool:

        ...

    @abstractmethod
    def is_root(self, node_id: NodeId) -> bool:

        ...

    @abstractmethod
    def get_last_hash_value(self, node_id: NodeId) -> Optional[str]:

        ...

    @abstractmethod
    def get_prefix_hash_values(self, node_id: NodeId) -> list[str]:

        ...

    @abstractmethod
    def get_hash_values(self, node_id: NodeId) -> list[str]:

        ...

    @abstractmethod
    def backfill_missing_hash_values(self) -> int:

        ...

    @abstractmethod
    def root_node_handle(self, extra_key: Optional[str] = None) -> NodeId:

        ...

    @abstractmethod
    def inc_lock_ref(
        self, node_id: NodeId, skip_lock_components: Sequence[ComponentType] = ()
    ) -> IncLockRefResult:

        ...

    @abstractmethod
    def dec_lock_ref(
        self,
        node_id: NodeId,
        params: Optional[DecLockRefParams] = None,
        skip_swa: bool = False,
    ) -> DecLockRefResult:

        ...

    @abstractmethod
    def dec_swa_lock_only(
        self,
        node_id: NodeId,
        swa_uuid_for_lock: Optional[int],
        skip_lock_node_ids: Optional[dict] = None,
    ) -> DecSwaLockOnlyResult:

        ...


    @abstractmethod
    def evict_device_start(
        self, component_type: ComponentType, request_cnt: int
    ) -> None:

        ...

    @abstractmethod
    def evict_device_next_node(
        self, component_type: ComponentType, tracker: dict[ComponentType, int]
    ) -> EvictDeviceNextNodeResult:

        ...

    @abstractmethod
    def evict_device_leaf(
        self, node_id: NodeId, is_write_back: bool
    ) -> EvictDeviceLeafResult:

        ...

    @abstractmethod
    def drop_subtree_no_host(self, node_id: NodeId) -> DropSubtreeNoHostResult:

        ...

    @abstractmethod
    def demote(self, node_id: NodeId) -> DemoteResult:

        ...

    @abstractmethod
    def evict_device_end(self, component_type: ComponentType) -> None:

        ...

    @abstractmethod
    def inc_host_lock_ref(self, node_id: NodeId) -> IncLockRefResult:

        ...

    @abstractmethod
    def dec_host_lock_ref(
        self, node_id: NodeId, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:

        ...

    @abstractmethod
    def evictable_size(self) -> int: ...

    @abstractmethod
    def protected_size(self) -> int: ...

    @abstractmethod
    def component_evictable_size(self, component_type: ComponentType) -> int:

        ...

    @abstractmethod
    def full_evictable_size(self) -> int: ...

    @abstractmethod
    def full_protected_size(self) -> int: ...

    @abstractmethod
    def swa_evictable_size(self) -> int: ...

    @abstractmethod
    def mamba_evictable_size(self) -> int: ...

    @abstractmethod
    def swa_protected_size(self) -> int: ...

    @abstractmethod
    def mamba_protected_size(self) -> int: ...

    @abstractmethod
    def total_size(self) -> tuple[int, int]:

        ...

    @abstractmethod
    def all_values_flatten(self) -> torch.Tensor: ...

    @abstractmethod
    def all_mamba_values_flatten(self) -> torch.Tensor: ...

    @abstractmethod
    def walk_for_kv_canary(
        self, unlocked_only: bool, swa_resident_only: bool
    ) -> RadixCacheWalkResult:

        ...

    @abstractmethod
    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:

        ...

    @property
    @abstractmethod
    def empty_match_result(self) -> MatchResult:

        ...

    @abstractmethod
    def is_full_device_evicted(self, node_id: NodeId) -> bool:

        ...

    @abstractmethod
    def collect_full_device_indices(
        self, from_node_id: NodeId, until_node_id: NodeId
    ) -> torch.Tensor:

        ...

    @abstractmethod
    def begin_insert(self, params: InsertParams) -> InsertStepResult:

        ...

    @abstractmethod
    def resume_insert(self) -> InsertStepResult:

        ...

    @abstractmethod
    def has_ongoing_insert(self) -> bool:

        ...

    @abstractmethod
    def end_insert(self) -> list[CacheAction | ComponentAction]:

        ...

    @abstractmethod
    def drive_host_eviction(
        self, component_type: ComponentType, num_tokens: int
    ) -> DriveHostEvictionResult:

        ...

    @abstractmethod
    def evict_excess_path_states(
        self,
        tail_node_id: NodeId,
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> None:

        ...


    @abstractmethod
    def set_hicache_enabled(self) -> None:

        ...

    @abstractmethod
    def insert_host(
        self,
        node_id: NodeId,
        key: RadixKey,
        host_value: torch.Tensor,
        hash_value: list[str],
    ) -> InsertResult:

        ...

    @abstractmethod
    def build_backup_spec(
        self, node_id: NodeId
    ) -> tuple[torch.Tensor, dict[ComponentType, list[PoolTransfer]]]:

        ...

    @abstractmethod
    def build_storage_backup_spec(
        self, node_id: NodeId, pass_prefix_keys: bool
    ) -> Optional[StorageBackupSpec]:

        ...

    @abstractmethod
    def build_hicache_transfers(
        self,
        component_type: ComponentType,
        node_id: NodeId,
        phase: CacheTransferPhase,
        *,
        host_indices: Optional[torch.Tensor] = None,
        token_ids: Optional[Sequence[int]] = None,
        prefetch_tokens: int = 0,
        last_hash: Optional[str] = None,
    ) -> Optional[list[PoolTransfer]]:

        ...

    @abstractmethod
    def build_load_back_spec(
        self, node_id: NodeId, req: Optional[Req] = None
    ) -> tuple[PoolTransfer, dict[ComponentType, list[PoolTransfer]]]:

        ...

    @abstractmethod
    def prefetch_anchor_info(
        self, node_id: NodeId
    ) -> tuple[Optional[str], Optional[str]]:

        ...

    @abstractmethod
    def commit_hicache_transfers(
        self,
        node_id: NodeId,
        phase: CacheTransferPhase,
        comp_xfers: dict[ComponentType, list[PoolTransfer]],
        *,
        cache_actions: list[CacheAction | ComponentAction],
        insert_result: Optional[InsertResult] = None,
        pool_storage_result: Optional[PoolTransferResult] = None,
    ) -> None:

        ...

    @abstractmethod
    def commit_backup(
        self,
        node_id: NodeId,
        host_indices: torch.Tensor,
        comp_xfers: dict[ComponentType, list[PoolTransfer]],
    ) -> None:

        ...

    @abstractmethod
    def commit_load_back(
        self,
        node_id: NodeId,
        device_indices: torch.Tensor,
        kv_xfer: PoolTransfer,
        comp_xfers: dict[ComponentType, list[PoolTransfer]],
    ) -> list[CacheAction | ComponentAction]:

        ...

    @abstractmethod
    def finish_load_back(self, anchor_node_id: NodeId) -> None:

        ...


    write_back_duplicate_reclaim_digest: int = 0

    @abstractmethod
    def mark_write_through_pending(self, node_id: NodeId) -> None:

        ...

    @abstractmethod
    def finish_write_through(self, node_ids: list[NodeId], ack_id: int) -> None:

        ...

    @abstractmethod
    def set_component_device_value(
        self, node_id: NodeId, component_type: ComponentType, value: torch.Tensor
    ) -> None:

        ...

    @abstractmethod
    def get_component_device_value(
        self, node_id: NodeId, component_type: ComponentType
    ) -> Optional[torch.Tensor]:

        ...

    @abstractmethod
    def component_has_host_value_only(
        self, node_id: NodeId, component_type: ComponentType
    ) -> bool:

        ...


    @abstractmethod
    def sanity_check(
        self,
        ongoing_write_through: list[tuple[int, NodeId]],
        ongoing_load_back: list[tuple[int, NodeId]],
    ) -> None:

        ...

    @abstractmethod
    def pretty_print(self) -> None:

        ...
