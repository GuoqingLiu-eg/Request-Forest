from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec

from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

if TYPE_CHECKING:
    import torch

    from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import NodeId


class ReplaceWriteThroughOnNodeSplit(msgspec.Struct, frozen=True):


    ack_id: int
    old_node_id: NodeId
    new_node_id: NodeId
    new_child_node_id: NodeId


class FreeDeviceKV(msgspec.Struct, frozen=True):


    indices: list[torch.Tensor]


class ComponentAction(msgspec.Struct, frozen=True):
    pass


class FreeComponentDeviceSlot(ComponentAction, frozen=True):


    indices: list[torch.Tensor]
    component_type: ComponentType


class FreeComponentHostSlot(ComponentAction, frozen=True):


    host_indices: list[torch.Tensor]
    component_type: ComponentType


class BackupKV(msgspec.Struct, frozen=True):


    node_ids: list[NodeId]


class MambaEvictExcessPathStates(ComponentAction, frozen=True):


    tail_node_id: NodeId
    component_type: ComponentType = ComponentType.MAMBA


class RebuildFullToSWAMapping(ComponentAction, frozen=True):


    full_indices: list[torch.Tensor]
    swa_indices: list[torch.Tensor]
    component_type: ComponentType = ComponentType.SWA


class RecoverSWAWithLockedFull(ComponentAction, frozen=True):


    node_id: NodeId
    kept_full: torch.Tensor
    incoming_full: torch.Tensor
    component_type: ComponentType = ComponentType.SWA


class SWARebuild(ComponentAction, frozen=True):


    node_id: NodeId
    source_value: torch.Tensor
    component_type: ComponentType = ComponentType.SWA


CacheAction = ReplaceWriteThroughOnNodeSplit | FreeDeviceKV | BackupKV
