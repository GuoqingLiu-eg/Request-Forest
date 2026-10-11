from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .commit import CommitResult, TargetDecision, TerminationReason
from .ir import ForestIR, ForestValidationError, RequestRef, _require_plain_int
from .plans import RequestAnchor, VerifyPlan

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = None


_INT64 = "int64"
_FLOAT64 = "float64"
_BOOL = "bool"

_TERMINATION_REASONS = (
    TerminationReason.TARGET_MISS,
    TerminationReason.LEAF_EXHAUSTED,
    TerminationReason.BUDGET_BOUNDARY,
    TerminationReason.NO_CANDIDATES,
    TerminationReason.EOS,
)
_TERMINATION_CODES = {
    reason: code for code, reason in enumerate(_TERMINATION_REASONS)
}
_NATIVE_OP_CACHE: dict[str, Any] = {}


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError(
            "Request Forest native adapters require the optional torch dependency"
        )
    return torch


def _torch_dtype(name: str) -> Any:
    torch_module = _require_torch()
    return getattr(torch_module, name)


def _require_cpu_tensor(
    name: str,
    value: Tensor,
    *,
    dtype: str,
    dimensions: int = 1,
) -> Tensor:
    torch_module = _require_torch()
    if not isinstance(value, torch_module.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor")
    if value.dtype != _torch_dtype(dtype):
        raise TypeError(f"{name} must have dtype torch.{dtype}")
    if value.dim() != dimensions:
        raise ValueError(f"{name} must be {dimensions}-dimensional")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return value


def _native_op(name: str) -> Any:
    cached = _NATIVE_OP_CACHE.get(name)
    if cached is not None:
        return cached
    torch_module = _require_torch()
    try:
        operation = getattr(torch_module.ops.sgl_kernel, name)
    except AttributeError as exc:
        raise RuntimeError(
            f"torch.ops.sgl_kernel.{name} is unavailable; load the SGLang "
            "native CPU extension before using Request Forest native adapters"
        ) from exc
    _NATIVE_OP_CACHE[name] = operation
    return operation


def _int64_tensor(values: Sequence[int]) -> Tensor:
    return _require_torch().tensor(values, dtype=_torch_dtype(_INT64))


def _bool_tensor(values: Sequence[bool]) -> Tensor:
    return _require_torch().tensor(values, dtype=_torch_dtype(_BOOL))


def _ordered_anchors(
    forest: ForestIR, anchors: Sequence[RequestAnchor]
) -> tuple[RequestAnchor, ...]:
    anchor_tuple = tuple(anchors)
    if any(not isinstance(anchor, RequestAnchor) for anchor in anchor_tuple):
        raise ForestValidationError("anchors must contain RequestAnchor values")
    if len(anchor_tuple) != len(forest.requests):
        raise ForestValidationError(
            "anchors must contain exactly one entry for every ForestIR request"
        )

    anchor_by_request: dict[RequestRef, RequestAnchor] = {}
    for anchor in anchor_tuple:
        if anchor.request in anchor_by_request:
            raise ForestValidationError(
                f"duplicate anchor for request {anchor.request!r}"
            )
        anchor_by_request[anchor.request] = anchor
    if set(anchor_by_request) != set(forest.requests):
        raise ForestValidationError(
            "anchor request references do not match ForestIR requests"
        )
    return tuple(anchor_by_request[request] for request in forest.requests)


@dataclass(frozen=True, slots=True, kw_only=True)
class ForestTensorView:


    request_refs: tuple[RequestRef, ...]
    node_parent_ids: Tensor
    node_request_slots: Tensor
    node_token_ids: Tensor
    node_depths: Tensor
    anchor_token_ids: Tensor
    anchor_positions: Tensor

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_refs", tuple(self.request_refs))
        node_tensors = (
            ("node_parent_ids", self.node_parent_ids),
            ("node_request_slots", self.node_request_slots),
            ("node_token_ids", self.node_token_ids),
            ("node_depths", self.node_depths),
        )
        for name, value in node_tensors:
            _require_cpu_tensor(name, value, dtype=_INT64)
        node_count = self.node_parent_ids.numel()
        if any(value.numel() != node_count for _, value in node_tensors[1:]):
            raise ValueError("ForestTensorView node tensors must have equal length")

        _require_cpu_tensor(
            "anchor_token_ids", self.anchor_token_ids, dtype=_INT64
        )
        _require_cpu_tensor(
            "anchor_positions", self.anchor_positions, dtype=_INT64
        )
        request_count = len(self.request_refs)
        if request_count == 0:
            raise ForestValidationError(
                "ForestTensorView requires at least one request"
            )
        if self.anchor_token_ids.numel() != request_count:
            raise ValueError("anchor_token_ids must have one entry per request")
        if self.anchor_positions.numel() != request_count:
            raise ValueError("anchor_positions must have one entry per request")

    @classmethod
    def from_forest(
        cls,
        forest: ForestIR,
        anchors: Sequence[RequestAnchor],
    ) -> ForestTensorView:


        if not isinstance(forest, ForestIR):
            raise ForestValidationError("forest must be a ForestIR")
        ordered_anchors = _ordered_anchors(forest, anchors)
        request_slots = {
            request: slot for slot, request in enumerate(forest.requests)
        }
        node_parent_ids: list[int] = []
        node_request_slots: list[int] = []
        node_token_ids: list[int] = []
        node_depths: list[int] = []
        for node in forest.nodes:
            node_parent_ids.append(
                -1 if node.parent_id is None else node.parent_id
            )
            node_request_slots.append(request_slots[node.request])
            node_token_ids.append(node.token_id)
            node_depths.append(node.depth)

        anchor_token_ids: list[int] = []
        anchor_positions: list[int] = []
        for anchor in ordered_anchors:
            anchor_token_ids.append(anchor.token_id)
            anchor_positions.append(anchor.position)

        return cls(
            request_refs=forest.requests,
            node_parent_ids=_int64_tensor(node_parent_ids),
            node_request_slots=_int64_tensor(node_request_slots),
            node_token_ids=_int64_tensor(node_token_ids),
            node_depths=_int64_tensor(node_depths),
            anchor_token_ids=_int64_tensor(anchor_token_ids),
            anchor_positions=_int64_tensor(anchor_positions),
        )

    from_ir = from_forest

    @property
    def node_count(self) -> int:
        return self.node_parent_ids.numel()

    @property
    def request_count(self) -> int:
        return len(self.request_refs)


class TensorForestArena:


    __slots__ = (
        "request_refs",
        "_node_parent_ids_buffer",
        "_node_request_slots_buffer",
        "_node_token_ids_buffer",
        "_node_depths_buffer",
        "_node_values_buffer",
        "_anchor_token_ids",
        "_anchor_positions",
        "_node_count",
    )

    def __init__(
        self,
        *,
        request_refs: Sequence[RequestRef],
        node_parent_ids_buffer: Tensor,
        node_request_slots_buffer: Tensor,
        node_token_ids_buffer: Tensor,
        node_depths_buffer: Tensor,
        node_values_buffer: Tensor,
        anchor_token_ids: Tensor,
        anchor_positions: Tensor,
        node_count: int,
    ) -> None:
        request_tuple = tuple(request_refs)


        ForestIR(requests=request_tuple)

        int_buffers = (
            ("node_parent_ids_buffer", node_parent_ids_buffer),
            ("node_request_slots_buffer", node_request_slots_buffer),
            ("node_token_ids_buffer", node_token_ids_buffer),
            ("node_depths_buffer", node_depths_buffer),
        )
        for name, value in int_buffers:
            _require_cpu_tensor(name, value, dtype=_INT64)
        _require_cpu_tensor(
            "node_values_buffer", node_values_buffer, dtype=_FLOAT64
        )
        capacity = node_parent_ids_buffer.numel()
        if any(value.numel() != capacity for _, value in int_buffers[1:]):
            raise ValueError("TensorForestArena node buffers must have equal length")
        if node_values_buffer.numel() != capacity:
            raise ValueError("TensorForestArena node buffers must have equal length")

        _require_cpu_tensor(
            "anchor_token_ids", anchor_token_ids, dtype=_INT64
        )
        _require_cpu_tensor(
            "anchor_positions", anchor_positions, dtype=_INT64
        )
        if anchor_token_ids.numel() != len(request_tuple):
            raise ValueError("anchor_token_ids must have one entry per request")
        if anchor_positions.numel() != len(request_tuple):
            raise ValueError("anchor_positions must have one entry per request")
        _require_plain_int("node_count", node_count)
        if node_count > capacity:
            raise ForestValidationError(
                f"node_count {node_count} exceeds arena capacity {capacity}"
            )

        self.request_refs = request_tuple
        self._node_parent_ids_buffer = node_parent_ids_buffer
        self._node_request_slots_buffer = node_request_slots_buffer
        self._node_token_ids_buffer = node_token_ids_buffer
        self._node_depths_buffer = node_depths_buffer
        self._node_values_buffer = node_values_buffer
        self._anchor_token_ids = anchor_token_ids
        self._anchor_positions = anchor_positions
        self._node_count = node_count

    @classmethod
    def empty(
        cls,
        *,
        request_refs: Sequence[RequestRef],
        anchor_token_ids: Tensor,
        anchor_positions: Tensor,
        node_capacity: int,
    ) -> TensorForestArena:


        _require_plain_int("node_capacity", node_capacity)
        torch_module = _require_torch()
        _require_cpu_tensor(
            "anchor_token_ids", anchor_token_ids, dtype=_INT64
        )
        _require_cpu_tensor(
            "anchor_positions", anchor_positions, dtype=_INT64
        )
        int_options = {"dtype": _torch_dtype(_INT64), "device": "cpu"}
        return cls(
            request_refs=request_refs,
            node_parent_ids_buffer=torch_module.empty(
                node_capacity, **int_options
            ),
            node_request_slots_buffer=torch_module.empty(
                node_capacity, **int_options
            ),
            node_token_ids_buffer=torch_module.empty(
                node_capacity, **int_options
            ),
            node_depths_buffer=torch_module.empty(
                node_capacity, **int_options
            ),
            node_values_buffer=torch_module.empty(
                node_capacity, dtype=_torch_dtype(_FLOAT64), device="cpu"
            ),


            anchor_token_ids=anchor_token_ids.clone(),
            anchor_positions=anchor_positions.clone(),
            node_count=0,
        )

    @classmethod
    def from_tensors(
        cls,
        *,
        request_refs: Sequence[RequestRef],
        node_parent_ids: Tensor,
        node_request_slots: Tensor,
        node_token_ids: Tensor,
        node_depths: Tensor,
        node_values: Tensor,
        anchor_token_ids: Tensor,
        anchor_positions: Tensor,
    ) -> TensorForestArena:


        _require_cpu_tensor(
            "node_parent_ids", node_parent_ids, dtype=_INT64
        )
        return cls(
            request_refs=request_refs,
            node_parent_ids_buffer=node_parent_ids,
            node_request_slots_buffer=node_request_slots,
            node_token_ids_buffer=node_token_ids,
            node_depths_buffer=node_depths,
            node_values_buffer=node_values,
            anchor_token_ids=anchor_token_ids,
            anchor_positions=anchor_positions,
            node_count=node_parent_ids.numel(),
        )

    @property
    def node_count(self) -> int:
        return self._node_count

    @property
    def request_count(self) -> int:
        return len(self.request_refs)

    @property
    def capacity(self) -> int:
        return self._node_parent_ids_buffer.numel()

    @staticmethod
    def _logical_prefix(buffer: Tensor, node_count: int) -> Tensor:
        if node_count == buffer.numel():
            return buffer
        return buffer.narrow(0, 0, node_count)

    @property
    def node_parent_ids(self) -> Tensor:
        return self._logical_prefix(
            self._node_parent_ids_buffer, self._node_count
        )

    @property
    def node_request_slots(self) -> Tensor:
        return self._logical_prefix(
            self._node_request_slots_buffer, self._node_count
        )

    @property
    def node_token_ids(self) -> Tensor:
        return self._logical_prefix(
            self._node_token_ids_buffer, self._node_count
        )

    @property
    def node_depths(self) -> Tensor:
        return self._logical_prefix(self._node_depths_buffer, self._node_count)

    @property
    def node_values(self) -> Tensor:
        return self._logical_prefix(self._node_values_buffer, self._node_count)

    @property
    def anchor_token_ids(self) -> Tensor:
        return self._anchor_token_ids

    @property
    def anchor_positions(self) -> Tensor:
        return self._anchor_positions

    def set_anchors(
        self, anchor_token_ids: Tensor, anchor_positions: Tensor
    ) -> None:


        _require_cpu_tensor(
            "anchor_token_ids", anchor_token_ids, dtype=_INT64
        )
        _require_cpu_tensor(
            "anchor_positions", anchor_positions, dtype=_INT64
        )
        if anchor_token_ids.numel() != self.request_count:
            raise ValueError("anchor_token_ids must have one entry per request")
        if anchor_positions.numel() != self.request_count:
            raise ValueError("anchor_positions must have one entry per request")
        self._anchor_token_ids.copy_(anchor_token_ids)
        self._anchor_positions.copy_(anchor_positions)

    def reset_round(
        self,
        *,
        request_refs: Sequence[RequestRef],
        anchor_token_ids: Tensor,
        anchor_positions: Tensor,
    ) -> None:


        request_tuple = tuple(request_refs)
        ForestIR(requests=request_tuple)
        if len(request_tuple) != self.request_count:
            raise ForestValidationError(
                "reset_round cannot change the arena request count"
            )
        _require_cpu_tensor(
            "anchor_token_ids", anchor_token_ids, dtype=_INT64
        )
        _require_cpu_tensor(
            "anchor_positions", anchor_positions, dtype=_INT64
        )
        if anchor_token_ids.numel() != len(request_tuple):
            raise ValueError("anchor_token_ids must have one entry per request")
        if anchor_positions.numel() != len(request_tuple):
            raise ValueError("anchor_positions must have one entry per request")

        self._anchor_token_ids.copy_(anchor_token_ids)
        self._anchor_positions.copy_(anchor_positions)
        self.request_refs = request_tuple
        self._node_count = 0

    def append_tensors(
        self,
        node_parent_ids: Tensor,
        node_request_slots: Tensor,
        node_token_ids: Tensor,
        node_depths: Tensor,
        node_values: Tensor,
    ) -> range:


        int_inputs = (
            ("node_parent_ids", node_parent_ids),
            ("node_request_slots", node_request_slots),
            ("node_token_ids", node_token_ids),
            ("node_depths", node_depths),
        )
        for name, value in int_inputs:
            _require_cpu_tensor(name, value, dtype=_INT64)
        _require_cpu_tensor("node_values", node_values, dtype=_FLOAT64)
        batch_size = node_parent_ids.numel()
        if any(value.numel() != batch_size for _, value in int_inputs[1:]):
            raise ValueError("appended node tensors must have equal length")
        if node_values.numel() != batch_size:
            raise ValueError("appended node tensors must have equal length")

        start = self._node_count
        end = start + batch_size
        if end > self.capacity:
            raise ForestValidationError(
                f"append would exceed arena capacity {self.capacity}: "
                f"start={start}, batch={batch_size}"
            )
        if batch_size == 0:
            return range(start, end)

        destination = slice(start, end)
        self._node_parent_ids_buffer[destination].copy_(node_parent_ids)
        self._node_request_slots_buffer[destination].copy_(node_request_slots)
        self._node_token_ids_buffer[destination].copy_(node_token_ids)
        self._node_depths_buffer[destination].copy_(node_depths)
        self._node_values_buffer[destination].copy_(node_values)
        self._node_count = end
        return range(start, end)

    def clear(self) -> None:


        self._node_count = 0

    def view(self) -> ForestTensorView:


        return ForestTensorView(
            request_refs=self.request_refs,
            node_parent_ids=self.node_parent_ids,
            node_request_slots=self.node_request_slots,
            node_token_ids=self.node_token_ids,
            node_depths=self.node_depths,
            anchor_token_ids=self.anchor_token_ids,
            anchor_positions=self.anchor_positions,
        )

    def select_prefix_closed(self, capacity: int) -> Tensor:
        return select_prefix_closed_arena(self, capacity)

    def select_compile_verify_plan(
        self,
        capacity: int,
        physical_rows: int,
        padding_token_id: int = 0,
    ) -> tuple[Tensor, NativeVerifyPlan]:
        return select_compile_verify_plan_arena(
            self,
            capacity,
            physical_rows,
            padding_token_id,
        )

    def compile_verify_plan(
        self,
        selected_node_ids: Tensor,
        physical_rows: int,
        padding_token_id: int = 0,
    ) -> NativeVerifyPlan:
        return compile_verify_plan_arena(
            self,
            selected_node_ids,
            physical_rows,
            padding_token_id,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class NativeVerifyPlan:


    token_ids: Tensor
    parent_rows: Tensor
    first_child: Tensor
    next_sibling: Tensor
    depths: Tensor
    positions: Tensor
    request_slots: Tensor
    valid_mask: Tensor
    candidate_mask: Tensor
    has_unverified_children: Tensor
    packed_to_node: Tensor
    node_to_packed: Tensor
    request_offsets: Tensor
    request_refs: tuple[RequestRef, ...]
    anchor_rows: Tensor
    ancestor_row_ptr: Tensor
    ancestor_columns: Tensor
    logical_rows: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_refs", tuple(self.request_refs))
        row_int_fields = (
            "token_ids",
            "parent_rows",
            "first_child",
            "next_sibling",
            "depths",
            "positions",
            "request_slots",
            "packed_to_node",
        )
        row_bool_fields = (
            "valid_mask",
            "candidate_mask",
            "has_unverified_children",
        )
        for name in row_int_fields:
            _require_cpu_tensor(name, getattr(self, name), dtype=_INT64)
        for name in row_bool_fields:
            _require_cpu_tensor(name, getattr(self, name), dtype=_BOOL)

        physical_rows = self.token_ids.numel()
        for name in row_int_fields[1:] + row_bool_fields:
            if getattr(self, name).numel() != physical_rows:
                raise ValueError(
                    "NativeVerifyPlan packed row tensors must have equal length"
                )

        for name in (
            "node_to_packed",
            "request_offsets",
            "anchor_rows",
            "ancestor_row_ptr",
            "ancestor_columns",
        ):
            _require_cpu_tensor(name, getattr(self, name), dtype=_INT64)
        _require_plain_int("logical_rows", self.logical_rows, minimum=1)
        if self.logical_rows > physical_rows:
            raise ForestValidationError(
                f"logical_rows {self.logical_rows} exceeds physical rows "
                f"{physical_rows}"
            )
        if not self.request_refs:
            raise ForestValidationError(
                "NativeVerifyPlan requires at least one request"
            )
        if self.request_offsets.numel() != len(self.request_refs) + 1:
            raise ValueError("request_offsets must have one boundary per request")
        if self.anchor_rows.numel() != len(self.request_refs):
            raise ValueError("anchor_rows must have one entry per request")
        if self.ancestor_row_ptr.numel() != physical_rows + 1:
            raise ValueError(
                "ancestor_row_ptr must have physical_rows + 1 entries"
            )

    @classmethod
    def from_kernel_outputs(
        cls,
        outputs: Sequence[Tensor],
        *,
        request_refs: Sequence[RequestRef],
    ) -> NativeVerifyPlan:


        output_tuple = tuple(outputs)
        if len(output_tuple) != 17:
            raise RuntimeError(
                "request_forest_compile_cpu must return exactly 17 tensors"
            )
        request_tuple = tuple(request_refs)
        if not request_tuple:
            raise ForestValidationError(
                "NativeVerifyPlan requires at least one request"
            )


        instance = object.__new__(cls)
        object.__setattr__(instance, "token_ids", output_tuple[0])
        object.__setattr__(instance, "parent_rows", output_tuple[1])
        object.__setattr__(instance, "first_child", output_tuple[2])
        object.__setattr__(instance, "next_sibling", output_tuple[3])
        object.__setattr__(instance, "depths", output_tuple[4])
        object.__setattr__(instance, "positions", output_tuple[5])
        object.__setattr__(instance, "request_slots", output_tuple[6])
        object.__setattr__(instance, "valid_mask", output_tuple[7])
        object.__setattr__(instance, "candidate_mask", output_tuple[8])
        object.__setattr__(
            instance, "has_unverified_children", output_tuple[9]
        )
        object.__setattr__(instance, "packed_to_node", output_tuple[10])
        object.__setattr__(instance, "node_to_packed", output_tuple[11])
        object.__setattr__(instance, "request_offsets", output_tuple[12])
        object.__setattr__(instance, "request_refs", request_tuple)
        object.__setattr__(instance, "anchor_rows", output_tuple[13])
        object.__setattr__(instance, "ancestor_row_ptr", output_tuple[14])
        object.__setattr__(instance, "ancestor_columns", output_tuple[15])
        object.__setattr__(
            instance, "logical_rows", int(output_tuple[16].item())
        )
        return instance

    @classmethod
    def from_verify_plan(cls, plan: VerifyPlan) -> NativeVerifyPlan:


        if not isinstance(plan, VerifyPlan):
            raise TypeError("plan must be a VerifyPlan")
        return cls(
            token_ids=_int64_tensor(plan.token_ids),
            parent_rows=_int64_tensor(plan.parent_rows),
            first_child=_int64_tensor(plan.first_child),
            next_sibling=_int64_tensor(plan.next_sibling),
            depths=_int64_tensor(plan.depths),
            positions=_int64_tensor(plan.positions),
            request_slots=_int64_tensor(plan.request_slots),
            valid_mask=_bool_tensor(plan.valid_mask),
            candidate_mask=_bool_tensor(plan.candidate_mask),
            has_unverified_children=_bool_tensor(
                plan.has_unverified_children
            ),
            packed_to_node=_int64_tensor(
                tuple(
                    -1 if node_id is None else node_id
                    for node_id in plan.packed_to_node
                )
            ),
            node_to_packed=_int64_tensor(plan.node_to_packed),
            request_offsets=_int64_tensor(plan.request_offsets),
            request_refs=plan.request_refs,
            anchor_rows=_int64_tensor(plan.anchor_rows),
            ancestor_row_ptr=_int64_tensor(plan.ancestor_row_ptr),
            ancestor_columns=_int64_tensor(plan.ancestor_columns),
            logical_rows=plan.logical_rows,
        )

    @property
    def physical_rows(self) -> int:
        return self.token_ids.numel()

    @property
    def candidate_rows(self) -> int:
        return self.logical_rows - len(self.request_refs)

    @property
    def ancestor_nnz(self) -> int:
        return self.ancestor_columns.numel()

    def to_verify_plan(self) -> VerifyPlan:


        packed_to_node = tuple(
            None if node_id == -1 else node_id
            for node_id in self.packed_to_node.tolist()
        )
        return VerifyPlan(
            token_ids=tuple(self.token_ids.tolist()),
            parent_rows=tuple(self.parent_rows.tolist()),
            first_child=tuple(self.first_child.tolist()),
            next_sibling=tuple(self.next_sibling.tolist()),
            depths=tuple(self.depths.tolist()),
            positions=tuple(self.positions.tolist()),
            request_slots=tuple(self.request_slots.tolist()),
            valid_mask=tuple(self.valid_mask.tolist()),
            candidate_mask=tuple(self.candidate_mask.tolist()),
            has_unverified_children=tuple(
                self.has_unverified_children.tolist()
            ),
            packed_to_node=packed_to_node,
            node_to_packed=tuple(self.node_to_packed.tolist()),
            request_offsets=tuple(self.request_offsets.tolist()),
            request_refs=self.request_refs,
            anchor_rows=tuple(self.anchor_rows.tolist()),
            ancestor_row_ptr=tuple(self.ancestor_row_ptr.tolist()),
            ancestor_columns=tuple(self.ancestor_columns.tolist()),
            logical_rows=self.logical_rows,
            physical_rows=self.physical_rows,
        )

    materialize = to_verify_plan


@dataclass(frozen=True, slots=True, kw_only=True)
class NativeCommitBatch:


    request_refs: tuple[RequestRef, ...]
    accepted_node_ids: Tensor
    accepted_offsets: Tensor
    output_token_ids: Tensor
    output_offsets: Tensor
    accepted_lengths: Tensor
    first_miss_positions: Tensor
    correction_token_ids: Tensor
    termination_reasons: Tensor
    visited_rows: Tensor

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_refs", tuple(self.request_refs))
        for name in (
            "accepted_node_ids",
            "accepted_offsets",
            "output_token_ids",
            "output_offsets",
            "accepted_lengths",
            "first_miss_positions",
            "correction_token_ids",
            "termination_reasons",
            "visited_rows",
        ):
            _require_cpu_tensor(name, getattr(self, name), dtype=_INT64)
        request_count = len(self.request_refs)
        if request_count == 0:
            raise ForestValidationError(
                "NativeCommitBatch requires at least one request"
            )
        if self.accepted_offsets.numel() != request_count + 1:
            raise ValueError("accepted_offsets must have request_count + 1 entries")
        if self.output_offsets.numel() != request_count + 1:
            raise ValueError("output_offsets must have request_count + 1 entries")
        for name in (
            "accepted_lengths",
            "first_miss_positions",
            "correction_token_ids",
            "termination_reasons",
            "visited_rows",
        ):
            if getattr(self, name).numel() != request_count:
                raise ValueError(f"{name} must have one entry per request")

    @classmethod
    def from_kernel_outputs(
        cls,
        outputs: Sequence[Tensor],
        *,
        request_refs: Sequence[RequestRef],
    ) -> NativeCommitBatch:


        output_tuple = tuple(outputs)
        if len(output_tuple) != 9:
            raise RuntimeError(
                "request_forest_commit_cpu must return exactly 9 tensors"
            )
        request_tuple = tuple(request_refs)
        if not request_tuple:
            raise ForestValidationError(
                "NativeCommitBatch requires at least one request"
            )
        instance = object.__new__(cls)
        object.__setattr__(instance, "request_refs", request_tuple)
        object.__setattr__(instance, "accepted_node_ids", output_tuple[0])
        object.__setattr__(instance, "accepted_offsets", output_tuple[1])
        object.__setattr__(instance, "output_token_ids", output_tuple[2])
        object.__setattr__(instance, "output_offsets", output_tuple[3])
        object.__setattr__(instance, "accepted_lengths", output_tuple[4])
        object.__setattr__(
            instance, "first_miss_positions", output_tuple[5]
        )
        object.__setattr__(
            instance, "correction_token_ids", output_tuple[6]
        )
        object.__setattr__(instance, "termination_reasons", output_tuple[7])
        object.__setattr__(instance, "visited_rows", output_tuple[8])
        return instance

    @classmethod
    def from_commit_results(
        cls,
        results: Sequence[CommitResult],
        *,
        visited_rows: Sequence[int] | None = None,
    ) -> NativeCommitBatch:


        result_tuple = tuple(results)
        if any(not isinstance(result, CommitResult) for result in result_tuple):
            raise TypeError("results must contain CommitResult values")
        if not result_tuple:
            raise ValueError("results must contain at least one request")

        accepted_flat: list[int] = []
        output_flat: list[int] = []
        accepted_offsets = [0]
        output_offsets = [0]
        for result in result_tuple:
            accepted_flat.extend(result.accepted_node_ids)
            output_flat.extend(result.output_token_ids)
            accepted_offsets.append(len(accepted_flat))
            output_offsets.append(len(output_flat))

        if visited_rows is None:
            visited_row_tuple = (-1,) * len(result_tuple)
        else:
            visited_row_tuple = tuple(visited_rows)
            if len(visited_row_tuple) != len(result_tuple):
                raise ValueError("visited_rows must have one entry per request")

        return cls(
            request_refs=tuple(result.request for result in result_tuple),
            accepted_node_ids=_int64_tensor(accepted_flat),
            accepted_offsets=_int64_tensor(accepted_offsets),
            output_token_ids=_int64_tensor(output_flat),
            output_offsets=_int64_tensor(output_offsets),
            accepted_lengths=_int64_tensor(
                tuple(result.accepted_length for result in result_tuple)
            ),
            first_miss_positions=_int64_tensor(
                tuple(
                    -1
                    if result.first_miss_position is None
                    else result.first_miss_position
                    for result in result_tuple
                )
            ),
            correction_token_ids=_int64_tensor(
                tuple(
                    -1
                    if result.correction_token_id is None
                    else result.correction_token_id
                    for result in result_tuple
                )
            ),
            termination_reasons=_int64_tensor(
                tuple(
                    _TERMINATION_CODES[result.termination_reason]
                    for result in result_tuple
                )
            ),
            visited_rows=_int64_tensor(visited_row_tuple),
        )

    @property
    def termination_rows(self) -> Tensor:


        return self.visited_rows

    def to_commit_results(self) -> tuple[CommitResult, ...]:


        accepted_flat = self.accepted_node_ids.tolist()
        accepted_offsets = self.accepted_offsets.tolist()
        output_flat = self.output_token_ids.tolist()
        output_offsets = self.output_offsets.tolist()
        accepted_lengths = self.accepted_lengths.tolist()
        first_misses = self.first_miss_positions.tolist()
        corrections = self.correction_token_ids.tolist()
        reason_codes = self.termination_reasons.tolist()

        results = []
        for slot, request in enumerate(self.request_refs):
            reason_code = reason_codes[slot]
            if reason_code < 0 or reason_code >= len(_TERMINATION_REASONS):
                raise RuntimeError(
                    f"native commit returned unknown termination reason {reason_code}"
                )
            results.append(
                CommitResult(
                    request=request,
                    accepted_node_ids=tuple(
                        accepted_flat[
                            accepted_offsets[slot] : accepted_offsets[slot + 1]
                        ]
                    ),
                    output_token_ids=tuple(
                        output_flat[output_offsets[slot] : output_offsets[slot + 1]]
                    ),
                    accepted_length=accepted_lengths[slot],
                    first_miss_position=(
                        None if first_misses[slot] == -1 else first_misses[slot]
                    ),
                    correction_token_id=(
                        None if corrections[slot] == -1 else corrections[slot]
                    ),
                    termination_reason=_TERMINATION_REASONS[reason_code],
                )
            )
        return tuple(results)

    materialize = to_commit_results


def tensorize_forest(
    forest: ForestIR, anchors: Sequence[RequestAnchor]
) -> ForestTensorView:


    return ForestTensorView.from_forest(forest, anchors)


def tensorize_node_values(
    view: ForestTensorView, node_values: Mapping[int, float]
) -> Tensor:


    if not isinstance(view, ForestTensorView):
        raise TypeError("view must be a ForestTensorView")
    missing = [
        node_id
        for node_id in range(view.node_count)
        if node_id not in node_values
    ]
    if missing:
        raise ValueError(f"missing values for node IDs: {missing}")
    if len(node_values) > view.node_count:
        unknown = set(node_values).difference(range(view.node_count))
        if unknown:
            raise ValueError(f"values contain unknown node IDs: {sorted(unknown)}")

    values = []
    for node_id in range(view.node_count):
        raw_value = node_values[node_id]
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError(f"value for node {node_id} must be a finite number")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(
                f"value for node {node_id} must be finite, got {value!r}"
            )
        values.append(value)
    return _require_torch().tensor(values, dtype=_torch_dtype(_FLOAT64))


def select_prefix_closed_tensor(
    parent_ids: Tensor,
    values: Tensor,
    capacity: int,
) -> Tensor:


    _require_cpu_tensor("parent_ids", parent_ids, dtype=_INT64)
    _require_cpu_tensor("values", values, dtype=_FLOAT64)
    _require_plain_int("capacity", capacity)
    return _native_op("request_forest_select_cpu")(parent_ids, values, capacity)


def select_prefix_closed_native(
    view: ForestTensorView,
    values: Tensor,
    capacity: int,
) -> Tensor:


    if not isinstance(view, ForestTensorView):
        raise TypeError("view must be a ForestTensorView")
    return select_prefix_closed_tensor(view.node_parent_ids, values, capacity)


def select_prefix_closed_arena(
    arena: TensorForestArena,
    capacity: int,
) -> Tensor:


    if not isinstance(arena, TensorForestArena):
        raise TypeError("arena must be a TensorForestArena")
    return select_prefix_closed_tensor(
        arena.node_parent_ids, arena.node_values, capacity
    )


def _select_compile_verify_plan_components(
    *,
    request_refs: Sequence[RequestRef],
    node_parent_ids: Tensor,
    node_request_slots: Tensor,
    node_token_ids: Tensor,
    node_depths: Tensor,
    node_values: Tensor,
    anchor_token_ids: Tensor,
    anchor_positions: Tensor,
    capacity: int,
    physical_rows: int,
    padding_token_id: int,
) -> tuple[Tensor, NativeVerifyPlan]:
    outputs = tuple(
        _native_op("request_forest_select_compile_cpu")(
            node_parent_ids,
            node_request_slots,
            node_token_ids,
            node_depths,
            node_values,
            anchor_token_ids,
            anchor_positions,
            capacity,
            physical_rows,
            padding_token_id,
        )
    )
    if len(outputs) != 18:
        raise RuntimeError(
            "request_forest_select_compile_cpu must return exactly 18 tensors"
        )
    return outputs[0], NativeVerifyPlan.from_kernel_outputs(
        outputs[1:], request_refs=request_refs
    )


def select_compile_verify_plan_tensor(
    view: ForestTensorView,
    values: Tensor,
    capacity: int,
    physical_rows: int,
    padding_token_id: int = 0,
) -> tuple[Tensor, NativeVerifyPlan]:


    if not isinstance(view, ForestTensorView):
        raise TypeError("view must be a ForestTensorView")
    _require_cpu_tensor("values", values, dtype=_FLOAT64)
    _require_plain_int("capacity", capacity)
    _require_plain_int("physical_rows", physical_rows, minimum=1)
    _require_plain_int("padding_token_id", padding_token_id)
    return _select_compile_verify_plan_components(
        request_refs=view.request_refs,
        node_parent_ids=view.node_parent_ids,
        node_request_slots=view.node_request_slots,
        node_token_ids=view.node_token_ids,
        node_depths=view.node_depths,
        node_values=values,
        anchor_token_ids=view.anchor_token_ids,
        anchor_positions=view.anchor_positions,
        capacity=capacity,
        physical_rows=physical_rows,
        padding_token_id=padding_token_id,
    )


select_compile_verify_plan_native = select_compile_verify_plan_tensor


def select_compile_verify_plan_arena(
    arena: TensorForestArena,
    capacity: int,
    physical_rows: int,
    padding_token_id: int = 0,
) -> tuple[Tensor, NativeVerifyPlan]:


    if not isinstance(arena, TensorForestArena):
        raise TypeError("arena must be a TensorForestArena")
    _require_plain_int("capacity", capacity)
    _require_plain_int("physical_rows", physical_rows, minimum=1)
    _require_plain_int("padding_token_id", padding_token_id)
    return _select_compile_verify_plan_components(
        request_refs=arena.request_refs,
        node_parent_ids=arena.node_parent_ids,
        node_request_slots=arena.node_request_slots,
        node_token_ids=arena.node_token_ids,
        node_depths=arena.node_depths,
        node_values=arena.node_values,
        anchor_token_ids=arena.anchor_token_ids,
        anchor_positions=arena.anchor_positions,
        capacity=capacity,
        physical_rows=physical_rows,
        padding_token_id=padding_token_id,
    )


def _compile_verify_plan_components(
    *,
    request_refs: Sequence[RequestRef],
    node_parent_ids: Tensor,
    node_request_slots: Tensor,
    node_token_ids: Tensor,
    node_depths: Tensor,
    anchor_token_ids: Tensor,
    anchor_positions: Tensor,
    selected_node_ids: Tensor,
    physical_rows: int,
    padding_token_id: int,
) -> NativeVerifyPlan:
    outputs = _native_op("request_forest_compile_cpu")(
        node_parent_ids,
        node_request_slots,
        node_token_ids,
        node_depths,
        anchor_token_ids,
        anchor_positions,
        selected_node_ids,
        physical_rows,
        padding_token_id,
    )
    return NativeVerifyPlan.from_kernel_outputs(
        outputs, request_refs=request_refs
    )


def compile_verify_plan_tensor(
    view: ForestTensorView,
    selected_node_ids: Tensor,
    physical_rows: int,
    padding_token_id: int = 0,
) -> NativeVerifyPlan:


    if not isinstance(view, ForestTensorView):
        raise TypeError("view must be a ForestTensorView")
    _require_cpu_tensor("selected_node_ids", selected_node_ids, dtype=_INT64)
    _require_plain_int("physical_rows", physical_rows, minimum=1)
    _require_plain_int("padding_token_id", padding_token_id)
    return _compile_verify_plan_components(
        request_refs=view.request_refs,
        node_parent_ids=view.node_parent_ids,
        node_request_slots=view.node_request_slots,
        node_token_ids=view.node_token_ids,
        node_depths=view.node_depths,
        anchor_token_ids=view.anchor_token_ids,
        anchor_positions=view.anchor_positions,
        selected_node_ids=selected_node_ids,
        physical_rows=physical_rows,
        padding_token_id=padding_token_id,
    )


compile_verify_plan_native = compile_verify_plan_tensor


def compile_verify_plan_arena(
    arena: TensorForestArena,
    selected_node_ids: Tensor,
    physical_rows: int,
    padding_token_id: int = 0,
) -> NativeVerifyPlan:


    if not isinstance(arena, TensorForestArena):
        raise TypeError("arena must be a TensorForestArena")
    _require_cpu_tensor("selected_node_ids", selected_node_ids, dtype=_INT64)
    _require_plain_int("physical_rows", physical_rows, minimum=1)
    _require_plain_int("padding_token_id", padding_token_id)
    return _compile_verify_plan_components(
        request_refs=arena.request_refs,
        node_parent_ids=arena.node_parent_ids,
        node_request_slots=arena.node_request_slots,
        node_token_ids=arena.node_token_ids,
        node_depths=arena.node_depths,
        anchor_token_ids=arena.anchor_token_ids,
        anchor_positions=arena.anchor_positions,
        selected_node_ids=selected_node_ids,
        physical_rows=physical_rows,
        padding_token_id=padding_token_id,
    )


def materialize_target_decisions(
    plan: NativeVerifyPlan,
    target_decisions: Mapping[int, TargetDecision],
) -> tuple[Tensor, Tensor]:


    if not isinstance(plan, NativeVerifyPlan):
        raise TypeError("plan must be a NativeVerifyPlan")
    target_token_ids = [-1] * plan.physical_rows
    target_is_eos = [False] * plan.physical_rows
    for row in range(plan.logical_rows):
        decision = target_decisions.get(row)
        if decision is None:
            continue
        if not isinstance(decision, TargetDecision):
            raise ValueError(
                f"Target decision for row {row} must be TargetDecision"
            )
        target_token_ids[row] = decision.token_id
        target_is_eos[row] = decision.is_eos
    return _int64_tensor(target_token_ids), _bool_tensor(target_is_eos)


def execute_commit_tensor(
    plan: NativeVerifyPlan,
    target_token_ids: Tensor,
    target_is_eos: Tensor,
) -> NativeCommitBatch:


    if not isinstance(plan, NativeVerifyPlan):
        raise TypeError("plan must be a NativeVerifyPlan")
    _require_cpu_tensor("target_token_ids", target_token_ids, dtype=_INT64)
    _require_cpu_tensor("target_is_eos", target_is_eos, dtype=_BOOL)
    outputs = _native_op("request_forest_commit_cpu")(
        plan.anchor_rows,
        plan.token_ids,
        plan.first_child,
        plan.next_sibling,
        plan.positions,
        plan.has_unverified_children,
        plan.packed_to_node,
        target_token_ids,
        target_is_eos,
        plan.logical_rows,
    )
    return NativeCommitBatch.from_kernel_outputs(
        outputs, request_refs=plan.request_refs
    )


def execute_commit_program_native(
    plan: NativeVerifyPlan,
    target_decisions: Mapping[int, TargetDecision],
) -> NativeCommitBatch:


    target_token_ids, target_is_eos = materialize_target_decisions(
        plan, target_decisions
    )
    return execute_commit_tensor(plan, target_token_ids, target_is_eos)


commit_native = execute_commit_tensor


__all__ = [
    "ForestTensorView",
    "NativeCommitBatch",
    "NativeVerifyPlan",
    "TensorForestArena",
    "commit_native",
    "compile_verify_plan_arena",
    "compile_verify_plan_native",
    "compile_verify_plan_tensor",
    "execute_commit_program_native",
    "execute_commit_tensor",
    "materialize_target_decisions",
    "select_prefix_closed_arena",
    "select_prefix_closed_native",
    "select_prefix_closed_tensor",
    "select_compile_verify_plan_arena",
    "select_compile_verify_plan_native",
    "select_compile_verify_plan_tensor",
    "tensorize_forest",
    "tensorize_node_values",
]
