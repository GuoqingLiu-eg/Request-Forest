from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .ir import (
    ForestValidationError,
    RequestRef,
    StateHandle,
    _require_plain_int,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class RequestAnchor:


    request: RequestRef
    token_id: int
    position: int
    state_handle: Optional[StateHandle] = None

    def __post_init__(self) -> None:
        if not isinstance(self.request, RequestRef):
            raise ForestValidationError(
                f"request must be RequestRef, got {self.request!r}"
            )
        _require_plain_int("token_id", self.token_id)
        _require_plain_int("position", self.position)
        if self.state_handle is not None and (
            isinstance(self.state_handle, bool)
            or not isinstance(self.state_handle, (int, str))
        ):
            raise ForestValidationError(
                "state_handle must be an int, str, or None, got "
                f"{self.state_handle!r}"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class ExpansionAction:


    request: Optional[RequestRef]
    parent_id: Optional[int]
    depth: int
    position: int
    state_handle: Optional[StateHandle] = None
    context_view_id: Optional[int] = None
    valid: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise ForestValidationError(f"valid must be bool, got {self.valid!r}")
        if self.valid:
            if not isinstance(self.request, RequestRef):
                raise ForestValidationError(
                    "a valid ExpansionAction needs a RequestRef"
                )
            if self.parent_id is not None:
                _require_plain_int("parent_id", self.parent_id)
            _require_plain_int("depth", self.depth)
            _require_plain_int("position", self.position)
            if self.parent_id is None and self.depth != 0:
                raise ForestValidationError(
                    "an anchor ExpansionAction must have depth 0"
                )
            if self.parent_id is not None and self.depth == 0:
                raise ForestValidationError(
                    "a candidate-parent ExpansionAction must have depth >= 1"
                )
            if self.context_view_id is not None:
                _require_plain_int("context_view_id", self.context_view_id)
        elif (
            self.request is not None
            or self.parent_id is not None
            or self.depth != -1
            or self.position != -1
            or self.state_handle is not None
            or self.context_view_id is not None
        ):
            raise ForestValidationError(
                "an invalid ExpansionAction must use only padding sentinels"
            )

        if self.state_handle is not None and (
            isinstance(self.state_handle, bool)
            or not isinstance(self.state_handle, (int, str))
        ):
            raise ForestValidationError(
                "state_handle must be an int, str, or None, got "
                f"{self.state_handle!r}"
            )

    @classmethod
    def padding(cls) -> ExpansionAction:
        return cls(request=None, parent_id=None, depth=-1, position=-1, valid=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class ExpansionPlan:
    actions: tuple[ExpansionAction, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "actions", tuple(self.actions))
        if not self.actions:
            raise ForestValidationError("ExpansionPlan width must be positive")

        saw_padding = False
        seen_parents: set[tuple[RequestRef, Optional[int]]] = set()
        for action in self.actions:
            if not action.valid:
                saw_padding = True
                continue
            if saw_padding:
                raise ForestValidationError(
                    "valid ExpansionAction rows must precede padding rows"
                )
            parent_key = (action.request, action.parent_id)
            if parent_key in seen_parents:
                raise ForestValidationError(
                    f"duplicate expansion parent in one plan: {parent_key!r}"
                )
            seen_parents.add(parent_key)

    @property
    def width(self) -> int:
        return len(self.actions)

    @property
    def valid_mask(self) -> tuple[bool, ...]:
        return tuple(action.valid for action in self.actions)


@dataclass(frozen=True, slots=True, kw_only=True)
class VerifyPlan:


    token_ids: tuple[int, ...]
    parent_rows: tuple[int, ...]
    first_child: tuple[int, ...]
    next_sibling: tuple[int, ...]
    depths: tuple[int, ...]
    positions: tuple[int, ...]
    request_slots: tuple[int, ...]
    valid_mask: tuple[bool, ...]
    candidate_mask: tuple[bool, ...]
    has_unverified_children: tuple[bool, ...]
    packed_to_node: tuple[Optional[int], ...]
    node_to_packed: tuple[int, ...]
    request_offsets: tuple[int, ...]
    request_refs: tuple[RequestRef, ...]
    anchor_rows: tuple[int, ...]
    ancestor_row_ptr: tuple[int, ...]
    ancestor_columns: tuple[int, ...]
    logical_rows: int
    physical_rows: int

    def __post_init__(self) -> None:
        tuple_fields = (
            "token_ids",
            "parent_rows",
            "first_child",
            "next_sibling",
            "depths",
            "positions",
            "request_slots",
            "valid_mask",
            "candidate_mask",
            "has_unverified_children",
            "packed_to_node",
            "node_to_packed",
            "request_offsets",
            "request_refs",
            "anchor_rows",
            "ancestor_row_ptr",
            "ancestor_columns",
        )
        for field_name in tuple_fields:
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))

        _require_plain_int("logical_rows", self.logical_rows, minimum=1)
        _require_plain_int("physical_rows", self.physical_rows, minimum=1)
        if self.logical_rows > self.physical_rows:
            raise ForestValidationError(
                f"logical_rows {self.logical_rows} exceeds physical_rows "
                f"{self.physical_rows}"
            )

        row_fields = (
            self.token_ids,
            self.parent_rows,
            self.first_child,
            self.next_sibling,
            self.depths,
            self.positions,
            self.request_slots,
            self.valid_mask,
            self.candidate_mask,
            self.has_unverified_children,
            self.packed_to_node,
        )
        if any(len(field) != self.physical_rows for field in row_fields):
            raise ForestValidationError(
                "all packed VerifyPlan fields must have physical_rows entries"
            )


        if len(self.ancestor_row_ptr) != self.physical_rows + 1:
            raise ForestValidationError(
                "ancestor_row_ptr must contain physical_rows + 1 entries"
            )
        previous_offset = -1
        column_count = len(self.ancestor_columns)
        for offset in self.ancestor_row_ptr:
            if type(offset) is not int:
                raise ForestValidationError(
                    "ancestor_row_ptr must contain plain integers"
                )
            if not previous_offset <= offset <= column_count:
                raise ForestValidationError(
                    "ancestor_row_ptr must be non-decreasing and within columns"
                )
            previous_offset = offset
        if self.ancestor_row_ptr[0] != 0:
            raise ForestValidationError(
                "ancestor_row_ptr must start at zero"
            )
        if self.ancestor_row_ptr[-1] != len(self.ancestor_columns):
            raise ForestValidationError(
                "ancestor_row_ptr must end at len(ancestor_columns)"
            )

        request_count = len(self.request_refs)
        if request_count == 0:
            raise ForestValidationError("VerifyPlan requires at least one request")
        if any(not isinstance(request, RequestRef) for request in self.request_refs):
            raise ForestValidationError(
                "VerifyPlan request_refs must contain RequestRef values"
            )
        if len(set(self.request_refs)) != request_count:
            raise ForestValidationError("VerifyPlan request_refs must be unique")
        if len({request.request_id for request in self.request_refs}) != request_count:
            raise ForestValidationError(
                "VerifyPlan cannot contain multiple versions of one request_id"
            )
        if len(self.request_offsets) != request_count + 1:
            raise ForestValidationError(
                "request_offsets must have len(request_refs) + 1 entries"
            )
        for field_name, values in (
            ("request_offsets", self.request_offsets),
            ("anchor_rows", self.anchor_rows),
        ):
            if any(type(value) is not int or value < 0 for value in values):
                raise ForestValidationError(
                    f"{field_name} must contain non-negative plain integers"
                )
        if (
            self.request_offsets[0] != 0
            or self.request_offsets[-1] != self.logical_rows
        ):
            raise ForestValidationError(
                "request_offsets must start at 0 and end at logical_rows"
            )
        if any(
            left >= right
            for left, right in zip(self.request_offsets, self.request_offsets[1:])
        ):
            raise ForestValidationError(
                "each request segment must contain at least its anchor"
            )
        if self.anchor_rows != self.request_offsets[:-1]:
            raise ForestValidationError(
                "anchor_rows must equal the start of each request segment"
            )

    @property
    def candidate_rows(self) -> int:
        return self.logical_rows - len(self.request_refs)

    @property
    def ancestor_nnz(self) -> int:


        return len(self.ancestor_columns)

    def packed_row_for(self, node_id: int) -> int:
        _require_plain_int("node_id", node_id)
        try:
            return self.node_to_packed[node_id]
        except IndexError as exc:
            raise ForestValidationError(f"unknown node_id {node_id}") from exc

    def ancestor_columns_for(self, row: int) -> tuple[int, ...]:


        _require_plain_int("row", row)
        if row >= self.physical_rows:
            raise ForestValidationError(f"unknown physical row {row}")
        start = self.ancestor_row_ptr[row]
        end = self.ancestor_row_ptr[row + 1]
        return self.ancestor_columns[start:end]

    def materialize_dense_ancestor_mask(
        self, *, max_physical_rows: int = 512
    ) -> tuple[tuple[bool, ...], ...]:


        _require_plain_int("max_physical_rows", max_physical_rows, minimum=1)
        if self.physical_rows > max_physical_rows:
            raise ForestValidationError(
                "refusing to materialize a dense ancestor mask for "
                f"{self.physical_rows} rows; debug limit is "
                f"{max_physical_rows}"
            )

        dense_rows = []
        for row in range(self.physical_rows):
            enabled = [False] * self.physical_rows
            start = self.ancestor_row_ptr[row]
            end = self.ancestor_row_ptr[row + 1]
            for column in self.ancestor_columns[start:end]:
                enabled[column] = True
            dense_rows.append(tuple(enabled))
        return tuple(dense_rows)
