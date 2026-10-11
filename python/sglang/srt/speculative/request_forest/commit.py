from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Mapping

from .ir import RequestRef

if TYPE_CHECKING:
    from .plans import VerifyPlan


class TerminationReason(str, Enum):
    TARGET_MISS = "target_miss"
    LEAF_EXHAUSTED = "leaf_exhausted"
    BUDGET_BOUNDARY = "budget_boundary"
    NO_CANDIDATES = "no_candidates"
    EOS = "eos"


@dataclass(frozen=True)
class TargetDecision:


    token_id: int
    is_eos: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.token_id, int)
            or isinstance(self.token_id, bool)
            or self.token_id < 0
        ):
            raise ValueError("Target token_id must be non-negative")
        if not isinstance(self.is_eos, bool):
            raise ValueError("Target is_eos must be bool")


@dataclass(frozen=True)
class CommitProgram:


    request_refs: tuple[RequestRef, ...]
    anchor_rows: tuple[int, ...]
    token_ids: tuple[int, ...]
    first_child: tuple[int, ...]
    next_sibling: tuple[int, ...]
    positions: tuple[int, ...]
    has_unverified_children: tuple[bool, ...]
    packed_to_node: tuple[int | None, ...]

    def __post_init__(self) -> None:
        row_count = len(self.token_ids)
        arrays = (
            self.first_child,
            self.next_sibling,
            self.positions,
            self.has_unverified_children,
            self.packed_to_node,
        )
        if any(len(values) != row_count for values in arrays):
            raise ValueError("CommitProgram row arrays must have equal length")
        if len(self.request_refs) != len(self.anchor_rows):
            raise ValueError("each request must have exactly one anchor row")
        if any(not isinstance(request, RequestRef) for request in self.request_refs):
            raise ValueError("request_refs must contain RequestRef values")
        if len(set(self.request_refs)) != len(self.request_refs):
            raise ValueError("request_refs must be unique")
        if any(
            not isinstance(row, int) or isinstance(row, bool) or row < 0
            for row in self.anchor_rows
        ):
            raise ValueError("anchor_rows must contain non-negative integers")
        if len(set(self.anchor_rows)) != len(self.anchor_rows):
            raise ValueError("anchor_rows must be unique")
        if any(
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or token_id < 0
            for token_id in self.token_ids
        ):
            raise ValueError("logical token_ids must be non-negative integers")
        if any(
            not isinstance(position, int)
            or isinstance(position, bool)
            or position < 0
            for position in self.positions
        ):
            raise ValueError("positions must be non-negative integers")
        for name, rows in (
            ("first_child", self.first_child),
            ("next_sibling", self.next_sibling),
        ):
            if any(
                not isinstance(row, int)
                or isinstance(row, bool)
                or row < -1
                for row in rows
            ):
                raise ValueError(f"{name} must contain integer row indices")
        if any(
            not isinstance(value, bool) for value in self.has_unverified_children
        ):
            raise ValueError("has_unverified_children must contain bool values")
        if any(
            node_id is not None
            and (
                not isinstance(node_id, int)
                or isinstance(node_id, bool)
                or node_id < 0
            )
            for node_id in self.packed_to_node
        ):
            raise ValueError(
                "packed_to_node must contain non-negative node IDs or None"
            )

        anchors = set(self.anchor_rows)
        if any(row < 0 or row >= row_count for row in anchors):
            raise ValueError("anchor row is outside the logical plan")
        for row, node_id in enumerate(self.packed_to_node):
            if row in anchors and node_id is not None:
                raise ValueError("anchor rows cannot map to candidate nodes")
            if row not in anchors and node_id is None:
                raise ValueError("every non-anchor logical row must map to a node")

        referenced_children: set[int] = set()
        for parent_row in range(row_count):
            seen_tokens: set[int] = set()
            seen_siblings: set[int] = set()
            child_row = self.first_child[parent_row]
            while child_row != -1:
                if child_row < 0 or child_row >= row_count:
                    raise ValueError("child row is outside the logical plan")
                if child_row in seen_siblings:
                    raise ValueError("cycle in next_sibling chain")
                if child_row <= parent_row:
                    raise ValueError("CommitProgram must be parent-before-child")
                if child_row in anchors:
                    raise ValueError("an anchor row cannot be a child")
                if self.positions[child_row] != self.positions[parent_row] + 1:
                    raise ValueError(
                        "child position must equal parent position plus one"
                    )
                token_id = self.token_ids[child_row]
                if token_id in seen_tokens:
                    raise ValueError("duplicate child token makes traversal ambiguous")
                if child_row in referenced_children:
                    raise ValueError("candidate row has multiple logical parents")
                seen_tokens.add(token_id)
                seen_siblings.add(child_row)
                referenced_children.add(child_row)
                child_row = self.next_sibling[child_row]

        expected_children = set(range(row_count)) - anchors
        if referenced_children != expected_children:
            raise ValueError("every candidate row must be reachable from one parent")
        if any(self.next_sibling[anchor] != -1 for anchor in anchors):
            raise ValueError("anchor rows cannot be siblings")

    def decision_position(self, row: int) -> int:
        if not isinstance(row, int) or isinstance(row, bool):
            raise ValueError("row must be an integer")
        if row < 0 or row >= len(self.positions):
            raise ValueError(f"row {row} is outside the logical plan")
        return self.positions[row] + 1


@dataclass(frozen=True)
class CommitResult:
    request: RequestRef
    accepted_node_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    accepted_length: int
    first_miss_position: int | None
    correction_token_id: int | None
    termination_reason: TerminationReason


def build_commit_program(verify_plan: VerifyPlan) -> CommitProgram:


    logical_rows = verify_plan.logical_rows
    return CommitProgram(
        request_refs=verify_plan.request_refs,
        anchor_rows=verify_plan.anchor_rows,
        token_ids=verify_plan.token_ids[:logical_rows],
        first_child=verify_plan.first_child[:logical_rows],
        next_sibling=verify_plan.next_sibling[:logical_rows],
        positions=verify_plan.positions[:logical_rows],
        has_unverified_children=verify_plan.has_unverified_children[:logical_rows],
        packed_to_node=verify_plan.packed_to_node[:logical_rows],
    )


def execute_commit_program(
    program: CommitProgram,
    target_decisions: Mapping[int, TargetDecision],
) -> tuple[CommitResult, ...]:


    results = []
    for request, anchor_row in zip(program.request_refs, program.anchor_rows):
        current_row = anchor_row
        accepted_node_ids: list[int] = []
        output_token_ids: list[int] = []

        while True:
            decision = target_decisions.get(current_row)
            if decision is None:
                raise ValueError(
                    f"missing Target decision for traversed row {current_row}"
                )
            if not isinstance(decision, TargetDecision):
                raise ValueError(
                    f"Target decision for row {current_row} must be TargetDecision"
                )

            matching_child = _find_child(program, current_row, decision.token_id)
            if matching_child != -1:
                node_id = program.packed_to_node[matching_child]
                if node_id is None:
                    raise RuntimeError(
                        "validated candidate row lost its logical node mapping"
                    )
                accepted_node_ids.append(node_id)
                output_token_ids.append(decision.token_id)
                if decision.is_eos:
                    results.append(
                        _make_result(
                            request=request,
                            accepted_node_ids=accepted_node_ids,
                            output_token_ids=output_token_ids,
                            first_miss_position=None,
                            correction_token_id=None,
                            reason=TerminationReason.EOS,
                        )
                    )
                    break
                current_row = matching_child
                continue

            output_token_ids.append(decision.token_id)
            miss_position = program.decision_position(current_row)
            if decision.is_eos:
                reason = TerminationReason.EOS
            elif program.first_child[current_row] == -1:
                if program.has_unverified_children[current_row]:
                    reason = TerminationReason.BUDGET_BOUNDARY
                elif current_row == anchor_row:
                    reason = TerminationReason.NO_CANDIDATES
                else:
                    reason = TerminationReason.LEAF_EXHAUSTED
            else:
                reason = TerminationReason.TARGET_MISS
            results.append(
                _make_result(
                    request=request,
                    accepted_node_ids=accepted_node_ids,
                    output_token_ids=output_token_ids,
                    first_miss_position=miss_position,
                    correction_token_id=decision.token_id,
                    reason=reason,
                )
            )
            break

    return tuple(results)


def _find_child(program: CommitProgram, parent_row: int, token_id: int) -> int:
    child_row = program.first_child[parent_row]
    while child_row != -1:
        if program.token_ids[child_row] == token_id:
            return child_row
        child_row = program.next_sibling[child_row]
    return -1


def _make_result(
    *,
    request: RequestRef,
    accepted_node_ids: list[int],
    output_token_ids: list[int],
    first_miss_position: int | None,
    correction_token_id: int | None,
    reason: TerminationReason,
) -> CommitResult:
    return CommitResult(
        request=request,
        accepted_node_ids=tuple(accepted_node_ids),
        output_token_ids=tuple(output_token_ids),
        accepted_length=len(accepted_node_ids),
        first_miss_position=first_miss_position,
        correction_token_id=correction_token_id,
        termination_reason=reason,
    )
