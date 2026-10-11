from __future__ import annotations

from collections.abc import Iterable, Sequence

from .ir import (
    ForestIR,
    ForestNode,
    ForestValidationError,
    _require_plain_int,
)
from .plans import RequestAnchor, VerifyPlan


def prefix_close_selection(
    forest: ForestIR, selected_node_ids: Iterable[int]
) -> tuple[int, ...]:


    requested = tuple(selected_node_ids)
    for node_id in requested:
        _require_plain_int("selected node_id", node_id)
    if len(set(requested)) != len(requested):
        raise ForestValidationError("selected_node_ids contains duplicates")

    nodes = forest.nodes
    node_count = len(nodes)
    closed: set[int] = set()
    for node_id in requested:
        if node_id >= node_count:
            raise ForestValidationError(f"unknown node_id {node_id}")
        node = nodes[node_id]
        while node.node_id not in closed:
            closed.add(node.node_id)
            if node.parent_id is None:
                break
            node = nodes[node.parent_id]
    return tuple(sorted(closed))


def compile_verify_plan(
    *,
    forest: ForestIR,
    anchors: Sequence[RequestAnchor],
    selected_node_ids: Iterable[int],
    physical_rows: int,
    padding_token_id: int = 0,
) -> VerifyPlan:


    if not isinstance(forest, ForestIR):
        raise ForestValidationError("forest must be a ForestIR")
    _require_plain_int("physical_rows", physical_rows, minimum=1)
    _require_plain_int("padding_token_id", padding_token_id)

    anchor_tuple = tuple(anchors)
    if any(not isinstance(anchor, RequestAnchor) for anchor in anchor_tuple):
        raise ForestValidationError("anchors must contain RequestAnchor values")
    if len(anchor_tuple) != len(forest.requests):
        raise ForestValidationError(
            "anchors must contain exactly one entry for every ForestIR request"
        )
    anchor_by_request: dict = {}
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

    selected = prefix_close_selection(forest, selected_node_ids)
    selected_set = set(selected)
    logical_rows = len(forest.requests) + len(selected)
    if logical_rows > physical_rows:
        raise ForestValidationError(
            f"logical verification rows {logical_rows} exceed physical bucket "
            f"{physical_rows}"
        )

    token_ids: list[int] = []
    parent_rows: list[int] = []
    first_child: list[int] = []
    next_sibling: list[int] = []
    last_child: list[int] = []
    depths: list[int] = []
    positions: list[int] = []
    request_slots: list[int] = []
    candidate_mask: list[bool] = []
    has_unverified_children: list[bool] = []
    packed_to_node: list[int | None] = []
    request_offsets = [0]
    anchor_rows: list[int] = []
    node_to_packed = [-1] * len(forest.nodes)
    request_to_slot = {
        request: slot for slot, request in enumerate(forest.requests)
    }
    selected_by_request: list[list[ForestNode]] = [
        [] for _ in forest.requests
    ]
    has_unselected_root = [False] * len(forest.requests)
    has_unselected_child = [False] * len(forest.nodes)
    for node in forest.nodes:
        request_slot = request_to_slot[node.request]
        if node.node_id in selected_set:
            selected_by_request[request_slot].append(node)
        elif node.parent_id is None:
            has_unselected_root[request_slot] = True
        else:
            has_unselected_child[node.parent_id] = True

    for request_slot, request in enumerate(forest.requests):
        anchor = anchor_by_request[request]
        anchor_row = len(token_ids)
        anchor_rows.append(anchor_row)

        token_ids.append(anchor.token_id)
        parent_rows.append(-1)
        first_child.append(-1)
        next_sibling.append(-1)
        last_child.append(-1)
        depths.append(0)
        positions.append(anchor.position)
        request_slots.append(request_slot)
        candidate_mask.append(False)
        has_unverified_children.append(has_unselected_root[request_slot])
        packed_to_node.append(None)

        for node in selected_by_request[request_slot]:
            packed_row = len(token_ids)
            node_to_packed[node.node_id] = packed_row

            if node.parent_id is None:
                parent_row = anchor_row
            else:
                parent_row = node_to_packed[node.parent_id]
                if parent_row == -1:
                    raise ForestValidationError(
                        f"selection is not prefix-closed at node {node.node_id}"
                    )

            token_ids.append(node.token_id)
            parent_rows.append(parent_row)
            first_child.append(-1)
            next_sibling.append(-1)
            last_child.append(-1)
            previous_child = last_child[parent_row]
            if previous_child == -1:
                first_child[parent_row] = packed_row
            else:
                next_sibling[previous_child] = packed_row
            last_child[parent_row] = packed_row
            depths.append(node.depth)
            positions.append(anchor.position + node.depth)
            request_slots.append(request_slot)
            candidate_mask.append(True)
            has_unverified_children.append(has_unselected_child[node.node_id])
            packed_to_node.append(node.node_id)
        request_offsets.append(len(token_ids))

    padding_rows = physical_rows - logical_rows
    token_ids.extend([padding_token_id] * padding_rows)
    parent_rows.extend([-1] * padding_rows)
    first_child.extend([-1] * padding_rows)
    next_sibling.extend([-1] * padding_rows)
    depths.extend([-1] * padding_rows)
    positions.extend([0] * padding_rows)
    request_slots.extend([-1] * padding_rows)
    candidate_mask.extend([False] * padding_rows)
    has_unverified_children.extend([False] * padding_rows)
    packed_to_node.extend([None] * padding_rows)
    valid_mask = [True] * logical_rows + [False] * padding_rows

    ancestor_row_ptr = [0]
    ancestor_columns: list[int] = []
    for row in range(logical_rows):
        parent = parent_rows[row]
        if parent != -1:
            parent_start = ancestor_row_ptr[parent]
            parent_end = ancestor_row_ptr[parent + 1]
            ancestor_columns.extend(ancestor_columns[parent_start:parent_end])
        ancestor_columns.append(row)
        ancestor_row_ptr.append(len(ancestor_columns))
    ancestor_row_ptr.extend([len(ancestor_columns)] * padding_rows)

    return VerifyPlan(
        token_ids=tuple(token_ids),
        parent_rows=tuple(parent_rows),
        first_child=tuple(first_child),
        next_sibling=tuple(next_sibling),
        depths=tuple(depths),
        positions=tuple(positions),
        request_slots=tuple(request_slots),
        valid_mask=tuple(valid_mask),
        candidate_mask=tuple(candidate_mask),
        has_unverified_children=tuple(has_unverified_children),
        packed_to_node=tuple(packed_to_node),
        node_to_packed=tuple(node_to_packed),
        request_offsets=tuple(request_offsets),
        request_refs=forest.requests,
        anchor_rows=tuple(anchor_rows),
        ancestor_row_ptr=tuple(ancestor_row_ptr),
        ancestor_columns=tuple(ancestor_columns),
        logical_rows=logical_rows,
        physical_rows=physical_rows,
    )
