from __future__ import annotations

import math
from collections.abc import Mapping

from .ir import ForestNode, ForestNodeArena


class CandidateSelectionError(ValueError):
    pass


class NonMonotoneValueError(CandidateSelectionError):
    pass


def _validated_values(
    nodes: tuple[ForestNode, ...], node_values: Mapping[int, float]
) -> list[float]:
    node_count = len(nodes)
    missing = [
        node_id for node_id in range(node_count) if node_id not in node_values
    ]
    if missing:
        raise CandidateSelectionError(
            f"missing values for node IDs: {missing}"
        )


    if len(node_values) > node_count:
        unknown = set(node_values).difference(range(node_count))
        if unknown:
            raise CandidateSelectionError(
                f"values contain unknown node IDs: {sorted(unknown)}"
            )

    values = [0.0] * node_count
    for node_id in range(node_count):
        raw_value = node_values[node_id]
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise CandidateSelectionError(
                f"value for node {node_id} must be a finite number"
            )
        try:
            value = float(raw_value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise CandidateSelectionError(
                f"value for node {node_id} must be a finite number"
            ) from exc
        if not math.isfinite(value):
            raise CandidateSelectionError(
                f"value for node {node_id} must be finite, got {value!r}"
            )
        values[node_id] = value

    return values


def _validate_monotone_edges(
    nodes: tuple[ForestNode, ...], values: list[float]
) -> None:
    for node in nodes:
        if node.parent_id is None:
            continue
        if values[node.node_id] > values[node.parent_id]:
            raise NonMonotoneValueError(
                "prefix-closed selection requires child value <= parent "
                f"value, but node {node.node_id} has {values[node.node_id]!r} "
                f"and parent {node.parent_id} has {values[node.parent_id]!r}"
            )


def select_prefix_closed(
    arena: ForestNodeArena,
    node_values: Mapping[int, float],
    capacity: int,
) -> tuple[int, ...]:


    if isinstance(capacity, bool) or not isinstance(capacity, int):
        raise TypeError("capacity must be an integer")
    if capacity < 0:
        raise ValueError("capacity must be non-negative")

    nodes = tuple(arena)
    values = _validated_values(nodes, node_values)
    _validate_monotone_edges(nodes, values)

    if capacity == 0 or not nodes:
        return ()

    ranked_ids = sorted(
        range(len(nodes)), key=values.__getitem__, reverse=True
    )
    limit = min(capacity, len(ranked_ids))
    selected: list[int] = []
    for node_id in ranked_ids:
        if len(selected) == limit or values[node_id] <= 0.0:
            break
        selected.append(node_id)

    return tuple(selected)


__all__ = [
    "CandidateSelectionError",
    "NonMonotoneValueError",
    "select_prefix_closed",
]
