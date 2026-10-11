from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class PolicyNode:
    request_index: int
    parent_index: int
    token_id: int
    depth: int
    path_probability: float
    conditional_probability: float
    candidate_rank: int

@dataclass(frozen=True)
class PolicySelection:
    nodes: tuple[PolicyNode, ...]
    stats: dict[str, Any]
