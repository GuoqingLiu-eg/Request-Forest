from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence, Union


StateHandle = Union[int, str]


class ForestValidationError(ValueError):
    pass


def _require_plain_int(name: str, value: int, *, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ForestValidationError(
            f"{name} must be an int >= {minimum}, got {value!r}"
        )


def _validate_probability(name: str, value: Optional[float]) -> None:
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ForestValidationError(
            f"{name} must be finite and in [0, 1], got {value!r}"
        )


def _validate_finite(name: str, value: Optional[float], *, minimum=None) -> None:
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (minimum is not None and value < minimum)
    ):
        suffix = "" if minimum is None else f" and >= {minimum}"
        raise ForestValidationError(
            f"{name} must be finite{suffix}, got {value!r}"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RequestRef:


    request_id: str
    generation: int
    forest_epoch: int

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ForestValidationError(
                f"request_id must be a non-empty string, got {self.request_id!r}"
            )
        _require_plain_int("generation", self.generation)
        _require_plain_int("forest_epoch", self.forest_epoch)


@dataclass(frozen=True, slots=True, kw_only=True)
class ForestNode:


    node_id: int
    request: RequestRef
    parent_id: Optional[int]
    token_id: int
    depth: int
    state_handle: Optional[StateHandle] = None
    draft_probability: Optional[float] = None
    conditional_acceptance: Optional[float] = None
    path_reach_probability: Optional[float] = None
    margin: Optional[float] = None
    entropy: Optional[float] = None
    sibling_rank: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.request, RequestRef):
            raise ForestValidationError("request must be a RequestRef")
        _require_plain_int("node_id", self.node_id)
        _require_plain_int("token_id", self.token_id)
        _require_plain_int("depth", self.depth, minimum=1)
        if self.parent_id is not None:
            _require_plain_int("parent_id", self.parent_id)
        if self.state_handle is not None and (
            isinstance(self.state_handle, bool)
            or not isinstance(self.state_handle, (int, str))
        ):
            raise ForestValidationError(
                "state_handle must be an int, str, or None, got "
                f"{self.state_handle!r}"
            )
        _validate_probability("draft_probability", self.draft_probability)
        _validate_probability("conditional_acceptance", self.conditional_acceptance)
        _validate_probability("path_reach_probability", self.path_reach_probability)
        _validate_finite("margin", self.margin, minimum=0.0)
        _validate_finite("entropy", self.entropy, minimum=0.0)
        if self.sibling_rank is not None:
            _require_plain_int("sibling_rank", self.sibling_rank)


@dataclass(frozen=True, slots=True, kw_only=True)
class ForestIR:


    requests: tuple[RequestRef, ...]
    nodes: tuple[ForestNode, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "requests", tuple(self.requests))
        object.__setattr__(self, "nodes", tuple(self.nodes))

        if not self.requests:
            raise ForestValidationError("ForestIR requires at least one request")
        if any(not isinstance(request, RequestRef) for request in self.requests):
            raise ForestValidationError("ForestIR requests must be RequestRef values")
        if len(set(self.requests)) != len(self.requests):
            raise ForestValidationError("ForestIR request references must be unique")
        request_ids = [request.request_id for request in self.requests]
        if len(set(request_ids)) != len(request_ids):
            raise ForestValidationError(
                "ForestIR cannot contain multiple generations or epochs for "
                "the same request_id"
            )
        if any(not isinstance(node, ForestNode) for node in self.nodes):
            raise ForestValidationError("ForestIR nodes must be ForestNode values")

        request_set = set(self.requests)
        sibling_tokens: set[tuple[RequestRef, Optional[int], int]] = set()
        for expected_id, node in enumerate(self.nodes):
            if node.node_id != expected_id:
                raise ForestValidationError(
                    "ForestIR node ids must be dense and match tuple order: "
                    f"expected {expected_id}, got {node.node_id}"
                )
            if node.request not in request_set:
                raise ForestValidationError(
                    f"node {node.node_id} references unknown request {node.request!r}"
                )

            if node.parent_id is None:
                expected_depth = 1
            else:
                if node.parent_id >= node.node_id:
                    raise ForestValidationError(
                        f"node {node.node_id} parent {node.parent_id} must "
                        "appear earlier"
                    )
                parent = self.nodes[node.parent_id]
                if parent.request != node.request:
                    raise ForestValidationError(
                        f"node {node.node_id} has a cross-request parent "
                        f"{node.parent_id}"
                    )
                expected_depth = parent.depth + 1
            if node.depth != expected_depth:
                raise ForestValidationError(
                    f"node {node.node_id} depth {node.depth} != expected "
                    f"{expected_depth}"
                )

            sibling_key = (node.request, node.parent_id, node.token_id)
            if sibling_key in sibling_tokens:
                raise ForestValidationError(
                    "duplicate sibling token for request "
                    f"{node.request.request_id!r}, parent={node.parent_id}, "
                    f"token={node.token_id}"
                )
            sibling_tokens.add(sibling_key)

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self) -> Iterator[ForestNode]:
        return iter(self.nodes)

    def get(self, node_id: int) -> ForestNode:
        _require_plain_int("node_id", node_id)
        try:
            return self.nodes[node_id]
        except IndexError as exc:
            raise ForestValidationError(f"unknown node_id {node_id}") from exc

    def children_of(
        self, request: RequestRef, parent_id: Optional[int]
    ) -> tuple[ForestNode, ...]:
        return tuple(
            node
            for node in self.nodes
            if node.request == request and node.parent_id == parent_id
        )

    def roots_for(self, request: RequestRef) -> tuple[ForestNode, ...]:
        return self.children_of(request, None)


class ForestNodeArena:


    def __init__(self, requests: Sequence[RequestRef]):
        request_tuple = tuple(requests)

        ForestIR(requests=request_tuple)
        self._requests = request_tuple
        self._request_set = set(request_tuple)
        self._nodes: list[ForestNode] = []
        self._children: dict[
            tuple[RequestRef, Optional[int]], list[int]
        ] = {}
        self._sibling_tokens: set[tuple[RequestRef, Optional[int], int]] = set()

    @property
    def requests(self) -> tuple[RequestRef, ...]:
        return self._requests

    @property
    def nodes(self) -> tuple[ForestNode, ...]:
        return tuple(self._nodes)

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self) -> Iterator[ForestNode]:
        return iter(self._nodes)

    def get(self, node_id: int) -> ForestNode:
        _require_plain_int("node_id", node_id)
        try:
            return self._nodes[node_id]
        except IndexError as exc:
            raise ForestValidationError(f"unknown node_id {node_id}") from exc

    def add_node(
        self,
        *,
        request: RequestRef,
        parent_id: Optional[int],
        token_id: int,
        state_handle: Optional[StateHandle] = None,
        draft_probability: Optional[float] = None,
        conditional_acceptance: Optional[float] = None,
        path_reach_probability: Optional[float] = None,
        margin: Optional[float] = None,
        entropy: Optional[float] = None,
        sibling_rank: Optional[int] = None,
    ) -> ForestNode:
        if request not in self._request_set:
            raise ForestValidationError(
                f"unknown or stale request reference {request!r}"
            )
        _require_plain_int("token_id", token_id)

        if parent_id is None:
            depth = 1
        else:
            parent = self.get(parent_id)
            if parent.request != request:
                raise ForestValidationError(
                    f"parent {parent_id} belongs to a different request"
                )
            depth = parent.depth + 1

        sibling_key = (request, parent_id, token_id)
        if sibling_key in self._sibling_tokens:
            raise ForestValidationError(
                "duplicate sibling token for request "
                f"{request.request_id!r}, parent={parent_id}, token={token_id}"
            )

        node = ForestNode(
            node_id=len(self._nodes),
            request=request,
            parent_id=parent_id,
            token_id=token_id,
            depth=depth,
            state_handle=state_handle,
            draft_probability=draft_probability,
            conditional_acceptance=conditional_acceptance,
            path_reach_probability=path_reach_probability,
            margin=margin,
            entropy=entropy,
            sibling_rank=sibling_rank,
        )
        self._nodes.append(node)
        self._children.setdefault((request, parent_id), []).append(node.node_id)
        self._sibling_tokens.add(sibling_key)
        return node

    def children_of(
        self, request: RequestRef, parent_id: Optional[int]
    ) -> tuple[ForestNode, ...]:
        return tuple(
            self._nodes[node_id]
            for node_id in self._children.get((request, parent_id), ())
        )

    def roots_for(self, request: RequestRef) -> tuple[ForestNode, ...]:
        return self.children_of(request, None)

    def freeze(self) -> ForestIR:
        return ForestIR(requests=self._requests, nodes=tuple(self._nodes))
