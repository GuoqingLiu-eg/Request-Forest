from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol, Sequence

from .ir import RequestRef, StateHandle
from .plans import ExpansionPlan


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class InvalidCandidateFragmentError(ValueError):
    pass


class TraceExhaustedError(LookupError):
    pass


@dataclass(frozen=True)
class DrafterCapabilities:
    supports_arbitrary_parent_state: bool
    max_fragment_nodes: int
    max_fanout: int
    provides_draft_features: bool = True
    supports_context_view: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.supports_arbitrary_parent_state, bool):
            raise ValueError("supports_arbitrary_parent_state must be bool")
        if not isinstance(self.provides_draft_features, bool):
            raise ValueError("provides_draft_features must be bool")
        if not isinstance(self.supports_context_view, bool):
            raise ValueError("supports_context_view must be bool")
        if not _is_plain_int(self.max_fragment_nodes) or self.max_fragment_nodes <= 0:
            raise ValueError("max_fragment_nodes must be positive")
        if not _is_plain_int(self.max_fanout) or self.max_fanout <= 0:
            raise ValueError("max_fanout must be positive")


@dataclass(frozen=True)
class CandidateFragmentNode:


    local_node_id: int
    parent_local_node_id: int | None
    token_id: int
    relative_depth: int
    draft_probability: float
    max_probability: float
    margin: float
    entropy: float
    state_handle: StateHandle | None = None
    valid: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise InvalidCandidateFragmentError("valid must be bool")
        if not self.valid:
            if (
                self.local_node_id != -1
                or self.parent_local_node_id is not None
                or self.token_id != -1
                or self.relative_depth != 0
                or self.state_handle is not None
                or self.draft_probability != 0.0
                or self.max_probability != 0.0
                or self.margin != 0.0
                or self.entropy != 0.0
            ):
                raise InvalidCandidateFragmentError(
                    "invalid fragment rows must use padding sentinels"
                )
            return
        if (
            not _is_plain_int(self.local_node_id)
            or not _is_plain_int(self.token_id)
            or not _is_plain_int(self.relative_depth)
            or self.local_node_id < 0
            or self.token_id < 0
            or self.relative_depth < 1
        ):
            raise InvalidCandidateFragmentError(
                "valid fragment nodes need non-negative IDs and positive depth"
            )
        if self.parent_local_node_id is not None and (
            not _is_plain_int(self.parent_local_node_id)
            or self.parent_local_node_id < 0
        ):
            raise InvalidCandidateFragmentError(
                "parent_local_node_id must be non-negative or None"
            )
        if self.state_handle is not None and (
            isinstance(self.state_handle, bool)
            or not isinstance(self.state_handle, (int, str))
        ):
            raise InvalidCandidateFragmentError(
                "state_handle must be an int, str, or None"
            )
        self._validate_features()

    def _validate_features(self) -> None:
        probabilities = (self.draft_probability, self.max_probability)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
            for value in probabilities
        ):
            raise InvalidCandidateFragmentError(
                "draft probabilities must be finite values in [0, 1]"
            )
        if self.draft_probability > self.max_probability:
            raise InvalidCandidateFragmentError(
                "draft_probability cannot exceed max_probability"
            )
        if (
            isinstance(self.margin, bool)
            or not isinstance(self.margin, (int, float))
            or not math.isfinite(self.margin)
            or not 0.0 <= self.margin <= 1.0
        ):
            raise InvalidCandidateFragmentError(
                "margin must be a finite value in [0, 1]"
            )
        if (
            isinstance(self.entropy, bool)
            or not isinstance(self.entropy, (int, float))
            or not math.isfinite(self.entropy)
            or self.entropy < 0.0
        ):
            raise InvalidCandidateFragmentError(
                "entropy must be a finite non-negative value"
            )

    @classmethod
    def padding(cls) -> CandidateFragmentNode:
        return cls(
            local_node_id=-1,
            parent_local_node_id=None,
            token_id=-1,
            relative_depth=0,
            draft_probability=0.0,
            max_probability=0.0,
            margin=0.0,
            entropy=0.0,
            state_handle=None,
            valid=False,
        )


@dataclass(frozen=True)
class CandidateFragment:
    action_index: int
    nodes: tuple[CandidateFragmentNode, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.action_index, int)
            or isinstance(self.action_index, bool)
            or self.action_index < 0
        ):
            raise InvalidCandidateFragmentError(
                "action_index must be a non-negative integer"
            )
        object.__setattr__(self, "nodes", tuple(self.nodes))
        saw_padding = False
        for node in self.nodes:
            if not isinstance(node, CandidateFragmentNode):
                raise InvalidCandidateFragmentError(
                    "fragment nodes must be CandidateFragmentNode values"
                )
            if not node.valid:
                saw_padding = True
            elif saw_padding:
                raise InvalidCandidateFragmentError(
                    "valid fragment rows must precede padding rows"
                )


@dataclass(frozen=True)
class CandidateFragmentBatch:
    fragments: tuple[CandidateFragment, ...]
    max_fragment_nodes: int

    def __post_init__(self) -> None:
        fragments = tuple(self.fragments)
        object.__setattr__(self, "fragments", fragments)
        if (
            not _is_plain_int(self.max_fragment_nodes)
            or self.max_fragment_nodes <= 0
        ):
            raise InvalidCandidateFragmentError(
                "max_fragment_nodes must be positive"
            )
        if not fragments:
            raise InvalidCandidateFragmentError("fragment batch width must be positive")
        for expected_index, fragment in enumerate(fragments):
            if not isinstance(fragment, CandidateFragment):
                raise InvalidCandidateFragmentError(
                    "fragments must contain CandidateFragment values"
                )
            if fragment.action_index != expected_index:
                raise InvalidCandidateFragmentError(
                    "fragment action indices must be dense and ordered"
                )
            if len(fragment.nodes) != self.max_fragment_nodes:
                raise InvalidCandidateFragmentError(
                    "every fragment must use the declared fixed physical shape"
                )

    @property
    def width(self) -> int:
        return len(self.fragments)


class ParallelDrafterAdapter(Protocol):


    @property
    def capabilities(self) -> DrafterCapabilities: ...

    def init_requests(self, requests: Sequence[RequestRef]) -> None: ...

    def expand(self, plan: ExpansionPlan) -> CandidateFragmentBatch: ...

    def after_target_commit(self, commit_results: Sequence[object]) -> None: ...

    def retire_request(self, request: RequestRef) -> None: ...


TraceKey = tuple[RequestRef, int | None]


class TraceDrafterAdapter:


    def __init__(
        self,
        *,
        traces: Mapping[TraceKey, Iterable[Iterable[CandidateFragmentNode]]],
        max_fragment_nodes: int,
        max_fanout: int,
        supports_arbitrary_parent_state: bool = True,
    ) -> None:
        self._capabilities = DrafterCapabilities(
            supports_arbitrary_parent_state=supports_arbitrary_parent_state,
            max_fragment_nodes=max_fragment_nodes,
            max_fanout=max_fanout,
        )
        self._traces = {
            key: deque(tuple(fragment) for fragment in fragments)
            for key, fragments in traces.items()
        }
        self._active_requests: set[RequestRef] = set()

    @property
    def capabilities(self) -> DrafterCapabilities:
        return self._capabilities

    def init_requests(self, requests: Sequence[RequestRef]) -> None:
        if any(not isinstance(request, RequestRef) for request in requests):
            raise ValueError("requests must contain RequestRef values")
        self._active_requests.update(requests)

    def expand(self, plan: ExpansionPlan) -> CandidateFragmentBatch:
        fragments = []
        for action_index, action in enumerate(plan.actions):
            if not action.valid:
                nodes = ()
            else:
                if action.request not in self._active_requests:
                    raise ValueError(f"request {action.request!r} is not active")
                if (
                    action.parent_id is not None
                    and not self.capabilities.supports_arbitrary_parent_state
                ):
                    raise ValueError(
                        "drafter does not support expansion from an arbitrary parent"
                    )
                key = (action.request, action.parent_id)
                queue = self._traces.get(key)
                if not queue:
                    raise TraceExhaustedError(f"no trace expansion remains for {key!r}")
                nodes = queue.popleft()
                self._validate_fragment(nodes)

            padded_nodes = tuple(nodes) + (CandidateFragmentNode.padding(),) * (
                self.capabilities.max_fragment_nodes - len(nodes)
            )
            fragments.append(
                CandidateFragment(action_index=action_index, nodes=padded_nodes)
            )

        return CandidateFragmentBatch(
            fragments=tuple(fragments),
            max_fragment_nodes=self.capabilities.max_fragment_nodes,
        )

    def after_target_commit(self, commit_results: Sequence[object]) -> None:
        del commit_results

    def retire_request(self, request: RequestRef) -> None:
        self._active_requests.discard(request)

    def _validate_fragment(
        self, nodes: Sequence[CandidateFragmentNode]
    ) -> None:
        if len(nodes) > self.capabilities.max_fragment_nodes:
            raise InvalidCandidateFragmentError(
                "fragment exceeds the declared max_fragment_nodes"
            )

        by_local_id: dict[int, CandidateFragmentNode] = {}
        sibling_tokens: set[tuple[int | None, int]] = set()
        fanout: dict[int | None, int] = {}
        for node in nodes:
            if not node.valid:
                raise InvalidCandidateFragmentError(
                    "trace fragments contain logical nodes only; padding is "
                    "added by the adapter"
                )
            if node.local_node_id < 0 or node.local_node_id in by_local_id:
                raise InvalidCandidateFragmentError(
                    f"duplicate or negative local node id {node.local_node_id}"
                )
            if node.token_id < 0:
                raise InvalidCandidateFragmentError("token_id must be non-negative")
            parent = None
            if node.parent_local_node_id is None:
                if node.relative_depth != 1:
                    raise InvalidCandidateFragmentError(
                        "a fragment root must have relative_depth=1"
                    )
            else:
                parent = by_local_id.get(node.parent_local_node_id)
                if parent is None:
                    raise InvalidCandidateFragmentError(
                        "a local parent must appear before its child"
                    )
                if node.relative_depth != parent.relative_depth + 1:
                    raise InvalidCandidateFragmentError(
                        "child relative_depth must equal parent depth plus one"
                    )

            sibling_key = (node.parent_local_node_id, node.token_id)
            if sibling_key in sibling_tokens:
                raise InvalidCandidateFragmentError(
                    "duplicate sibling token makes Target traversal ambiguous"
                )
            sibling_tokens.add(sibling_key)

            fanout[node.parent_local_node_id] = (
                fanout.get(node.parent_local_node_id, 0) + 1
            )
            if fanout[node.parent_local_node_id] > self.capabilities.max_fanout:
                raise InvalidCandidateFragmentError(
                    "fragment exceeds the declared max_fanout"
                )
            by_local_id[node.local_node_id] = node
