from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, cast

from .profile import (
    ProfileEstimate,
    ProfileFingerprint,
    VerifyShape,
    VerifyShapeProfileDB,
)


@dataclass(frozen=True)
class BucketCandidate:


    shape: VerifyShape
    logical_num_tokens: int
    expected_progress: float
    selected_node_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.shape, VerifyShape):
            raise ValueError(f"shape must be VerifyShape, got {type(self.shape)!r}")
        selected_node_ids = tuple(self.selected_node_ids)
        object.__setattr__(self, "selected_node_ids", selected_node_ids)
        if (
            not isinstance(self.logical_num_tokens, int)
            or isinstance(self.logical_num_tokens, bool)
            or self.logical_num_tokens < self.shape.num_requests
        ):
            raise ValueError(
                "logical_num_tokens must include at least one anchor per request, "
                f"got logical={self.logical_num_tokens}, "
                f"requests={self.shape.num_requests}"
            )
        if self.logical_num_tokens > self.shape.physical_row_bucket:
            raise ValueError(
                "logical_num_tokens cannot exceed the physical row bucket, "
                f"got logical={self.logical_num_tokens}, "
                f"bucket={self.shape.physical_row_bucket}"
            )
        if (
            isinstance(self.expected_progress, bool)
            or not isinstance(self.expected_progress, (int, float))
            or not math.isfinite(self.expected_progress)
            or self.expected_progress < self.shape.num_requests
        ):
            raise ValueError(
                "expected_progress must be finite and include one base Target "
                f"token per request, got progress={self.expected_progress!r}, "
                f"requests={self.shape.num_requests}"
            )
        if any(
            not isinstance(node_id, int)
            or isinstance(node_id, bool)
            or node_id < 0
            for node_id in selected_node_ids
        ):
            raise ValueError("selected_node_ids must contain non-negative integers")
        if len(set(selected_node_ids)) != len(selected_node_ids):
            raise ValueError("selected_node_ids must be unique")
        expected_candidates = self.logical_num_tokens - self.shape.num_requests
        if len(selected_node_ids) != expected_candidates:
            raise ValueError(
                "selected_node_ids must account for every non-anchor logical row, "
                f"got selected={len(selected_node_ids)}, expected={expected_candidates}"
            )

    @property
    def physical_row_bucket(self) -> int:
        return self.shape.physical_row_bucket


@dataclass(frozen=True)
class BucketDecision:
    candidate: BucketCandidate
    estimate: ProfileEstimate
    score: float
    total_gpu_ms: float
    marginal_gpu_ms: Optional[float]


@dataclass(frozen=True, slots=True)
class _CandidateView:


    shape: VerifyShape
    logical_num_tokens: int
    expected_progress: float
    ordinal: int
    payload: object

    @property
    def physical_row_bucket(self) -> int:
        return self.shape.physical_row_bucket


@dataclass(frozen=True, slots=True)
class _CandidateSelection:
    view: _CandidateView
    estimate: ProfileEstimate
    score: float
    total_gpu_ms: float
    marginal_gpu_ms: Optional[float]


@dataclass(frozen=True, slots=True)
class _EvaluatedView:
    view: _CandidateView
    estimate: ProfileEstimate
    score: float
    total_gpu_ms: float


def _predecessor_across_physical_bucket(
    selected: _EvaluatedView,
    evaluated: Sequence[_EvaluatedView],
) -> Optional[_EvaluatedView]:


    lower_buckets = {
        item.view.physical_row_bucket
        for item in evaluated
        if item.view.physical_row_bucket < selected.view.physical_row_bucket
    }
    if not lower_buckets:
        return None
    previous_bucket = max(lower_buckets)
    same_bucket = [
        item
        for item in evaluated
        if item.view.physical_row_bucket == previous_bucket
    ]
    return max(
        same_bucket,
        key=lambda item: (
            item.view.logical_num_tokens,
            item.view.expected_progress,
            -item.view.ordinal,
        ),
    )


def _select_candidate_views(
    *,
    views: Sequence[_CandidateView],
    profile_db: VerifyShapeProfileDB,
    runtime_fingerprint: ProfileFingerprint,
    fixed_gpu_ms: float = 0.0,
) -> Optional[_CandidateSelection]:


    if (
        isinstance(fixed_gpu_ms, bool)
        or not isinstance(fixed_gpu_ms, (int, float))
        or not math.isfinite(fixed_gpu_ms)
        or fixed_gpu_ms < 0.0
    ):
        raise ValueError("fixed_gpu_ms must be finite and non-negative")
    fixed_gpu_ms = float(fixed_gpu_ms)
    if not views:
        return None

    reference_shape = views[0].shape
    round_key = (
        reference_shape.execution_mode,
        reference_shape.num_requests,
        reference_shape.total_context_tokens,
        reference_shape.max_context_tokens,
    )
    for view in views[1:]:
        candidate_round_key = (
            view.shape.execution_mode,
            view.shape.num_requests,
            view.shape.total_context_tokens,
            view.shape.max_context_tokens,
        )
        if candidate_round_key != round_key:
            raise ValueError(
                "all bucket candidates must describe the same verification round"
            )

    evaluated: list[_EvaluatedView] = []
    for view in views:
        estimate = profile_db.estimate(
            view.shape,
            runtime_fingerprint=runtime_fingerprint,
        )
        if estimate is None:
            continue
        total_gpu_ms = fixed_gpu_ms + estimate.p50_gpu_ms
        evaluated.append(
            _EvaluatedView(
                view=view,
                estimate=estimate,
                score=view.expected_progress / total_gpu_ms,
                total_gpu_ms=total_gpu_ms,
            )
        )

    if not evaluated:
        return None

    selected = max(
        evaluated,
        key=lambda item: (
            item.score,
            -item.view.physical_row_bucket,
            -item.view.logical_num_tokens,
            -item.view.ordinal,
        ),
    )
    predecessor = _predecessor_across_physical_bucket(selected, evaluated)
    marginal_gpu_ms = (
        None
        if predecessor is None
        else max(
            0.0,
            selected.estimate.p50_gpu_ms - predecessor.estimate.p50_gpu_ms,
        )
    )
    return _CandidateSelection(
        view=selected.view,
        estimate=selected.estimate,
        score=selected.score,
        total_gpu_ms=selected.total_gpu_ms,
        marginal_gpu_ms=marginal_gpu_ms,
    )


def select_verify_bucket(
    *,
    candidates: Sequence[BucketCandidate],
    profile_db: VerifyShapeProfileDB,
    runtime_fingerprint: ProfileFingerprint,
    fixed_gpu_ms: float = 0.0,
) -> Optional[BucketDecision]:


    selection = _select_candidate_views(
        views=tuple(
            _CandidateView(
                shape=candidate.shape,
                logical_num_tokens=candidate.logical_num_tokens,
                expected_progress=candidate.expected_progress,
                ordinal=ordinal,
                payload=candidate,
            )
            for ordinal, candidate in enumerate(candidates)
        ),
        profile_db=profile_db,
        runtime_fingerprint=runtime_fingerprint,
        fixed_gpu_ms=fixed_gpu_ms,
    )
    if selection is None:
        return None
    candidate = cast(BucketCandidate, selection.view.payload)
    return BucketDecision(
        candidate=candidate,
        estimate=selection.estimate,
        score=selection.score,
        total_gpu_ms=selection.total_gpu_ms,
        marginal_gpu_ms=selection.marginal_gpu_ms,
    )
