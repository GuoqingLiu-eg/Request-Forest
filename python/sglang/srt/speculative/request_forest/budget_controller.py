from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch

from .profile import (
    ProfileEstimate,
    ProfileFingerprint,
    VerifyShape,
    VerifyShapeProfileDB,
)
from .target_timing import (
    TargetTimingObservation,
    TargetTimingScope,
)


class BudgetDecisionSource(str, Enum):


    STATIC_FALLBACK = "static_fallback"
    OFFLINE_PROFILE = "offline_profile"
    ONLINE_EMA = "online_ema"


@dataclass(frozen=True, slots=True)
class OnlineCostEstimate:


    sample_count: int
    last_gpu_ms: float
    ema_gpu_ms: float


@dataclass(frozen=True, slots=True)
class VerificationBudgetDecision:


    selected_node_ids: tuple[int, ...]
    shape: VerifyShape
    expected_progress: float
    source: BudgetDecisionSource
    profiled_shape: Optional[VerifyShape] = None
    fixed_draft_gpu_ms: float = 0.0
    estimated_gpu_ms: Optional[float] = None
    estimated_total_gpu_ms: Optional[float] = None
    goodput: Optional[float] = None
    marginal_gpu_ms: Optional[float] = None

    @property
    def num_candidate_nodes(self) -> int:
        return len(self.selected_node_ids)

    @property
    def logical_num_tokens(self) -> int:
        return self.shape.num_requests + self.num_candidate_nodes

    @property
    def physical_row_bucket(self) -> int:
        return self.shape.physical_row_bucket


@dataclass(frozen=True, slots=True)
class PreDraftBudgetDecision:


    use_target_only: bool
    target_only_shape: VerifyShape
    request_forest_shape: VerifyShape
    target_only_profiled_shape: VerifyShape
    request_forest_profiled_shape: VerifyShape
    target_only_total_gpu_ms: float
    request_forest_target_gpu_ms: float
    request_forest_total_gpu_ms: float
    target_only_goodput: float
    request_forest_goodput: float
    target_only_source: BudgetDecisionSource = BudgetDecisionSource.OFFLINE_PROFILE
    request_forest_source: BudgetDecisionSource = (
        BudgetDecisionSource.OFFLINE_PROFILE
    )


class _ExactOnlineProfileOverlay:


    def __init__(
        self,
        *,
        offline: VerifyShapeProfileDB,
        online: dict[VerifyShape, OnlineCostEstimate],
    ) -> None:
        self._offline = offline
        self._online = online

    def estimate(
        self,
        shape: VerifyShape,
        *,
        runtime_fingerprint: ProfileFingerprint,
    ) -> Optional[ProfileEstimate]:
        offline = self._offline.estimate(
            shape,
            runtime_fingerprint=runtime_fingerprint,
        )
        if offline is None:
            return None


        online = self._online.get(offline.profiled_shape)
        if online is not None:
            return ProfileEstimate(
                requested_shape=shape,
                profiled_shape=offline.profiled_shape,
                p50_gpu_ms=online.ema_gpu_ms,
                sample_count=online.sample_count,
                exact=offline.exact,
            )
        return offline


def _require_non_negative_int(value: int, *, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validated_path_values(path_values: Sequence[float]) -> tuple[float, ...]:

    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None and isinstance(path_values, torch.Tensor):
        if path_values.device.type != "cpu":
            raise ValueError("path_values tensor must reside on CPU")
        if path_values.dtype != torch.float64:
            raise ValueError("path_values tensor must use torch.float64")
        if path_values.ndim != 1:
            raise ValueError("path_values tensor must be one-dimensional")
        raw_values = path_values.tolist()
    else:
        if isinstance(path_values, (str, bytes)) or not isinstance(
            path_values, Sequence
        ):
            raise TypeError("path_values must be a sequence or CPU float64 tensor")
        raw_values = path_values

    values: list[float] = []
    for raw in raw_values:
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError("path_values must contain finite probabilities")
        value = float(raw)
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("path_values must contain probabilities in [0, 1]")
        values.append(value)
    return tuple(values)


def _validated_node_ids(
    ranked_node_ids: Optional[Sequence[int]],
    *,
    count: int,
) -> tuple[int, ...]:
    if ranked_node_ids is None:
        return tuple(range(count))
    if isinstance(ranked_node_ids, (str, bytes)) or not isinstance(
        ranked_node_ids, Sequence
    ):
        raise TypeError("ranked_node_ids must be a sequence")
    node_ids = tuple(ranked_node_ids)
    if len(node_ids) != count:
        raise ValueError("ranked_node_ids and path_values must have equal length")
    if any(
        not isinstance(node_id, int)
        or isinstance(node_id, bool)
        or node_id < 0
        for node_id in node_ids
    ):
        raise ValueError("ranked_node_ids must contain non-negative integers")
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("ranked_node_ids must be unique")
    return node_ids


class RequestForestBudgetController:


    def __init__(
        self,
        *,
        static_max_candidate_nodes: int,
        eager_row_budgets: Sequence[int],
        profile_db: Optional[VerifyShapeProfileDB],
        runtime_fingerprint: Optional[ProfileFingerprint],
        enabled: bool = True,
        online_ema_alpha: float = 0.2,
        fixed_draft_cost_ms: float = 0.0,
    ) -> None:
        _require_non_negative_int(
            static_max_candidate_nodes,
            name="static_max_candidate_nodes",
        )
        if isinstance(eager_row_budgets, (str, bytes)) or not isinstance(
            eager_row_budgets, Sequence
        ):
            raise TypeError("eager_row_budgets must be a sequence")
        if any(
            not isinstance(rows, int)
            or isinstance(rows, bool)
            or rows < 1
            for rows in eager_row_budgets
        ):
            raise ValueError("eager_row_budgets must contain positive integers")
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a bool")
        if profile_db is not None and not isinstance(
            profile_db, VerifyShapeProfileDB
        ):
            raise TypeError("profile_db must be VerifyShapeProfileDB or None")
        if runtime_fingerprint is not None and not isinstance(
            runtime_fingerprint, ProfileFingerprint
        ):
            raise TypeError(
                "runtime_fingerprint must be ProfileFingerprint or None"
            )
        if (
            isinstance(online_ema_alpha, bool)
            or not isinstance(online_ema_alpha, (int, float))
            or not math.isfinite(online_ema_alpha)
            or not 0.0 < online_ema_alpha <= 1.0
        ):
            raise ValueError("online_ema_alpha must be finite and in (0, 1]")
        if (
            isinstance(fixed_draft_cost_ms, bool)
            or not isinstance(fixed_draft_cost_ms, (int, float))
            or not math.isfinite(fixed_draft_cost_ms)
            or fixed_draft_cost_ms < 0.0
        ):
            raise ValueError(
                "fixed_draft_cost_ms must be finite and non-negative"
            )

        self.static_max_candidate_nodes = static_max_candidate_nodes
        self.eager_row_budgets = tuple(sorted(set(eager_row_budgets)))
        self.profile_db = profile_db
        self.runtime_fingerprint = runtime_fingerprint
        self.enabled = enabled
        self.online_ema_alpha = float(online_ema_alpha)
        self.fixed_draft_cost_ms = float(fixed_draft_cost_ms)
        self._online: dict[VerifyShape, OnlineCostEstimate] = {}
        self._profile_cells = (
            frozenset(point.shape for point in profile_db.points)
            if profile_db is not None
            else frozenset()
        )

    def ingest_observation(
        self, observation: TargetTimingObservation
    ) -> bool:


        if not isinstance(observation, TargetTimingObservation):
            raise TypeError("observation must be TargetTimingObservation")
        if (
            not self.enabled
            or self.profile_db is None
            or self.runtime_fingerprint is None
            or self.runtime_fingerprint != self.profile_db.fingerprint
            or observation.key.fingerprint != self.runtime_fingerprint
            or observation.key.scope != TargetTimingScope.VERIFICATION_STAGE
            or observation.key.shape.execution_mode
            not in ("eager", "cuda_graph")
            or observation.key.shape not in self._profile_cells
            or observation.key.shape.physical_row_bucket
            not in self.eager_row_budgets
            or not math.isfinite(observation.gpu_ms)
            or observation.gpu_ms <= 0.0
        ):
            return False

        previous = self._online.get(observation.key.shape)
        if previous is None:
            estimate = OnlineCostEstimate(
                sample_count=1,
                last_gpu_ms=float(observation.gpu_ms),
                ema_gpu_ms=float(observation.gpu_ms),
            )
        else:
            gpu_ms = float(observation.gpu_ms)
            estimate = OnlineCostEstimate(
                sample_count=previous.sample_count + 1,
                last_gpu_ms=gpu_ms,
                ema_gpu_ms=previous.ema_gpu_ms
                + self.online_ema_alpha * (gpu_ms - previous.ema_gpu_ms),
            )
        self._online[observation.key.shape] = estimate
        return True

    def online_estimate(self, shape: VerifyShape) -> Optional[OnlineCostEstimate]:


        if not isinstance(shape, VerifyShape):
            raise TypeError("shape must be VerifyShape")
        return self._online.get(shape)

    def choose_pre_draft_mode(
        self,
        *,
        target_only_shape: VerifyShape,
        request_forest_shape: VerifyShape,
        request_forest_path_values: Sequence[float],
    ) -> Optional[PreDraftBudgetDecision]:


        if not isinstance(target_only_shape, VerifyShape):
            raise TypeError("target_only_shape must be VerifyShape")
        if not isinstance(request_forest_shape, VerifyShape):
            raise TypeError("request_forest_shape must be VerifyShape")
        values = _validated_path_values(request_forest_path_values)
        round_key = (
            target_only_shape.num_requests,
            target_only_shape.total_context_tokens,
            target_only_shape.max_context_tokens,
        )
        if (
            request_forest_shape.num_requests,
            request_forest_shape.total_context_tokens,
            request_forest_shape.max_context_tokens,
        ) != round_key:
            raise ValueError(
                "target-only and Request Forest shapes must describe the same round"
            )
        num_requests = target_only_shape.num_requests
        if target_only_shape.physical_row_bucket < num_requests:
            raise ValueError("target-only shape must cover every request anchor")
        if request_forest_shape.physical_row_bucket < num_requests + len(values):
            raise ValueError(
                "Request Forest shape must cover every lagged candidate row"
            )
        if (
            not self.enabled
            or self.profile_db is None
            or self.runtime_fingerprint is None
        ):
            return None

        overlay = _ExactOnlineProfileOverlay(
            offline=self.profile_db,
            online=self._online,
        )
        target_only_estimate = overlay.estimate(
            target_only_shape,
            runtime_fingerprint=self.runtime_fingerprint,
        )
        request_forest_estimate = overlay.estimate(
            request_forest_shape,
            runtime_fingerprint=self.runtime_fingerprint,
        )
        if target_only_estimate is None or request_forest_estimate is None:
            return None

        target_only_total_gpu_ms = target_only_estimate.p50_gpu_ms
        request_forest_total_gpu_ms = (
            self.fixed_draft_cost_ms + request_forest_estimate.p50_gpu_ms
        )
        target_only_goodput = num_requests / target_only_total_gpu_ms
        request_forest_goodput = (
            num_requests + sum(values)
        ) / request_forest_total_gpu_ms
        target_only_source = (
            BudgetDecisionSource.ONLINE_EMA
            if target_only_estimate.profiled_shape in self._online
            else BudgetDecisionSource.OFFLINE_PROFILE
        )
        request_forest_source = (
            BudgetDecisionSource.ONLINE_EMA
            if request_forest_estimate.profiled_shape in self._online
            else BudgetDecisionSource.OFFLINE_PROFILE
        )
        return PreDraftBudgetDecision(
            use_target_only=target_only_goodput > request_forest_goodput,
            target_only_shape=target_only_shape,
            request_forest_shape=request_forest_shape,
            target_only_profiled_shape=target_only_estimate.profiled_shape,
            request_forest_profiled_shape=request_forest_estimate.profiled_shape,
            target_only_total_gpu_ms=target_only_total_gpu_ms,
            request_forest_target_gpu_ms=request_forest_estimate.p50_gpu_ms,
            request_forest_total_gpu_ms=request_forest_total_gpu_ms,
            target_only_goodput=target_only_goodput,
            request_forest_goodput=request_forest_goodput,
            target_only_source=target_only_source,
            request_forest_source=request_forest_source,
        )

    def choose(
        self,
        *,
        path_values: Sequence[float],
        num_requests: int,
        total_context_tokens: int,
        max_context_tokens: int,
        max_tree_depth: int = 0,
        ancestor_edges: int = 0,
        kv_page_runs: int = 0,
        ranked_node_ids: Optional[Sequence[int]] = None,
        node_depths: Optional[Sequence[int]] = None,
        execution_shape_resolver: Optional[
            Callable[[int, int], tuple[int, str]]
        ] = None,
        _trusted_internal: bool = False,
    ) -> VerificationBudgetDecision:


        _require_non_negative_int(num_requests, name="num_requests")
        if num_requests == 0:
            raise ValueError("num_requests must be positive")
        _require_non_negative_int(
            total_context_tokens, name="total_context_tokens"
        )
        _require_non_negative_int(max_context_tokens, name="max_context_tokens")
        _require_non_negative_int(max_tree_depth, name="max_tree_depth")
        _require_non_negative_int(ancestor_edges, name="ancestor_edges")
        _require_non_negative_int(kv_page_runs, name="kv_page_runs")
        if max_context_tokens > total_context_tokens:
            raise ValueError(
                "max_context_tokens cannot exceed total_context_tokens"
            )

        if _trusted_internal and self.profile_db is None and ranked_node_ids is None:


            fast = self._unprofiled_trusted_prefix(
                path_values=path_values,
                node_depths=node_depths,
                num_requests=num_requests,
                total_context_tokens=total_context_tokens,
                max_context_tokens=max_context_tokens,
                max_tree_depth=max_tree_depth,
                ancestor_edges=ancestor_edges,
                kv_page_runs=kv_page_runs,
                execution_shape_resolver=execution_shape_resolver,
            )
            if fast is not None:
                return fast

        if _trusted_internal:


            values = tuple(path_values.tolist())
            node_ids = (
                tuple(range(len(values)))
                if ranked_node_ids is None
                else tuple(ranked_node_ids)
            )
        else:
            values = _validated_path_values(path_values)
            node_ids = _validated_node_ids(ranked_node_ids, count=len(values))
        depths = None
        if node_depths is not None:
            if _trusted_internal:
                depths = tuple(node_depths.tolist())
            else:
                if isinstance(node_depths, (str, bytes)) or not isinstance(
                    node_depths, Sequence
                ):
                    raise TypeError("node_depths must be a sequence")
                if len(node_depths) != len(values):
                    raise ValueError("node_depths must align with path_values")
                depth_values: list[int] = []
                for depth in node_depths:
                    _require_non_negative_int(depth, name="node depth")
                    depth_values.append(int(depth))
                depths = tuple(depth_values)
        max_k = min(self.static_max_candidate_nodes, len(values))

        prefix_progress = [float(num_requests)]
        prefix_max_depth = [0]
        prefix_ancestor_edges = [num_requests]
        running = float(num_requests)
        running_max_depth = 0
        running_ancestor_edges = num_requests
        for index, value in enumerate(values[:max_k]):
            running += value
            prefix_progress.append(running)
            if depths is not None:
                depth = depths[index]
                running_max_depth = max(running_max_depth, depth)
                running_ancestor_edges += depth + 1
                prefix_max_depth.append(running_max_depth)
                prefix_ancestor_edges.append(running_ancestor_edges)

        def resolve_execution_shape(
            logical_rows: int,
            candidate_k: int,
        ) -> tuple[int, str]:
            candidate_max_depth = (
                max_tree_depth
                if depths is None
                else prefix_max_depth[candidate_k]
            )
            if execution_shape_resolver is None:
                physical_rows, execution_mode = logical_rows, "eager"
            else:
                resolved = execution_shape_resolver(
                    logical_rows,
                    candidate_max_depth,
                )
                if _trusted_internal:
                    physical_rows, execution_mode = resolved
                else:
                    if (
                        not isinstance(resolved, tuple)
                        or len(resolved) != 2
                    ):
                        raise ValueError(
                            "execution_shape_resolver must return (rows, mode)"
                        )
                    physical_rows, execution_mode = resolved
                    if (
                        not isinstance(physical_rows, int)
                        or isinstance(physical_rows, bool)
                        or physical_rows < logical_rows
                    ):
                        raise ValueError(
                            "resolved physical rows must cover logical rows"
                        )
                    if (
                        not isinstance(execution_mode, str)
                        or not execution_mode.strip()
                    ):
                        raise ValueError(
                            "execution_shape_resolver must return a non-empty mode"
                        )
            return physical_rows, execution_mode

        if _trusted_internal and self.profile_db is None:


            logical_rows = num_requests + max_k
            physical_rows, candidate_mode = resolve_execution_shape(logical_rows, max_k)
            if (
                not isinstance(physical_rows, int)
                or isinstance(physical_rows, bool)
                or physical_rows < logical_rows
            ):
                raise ValueError("resolved physical rows must cover selected logical rows")
            if candidate_mode not in ("eager", "cuda_graph"):
                raise ValueError("trusted static execution mode must be eager or cuda_graph")
            full_shape = self._shape(
                physical_rows=physical_rows,
                execution_mode=candidate_mode,
                num_requests=num_requests,
                total_context_tokens=total_context_tokens,
                max_context_tokens=max_context_tokens,
                max_tree_depth=max_tree_depth if depths is None else prefix_max_depth[max_k],
                ancestor_edges=ancestor_edges if depths is None else prefix_ancestor_edges[max_k],
                kv_page_runs=kv_page_runs,
            )
            return VerificationBudgetDecision(
                selected_node_ids=node_ids[:max_k],
                shape=full_shape,
                expected_progress=prefix_progress[max_k],
                source=BudgetDecisionSource.STATIC_FALLBACK,
            )

        execution_shapes = tuple(
            resolve_execution_shape(num_requests + k, k)
            for k in range(max_k + 1)
        )
        execution_mode = execution_shapes[0][1]
        if any(mode != execution_mode for _, mode in execution_shapes[1:]):
            raise ValueError(
                "all budget candidates must use one verification execution mode"
            )

        def make_shape(candidate_k: int) -> VerifyShape:
            physical_rows, candidate_mode = execution_shapes[candidate_k]
            candidate_max_depth = (
                max_tree_depth
                if depths is None
                else prefix_max_depth[candidate_k]
            )
            candidate_ancestor_edges = (
                ancestor_edges
                if depths is None
                else prefix_ancestor_edges[candidate_k]
            )
            return self._shape(
                physical_rows=physical_rows,
                execution_mode=candidate_mode,
                num_requests=num_requests,
                total_context_tokens=total_context_tokens,
                max_context_tokens=max_context_tokens,
                max_tree_depth=candidate_max_depth,
                ancestor_edges=candidate_ancestor_edges,
                kv_page_runs=kv_page_runs,
            )

        full_shape = make_shape(max_k)

        can_select = (
            self.enabled
            and self.profile_db is not None
            and self.runtime_fingerprint is not None
            and self.runtime_fingerprint == self.profile_db.fingerprint
            and full_shape.physical_row_bucket in self.eager_row_budgets
        )
        if can_select:
            overlay = _ExactOnlineProfileOverlay(
                offline=self.profile_db,
                online=self._online,
            )
            best = None
            fullest_by_bucket = {}
            interval_start = 0
            while interval_start <= max_k:
                physical_rows, mode = execution_shapes[interval_start]
                interval_end = interval_start
                while (
                    interval_end < max_k
                    and execution_shapes[interval_end + 1]
                    == (physical_rows, mode)
                ):
                    interval_end += 1

                k = interval_start
                while k <= interval_end:
                    query_shape = make_shape(k)
                    estimate = overlay.estimate(
                        query_shape,
                        runtime_fingerprint=self.runtime_fingerprint,
                    )
                    if estimate is None:


                        break

                    covered_end = interval_end
                    if depths is not None:
                        cell = estimate.profiled_shape
                        covered_end = min(
                            covered_end,
                            bisect_right(
                                prefix_max_depth,
                                cell.max_tree_depth,
                                k,
                                interval_end + 1,
                            )
                            - 1,
                            bisect_right(
                                prefix_ancestor_edges,
                                cell.ancestor_edges,
                                k,
                                interval_end + 1,
                            )
                            - 1,
                        )


                    interval_progress = prefix_progress[covered_end]
                    candidate_k = bisect_left(
                        prefix_progress,
                        interval_progress,
                        k,
                        covered_end + 1,
                    )
                    candidate_shape = make_shape(candidate_k)


                    candidate_estimate = estimate
                    total_gpu_ms = (
                        self.fixed_draft_cost_ms
                        + candidate_estimate.p50_gpu_ms
                    )
                    score = interval_progress / total_gpu_ms
                    candidate_key = (
                        score,
                        score,
                        -candidate_shape.physical_row_bucket,
                        -(num_requests + candidate_k),
                        -candidate_k,
                    )
                    if best is None or candidate_key > best[0]:
                        best = (
                            candidate_key,
                            candidate_k,
                            candidate_shape,
                            candidate_estimate,
                            score,
                            total_gpu_ms,
                        )


                    previous_fullest = fullest_by_bucket.get(physical_rows)
                    if previous_fullest is None or covered_end > previous_fullest[0]:
                        fullest_by_bucket[physical_rows] = (
                            covered_end,
                            estimate,
                        )
                    k = covered_end + 1
                interval_start = interval_end + 1

            if best is not None:
                (
                    _,
                    selected_k,
                    selected_shape,
                    selected_estimate,
                    selected_score,
                    selected_total_gpu_ms,
                ) = best
                lower_buckets = tuple(
                    bucket
                    for bucket in fullest_by_bucket
                    if bucket < selected_shape.physical_row_bucket
                )
                predecessor = (
                    None
                    if not lower_buckets
                    else fullest_by_bucket[max(lower_buckets)][1]
                )
                marginal_gpu_ms = (
                    None
                    if predecessor is None
                    else max(
                        0.0,
                        selected_estimate.p50_gpu_ms
                        - predecessor.p50_gpu_ms,
                    )
                )
                source = (
                    BudgetDecisionSource.ONLINE_EMA
                    if selected_estimate.profiled_shape in self._online
                    else BudgetDecisionSource.OFFLINE_PROFILE
                )
                return VerificationBudgetDecision(
                    selected_node_ids=node_ids[:selected_k],
                    shape=selected_shape,
                    expected_progress=prefix_progress[selected_k],
                    source=source,
                    profiled_shape=selected_estimate.profiled_shape,
                    fixed_draft_gpu_ms=self.fixed_draft_cost_ms,
                    estimated_gpu_ms=selected_estimate.p50_gpu_ms,
                    estimated_total_gpu_ms=selected_total_gpu_ms,
                    goodput=selected_score,
                    marginal_gpu_ms=marginal_gpu_ms,
                )

        return VerificationBudgetDecision(
            selected_node_ids=node_ids[:max_k],
            shape=full_shape,
            expected_progress=prefix_progress[max_k],
            source=BudgetDecisionSource.STATIC_FALLBACK,
        )

    def _unprofiled_trusted_prefix(
        self,
        *,
        path_values,
        node_depths,
        num_requests: int,
        total_context_tokens: int,
        max_context_tokens: int,
        max_tree_depth: int,
        ancestor_edges: int,
        kv_page_runs: int,
        execution_shape_resolver,
    ) -> Optional[VerificationBudgetDecision]:


        if (
            not isinstance(path_values, torch.Tensor)
            or path_values.ndim != 1
            or path_values.device.type != "cpu"
            or not path_values.dtype.is_floating_point
        ):
            return None
        count = int(path_values.numel())
        max_k = min(self.static_max_candidate_nodes, count)
        if node_depths is None:
            resolved_depth = max_tree_depth
            resolved_edges = ancestor_edges
        else:
            if (
                not isinstance(node_depths, torch.Tensor)
                or node_depths.ndim != 1
                or node_depths.device.type != "cpu"
                or node_depths.dtype != torch.int64
                or int(node_depths.numel()) != count
            ):
                return None
            if max_k == 0:
                resolved_depth = 0
                resolved_edges = num_requests
            else:
                depth_prefix = node_depths[:max_k]
                resolved_depth = int(depth_prefix.max().item())
                resolved_edges = (
                    num_requests + int(depth_prefix.sum().item()) + max_k
                )
        logical_rows = num_requests + max_k
        if execution_shape_resolver is None:
            physical_rows, candidate_mode = logical_rows, "eager"
        else:
            resolved = execution_shape_resolver(logical_rows, resolved_depth)
            if not isinstance(resolved, tuple) or len(resolved) != 2:
                raise ValueError(
                    "execution_shape_resolver must return (rows, mode)"
                )
            physical_rows, candidate_mode = resolved
        if (
            not isinstance(physical_rows, int)
            or isinstance(physical_rows, bool)
            or physical_rows < logical_rows
        ):
            raise ValueError(
                "resolved physical rows must cover selected logical rows"
            )
        if candidate_mode not in ("eager", "cuda_graph"):
            raise ValueError(
                "trusted static execution mode must be eager or cuda_graph"
            )
        if max_k == 0:
            expected_progress = float(num_requests)
        else:
            expected_progress = float(num_requests) + float(
                path_values[:max_k].sum().item()
            )
        return VerificationBudgetDecision(
            selected_node_ids=tuple(range(max_k)),
            shape=self._shape(
                physical_rows=physical_rows,
                execution_mode=candidate_mode,
                num_requests=num_requests,
                total_context_tokens=total_context_tokens,
                max_context_tokens=max_context_tokens,
                max_tree_depth=resolved_depth,
                ancestor_edges=resolved_edges,
                kv_page_runs=kv_page_runs,
            ),
            expected_progress=expected_progress,
            source=BudgetDecisionSource.STATIC_FALLBACK,
        )

    @staticmethod
    def _shape(
        *,
        physical_rows: int,
        execution_mode: str,
        num_requests: int,
        total_context_tokens: int,
        max_context_tokens: int,
        max_tree_depth: int,
        ancestor_edges: int,
        kv_page_runs: int,
    ) -> VerifyShape:
        return VerifyShape(
            physical_row_bucket=physical_rows,
            execution_mode=execution_mode,
            num_requests=num_requests,
            total_context_tokens=total_context_tokens,
            max_context_tokens=max_context_tokens,
            max_tree_depth=max_tree_depth,
            ancestor_edges=ancestor_edges,
            kv_page_runs=kv_page_runs,
        )


__all__ = [
    "BudgetDecisionSource",
    "OnlineCostEstimate",
    "PreDraftBudgetDecision",
    "RequestForestBudgetController",
    "VerificationBudgetDecision",
]
