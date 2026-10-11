from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from .external_policy_types import PolicyNode, PolicySelection


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite probability in [0, 1]")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite probability in [0, 1]")
    return value


def _sequence(value: Any, name: str) -> Sequence:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    return value


def _checkpoints(values: Mapping, *, json_keys: bool = False) -> dict[int, float]:
    if not isinstance(values, Mapping):
        raise ValueError("checkpoints must map depths to calibrated thresholds")
    result = {}
    for depth, threshold in values.items():
        if json_keys:
            if not isinstance(depth, str) or not depth.isdecimal():
                raise ValueError(
                    "profile checkpoint keys must be positive depth strings"
                )
            if str(int(depth)) != depth:
                raise ValueError(
                    "profile checkpoint depths must use canonical integers"
                )
            depth = int(depth)
        depth = _integer(depth, "checkpoint depth", minimum=1)
        result[depth] = _probability(threshold, f"threshold at depth {depth}")
    return result


def _validate_lattice(candidate_ids, edge_probabilities) -> tuple[int, int, int]:
    candidate_ids = _sequence(candidate_ids, "candidate_ids")
    edge_probabilities = _sequence(edge_probabilities, "edge_probabilities")
    batch_size = len(candidate_ids)
    if len(edge_probabilities) != batch_size:
        raise ValueError("candidate_ids and edge_probabilities disagree on batch size")
    depth_count = None
    support_size = None
    for request, (ids, edges) in enumerate(zip(candidate_ids, edge_probabilities)):
        ids = _sequence(ids, "request candidate_ids")
        edges = _sequence(edges, "request edge_probabilities")
        if depth_count is None:
            depth_count = len(ids)
        if len(ids) != depth_count or len(edges) != depth_count:
            raise ValueError("the lattice must have a common depth count E")
        for depth, (tokens, rows) in enumerate(zip(ids, edges)):
            tokens = _sequence(tokens, "candidate token row")
            rows = _sequence(rows, "predecessor rows")
            if support_size is None:
                support_size = len(tokens)
                if support_size == 0:
                    raise ValueError("nonempty depths require candidate support K > 0")
            if len(tokens) != support_size:
                raise ValueError("the lattice must have a common candidate support K")
            token_values = [_integer(token, "token_id") for token in tokens]
            if len(set(token_values)) != len(token_values):
                raise ValueError("candidate IDs must be unique within a depth")
            allowed_rows = (1, support_size) if depth == 0 else (support_size,)
            if len(rows) not in allowed_rows:
                raise ValueError("first depth needs P=1 or K; later depths need P=K")
            for row in rows:
                row = _sequence(row, "edge probability row")
                if len(row) != support_size:
                    raise ValueError("edge probability rows must have length K")
                probabilities = [
                    _probability(value, "edge probability") for value in row
                ]
                if not math.isclose(
                    math.fsum(probabilities), 1.0, rel_tol=1e-6, abs_tol=1e-6
                ):
                    raise ValueError("edge probabilities must already be normalized")
    return batch_size, depth_count or 0, support_size or 0


@dataclass(frozen=True)
class _Candidate:
    parent_index: int
    candidate_rank: int
    log_probability: float
    conditional_probability: float


def select_echo_batch(
    candidate_ids,
    edge_probabilities,
    *,
    capacity: int,
    checkpoints: dict[int, float],
    per_request_cap: int,
    initial_width: int = 1,
    max_width: int | None = None,
) -> PolicySelection:

    capacity = _integer(capacity, "capacity")
    per_request_cap = _integer(per_request_cap, "per_request_cap")
    initial_width = _integer(initial_width, "initial_width", minimum=1)
    gates = _checkpoints(checkpoints)
    batch_size, depth_count, support_size = _validate_lattice(
        candidate_ids, edge_probabilities
    )
    if depth_count and any(depth > depth_count for depth in gates):
        raise ValueError("checkpoint depth exceeds the available draft window")
    if max_width is None:
        max_width = max(support_size, initial_width)
    max_width = _integer(max_width, "max_width", minimum=1)
    if max_width < initial_width:
        raise ValueError("max_width must be >= initial_width")

    nodes: list[PolicyNode] = []
    node_logs: list[float] = []
    counts = [0] * batch_size
    depths = [0] * batch_size
    widths = [0] * batch_size
    frontiers = [[-1] for _ in range(batch_size)]
    active = list(range(batch_size)) if per_request_cap else []
    truncated: dict[int, tuple[int, list[_Candidate]]] = {}
    gate_events: list[dict[str, Any]] = []
    depth_nodes = 0
    width_nodes = 0

    def expand(request: int, depth: int) -> list[_Candidate]:
        candidates = []
        for parent_index in frontiers[request]:
            rank = 0 if parent_index == -1 else nodes[parent_index].candidate_rank
            parent_log = 0.0 if parent_index == -1 else node_logs[parent_index]
            for candidate_rank, probability in enumerate(
                edge_probabilities[request][depth - 1][rank]
            ):
                probability = float(probability)
                log_q = math.log(probability) if probability else -math.inf
                candidates.append(
                    _Candidate(
                        parent_index, candidate_rank, parent_log + log_q, probability
                    )
                )
        candidates.sort(
            key=lambda item: (
                -item.log_probability,
                item.parent_index,
                item.candidate_rank,
            )
        )
        return candidates

    def append(request: int, depth: int, candidates: list[_Candidate], limit: int):
        take = min(limit, capacity - len(nodes), per_request_cap - counts[request])
        selected_indices = []
        for candidate in candidates[:take]:
            selected_indices.append(len(nodes))
            nodes.append(
                PolicyNode(
                    request_index=request,
                    parent_index=candidate.parent_index,
                    token_id=int(
                        candidate_ids[request][depth - 1][candidate.candidate_rank]
                    ),
                    depth=depth,
                    path_probability=math.exp(candidate.log_probability),
                    conditional_probability=candidate.conditional_probability,
                    candidate_rank=candidate.candidate_rank,
                )
            )
            node_logs.append(candidate.log_probability)
        counts[request] += len(selected_indices)
        if selected_indices:
            depths[request] = max(depths[request], depth)
            widths[request] = max(widths[request], len(selected_indices))
        return selected_indices


    for depth in range(1, depth_count + 1):
        if not active or len(nodes) == capacity:
            break
        next_active = []
        for request in active:
            if len(nodes) == capacity:
                break
            if counts[request] == per_request_cap:
                continue
            candidates = expand(request, depth)
            if depth in gates:
                threshold = gates[depth]
                log_threshold = math.log(threshold) if threshold else -math.inf
                passed = candidates[0].log_probability > log_threshold
                gate_events.append(
                    {
                        "request_index": request,
                        "depth": depth,
                        "confidence": math.exp(candidates[0].log_probability),
                        "threshold": threshold,
                        "passed": passed,
                    }
                )
                if not passed:
                    truncated[request] = (depth, candidates)
                    continue
            frontiers[request] = append(request, depth, candidates, initial_width)
            depth_nodes += len(frontiers[request])
            if frontiers[request] and counts[request] < per_request_cap:
                next_active.append(request)
        active = next_active


    for request in sorted(truncated):
        if len(nodes) == capacity:
            break
        depth, candidates = truncated[request]
        width_nodes += len(append(request, depth, candidates, max_width))

    equal_shares = (
        [
            capacity // batch_size + (i < capacity % batch_size)
            for i in range(batch_size)
        ]
        if batch_size
        else []
    )
    stats = {
        "policy": "echo-style",
        "capacity": capacity,
        "selected_nodes": len(nodes),
        "unused_capacity": capacity - len(nodes),
        "per_request_counts": counts,
        "per_request_depths": depths,
        "per_request_max_widths": widths,
        "gate_checks": len(gate_events),
        "gate_failures": sum(not event["passed"] for event in gate_events),
        "gate_events": gate_events,
        "depth_nodes": depth_nodes,
        "width_nodes": width_nodes,
        "equal_share_reference": equal_shares,
        "borrowed_nodes": sum(
            max(0, used - share) for used, share in zip(counts, equal_shares)
        ),
    }
    return PolicySelection(nodes=tuple(nodes), stats=stats)


def _auc(samples: list[tuple[float, bool]]) -> float:

    samples = sorted(samples, key=lambda sample: sample[0])
    positives = sum(label for _, label in samples)
    negatives = len(samples) - positives
    negatives_below = 0
    favorable = 0.0
    index = 0
    while index < len(samples):
        end = index + 1
        while end < len(samples) and samples[end][0] == samples[index][0]:
            end += 1
        positive_ties = sum(label for _, label in samples[index:end])
        negative_ties = end - index - positive_ties
        favorable += positive_ties * (negatives_below + 0.5 * negative_ties)
        negatives_below += negative_ties
        index = end
    return favorable / (positives * negatives)


def _calibrated_threshold(samples: list[tuple[float, bool]]) -> tuple[float, float]:

    samples = sorted(samples, key=lambda sample: sample[0])
    unique = sorted({score for score, _ in samples})
    thresholds = sorted(
        {0.0, 1.0, *[(left + right) / 2.0 for left, right in zip(unique, unique[1:])]}
    )
    positives = sum(label for _, label in samples)
    negatives = len(samples) - positives
    best_threshold, best_j = 0.0, -math.inf
    true_positives = positives
    false_positives = negatives
    index = 0
    for threshold in thresholds:
        while index < len(samples) and samples[index][0] <= threshold:
            if samples[index][1]:
                true_positives -= 1
            else:
                false_positives -= 1
            index += 1
        objective = true_positives / positives - false_positives / negatives
        if objective > best_j:
            best_threshold, best_j = threshold, objective
    return best_threshold, best_j


def calibrate_echo_checkpoints(
    records: Iterable[Mapping[str, Any]],
    *,
    min_auc: float,
    min_samples_per_class: int = 2,
    max_checkpoints: int | None = None,
) -> dict[str, Any]:

    min_auc = _probability(min_auc, "min_auc")
    if min_auc < 0.5:
        raise ValueError("min_auc must be >= 0.5 for high-confidence pass gates")
    min_samples_per_class = _integer(
        min_samples_per_class, "min_samples_per_class", minimum=1
    )
    if max_checkpoints is not None:
        max_checkpoints = _integer(max_checkpoints, "max_checkpoints", minimum=2)
    grouped = defaultdict(list)
    record_count = 0
    for record in records:
        if not isinstance(record, Mapping) or record.get("split") != "calibration":
            raise ValueError("all records must explicitly have split='calibration'")
        depth = _integer(record.get("depth"), "calibration depth", minimum=1)
        confidence = _probability(record.get("confidence"), "calibration confidence")
        accepted = record.get("accepted")
        if not isinstance(accepted, bool):
            raise ValueError("accepted must be a boolean verification label")
        grouped[depth].append((confidence, accepted))
        record_count += 1

    checkpoints = {}
    statistics = {}
    for depth in sorted(grouped):
        samples = grouped[depth]
        positives = sum(label for _, label in samples)
        negatives = len(samples) - positives
        detail = {"accepted": positives, "rejected": negatives, "selected": False}
        statistics[str(depth)] = detail
        if min(positives, negatives) < min_samples_per_class:
            detail["reason"] = "insufficient_class_support"
            continue
        auc = _auc(samples)
        detail["auc"] = auc
        if auc > min_auc:
            threshold, objective = _calibrated_threshold(samples)
            checkpoints[str(depth)] = threshold
            detail.update(selected=True, threshold=threshold, youden_j=objective)
        else:
            detail["reason"] = "auc_not_above_cutoff"
    if not checkpoints:
        raise ValueError("no calibration depth has sufficient support and AUC")
    if max_checkpoints is not None and len(checkpoints) > max_checkpoints:
        endpoints = {min(grouped), max(grouped)}
        retained = [d for d in sorted(endpoints) if str(d) in checkpoints]
        interior = sorted(
            (int(d) for d in checkpoints if int(d) not in endpoints),
            key=lambda d: (-statistics[str(d)]["auc"], d),
        )
        retained.extend(interior[:max_checkpoints - len(retained)])
        for depth in list(checkpoints):
            if int(depth) not in retained:
                del checkpoints[depth]
                statistics[depth].update(selected=False, reason="sparse_checkpoint_budget")
        checkpoints = dict(sorted(checkpoints.items(), key=lambda item: int(item[0])))
    return {
        "schema_version": 1,
        "policy": "echo-style",
        "calibration_split": "calibration",
        "calibration_record_count": record_count,
        "min_auc": min_auc,
        "min_samples_per_class": min_samples_per_class,
        "max_checkpoints": max_checkpoints,
        "checkpoint_selection": "eligible endpoints, then descending calibration AUC; shallow ties",
        "threshold_objective": "youden_j; ties choose smaller threshold",
        "checkpoints": checkpoints,
        "depth_statistics": statistics,
    }


def load_echo_profile(path: str | Path) -> dict[int, float]:

    with Path(path).open(encoding="utf-8") as handle:
        profile = json.load(handle)
    if not isinstance(profile, dict):
        raise ValueError("ECHO profile must be a JSON object")
    version = profile.get("schema_version")
    if (
        type(version) is not int
        or version != 1
        or profile.get("policy") != "echo-style"
    ):
        raise ValueError("unsupported ECHO profile schema or policy")
    if profile.get("calibration_split") != "calibration":
        raise ValueError("ECHO profile must identify its calibration split")
    _integer(
        profile.get("calibration_record_count"), "calibration_record_count", minimum=2
    )
    checkpoints = _checkpoints(profile.get("checkpoints"), json_keys=True)
    if not checkpoints:
        raise ValueError("ECHO profile contains no calibrated checkpoints")
    return checkpoints
