from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Sequence
from numbers import Real

from .external_policy_types import PolicyNode, PolicySelection

CycleCostModel = Callable[[int, int, int], float]
_MAX_TOP_K = 128
_PROBABILITY_SUM_TOLERANCE = 1e-6


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _sequence(value: object, name: str) -> Sequence:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    return value


def _validated_candidates(candidate_ids, marginal_probabilities):
    ids = _sequence(candidate_ids, "candidate_ids")
    probs = _sequence(marginal_probabilities, "marginal_probabilities")
    if len(ids) != len(probs):
        raise ValueError("candidate_ids and marginal_probabilities batch sizes differ")
    result = []
    expected_depth = expected_width = None
    for request_index, (request_ids, request_probs) in enumerate(zip(ids, probs)):
        request_ids = _sequence(request_ids, f"candidate_ids[{request_index}]")
        request_probs = _sequence(request_probs, f"probabilities[{request_index}]")
        if expected_depth is None:
            expected_depth = len(request_ids)
        if len(request_ids) != expected_depth or len(request_probs) != expected_depth:
            raise ValueError(
                "candidate tensors must have matching rectangular [B,E,K] shapes"
            )
        rows = []
        for token_row, prob_row in zip(request_ids, request_probs):
            token_row = _sequence(token_row, "candidate token row")
            prob_row = _sequence(prob_row, "candidate probability row")
            if expected_width is None:
                expected_width = len(token_row)
            if len(token_row) != expected_width or len(prob_row) != expected_width:
                raise ValueError(
                    "candidate tensors must have matching rectangular [B,E,K] shapes"
                )
            row = []
            seen = set()
            for rank, (token, probability) in enumerate(zip(token_row, prob_row)):
                _integer(token, "candidate token id")
                if token in seen:
                    raise ValueError(
                        "candidate token ids must be unique within each position"
                    )
                seen.add(token)
                if isinstance(probability, bool) or not isinstance(probability, Real):
                    raise ValueError("marginal probabilities must be real numbers")
                try:
                    probability = float(probability)
                except OverflowError as exc:
                    raise ValueError("marginal probabilities must be finite") from exc
                if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                    raise ValueError(
                        "marginal probabilities must be finite and in [0,1]"
                    )
                row.append((token, probability, rank))
            if math.fsum(p for _, p, _ in row) > 1.0 + _PROBABILITY_SUM_TOLERANCE:
                raise ValueError(
                    "retained marginal probabilities must sum to at most one"
                )

            row.sort(key=lambda item: (-item[1], item[2]))
            rows.append(row[:_MAX_TOP_K])
        result.append(rows)
    return result


def _select_request(
    rows,
    *,
    request_index: int,
    quota: int,
    context_length: int,
    cost_model: CycleCostModel,
    minimum: int,
    maximum: int,
    bucket_step: int,
):
    maximum = min(maximum, quota + 1)

    effective_minimum = min(minimum, maximum)
    nodes = []
    node_log_probabilities = []
    costs = {}
    return_size = effective_minimum
    path_probability_sum = 1.0
    stop_reason = "no_candidates"

    def cost(total_nodes: int) -> float:
        if total_nodes not in costs:
            value = cost_model(request_index, total_nodes, context_length)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError("cost_model must return a real cycle duration")
            try:
                value = float(value)
            except OverflowError as exc:
                raise ValueError("cost_model cycle duration must be finite") from exc
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    "cost_model cycle duration must be positive and finite"
                )
            previous = costs.get(total_nodes - 1)
            if previous is not None and value < previous:
                raise ValueError("cost_model must be nondecreasing in total tree nodes")
            costs[total_nodes] = value
        return costs[total_nodes]


    heap = []
    serial = 0

    def push(parent_index: int, depth: int, rank: int, path_log_probability=None):
        nonlocal serial
        probability = rows[depth - 1][rank][1]
        node_log = math.log(probability) if probability else -math.inf
        if path_log_probability is None:
            parent_log = (
                0.0 if parent_index < 0 else node_log_probabilities[parent_index]
            )
            path_log_probability = parent_log + node_log
        heapq.heappush(
            heap, (-path_log_probability, serial, parent_index, depth, rank, node_log)
        )
        serial += 1

    if maximum > 1 and rows and rows[0]:
        cost(1)
        push(-1, 1, 0)
        while heap:
            negative_log, _, parent, depth, rank, node_log = heapq.heappop(heap)
            path_log = -negative_log
            probability = math.exp(path_log)
            total_nodes = len(nodes) + 1
            previous_cost = cost(total_nodes)
            next_cost = cost(total_nodes + 1)
            if (
                probability * previous_cost
                <= path_probability_sum * (next_cost - previous_cost)
                and total_nodes >= effective_minimum
            ):
                stop_reason = "marginal_cost"
                break
            token, conditional_probability, original_rank = rows[depth - 1][rank]
            nodes.append(
                PolicyNode(
                    request_index=request_index,
                    parent_index=parent,
                    token_id=token,
                    depth=depth,
                    path_probability=probability,
                    conditional_probability=conditional_probability,
                    candidate_rank=original_rank,
                )
            )
            node_log_probabilities.append(path_log)
            path_probability_sum += probability
            current_size = len(nodes) + 1
            if current_size >= return_size + bucket_step:
                return_size += bucket_step
            if current_size == maximum:
                stop_reason = "node_limit"
                break
            if depth < len(rows):
                push(len(nodes) - 1, depth + 1, 0)
            if rank + 1 < len(rows[depth - 1]):
                sibling_probability = rows[depth - 1][rank + 1][1]
                sibling_log = (
                    math.log(sibling_probability) if sibling_probability else -math.inf
                )


                sibling_path_log = (
                    path_log - node_log + sibling_log
                    if math.isfinite(node_log)
                    else None
                )
                push(parent, depth, rank + 1, sibling_path_log)
        else:
            stop_reason = "exhausted_candidates"
    elif maximum == 1:
        stop_reason = "node_limit"

    constructed_size = len(nodes) + 1
    if return_size < constructed_size:
        nodes = nodes[: return_size - 1]
    returned_size = len(nodes) + 1
    return nodes, {
        "request_index": request_index,
        "candidate_capacity": quota,
        "effective_min_tree_nodes": effective_minimum,
        "effective_max_tree_nodes": maximum,
        "constructed_tree_nodes": constructed_size,
        "returned_tree_nodes": returned_size,
        "bucket_truncated": returned_size < constructed_size,
        "acceptance_surrogate": 1.0 + math.fsum(n.path_probability for n in nodes),
        "estimated_cycle_seconds": costs.get(returned_size),
        "stop_reason": stop_reason,
    }


def select_bastion_batch(
    candidate_ids,
    marginal_probabilities,
    *,
    capacity: int,
    context_lengths,
    cost_model: CycleCostModel,
    min_nodes_per_request: int = 32,
    max_nodes_per_request: int = 8192,
    bucket_step: int = 32,
) -> PolicySelection:

    _integer(capacity, "capacity")
    _integer(min_nodes_per_request, "min_nodes_per_request", 1)
    _integer(max_nodes_per_request, "max_nodes_per_request", 1)
    _integer(bucket_step, "bucket_step", 1)
    if min_nodes_per_request > max_nodes_per_request:
        raise ValueError("min_nodes_per_request must not exceed max_nodes_per_request")
    if not callable(cost_model):
        raise ValueError("cost_model must be callable")
    rows = _validated_candidates(candidate_ids, marginal_probabilities)
    contexts = _sequence(context_lengths, "context_lengths")
    if len(contexts) != len(rows):
        raise ValueError("context_lengths must match the candidate batch size")
    for length in contexts:
        _integer(length, "context length")

    nodes = []
    request_stats = []
    base, remainder = divmod(capacity, len(rows)) if rows else (0, 0)
    for index, (request_rows, context_length) in enumerate(zip(rows, contexts)):
        selected, stats = _select_request(
            request_rows,
            request_index=index,
            quota=base + (index < remainder),
            context_length=context_length,
            cost_model=cost_model,
            minimum=min_nodes_per_request,
            maximum=max_nodes_per_request,
            bucket_step=bucket_step,
        )
        offset = len(nodes)
        nodes.extend(
            PolicyNode(
                request_index=n.request_index,
                parent_index=n.parent_index + offset if n.parent_index >= 0 else -1,
                token_id=n.token_id,
                depth=n.depth,
                path_probability=n.path_probability,
                conditional_probability=n.conditional_probability,
                candidate_rank=n.candidate_rank,
            )
            for n in selected
        )
        request_stats.append(stats)
    return PolicySelection(
        nodes=tuple(nodes),
        stats={
            "policy": "bastion_style_independent_per_request",
            "budget_scope": "equal_per_request_no_redistribution",
            "capacity_excludes_roots": capacity,
            "selected_nodes": len(nodes),
            "unused_capacity": capacity - len(nodes),
            "root_nodes": len(rows),
            "bucket_step": bucket_step,
            "requests": request_stats,
        },
    )
