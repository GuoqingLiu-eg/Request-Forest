from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Optional

import torch

from .adapter import DrafterCapabilities
from .acceptance_calibrator import calibrate_probability_row
from .native import TensorForestArena


DFLASH2_SELECTION_MODES = (
    "global-forest",
    "global-chain",
    "equal-forest",
    "equal-chain",
)
_DFLASH2_SELECTION_MODE_IDS = {
    mode: index for index, mode in enumerate(DFLASH2_SELECTION_MODES)
}


@dataclass(frozen=True, slots=True)
class DFlashLoweringResult:


    node_ids: range
    conditional_probabilities: torch.Tensor
    max_probabilities: torch.Tensor
    margins: torch.Tensor
    entropies: torch.Tensor
    truncated_root_request_slots: tuple[int, ...] = ()
    truncated_parent_node_ids: tuple[int, ...] = ()

    @property
    def node_count(self) -> int:
        return len(self.node_ids)


def _require_plain_int(name: str, value: int, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


def _request_slot_list(
    *,
    batch_size: int,
    request_slots: Optional[torch.Tensor],
    arena: TensorForestArena,
) -> list[int]:
    if request_slots is None:
        if batch_size != arena.request_count:
            raise ValueError(
                "identity request slots require one proposal row per Arena request"
            )
        return list(range(batch_size))
    if not isinstance(request_slots, torch.Tensor):
        raise TypeError("request_slots must be a torch.Tensor or None")
    if request_slots.ndim != 1 or request_slots.numel() != batch_size:
        raise ValueError("request_slots must have shape [batch_size]")
    if request_slots.dtype == torch.bool or torch.is_floating_point(request_slots):
        raise TypeError("request_slots must have an integer dtype")
    slots = request_slots.detach().to(device="cpu", dtype=torch.int64).tolist()
    for slot in slots:
        if slot < 0 or slot >= arena.request_count:
            raise ValueError(
                f"request slot {slot} is outside [0, {arena.request_count})"
            )
    return slots


def _check_capacity(
    *, arena: TensorForestArena, capacity: int, configured_max_nodes: int
) -> None:
    _require_plain_int("capacity", capacity, minimum=0)
    if capacity > configured_max_nodes:
        raise ValueError(
            f"capacity {capacity} exceeds configured max_nodes "
            f"{configured_max_nodes}"
        )
    remaining = arena.capacity - arena.node_count
    if capacity > remaining:
        raise ValueError(
            f"capacity {capacity} exceeds remaining Arena capacity {remaining}"
        )


def _empty_feature_tensor() -> torch.Tensor:
    return torch.empty((0,), dtype=torch.float64, device="cpu")


def _truncated_frontier(
    heap: list[tuple],
    *,
    request_slot_index: int = 1,
    parent_id_index: int = 3,
) -> tuple[tuple[int, ...], tuple[int, ...]]:


    root_request_slots = tuple(
        sorted(
            {
                int(entry[request_slot_index])
                for entry in heap
                if int(entry[parent_id_index]) == -1
            }
        )
    )
    parent_node_ids = tuple(
        sorted(
            {
                int(entry[parent_id_index])
                for entry in heap
                if int(entry[parent_id_index]) >= 0
            }
        )
    )
    return root_request_slots, parent_node_ids


def _try_native_dflash2_lower(
    *,
    candidate_ids: torch.Tensor,
    probabilities: torch.Tensor,
    request_slots: torch.Tensor,
    capacity: int,
    start_node_id: int,
    calibration_multipliers: torch.Tensor | None = None,
    max_nodes_per_request: int | None = None,
    selection_mode: int = 0,
    d2_mix_lambda: float = 0.0,
) -> Optional[tuple[torch.Tensor, ...]]:


    try:
        operation = torch.ops.sgl_kernel.request_forest_dflash2_lower_cpu
    except AttributeError:
        return None
    outputs = tuple(
        operation(
            candidate_ids,
            probabilities,
            request_slots,
            capacity,
            start_node_id,
            calibration_multipliers,
            -1 if max_nodes_per_request is None else max_nodes_per_request,
            selection_mode,
            d2_mix_lambda,
        )
    )
    if len(outputs) not in (7, 8):
        raise RuntimeError(
            "request_forest_dflash2_lower_cpu must return 7 or 8 tensors"
        )
    if len(outputs) == 7:
        outputs = (*outputs, _empty_feature_tensor())
    return outputs


def _append_lowered_nodes(
    *,
    arena: TensorForestArena,
    parent_ids: list[int],
    request_slots: list[int],
    token_ids: list[int],
    depths: list[int],
    path_probabilities: list[float],
    conditional_probabilities: list[float],
    max_probabilities: list[float],
    margins: list[float],
    entropies: list[float],
    truncated_root_request_slots: tuple[int, ...] = (),
    truncated_parent_node_ids: tuple[int, ...] = (),
) -> DFlashLoweringResult:
    if not parent_ids:
        empty = torch.empty((0,), dtype=torch.int64, device="cpu")
        empty_values = _empty_feature_tensor()
        appended = arena.append_tensors(
            empty, empty, empty, empty, empty_values
        )
        return DFlashLoweringResult(
            node_ids=appended,
            conditional_probabilities=empty_values,
            max_probabilities=empty_values,
            margins=empty_values,
            entropies=empty_values,
            truncated_root_request_slots=truncated_root_request_slots,
            truncated_parent_node_ids=truncated_parent_node_ids,
        )

    appended = arena.append_tensors(
        torch.tensor(parent_ids, dtype=torch.int64, device="cpu"),
        torch.tensor(request_slots, dtype=torch.int64, device="cpu"),
        torch.tensor(token_ids, dtype=torch.int64, device="cpu"),
        torch.tensor(depths, dtype=torch.int64, device="cpu"),
        torch.tensor(path_probabilities, dtype=torch.float64, device="cpu"),
    )
    return DFlashLoweringResult(
        node_ids=appended,
        conditional_probabilities=torch.tensor(
            conditional_probabilities, dtype=torch.float64, device="cpu"
        ),
        max_probabilities=torch.tensor(
            max_probabilities, dtype=torch.float64, device="cpu"
        ),
        margins=torch.tensor(margins, dtype=torch.float64, device="cpu"),
        entropies=torch.tensor(entropies, dtype=torch.float64, device="cpu"),
        truncated_root_request_slots=truncated_root_request_slots,
        truncated_parent_node_ids=truncated_parent_node_ids,
    )


class DFlashChainLowerer:


    def __init__(self, *, max_nodes: int) -> None:
        _require_plain_int("max_nodes", max_nodes, minimum=1)
        self._max_nodes = max_nodes
        self._capabilities = DrafterCapabilities(
            supports_arbitrary_parent_state=False,
            max_fragment_nodes=max_nodes,
            max_fanout=1,
            provides_draft_features=False,
        )
        self._uniform_gather_indices: dict[
            tuple[torch.device, int, int], torch.Tensor
        ] = {}

    @property
    def capabilities(self) -> DrafterCapabilities:
        return self._capabilities

    def _lower_uniform_identity_chain(
        self,
        *,
        arena: TensorForestArena,
        candidate_token_ids: torch.Tensor,
        capacity: int,
    ) -> DFlashLoweringResult:


        batch_size, depth_count = candidate_token_ids.shape
        total_nodes = batch_size * depth_count
        selected_count = min(capacity, total_nodes)
        if batch_size == 0 or selected_count == 0:
            empty = torch.empty((0,), dtype=torch.int64, device="cpu")
            empty_values = _empty_feature_tensor()
            appended = arena.append_tensors(
                empty, empty, empty, empty, empty_values
            )
            return DFlashLoweringResult(
                node_ids=appended,
                conditional_probabilities=empty_values,
                max_probabilities=empty_values,
                margins=empty_values,
                entropies=empty_values,
                truncated_root_request_slots=(
                    tuple(range(batch_size)) if total_nodes else ()
                ),
            )

        key = (candidate_token_ids.device, batch_size, depth_count)
        gather_indices = self._uniform_gather_indices.get(key)
        if gather_indices is None:
            gather_indices = (
                torch.arange(
                    total_nodes,
                    dtype=torch.int64,
                    device=candidate_token_ids.device,
                )
                .view(batch_size, depth_count)
                .transpose(0, 1)
                .reshape(-1)
            )
            self._uniform_gather_indices[key] = gather_indices
        token_ids = (
            candidate_token_ids.reshape(-1)
            .index_select(0, gather_indices[:selected_count])
            .to(device="cpu", dtype=torch.int64)
        )
        if bool(torch.any(token_ids < 0)):
            raise ValueError("candidate token IDs must be non-negative")

        start = arena.node_count
        ranks = torch.arange(selected_count, dtype=torch.int64, device="cpu")
        request_slots = torch.remainder(ranks, batch_size)
        zero_based_depths = torch.div(
            ranks, batch_size, rounding_mode="floor"
        )
        depths = zero_based_depths + 1
        parent_ids = ranks + start - batch_size
        parent_ids.masked_fill_(zero_based_depths == 0, -1)
        path_probabilities = torch.ones(
            (selected_count,), dtype=torch.float64, device="cpu"
        )
        appended = arena.append_tensors(
            parent_ids,
            request_slots,
            token_ids,
            depths,
            path_probabilities,
        )

        truncated_roots: list[int] = []
        truncated_parents: list[int] = []
        full_depths, partial_requests = divmod(selected_count, batch_size)
        for request_slot in range(batch_size):
            selected_depths = full_depths + int(request_slot < partial_requests)
            if selected_depths == 0:
                truncated_roots.append(request_slot)
            elif selected_depths < depth_count:
                truncated_parents.append(
                    start + (selected_depths - 1) * batch_size + request_slot
                )

        zeros = torch.zeros_like(path_probabilities)
        return DFlashLoweringResult(
            node_ids=appended,
            conditional_probabilities=path_probabilities,
            max_probabilities=path_probabilities,
            margins=zeros,
            entropies=zeros,
            truncated_root_request_slots=tuple(truncated_roots),
            truncated_parent_node_ids=tuple(sorted(truncated_parents)),
        )

    def lower(
        self,
        *,
        arena: TensorForestArena,
        candidate_token_ids: torch.Tensor,
        capacity: int,
        request_slots: Optional[torch.Tensor] = None,
        conditional_probabilities: Optional[torch.Tensor] = None,
    ) -> DFlashLoweringResult:


        _check_capacity(
            arena=arena,
            capacity=capacity,
            configured_max_nodes=self._max_nodes,
        )
        if not isinstance(candidate_token_ids, torch.Tensor):
            raise TypeError("candidate_token_ids must be a torch.Tensor")
        if candidate_token_ids.ndim != 2:
            raise ValueError("candidate_token_ids must have shape [B, E]")
        if candidate_token_ids.dtype == torch.bool or torch.is_floating_point(
            candidate_token_ids
        ):
            raise TypeError("candidate_token_ids must have an integer dtype")
        batch_size, depth_count = candidate_token_ids.shape
        slots = _request_slot_list(
            batch_size=batch_size, request_slots=request_slots, arena=arena
        )
        if depth_count == 0:
            return _append_lowered_nodes(
                arena=arena,
                parent_ids=[],
                request_slots=[],
                token_ids=[],
                depths=[],
                path_probabilities=[],
                conditional_probabilities=[],
                max_probabilities=[],
                margins=[],
                entropies=[],
            )
        if conditional_probabilities is None and request_slots is None:
            return self._lower_uniform_identity_chain(
                arena=arena,
                candidate_token_ids=candidate_token_ids,
                capacity=capacity,
            )
        tokens = candidate_token_ids.detach().to(
            device="cpu", dtype=torch.int64
        ).tolist()

        if conditional_probabilities is None:
            probabilities = [[1.0] * depth_count for _ in range(batch_size)]
        else:
            if not isinstance(conditional_probabilities, torch.Tensor):
                raise TypeError("conditional_probabilities must be a torch.Tensor")
            if tuple(conditional_probabilities.shape) != (batch_size, depth_count):
                raise ValueError(
                    "conditional_probabilities must match candidate_token_ids"
                )
            probabilities = (
                conditional_probabilities.detach()
                .to(device="cpu", dtype=torch.float64)
                .tolist()
            )

        start = arena.node_count


        heap: list[tuple[float, int, int, int, int, int, int, float]] = []

        def push(
            *, batch_row: int, depth: int, parent_id: int, path_probability: float
        ) -> None:
            token_id = int(tokens[batch_row][depth - 1])
            conditional = float(probabilities[batch_row][depth - 1])
            if token_id < 0:
                raise ValueError("candidate token IDs must be non-negative")
            if not math.isfinite(conditional) or not 0.0 <= conditional <= 1.0:
                raise ValueError(
                    "conditional probabilities must be finite values in [0, 1]"
                )
            next_path_probability = path_probability * conditional
            if next_path_probability <= 0.0:
                return
            heapq.heappush(
                heap,
                (
                    -next_path_probability,
                    depth,
                    slots[batch_row],
                    parent_id,
                    0,
                    batch_row,
                    token_id,
                    conditional,
                ),
            )

        if depth_count > 0:
            for batch_row in range(batch_size):
                push(
                    batch_row=batch_row,
                    depth=1,
                    parent_id=-1,
                    path_probability=1.0,
                )

        parent_ids: list[int] = []
        output_slots: list[int] = []
        output_tokens: list[int] = []
        depths: list[int] = []
        path_probabilities: list[float] = []
        output_conditionals: list[float] = []
        max_probabilities: list[float] = []
        margins: list[float] = []
        entropies: list[float] = []
        while heap and len(parent_ids) < capacity:
            (
                negative_path_probability,
                depth,
                request_slot,
                parent_id,
                _token_rank,
                batch_row,
                token_id,
                conditional,
            ) = heapq.heappop(heap)
            path_probability = -negative_path_probability
            node_id = start + len(parent_ids)
            parent_ids.append(parent_id)
            output_slots.append(request_slot)
            output_tokens.append(token_id)
            depths.append(depth)
            path_probabilities.append(path_probability)
            output_conditionals.append(conditional)
            max_probabilities.append(conditional)
            margins.append(0.0)
            entropies.append(0.0)
            if depth < depth_count:
                push(
                    batch_row=batch_row,
                    depth=depth + 1,
                    parent_id=node_id,
                    path_probability=path_probability,
                )

        truncated_roots, truncated_parents = _truncated_frontier(
            heap,
            request_slot_index=2,
        )
        return _append_lowered_nodes(
            arena=arena,
            parent_ids=parent_ids,
            request_slots=output_slots,
            token_ids=output_tokens,
            depths=depths,
            path_probabilities=path_probabilities,
            conditional_probabilities=output_conditionals,
            max_probabilities=max_probabilities,
            margins=margins,
            entropies=entropies,
            truncated_root_request_slots=truncated_roots,
            truncated_parent_node_ids=truncated_parents,
        )


class DFlash2LatticeLowerer:


    def __init__(
        self,
        *,
        max_nodes: int,
        top_k: int,
        max_nodes_per_request: int | None = None,
        selection_mode: str = "global-forest",
    ) -> None:
        _require_plain_int("max_nodes", max_nodes, minimum=1)
        _require_plain_int("top_k", top_k, minimum=1)
        if max_nodes_per_request is not None:
            _require_plain_int(
                "max_nodes_per_request",
                max_nodes_per_request,
                minimum=1,
            )
        if selection_mode not in _DFLASH2_SELECTION_MODE_IDS:
            raise ValueError(
                "selection_mode must be one of "
                f"{DFLASH2_SELECTION_MODES}, got {selection_mode!r}"
            )
        self._max_nodes = max_nodes
        self._top_k = top_k
        self._max_nodes_per_request = max_nodes_per_request
        self._selection_mode = selection_mode
        self._selection_mode_id = _DFLASH2_SELECTION_MODE_IDS[selection_mode]
        self._capabilities = DrafterCapabilities(
            supports_arbitrary_parent_state=False,
            max_fragment_nodes=max_nodes,
            max_fanout=top_k,
            provides_draft_features=True,
        )

    @property
    def capabilities(self) -> DrafterCapabilities:
        return self._capabilities

    @property
    def selection_mode(self) -> str:
        return self._selection_mode

    def lower(
        self,
        *,
        arena: TensorForestArena,
        candidate_ids: torch.Tensor,
        scores: torch.Tensor,
        capacity: int,
        request_slots: Optional[torch.Tensor] = None,
        collect_features: bool = True,
        calibration_multipliers: torch.Tensor | None = None,
        d2_mix_lambda: float = 0.0,
    ) -> DFlashLoweringResult:


        _check_capacity(
            arena=arena,
            capacity=capacity,
            configured_max_nodes=self._max_nodes,
        )
        if not isinstance(candidate_ids, torch.Tensor) or not isinstance(
            scores, torch.Tensor
        ):
            raise TypeError("candidate_ids and scores must be torch.Tensor values")
        if candidate_ids.ndim != 3:
            raise ValueError("candidate_ids must have shape [B, E, K]")
        if scores.ndim != 4:
            raise ValueError("scores must have shape [B, E, P, K]")
        if candidate_ids.dtype == torch.bool or torch.is_floating_point(candidate_ids):
            raise TypeError("candidate_ids must have an integer dtype")
        if not torch.is_floating_point(scores):
            raise TypeError("scores must have a floating-point dtype")
        d2_mix_lambda = float(d2_mix_lambda)
        if not math.isfinite(d2_mix_lambda) or not 0.0 <= d2_mix_lambda <= 1.0:
            raise ValueError("d2_mix_lambda must be finite and in [0, 1]")

        batch_size, depth_count, top_k = candidate_ids.shape
        score_batch, score_depth, predecessor_count, score_top_k = scores.shape
        if (score_batch, score_depth, score_top_k) != (
            batch_size,
            depth_count,
            top_k,
        ):
            raise ValueError("candidate_ids and scores disagree on B, E, or K")
        if top_k != self._top_k:
            raise ValueError(
                f"proposal top_k={top_k} does not match configured top_k={self._top_k}"
            )
        if depth_count > 1 and predecessor_count != top_k:
            raise ValueError("DFlash2 transition scores require P == K")
        if depth_count > 0 and predecessor_count < 1:
            raise ValueError("DFlash2 root scores require at least one predecessor row")

        slots = _request_slot_list(
            batch_size=batch_size, request_slots=request_slots, arena=arena
        )
        if depth_count == 0:
            return _append_lowered_nodes(
                arena=arena,
                parent_ids=[],
                request_slots=[],
                token_ids=[],
                depths=[],
                path_probabilities=[],
                conditional_probabilities=[],
                max_probabilities=[],
                margins=[],
                entropies=[],
            )
        candidate_ids_cpu = (
            candidate_ids.detach()
            .to(device="cpu", dtype=torch.int64)
            .contiguous()
        )
        score_cpu = (
            scores.detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
        )
        probabilities_tensor = torch.softmax(score_cpu, dim=-1).contiguous()
        if not collect_features and predecessor_count == top_k:
            native_outputs = _try_native_dflash2_lower(
                candidate_ids=candidate_ids_cpu,
                probabilities=probabilities_tensor,
                request_slots=torch.tensor(
                    slots, dtype=torch.int64, device="cpu"
                ),
                capacity=capacity,
                start_node_id=arena.node_count,
                calibration_multipliers=calibration_multipliers,
                max_nodes_per_request=self._max_nodes_per_request,
                selection_mode=self._selection_mode_id,
                d2_mix_lambda=d2_mix_lambda,
            )
            if native_outputs is not None:
                (
                    parent_ids_tensor,
                    request_slots_tensor,
                    token_ids_tensor,
                    depths_tensor,
                    path_values_tensor,
                    truncated_roots_tensor,
                    truncated_parents_tensor,
                    raw_conditionals_tensor,
                ) = native_outputs
                appended = arena.append_tensors(
                    parent_ids_tensor,
                    request_slots_tensor,
                    token_ids_tensor,
                    depths_tensor,
                    path_values_tensor,
                )
                empty_values = _empty_feature_tensor()
                return DFlashLoweringResult(
                    node_ids=appended,
                    conditional_probabilities=raw_conditionals_tensor,
                    max_probabilities=empty_values,
                    margins=empty_values,
                    entropies=empty_values,
                    truncated_root_request_slots=tuple(
                        truncated_roots_tensor.tolist()
                    ),
                    truncated_parent_node_ids=tuple(
                        truncated_parents_tensor.tolist()
                    ),
                )

        tokens = candidate_ids_cpu.tolist()
        probabilities = probabilities_tensor.tolist()
        row_maxima = row_margins = row_entropies = None
        if collect_features:
            max_tensor = probabilities_tensor.max(dim=-1).values
            if top_k == 1:
                margin_tensor = max_tensor
            else:
                top_two = torch.topk(probabilities_tensor, k=2, dim=-1).values
                margin_tensor = top_two[..., 0] - top_two[..., 1]
            entropy_tensor = -torch.where(
                probabilities_tensor > 0,
                probabilities_tensor * probabilities_tensor.log(),
                torch.zeros_like(probabilities_tensor),
            ).sum(dim=-1)
            row_maxima = max_tensor.tolist()
            row_margins = margin_tensor.tolist()
            row_entropies = entropy_tensor.tolist()

        start = arena.node_count
        chain_mode = self._selection_mode in ("global-chain", "equal-chain")
        equal_mode = self._selection_mode in ("equal-forest", "equal-chain")
        mix_enabled = d2_mix_lambda > 0.0 and not chain_mode
        depth_discounts = (
            tuple(
                1.0 if depth == 0 else depth ** (-d2_mix_lambda)
                for depth in range(depth_count + 1)
            )
            if mix_enabled
            else ()
        )
        if equal_mode and batch_size:
            equal_quota, extra_quota = divmod(capacity, batch_size)
            request_limits: list[int | None] = [
                equal_quota + int(batch_row < extra_quota)
                for batch_row in range(batch_size)
            ]
            if self._max_nodes_per_request is not None:
                request_limits = [
                    min(limit, self._max_nodes_per_request)
                    for limit in request_limits
                ]
        else:
            request_limits = [self._max_nodes_per_request] * batch_size


        heap: list[
            tuple[
                float,
                int,
                int,
                int,
                int,
                int,
                int,
                float,
                float,
                float,
                float,
                float,
                float,
            ]
        ] = []

        def push_children(
            *,
            batch_row: int,
            depth: int,
            predecessor_rank: int,
            parent_id: int,
            parent_rf_path_probability: float,
            parent_mix_path_probability: float,
        ) -> None:
            probability_row = probabilities[batch_row][depth - 1][predecessor_rank]
            if chain_mode:
                token_row = tokens[batch_row][depth - 1]
                if any(int(token_id) < 0 for token_id in token_row):
                    raise ValueError("candidate token IDs must be non-negative")
                if any(
                    not math.isfinite(float(probability))
                    or float(probability) < 0.0
                    for probability in probability_row
                ):
                    raise ValueError(
                        "DFlash2 scores produced an invalid probability"
                    )
                calibrated_row = probability_row
                token_ranks = (
                    max(range(top_k), key=probability_row.__getitem__),
                )
            else:
                calibrated_row, _ = calibrate_probability_row(
                    probability_row,
                    depth=depth,
                    multipliers=calibration_multipliers,
                )
                token_ranks = range(top_k)
            argmax_rank = (
                max(range(top_k), key=probability_row.__getitem__)
                if mix_enabled
                else -1
            )
            if collect_features:
                assert row_maxima is not None
                assert row_margins is not None
                assert row_entropies is not None
                maximum = float(
                    row_maxima[batch_row][depth - 1][predecessor_rank]
                )
                margin = float(
                    row_margins[batch_row][depth - 1][predecessor_rank]
                )
                entropy = float(
                    row_entropies[batch_row][depth - 1][predecessor_rank]
                )
                if not all(
                    math.isfinite(value) for value in (maximum, margin, entropy)
                ):
                    raise ValueError(
                        "DFlash2 scores produced non-finite probabilities"
                    )
            else:


                maximum = margin = entropy = 0.0
            for token_rank in token_ranks:
                token_id = int(tokens[batch_row][depth - 1][token_rank])
                conditional = float(probability_row[token_rank])
                calibrated = float(calibrated_row[token_rank])
                if token_id < 0:
                    raise ValueError("candidate token IDs must be non-negative")
                if not math.isfinite(conditional) or conditional < 0.0:
                    raise ValueError("DFlash2 scores produced an invalid probability")
                rf_path_probability = parent_rf_path_probability * calibrated
                if mix_enabled:
                    mixed_conditional = (
                        (1.0 - d2_mix_lambda) * calibrated
                        + (d2_mix_lambda if token_rank == argmax_rank else 0.0)
                    )
                    mix_path_probability = (
                        parent_mix_path_probability * mixed_conditional
                    )
                    priority = mix_path_probability * depth_discounts[depth]
                else:
                    mix_path_probability = rf_path_probability
                    priority = rf_path_probability
                if priority <= 0.0:
                    continue
                heapq.heappush(
                    heap,
                    (
                        -priority,
                        slots[batch_row],
                        depth,
                        parent_id,
                        token_rank,
                        batch_row,
                        token_id,
                        conditional,
                        maximum,
                        margin,
                        entropy,
                        rf_path_probability,
                        mix_path_probability,
                    ),
                )

        if depth_count > 0:
            for batch_row in range(batch_size):
                push_children(
                    batch_row=batch_row,
                    depth=1,
                    predecessor_rank=0,
                    parent_id=-1,
                    parent_rf_path_probability=1.0,
                    parent_mix_path_probability=1.0,
                )

        parent_ids: list[int] = []
        output_slots: list[int] = []
        output_tokens: list[int] = []
        depths: list[int] = []
        path_probabilities: list[float] = []
        output_conditionals: list[float] = []
        max_probabilities: list[float] = []
        margins: list[float] = []
        entropies: list[float] = []
        selected_per_request = [0] * batch_size
        capped_root_requests: set[int] = set()
        capped_parent_nodes: set[int] = set()
        while heap and len(parent_ids) < capacity:
            (
                _negative_priority,
                request_slot,
                depth,
                parent_id,
                token_rank,
                batch_row,
                token_id,
                conditional,
                maximum,
                margin,
                entropy,
                rf_path_probability,
                mix_path_probability,
            ) = heapq.heappop(heap)
            request_limit = request_limits[batch_row]
            if (
                request_limit is not None
                and selected_per_request[batch_row] >= request_limit
            ):
                if parent_id < 0:
                    capped_root_requests.add(request_slot)
                else:
                    capped_parent_nodes.add(parent_id)
                continue
            node_id = start + len(parent_ids)
            parent_ids.append(parent_id)
            output_slots.append(request_slot)
            output_tokens.append(token_id)
            depths.append(depth)
            path_probabilities.append(rf_path_probability)
            selected_per_request[batch_row] += 1
            if collect_features or calibration_multipliers is not None:
                output_conditionals.append(conditional)
            if collect_features:
                max_probabilities.append(maximum)
                margins.append(margin)
                entropies.append(entropy)
            if (
                depth < depth_count
                and (
                    request_limit is None
                    or selected_per_request[batch_row] < request_limit
                )
            ):
                push_children(
                    batch_row=batch_row,
                    depth=depth + 1,
                    predecessor_rank=token_rank,
                    parent_id=node_id,
                    parent_rf_path_probability=rf_path_probability,
                    parent_mix_path_probability=mix_path_probability,
                )
            elif depth < depth_count:
                capped_parent_nodes.add(node_id)

        truncated_roots, truncated_parents = _truncated_frontier(heap)
        truncated_roots = tuple(
            sorted(set(truncated_roots).union(capped_root_requests))
        )
        truncated_parents = tuple(
            sorted(set(truncated_parents).union(capped_parent_nodes))
        )
        return _append_lowered_nodes(
            arena=arena,
            parent_ids=parent_ids,
            request_slots=output_slots,
            token_ids=output_tokens,
            depths=depths,
            path_probabilities=path_probabilities,
            conditional_probabilities=output_conditionals,
            max_probabilities=max_probabilities,
            margins=margins,
            entropies=entropies,
            truncated_root_request_slots=truncated_roots,
            truncated_parent_node_ids=truncated_parents,
        )


__all__ = [
    "DFLASH2_SELECTION_MODES",
    "DFlash2LatticeLowerer",
    "DFlashChainLowerer",
    "DFlashLoweringResult",
]
