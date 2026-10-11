from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class AcceptanceFeedbackMetadata:


    token_ids: tuple[int, ...]
    first_child: tuple[int, ...]
    next_sibling: tuple[int, ...]
    depths: tuple[int, ...]
    confidence_bins: tuple[int, ...]
    anchor_rows: tuple[int, ...]
    logical_rows: int


def confidence_bin(probability: float, bin_count: int) -> int:


    if bin_count < 1:
        raise ValueError("bin_count must be positive")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    return min(int(probability * bin_count), bin_count - 1)


def calibrate_probability_row(
    probabilities: list[float],
    *,
    depth: int,
    multipliers: torch.Tensor | None,
) -> tuple[list[float], list[int]]:


    if multipliers is None:
        return probabilities, []
    if (
        multipliers.device.type != "cpu"
        or multipliers.dtype != torch.float64
        or multipliers.dim() != 2
        or not multipliers.is_contiguous()
        or multipliers.shape[0] < depth
        or multipliers.shape[1] < 1
    ):
        raise ValueError(
            "multipliers must be contiguous CPU float64 [depth, confidence_bin]"
        )

    bin_count = int(multipliers.shape[1])
    depth_row = multipliers[depth - 1]
    calibrated: list[float] = []
    bins: list[int] = []
    changed = False
    for probability in probabilities:
        bucket = confidence_bin(probability, bin_count)
        multiplier = float(depth_row[bucket])
        if not math.isfinite(multiplier) or multiplier < 0.0:
            raise ValueError("calibration multipliers must be finite and non-negative")
        bins.append(bucket)
        changed = changed or multiplier != 1.0
        calibrated.append(
            0.0 if probability == 0.0 else min(1.0, probability * multiplier)
        )

    total = sum(calibrated)
    if changed and total > 1.0:
        inverse = 1.0 / total
        calibrated = [value * inverse for value in calibrated]
    return calibrated, bins


class DepthConfidenceBetaEMA:


    __slots__ = (
        "_bin_count",
        "_correction_strength",
        "_decay",
        "_denominator",
        "_depth_count",
        "_multipliers",
        "_inverse_centers",
        "_negative",
        "_positive",
        "_prior_strength",
        "_prior_positive",
    )

    def __init__(
        self,
        *,
        depth_count: int,
        bin_count: int = 16,
        prior_strength: float = 16.0,
        decay: float = 0.98,
        correction_strength: float = 0.25,
    ) -> None:
        if depth_count < 1 or bin_count < 1:
            raise ValueError("depth_count and bin_count must be positive")
        if prior_strength <= 0.0:
            raise ValueError("prior_strength must be positive")
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        if not 0.0 <= correction_strength <= 1.0:
            raise ValueError("correction_strength must be in [0, 1]")
        self._depth_count = depth_count
        self._bin_count = bin_count
        self._prior_strength = float(prior_strength)
        self._decay = float(decay)
        self._correction_strength = float(correction_strength)
        shape = (depth_count, bin_count)
        self._positive = torch.zeros(shape, dtype=torch.float64, device="cpu")
        self._negative = torch.zeros_like(self._positive)
        self._multipliers = torch.ones_like(self._positive)
        centers = (
            torch.arange(bin_count, dtype=torch.float64).add_(0.5) / bin_count
        )
        self._prior_positive = centers.unsqueeze(0) * self._prior_strength
        self._inverse_centers = centers.reciprocal().unsqueeze(0)
        self._denominator = torch.empty_like(self._positive)

    @property
    def multipliers(self) -> torch.Tensor:


        return self._multipliers

    @property
    def bin_count(self) -> int:


        return self._bin_count

    @property
    def sample_mass(self) -> float:
        return float((self._positive + self._negative).sum())

    @property
    def correction_strength(self) -> float:


        return self._correction_strength

    def set_correction_strength(self, value: float) -> None:


        if not 0.0 <= value <= 1.0:
            raise ValueError("correction_strength must be in [0, 1]")
        value = float(value)
        if value == self._correction_strength:
            return
        self._correction_strength = value
        self._refresh_multipliers()

    def _refresh_multipliers(self) -> None:


        self._denominator.copy_(self._positive).add_(self._negative).add_(
            self._prior_strength
        )
        self._multipliers.copy_(self._positive).add_(
            self._prior_positive
        ).div_(self._denominator).mul_(self._inverse_centers)
        self._multipliers.sub_(1.0).mul_(self._correction_strength).add_(1.0)

    def ingest(
        self,
        metadata: AcceptanceFeedbackMetadata,
        target_predictions: torch.Tensor,
        *,
        eos_token_ids: frozenset[int] = frozenset(),
        request_ignore_eos: tuple[bool, ...] = (),
    ) -> int:


        predictions = target_predictions[: metadata.logical_rows].tolist()
        cell_count = self._depth_count * self._bin_count
        flat_samples: list[int] = []
        ignore_eos = request_ignore_eos or (False,) * len(metadata.anchor_rows)

        for request_slot, anchor_row in enumerate(metadata.anchor_rows):
            parent_row = anchor_row
            while True:
                child_row = metadata.first_child[parent_row]
                if child_row < 0:
                    break
                target_token = int(predictions[parent_row])
                matching_row = -1
                while child_row >= 0:
                    bucket = metadata.confidence_bins[child_row]
                    if bucket >= 0:
                        depth = metadata.depths[child_row]
                        flat_bin = (
                            (depth - 1) * self._bin_count + bucket
                        )
                        if (
                            matching_row < 0
                            and metadata.token_ids[child_row] == target_token
                        ):
                            flat_samples.append(flat_bin)
                            matching_row = child_row
                        else:
                            flat_samples.append(cell_count + flat_bin)
                    child_row = metadata.next_sibling[child_row]
                if matching_row < 0:
                    break
                if target_token in eos_token_ids and not ignore_eos[request_slot]:
                    break
                parent_row = matching_row

        sample_count = len(flat_samples)
        if sample_count == 0:
            return 0
        counts = torch.bincount(
            torch.tensor(flat_samples, dtype=torch.int64),
            minlength=2 * cell_count,
        )
        positive, negative = counts.view(2, self._depth_count, self._bin_count)
        self._positive.mul_(self._decay).add_(positive)
        self._negative.mul_(self._decay).add_(negative)
        self._refresh_multipliers()
        return sample_count
