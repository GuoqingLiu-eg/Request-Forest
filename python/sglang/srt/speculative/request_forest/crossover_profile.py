from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CrossoverPoint:
    batch_size: int
    lambda_mix: float
    verify_nodes: int


@dataclass(frozen=True, slots=True)
class CrossoverDecision:
    lambda_mix: float
    verify_nodes: int


@dataclass(frozen=True, slots=True)
class CrossoverProfile:


    points: tuple[CrossoverPoint, ...]
    crossing_batch: float | None = None
    schema_version: int = 1

    @classmethod
    def load(cls, path: str | Path) -> "CrossoverProfile":
        payload = json.loads(Path(path).read_text())
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("unsupported crossover profile schema")
        raw_points = payload.get("points")
        if not isinstance(raw_points, list) or not raw_points:
            raise ValueError("crossover profile has no points")
        points: list[CrossoverPoint] = []
        previous_batch = 0
        for raw in raw_points:
            if not isinstance(raw, dict):
                raise ValueError("crossover profile point must be an object")
            batch = int(raw["batch_size"])
            lam = float(raw["lambda"])
            nodes = int(raw["verify_nodes"])
            if batch <= previous_batch or not 0.0 <= lam <= 1.0 or nodes < 0:
                raise ValueError("invalid crossover profile point")
            points.append(CrossoverPoint(batch, lam, nodes))
            previous_batch = batch
        fit = payload.get("fit", {})
        crossing = fit.get("provisional_crossing_batch")
        if crossing is not None:
            crossing = float(crossing)
            if not math.isfinite(crossing) or crossing < 0.0:
                raise ValueError("invalid crossover profile crossing")
        return cls(tuple(points), crossing)

    def resolve(
        self,
        batch_size: int,
        *,
        generated_nodes: int,
        configured_max_nodes: int | None = None,
    ) -> CrossoverDecision:


        if batch_size < 1 or generated_nodes < 0:
            raise ValueError("batch_size and generated_nodes must be positive")
        if len(self.points) == 1:
            left = right = self.points[0]
        elif batch_size <= self.points[0].batch_size:
            left = right = self.points[0]
        elif batch_size >= self.points[-1].batch_size:
            left = right = self.points[-1]
        else:
            left = right = self.points[-1]
            for candidate_left, candidate_right in zip(self.points, self.points[1:]):
                if candidate_left.batch_size <= batch_size <= candidate_right.batch_size:
                    left, right = candidate_left, candidate_right
                    break
        if left is right and batch_size > left.batch_size and left.lambda_mix == 0.0:


            lam = 0.0
            nodes = left.verify_nodes * batch_size / left.batch_size
        elif left is right:
            lam = left.lambda_mix
            nodes = float(left.verify_nodes)
        else:
            span = right.batch_size - left.batch_size
            ratio = (batch_size - left.batch_size) / span
            lam = left.lambda_mix + ratio * (right.lambda_mix - left.lambda_mix)
            nodes = left.verify_nodes + ratio * (right.verify_nodes - left.verify_nodes)
        capacity = min(generated_nodes, max(0, int(round(nodes))))
        if configured_max_nodes is not None:
            capacity = min(capacity, configured_max_nodes)
        return CrossoverDecision(float(lam), capacity)


def load_crossover_profile(path: str | Path | None) -> CrossoverProfile | None:
    if path is None:
        return None
    return CrossoverProfile.load(path)
