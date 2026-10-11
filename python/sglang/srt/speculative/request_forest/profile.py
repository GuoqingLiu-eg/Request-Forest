from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence


PROFILE_SCHEMA_VERSION = 1


def _installed_sglang_revision() -> str:
    try:
        from sglang.version import __version__

        return str(__version__)
    except (ImportError, AttributeError):
        return "unknown"


def _installed_kernel_revision() -> str:
    try:
        from sgl_kernel import __version__

        return str(__version__)
    except (ImportError, AttributeError):
        return "unknown"


def _require_non_empty(value: str, *, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")


def _require_non_negative(value: int, *, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")


def _require_positive(value: int, *, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def build_alpha_grid(segments: int) -> tuple[float, ...]:


    if isinstance(segments, bool) or not isinstance(segments, int) or segments < 1:
        raise ValueError("alpha grid segments must be a positive integer")
    return tuple(index / segments for index in range(segments + 1))


@dataclass(frozen=True)
class ProfileFingerprint:


    model_id: str
    model_revision: str
    dtype: str
    quantization: str
    device_name: str
    attention_backend: str
    sglang_revision: str
    kernel_revision: str
    tp_size: int
    physical_row_grid: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in (
            "model_id",
            "model_revision",
            "dtype",
            "quantization",
            "device_name",
            "attention_backend",
            "sglang_revision",
            "kernel_revision",
        ):
            _require_non_empty(getattr(self, name), name=name)
        _require_positive(self.tp_size, name="tp_size")

        grid = tuple(self.physical_row_grid)
        if not grid:
            raise ValueError("physical_row_grid must contain at least one bucket")
        if any(
            not isinstance(bucket, int) or isinstance(bucket, bool) or bucket < 1
            for bucket in grid
        ):
            raise ValueError(
                "physical_row_grid must contain positive integer buckets, "
                f"got {grid!r}"
            )
        if tuple(sorted(set(grid))) != grid:
            raise ValueError(
                "physical_row_grid must be strictly increasing, " f"got {grid!r}"
            )
        object.__setattr__(self, "physical_row_grid", grid)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "device_name": self.device_name,
            "attention_backend": self.attention_backend,
            "sglang_revision": self.sglang_revision,
            "kernel_revision": self.kernel_revision,
            "tp_size": self.tp_size,
            "physical_row_grid": list(self.physical_row_grid),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProfileFingerprint:
        return cls(
            model_id=value["model_id"],
            model_revision=value["model_revision"],
            dtype=value["dtype"],
            quantization=value["quantization"],
            device_name=value["device_name"],
            attention_backend=value["attention_backend"],
            sglang_revision=value["sglang_revision"],
            kernel_revision=value["kernel_revision"],
            tp_size=value["tp_size"],
            physical_row_grid=tuple(value["physical_row_grid"]),
        )


def build_runtime_fingerprint(
    *,
    model_id: str,
    model_revision: Optional[str],
    dtype: object,
    quantization: Optional[str],
    device_name: str,
    attention_backend: str,
    tp_size: int,
    physical_row_grid: Sequence[int],
    sglang_revision: Optional[str] = None,
    kernel_revision: Optional[str] = None,
) -> ProfileFingerprint:


    normalized_dtype = str(dtype).removeprefix("torch.")
    return ProfileFingerprint(
        model_id=str(model_id),
        model_revision=str(model_revision or "default"),
        dtype=normalized_dtype,
        quantization=str(quantization or "none"),
        device_name=str(device_name),
        attention_backend=str(attention_backend),
        sglang_revision=str(sglang_revision or _installed_sglang_revision()),
        kernel_revision=str(kernel_revision or _installed_kernel_revision()),
        tp_size=tp_size,
        physical_row_grid=tuple(physical_row_grid),
    )


@dataclass(frozen=True)
class VerifyShape:


    physical_row_bucket: int
    execution_mode: str
    num_requests: int
    total_context_tokens: int
    max_context_tokens: int
    max_tree_depth: int = 0
    ancestor_edges: int = 0
    kv_page_runs: int = 0

    def __post_init__(self) -> None:
        _require_positive(self.physical_row_bucket, name="physical_row_bucket")
        _require_non_empty(self.execution_mode, name="execution_mode")
        _require_positive(self.num_requests, name="num_requests")
        _require_non_negative(self.total_context_tokens, name="total_context_tokens")
        _require_non_negative(self.max_context_tokens, name="max_context_tokens")
        _require_non_negative(self.max_tree_depth, name="max_tree_depth")
        _require_non_negative(self.ancestor_edges, name="ancestor_edges")
        _require_non_negative(self.kv_page_runs, name="kv_page_runs")

        if self.physical_row_bucket < self.num_requests:
            raise ValueError(
                "physical_row_bucket must cover at least one anchor per request, "
                f"got bucket={self.physical_row_bucket}, requests={self.num_requests}"
            )
        if self.max_context_tokens > self.total_context_tokens:
            raise ValueError(
                "max_context_tokens cannot exceed total_context_tokens, "
                f"got max={self.max_context_tokens}, total={self.total_context_tokens}"
            )

    def covers(self, query: VerifyShape) -> bool:


        return (
            self.physical_row_bucket == query.physical_row_bucket
            and self.execution_mode == query.execution_mode
            and self.num_requests >= query.num_requests
            and self.total_context_tokens >= query.total_context_tokens
            and self.max_context_tokens >= query.max_context_tokens
            and self.max_tree_depth >= query.max_tree_depth
            and self.ancestor_edges >= query.ancestor_edges
            and self.kv_page_runs >= query.kv_page_runs
        )

    def covering_rank(self, query: VerifyShape) -> tuple[int, ...]:


        return (
            self.num_requests - query.num_requests,
            self.total_context_tokens - query.total_context_tokens,
            self.max_context_tokens - query.max_context_tokens,
            self.max_tree_depth - query.max_tree_depth,
            self.ancestor_edges - query.ancestor_edges,
            self.kv_page_runs - query.kv_page_runs,
            self.num_requests,
            self.total_context_tokens,
            self.max_context_tokens,
            self.max_tree_depth,
            self.ancestor_edges,
            self.kv_page_runs,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "physical_row_bucket": self.physical_row_bucket,
            "execution_mode": self.execution_mode,
            "num_requests": self.num_requests,
            "total_context_tokens": self.total_context_tokens,
            "max_context_tokens": self.max_context_tokens,
            "max_tree_depth": self.max_tree_depth,
            "ancestor_edges": self.ancestor_edges,
            "kv_page_runs": self.kv_page_runs,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> VerifyShape:
        return cls(
            physical_row_bucket=value["physical_row_bucket"],
            execution_mode=value["execution_mode"],
            num_requests=value["num_requests"],
            total_context_tokens=value["total_context_tokens"],
            max_context_tokens=value["max_context_tokens"],
            max_tree_depth=value.get("max_tree_depth", 0),
            ancestor_edges=value.get("ancestor_edges", 0),
            kv_page_runs=value.get("kv_page_runs", 0),
        )


def _covering_coordinates(shape: VerifyShape) -> tuple[int, ...]:


    return (
        shape.num_requests,
        shape.total_context_tokens,
        shape.max_context_tokens,
        shape.max_tree_depth,
        shape.ancestor_edges,
        shape.kv_page_runs,
    )


@dataclass(frozen=True)
class VerifyProfilePoint:
    shape: VerifyShape
    p50_gpu_ms: float
    sample_count: int = 1
    p90_gpu_ms: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.shape, VerifyShape):
            raise ValueError(f"shape must be VerifyShape, got {type(self.shape)!r}")
        if not _is_finite_number(self.p50_gpu_ms) or self.p50_gpu_ms <= 0:
            raise ValueError(
                f"p50_gpu_ms must be finite and positive, got {self.p50_gpu_ms!r}"
            )
        _require_positive(self.sample_count, name="sample_count")
        if self.p90_gpu_ms is not None:
            if not _is_finite_number(self.p90_gpu_ms) or self.p90_gpu_ms <= 0:
                raise ValueError(
                    "p90_gpu_ms must be finite and positive when present, "
                    f"got {self.p90_gpu_ms!r}"
                )
            if self.p90_gpu_ms < self.p50_gpu_ms:
                raise ValueError(
                    "p90_gpu_ms must be greater than or equal to p50_gpu_ms, "
                    f"got p50={self.p50_gpu_ms}, p90={self.p90_gpu_ms}"
                )

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "shape": self.shape.to_dict(),
            "p50_gpu_ms": self.p50_gpu_ms,
            "sample_count": self.sample_count,
        }
        if self.p90_gpu_ms is not None:
            value["p90_gpu_ms"] = self.p90_gpu_ms
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> VerifyProfilePoint:
        return cls(
            shape=VerifyShape.from_dict(value["shape"]),
            p50_gpu_ms=value["p50_gpu_ms"],
            sample_count=value.get("sample_count", 1),
            p90_gpu_ms=value.get("p90_gpu_ms"),
        )


@dataclass(frozen=True)
class ProfileEstimate:
    requested_shape: VerifyShape
    profiled_shape: VerifyShape
    p50_gpu_ms: float
    sample_count: int
    exact: bool


@dataclass(frozen=True)
class AcceptanceCorrectionCell:


    batch_size: int
    context_length: int
    alpha: float

    def __post_init__(self) -> None:
        _require_positive(self.batch_size, name="batch_size")
        _require_positive(self.context_length, name="context_length")
        if (
            not _is_finite_number(self.alpha)
            or not 0.0 <= self.alpha <= 1.0
        ):
            raise ValueError("alpha must be finite and in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "context_length": self.context_length,
            "alpha": self.alpha,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AcceptanceCorrectionCell:
        return cls(
            batch_size=value["batch_size"],
            context_length=value["context_length"],
            alpha=value["alpha"],
        )


@dataclass(frozen=True)
class AcceptanceCorrectionTable:


    cells: tuple[AcceptanceCorrectionCell, ...]
    _log_cells: tuple[tuple[float, float, AcceptanceCorrectionCell], ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        cells = tuple(self.cells)
        if not cells:
            raise ValueError("acceptance correction table must contain a cell")
        if any(not isinstance(cell, AcceptanceCorrectionCell) for cell in cells):
            raise ValueError(
                "acceptance correction table must contain only "
                "AcceptanceCorrectionCell values"
            )
        coordinates = [(cell.batch_size, cell.context_length) for cell in cells]
        if len(set(coordinates)) != len(coordinates):
            raise ValueError("acceptance correction cells must have unique scenes")
        cells = tuple(
            sorted(cells, key=lambda cell: (cell.batch_size, cell.context_length))
        )
        object.__setattr__(self, "cells", cells)
        object.__setattr__(
            self,
            "_log_cells",
            tuple(
                (math.log2(cell.batch_size), math.log2(cell.context_length), cell)
                for cell in cells
            ),
        )

    @property
    def has_positive_alpha(self) -> bool:
        return any(cell.alpha > 0.0 for cell in self.cells)

    def estimate(self, *, batch_size: int, context_length: int) -> float:
        _require_positive(batch_size, name="batch_size")
        _require_positive(context_length, name="context_length")
        log_batch = math.log2(batch_size)
        log_context = math.log2(context_length)
        return min(
            self._log_cells,
            key=lambda item: (
                (log_batch - item[0]) ** 2 + (log_context - item[1]) ** 2,
                item[2].alpha,
                item[2].batch_size,
                item[2].context_length,
            ),
        )[2].alpha

    def to_dict(self) -> dict[str, Any]:
        return {"cells": [cell.to_dict() for cell in self.cells]}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AcceptanceCorrectionTable:
        return cls(
            cells=tuple(
                AcceptanceCorrectionCell.from_dict(cell) for cell in value["cells"]
            )
        )


@dataclass(frozen=True)
class AcceptanceCorrectionSurface:


    coefficients: tuple[float, float, float, float, float, float]
    log2_batch_center: float
    log2_batch_scale: float
    log2_context_center: float
    log2_context_scale: float
    batch_size_min: int
    batch_size_max: int
    context_length_min: int
    context_length_max: int
    alpha_min: float = 0.0
    alpha_max: float = 1.0

    def __post_init__(self) -> None:
        coefficients = tuple(self.coefficients)
        if len(coefficients) != 6 or not all(
            _is_finite_number(value) for value in coefficients
        ):
            raise ValueError("coefficients must contain six finite numbers")
        object.__setattr__(self, "coefficients", coefficients)
        for name in ("log2_batch_center", "log2_context_center"):
            if not _is_finite_number(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        for name in ("log2_batch_scale", "log2_context_scale"):
            value = getattr(self, name)
            if not _is_finite_number(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for prefix in ("batch_size", "context_length"):
            lower = getattr(self, f"{prefix}_min")
            upper = getattr(self, f"{prefix}_max")
            _require_positive(lower, name=f"{prefix}_min")
            _require_positive(upper, name=f"{prefix}_max")
            if lower > upper:
                raise ValueError(f"{prefix}_min cannot exceed {prefix}_max")
        if (
            not _is_finite_number(self.alpha_min)
            or not _is_finite_number(self.alpha_max)
            or not 0.0 <= self.alpha_min <= self.alpha_max <= 1.0
        ):
            raise ValueError("alpha range must satisfy 0 <= min <= max <= 1")

    def estimate(self, *, batch_size: int, context_length: int) -> float:
        _require_positive(batch_size, name="batch_size")
        _require_positive(context_length, name="context_length")
        bounded_batch = min(max(batch_size, self.batch_size_min), self.batch_size_max)
        bounded_context = min(
            max(context_length, self.context_length_min), self.context_length_max
        )
        x_batch = (
            math.log2(bounded_batch) - self.log2_batch_center
        ) / self.log2_batch_scale
        x_context = (
            math.log2(bounded_context) - self.log2_context_center
        ) / self.log2_context_scale
        features = (
            1.0,
            x_batch,
            x_context,
            x_batch * x_context,
            x_batch * x_batch,
            x_context * x_context,
        )
        alpha = sum(
            coefficient * feature
            for coefficient, feature in zip(self.coefficients, features)
        )
        return min(max(alpha, self.alpha_min), self.alpha_max)

    def to_dict(self) -> dict[str, Any]:
        return {
            "coefficients": list(self.coefficients),
            "log2_batch_center": self.log2_batch_center,
            "log2_batch_scale": self.log2_batch_scale,
            "log2_context_center": self.log2_context_center,
            "log2_context_scale": self.log2_context_scale,
            "batch_size_min": self.batch_size_min,
            "batch_size_max": self.batch_size_max,
            "context_length_min": self.context_length_min,
            "context_length_max": self.context_length_max,
            "alpha_min": self.alpha_min,
            "alpha_max": self.alpha_max,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AcceptanceCorrectionSurface:
        return cls(
            coefficients=tuple(value["coefficients"]),
            log2_batch_center=value["log2_batch_center"],
            log2_batch_scale=value["log2_batch_scale"],
            log2_context_center=value["log2_context_center"],
            log2_context_scale=value["log2_context_scale"],
            batch_size_min=value["batch_size_min"],
            batch_size_max=value["batch_size_max"],
            context_length_min=value["context_length_min"],
            context_length_max=value["context_length_max"],
            alpha_min=value.get("alpha_min", 0.0),
            alpha_max=value.get("alpha_max", 1.0),
        )


@dataclass(frozen=True)
class VerifyShapeProfileDB:
    fingerprint: ProfileFingerprint
    points: tuple[VerifyProfilePoint, ...]
    schema_version: int = PROFILE_SCHEMA_VERSION
    acceptance_correction_surface: Optional[AcceptanceCorrectionSurface] = None
    acceptance_correction_table: Optional[AcceptanceCorrectionTable] = None
    _points_by_exact_axes: Mapping[
        tuple[int, str], tuple[VerifyProfilePoint, ...]
    ] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != PROFILE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported profile schema_version={self.schema_version}; "
                f"expected {PROFILE_SCHEMA_VERSION}"
            )
        if not isinstance(self.fingerprint, ProfileFingerprint):
            raise ValueError(
                "fingerprint must be ProfileFingerprint, "
                f"got {type(self.fingerprint)!r}"
            )
        if self.acceptance_correction_surface is not None and not isinstance(
            self.acceptance_correction_surface, AcceptanceCorrectionSurface
        ):
            raise ValueError(
                "acceptance_correction_surface must be "
                "AcceptanceCorrectionSurface when present"
            )
        if self.acceptance_correction_table is not None and not isinstance(
            self.acceptance_correction_table, AcceptanceCorrectionTable
        ):
            raise ValueError(
                "acceptance_correction_table must be "
                "AcceptanceCorrectionTable when present"
            )

        points = tuple(self.points)
        if any(not isinstance(point, VerifyProfilePoint) for point in points):
            raise ValueError("points must contain only VerifyProfilePoint values")
        shapes = [point.shape for point in points]
        if len(set(shapes)) != len(shapes):
            raise ValueError("profile points must have unique VerifyShape keys")
        for point in points:
            if (
                point.shape.physical_row_bucket
                not in self.fingerprint.physical_row_grid
            ):
                raise ValueError(
                    "profile point row bucket is absent from fingerprint grid: "
                    f"{point.shape.physical_row_bucket} not in "
                    f"{self.fingerprint.physical_row_grid}"
                )
        object.__setattr__(self, "points", points)

        points_by_exact_axes: dict[tuple[int, str], list[VerifyProfilePoint]] = {}
        for point in points:
            key = (point.shape.physical_row_bucket, point.shape.execution_mode)
            points_by_exact_axes.setdefault(key, []).append(point)
        object.__setattr__(
            self,
            "_points_by_exact_axes",
            MappingProxyType(
                {
                    key: tuple(
                        sorted(
                            group,
                            key=lambda point: _covering_coordinates(point.shape),
                        )
                    )
                    for key, group in points_by_exact_axes.items()
                }
            ),
        )

    def __reduce__(self):


        return (
            type(self),
            (
                self.fingerprint,
                self.points,
                self.schema_version,
                self.acceptance_correction_surface,
                self.acceptance_correction_table,
            ),
        )

    def estimate(
        self,
        shape: VerifyShape,
        *,
        runtime_fingerprint: ProfileFingerprint,
    ) -> Optional[ProfileEstimate]:


        if runtime_fingerprint != self.fingerprint:
            return None
        if shape.physical_row_bucket not in runtime_fingerprint.physical_row_grid:
            return None

        candidates = self._points_by_exact_axes.get(
            (shape.physical_row_bucket, shape.execution_mode),
            (),
        )
        query_coordinates = _covering_coordinates(shape)


        low = 0
        high = len(candidates)
        while low < high:
            middle = (low + high) // 2
            middle_coordinates = _covering_coordinates(candidates[middle].shape)
            if middle_coordinates < query_coordinates:
                low = middle + 1
            else:
                high = middle

        point: Optional[VerifyProfilePoint] = None
        for index in range(low, len(candidates)):
            candidate = candidates[index]
            if candidate.shape.covers(shape):
                point = candidate
                break
        if point is None:
            return None
        return ProfileEstimate(
            requested_shape=shape,
            profiled_shape=point.shape,
            p50_gpu_ms=point.p50_gpu_ms,
            sample_count=point.sample_count,
            exact=point.shape == shape,
        )

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "fingerprint": self.fingerprint.to_dict(),
            "points": [point.to_dict() for point in self.points],
        }
        if self.acceptance_correction_surface is not None:
            value["acceptance_correction_surface"] = (
                self.acceptance_correction_surface.to_dict()
            )
        if self.acceptance_correction_table is not None:
            value["acceptance_correction_table"] = (
                self.acceptance_correction_table.to_dict()
            )
        return value

    def to_json(self, *, indent: Optional[int] = None) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            separators=None if indent is not None else (",", ":"),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> VerifyShapeProfileDB:
        try:
            return cls(
                schema_version=value["schema_version"],
                fingerprint=ProfileFingerprint.from_dict(value["fingerprint"]),
                points=tuple(
                    VerifyProfilePoint.from_dict(point) for point in value["points"]
                ),
                acceptance_correction_surface=(
                    None
                    if value.get("acceptance_correction_surface") is None
                    else AcceptanceCorrectionSurface.from_dict(
                        value["acceptance_correction_surface"]
                    )
                ),
                acceptance_correction_table=(
                    None
                    if value.get("acceptance_correction_table") is None
                    else AcceptanceCorrectionTable.from_dict(
                        value["acceptance_correction_table"]
                    )
                ),
            )
        except (KeyError, TypeError) as error:
            raise ValueError(f"invalid shape profile payload: {error}") from error

    @classmethod
    def from_json(cls, data: str) -> VerifyShapeProfileDB:
        try:
            value = json.loads(data)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid shape profile JSON: {error}") from error
        if not isinstance(value, dict):
            raise ValueError("shape profile JSON root must be an object")
        return cls.from_dict(value)

    @classmethod
    def load(cls, path: str | Path) -> VerifyShapeProfileDB:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def dump(self, path: str | Path, *, indent: int = 2) -> None:
        Path(path).write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")


def build_profile_db(
    *,
    fingerprint: ProfileFingerprint,
    points: Sequence[VerifyProfilePoint],
    acceptance_correction_surface: Optional[AcceptanceCorrectionSurface] = None,
    acceptance_correction_table: Optional[AcceptanceCorrectionTable] = None,
) -> VerifyShapeProfileDB:


    return VerifyShapeProfileDB(
        fingerprint=fingerprint,
        points=tuple(points),
        acceptance_correction_surface=acceptance_correction_surface,
        acceptance_correction_table=acceptance_correction_table,
    )
