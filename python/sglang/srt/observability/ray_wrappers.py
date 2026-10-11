# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from __future__ import annotations

import copy
import re
import time
from typing import List, Optional

try:
    from ray import serve as ray_serve
    from ray.util import metrics as ray_metrics
    from ray.util.metrics import Metric
except ImportError:
    ray_metrics = None
    ray_serve = None
    Metric = None  # type: ignore[assignment]

from sglang.srt.observability.metrics_collector import (
    ExpertDispatchCollector,
    RadixCacheMetricsCollector,
    SchedulerMetricsCollector,
    StorageMetricsCollector,
    TokenizerMetricsCollector,
)


def _get_replica_id() -> Optional[str]:

    if ray_serve is None:
        return None
    try:
        return ray_serve.get_replica_context().replica_id.unique_id
    except ray_serve.exceptions.RayServeException:
        return None


class RayPrometheusMetric:


    _is_labeled: bool = False

    def __init__(self) -> None:
        if ray_metrics is None:
            raise ImportError(
                "RayPrometheusMetric requires Ray to be installed. "
                "Install with: pip install 'ray[serve]'"
            )
        self.metric: Optional[Metric] = None
        self._tags: dict = {"ReplicaId": _get_replica_id() or ""}

    @staticmethod
    def _get_tag_keys(labelnames: Optional[List[str]]) -> tuple:
        labels = list(labelnames) if labelnames else []
        labels.append("ReplicaId")
        return tuple(labels)

    def _build_tags(self, *labels: str, **labelskwargs: str) -> dict:
        if labels:


            expected = len(self.metric._tag_keys) - 1
            if len(labels) != expected:
                raise ValueError(
                    "Number of labels must match the number of tag keys. "
                    f"Expected {expected}, got {len(labels)}"
                )
            labelskwargs.update(zip(self.metric._tag_keys, labels))
        labelskwargs["ReplicaId"] = _get_replica_id() or ""
        return {k: v if isinstance(v, str) else str(v) for k, v in labelskwargs.items()}

    def labels(self, *labels: str, **labelskwargs: str) -> RayPrometheusMetric:
        if self._is_labeled:
            raise ValueError("labels() cannot be called on an already-labeled metric.")
        clone = copy.copy(self)
        clone._tags = self._build_tags(*labels, **labelskwargs)
        clone._is_labeled = True
        return clone

    @staticmethod
    def _coerce_positive_boundaries(buckets):


        if not buckets:
            return []
        return [b for b in buckets if b > 0]

    @staticmethod
    def _get_sanitized_opentelemetry_name(name: str) -> str:

        return re.sub(r"[^a-zA-Z0-9_]", "_", name)


class RayCounterWrapper(RayPrometheusMetric):


    def __init__(
        self,
        name: str,
        documentation: Optional[str] = "",
        labelnames: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        tag_keys = self._get_tag_keys(labelnames)
        name = self._get_sanitized_opentelemetry_name(name)
        self.metric = ray_metrics.Counter(
            name=name,
            description=documentation,
            tag_keys=tag_keys,
        )

    def inc(self, value: float = 1.0) -> None:
        if value == 0:
            return
        return self.metric.inc(value, tags=self._tags)


class RayGaugeWrapper(RayPrometheusMetric):


    def __init__(
        self,
        name: str,
        documentation: Optional[str] = "",
        labelnames: Optional[List[str]] = None,
        multiprocess_mode: Optional[str] = "",
    ) -> None:


        del multiprocess_mode
        super().__init__()
        tag_keys = self._get_tag_keys(labelnames)
        name = self._get_sanitized_opentelemetry_name(name)
        self.metric = ray_metrics.Gauge(
            name=name,
            description=documentation,
            tag_keys=tag_keys,
        )

    def set(self, value: float) -> None:
        return self.metric.set(value, tags=self._tags)

    def set_to_current_time(self) -> None:
        return self.set(time.time())


class RayHistogramWrapper(RayPrometheusMetric):


    def __init__(
        self,
        name: str,
        documentation: Optional[str] = "",
        labelnames: Optional[List[str]] = None,
        buckets: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        tag_keys = self._get_tag_keys(labelnames)
        name = self._get_sanitized_opentelemetry_name(name)
        self.metric = ray_metrics.Histogram(
            name=name,
            description=documentation,
            tag_keys=tag_keys,
            boundaries=self._coerce_positive_boundaries(buckets),
        )

    def observe(self, value: float) -> None:
        return self.metric.observe(value, tags=self._tags)


class RaySummaryWrapper(RayPrometheusMetric):


    DEFAULT_BOUNDARIES: List[float] = [
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
    ]

    def __init__(
        self,
        name: str,
        documentation: Optional[str] = "",
        labelnames: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        tag_keys = self._get_tag_keys(labelnames)
        name = self._get_sanitized_opentelemetry_name(name)
        self.metric = ray_metrics.Histogram(
            name=name,
            description=documentation,
            tag_keys=tag_keys,
            boundaries=self._coerce_positive_boundaries(self.DEFAULT_BOUNDARIES),
        )

    def observe(self, value: float) -> None:
        return self.metric.observe(value, tags=self._tags)


class RaySchedulerMetricsCollector(SchedulerMetricsCollector):


    _counter_cls = RayCounterWrapper
    _gauge_cls = RayGaugeWrapper
    _histogram_cls = RayHistogramWrapper
    _summary_cls = RaySummaryWrapper


class RayTokenizerMetricsCollector(TokenizerMetricsCollector):


    _counter_cls = RayCounterWrapper
    _gauge_cls = RayGaugeWrapper
    _histogram_cls = RayHistogramWrapper


class RayStorageMetricsCollector(StorageMetricsCollector):


    _counter_cls = RayCounterWrapper
    _histogram_cls = RayHistogramWrapper


class RayRadixCacheMetricsCollector(RadixCacheMetricsCollector):


    _counter_cls = RayCounterWrapper
    _histogram_cls = RayHistogramWrapper


class RayExpertDispatchCollector(ExpertDispatchCollector):


    _histogram_cls = RayHistogramWrapper
