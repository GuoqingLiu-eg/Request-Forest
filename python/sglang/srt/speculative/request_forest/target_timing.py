from __future__ import annotations

import math
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterator, Optional, Protocol

from .profile import ProfileFingerprint, VerifyShape


class TargetTimingScope(str, Enum):


    DRAFT_STAGE = "draft_stage"
    TARGET_FORWARD = "target_forward"
    VERIFICATION_STAGE = "verification_stage"


@dataclass(frozen=True, slots=True)
class TargetTimingKey:


    fingerprint: ProfileFingerprint
    shape: VerifyShape
    scope: TargetTimingScope

    def __post_init__(self) -> None:
        if not isinstance(self.fingerprint, ProfileFingerprint):
            raise TypeError("fingerprint must be ProfileFingerprint")
        if not isinstance(self.shape, VerifyShape):
            raise TypeError("shape must be VerifyShape")
        if not isinstance(self.scope, TargetTimingScope):
            raise TypeError("scope must be TargetTimingScope")


@dataclass(frozen=True, slots=True)
class TargetTimingObservation:


    sequence: int
    key: TargetTimingKey
    gpu_ms: float


    round_id: int | None = None


@dataclass(frozen=True, slots=True)
class TargetTimingSnapshot:


    key: TargetTimingKey
    sample_count: int
    last_gpu_ms: float
    ema_gpu_ms: float


@dataclass(frozen=True, slots=True)
class TargetTimingTelemetry:


    pending: int
    active: int
    submitted: int
    completed: int
    skipped_capacity: int
    invalid_samples: int


class _TimingEvent(Protocol):
    def record(self, stream: Any = None) -> None: ...

    def query(self) -> bool: ...

    def elapsed_time(self, end_event: Any) -> float: ...


@dataclass(slots=True)
class _EmaState:
    sample_count: int
    last_gpu_ms: float
    ema_gpu_ms: float


@dataclass(slots=True)
class _PendingTiming:
    key: TargetTimingKey
    start_event: _TimingEvent
    end_event: _TimingEvent
    round_id: int | None


@dataclass(slots=True)
class TargetTimingHandle:


    key: TargetTimingKey
    _start_event: _TimingEvent
    _stream: Any
    _owner: Any
    round_id: int | None = None
    _closed: bool = False


def _make_cuda_event() -> _TimingEvent:


    import torch

    return torch.cuda.Event(enable_timing=True)


def _record_event(event: _TimingEvent, stream: Any) -> None:
    if stream is None:
        event.record()
    else:
        event.record(stream)


class AsyncTargetVerificationTimer:


    def __init__(
        self,
        *,
        ema_alpha: float = 0.2,
        max_pending: int = 128,
        telemetry_capacity: int = 512,
        event_factory: Optional[Callable[[], _TimingEvent]] = None,
    ) -> None:
        if (
            not isinstance(ema_alpha, (int, float))
            or isinstance(ema_alpha, bool)
            or not math.isfinite(ema_alpha)
            or not 0.0 < ema_alpha <= 1.0
        ):
            raise ValueError("ema_alpha must be finite and in (0, 1]")
        if (
            not isinstance(max_pending, int)
            or isinstance(max_pending, bool)
            or max_pending < 1
        ):
            raise ValueError("max_pending must be a positive integer")
        if (
            not isinstance(telemetry_capacity, int)
            or isinstance(telemetry_capacity, bool)
            or telemetry_capacity < 1
        ):
            raise ValueError("telemetry_capacity must be a positive integer")

        self._ema_alpha = float(ema_alpha)
        self._max_pending = max_pending
        self._event_factory = event_factory or _make_cuda_event


        self._reuse_events = event_factory is None
        self._free_events: list[_TimingEvent] = []
        self._pending: deque[_PendingTiming] = deque()
        self._recent: deque[TargetTimingObservation] = deque(
            maxlen=telemetry_capacity
        )
        self._states: dict[TargetTimingKey, _EmaState] = {}
        self._active = 0
        self._submitted = 0
        self._completed = 0
        self._skipped_capacity = 0
        self._invalid_samples = 0
        self._next_sequence = 1

    def begin(
        self,
        key: TargetTimingKey,
        *,
        stream: Any = None,
        round_id: int | None = None,
    ) -> Optional[TargetTimingHandle]:


        if not isinstance(key, TargetTimingKey):
            raise TypeError("key must be TargetTimingKey")


        self._poll_oldest()
        if len(self._pending) + self._active >= self._max_pending:
            self._skipped_capacity += 1
            return None

        start_event = self._take_event()
        _record_event(start_event, stream)
        self._active += 1
        return TargetTimingHandle(
            key=key,
            _start_event=start_event,
            _stream=stream,
            _owner=self,
            round_id=round_id,
        )

    def end(self, handle: Optional[TargetTimingHandle]) -> None:


        if handle is None:
            return
        if not isinstance(handle, TargetTimingHandle) or handle._owner is not self:
            raise ValueError("timing handle belongs to a different timer")
        if handle._closed:
            raise RuntimeError("timing handle was already ended")

        handle._closed = True
        self._active -= 1
        end_event = self._take_event()
        _record_event(end_event, handle._stream)
        self._pending.append(
            _PendingTiming(
                key=handle.key,
                start_event=handle._start_event,
                end_event=end_event,
                round_id=handle.round_id,
            )
        )
        self._submitted += 1

    @contextmanager
    def measure(
        self,
        key: TargetTimingKey,
        *,
        stream: Any = None,
        round_id: int | None = None,
    ) -> Iterator[Optional[TargetTimingHandle]]:


        handle = self.begin(key, stream=stream, round_id=round_id)
        try:
            yield handle
        finally:
            self.end(handle)

    def poll(
        self,
        *,
        max_results: Optional[int] = None,
    ) -> tuple[TargetTimingObservation, ...]:


        if max_results is not None and (
            not isinstance(max_results, int)
            or isinstance(max_results, bool)
            or max_results < 0
        ):
            raise ValueError("max_results must be a non-negative integer or None")
        if max_results == 0 or not self._pending:
            return ()

        remaining: deque[_PendingTiming] = deque()
        observations: list[TargetTimingObservation] = []
        while self._pending:
            interval = self._pending.popleft()
            if max_results is not None and len(observations) >= max_results:
                remaining.append(interval)
                continue
            if not interval.end_event.query():
                remaining.append(interval)
                continue

            observation = self._complete(interval)
            if observation is not None:
                observations.append(observation)

        self._pending = remaining
        return tuple(observations)

    def _poll_oldest(self) -> Optional[TargetTimingObservation]:
        if not self._pending:
            return None
        interval = self._pending[0]
        if not interval.end_event.query():
            return None
        self._pending.popleft()
        return self._complete(interval)

    def _complete(
        self, interval: _PendingTiming
    ) -> Optional[TargetTimingObservation]:
        gpu_ms = float(interval.start_event.elapsed_time(interval.end_event))
        self._recycle(interval.start_event, interval.end_event)
        if not math.isfinite(gpu_ms) or gpu_ms < 0.0:
            self._invalid_samples += 1
            return None
        return self._observe(interval.key, gpu_ms, round_id=interval.round_id)

    def _take_event(self) -> _TimingEvent:
        if self._free_events:
            return self._free_events.pop()
        return self._event_factory()

    def _recycle(self, *events: _TimingEvent) -> None:
        if self._reuse_events:
            self._free_events.extend(events)

    def _observe(
        self,
        key: TargetTimingKey,
        gpu_ms: float,
        *,
        round_id: int | None,
    ) -> TargetTimingObservation:
        state = self._states.get(key)
        if state is None:
            state = _EmaState(
                sample_count=1,
                last_gpu_ms=gpu_ms,
                ema_gpu_ms=gpu_ms,
            )
            self._states[key] = state
        else:
            state.sample_count += 1
            state.last_gpu_ms = gpu_ms
            state.ema_gpu_ms += self._ema_alpha * (gpu_ms - state.ema_gpu_ms)

        observation = TargetTimingObservation(
            sequence=self._next_sequence,
            key=key,
            gpu_ms=gpu_ms,
            round_id=round_id,
        )
        self._next_sequence += 1
        self._completed += 1
        self._recent.append(observation)
        return observation

    def snapshot(self, key: TargetTimingKey) -> Optional[TargetTimingSnapshot]:


        state = self._states.get(key)
        if state is None:
            return None
        return TargetTimingSnapshot(
            key=key,
            sample_count=state.sample_count,
            last_gpu_ms=state.last_gpu_ms,
            ema_gpu_ms=state.ema_gpu_ms,
        )

    def snapshots(self) -> tuple[TargetTimingSnapshot, ...]:


        return tuple(
            TargetTimingSnapshot(
                key=key,
                sample_count=state.sample_count,
                last_gpu_ms=state.last_gpu_ms,
                ema_gpu_ms=state.ema_gpu_ms,
            )
            for key, state in self._states.items()
        )

    def recent_observations(self) -> tuple[TargetTimingObservation, ...]:
        return tuple(self._recent)

    def telemetry(self) -> TargetTimingTelemetry:
        return TargetTimingTelemetry(
            pending=len(self._pending),
            active=self._active,
            submitted=self._submitted,
            completed=self._completed,
            skipped_capacity=self._skipped_capacity,
            invalid_samples=self._invalid_samples,
        )


__all__ = [
    "AsyncTargetVerificationTimer",
    "TargetTimingHandle",
    "TargetTimingKey",
    "TargetTimingObservation",
    "TargetTimingScope",
    "TargetTimingSnapshot",
    "TargetTimingTelemetry",
]
