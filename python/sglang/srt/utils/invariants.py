from __future__ import annotations

import enum
import logging
from typing import Callable, Optional

import torch

from sglang.srt.environ import InvariantCheckLevel, envs

logger = logging.getLogger(__name__)


class Bucket(enum.Enum):


    SOFTEN = "soften"
    GUARD = "guard"
    FATAL_CONTAINABLE = "fatal_containable"
    FATAL_UNCONTAINABLE = "fatal_uncontainable"


def resolve_level() -> InvariantCheckLevel:

    if envs.SGLANG_INVARIANT_CHECK.is_set():
        return InvariantCheckLevel(envs.SGLANG_INVARIANT_CHECK.get())
    if envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        return InvariantCheckLevel.STRICT
    return InvariantCheckLevel.OFF


class Property:


    name: str

    def ok(self, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class NotNaN(Property):


    name = "not_nan"

    def ok(self, value: torch.Tensor) -> torch.Tensor:
        return ~torch.isnan(value)


class NotInf(Property):


    name = "not_inf"

    def ok(self, value: torch.Tensor) -> torch.Tensor:
        return ~torch.isinf(value)


class Finite(Property):


    name = "finite"

    def ok(self, value: torch.Tensor) -> torch.Tensor:
        return torch.isfinite(value)


class InRange(Property):


    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi
        self.name = f"in_range[{lo},{hi})"

    def ok(self, value: torch.Tensor) -> torch.Tensor:
        return (value >= self.lo) & (value < self.hi)


class InClosedRange(Property):


    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi
        self.name = f"in_closed_range[{lo},{hi}]"

    def ok(self, value: torch.Tensor) -> torch.Tensor:
        return (value >= self.lo) & (value <= self.hi)


class PageAligned(Property):
    def __init__(self, page_size: int):
        self.page_size = page_size
        self.name = f"page_aligned[{page_size}]"

    def ok(self, value: torch.Tensor) -> torch.Tensor:
        return value % self.page_size == 0


class IsTrue(Property):


    name = "is_true"

    def ok(self, value: torch.Tensor) -> torch.Tensor:
        return value


_REGISTRY: dict[str, Invariant] = {}


class Invariant:


    def __init__(
        self,
        name: str,
        bucket: Bucket,
        prop: Property,
        *,
        recover: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        if bucket is Bucket.FATAL_UNCONTAINABLE and recover is not None:
            raise ValueError(f"uncontainable FATAL {name!r} cannot have a recover")
        if name in _REGISTRY:
            raise ValueError(f"duplicate invariant {name!r}")
        self.name = name
        self.bucket = bucket
        self.prop = prop
        self.recover = recover
        _REGISTRY[name] = self


def registered_invariants() -> dict[str, Invariant]:
    return dict(_REGISTRY)


def _get_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


_FLUSH_EVERY = 512


class _CheckReporter:


    def __init__(self):
        self._dev: dict[str, torch.Tensor] = {}
        self._host: dict[str, torch.Tensor] = {}
        self._calls: dict[str, int] = {}
        self._logged_total: dict[str, int] = {}

    def record(self, key: str, bucket: Bucket, hit_count: torch.Tensor, msg: str):
        dev = self._dev.get(key)
        if dev is None:
            dev = torch.zeros(1, dtype=torch.int64, device=hit_count.device)
            self._dev[key] = dev
            self._host[key] = torch.zeros(
                1, dtype=torch.int64, pin_memory=hit_count.is_cuda
            )
            self._calls[key] = 0
            self._logged_total[key] = 0

        dev.add_(hit_count.reshape(1).to(torch.int64))
        self._host[key].copy_(dev, non_blocking=True)
        self._calls[key] += 1

        total = int(self._host[key][0])
        last = self._logged_total[key]
        first_hit = last == 0 and total > 0
        cadence = self._calls[key] % _FLUSH_EVERY == 0
        if total > last and (first_hit or cadence):
            level = logging.INFO if bucket is Bucket.SOFTEN else logging.WARNING
            logger.log(
                level,
                "invariant-check [%s]: +%d hit(s), %d total. %s",
                key,
                total - last,
                total,
                msg,
            )
            self._logged_total[key] = total


_reporter = _CheckReporter()


def _crashes(bucket: Bucket, level: InvariantCheckLevel) -> bool:

    if bucket is Bucket.SOFTEN:
        return False
    if bucket is Bucket.FATAL_UNCONTAINABLE:
        return level >= InvariantCheckLevel.WARN
    return level == InvariantCheckLevel.STRICT


def _signal(ok: torch.Tensor, *, inv: Invariant, level: InvariantCheckLevel, msg: str):
    if _crashes(inv.bucket, level):

        torch._assert_async(ok.all(), f"invariant-check FAILED [{inv.name}]: {msg}")
        return
    _reporter.record(f"{inv.name}@rank{_get_rank()}", inv.bucket, (~ok).sum(), msg)


def expect(
    inv: Invariant,
    value: Optional[torch.Tensor],
    *,
    msg: str = "",
) -> Optional[torch.Tensor]:

    level = resolve_level()
    if level >= InvariantCheckLevel.WARN and value is not None and value.numel() > 0:
        detail = f"{inv.prop.name}: {msg}" if msg else inv.prop.name
        _signal(inv.prop.ok(value), inv=inv, level=level, msg=detail)

    if inv.recover is not None and value is not None:
        value = inv.recover(value)
    return value
