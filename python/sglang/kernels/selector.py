from __future__ import annotations

from functools import lru_cache
from typing import Callable, Optional

from sglang.kernels.registry import registry
from sglang.kernels.spec import KernelBackend, KernelSpec, PlatformInfo


@lru_cache(maxsize=1)
def _platform() -> PlatformInfo:
    return PlatformInfo.detect()


def select_kernel(op: str, backend: Optional[KernelBackend] = None) -> KernelSpec:

    specs = registry.get(op)
    if not specs:
        raise KeyError(f"No kernels registered for op {op!r}")

    if backend is not None:
        for spec in specs:
            if spec.backend == backend:
                return spec
        raise KeyError(f"No '{backend.value}' backend registered for op {op!r}")

    if len(specs) == 1:
        return specs[0]


    platform = _platform()
    eligible = [s for s in specs if s.is_available(platform)]
    if len(eligible) == 1:
        return eligible[0]
    if not eligible:
        raise ValueError(
            f"op {op!r} has no backend usable on device {platform.device.value!r} "
            f"(registered: {[s.backend.value for s in specs]})"
        )
    raise ValueError(
        f"op {op!r} has multiple backends usable on device "
        f"{platform.device.value!r} ({[s.backend.value for s in eligible]}); "
        f"pass backend=... to choose one"
    )


@lru_cache(maxsize=None)
def _resolve(op: str, backend: Optional[KernelBackend]) -> Callable:
    return select_kernel(op, backend=backend).load()


def get_kernel(op: str, backend: Optional[KernelBackend] = None) -> Callable:

    return _resolve(op, backend)


def clear_cache() -> None:

    _resolve.cache_clear()
