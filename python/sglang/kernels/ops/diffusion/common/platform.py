from __future__ import annotations

from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)

_CUDA_LIKE = frozenset({"cuda", "hip"})


def platform_key() -> str:

    from sglang.multimodal_gen.runtime.platforms import current_platform

    for name in ("cuda", "hip", "npu", "mps", "musa"):
        if getattr(current_platform, f"is_{name}")():
            return name
    return "cpu"


def is_cuda() -> bool:

    from sglang.multimodal_gen.runtime.platforms import current_platform

    return current_platform.is_cuda()


def is_hip() -> bool:

    from sglang.multimodal_gen.runtime.platforms import current_platform

    return current_platform.is_hip()


def has_triton() -> bool:

    return platform_key() in _CUDA_LIKE


def lazy_fallback(kind: str, name: str) -> Callable:


    def _call(*args, **kwargs):
        from importlib import import_module

        impl = getattr(
            import_module(f"sglang.kernels.ops.diffusion.common.fallback_{kind}"), name
        )
        return impl(*args, **kwargs)

    _call.__name__ = name
    _call.__qualname__ = f"{kind}_fallback.{name}"
    return _call


def select_impl(triton_impl: F, **fallbacks: F) -> F:

    return fallbacks.get(platform_key(), triton_impl)
