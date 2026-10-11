from __future__ import annotations

import functools
from typing import Any, Callable

_WRAPPED_MARKER_ATTR = "_kv_canary_wrapped_by"


def wrap_method(
    obj: object,
    method_name: str,
    *,
    wrapper: Callable[..., Any],
) -> None:

    if not hasattr(obj, method_name):
        raise AttributeError(
            f"kv-canary: {type(obj).__name__} missing required method {method_name!r}"
        )
    original = getattr(obj, method_name)
    if getattr(original, _WRAPPED_MARKER_ATTR, None) is not None:
        raise RuntimeError(
            f"kv-canary: {type(obj).__name__}.{method_name} already wrapped by kv-canary"
        )

    @functools.wraps(original)
    def patched(*args: Any, **kwargs: Any) -> Any:
        return wrapper(original, *args, **kwargs)

    setattr(patched, _WRAPPED_MARKER_ATTR, method_name)
    setattr(obj, method_name, patched)
