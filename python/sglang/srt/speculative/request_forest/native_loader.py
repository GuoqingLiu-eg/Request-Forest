from __future__ import annotations

import os
from pathlib import Path

import torch


CPU_LIBRARY_ENV = "SGLANG_REQUEST_FOREST_CPU_OPS_LIBRARY"
CUDA_LIBRARY_ENV = "SGLANG_REQUEST_FOREST_CUDA_OPS_LIBRARY"

_CPU_OPS = (
    "request_forest_select_cpu",
    "request_forest_dflash2_lower_cpu",
    "request_forest_compile_cpu",
    "request_forest_select_compile_cpu",
    "request_forest_commit_cpu",
)
_CUDA_OPS = (
    "request_forest_build_tree_attention",
    "request_forest_commit_kv",
    "request_forest_verify_greedy",
)


def _missing_ops(names: tuple[str, ...]) -> tuple[str, ...]:
    namespace = torch.ops.sgl_kernel
    return tuple(name for name in names if not hasattr(namespace, name))


def _load_library_for_ops(
    *,
    kind: str,
    env_name: str,
    required_ops: tuple[str, ...],
) -> None:
    missing = _missing_ops(required_ops)
    if not missing:
        return

    configured_path = os.environ.get(env_name)
    if not configured_path:
        raise RuntimeError(
            f"Request Forest {kind} operators are not loaded ({', '.join(missing)}). "
            f"Build the standalone {kind} library and set {env_name} to its "
            "absolute .so path before launching the server."
        )
    library_path = Path(configured_path).expanduser()
    if not library_path.is_absolute() or not library_path.is_file():
        raise RuntimeError(
            f"{env_name} must name an existing absolute .so file; got "
            f"{configured_path!r}"
        )

    torch.ops.load_library(str(library_path))
    missing_after_load = _missing_ops(required_ops)
    if missing_after_load:
        raise RuntimeError(
            f"{library_path} did not register the required Request Forest "
            f"{kind} operators: {', '.join(missing_after_load)}"
        )


def load_request_forest_native_ops() -> None:


    _load_library_for_ops(
        kind="CPU",
        env_name=CPU_LIBRARY_ENV,
        required_ops=_CPU_OPS,
    )
    _load_library_for_ops(
        kind="CUDA",
        env_name=CUDA_LIBRARY_ENV,
        required_ops=_CUDA_OPS,
    )


__all__ = [
    "CPU_LIBRARY_ENV",
    "CUDA_LIBRARY_ENV",
    "load_request_forest_native_ops",
]
