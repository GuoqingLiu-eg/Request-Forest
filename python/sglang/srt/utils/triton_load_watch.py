from __future__ import annotations

import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.utils.common import get_available_gpu_memory

logger = logging.getLogger(__name__)

_serving_started = False
_prev_compile_listener = None
_installed = False


def install() -> None:

    global _installed, _prev_compile_listener
    if _installed:
        return
    try:
        import triton.knobs as knobs
    except ImportError:
        return


    runtime = getattr(knobs, "runtime", None)
    compilation = getattr(knobs, "compilation", None)
    hook = getattr(runtime, "kernel_load_start_hook", None)
    if hook is None or compilation is None or not hasattr(compilation, "listener"):
        return

    _prev_compile_listener = compilation.listener
    hook.add(_on_kernel_load)
    compilation.listener = _on_compilation
    _installed = True


def mark_serving_started() -> None:

    global _serving_started
    _serving_started = True


def _on_compilation(*, src, metadata, metadata_group, times, cache_hit) -> None:
    if _prev_compile_listener is not None:
        _prev_compile_listener(
            src=src,
            metadata=metadata,
            metadata_group=metadata_group,
            times=times,
            cache_hit=cache_hit,
        )
    if not _serving_started or cache_hit:
        return

    compile_time_secs = times.total / 1e6
    if compile_time_secs < envs.SGLANG_TRITON_SLOW_COMPILE_THRESHOLD_SECS.get():
        return

    logger.warning(
        "Triton kernel '%s' took %.2f s to compile after serving started. "
        "Serving-time compilation can stall the engine; pre-compile it during "
        "engine init.",
        src.name,
        compile_time_secs,
    )


def _on_kernel_load(module, function, name, metadata_group, hash) -> None:
    if not _serving_started:
        return

    free_gb = None
    if torch.cuda.is_available():
        try:
            free_gb = get_available_gpu_memory(
                "cuda", torch.cuda.current_device(), empty_cache=False
            )
        except RuntimeError:
            logger.debug("Unable to query free device memory", exc_info=True)

    should_crash = envs.SGLANG_CRASH_ON_TRITON_LOAD_AFTER_READY.get()
    if not should_crash and (
        free_gb is None or free_gb >= envs.SGLANG_TRITON_LOAD_WARNING_THRESHOLD_GB.get()
    ):
        return

    free_memory = f"{free_gb:.2f} GiB" if free_gb is not None else "unknown"
    msg = (
        f"Triton kernel '{name}' device-loaded after serving started "
        f"(free device mem: {free_memory}). Pre-load it during engine init "
        f"to avoid CUDA OOM."
    )
    if should_crash:
        raise RuntimeError(msg)
    logger.warning(msg)
