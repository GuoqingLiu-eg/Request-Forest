import logging
import os
import pkgutil
from importlib.metadata import entry_points

import torch

from sglang.srt.environ import envs
from sglang.srt.platforms.cpu import CpuSRTPlatform
from sglang.srt.platforms.cuda import CudaSRTPlatform
from sglang.srt.platforms.interface import SRTPlatform
from sglang.srt.platforms.rocm import RocmSRTPlatform
from sglang.srt.platforms.xpu import XpuSRTPlatform
from sglang.srt.plugins import PLATFORM_PLUGINS_GROUP, load_plugins_by_group

logger = logging.getLogger(__name__)

_current_platform: SRTPlatform | None = None


def _is_cuda_available() -> bool:
    return bool(torch.cuda.is_available() and torch.version.hip is None)


def _is_rocm_available() -> bool:
    return bool(torch.cuda.is_available() and torch.version.hip is not None)


def _is_cpu_available() -> bool:
    return os.getenv("SGLANG_USE_CPU_ENGINE", "0") == "1"


def _is_xpu_available() -> bool:
    return torch.xpu.is_available()


def _resolve_platform() -> SRTPlatform:

    selected = envs.SGLANG_PLATFORM.get()

    if selected:


        discovered = entry_points(group=PLATFORM_PLUGINS_GROUP)
        ep_map = {ep.name: ep for ep in discovered}

        if selected not in ep_map:
            available = ", ".join(f"'{n}'" for n in ep_map) if ep_map else "none"
            raise RuntimeError(
                f"SGLANG_PLATFORM={selected!r} not found in discovered platform plugins "
                f"(available: {available}). Install the plugin with 'pip install -e' "
                f"to register its entry_points."
            )

        try:
            plugin_fn = ep_map[selected].load()
            result = plugin_fn()
        except Exception:
            logger.exception("Failed to activate platform plugin: %s", selected)
            raise

        if result is None:
            raise RuntimeError(
                f"Platform plugin {selected!r} is installed but activate() returned None "
                f"(hardware not available on this machine?)."
            )
        logger.info("OOT platform plugin activated: %s -> %s", selected, result)
        return _load_platform_class(result)()


    all_plugins = load_plugins_by_group(PLATFORM_PLUGINS_GROUP)

    activated: dict[str, str] = {}
    for name, (plugin_fn, _dist) in all_plugins.items():
        try:
            result = plugin_fn()
            if result is not None:
                activated[name] = result
                logger.info("OOT platform plugin activated: %s -> %s", name, result)
        except Exception:
            logger.exception("Failed to activate platform plugin: %s", name)

    if len(activated) == 0:
        if _is_cpu_available():
            logger.debug("SGLANG_USE_CPU_ENGINE=1. Using CPU SRTPlatform defaults.")
            return CpuSRTPlatform()
        if _is_cuda_available():
            logger.debug(
                "No platform plugin detected. Using CUDA SRTPlatform defaults."
            )
            return CudaSRTPlatform()
        if _is_rocm_available():
            logger.debug(
                "No platform plugin detected. Using ROCm SRTPlatform defaults."
            )
            return RocmSRTPlatform()
        if _is_xpu_available():
            logger.debug("No platform plugin detected. Using XPU SRTPlatform defaults.")
            return XpuSRTPlatform()
        logger.debug("No platform detected. Using base SRTPlatform.")
        return SRTPlatform()

    if len(activated) == 1:
        name, qualname = next(iter(activated.items()))
        return _load_platform_class(qualname)()


    names_str = ", ".join(f"'{n}'" for n in activated)
    raise RuntimeError(
        f"Multiple platform plugins activated: {names_str}. "
        f"Set SGLANG_PLATFORM to select one."
    )


def _load_platform_class(qualname: str) -> type:

    cls = pkgutil.resolve_name(qualname)
    if not isinstance(cls, type) or not issubclass(cls, SRTPlatform):
        raise TypeError(
            f"Expected an SRTPlatform subclass, got {type(cls)}: {qualname}"
        )
    return cls


current_platform: SRTPlatform


def __getattr__(name: str):

    if name == "current_platform":
        global _current_platform
        if _current_platform is None:
            _current_platform = _resolve_platform()
        return _current_platform
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
