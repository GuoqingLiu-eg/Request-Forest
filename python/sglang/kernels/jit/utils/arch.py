from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List

import torch

from sglang.kernels.jit.utils.common import (
    cache_once,
    is_hip_runtime,
    is_musa_runtime,
)

logger = logging.getLogger(__name__)


@dataclass
class ArchInfo:
    major: int
    minor: int
    suffix: str

    @property
    def target_name(self) -> str:
        return f"{self.major}.{self.minor}{self.suffix}"

    @property
    def jit_flag(self) -> str:
        return f"-DSGL_CUDA_ARCH={self.major * 100 + self.minor * 10}"


@cache_once
def _jit_cuda_version() -> tuple[int, ...]:

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is None:
        nvcc_path = shutil.which("nvcc")
        cuda_home = (
            os.path.dirname(os.path.dirname(nvcc_path))
            if nvcc_path is not None
            else "/usr/local/cuda"
        )
    nvcc = os.path.join(cuda_home, "bin", "nvcc")
    try:
        output = subprocess.check_output([nvcc, "--version"], text=True)
        match = re.search(r"release (\d+)\.(\d+)", output)
        if match is not None:
            return int(match.group(1)), int(match.group(2))
        logger.warning("Cannot parse `%s --version` output: %s", nvcc, output)
    except (OSError, subprocess.SubprocessError) as error:
        logger.warning("Cannot run `%s --version`: %s", nvcc, error)
    from sglang.srt.utils.common import get_cuda_version

    return get_cuda_version()


def _cuda_arch_suffix(major: int, minor: int) -> str:

    if major == 9:
        return "a"
    if major == 12:
        if minor == 0 and _jit_cuda_version() >= (12, 9):
            return "f"
        return "a"
    if major >= 10:
        return "a"
    return ""


@cache_once
def _init_jit_cuda_arch_once():
    global _CUDA_ARCH
    try:
        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
    except Exception:
        logger.warning("Cannot detect CUDA architecture.")
        major, minor = 0, 0


    suffix = (
        ""
        if (is_hip_runtime() or is_musa_runtime())
        else _cuda_arch_suffix(major, minor)
    )
    _CUDA_ARCH = ArchInfo(major, minor, suffix)


def get_default_target_flags(arch: ArchInfo | None = None) -> List[str]:

    if is_hip_runtime():
        flags = ["-DUSE_ROCM", "-std=c++20", "-O3"]

        try:
            device = torch.cuda.current_device()
            gcn_arch = torch.cuda.get_device_properties(device).gcnArchName
            if "gfx942" in gcn_arch:
                flags.append("-DHIP_FP8_TYPE_FNUZ=1")
            else:
                flags.append("-DHIP_FP8_TYPE_E4M3=1")
        except Exception:
            flags.append("-DHIP_FP8_TYPE_E4M3=1")
        return flags
    else:
        if arch is None:
            arch = get_jit_cuda_arch()
        return [
            arch.jit_flag,
            "-std=c++20",
            "-O3",
            "--expt-relaxed-constexpr",
        ]


def make_jit_cuda_arch(major: int, minor: int) -> ArchInfo:

    return ArchInfo(major, minor, _cuda_arch_suffix(major, minor))


@contextmanager
def override_jit_cuda_arch(major: int, minor: int, suffix: str | None = None):

    global _CUDA_ARCH
    old_value = get_jit_cuda_arch()
    _CUDA_ARCH = (
        make_jit_cuda_arch(major, minor)
        if suffix is None
        else ArchInfo(major, minor, suffix)
    )
    try:
        yield
    finally:
        _CUDA_ARCH = old_value


def get_jit_cuda_arch() -> ArchInfo:

    _init_jit_cuda_arch_once()
    return _CUDA_ARCH


@cache_once
def is_arch_support_pdl() -> bool:
    if is_hip_runtime() or is_musa_runtime():
        return False
    return get_jit_cuda_arch().major >= 9
