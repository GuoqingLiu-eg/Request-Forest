from __future__ import annotations

import logging
import os
import pathlib
import shutil
from typing import List, Tuple

import torch

from sglang.kernels.jit.utils.arch import get_jit_cuda_arch
from sglang.kernels.jit.utils.common import cache_once, is_hip_runtime

logger = logging.getLogger(__name__)


@cache_once
def cuda_home() -> str:

    configured = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if configured is not None:
        return configured
    nvcc_path = shutil.which("nvcc")
    if nvcc_path is not None:
        return os.path.dirname(os.path.dirname(nvcc_path))
    return "/usr/local/cuda"


@cache_once
def rocm_home() -> str:

    return os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH") or "/opt/rocm"


@cache_once
def device_compiler_path() -> str:

    if is_hip_runtime():
        return os.path.join(rocm_home(), "bin", "hipcc")
    return os.path.join(cuda_home(), "bin", "nvcc")


@cache_once
def host_compiler_path() -> str:

    return os.environ.get("CXX", "c++")


@cache_once
def gpu_arch_name() -> str:

    if not is_hip_runtime():
        return get_jit_cuda_arch().target_name
    try:
        device = torch.cuda.current_device()
        return str(torch.cuda.get_device_properties(device).gcnArchName)
    except Exception:
        logger.warning("Cannot detect ROCm gcnArchName; the JIT cache target degrades.")
        return "unknown"


@cache_once
def toolkit_home() -> pathlib.Path:

    return pathlib.Path(device_compiler_path()).parent.parent


@cache_once
def tvm_ffi_paths() -> Tuple[Tuple[str, ...], str, str]:

    from tvm_ffi.libinfo import (
        find_dlpack_include_path,
        find_include_path,
        find_libtvm_ffi,
    )

    lib = pathlib.Path(find_libtvm_ffi())
    includes = tuple(dict.fromkeys([find_include_path(), find_dlpack_include_path()]))
    return includes, str(lib.parent), lib.stem.removeprefix("lib")


def target_flags() -> List[str]:

    if is_hip_runtime():
        return [f"--offload-arch={gpu_arch_name()}"]
    arch = get_jit_cuda_arch()
    target = f"{arch.major}{arch.minor}{arch.suffix}"
    return [f"-gencode=arch=compute_{target},code=sm_{target}"]


def base_cxx_flags() -> List[str]:

    return ["-fPIC"]


def base_cuda_flags() -> List[str]:
    if is_hip_runtime():
        return ["-fPIC", "-D__HIP_PLATFORM_AMD__=1", "-fno-gpu-rdc"]
    return ["-Xcompiler", "-fPIC"]


def base_include_paths() -> List[str]:
    includes, _, _ = tvm_ffi_paths()
    if is_hip_runtime():
        return [*includes, f"{rocm_home()}/include"]
    return list(includes)


def base_link_flags(*, with_device: bool) -> List[str]:

    _, lib_dir, lib_name = tvm_ffi_paths()
    flags = ["-shared", f"-L{lib_dir}", f"-l{lib_name}"]
    if not with_device:
        return flags
    if is_hip_runtime():
        return flags + [f"-L{rocm_home()}/lib", "-lamdhip64"]
    return flags + [f"-L{cuda_home()}/lib64", "-lcudart"]


def compilers() -> Tuple[str, str]:

    return host_compiler_path(), device_compiler_path()
