from __future__ import annotations

import contextlib
import fcntl
import logging
import os
import pathlib
import shutil
import uuid
from typing import TYPE_CHECKING, List, Tuple

import torch

from sglang.kernels.jit.utils.arch import get_default_target_flags
from sglang.kernels.jit.utils.common import is_hip_runtime
from sglang.kernels.jit.utils.compile import cache, ninja
from sglang.kernels.jit.utils.compile.paths import (
    DEFAULT_CFLAGS,
    DEFAULT_INCLUDE,
    DEFAULT_LDFLAGS,
)
from sglang.kernels.jit.utils.compile.spec import BuildSpec, resolve_sources
from sglang.kernels.jit.utils.deps import REGISTERED_DEPENDENCIES

if TYPE_CHECKING:
    from tvm_ffi import Module

    _DISABLE_TORCH_COMPILE = lambda f: f
else:

    _DISABLE_TORCH_COMPILE = torch.compiler.disable


logger = logging.getLogger(__name__)

_LOCK_FILE = ".lock"


@_DISABLE_TORCH_COMPILE
def load_jit(
    *args: str,
    cpp_files: List[str] | None = None,
    cuda_files: List[str] | None = None,
    cpp_wrappers: List[Tuple[str, str]] | None = None,
    cuda_wrappers: List[Tuple[str, str]] | None = None,
    extra_cflags: List[str] | None = None,
    extra_cuda_cflags: List[str] | None = None,
    extra_ldflags: List[str] | None = None,
    extra_include_paths: List[str] | None = None,
    extra_dependencies: List[str] | None = None,
    header_only: bool = True,
) -> Module:

    if is_hip_runtime():
        extra_cuda_cflags = [
            flag
            for flag in (extra_cuda_cflags or [])
            if flag not in ("--use_fast_math", "-use_fast_math")
        ]

    includes = list(DEFAULT_INCLUDE) + (extra_include_paths or [])
    for dep in sorted(set(extra_dependencies or [])):
        if dep not in REGISTERED_DEPENDENCIES:
            raise ValueError(f"Dependency {dep} is not registered.")
        includes += REGISTERED_DEPENDENCIES[dep]()

    module_args = tuple(str(arg) for arg in args)
    spec = BuildSpec(
        module_args=module_args,
        cpp_files=resolve_sources(cpp_files),
        cuda_files=resolve_sources(cuda_files),
        cpp_wrappers=tuple(cpp_wrappers or ()),
        cuda_wrappers=tuple(cuda_wrappers or ()),
        cflags=tuple(DEFAULT_CFLAGS + (extra_cflags or [])),
        cuda_cflags=tuple(get_default_target_flags() + (extra_cuda_cflags or [])),
        ldflags=tuple(DEFAULT_LDFLAGS + (extra_ldflags or [])),
        include_paths=tuple(includes),
        header_only=header_only,
    )


    build_file = ninja.generate(spec)

    build_key = cache.compute_build_key(spec, build_file=build_file)
    scope = cache.build_key_dir(module_name=spec.module_name, build_key=build_key)

    prebuilt = cache.find_prebuilt(scope=scope, module_name=spec.module_name)
    if prebuilt is not None:
        try:
            return _load(prebuilt)
        except Exception as e:


            logger.warning(
                "Cached JIT module %s failed to load; rebuilding. " "Got error: %s",
                spec.module_name,
                e,
            )

    with _build_lock(scope):


        prebuilt = cache.find_prebuilt(scope=scope, module_name=spec.module_name)
        if prebuilt is not None:
            try:
                return _load(prebuilt)
            except Exception as e:
                logger.warning(
                    "Cached JIT module %s failed to load; rebuilding. Got error: %s",
                    spec.module_name,
                    e,
                )


                shutil.rmtree(prebuilt.parent, ignore_errors=True)


        staging = scope / f".staging-{uuid.uuid4().hex}"
        try:
            library = ninja.build(spec=spec, build_dir=staging, build_file=build_file)


            module = _load(library)
            cache.commit_build(
                spec,
                scope=scope,
                staging=staging,
                dependencies=ninja.scan_dependencies(staging),
            )
            return module
        finally:
            shutil.rmtree(staging, ignore_errors=True)


@contextlib.contextmanager
def _build_lock(scope: pathlib.Path):

    scope.mkdir(parents=True, exist_ok=True)
    handle = os.open(scope / _LOCK_FILE, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield
    finally:
        os.close(handle)


def _load(library: pathlib.Path) -> Module:
    from tvm_ffi import load_module

    return load_module(str(library))
