#!/usr/bin/env python3
"""JIT-load the standalone Request Forest CUDA operators."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


def load_native_cuda_ops(
    build_directory: str | Path,
    *,
    verbose: bool = False,
) -> Any:
    """Compile Request Forest CUDA ops into ``torch.ops.sgl_kernel``."""

    python_bin = str(Path(sys.executable).absolute().parent)
    inherited_path = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join(
        part for part in (python_bin, inherited_path) if part
    )

    from torch.utils.cpp_extension import load

    repo_root = Path(__file__).resolve().parents[2]
    source = (
        repo_root
        / "python"
        / "sglang"
        / "kernels"
        / "aot"
        / "csrc"
        / "speculative"
        / "request_forest.cu"
    )
    registration = (
        repo_root
        / "benchmark"
        / "request_forest"
        / "native_cuda_registration.cpp"
    )
    build_path = Path(build_directory).expanduser().resolve()
    build_path.mkdir(parents=True, exist_ok=True)
    return load(
        name="request_forest_native_cuda",
        sources=[str(source), str(registration)],
        extra_cflags=["-O3", "-DNDEBUG", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-DNDEBUG", "-std=c++17"],
        with_cuda=True,
        is_python_module=False,
        build_directory=str(build_path),
        verbose=verbose,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-directory", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    library_path = load_native_cuda_ops(
        args.build_directory,
        verbose=args.verbose,
    )
    print(library_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
