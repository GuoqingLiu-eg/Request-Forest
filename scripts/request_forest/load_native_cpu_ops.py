#!/usr/bin/env python3
"""JIT-load the standalone Request Forest CPU operators."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


def load_native_cpu_ops(
    build_directory: str | Path,
    include_eagle: bool = False,
    verbose: bool = False,
) -> Any:
    """Compile and load Request Forest CPU ops into ``torch.ops.sgl_kernel``.

    The caller owns the build directory so tests and benchmarks can keep JIT
    artifacts outside the source tree.  EAGLE support adds the repository's
    existing CPU tree builder to the same shared library.
    """

    # ``cpp_extension`` locates Ninja through PATH.  An isolated interpreter can
    # have Ninja beside ``python`` without its bin directory being inherited by
    # the parent shell (for example when invoked through an absolute path).
    # Do not resolve the interpreter symlink: in a virtual environment the
    # sibling ``ninja`` executable lives beside the symlink, not beside the
    # base interpreter it points to.
    python_bin = str(Path(sys.executable).absolute().parent)
    inherited_path = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join(
        part for part in (python_bin, inherited_path) if part
    )

    from torch.utils.cpp_extension import load

    repo_root = Path(__file__).resolve().parents[2]
    cpu_source_dir = (
        repo_root / "python" / "sglang" / "kernels" / "aot" / "csrc" / "cpu"
    )
    sources = [
        cpu_source_dir / "request_forest.cpp",
        repo_root / "benchmark" / "request_forest" / "native_cpu_registration.cpp",
    ]
    extra_cflags = ["-O3", "-DNDEBUG", "-std=c++17"]
    extension_name = "request_forest_native_cpu"
    if include_eagle:
        sources.append(cpu_source_dir / "spec.cpp")
        extra_cflags.append("-DREQUEST_FOREST_WITH_EAGLE=1")
        extension_name += "_with_eagle"

    build_path = Path(build_directory).expanduser().resolve()
    build_path.mkdir(parents=True, exist_ok=True)
    return load(
        name=extension_name,
        sources=[str(source) for source in sources],
        extra_cflags=extra_cflags,
        extra_include_paths=[str(cpu_source_dir)],
        with_cuda=False,
        is_python_module=False,
        build_directory=str(build_path),
        verbose=verbose,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-directory", type=Path, required=True)
    parser.add_argument("--include-eagle", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    library_path = load_native_cpu_ops(
        args.build_directory,
        include_eagle=args.include_eagle,
        verbose=args.verbose,
    )
    print(library_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
