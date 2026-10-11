from __future__ import annotations

import logging
import os
import pathlib
import shlex
import subprocess
from typing import List

from sglang.kernels.jit.utils.compile import toolchain
from sglang.kernels.jit.utils.compile.spec import BuildSpec

logger = logging.getLogger(__name__)

_BUILD_FILE = "build.ninja"
_NINJA_TIMEOUT_S = 1800


def _escape(path: str) -> str:

    return path.replace("$", "$$").replace(":", "$:").replace(" ", "$ ")


def _arg(path: str) -> str:

    return shlex.quote(path).replace("$", "$$")


def _quote_path_flags(flags: List[str]) -> List[str]:

    quoted: List[str] = []
    for flag in flags:
        for prefix in ("-I", "-L"):
            if flag.startswith(prefix) and len(flag) > len(prefix):
                quoted.append(prefix + _arg(flag[len(prefix) :]))
                break
        else:
            quoted.append(flag)
    return quoted


def generate(spec: BuildSpec) -> str:

    units = spec.translation_units()
    with_device = any(unit.is_cuda for unit in units)

    host_cc, device_cc = toolchain.compilers()
    includes = toolchain.base_include_paths() + list(spec.include_paths)
    include_flags = [f"-I{path}" for path in includes]

    cxxflags = _quote_path_flags(
        toolchain.base_cxx_flags() + list(spec.cflags) + include_flags
    )
    cudaflags = _quote_path_flags(
        toolchain.base_cuda_flags()
        + toolchain.target_flags()
        + list(spec.cuda_cflags)
        + include_flags
    )
    ldflags = _quote_path_flags(
        toolchain.base_link_flags(with_device=with_device) + list(spec.ldflags)
    )

    lines = [
        "ninja_required_version = 1.3",
        f"cxx = {_arg(host_cc)}",
        f"nvcc = {_arg(device_cc)}",
        f"cxxflags = {' '.join(cxxflags)}",
        f"cudaflags = {' '.join(cudaflags)}",
        f"ldflags = {' '.join(ldflags)}",
        "",
        "rule compile_cxx",
        "  depfile = $out.d",
        '  command = $cxx -MD -MF "$out.d" $cxxflags -c "$in" -o "$out"',
        "",
        "rule compile_cuda",
        "  depfile = $out.d",
        '  command = $nvcc -MD -MF "$out.d" $cudaflags -c "$in" -o "$out"',
        "",
        "rule link",
        '  command = $cxx $in $ldflags -o "$out"',
        "",
    ]

    objects: List[str] = []
    for index, unit in enumerate(units):
        obj = f"{unit.stem}_{index}.o"
        rule = "compile_cuda" if unit.is_cuda else "compile_cxx"
        lines.append(f"build {obj}: {rule} {_escape(unit.filename)}")
        objects.append(obj)

    lines += [
        "",
        f"build {spec.module_name}.so: link {' '.join(objects)}",
        "",
        f"default {spec.module_name}.so",
        "",
    ]
    return "\n".join(lines)


def build(*, spec: BuildSpec, build_dir: pathlib.Path, build_file: str) -> pathlib.Path:

    build_dir.mkdir(parents=True, exist_ok=True)
    for unit in spec.translation_units():


        if unit.source is not None:
            (build_dir / unit.filename).write_text(unit.source)
    (build_dir / _BUILD_FILE).write_text(build_file)

    command = ["ninja", "-f", _BUILD_FILE]
    jobs = os.environ.get("MAX_JOBS")
    if jobs:
        command += ["-j", jobs]
    completed = subprocess.run(
        command,
        cwd=str(build_dir),
        capture_output=True,
        text=True,
        timeout=_NINJA_TIMEOUT_S,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Failed to build JIT module {spec.module_name} in {build_dir}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return build_dir / f"{spec.module_name}.so"


def scan_dependencies(build_dir: pathlib.Path) -> List[pathlib.Path]:

    paths: List[pathlib.Path] = []
    seen = set()
    for depfile in sorted(build_dir.glob("*.o.d")):
        try:
            text = depfile.read_text(errors="ignore")
        except OSError:
            continue
        for candidate in _parse_depfile(text):
            if candidate in seen:
                continue
            seen.add(candidate)
            paths.append(pathlib.Path(candidate))
    return paths


def _parse_depfile(text: str) -> List[str]:

    joined = text.replace("\\\r\n", " ").replace("\\\n", " ")
    tokens: List[str] = []
    for line in joined.split("\n"):
        _, separator, prerequisites = line.partition(":")
        if not separator:
            continue
        current: List[str] = []
        index = 0
        while index < len(prerequisites):
            char = prerequisites[index]
            if char == "\\" and index + 1 < len(prerequisites):
                if prerequisites[index + 1] in " \\":
                    current.append(prerequisites[index + 1])
                    index += 2
                    continue
            if char.isspace():
                if current:
                    tokens.append("".join(current))
                    current = []
                index += 1
                continue
            current.append(char)
            index += 1
        if current:
            tokens.append("".join(current))
    return tokens
