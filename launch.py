#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


PROFILES = ("qwen3.6", "qwen3.8", "ornith")


def load_config(root: Path, name: str) -> dict:
    with (root / "configs" / f"{name}.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def server_argv(options: dict) -> list[str]:
    result = []
    for name, value in options.items():
        if value is False or value is None:
            continue
        result.append("--" + name)
        if value is True:
            continue
        if isinstance(value, list):
            result.extend(str(item) for item in value)
        elif isinstance(value, dict):
            result.append(json.dumps(value, separators=(",", ":")))
        else:
            result.append(str(value))
    return result


def last_option(argv: list[str], option: str) -> str:
    value = None
    for index, token in enumerate(argv):
        if token == option:
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                raise ValueError(f"{option} requires a value")
            value = argv[index + 1]
        elif token.startswith(option + "="):
            value = token.split("=", 1)[1]
    if not value:
        raise ValueError(f"{option} requires a value")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the shared Request Forest runtime with a model profile.",
        epilog="Append -- followed by SGLang options to override server arguments. Repeating a list option replaces the whole list.",
    )
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--draft-model-path", required=True)
    parser.add_argument("--gpus")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--native-dir")
    parser.add_argument("--dry-run", action="store_true")
    supplied = sys.argv[1:]
    if "--" in supplied:
        boundary = supplied.index("--")
        args = parser.parse_args(supplied[:boundary])
        extra = supplied[boundary + 1:]
    else:
        args = parser.parse_args(supplied)
        extra = []
    if extra and not extra[0].startswith("--"):
        parser.error("Additional SGLang arguments must start with an option")
    if args.gpus is not None and not args.gpus.strip():
        parser.error("--gpus must not be empty")
    root = Path(__file__).resolve().parent
    common = load_config(root, "common")
    profile = load_config(root, args.profile)
    options = dict(common["server_args"])
    options["model-path"] = str(Path(args.model_path).expanduser().resolve())
    options["speculative-draft-model-path"] = str(Path(args.draft_model_path).expanduser().resolve())
    if args.host is not None:
        options["host"] = args.host
    if args.port is not None:
        options["port"] = args.port
    native_root = Path(args.native_dir).expanduser().resolve() if args.native_dir else root / "native"
    libraries = {
        "SGLANG_REQUEST_FOREST_CPU_OPS_LIBRARY": str(native_root / "request_forest_native_cpu.so"),
        "SGLANG_REQUEST_FOREST_CUDA_OPS_LIBRARY": str(native_root / "request_forest_native_cuda.so"),
    }
    overrides = dict(common["environment"])
    overrides.update(profile["environment"])
    overrides.update(libraries)
    python_paths = [str(root / "python")]
    if (root / "build" / "deps" / "flashinfer" / "__init__.py").is_file():
        python_paths.append(str(root / "build" / "deps"))
    overrides["PYTHONPATH"] = os.pathsep.join(python_paths)
    overrides["CUDA_VISIBLE_DEVICES"] = args.gpus if args.gpus is not None else os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    overrides["SGLANG_CACHE_DIR"] = str(root / "build" / "cache")
    overrides["FLASHINFER_CUDA_ARCH_LIST"] = os.environ.get("FLASHINFER_CUDA_ARCH_LIST") or "12.0f"
    environment = os.environ.copy()
    environment.update(overrides)
    command = [sys.executable, "-m", "sglang.launch_server", *server_argv(options), *extra]
    if args.dry_run:
        print(json.dumps({"profile": args.profile, "model_id": profile["model_id"], "command": command, "environment": overrides}, indent=2))
        return
    try:
        for option in ("--model-path", "--speculative-draft-model-path"):
            path = Path(last_option(command[3:], option)).expanduser().resolve()
            if not path.is_dir():
                parser.error(f"{option} must name an existing model directory: {path}")
    except ValueError as error:
        parser.error(str(error))
    for name, library in libraries.items():
        if not Path(library).is_file():
            parser.error(f"{name} does not exist: {library}; check the bundled native directory or --native-dir")
    Path(overrides["SGLANG_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    os.execvpe(sys.executable, command, environment)


if __name__ == "__main__":
    main()
