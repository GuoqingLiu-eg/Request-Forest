import __future__

import importlib
import inspect
import textwrap
import types
from collections.abc import Callable
from typing import Any, Optional

import yaml

from sglang.srt.debug_utils.source_patcher.source_editor import apply_edits
from sglang.srt.debug_utils.source_patcher.types import (
    EditSpec,
    PatchConfig,
    PatchSpec,
    PatchState,
)


def apply_patches_from_config(
    yaml_content: str,
    *,
    extra_imports: Optional[list[str]] = None,
) -> list[PatchState]:

    raw: dict[str, Any] = yaml.safe_load(yaml_content)
    config: PatchConfig = PatchConfig(**raw)

    if extra_imports:
        config = _inject_preamble(config=config, extra_imports=extra_imports)

    return _apply_specs(config.patches)


class CodePatcher:


    def __init__(self, *, patches: list[PatchSpec]) -> None:
        self._patches = patches
        self._states: list[PatchState] = []

    def __enter__(self) -> "CodePatcher":
        self._states = _apply_specs(self._patches)
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        for state in reversed(self._states):
            state.restore()
        self._states.clear()


def patch_function(
    *,
    target: Callable[..., Any],
    edits: list[EditSpec],
    preamble: str = "",
) -> PatchState:

    original_code: types.CodeType = target.__code__

    source: str = inspect.getsource(target)
    modified_source: str = apply_edits(source=source, edits=edits)
    modified_source = textwrap.dedent(modified_source)

    if preamble.strip():
        modified_source = _insert_preamble(source=modified_source, preamble=preamble)

    code: types.CodeType = compile(
        modified_source,
        inspect.getfile(target),
        "exec",
        flags=__future__.annotations.compiler_flag,
    )
    temp_namespace: dict[str, Any] = {}
    exec(code, target.__globals__, temp_namespace)

    new_fn: Any = temp_namespace[target.__name__]
    target.__code__ = new_fn.__code__

    return PatchState(target_fn=target, original_code=original_code)


def _apply_specs(specs: list[PatchSpec]) -> list[PatchState]:
    states: list[PatchState] = []
    for spec in specs:
        target_fn: Callable[..., Any] = _resolve_target(spec.target)
        print(f"[source_patcher] patching {spec.target}")
        state: PatchState = patch_function(
            target=target_fn, edits=spec.edits, preamble=spec.preamble
        )
        states.append(state)
    return states


def _inject_preamble(*, config: PatchConfig, extra_imports: list[str]) -> PatchConfig:

    import_block: str = "\n".join(extra_imports)
    new_patches: list[PatchSpec] = []

    for spec in config.patches:
        existing: str = spec.preamble
        combined: str = (
            import_block + "\n" + existing if existing.strip() else import_block
        )
        new_patches.append(
            PatchSpec(target=spec.target, edits=spec.edits, preamble=combined)
        )

    return PatchConfig(patches=new_patches)


def _insert_preamble(*, source: str, preamble: str) -> str:

    lines: list[str] = source.splitlines()

    signature_end: int = _find_signature_end(lines)

    body_start: int = signature_end + 1
    body_indent: str = ""
    for i in range(body_start, len(lines)):
        if lines[i].strip():
            body_indent = " " * (len(lines[i]) - len(lines[i].lstrip()))
            body_start = i
            break

    preamble_lines: list[str] = [
        body_indent + pl for pl in preamble.strip().splitlines()
    ]
    return "\n".join(lines[:body_start] + preamble_lines + lines[body_start:])


def _find_signature_end(lines: list[str]) -> int:

    for i, line in enumerate(lines):
        if line.rstrip().endswith(":"):
            return i
    return 0


def _resolve_target(qualified_name: str) -> Callable[..., Any]:

    parts: list[str] = qualified_name.split(".")

    target: Any = None
    for split_idx in range(len(parts), 0, -1):
        module_path: str = ".".join(parts[:split_idx])
        try:
            target = importlib.import_module(module_path)
            attr_parts: list[str] = parts[split_idx:]
            break
        except ImportError:
            continue
    else:
        raise ImportError(f"could not import any module prefix of '{qualified_name}'")

    for attr_name in attr_parts:
        target = getattr(target, attr_name)

    if isinstance(target, classmethod):
        target = target.__func__
    if not callable(target):
        raise TypeError(
            f"resolved target '{qualified_name}' is not callable: {type(target)}"
        )

    return target
