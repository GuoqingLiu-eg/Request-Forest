from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal, Optional

import yaml

from sglang.srt.debug_utils.comparator.utils import _StrictBase


class MetaOverrideRule(_StrictBase):


    match: str
    dims: str
    side: Literal["both", "baseline", "target"] = "both"


class MetaOverrideConfig(_StrictBase):


    overrides: list[MetaOverrideRule] = []


class MetaOverrider:


    def __init__(self, rules: list[MetaOverrideRule]) -> None:
        self._rules: list[MetaOverrideRule] = rules

    @property
    def is_empty(self) -> bool:
        return len(self._rules) == 0

    @classmethod
    def from_args_and_config(
        cls,
        *,
        override_dims: list[str],
        override_baseline_dims: list[str],
        override_target_dims: list[str],
        override_config: Optional[Path],
    ) -> MetaOverrider:
        per_side_args: list[tuple[list[str], Literal["both", "baseline", "target"]]] = [
            (override_dims, "both"),
            (override_baseline_dims, "baseline"),
            (override_target_dims, "target"),
        ]
        cli_rules: list[MetaOverrideRule] = [
            MetaOverrideRule(match=name, dims=dims_str, side=side)
            for raw_args, side in per_side_args
            for name, dims_str in [_parse_cli_override_arg(raw) for raw in raw_args]
        ]

        yaml_rules: list[MetaOverrideRule] = (
            _load_yaml_rules(override_config) if override_config is not None else []
        )

        return cls(rules=cli_rules + yaml_rules)

    def apply_to_meta(
        self,
        *,
        name: str,
        meta: dict[str, Any],
        side: Literal["baseline", "target"],
    ) -> dict[str, Any]:

        for rule in self._rules:
            if rule.side not in ("both", side):
                continue
            if re.search(rule.match, name):
                return {**meta, "dims": rule.dims}

        return meta


def _parse_cli_override_arg(raw: str) -> tuple[str, str]:

    parts: list[str] = raw.split(":", maxsplit=1)
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        raise ValueError(
            f"Invalid override format: {raw!r}; expected 'name:dims_string'"
        )
    return parts[0].strip(), parts[1].strip()


def _load_yaml_rules(path: Path) -> list[MetaOverrideRule]:

    with open(path) as f:
        raw_data: Any = yaml.safe_load(f)

    if raw_data is None:
        return []

    config: MetaOverrideConfig = MetaOverrideConfig.model_validate(raw_data)
    return config.overrides
