# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Mapping, TypeAlias

__all__ = [
    "CheckpointQuantSpec",
    "QuantMetadataSource",
    "resolve_checkpoint_quant_spec",
]


QuantMetadataSource: TypeAlias = Literal[
    "quantization_config",
    "text_config.quantization_config",
    "compression_config",
]


@dataclass(slots=True)
class CheckpointQuantSpec:


    declared_method: str | None
    config: dict[str, Any]
    source: QuantMetadataSource


def _get_field(config: object, name: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def _to_metadata_dict(value: object, source: QuantMetadataSource) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return deepcopy(dict(value))

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        metadata = to_dict()
        if isinstance(metadata, Mapping):
            return deepcopy(dict(metadata))

    raise TypeError(
        f"{source} must be a mapping or expose to_dict(), "
        f"got {type(value).__name__}"
    )


def _select_hf_quant_metadata(
    hf_config: object,
) -> tuple[QuantMetadataSource, object] | None:
    value = _get_field(hf_config, "quantization_config")
    if value is not None:
        return "quantization_config", value

    text_config = _get_field(hf_config, "text_config")
    value = _get_field(text_config, "quantization_config")
    if value is not None:
        return "text_config.quantization_config", value

    value = _get_field(hf_config, "compression_config")
    if value is not None:
        return "compression_config", value

    return None


def resolve_checkpoint_quant_spec(hf_config: object) -> CheckpointQuantSpec | None:


    selected = _select_hf_quant_metadata(hf_config)
    if selected is None:
        return None

    source, value = selected
    config = _to_metadata_dict(value, source)
    declared_method = config.get("quant_method")
    return CheckpointQuantSpec(
        declared_method=(declared_method if isinstance(declared_method, str) else None),
        config=config,
        source=source,
    )
