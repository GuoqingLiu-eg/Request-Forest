from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Optional, Protocol

import torch

from sglang.srt.multimodal.kimi_k3_image_processing import (
    KimiK3DeferredPreprocessing,
)


class KimiK3MediaProcessorConfigProvider(Protocol):


    media_proc_cfg: Mapping[str, Any]


@dataclass(frozen=True)
class KimiK3PreprocessConfig:


    patch_size: int
    merge_kernel_size: int
    in_patch_limit: int
    patch_limit_on_one_side: int
    fixed_output_tokens: Optional[int]
    image_mean: tuple[float, ...]
    image_std: tuple[float, ...]
    transparent_bg_config: Optional[dict]

    @classmethod
    def from_media_processor(
        cls, media_processor: KimiK3MediaProcessorConfigProvider
    ) -> KimiK3PreprocessConfig:
        config = media_processor.media_proc_cfg
        return cls(
            patch_size=int(config["patch_size"]),
            merge_kernel_size=int(config["merge_kernel_size"]),
            in_patch_limit=int(config["in_patch_limit"]),
            patch_limit_on_one_side=int(config["patch_limit_on_one_side"]),
            fixed_output_tokens=(
                None
                if config.get("fixed_output_tokens") is None
                else int(config["fixed_output_tokens"])
            ),
            image_mean=tuple(float(value) for value in config["image_mean"]),
            image_std=tuple(float(value) for value in config["image_std"]),
            transparent_bg_config=deepcopy(config.get("transparent_bg_config")),
        )


@dataclass(frozen=True)
class KimiK3ResizeConfig:
    num_tokens: int
    new_width: int
    new_height: int
    pad_width: int
    pad_height: int

    @classmethod
    def from_dict(cls, value: dict) -> KimiK3ResizeConfig:
        return cls(
            num_tokens=int(value["num_tokens"]),
            new_width=int(value["new_width"]),
            new_height=int(value["new_height"]),
            pad_width=int(value["pad_width"]),
            pad_height=int(value["pad_height"]),
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "num_tokens": self.num_tokens,
            "new_width": self.new_width,
            "new_height": self.new_height,
            "pad_width": self.pad_width,
            "pad_height": self.pad_height,
        }


@dataclass(frozen=True)
class KimiK3ImagePreprocessArtifact:


    content_digest: str
    artifact_key: str
    feature_hash: int
    original_size: tuple[int, int]
    resize_config: KimiK3ResizeConfig
    grid_thw: tuple[int, int, int]
    feature: Optional[torch.Tensor]
    deferred: Optional[KimiK3DeferredPreprocessing] = None

    @property
    def has_feature(self) -> bool:
        return self.feature is not None

    def cache_value(self) -> KimiK3ImagePreprocessArtifact:

        if self.feature is None or self.feature.device.type == "cpu":
            return self
        return replace(self, feature=None)

    def cache_size_items(self) -> tuple:

        deferred = None
        if self.deferred is not None:
            deferred = (
                self.deferred.backend,
                self.deferred.image_mean,
                self.deferred.image_std,
                self.deferred.transparent_bg_config,
                self.deferred.resize_config,
            )
        return (
            self.content_digest,
            self.artifact_key,
            self.feature_hash,
            self.original_size,
            (
                self.resize_config.num_tokens,
                self.resize_config.new_width,
                self.resize_config.new_height,
                self.resize_config.pad_width,
                self.resize_config.pad_height,
            ),
            self.grid_thw,
            self.feature,
            deferred,
        )
