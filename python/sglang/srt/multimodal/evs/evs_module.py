# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from transformers import PretrainedConfig

from sglang.srt.managers.schedule_batch import MultimodalDataItem
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.utils import logger

from .evs_core import compute_retention_mask, replace_offsets_with_tokens_per_frame


@dataclass(kw_only=True)
class EVSEmbeddingResult(EmbeddingResult):


    num_tokens_per_frame: list[int]

    def redistribute_pruned_frames_placeholders(
        self,
        input_ids: torch.Tensor,
        offsets: list[tuple[int, int]],
        *,
        item: MultimodalDataItem,
        extend_prefix_len: int,
        extend_seq_len: int,
    ) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        assert len(input_ids) == extend_seq_len
        assert item.is_video() and "pre_chunked_input_ids" in item.model_specific_data
        pre_chunked_input_ids = item.pre_chunked_input_ids
        filler_token_id = item.pad_value
        input_ids_list = replace_offsets_with_tokens_per_frame(
            pre_chunked_input_ids=pre_chunked_input_ids,
            num_tokens_per_frame=self.num_tokens_per_frame,
            frame_offsets_inclusive=offsets,
            filler_token_id=filler_token_id,
        )
        input_ids = torch.tensor(
            input_ids_list, dtype=input_ids.dtype, device=input_ids.device
        )
        offsets = BaseMultimodalProcessor.get_mm_items_offset(
            input_ids, filler_token_id
        )
        input_ids = input_ids[extend_prefix_len : extend_prefix_len + extend_seq_len]
        assert (
            len(input_ids) == extend_seq_len
        ), f"Input ids length changed after redistribution, got {len(input_ids)} != {extend_seq_len}"
        return input_ids, offsets


@dataclass(frozen=True, kw_only=True)
class EVSConfig:
    video_pruning_rate: float
    spatial_merge_size: int = 1

    def __post_init__(self):
        assert (
            self.video_pruning_rate >= 0.0 and self.video_pruning_rate < 1.0
        ), f"Video pruning rate must be between 0.0 and 1.0, got {self.video_pruning_rate=}"


class EVS(torch.nn.Module, ABC):


    @staticmethod
    @abstractmethod
    def create_evs_config(config: PretrainedConfig) -> EVSConfig:

        raise NotImplementedError

    @abstractmethod
    def get_video_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:

        raise NotImplementedError

    def __init__(
        self,
        config: PretrainedConfig,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> None:
        super().__init__()
        model_name = self.__class__.__name__
        self.original_get_video_feature = self.get_video_feature
        self.evs_config = self.create_evs_config(config)
        self.evs_enabled = self.evs_config.video_pruning_rate > 0.0
        if self.evs_enabled:
            logger.info(f"[EVS] enabled for {model_name} [{self.evs_config}]")
            self.get_video_feature = self.evs_video
        else:
            logger.info(
                f"[EVS] requested on model {model_name} but is disabled for pruning_rate == 0.0."
            )

    def evs_video(self, items: list[MultimodalDataItem]) -> EVSEmbeddingResult:

        logger.debug(
            f"[EVS] beginning for model {self.__class__.__name__} [evs_config={self.evs_config=}]"
        )
        assert len(items) == 1, f"Expected 1 item, got {len(items)}"
        item = items[0]
        assert item.is_video() and "thw_grids" in item.model_specific_data

        q = self.evs_config.video_pruning_rate
        merge = self.evs_config.spatial_merge_size
        videos_features = self.original_get_video_feature([item])
        if videos_features.ndim == 3:
            videos_features = videos_features.flatten(0, 1)
        assert videos_features.ndim == 2, videos_features.ndim

        final_embeddings: list[torch.Tensor] = []
        num_tokens_per_frame: list[int] = []

        sizes = [(t * h * w // merge**2) for t, h, w in item.thw_grids]
        for single_video, video_size_thw in zip(
            videos_features.split(sizes),
            item.thw_grids,
            strict=True,
        ):
            retention_mask = compute_retention_mask(
                single_video,
                video_size_thw=video_size_thw,
                spatial_merge_size=merge,
                q=q,
            )
            preserved = single_video[retention_mask]
            final_embeddings.append(preserved)
            num_frames = video_size_thw[0]
            tokens_per_frame = (
                retention_mask.reshape(num_frames, -1).sum(dim=-1).tolist()
            )
            num_tokens_per_frame.extend(tokens_per_frame)
        final_embeddings_tensor = torch.cat(final_embeddings)
        return EVSEmbeddingResult(
            embedding=final_embeddings_tensor,
            num_tokens_per_frame=num_tokens_per_frame,
        )
