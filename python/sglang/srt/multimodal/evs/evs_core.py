# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/evs.py

import torch


def compute_retained_tokens_count(
    tokens_per_frame: int, num_frames: int, q: float
) -> int:

    total_tokens = tokens_per_frame * num_frames
    evs_num_tokens = int(total_tokens * (1 - q))
    min_num_tokens = tokens_per_frame
    return max(min_num_tokens, evs_num_tokens)


def compute_retention_mask(
    video_embeds: torch.Tensor,
    video_size_thw: torch.LongTensor | tuple[int, int, int],
    spatial_merge_size: int,
    q: float,
) -> torch.Tensor:

    T, H, W = map(int, video_size_thw)


    video_embeds = video_embeds.reshape(
        T,
        H // spatial_merge_size,
        W // spatial_merge_size,
        video_embeds.size(-1),
    )
    tokens_per_frame = (H // spatial_merge_size) * (W // spatial_merge_size)

    similarity = torch.nn.functional.cosine_similarity(
        video_embeds[1:, ...], video_embeds[:-1, ...], dim=-1
    )
    dissimilarity = 1 - similarity


    dissimilarity = torch.cat(
        [255 * torch.ones_like(video_embeds[:1, :, :, 0]), dissimilarity], dim=0
    )

    dissimilarity_flat = dissimilarity.view(-1)
    order = torch.argsort(dissimilarity_flat, dim=-1, descending=True, stable=True)
    retain_num_tokens = compute_retained_tokens_count(
        tokens_per_frame=tokens_per_frame, num_frames=T, q=q
    )
    topk_indices = order[:retain_num_tokens]

    retention_mask = torch.zeros_like(dissimilarity_flat, dtype=torch.bool)
    retention_mask[topk_indices] = True
    retention_mask = retention_mask.reshape(dissimilarity.size())

    mask = retention_mask.view(-1)
    return mask


def tokens_per_frame(
    *,
    q: float,
    num_frames: int,
    frame_num_tokens: int,
) -> list[int]:

    retained = compute_retained_tokens_count(
        tokens_per_frame=frame_num_tokens, num_frames=num_frames, q=q
    )
    base = retained // num_frames
    rem = retained % num_frames
    tpf = [base] * (num_frames - 1) + [base + rem]
    assert sum(tpf) == retained
    return tpf


def replace_offsets_with_tokens_per_frame(
    *,
    pre_chunked_input_ids: list[int],
    num_tokens_per_frame: list[int],
    frame_offsets_inclusive: list[tuple[int, int]],
    filler_token_id: int,
) -> list[int]:

    assert isinstance(pre_chunked_input_ids, list)
    ids = pre_chunked_input_ids

    if len(frame_offsets_inclusive) == 1:
        """There might be no frame separators, in which case there will be one contiguous span of tokens"""
        final = ids[0 : frame_offsets_inclusive[0][0]]
        frames = [filler_token_id] * sum(num_tokens_per_frame)
        final.extend(frames)
    else:
        cursor = 0
        final = []
        for (start, end), num_tokens in zip(
            frame_offsets_inclusive, num_tokens_per_frame, strict=True
        ):
            final.extend(ids[cursor:start])
            final.extend([filler_token_id] * num_tokens)
            cursor = end + 1
    final.extend(ids[frame_offsets_inclusive[-1][1] + 1 :])
    return final
