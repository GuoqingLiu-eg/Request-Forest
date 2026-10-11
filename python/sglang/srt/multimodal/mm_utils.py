# Copyright 2023-2024 SGLang Team
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


import ast
import itertools
import math
import re
from io import BytesIO
from typing import Callable, Literal, Optional, Sequence

import numpy as np
import pybase64
import torch
from PIL import Image

from sglang.srt.distributed.communication_op import tensor_model_parallel_all_gather
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import flatten_nested_list


def ensure_numpy(x):

    return x.numpy() if isinstance(x, torch.Tensor) else x


def has_valid_data(data) -> bool:
    if data is None:
        return False
    if isinstance(data, list):
        return any(has_valid_data(item) for item in flatten_nested_list(data))
    return True


def materialize_multimodal_features(
    features: Sequence[torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:


    if not features:
        raise ValueError("features must contain at least one tensor")

    first = features[0]
    if not isinstance(first, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(first)}")
    if first.ndim == 0:
        raise ValueError("multimodal feature tensors must have a leading dimension")
    trailing_shape = first.shape[1:]
    total_tokens = 0
    for feature in features:
        if not isinstance(feature, torch.Tensor):
            raise TypeError(f"expected torch.Tensor, got {type(feature)}")
        if feature.ndim == 0 or feature.shape[1:] != trailing_shape:
            raise ValueError(
                "multimodal feature tensors must have matching trailing shapes: "
                f"expected {trailing_shape}, got {feature.shape}"
            )
        total_tokens += feature.shape[0]

    output = torch.empty(
        (total_tokens, *trailing_shape),
        device=device,
        dtype=dtype,
    )
    offset = 0
    for feature in features:
        length = feature.shape[0]
        output[offset : offset + length].copy_(feature, non_blocking=True)
        offset += length
    return output


def select_best_resolution(original_size, possible_resolutions):

    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_resolutions:

        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(
            original_height * scale
        )


        effective_resolution = min(
            downscaled_width * downscaled_height, original_width * original_height
        )
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution
            and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def resize_and_pad_image(image, target_resolution):

    original_width, original_height = image.size
    target_width, target_height = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)


    resized_image = image.resize((new_width, new_height))

    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def divide_to_patches(image, patch_size):

    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):

    if isinstance(grid_pinpoints, str) and "x" in grid_pinpoints:
        assert patch_size in [
            224,
            336,
            384,
            448,
            512,
        ], "patch_size should be in [224, 336, 384, 448, 512]"

        matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))

        grid_pinpoints = [
            (i, j)
            for i in range(range_start[0], range_end[0] + 1)
            for j in range(range_start[1], range_end[1] + 1)
        ]

        grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size


def process_anyres_image(image, processor, grid_pinpoints):

    if isinstance(grid_pinpoints, str) and "x" in grid_pinpoints:
        try:
            patch_size = processor.size[0]
        except Exception:
            patch_size = processor.size["shortest_edge"]
        assert patch_size in [
            224,
            336,
            384,
            448,
            512,
        ], "patch_size should be in [224, 336, 384, 448, 512]"

        matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))

        grid_pinpoints = [
            (i, j)
            for i in range(range_start[0], range_end[0] + 1)
            for j in range(range_start[1], range_end[1] + 1)
        ]

        grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]

    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)


    crop_size = (
        processor.crop_size["height"]
        if getattr(processor, "crop_size", None) is not None
        else processor.size["height"]
    )
    shortest_edge = (
        processor.size["shortest_edge"]
        if "shortest_edge" in processor.size
        else processor.size["height"]
    )
    patches = divide_to_patches(image_padded, crop_size)

    image_original_resize = image.resize((shortest_edge, shortest_edge))

    image_patches = [image_original_resize] + patches
    image_patches = [
        processor.preprocess(image_patch.convert("RGB"))["pixel_values"][0]
        for image_patch in image_patches
    ]

    image_patches = [ensure_numpy(p) for p in image_patches]
    return np.stack(image_patches, axis=0)


def load_image_from_base64(image):
    return Image.open(BytesIO(pybase64.b64decode(image, validate=True)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    if pil_img.mode == "L":
        pil_img = pil_img.convert("RGB")
    if width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def unpad_image(tensor, original_size):

    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


def unpad_image_shape(current_height, current_width, original_size):

    original_width, original_height = original_size

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        new_shape = (current_height - 2 * padding, current_width)
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        new_shape = (current_height, current_width - 2 * padding)

    return new_shape


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == "pad":
        for image in images:
            image = expand2square(
                image, tuple(int(x * 255) for x in image_processor.image_mean)
            )
            image = image_processor.preprocess(image)["pixel_values"][0]
            new_images.append(image)
    elif "anyres" in image_aspect_ratio:
        for image in images:
            image = process_anyres_image(
                image, image_processor, model_cfg.image_grid_pinpoints
            )
            new_images.append(image)
    else:
        return image_processor(images)["pixel_values"]
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = np.stack(new_images, axis=0)
    return new_images


def get_dp_encoder_lb_assignment(
    sizes: list[int],
    num_gpus: int = 2,
) -> tuple[list[int], list[int], list[int]]:


    n_samples = len(sizes)


    if n_samples == 0:
        return [], [0] * num_gpus, [0] * num_gpus


    gpu_assignments = [list[int]() for _ in range(num_gpus)]
    gpu_loads = [0] * num_gpus


    large_to_small_indices = sorted(
        range(n_samples), key=lambda i: sizes[i], reverse=True
    )

    for idx in large_to_small_indices:

        min_gpu = min(range(num_gpus), key=lambda i: gpu_loads[i])
        gpu_assignments[min_gpu].append(idx)
        gpu_loads[min_gpu] += sizes[idx]


    shuffle_indices = list[int]()
    gpu_sample_counts = list[int]()
    for gpu_id in range(num_gpus):


        shuffle_indices.extend(gpu_assignments[gpu_id])


        gpu_sample_counts.append(len(gpu_assignments[gpu_id]))

    return (shuffle_indices, gpu_sample_counts, gpu_loads)


def _pad_mrope_vision_embeddings_for_tp_gather(
    image_embeds_local: torch.Tensor, max_len_per_rank: int
) -> torch.Tensor:


    current_len = image_embeds_local.shape[0]
    if current_len >= max_len_per_rank:
        return image_embeds_local

    padded = torch.empty(
        (max_len_per_rank, *image_embeds_local.shape[1:]),
        dtype=image_embeds_local.dtype,
        device=image_embeds_local.device,
    )
    if current_len > 0:
        padded[:current_len].copy_(image_embeds_local)
    return padded


def run_dp_sharded_vision_model(
    image_input: torch.Tensor, vision_model: torch.nn.Module
) -> torch.Tensor:


    num_chunks = image_input.shape[0]
    mp_world_size = get_parallel().tp_size
    num_chunks_per_rank = (num_chunks + mp_world_size - 1) // mp_world_size
    num_padded_chunks = num_chunks_per_rank * mp_world_size - num_chunks
    pad = (0,) * (2 * (image_input.dim() - 1)) + (0, num_padded_chunks)
    image_input_padded = torch.nn.functional.pad(image_input, pad)
    rank = get_parallel().tp_rank
    image_input_per_rank = image_input_padded[
        rank * num_chunks_per_rank : (rank + 1) * num_chunks_per_rank, ...
    ]

    vision_embeddings = vision_model(image_input_per_rank)

    vision_embeddings = vision_embeddings.last_hidden_state.contiguous()
    vision_embeddings = tensor_model_parallel_all_gather(vision_embeddings, dim=0)
    vision_embeddings = vision_embeddings[:num_chunks, ...]
    return vision_embeddings


def concat_or_single(tensors: Sequence[torch.Tensor], dim: int = 0) -> torch.Tensor:


    return tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=dim)


def run_dp_sharded_mrope_vision_model(
    vision_model: torch.nn.Module,
    pixel_values: Optional[torch.Tensor],
    grid_thw_list: list,
    *,
    rope_type: Literal["rope_3d", "rope_2d", "rope_2d_packed"],
    pool_temporal_dimension: bool = False,
    load_local_pixel_values: Optional[Callable[[list[int]], torch.Tensor]] = None,
    pixel_values_device: Optional[torch.device] = None,
    pixel_values_dtype: Optional[torch.dtype] = None,
    pass_grid_thw_list: bool = False,
):

    if pixel_values is None and load_local_pixel_values is None:
        raise ValueError("pixel_values or load_local_pixel_values must be provided")

    input_device = (
        pixel_values.device if pixel_values is not None else pixel_values_device
    )
    input_dtype = pixel_values.dtype if pixel_values is not None else pixel_values_dtype
    if input_device is None or input_dtype is None:
        raise ValueError(
            "pixel_values_device and pixel_values_dtype are required with a local loader"
        )

    tp_size = get_parallel().attn_tp_size
    if tp_size == 1:
        if pixel_values is None:
            pixel_values = load_local_pixel_values(list(range(len(grid_thw_list))))
        grid_thw = torch.tensor(
            grid_thw_list,


            device=pixel_values.device if rope_type == "rope_2d" else None,
        )
        if rope_type == "rope_2d":
            kwargs = {
                "grid_hw": grid_thw,
                "max_seqlen": max(math.prod(grid) for grid in grid_thw_list),
            }
            if pass_grid_thw_list:
                kwargs["grid_thw_list"] = grid_thw_list
            image_embeds = vision_model(pixel_values, **kwargs)


            if isinstance(image_embeds, list):
                return concat_or_single(image_embeds, dim=0)
            return image_embeds
        if rope_type == "rope_2d_packed":
            image_embeds = vision_model(pixel_values, grid_thw)
            if isinstance(image_embeds, list):
                return concat_or_single(image_embeds, dim=0)
            return image_embeds
        return vision_model(pixel_values, grid_thw=grid_thw)


    tp_rank_local = get_parallel().attn_tp_rank


    patches_per_image = [math.prod(grid_thw) for grid_thw in grid_thw_list]


    cum_patches_per_image = [0, *itertools.accumulate(patches_per_image)]


    image_to_tp_rank, gpu_sample_counts, _ = get_dp_encoder_lb_assignment(
        patches_per_image, tp_size
    )


    cum_gpu_sample_counts = [0, *itertools.accumulate(gpu_sample_counts)]


    image_idxs_local = image_to_tp_rank[
        cum_gpu_sample_counts[tp_rank_local] : cum_gpu_sample_counts[tp_rank_local + 1]
    ]


    if len(image_idxs_local) > 0:
        if load_local_pixel_values is not None:
            pixel_values_local = load_local_pixel_values(image_idxs_local)
        else:
            assert pixel_values is not None
            pixel_values_local = torch.cat(
                [
                    pixel_values[
                        cum_patches_per_image[i] : cum_patches_per_image[i + 1]
                    ]
                    for i in image_idxs_local
                ]
            )
    else:
        pixel_values_local = None

    packed_2d_rope = rope_type in ("rope_2d", "rope_2d_packed")
    if packed_2d_rope:
        embed_dim_reduction_factor = (
            vision_model.merge_kernel_size[0] * vision_model.merge_kernel_size[1]
        )
    else:
        embed_dim_reduction_factor = (
            vision_model.spatial_merge_size * vision_model.spatial_merge_size
        )

    output_tokens_per_image = [
        math.prod(grid[1:] if pool_temporal_dimension else grid)
        // embed_dim_reduction_factor
        for grid in grid_thw_list
    ]
    grouped_output_lengths = []
    assignment_offset = 0
    for sample_count in gpu_sample_counts:
        rank_images = image_to_tp_rank[
            assignment_offset : assignment_offset + sample_count
        ]
        grouped_output_lengths.append(
            sum(output_tokens_per_image[i] for i in rank_images)
        )
        assignment_offset += sample_count


    max_len_per_rank = max(grouped_output_lengths)
    local_grid_thw_list = [grid_thw_list[i] for i in image_idxs_local]


    if packed_2d_rope:
        if pixel_values_local is not None and pixel_values_local.shape[0] > 0:


            local_grid_thw = torch.tensor(
                local_grid_thw_list,
                device=(pixel_values_local.device if rope_type == "rope_2d" else None),
            )
            if rope_type == "rope_2d":
                kwargs = {
                    "grid_hw": local_grid_thw,
                    "max_seqlen": max(math.prod(grid) for grid in local_grid_thw_list),
                }
                if pass_grid_thw_list:
                    kwargs["grid_thw_list"] = local_grid_thw_list
                image_embeds_local = vision_model(pixel_values_local, **kwargs)
            else:
                image_embeds_local = vision_model(pixel_values_local, local_grid_thw)
            if isinstance(image_embeds_local, list):
                image_embeds_local = concat_or_single(image_embeds_local, dim=0)
        else:
            out_dim = getattr(vision_model.config, "hidden_size", None)
            image_embeds_local = torch.empty(
                (0, embed_dim_reduction_factor, out_dim),
                device=vision_model.device,
                dtype=input_dtype,
            )
    else:
        if pixel_values_local is not None and pixel_values_local.shape[0] > 0:

            image_embeds_local = vision_model(
                pixel_values_local, torch.tensor(local_grid_thw_list)
            )
            if isinstance(image_embeds_local, list):
                image_embeds_local = concat_or_single(image_embeds_local, dim=0)
        else:

            out_dim = getattr(vision_model, "out_hidden_size", None)
            if out_dim is None:
                out_dim = vision_model.config.hidden_size
            image_embeds_local = torch.empty(
                (0, out_dim),
                device=vision_model.device,
                dtype=input_dtype,
            )


    if len(grid_thw_list) == 1:
        owner_local = image_to_tp_rank[0]
        n_tok = output_tokens_per_image[0]
        if tp_rank_local == owner_local:
            out_embeddings = image_embeds_local.contiguous()
        else:
            out_embeddings = torch.empty(
                (n_tok, *image_embeds_local.shape[1:]),
                dtype=input_dtype,
                device=input_device,
            )
        get_parallel().attn_tp_group.broadcast(out_embeddings, src=owner_local)
        return out_embeddings


    image_embeds_local_padded = _pad_mrope_vision_embeddings_for_tp_gather(
        image_embeds_local, max_len_per_rank
    )


    gathered_embeds = get_parallel().attn_tp_group.all_gather(
        image_embeds_local_padded, dim=0
    )


    rank_embeddings = list[torch.Tensor]()
    for rank in range(tp_size):
        start_idx = rank * max_len_per_rank
        end_idx = start_idx + grouped_output_lengths[rank]
        rank_embeddings.append(gathered_embeds[start_idx:end_idx])


    original_order_embeddings = [None] * len(grid_thw_list)
    current_idx = 0
    for rank in range(tp_size):
        count = gpu_sample_counts[rank]
        if count > 0:


            rank_images = image_to_tp_rank[current_idx : current_idx + count]

            rank_embed = rank_embeddings[rank]

            embed_start = 0
            for img_idx in rank_images:
                img_patches = output_tokens_per_image[img_idx]
                original_order_embeddings[img_idx] = rank_embed[
                    embed_start : embed_start + img_patches
                ]
                embed_start += img_patches
            current_idx += count
    out_embeddings = torch.cat(original_order_embeddings, dim=0)
    return out_embeddings
