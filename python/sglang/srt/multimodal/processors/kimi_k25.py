import math
import re
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sglang.kernels.ops.mm.process import normalize_and_patchify
from sglang.srt.managers.schedule_batch import (
    MultimodalProcessorOutput,
)
from sglang.srt.models.kimi_k25 import KimiK25ForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import (
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.kimi_common import KimiGridMMDataMixin
from sglang.srt.multimodal.transport.cuda_ipc import (
    DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY,
)


def navit_resize_config(
    width: int,
    height: int,
    patch_size: int,
    merge_kernel_size: int,
    in_patch_limit: int,
    patch_limit_on_one_side: int,
    fixed_output_tokens: int | None = None,
) -> dict:

    s1 = math.sqrt(
        in_patch_limit
        / (max(1.0, width // patch_size) * max(1.0, height // patch_size))
    )
    s2 = patch_limit_on_one_side * patch_size / width
    s3 = patch_limit_on_one_side * patch_size / height
    scale = min(1.0, s1, s2, s3)
    new_w = min(max(1, int(width * scale)), patch_limit_on_one_side * patch_size)
    new_h = min(max(1, int(height * scale)), patch_limit_on_one_side * patch_size)

    factor = merge_kernel_size * patch_size
    pad_height = (factor - new_h % factor) % factor
    pad_width = (factor - new_w % factor) % factor

    if fixed_output_tokens is not None:
        num_tokens = fixed_output_tokens
    else:
        token_height = (new_h + pad_height) // factor
        token_width = (new_w + pad_width) // factor
        num_tokens = token_height * token_width

    return {
        "num_tokens": num_tokens,
        "new_width": new_w,
        "new_height": new_h,
        "pad_width": pad_width,
        "pad_height": pad_height,
    }


def _get_image_dimensions(image: Union[torch.Tensor, Image.Image]) -> tuple[int, int]:

    if isinstance(image, torch.Tensor):

        return image.shape[2], image.shape[1]
    return image.size


def _expand_image_token_ids(
    input_ids: Union[List[int], torch.Tensor],
    image_token_id: int,
    image_token_counts: List[int],
) -> torch.Tensor:

    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.detach().flatten().cpu().numpy()
    input_ids = np.asarray(input_ids, dtype=np.int64)

    placeholder_mask = input_ids == image_token_id
    placeholder_count = np.count_nonzero(placeholder_mask)
    if placeholder_count != len(image_token_counts):
        raise ValueError(
            f"Expected {len(image_token_counts)} image placeholder token(s), "
            f"found {placeholder_count}."
        )

    repeats = np.ones(input_ids.shape, dtype=np.int64)
    repeats[placeholder_mask] = image_token_counts
    return torch.from_numpy(np.repeat(input_ids, repeats)).unsqueeze(0)


def _pil_to_cuda_chw(image: Image.Image) -> torch.Tensor:

    arr = np.asarray(image.convert("RGB"))
    return torch.from_numpy(arr).permute(2, 0, 1).cuda()


def _ensure_chw_rgb(image: torch.Tensor) -> torch.Tensor:

    if image.dtype != torch.uint8:


        raise ValueError(
            f"Kimi GPU preprocessing expects raw uint8 pixels, got {image.dtype}"
        )
    image = image.cuda()
    if image.dim() == 2:
        image = image.unsqueeze(0)
    c = image.shape[0]
    if c == 3:
        return image
    if c == 1:
        return image.repeat(3, 1, 1)

    return image[:3]


def _resize_bicubic_if_needed(
    image: torch.Tensor, target_height: int, target_width: int
) -> torch.Tensor:

    image = image.float()
    if image.shape[-2:] == (target_height, target_width):
        return image
    return (
        F.interpolate(
            image,
            size=(target_height, target_width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        .round_()
        .clamp_(0.0, 255.0)
    )


def _grid_thw_from_resize_config(config: dict, patch_size: int) -> tuple[int, int, int]:
    height = config["new_height"] + config["pad_height"]
    width = config["new_width"] + config["pad_width"]
    return 1, height // patch_size, width // patch_size


def _to_cuda_chw(image: Union[torch.Tensor, Image.Image]) -> torch.Tensor:
    if isinstance(image, Image.Image):
        return _pil_to_cuda_chw(image)
    return _ensure_chw_rgb(image)


def _process_single_image(
    image: Union[torch.Tensor, Image.Image],
    config: dict,
    image_scale: torch.Tensor,
    image_bias: torch.Tensor,
    patch_size: int,
    to_chw: Callable[[Union[torch.Tensor, Image.Image]], torch.Tensor] = _to_cuda_chw,
    post_resize: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:

    image = to_chw(image)

    new_h, new_w = config["new_height"], config["new_width"]
    padded_h = new_h + config["pad_height"]
    padded_w = new_w + config["pad_width"]

    x = _resize_bicubic_if_needed(image.unsqueeze(0), new_h, new_w)
    if post_resize is not None:
        x = post_resize(x)

    return normalize_and_patchify(
        x, image_scale, image_bias, patch_size, padded_h, padded_w
    ).squeeze(0)


def _resize_images_by_source_shape(
    indexed_images: list[tuple[int, torch.Tensor]],
    target_height: int,
    target_width: int,
) -> list[torch.Tensor]:

    by_source_shape = defaultdict(list)
    for index, image in indexed_images:
        by_source_shape[(tuple(image.shape), image.dtype)].append((index, image))

    resized_by_index = {}
    for images in by_source_shape.values():
        if len(images) == 1:
            index, image = images[0]
            resized_by_index[index] = _resize_bicubic_if_needed(
                image.unsqueeze(0), target_height, target_width
            )
            continue

        source_batch = torch.cat([image.unsqueeze(0) for _, image in images], dim=0)
        resized_batch = _resize_bicubic_if_needed(
            source_batch, target_height, target_width
        )
        for local_index, (index, _) in enumerate(images):
            resized_by_index[index] = resized_batch[local_index : local_index + 1]

    return [resized_by_index[index] for index, _ in indexed_images]


def _gpu_preprocess_images(
    images: list[Union[torch.Tensor, Image.Image]],
    resize_configs: list[dict],
    image_scale: torch.Tensor,
    image_bias: torch.Tensor,
    patch_size: int,
    to_chw: Callable[[Union[torch.Tensor, Image.Image]], torch.Tensor] = _to_cuda_chw,
    post_resize: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:

    n = len(images)
    if n == 0:
        device = image_scale.device
        return (
            torch.empty(0, 3, patch_size, patch_size, device=device),
            torch.empty(0, 3, dtype=torch.int64),
        )

    groups = defaultdict(list)
    for idx, (image, config) in enumerate(zip(images, resize_configs)):
        padded_h = config["new_height"] + config["pad_height"]
        padded_w = config["new_width"] + config["pad_width"]
        target_h = config["new_height"]
        target_w = config["new_width"]
        groups[(target_h, target_w, padded_h, padded_w)].append((idx, image, config))

    all_patches = [None] * n
    all_grids = [None] * n

    for (target_h, target_w, padded_h, padded_w), group in groups.items():
        if len(group) == 1:
            idx, image, config = group[0]
            patches = _process_single_image(
                image,
                config,
                image_scale,
                image_bias,
                patch_size,
                to_chw=to_chw,
                post_resize=post_resize,
            )
            all_patches[idx] = patches
            all_grids[idx] = _grid_thw_from_resize_config(config, patch_size)
        else:
            indexed_images = [(idx, to_chw(image)) for idx, image, _ in group]


            resized = _resize_images_by_source_shape(indexed_images, target_h, target_w)
            if post_resize is not None:


                resized = [post_resize(part) for part in resized]
            batch = torch.cat(resized, dim=0)

            T = 1
            gh, gw = padded_h // patch_size, padded_w // patch_size
            batch = normalize_and_patchify(
                batch,
                image_scale,
                image_bias,
                patch_size,
                padded_h,
                padded_w,
            )

            grid = (T, gh, gw)
            for i, (idx, _, _) in enumerate(group):
                all_patches[idx] = batch[i]
                all_grids[idx] = grid

    pixel_values = torch.cat(all_patches, dim=0)
    grid_thws = torch.tensor(all_grids, dtype=torch.int64)
    return pixel_values, grid_thws


class KimiGPUProcessorWrapper:


    def __init__(
        self,
        hf_processor,
        image_token,
        image_token_id,
        patch_size,
        merge_kernel_size,
        in_patch_limit,
        patch_limit_on_one_side,
        fixed_output_tokens,
        image_mean,
        image_std,
    ):
        self._hf_processor = hf_processor
        self._image_token = image_token
        self._image_token_id = image_token_id
        self._patch_size = patch_size
        self._merge_kernel_size = merge_kernel_size
        self._in_patch_limit = in_patch_limit
        self._patch_limit_on_one_side = patch_limit_on_one_side
        self._fixed_output_tokens = fixed_output_tokens
        self._image_mean = image_mean
        self._image_std = image_std
        self._gpu_norm_tensors = None


        self.image_processor = hf_processor.image_processor
        self.tokenizer = hf_processor.tokenizer
        self.media_processor = hf_processor.media_processor

    def __call__(self, text=None, images=None, **kwargs):

        images = images or kwargs.pop("images", None)
        original_input_ids = kwargs.pop("sglang_original_input_ids", None)

        if images and torch.cuda.is_available():
            return self._gpu_call(text, images, original_input_ids)
        return self._cpu_call(text, images, original_input_ids, **kwargs)

    def _prepare_input_ids(self, input_text, resize_configs, original_input_ids):
        if original_input_ids is not None:
            return _expand_image_token_ids(
                original_input_ids,
                self._image_token_id,
                [config["num_tokens"] for config in resize_configs],
            )

        parts = input_text.split(self._image_token)
        result = [parts[0]]
        for config, part in zip(resize_configs, parts[1:]):
            result.append(self._image_token * config["num_tokens"] + part)
        expanded_text = "".join(result)
        return self._hf_processor.tokenizer(expanded_text, return_tensors="pt")[
            "input_ids"
        ]

    def _gpu_call(self, text, images, original_input_ids=None):

        input_text = text[0] if isinstance(text, list) else text


        resize_configs = []
        for image in images:
            w, h = _get_image_dimensions(image)
            resize_configs.append(
                navit_resize_config(
                    w,
                    h,
                    self._patch_size,
                    self._merge_kernel_size,
                    self._in_patch_limit,
                    self._patch_limit_on_one_side,
                    self._fixed_output_tokens,
                )
            )


        input_ids = self._prepare_input_ids(
            input_text, resize_configs, original_input_ids
        )


        image_scale, image_bias = self._get_gpu_norm_tensors()
        pixel_values, grid_thws = _gpu_preprocess_images(
            images, resize_configs, image_scale, image_bias, self._patch_size
        )

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,


            "image_grid_thw": grid_thws,
        }

    def _cpu_call(self, text, images, original_input_ids=None, **kwargs):

        input_text = text[0] if isinstance(text, list) else text

        if images:

            image_token_counts = [
                int(
                    self._hf_processor.media_processor.media_tokens_calculator(
                        {"type": "image", "image": image}
                    )
                )
                for image in images
            ]
            parts = input_text.split(self._image_token)
            result = [parts[0]]
            for num_tokens, part in zip(image_token_counts, parts[1:]):
                result.append(self._image_token * num_tokens + part)
            input_text = "".join(result)


            kwargs["medias"] = [{"type": "image", "image": img} for img in images]

        out = self._hf_processor(text=[input_text], **kwargs)
        if images and original_input_ids is not None:


            out["input_ids"] = _expand_image_token_ids(
                original_input_ids, self._image_token_id, image_token_counts
            )
        grid_thws = out.pop("grid_thws", None)
        if grid_thws is not None:
            out["image_grid_thw"] = grid_thws
        return out

    def _get_gpu_norm_tensors(self, device="cuda"):
        if self._gpu_norm_tensors is None:
            image_scale = torch.tensor(
                [1.0 / (255.0 * std) for std in self._image_std],
                device=device,
                dtype=torch.float32,
            ).view(1, 3, 1, 1)
            image_bias = torch.tensor(
                [-mean / std for mean, std in zip(self._image_mean, self._image_std)],
                device=device,
                dtype=torch.float32,
            ).view(1, 3, 1, 1)
            self._gpu_norm_tensors = (image_scale, image_bias)
        return self._gpu_norm_tensors


class KimiK2_5VLImageProcessor(KimiGridMMDataMixin, SGLangBaseProcessor):
    models = [KimiK25ForConditionalGeneration]
    gpu_image_decode = True
    prefer_tokenized_input = True
    precompute_hash_before_cpu_transfer = True

    preserve_processor_input_ids = True
    auto_mm_processor_worker_num = 2
    auto_mm_io_worker_num = 16
    supports_mm_processor_concurrency = True

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        mm_tokens = MultimodalSpecialTokens(
            image_token="<|media_pad|>",

            image_token_id=hf_config.media_placeholder_token_id,
            image_token_regex=re.compile(r"(?:<\|media_pad\|>)+"),
        ).build(_processor)

        media_proc_cfg = _processor.media_processor.media_proc_cfg
        processor = KimiGPUProcessorWrapper(
            _processor,
            image_token=mm_tokens.image_token,
            image_token_id=mm_tokens.image_token_id,
            patch_size=media_proc_cfg["patch_size"],
            merge_kernel_size=media_proc_cfg["merge_kernel_size"],
            in_patch_limit=media_proc_cfg["in_patch_limit"],
            patch_limit_on_one_side=media_proc_cfg["patch_limit_on_one_side"],
            fixed_output_tokens=media_proc_cfg.get("fixed_output_tokens"),
            image_mean=media_proc_cfg["image_mean"],
            image_std=media_proc_cfg["image_std"],
        )


        super().__init__(hf_config, server_args, processor, *args, **kwargs)
        self.mm_tokens = mm_tokens

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes, Dict]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        expected_image_count = len(image_data or [])
        placeholder_count = self.count_image_placeholders(
            input_text, self.mm_tokens.image_token_id
        )
        if placeholder_count is not None:
            if placeholder_count != expected_image_count:
                raise ValueError(
                    "Kimi image placeholders must map one-to-one to image data: "
                    f"expected {expected_image_count}, found {placeholder_count} token(s)"
                )
            base_output = await self.fast_load_mm_data(
                prompt=input_text,
                image_data=image_data,
                multimodal_tokens=self.mm_tokens,


                input_ids=input_text,
            )
        else:
            base_output = await self.load_mm_data(
                prompt=input_text,
                image_data=image_data,
                multimodal_tokens=self.mm_tokens,
            )


            if len(base_output.images) != expected_image_count:
                raise ValueError(
                    "Kimi image placeholders must map one-to-one to image data: "
                    f"expected {expected_image_count}, loaded {len(base_output.images)}"
                )

        mm_items, input_ids, _ = await self.process_and_combine_mm_data_async(
            base_output,
            self.mm_tokens,
            sglang_original_input_ids=base_output.input_ids,
        )


        if self.keep_mm_features_on_device and self.server_args.mm_enable_dp_encoder:
            for item in mm_items:
                item.model_specific_data[DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY] = (
                    True
                )

        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_token_id=self.mm_tokens.image_token_id,
        )

    def get_mm_data(self, prompt, embeddings, **kwargs):
        img_grid_thw = kwargs.get("img_grid_thw", None)
        return self._build_kimi_mm_data_from_grids(
            prompt=prompt,
            embeddings=embeddings,
            image_token_id=self.mm_tokens.image_token_id,
            img_grid_thw=img_grid_thw,
        )
