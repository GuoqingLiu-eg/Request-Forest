import asyncio
import os
from typing import Dict, List, Optional, Union

import numpy as np
import requests
from transformers.models.auto.processing_auto import (
    PROCESSOR_MAPPING_NAMES as HF_MAPPING_NAMES,
)

import sglang.srt.managers.multimodal_processor as sgl_mm_processor_utils
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.llava import (
    LlavaForConditionalGeneration,
    LlavaLlamaForCausalLM,
    LlavaMistralForCausalLM,
    LlavaQwenForCausalLM,
)
from sglang.srt.models.llavavid import LlavaVidForCausalLM
from sglang.srt.models.mistral import Mistral3ForConditionalGeneration
from sglang.srt.multimodal.mm_utils import (
    ensure_numpy,
    expand2square,
    process_anyres_image,
)
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.utils import ImageData, get_image_bytes, load_image, logger
from sglang.utils import get_exception_traceback


class LlavaImageProcessor(BaseMultimodalProcessor):
    models = [
        LlavaLlamaForCausalLM,
        LlavaVidForCausalLM,
        LlavaQwenForCausalLM,
        LlavaMistralForCausalLM,
    ]
    gpu_image_decode = False

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

    @staticmethod
    def _preprocess_image_task(
        image_input,
        image_hash,
        image_aspect_ratio: Optional[str] = None,
        image_grid_pinpoints: Optional[str] = None,
        processor=None,
    ):


        image_processor = processor.image_processor

        try:
            image, image_size = load_image(image_input, False)
            if image_size is not None:

                pixel_values = image_processor(image)["pixel_values"]
                for i in range(len(pixel_values)):
                    pixel_values[i] = ensure_numpy(pixel_values[i]).astype(np.float16)
                pixel_values = np.stack(pixel_values, axis=0)
                return pixel_values, image_hash, image_size
            else:

                if image_aspect_ratio == "pad":
                    image = expand2square(
                        image,
                        tuple(int(x * 255) for x in image_processor.image_mean),
                    )
                    pixel_values = image_processor(image.convert("RGB"))[
                        "pixel_values"
                    ][0]
                elif image_aspect_ratio == "anyres" or (
                    image_aspect_ratio is not None
                    and "anyres_max" in image_aspect_ratio
                ):
                    pixel_values = process_anyres_image(
                        image, image_processor, image_grid_pinpoints
                    )
                else:
                    pixel_values = image_processor(image)["pixel_values"][0]

                pixel_values = ensure_numpy(pixel_values)
                if isinstance(pixel_values, np.ndarray):
                    pixel_values = pixel_values.astype(np.float16)

                return pixel_values, image_hash, image.size
        except Exception:
            logger.error("Exception in TokenizerManager:\n" + get_exception_traceback())

    async def _fetch_remote_image_bytes(self, url):


        loop = asyncio.get_running_loop()
        max_retries = max(0, int(os.environ.get("SGLANG_MM_LOAD_MAX_RETRIES", "2")))
        delay = 0.5
        for attempt in range(max_retries + 1):
            try:
                return await loop.run_in_executor(
                    self.io_executor, get_image_bytes, url
                )
            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
            ):
                if attempt >= max_retries:
                    raise
                await asyncio.sleep(delay)
                delay *= 2

    async def _process_single_image(
        self,
        image_data: Union[bytes, str, ImageData],
        aspect_ratio: str,
        grid_pinpoints: str,
    ):
        url = image_data.url if isinstance(image_data, ImageData) else image_data
        image_hash = hash(url) if isinstance(url, (str, bytes)) else None


        image_input = url
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            image_input = await self._fetch_remote_image_bytes(url)

        if self.cpu_executor is not None:
            loop = asyncio.get_running_loop()
            fut = loop.run_in_executor(
                self.cpu_executor,
                LlavaImageProcessor._preprocess_image_task,
                image_input,
                image_hash,
                aspect_ratio,
                grid_pinpoints,
                self._processor,
            )
            timeout = int(os.environ.get("REQUEST_TIMEOUT", "10"))
            return await asyncio.wait_for(fut, timeout=timeout)
        else:
            return LlavaImageProcessor._preprocess_image_task(
                image_input,
                image_hash,
                aspect_ratio,
                grid_pinpoints,
                self._processor,
            )

    def _process_precomputed_image_data(self, image_data: List[Dict]) -> Dict:
        mm_items = []
        for item in image_data:

            if "image_sizes" not in item:
                if "pixel_values" in item:
                    pv = item["pixel_values"]

                    h, w = (
                        (pv.shape[2], pv.shape[3])
                        if len(pv.shape) == 4
                        else (pv.shape[1], pv.shape[2])
                    )
                    item["image_sizes"] = [(w, h)]
                else:
                    item["image_sizes"] = [(336, 336)]

            mm_items.append(
                MultimodalDataItem(
                    feature=item["feature"],
                    modality=Modality.IMAGE,
                    model_specific_data=item,
                )
            )
        return MultimodalProcessorOutput(mm_items=mm_items)

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes, ImageData]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):


        if (
            isinstance(image_data, list)
            and len(image_data) > 0
            and isinstance(image_data[0], dict)
        ):
            return self._process_precomputed_image_data(image_data)

        modalities = request_obj.modalities or ["image"]
        aspect_ratio = getattr(self.hf_config, "image_aspect_ratio", None)
        grid_pinpoints = (
            self.hf_config.image_grid_pinpoints
            if hasattr(self.hf_config, "image_grid_pinpoints")
            and "anyres" in aspect_ratio
            else None
        )

        if isinstance(image_data, list) and len(image_data) > 0:
            if "multi-images" in modalities or "video" in modalities:

                aspect_ratio = "pad"
                pixel_values, data_hashes, image_sizes = [], [], []
                res = []
                for img_data in image_data:
                    res.append(
                        self._process_single_image(
                            img_data, aspect_ratio, grid_pinpoints
                        )
                    )

                res = await asyncio.gather(*res)
                for pixel_v, image_h, image_s in res:
                    pixel_values.append(pixel_v)
                    data_hashes.append(image_h)
                    image_sizes.append(image_s)
            else:

                pixel_values, image_hash, image_size = await self._process_single_image(
                    image_data[0], aspect_ratio, grid_pinpoints
                )
                pixel_values = [pixel_values]
                image_sizes = [image_size]
        else:
            raise ValueError(f"Invalid image data: {image_data}")
        modality = Modality.IMAGE
        if isinstance(request_obj.modalities, list):
            if request_obj.modalities[0] == "video":
                modality = Modality.VIDEO


        mm_items = []
        for pixel_v, image_s in zip(pixel_values, image_sizes):

            if isinstance(pixel_v, np.ndarray) and pixel_v.ndim == 3:
                pixel_v = np.expand_dims(pixel_v, 0)
            mm_items.append(
                MultimodalDataItem(
                    feature=pixel_v,
                    model_specific_data={
                        "image_sizes": [image_s],
                        "image_aspect_ratio": aspect_ratio,
                    },
                    modality=modality,
                )
            )

        return MultimodalProcessorOutput(
            mm_items=mm_items,
        )


class LlavaMultimodalProcessor(BaseMultimodalProcessor):


    models = [LlavaForConditionalGeneration, Mistral3ForConditionalGeneration]

    def _get_sgl_processor_cls(self, model_type: str):
        if model_type == "clip_vision_model":
            return LlavaImageProcessor
        if hf_name := HF_MAPPING_NAMES.get(model_type):
            sgl_mm_processor_set = sgl_mm_processor_utils.PROCESSOR_MAPPING.values()
            sgl_processor_cls = list(
                filter(lambda p: p.__name__ == hf_name, sgl_mm_processor_set)
            )
            if sgl_processor_cls:
                return sgl_processor_cls[0]
        raise ValueError(
            f"Cannot find corresponding multimodal processor registered in sglang for model type `{model_type}`"
        )

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        assert hasattr(hf_config, "vision_config")
        assert hasattr(hf_config, "text_config")
        self.vision_config = hf_config.vision_config
        self.text_config = hf_config.text_config
        self.hf_config = hf_config

        if vision_type := getattr(self.vision_config, "model_type"):
            self.inner = self._get_sgl_processor_cls(vision_type)(
                hf_config, server_args, _processor, *args, **kwargs
            )
        else:
            raise ValueError(
                f"Required `vision_config.model_type` is not found in hf_config: `{hf_config}`"
            )

    async def process_mm_data_async(self, *args, **kwargs):
        return await self.inner.process_mm_data_async(*args, **kwargs)
