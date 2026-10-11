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


from __future__ import annotations

import copy
import logging
import pickle
import uuid
from array import array
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
)

import msgspec
import numpy as np
import torch
import zmq
import zmq.asyncio
from pydantic import PlainValidator

from sglang.srt.environ import envs
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.managers.embed_types import PositionalEmbeds
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalProcessorOutput,
    ReturnHiddenStatesMode,
    get_return_hidden_states_mode,
)
from sglang.srt.multimodal.mm_utils import has_valid_data
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils import ImageData, VideoData
from sglang.srt.utils.field_validators import validate_optional_list_i64_1d_2d
from sglang.srt.utils.msgpack_utils import dec_hook, enc_hook, ext_hook
from sglang.srt.utils.msgspec_utils import (
    Base64Bytes,
    msgspec_struct_pydantic_core_schema,
)
from sglang.srt.utils.weight_versions import WeightVersionSpans


if TYPE_CHECKING:
    from PIL.Image import Image
else:
    Image = Any

logger = logging.getLogger(__name__)


class BaseReq(msgspec.Struct, tag=True, kw_only=True, array_like=True):


    rid: Optional[str] = None
    http_worker_ipc: Optional[str] = None

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        return msgspec_struct_pydantic_core_schema(cls, handler)


class BaseBatchReq(msgspec.Struct, tag=True, kw_only=True, array_like=True):


    rids: Optional[List[str]] = None


    http_worker_ipcs: Optional[List[Optional[str]]] = None

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        return msgspec_struct_pydantic_core_schema(cls, handler)


class PickleWrapper(msgspec.Struct, tag=True, array_like=True):


    data: bytes


class SessionParams(msgspec.Struct, kw_only=True, array_like=True):


    id: Optional[str] = None


    rid: Optional[str] = None


    offset: Optional[int] = None


    replace: Optional[bool] = None


    drop_previous_output: Optional[bool] = None


ImageDataInputItem = Union[str, bytes, Dict[str, Any], ImageData, Image]
AudioDataInputItem = Union[str, bytes, Dict[str, Any]]
VideoDataInputItem = Union[str, bytes, Dict[str, Any], VideoData]

MultimodalDataInputItem = Union[
    ImageDataInputItem, VideoDataInputItem, AudioDataInputItem
]

MultimodalDataInputFormat = Union[
    List[List[MultimodalDataInputItem]],
    List[MultimodalDataInputItem],
    MultimodalDataInputItem,
]


@dataclass
class GenerateReqInput:


    rid: Optional[Union[str, List[str]]] = field(default=None, kw_only=True)


    session_id: Optional[str] = field(default=None, kw_only=True)

    text: Optional[Union[List[str], str]] = None


    input_ids: Annotated[
        Optional[Union[List[List[int]], List[int]]],
        PlainValidator(validate_optional_list_i64_1d_2d),
    ] = None

    input_embeds: Optional[Union[List[List[List[float]]], List[List[float]]]] = None


    image_data: Optional[MultimodalDataInputFormat] = None

    video_data: Optional[MultimodalDataInputFormat] = None

    audio_data: Optional[MultimodalDataInputFormat] = None


    mm_hashes: Optional[Union[List[str], List[List[str]]]] = None


    mm_content_hashes: Optional[
        Union[List[Optional[str]], List[List[Optional[str]]]]
    ] = None

    use_audio_in_video: bool = False

    video_config: Optional[Dict[str, Any]] = None

    sampling_params: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None

    return_logprob: Optional[Union[List[bool], bool]] = None


    logprob_start_len: Optional[Union[List[int], int]] = None

    top_logprobs_num: Optional[Union[List[int], int]] = None

    token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None

    return_sampling_mask: Optional[Union[List[bool], bool]] = None

    return_text_in_logprobs: bool = False


    return_flat_raw_top_logprobs: bool = False

    return_flat_raw_top_logprobs_b64: bool = False

    stream: bool = False

    log_metrics: bool = True

    return_hidden_states: Union[
        List[ReturnHiddenStatesMode], ReturnHiddenStatesMode
    ] = False

    return_routed_experts: bool = False


    routed_experts_start_len: int = 0
    return_indexer_topk: bool = False


    modalities: Optional[List[str]] = None

    session_params: Optional[Dict[str, Any]] = None


    lora_path: Optional[Union[List[Optional[str]], str]] = None

    lora_id: Optional[Union[List[Optional[str]], str]] = None


    custom_logit_processor: Optional[Union[List[Optional[str]], str]] = None


    positional_embed_overrides: Any = None


    bootstrap_host: Optional[Union[List[Optional[str]], str]] = None
    bootstrap_port: Optional[Union[List[Optional[int]], int]] = None
    bootstrap_room: Optional[Union[List[Optional[int]], int]] = None
    bootstrap_pair_key: Optional[Union[List[Optional[str]], str]] = None
    decode_tp_size: Optional[Union[List[Optional[int]], int]] = None


    routed_dp_rank: Optional[int] = None


    data_parallel_rank: Optional[int] = None

    disagg_prefill_dp_rank: Optional[int] = None

    routing_key: Optional[str] = None

    conversation_id: Optional[str] = None


    http_worker_ipc: Optional[str] = field(default=None, kw_only=True)


    background: bool = False

    require_reasoning: bool = False


    max_thinking_tokens: Optional[int] = None


    priority: Optional[int] = None

    extra_key: Optional[Union[List[str], str]] = None


    no_logs: bool = False

    custom_labels: Optional[Dict[str, str]] = None


    return_bytes: bool = False

    return_entropy: bool = False

    return_prompt_token_ids: bool = False


    external_trace_header: Optional[Dict[str, Any]] = None
    received_time: Optional[float] = None


    need_wait_for_mm_inputs: Optional[bool] = None
    num_items_assigned: Optional[Dict[Modality, List[int]]] = None


    encoder_urls: Optional[List[str]] = None


    max_dynamic_patch: Optional[int] = None
    min_dynamic_patch: Optional[int] = None
    image_max_dynamic_patch: Optional[int] = None
    video_max_dynamic_patch: Optional[int] = None


    images_config: Optional[dict] = None


    multi_item_delimiter_indices: Optional[Union[List[List[int]], List[int]]] = None


    cache_salt: Optional[Union[List[str], str]] = None

    def regenerate_rid(self):

        if isinstance(self.rid, list):
            self.rid = [uuid.uuid4().hex for _ in range(len(self.rid))]
        else:
            self.rid = uuid.uuid4().hex
        return self.rid

    def _validate_rid_uniqueness(self):

        if isinstance(self.rid, list) and len(set(self.rid)) != len(self.rid):
            counts = Counter(self.rid)
            duplicates = [rid for rid, count in counts.items() if count > 1]
            raise ValueError(
                f"Duplicate request IDs detected within the request: {duplicates}"
            )

    def contains_mm_input(self) -> bool:
        return (
            has_valid_data(self.image_data)
            or has_valid_data(self.video_data)
            or has_valid_data(self.audio_data)
        )

    def normalize_batch_and_arguments(self):

        if self.data_parallel_rank is not None:
            import warnings

            warnings.warn(
                "'data_parallel_rank' is deprecated, use 'routed_dp_rank' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if self.routed_dp_rank is None:
                self.routed_dp_rank = self.data_parallel_rank
            self.data_parallel_rank = None

        self._validate_inputs()
        self._determine_batch_size()
        if self.session_id is not None and self.session_params is not None:
            raise ValueError("session_id and session_params cannot both be set.")
        self._handle_parallel_sampling()

        if self.is_single:
            self._normalize_single_inputs()
        else:
            self._normalize_batch_inputs()

        self._validate_rid_uniqueness()

    def _validate_inputs(self):

        if (
            self.text is None and self.input_ids is None and self.input_embeds is None
        ) or (
            self.text is not None
            and self.input_ids is not None
            and self.input_embeds is not None
        ):
            raise ValueError(
                "Either text, input_ids or input_embeds should be provided."
            )
        if (
            self.return_flat_raw_top_logprobs
            and self.multi_item_delimiter_indices is not None
        ):
            raise ValueError(
                "return_flat_raw_top_logprobs does not support multi-item "
                "scoring: delimiter-sparse top logprob rows have no contiguous "
                "position mapping."
            )
        if (
            self.return_flat_raw_top_logprobs_b64
            and not self.return_flat_raw_top_logprobs
        ):
            raise ValueError(
                "return_flat_raw_top_logprobs_b64 requires return_flat_raw_top_logprobs."
            )

    def _determine_batch_size(self):

        if self.text is not None:
            if isinstance(self.text, str):
                self.is_single = True
                self.batch_size = 1
            else:
                self.is_single = False
                self.batch_size = len(self.text)
            self.input_embeds = None
        elif self.input_ids is not None:
            if len(self.input_ids) == 0:
                raise ValueError("input_ids cannot be empty.")
            if isinstance(self.input_ids[0], int):
                self.is_single = True
                self.batch_size = 1
            else:
                self.is_single = False
                self.batch_size = len(self.input_ids)
            self.input_embeds = None
        else:
            if isinstance(self.input_embeds[0][0], float):
                self.is_single = True
                self.batch_size = 1
            else:
                self.is_single = False
                self.batch_size = len(self.input_embeds)

    def _handle_parallel_sampling(self):


        if self.sampling_params is None:
            self.parallel_sample_num = 1
            return
        elif isinstance(self.sampling_params, dict):
            self.parallel_sample_num = self.sampling_params.get("n", 1)
        else:
            self.parallel_sample_num = self.sampling_params[0].get("n", 1)
            for sampling_params in self.sampling_params:
                if self.parallel_sample_num != sampling_params.get("n", 1):
                    raise ValueError(
                        "The parallel_sample_num should be the same for all samples in sample params."
                    )


        if self.parallel_sample_num > 1 and self.is_single:
            self.is_single = False
            if self.text is not None:
                self.text = [self.text]
            if self.input_ids is not None:
                self.input_ids = [self.input_ids]
            if self.input_embeds is not None:
                self.input_embeds = [self.input_embeds]

    def _normalize_single_inputs(self):

        if self.sampling_params is None:
            self.sampling_params = {}
        if self.rid is None:
            self.rid = uuid.uuid4().hex
        if self.return_logprob is None:
            self.return_logprob = False
        if self.logprob_start_len is None:
            self.logprob_start_len = -1
        if self.top_logprobs_num is None:
            self.top_logprobs_num = 0
        if not self.token_ids_logprob:
            self.token_ids_logprob = None
        if self.return_sampling_mask is None:
            self.return_sampling_mask = False
        for field_name in ("extra_key", "cache_salt"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise ValueError(
                    f"{field_name} should be a string for a single request."
                )
            if value == "":
                setattr(self, field_name, None)

    def _normalize_batch_inputs(self):


        if self.parallel_sample_num == 1:
            num = self.batch_size
        else:

            num = self.batch_size * self.parallel_sample_num


        self._expand_inputs(num)
        self._normalize_rid(num)
        self._normalize_lora_paths(num)
        self._normalize_image_data(num)
        self._normalize_mm_hashes(num)
        self._normalize_video_data(num)
        self._normalize_audio_data(num)
        self._normalize_sampling_params(num)
        self._normalize_logprob_params(num)
        self._normalize_return_hidden_states(num)
        self._normalize_custom_logit_processor(num)
        self._normalize_extra_key(num)
        self._normalize_cache_salt(num)
        self._normalize_bootstrap_params(num)

    def _expand_inputs(self, num):

        if self.text is not None:
            if not isinstance(self.text, list):
                raise ValueError("Text should be a list for batch processing.")
            self.text = self.text * self.parallel_sample_num
        elif self.input_ids is not None:
            if not isinstance(self.input_ids, list) or not isinstance(
                self.input_ids[0], list
            ):
                raise ValueError(
                    "input_ids should be a list of lists for batch processing."
                )
            self.input_ids = self.input_ids * self.parallel_sample_num
        elif self.input_embeds is not None:
            if not isinstance(self.input_embeds, list):
                raise ValueError("input_embeds should be a list for batch processing.")
            self.input_embeds = self.input_embeds * self.parallel_sample_num

    def _normalize_lora_paths(self, num):

        if self.lora_path is not None:
            if isinstance(self.lora_path, str):
                self.lora_path = [self.lora_path] * num
            elif isinstance(self.lora_path, list):
                self.lora_path = self.lora_path * self.parallel_sample_num
            else:
                raise ValueError("lora_path should be a list or a string.")

    def _normalize_image_data(self, num):

        if self.image_data is None:
            self.image_data = [None] * num
        elif not isinstance(self.image_data, list):

            self.image_data = [[self.image_data]] * num
            self.modalities = ["image"] * num
        elif isinstance(self.image_data, list):

            if len(self.image_data) == 0:
                self.image_data = [None] * num
                return

            if len(self.image_data) != self.batch_size:
                raise ValueError(
                    "The length of image_data should be equal to the batch size."
                )

            self.modalities = []
            if len(self.image_data) > 0 and isinstance(self.image_data[0], list):

                for i in range(len(self.image_data)):
                    if self.image_data[i] is None or self.image_data[i] == [None]:
                        self.modalities.append(None)
                    elif len(self.image_data[i]) == 1:
                        self.modalities.append("image")
                    elif len(self.image_data[i]) > 1:
                        self.modalities.append("multi-images")
                    else:

                        self.modalities.append(None)

                self.image_data = self.image_data * self.parallel_sample_num
                self.modalities = self.modalities * self.parallel_sample_num
            else:

                wrapped_images = [[img] for img in self.image_data]

                self.image_data = wrapped_images * self.parallel_sample_num
                self.modalities = ["image"] * num

    def _normalize_mm_hashes(self, num):

        for field_name in ("mm_hashes", "mm_content_hashes"):
            hashes = getattr(self, field_name)
            if hashes is None:
                setattr(self, field_name, [None] * num)
                continue
            if not isinstance(hashes, list):
                raise ValueError(f"{field_name} must be a list")
            if len(hashes) != self.batch_size:
                raise ValueError(
                    f"The length of {field_name} should equal the batch size"
                )

            normalized = []
            for request_index, request_hashes in enumerate(hashes):
                images = self.image_data[request_index]
                image_count = len(images or [])
                if isinstance(request_hashes, list):
                    per_request = request_hashes
                elif image_count == 1:
                    per_request = [request_hashes]
                else:
                    raise ValueError(
                        f"{field_name}[{request_index}] must be a list with one "
                        "entry per image"
                    )
                if len(per_request) != image_count:
                    raise ValueError(
                        f"{field_name}[{request_index}] has {len(per_request)} "
                        f"entries for {image_count} images"
                    )
                normalized.append(per_request)
            setattr(self, field_name, normalized * self.parallel_sample_num)

    def _normalize_video_data(self, num):

        if self.video_data is None:
            self.video_data = [None] * num
        elif not isinstance(self.video_data, list):
            self.video_data = [self.video_data] * num
        elif isinstance(self.video_data, list):
            self.video_data = self.video_data * self.parallel_sample_num

    def _normalize_audio_data(self, num):

        if self.audio_data is None:
            self.audio_data = [None] * num
        elif not isinstance(self.audio_data, list):
            self.audio_data = [self.audio_data] * num
        elif isinstance(self.audio_data, list):
            self.audio_data = self.audio_data * self.parallel_sample_num

    def _normalize_sampling_params(self, num):

        if self.sampling_params is None:
            self.sampling_params = [{}] * num
        elif isinstance(self.sampling_params, dict):
            self.sampling_params = [self.sampling_params] * num
        else:
            self.sampling_params = self.sampling_params * self.parallel_sample_num

    def _normalize_rid(self, num):

        if self.rid is None:
            self.rid = [uuid.uuid4().hex for _ in range(num)]
        elif isinstance(self.rid, str):
            new_rids = [f"{self.rid}_{i}" for i in range(num)]
            self.rid = new_rids
        elif isinstance(self.rid, list):


            if len(self.rid) != self.batch_size:
                raise ValueError(
                    "The specified rids length mismatch with the batch_size for batch processing."
                )
        else:
            raise ValueError("The rid should be a string or a list of strings.")

    def _normalize_logprob_params(self, num):


        def normalize_param(param, default_value, param_name):
            if param is None:
                return [default_value] * num
            elif not isinstance(param, list):
                return [param] * num
            else:
                if self.parallel_sample_num > 1:
                    raise ValueError(
                        f"Cannot use list {param_name} with parallel_sample_num > 1"
                    )
                return param


        self.return_logprob = normalize_param(
            self.return_logprob, False, "return_logprob"
        )
        self.logprob_start_len = normalize_param(
            self.logprob_start_len, -1, "logprob_start_len"
        )
        self.top_logprobs_num = normalize_param(
            self.top_logprobs_num, 0, "top_logprobs_num"
        )
        self.return_sampling_mask = normalize_param(
            self.return_sampling_mask, False, "return_sampling_mask"
        )


        if not self.token_ids_logprob:
            self.token_ids_logprob = [None] * num
        elif not isinstance(self.token_ids_logprob, list):
            self.token_ids_logprob = [[self.token_ids_logprob] for _ in range(num)]
        elif not isinstance(self.token_ids_logprob[0], list):
            self.token_ids_logprob = [
                copy.deepcopy(self.token_ids_logprob) for _ in range(num)
            ]
        elif self.parallel_sample_num > 1:
            raise ValueError(
                "Cannot use list token_ids_logprob with parallel_sample_num > 1"
            )

    def _normalize_return_hidden_states(self, num):

        if isinstance(self.return_hidden_states, list):
            if len(self.return_hidden_states) != self.batch_size:
                raise ValueError(
                    "The length of return_hidden_states should be equal to the batch size."
                )
            for mode in self.return_hidden_states:
                get_return_hidden_states_mode(mode)
            self.return_hidden_states = (
                self.return_hidden_states * self.parallel_sample_num
            )
        else:
            get_return_hidden_states_mode(self.return_hidden_states)
            self.return_hidden_states = [self.return_hidden_states] * num

    def _normalize_custom_logit_processor(self, num):

        if self.custom_logit_processor is None:
            self.custom_logit_processor = [None] * num
        elif not isinstance(self.custom_logit_processor, list):
            self.custom_logit_processor = [self.custom_logit_processor] * num
        elif self.parallel_sample_num > 1:
            raise ValueError(
                "Cannot use list custom_logit_processor with parallel_sample_num > 1"
            )

    def _normalize_extra_key(self, num):

        if self.extra_key is None:
            return
        if isinstance(self.extra_key, str):
            value = self.extra_key or None
            self.extra_key = [value] * num
        elif isinstance(self.extra_key, list):
            if len(self.extra_key) != self.batch_size:
                raise ValueError(
                    "The length of extra_key should be equal to the batch size."
                )
            if any(not isinstance(value, str) for value in self.extra_key):
                raise ValueError("Every extra_key should be a string.")
            self.extra_key = [value or None for value in self.extra_key]
            self.extra_key = self.extra_key * self.parallel_sample_num
        else:
            raise ValueError("extra_key should be a list or a string.")

    def _normalize_cache_salt(self, num):

        if self.cache_salt is None:
            return
        if isinstance(self.cache_salt, str):
            value = self.cache_salt or None
            self.cache_salt = [value] * num
        elif isinstance(self.cache_salt, list):
            if len(self.cache_salt) != self.batch_size:
                raise ValueError(
                    "The length of cache_salt should be equal to the batch size."
                )
            if any(not isinstance(value, str) for value in self.cache_salt):
                raise ValueError("Every cache_salt should be a string.")
            self.cache_salt = [value or None for value in self.cache_salt]
            self.cache_salt = self.cache_salt * self.parallel_sample_num
        else:
            raise ValueError("cache_salt should be a list or a string.")

    def _normalize_bootstrap_params(self, num):


        if self.bootstrap_host is None:
            self.bootstrap_host = [None] * num
        elif not isinstance(self.bootstrap_host, list):
            self.bootstrap_host = [self.bootstrap_host] * num
        elif isinstance(self.bootstrap_host, list):
            self.bootstrap_host = self.bootstrap_host * self.parallel_sample_num


        if self.bootstrap_port is None:
            self.bootstrap_port = [None] * num
        elif not isinstance(self.bootstrap_port, list):
            self.bootstrap_port = [self.bootstrap_port] * num
        elif isinstance(self.bootstrap_port, list):
            self.bootstrap_port = self.bootstrap_port * self.parallel_sample_num


        if self.bootstrap_room is None:
            self.bootstrap_room = [None] * num
        elif not isinstance(self.bootstrap_room, list):
            self.bootstrap_room = [self.bootstrap_room + i for i in range(num)]
        elif isinstance(self.bootstrap_room, list):
            self.bootstrap_room = self.bootstrap_room * self.parallel_sample_num


        if self.bootstrap_pair_key is None:
            self.bootstrap_pair_key = [None] * num
        elif not isinstance(self.bootstrap_pair_key, list):
            self.bootstrap_pair_key = [self.bootstrap_pair_key] * num
        elif isinstance(self.bootstrap_pair_key, list):
            self.bootstrap_pair_key = self.bootstrap_pair_key * self.parallel_sample_num


        if self.decode_tp_size is None:
            self.decode_tp_size = [None] * num
        elif not isinstance(self.decode_tp_size, list):
            self.decode_tp_size = [self.decode_tp_size] * num
        elif isinstance(self.decode_tp_size, list):
            self.decode_tp_size = self.decode_tp_size * self.parallel_sample_num

    def _get_positional_embed_overrides_item(
        self, i: int
    ) -> Optional[PositionalEmbeds]:

        if self.positional_embed_overrides is None:
            return None
        if isinstance(self.positional_embed_overrides, PositionalEmbeds):
            return self.positional_embed_overrides
        return self.positional_embed_overrides[i]

    def __getitem__(self, i):


        cache = self.__dict__.setdefault("_sub_obj_cache", {})
        if i in cache:
            return cache[i]
        sub = GenerateReqInput(
            rid=self.rid[i],
            session_id=self.session_id,
            text=self.text[i] if self.text is not None else None,
            input_ids=self.input_ids[i] if self.input_ids is not None else None,
            input_embeds=(
                self.input_embeds[i] if self.input_embeds is not None else None
            ),
            image_data=self.image_data[i],
            video_data=self.video_data[i],
            audio_data=self.audio_data[i],
            mm_hashes=self.mm_hashes[i] if self.mm_hashes is not None else None,
            mm_content_hashes=(
                self.mm_content_hashes[i]
                if self.mm_content_hashes is not None
                else None
            ),
            sampling_params=self.sampling_params[i],
            return_logprob=self.return_logprob[i],
            logprob_start_len=self.logprob_start_len[i],
            top_logprobs_num=self.top_logprobs_num[i],
            token_ids_logprob=self.token_ids_logprob[i],
            return_sampling_mask=self.return_sampling_mask[i],
            return_text_in_logprobs=self.return_text_in_logprobs,
            return_flat_raw_top_logprobs=self.return_flat_raw_top_logprobs,
            return_flat_raw_top_logprobs_b64=self.return_flat_raw_top_logprobs_b64,
            stream=self.stream,
            log_metrics=self.log_metrics,
            return_hidden_states=(
                self.return_hidden_states[i]
                if isinstance(self.return_hidden_states, list)
                else self.return_hidden_states
            ),
            return_routed_experts=self.return_routed_experts,
            routed_experts_start_len=self.routed_experts_start_len,
            return_indexer_topk=self.return_indexer_topk,
            modalities=self.modalities[i] if self.modalities else None,
            session_params=self.session_params,
            lora_path=self.lora_path[i] if self.lora_path is not None else None,
            lora_id=self.lora_id[i] if self.lora_id is not None else None,
            custom_logit_processor=(
                self.custom_logit_processor[i]
                if self.custom_logit_processor is not None
                else None
            ),
            positional_embed_overrides=self._get_positional_embed_overrides_item(i),

            bootstrap_host=(
                self.bootstrap_host[i] if self.bootstrap_host is not None else None
            ),
            bootstrap_port=(
                self.bootstrap_port[i] if self.bootstrap_port is not None else None
            ),
            bootstrap_room=(
                self.bootstrap_room[i] if self.bootstrap_room is not None else None
            ),
            bootstrap_pair_key=(
                self.bootstrap_pair_key[i]
                if self.bootstrap_pair_key is not None
                else None
            ),
            decode_tp_size=(
                self.decode_tp_size[i] if self.decode_tp_size is not None else None
            ),
            routed_dp_rank=self.routed_dp_rank,
            disagg_prefill_dp_rank=self.disagg_prefill_dp_rank,
            conversation_id=self.conversation_id,
            http_worker_ipc=self.http_worker_ipc,
            require_reasoning=self.require_reasoning,
            max_thinking_tokens=self.max_thinking_tokens,
            priority=self.priority,
            extra_key=self.extra_key[i] if self.extra_key is not None else None,
            cache_salt=(self.cache_salt[i] if self.cache_salt is not None else None),
            no_logs=self.no_logs,
            custom_labels=self.custom_labels,
            return_bytes=self.return_bytes,
            return_entropy=self.return_entropy,
            return_prompt_token_ids=self.return_prompt_token_ids,
            external_trace_header=self.external_trace_header,
            received_time=self.received_time,
            multi_item_delimiter_indices=(
                self.multi_item_delimiter_indices[i]
                if self.multi_item_delimiter_indices is not None
                else None
            ),
        )
        cache[i] = sub
        return sub


class TokenizedGenerateReqInput(BaseReq, kw_only=True):
    input_text: Optional[Union[str, List[Union[str, List[str]]]]]

    input_ids: Optional[array]

    input_embeds: Optional[List[List[float]]]

    mm_inputs: Optional[MultimodalProcessorOutput]
    token_type_ids: Optional[List[int]]

    sampling_params: SamplingParams

    return_logprob: bool

    logprob_start_len: int

    top_logprobs_num: int

    token_ids_logprob: Optional[List[int]]

    stream: bool

    return_sampling_mask: bool = False


    return_flat_raw_top_logprobs: bool = False


    return_hidden_states: ReturnHiddenStatesMode = False


    return_routed_experts: bool = False

    routed_experts_start_len: int = 0
    return_indexer_topk: bool = False


    session_id: Optional[str] = None
    session_params: Optional[SessionParams] = None


    lora_id: Optional[str] = None


    custom_logit_processor: Optional[str] = None

    positional_embed_overrides: Optional[PositionalEmbeds] = None


    bootstrap_host: Optional[str] = None
    bootstrap_port: Optional[int] = None
    bootstrap_room: Optional[int] = None
    bootstrap_pair_key: Optional[str] = None
    decode_tp_size: Optional[int] = None


    routed_dp_rank: Optional[int] = None

    disagg_prefill_dp_rank: Optional[int] = None


    routing_key: Optional[str] = None

    require_reasoning: bool = False


    priority: Optional[int] = None


    extra_key: Optional[str] = None


    no_logs: bool = False


    return_bytes: bool = False

    return_entropy: bool = False

    need_wait_for_mm_inputs: Optional[bool] = None
    num_items_assigned: Optional[Dict[Modality, List[int]]] = None


    encoder_urls: Optional[List[str]] = None


    multi_item_delimiter_indices: Optional[List[int]] = None


    time_stats: Optional[PickleWrapper] = None


    cache_salt: Optional[str] = None

    def wrap_pickle_fields(self):
        self.time_stats = wrap_as_pickle(self.time_stats)

    def unwrap_pickle_fields(self):
        self.time_stats = unwrap_from_pickle(self.time_stats)


class BatchTokenizedGenerateReqInput(BaseBatchReq, kw_only=True):


    batch: List[TokenizedGenerateReqInput]

    def __len__(self):
        return len(self.batch)

    def __getitem__(self, i):
        return self.batch[i]

    def __iter__(self):
        return iter(self.batch)


@dataclass
class EmbeddingReqInput:


    rid: Optional[Union[str, List[str]]] = field(default=None, kw_only=True)

    text: Optional[Union[List[List[str]], List[str], str]] = None

    input_ids: Optional[Union[List[List[int]], List[int]]] = None

    input_embeds: Optional[Union[List[List[List[float]]], List[List[float]]]] = None


    image_data: Optional[MultimodalDataInputFormat] = None

    video_data: Optional[MultimodalDataInputFormat] = None

    audio_data: Optional[MultimodalDataInputFormat] = None

    embed_override_token_id: Optional[int] = None


    embed_overrides: Any = None

    sampling_params: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None

    log_metrics: bool = True

    modalities: Optional[List[str]] = None

    is_cross_encoder_request: bool = False

    lora_path: Optional[Union[List[Optional[str]], str]] = None

    lora_id: Optional[Union[List[Optional[str]], str]] = None


    positional_embed_overrides: Any = None

    routing_key: Optional[str] = None


    http_worker_ipc: Optional[str] = field(default=None, kw_only=True)


    background: bool = False


    priority: Optional[int] = None


    dimensions: Optional[int] = None

    return_pooled_hidden_states: bool = False

    return_prompt_token_ids: bool = False


    external_trace_header: Optional[Dict[str, Any]] = None
    received_time: Optional[float] = None


    multi_item_delimiter_indices: Optional[Union[List[List[int]], List[int]]] = None

    def regenerate_rid(self):

        if isinstance(self.rid, list):
            self.rid = [uuid.uuid4().hex for _ in range(len(self.rid))]
        else:
            self.rid = uuid.uuid4().hex
        return self.rid

    def _validate_rid_uniqueness(self):

        if isinstance(self.rid, list) and len(set(self.rid)) != len(self.rid):
            counts = Counter(self.rid)
            duplicates = [rid for rid, count in counts.items() if count > 1]
            raise ValueError(
                f"Duplicate request IDs detected within the request: {duplicates}"
            )

    def normalize_batch_and_arguments(self):

        if self.text is None and self.input_ids is None and self.image_data is None:
            raise ValueError(
                "At least one of text, input_ids, or image should be provided"
            )


        if self.text is not None and self.input_ids is not None:
            raise ValueError("text and input_ids cannot be provided at the same time")


        self.batch_size = 0
        self.is_single = True


        if self.text is not None:
            if isinstance(self.text, list):
                self.batch_size += len(self.text)
                self.is_single = False
            else:
                self.batch_size += 1


        if self.input_ids is not None:
            if isinstance(self.input_ids[0], list):
                self.batch_size += len(self.input_ids)
                self.is_single = False
            else:
                self.batch_size += 1


        if self.is_single:
            if self.rid is None:
                self.rid = uuid.uuid4().hex
            if self.sampling_params is None:
                self.sampling_params = {}
            self.sampling_params["max_new_tokens"] = 0
        else:
            if self.rid is None:
                self.rid = [uuid.uuid4().hex for _ in range(self.batch_size)]
            else:
                assert isinstance(self.rid, list), "The rid should be a list."

            if self.sampling_params is None:
                self.sampling_params = [{}] * self.batch_size
            elif isinstance(self.sampling_params, dict):
                self.sampling_params = [self.sampling_params] * self.batch_size
            for i in range(self.batch_size):
                self.sampling_params[i]["max_new_tokens"] = 0

            self._normalize_lora_paths(self.batch_size)

        self._validate_rid_uniqueness()

    def _normalize_lora_paths(self, num):

        if self.lora_path is not None:
            if isinstance(self.lora_path, str):
                self.lora_path = [self.lora_path] * num
            elif isinstance(self.lora_path, list):
                if len(self.lora_path) != num:
                    raise ValueError(
                        f"lora_path list length ({len(self.lora_path)}) must match batch size ({num})"
                    )
            else:
                raise ValueError("lora_path should be a list or a string.")

    def contains_mm_input(self) -> bool:
        return (
            has_valid_data(self.image_data)
            or has_valid_data(self.video_data)
            or has_valid_data(self.audio_data)
        )

    def _get_positional_embed_overrides_item(
        self, i: int
    ) -> Optional[PositionalEmbeds]:

        if self.positional_embed_overrides is None:
            return None
        if isinstance(self.positional_embed_overrides, PositionalEmbeds):
            return self.positional_embed_overrides
        return self.positional_embed_overrides[i]

    def __getitem__(self, i):

        cache = self.__dict__.setdefault("_sub_obj_cache", {})
        if i in cache:
            return cache[i]

        if self.is_cross_encoder_request:
            sub = EmbeddingReqInput(
                rid=self.rid[i],
                text=[self.text[i]] if self.text is not None else None,
                sampling_params=self.sampling_params[i],
                is_cross_encoder_request=True,
                lora_path=self.lora_path[i] if self.lora_path is not None else None,
                lora_id=self.lora_id[i] if self.lora_id is not None else None,
                positional_embed_overrides=self._get_positional_embed_overrides_item(i),
                http_worker_ipc=self.http_worker_ipc,
                priority=self.priority,
                return_pooled_hidden_states=self.return_pooled_hidden_states,
                return_prompt_token_ids=self.return_prompt_token_ids,
                multi_item_delimiter_indices=(
                    self.multi_item_delimiter_indices[i]
                    if self.multi_item_delimiter_indices is not None
                    else None
                ),
            )
        else:
            sub = EmbeddingReqInput(
                rid=self.rid[i],
                text=self.text[i] if self.text is not None else None,
                input_ids=self.input_ids[i] if self.input_ids is not None else None,
                image_data=self.image_data[i] if self.image_data is not None else None,
                video_data=self.video_data[i] if self.video_data is not None else None,
                audio_data=self.audio_data[i] if self.audio_data is not None else None,
                embed_override_token_id=self.embed_override_token_id,
                embed_overrides=(
                    self.embed_overrides[i]
                    if self.embed_overrides is not None
                    else None
                ),
                sampling_params=self.sampling_params[i],
                lora_path=self.lora_path[i] if self.lora_path is not None else None,
                lora_id=self.lora_id[i] if self.lora_id is not None else None,
                positional_embed_overrides=self._get_positional_embed_overrides_item(i),
                http_worker_ipc=self.http_worker_ipc,
                priority=self.priority,
                dimensions=self.dimensions,
                return_pooled_hidden_states=self.return_pooled_hidden_states,
                return_prompt_token_ids=self.return_prompt_token_ids,
                external_trace_header=self.external_trace_header,
                received_time=self.received_time,
                multi_item_delimiter_indices=(
                    self.multi_item_delimiter_indices[i]
                    if self.multi_item_delimiter_indices is not None
                    else None
                ),
            )
        cache[i] = sub
        return sub


class TokenizedEmbeddingReqInput(BaseReq, kw_only=True):
    input_text: Optional[Union[str, List[Union[str, List[str]]]]]

    input_ids: Optional[array]

    mm_inputs: Optional[MultimodalProcessorOutput]

    token_type_ids: Optional[List[int]]

    sampling_params: SamplingParams

    lora_id: Optional[str] = None

    positional_embed_overrides: Optional[PositionalEmbeds] = None

    routed_dp_rank: Optional[int] = None

    priority: Optional[int] = None

    dimensions: Optional[int] = None

    return_pooled_hidden_states: bool = False

    multi_item_delimiter_indices: Optional[List[int]] = None


    time_stats: Optional[PickleWrapper] = None

    def wrap_pickle_fields(self):
        self.time_stats = wrap_as_pickle(self.time_stats)

    def unwrap_pickle_fields(self):
        self.time_stats = unwrap_from_pickle(self.time_stats)


class BatchTokenizedEmbeddingReqInput(BaseBatchReq, kw_only=True):


    batch: List[TokenizedEmbeddingReqInput]

    def __len__(self):
        return len(self.batch)

    def __getitem__(self, i):
        return self.batch[i]

    def __iter__(self):
        return iter(self.batch)


TokenLogprobValues = Optional[List[Optional[List[Optional[float]]]]]
TokenLogprobIndices = Optional[List[Optional[List[Optional[int]]]]]
TopLogprobValues = Optional[List[Optional[List[Optional[List[float]]]]]]
TopLogprobIndices = Optional[List[Optional[List[Optional[List[int]]]]]]
TokenIdsLogprobValues = Optional[List[Optional[List[Optional[List[float]]]]]]
TokenIdsLogprobIndices = Optional[List[Optional[List[Optional[List[int]]]]]]
HiddenStateChunk = List[Optional[Union[float, List[float]]]]
OutputHiddenStates = Optional[List[Optional[List[HiddenStateChunk]]]]
CachedTokensDetails = Dict[str, Union[int, str]]

FinishReasonDict = Dict[str, Optional[Union[str, int, List[int]]]]


def build_flat_input_top_logprobs_arrays(
    input_top_logprobs_val: List[Optional[List[float]]],
    input_top_logprobs_idx: List[Optional[List[int]]],
    top_logprobs_num: int,
) -> Tuple[np.ndarray, np.ndarray, int]:

    num_rows = len(input_top_logprobs_val)
    null_prefix = 0
    while null_prefix < num_rows and not input_top_logprobs_val[null_prefix]:
        null_prefix += 1
    val_rows = input_top_logprobs_val[null_prefix:]
    idx_rows = input_top_logprobs_idx[null_prefix:]
    k = len(val_rows[0]) if val_rows else top_logprobs_num
    for offset, row in enumerate(val_rows):
        if row is None or len(row) != k:
            raise ValueError(
                "return_flat_raw_top_logprobs requires rectangular top logprob "
                f"rows with nulls only in the leading prefix; row {null_prefix + offset} "
                f"has {None if row is None else len(row)} entries (expected {k})."
            )
    val_arr = np.asarray(val_rows, dtype=np.float32).reshape(len(val_rows), k)
    idx_arr = np.asarray(idx_rows, dtype=np.int32).reshape(len(idx_rows), k)
    return val_arr, idx_arr, null_prefix


class BatchTokenIDOutput(BaseBatchReq, kw_only=True):

    finished_reasons: List[Optional[FinishReasonDict]]

    decoded_texts: List[str]
    decode_ids: List[array]
    read_offsets: List[int]

    output_ids: Optional[List[array]]

    skip_special_tokens: List[bool]
    spaces_between_special_tokens: List[bool]
    no_stop_trim: List[bool]


    prompt_tokens: List[int]
    reasoning_tokens: List[int]
    completion_tokens: List[int]
    cached_tokens: List[int]


    input_token_logprobs_val: TokenLogprobValues
    input_token_logprobs_idx: TokenLogprobIndices
    output_token_logprobs_val: TokenLogprobValues
    output_token_logprobs_idx: TokenLogprobIndices
    input_top_logprobs_val: TopLogprobValues
    input_top_logprobs_idx: TopLogprobIndices
    output_top_logprobs_val: TopLogprobValues
    output_top_logprobs_idx: TopLogprobIndices
    input_token_ids_logprobs_val: TokenIdsLogprobValues
    input_token_ids_logprobs_idx: TokenIdsLogprobIndices
    output_token_ids_logprobs_val: TokenIdsLogprobValues
    output_token_ids_logprobs_idx: TokenIdsLogprobIndices
    output_token_entropy_val: Optional[List[Optional[float]]]


    output_token_sampling_mask: Optional[List[List]]


    output_token_sampling_logprobs: Optional[List[List]]


    output_hidden_states: OutputHiddenStates


    routed_experts: Optional[List[Optional[torch.Tensor]]]

    indexer_topk: Optional[List[Optional[torch.Tensor]]]


    placeholder_tokens_idx: Optional[List[Optional[List[int]]]]
    placeholder_tokens_val: Optional[List[Optional[List[int]]]]


    retraction_counts: Optional[List[int]] = None

    weight_versions: Optional[List[Optional[WeightVersionSpans]]] = None


    token_steps: Optional[List[List[int]]] = None


    customized_info: Optional[PickleWrapper] = None

    cached_tokens_details: Optional[List[Optional[CachedTokensDetails]]] = None

    dp_ranks: Optional[List[Optional[int]]] = None


    time_stats: Optional[PickleWrapper] = None


    image_tokens: Optional[List[int]] = None
    audio_tokens: Optional[List[int]] = None
    video_tokens: Optional[List[int]] = None


    spec_verify_ct: Optional[List[int]] = None

    spec_num_correct_drafts: Optional[List[int]] = None

    spec_num_proposed_drafts: Optional[List[Optional[int]]] = None
    spec_num_block_accept_tokens: Optional[List[int]] = None
    spec_num_cap_tokens: Optional[List[int]] = None

    spec_correct_drafts_histogram: Optional[List[List[int]]] = None
    spec_cap_lens_histogram: Optional[List[List[int]]] = None


    input_top_logprobs_val_flat: Optional[List[Optional[np.ndarray]]] = None
    input_top_logprobs_idx_flat: Optional[List[Optional[np.ndarray]]] = None
    input_top_logprobs_flat_null_prefix: Optional[List[Optional[int]]] = None


class BatchStrOutput(BaseBatchReq, kw_only=True):

    finished_reasons: List[Optional[FinishReasonDict]]

    output_strs: List[str]

    output_ids: Optional[List[array]]


    prompt_tokens: List[int]
    completion_tokens: List[int]
    reasoning_tokens: List[int]
    cached_tokens: List[int]


    input_token_logprobs_val: TokenLogprobValues
    input_token_logprobs_idx: TokenLogprobIndices
    output_token_logprobs_val: TokenLogprobValues
    output_token_logprobs_idx: TokenLogprobIndices
    input_top_logprobs_val: TopLogprobValues
    input_top_logprobs_idx: TopLogprobIndices
    output_top_logprobs_val: TopLogprobValues
    output_top_logprobs_idx: TopLogprobIndices
    input_token_ids_logprobs_val: TokenIdsLogprobValues
    input_token_ids_logprobs_idx: TokenIdsLogprobIndices
    output_token_ids_logprobs_val: TokenIdsLogprobValues
    output_token_ids_logprobs_idx: TokenIdsLogprobIndices
    output_token_entropy_val: Optional[List[Optional[float]]]


    output_token_sampling_mask: Optional[List[List]]
    output_token_sampling_logprobs: Optional[List[List]]


    output_hidden_states: OutputHiddenStates


    routed_experts: Optional[List[Optional[str]]]

    indexer_topk: Optional[List[Optional[str]]]


    placeholder_tokens_idx: Optional[List[Optional[List[int]]]]
    placeholder_tokens_val: Optional[List[Optional[List[int]]]]


    retraction_counts: Optional[List[int]] = None

    weight_versions: Optional[List[Optional[WeightVersionSpans]]] = None


    token_steps: Optional[List[List[int]]] = None


    customized_info: Optional[PickleWrapper] = None

    cached_tokens_details: Optional[List[Optional[CachedTokensDetails]]] = None

    dp_ranks: Optional[List[Optional[int]]] = None


    time_stats: Optional[PickleWrapper] = None


    image_tokens: Optional[List[int]] = None
    audio_tokens: Optional[List[int]] = None
    video_tokens: Optional[List[int]] = None


    spec_verify_ct: Optional[List[int]] = None

    spec_num_correct_drafts: Optional[List[int]] = None

    spec_num_proposed_drafts: Optional[List[Optional[int]]] = None
    spec_num_block_accept_tokens: Optional[List[int]] = None
    spec_num_cap_tokens: Optional[List[int]] = None

    spec_correct_drafts_histogram: Optional[List[List[int]]] = None
    spec_cap_lens_histogram: Optional[List[List[int]]] = None


    input_top_logprobs_val_flat: Optional[List[Optional[np.ndarray]]] = None
    input_top_logprobs_idx_flat: Optional[List[Optional[np.ndarray]]] = None
    input_top_logprobs_flat_null_prefix: Optional[List[Optional[int]]] = None


class BatchEmbeddingOutput(BaseBatchReq, kw_only=True):

    finished_reasons: List[Optional[FinishReasonDict]]

    embeddings: List[Union[List[Union[float, List[float]]], Dict[int, float], float]]

    prompt_tokens: List[int]
    cached_tokens: List[int]

    placeholder_tokens_idx: Optional[List[Optional[List[int]]]]
    placeholder_tokens_val: Optional[List[Optional[List[int]]]]


    retraction_counts: Optional[List[int]] = None

    cached_tokens_details: Optional[List[Optional[CachedTokensDetails]]] = None


    time_stats: Optional[PickleWrapper] = None


    pooled_hidden_states: Optional[List[Optional[torch.Tensor]]] = None


class ClearHiCacheReqInput(BaseReq, kw_only=True):
    pass


class ClearHiCacheReqOutput(BaseReq, kw_only=True):
    success: bool


class FlushCacheReqInput(BaseReq, kw_only=True):
    timeout_s: Optional[float] = None


class FlushCacheReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str = ""


class AddExternalCorpusReqInput(BaseReq, kw_only=True):
    corpus_id: Optional[str] = None
    file_path: Optional[str] = None
    documents: Optional[List[str]] = None
    token_chunks: Optional[List[List[int]]] = None


class AddExternalCorpusReqOutput(BaseReq, kw_only=True):
    success: bool
    corpus_id: str = ""
    message: str = ""
    loaded_token_count: int = 0


class RemoveExternalCorpusReqInput(BaseReq, kw_only=True):
    corpus_id: str


class RemoveExternalCorpusReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str = ""


class ListExternalCorporaReqInput(BaseReq, kw_only=True):
    pass


class ListExternalCorporaReqOutput(BaseReq, kw_only=True):
    success: bool
    corpus_token_counts: Dict[str, int] = msgspec.field(default_factory=dict)
    message: str = ""


class AttachHiCacheStorageReqInput(BaseReq, kw_only=True):


    hicache_storage_backend: str
    hicache_storage_backend_extra_config_json: Optional[str] = None
    hicache_storage_prefetch_policy: Optional[str] = None
    hicache_write_policy: Optional[str] = None


class AttachHiCacheStorageReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str = ""


class DetachHiCacheStorageReqInput(BaseReq, kw_only=True):


    pass


class DetachHiCacheStorageReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str = ""


class PauseGenerationReqInput(BaseReq, kw_only=True):


    mode: Literal["abort", "retract", "in_place"] = "abort"


class ContinueGenerationReqInput(BaseReq, kw_only=True):


    torch_empty_cache: bool = True


class TokenizerWorkerRegistrationReq(BaseReq, kw_only=True):


    worker_ipc_name: str


class PauseContinueBroadcastReq(BaseReq, kw_only=True):


    is_pause: bool


class UpdateWeightFromDiskReqInput(BaseReq, kw_only=True):

    model_path: str

    load_format: Optional[str] = None

    abort_all_requests: bool = False

    weight_version: Optional[str] = None

    is_async: bool = False

    torch_empty_cache: bool = False

    keep_pause: bool = False

    recapture_cuda_graph: bool = False

    token_step: int = 0

    flush_cache: bool = True

    manifest: Optional[Dict[str, Any]] = None


class UpdateWeightFromDiskReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str

    num_paused_requests: int = 0


class UpdateWeightsFromDistributedReqInput(BaseReq, kw_only=True):
    names: List[str]
    dtypes: List[str]
    shapes: List[List[int]]

    group_name: str = "weight_update_group"

    flush_cache: bool = True

    abort_all_requests: bool = False

    weight_version: Optional[str] = None

    load_format: Optional[str] = None

    torch_empty_cache: bool = False


class UpdateWeightsFromDistributedReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str


class UpdateWeightsFromTensorReqInput(BaseReq, kw_only=True):


    serialized_named_tensors: Annotated[List[bytes], Base64Bytes()]

    load_format: Optional[str] = None

    flush_cache: bool = True

    abort_all_requests: bool = False

    weight_version: Optional[str] = None

    disable_draft_model: Optional[bool] = None

    torch_empty_cache: bool = False


class UpdateWeightsFromTensorReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str


class InitWeightsSendGroupForRemoteInstanceReqInput(BaseReq, kw_only=True):

    master_address: str

    ports: str

    group_rank: int

    world_size: int

    group_name: str = "weight_send_group"

    backend: str = "nccl"


class UpdateWeightsFromIPCReqInput(BaseReq, kw_only=True):

    zmq_handles: Dict[str, str]

    flush_cache: bool = True

    weight_version: Optional[str] = None

    torch_empty_cache: bool = False


class UpdateWeightsFromIPCReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str


class InitWeightsSendGroupForRemoteInstanceReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str


class SendWeightsToRemoteInstanceReqInput(BaseReq, kw_only=True):

    master_address: str

    ports: str

    group_name: str = "weight_send_group"


class SendWeightsToRemoteInstanceReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str


class UpdateExpertBackupReq(BaseReq, kw_only=True):
    pass


class ExpertWeightPointer(msgspec.Struct, kw_only=True, array_like=True):


    weight_ptr: int
    byte_size: int


class BackupDramReq(BaseReq, kw_only=True):
    rank: int
    weight_pointer_map: Dict[str, ExpertWeightPointer]
    session_id: str
    buffer_size: int


class InitWeightsUpdateGroupReqInput(BaseReq, kw_only=True):

    master_address: str

    master_port: int

    rank_offset: int

    world_size: int

    group_name: str = "weight_update_group"

    backend: str = "nccl"


class InitWeightsUpdateGroupReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str


class DestroyWeightsUpdateGroupReqInput(BaseReq, kw_only=True):
    group_name: str = "weight_update_group"


class DestroyWeightsUpdateGroupReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str


class UpdateWeightVersionReqInput(BaseReq, kw_only=True):

    new_version: str

    abort_all_requests: bool = True


class UpdateWeightVersionReqOutput(BaseReq, kw_only=True):
    pass


class GetWeightsByNameReqInput(BaseReq, kw_only=True):
    name: str
    truncate_size: int = 100


class GetWeightsByNameReqOutput(BaseReq, kw_only=True):


    parameter: Optional[List[Union[float, List[float]]]]


class ReleaseMemoryOccupationReqInput(BaseReq, kw_only=True):


    tags: Optional[List[str]] = None


class ReleaseMemoryOccupationReqOutput(BaseReq, kw_only=True):
    pass


class ResumeMemoryOccupationReqInput(BaseReq, kw_only=True):


    tags: Optional[List[str]] = None


class ResumeMemoryOccupationReqOutput(BaseReq, kw_only=True):
    pass


class CheckWeightsReqInput(BaseReq, kw_only=True):
    action: str = "checksum"
    allow_quant_error: bool = False


class ParallelismInfo(msgspec.Struct, kw_only=True):
    tp_rank: int
    tp_size: int
    dp_rank: int
    dp_size: int
    pp_rank: int
    pp_size: int
    rank: int
    size: int


class ChecksumInfo(msgspec.Struct, kw_only=True):
    checksums: Dict[str, str]
    per_gpu_checksum: str
    parallelism_info: ParallelismInfo


class CheckWeightsReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str


    payload: Optional[List[ChecksumInfo]] = None


class SlowDownReqInput(BaseReq, kw_only=True):
    forward_sleep_time: Optional[float]


class SlowDownReqOutput(BaseReq, kw_only=True):
    pass


class AbortReq(BaseReq, kw_only=True):

    abort_all: bool = False

    finished_reason: Optional[FinishReasonDict] = None
    abort_message: Optional[str] = None
    weight_versions: Optional[WeightVersionSpans] = None

    def __post_init__(self):

        if self.rid is None:
            self.rid = ""


class ActiveRanksOutput(BaseReq, kw_only=True):
    status: List[bool]


class ElasticScaleUpdateReq(BaseReq, kw_only=True):


    success: bool
    effective_ep_size: int
    slot_offset: int = 0
    slot_count: int = 0
    error: Optional[str] = None


class ScaleElasticEPReqInput(BaseReq, kw_only=True):


    new_ep_size: int


class ScaleElasticEPReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str
    old_ep_size: int = 0
    new_ep_size: int = 0
    pending_ep_size: Optional[int] = None
    scale_phase: str = "idle"


class GetInternalStateReq(BaseReq, kw_only=True):
    pass


class GetInternalStateReqOutput(BaseReq, kw_only=True):


    internal_state: Dict[str, Any]


class SetInternalStateReq(BaseReq, kw_only=True):

    server_args: Dict[str, Union[int, float]]


class SetInternalStateReqOutput(BaseReq, kw_only=True):
    updated: bool


class ProfileReqType(Enum):
    START_PROFILE = 1
    STOP_PROFILE = 2


class ProfileReq(BaseReq, kw_only=True):
    req_type: ProfileReqType = ProfileReqType.START_PROFILE

    output_dir: Optional[str] = None

    start_step: Optional[int] = None


    num_steps: Optional[int] = None

    activities: Optional[List[str]] = None

    profile_by_stage: bool = False

    with_stack: Optional[bool] = None

    record_shapes: Optional[bool] = None
    profile_id: Optional[str] = None

    merge_profiles: bool = False

    profile_prefix: Optional[str] = None

    profile_stages: Optional[List[str]] = None

    detailed_annotations: bool = False


class ProfileReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str


class FreezeGCReq(BaseReq, kw_only=True):
    pass


class ShutdownReq(BaseReq, kw_only=True):


    pass


class ConfigureLoggingReq(BaseReq, kw_only=True):
    log_requests: Optional[bool] = None
    log_requests_level: Optional[int] = None
    log_requests_format: Optional[str] = None
    log_level: Optional[str] = None
    dump_requests_folder: Optional[str] = None
    dump_requests_threshold: Optional[int] = None
    crash_dump_folder: Optional[str] = None
    dump_requests_exclude_meta_keys: Optional[List[str]] = None


class OpenSessionReqInput(BaseReq, kw_only=True):
    capacity_of_str_len: int
    session_id: Optional[str] = None
    streaming: Optional[bool] = None
    timeout: Optional[float] = None


class CloseSessionReqInput(BaseReq, kw_only=True):
    session_id: str


class OpenSessionReqOutput(BaseReq, kw_only=True):
    session_id: Optional[str]
    success: bool


class HealthCheckOutput(BaseReq, kw_only=True):
    pass


class ExpertDistributionReqType(Enum):
    START_RECORD = 1
    STOP_RECORD = 2
    DUMP_RECORD = 3


class ExpertDistributionReq(BaseReq, kw_only=True):
    action: ExpertDistributionReqType


class ExpertDistributionReqOutput(BaseReq, kw_only=True):
    pass


class Function(msgspec.Struct, kw_only=True, array_like=True):
    description: Optional[str] = None
    name: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        return msgspec_struct_pydantic_core_schema(cls, handler)


class Tool(msgspec.Struct, kw_only=True, array_like=True):
    function: Function
    type: str = "function"

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        return msgspec_struct_pydantic_core_schema(cls, handler)


class ParseFunctionCallReq(BaseReq, kw_only=True):
    text: str
    tools: List[Tool] = msgspec.field(
        default_factory=list
    )
    tool_call_parser: Optional[str] = (
        None
    )


class SeparateReasoningReqInput(BaseReq, kw_only=True):
    text: str
    reasoning_parser: str
    return_blocks: bool = False


class VertexGenerateReqInput(BaseReq, kw_only=True):


    instances: List[Dict[str, Any]]
    parameters: Optional[Dict[str, Any]] = None


class RpcReqInput(BaseReq, kw_only=True):
    method: str

    parameters: Optional[Dict[str, Union[bool, int, float, str, None]]] = None


class RpcReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str


class LoadLoRAAdapterReqInput(BaseReq, kw_only=True):

    lora_name: str

    lora_path: str

    pinned: bool = False

    lora_id: Optional[str] = None

    def to_ref(self) -> LoRARef:
        return LoRARef(
            lora_id=self.lora_id,
            lora_name=self.lora_name,
            lora_path=self.lora_path,
            pinned=self.pinned,
        )


class UnloadLoRAAdapterReqInput(BaseReq, kw_only=True):

    lora_name: str

    lora_id: Optional[str] = None

    def to_ref(self) -> LoRARef:
        return LoRARef(
            lora_id=self.lora_id,
            lora_name=self.lora_name,
        )


class LoadLoRAAdapterFromTensorsReqInput(BaseReq, kw_only=True):
    lora_name: str


    config_dict: Dict[str, Any]


    serialized_named_tensors: Annotated[List[bytes], Base64Bytes()]
    pinned: bool = False
    added_tokens_config: Optional[Dict[str, int]] = None
    lora_id: Optional[str] = None
    load_format: Optional[str] = None
    expected_checksums: Optional[Dict[str, str]] = None

    def to_ref(self) -> LoRARef:
        return LoRARef(
            lora_id=self.lora_id,
            lora_name=self.lora_name,
            lora_path="__tensor__",
            pinned=self.pinned,
        )


class LoRAUpdateOutput(BaseReq, kw_only=True):
    success: bool
    error_message: Optional[str] = None
    loaded_adapters: Optional[Dict[str, Union[str, LoRARef]]] = None


LoadLoRAAdapterReqOutput = UnloadLoRAAdapterReqOutput = (
    LoadLoRAAdapterFromTensorsReqOutput
) = LoRAUpdateOutput


class BlockReqType(Enum):
    BLOCK = 1
    UNBLOCK = 2


class BlockReqInput(BaseReq, kw_only=True):
    req_type: BlockReqType


class SetInjectDumpMetadataReqOutput(BaseReq, kw_only=True):
    success: bool


class LazyDumpTensorsReqInput(BaseReq, kw_only=True):
    pass


class LazyDumpTensorsReqOutput(BaseReq, kw_only=True):
    success: bool


class DumperControlReqInput(BaseReq, kw_only=True):
    method: str

    body: Dict[str, Any]


class DumperControlReqOutput(BaseReq, kw_only=True):
    success: bool

    response: List[Dict[str, Any]]
    error: str = ""


_IGNORE_REQ_TYPES_CHECK = (
    GenerateReqInput.__name__,
    EmbeddingReqInput.__name__,
)


def _check_all_req_types():

    import inspect
    import sys

    all_classes = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    for class_type in all_classes:

        name = class_type[0]
        if class_type[1].__module__ != __name__:
            continue
        if name in _IGNORE_REQ_TYPES_CHECK:
            continue
        is_io_struct = (
            name.endswith("Req") or name.endswith("Input") or name.endswith("Output")
        )
        is_base_req = issubclass(class_type[1], BaseReq) or issubclass(
            class_type[1], BaseBatchReq
        )
        if is_io_struct and not is_base_req:
            raise ValueError(f"{name} is not a subclass of BaseReq or BaseBatchReq.")
        if is_base_req and not is_io_struct:
            raise ValueError(
                f"{name} is a subclass of BaseReq but not follow the naming convention."
            )


_check_all_req_types()


def wrap_as_pickle(obj: object) -> object:
    if obj is None:
        return None
    if _USE_PICKLE_IPC:
        return obj
    return PickleWrapper(pickle.dumps(obj))


def unwrap_from_pickle(obj: Optional[object]) -> Optional[object]:
    if obj is None:
        return None
    if _USE_PICKLE_IPC:
        return obj
    if not isinstance(obj, PickleWrapper):


        return obj
    return pickle.loads(obj.data)


_struct_types = tuple(
    cls
    for cls in BaseReq.__subclasses__()
    + BaseBatchReq.__subclasses__()
    + [PickleWrapper]
)


_primitive_types = (int, float, bool, bytes)
_all_types = _struct_types + _primitive_types

_msgpack_encoder = msgspec.msgpack.Encoder(enc_hook=enc_hook)
_msgpack_decoder = msgspec.msgpack.Decoder(
    Union[_all_types], dec_hook=dec_hook, ext_hook=ext_hook
)
_USE_PICKLE_IPC = envs.SGLANG_USE_PICKLE_IPC.get()


def hook_custom_types(*new_types: Type):
    global _msgpack_decoder, _all_types
    _all_types = tuple(dict.fromkeys(_all_types + new_types))
    _msgpack_decoder = msgspec.msgpack.Decoder(
        Union[_all_types], dec_hook=dec_hook, ext_hook=ext_hook
    )


def _maybe_wrap_pickle(obj: Any) -> Any:
    if isinstance(obj, (msgspec.Struct, *_primitive_types)):
        return obj

    raise TypeError(
        f"Cannot serialize object of type {type(obj)} over msgpack IPC. "
        "Add a precise msgspec-compatible type, or use an explicit PickleWrapper "
        "field via wrap_as_pickle(...) for the opaque payload."
    )


def _maybe_unwrap_pickle(obj: Any) -> Any:
    if isinstance(obj, PickleWrapper):
        obj = pickle.loads(obj.data)
        if envs.SGLANG_LOG_PICKLE_IPC_OBJECTS.get():
            logger.info(f"Object of type {type(obj)} is unwrapped from PickleWrapper.")
        return obj

    return obj


def msgpack_encode(obj: Any) -> bytes:
    return _msgpack_encoder.encode(_maybe_wrap_pickle(obj))


def msgpack_decode(data: bytes) -> Any:
    return _maybe_unwrap_pickle(_msgpack_decoder.decode(data))


def sock_send(socket: zmq.Socket, obj: Any, flags: int = 0) -> None:
    if _USE_PICKLE_IPC:
        socket.send_pyobj(obj, flags=flags, protocol=pickle.HIGHEST_PROTOCOL)
        return

    socket.send(msgpack_encode(obj), flags=flags)


def sock_recv(socket: zmq.Socket, flags: int = 0) -> Any:
    if _USE_PICKLE_IPC:
        return socket.recv_pyobj(flags=flags)

    data = socket.recv(flags=flags)
    return msgpack_decode(data)


async def async_sock_send(socket: zmq.asyncio.Socket, obj: Any, flags: int = 0) -> None:
    if _USE_PICKLE_IPC:
        await socket.send_pyobj(obj, flags=flags, protocol=pickle.HIGHEST_PROTOCOL)
        return

    await socket.send(msgpack_encode(obj), flags=flags)


async def async_sock_recv(socket: zmq.asyncio.Socket, flags: int = 0) -> Any:
    if _USE_PICKLE_IPC:
        return await socket.recv_pyobj(flags=flags)

    data = await socket.recv(flags=flags)
    return msgpack_decode(data)
