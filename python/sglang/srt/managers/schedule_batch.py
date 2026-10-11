from __future__ import annotations

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import (
    get_disagg,
    get_schedule,
    get_serving,
    get_spec,
    mamba_cache_chunk_size,
    mamba_checkpoint_grid,
    mamba_extra_buffer_enabled,
    mamba_extra_buffer_lazy_enabled,
    mamba_track_grid,
)
from sglang.srt.utils.common import (
    Range,
    ceil_align,
    flatten_arrays_to_pinned_cpu,
    is_pin_memory_available,
)
from sglang.srt.utils.weight_versions import (
    WeightVersionEvent,
    truncate_weight_version_events,
)

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
"""
Store information about requests and batches.

The following is the flow of data structures for a batch:

ScheduleBatch -> ForwardBatch

- ScheduleBatch is managed by `scheduler.py::Scheduler`.
  It contains high-level scheduling data. Most of the data is on the CPU.
- ForwardBatch is managed by `model_runner.py::ModelRunner`.
  It contains low-level tensor data. Most of the data consists of GPU tensors.
  It is constructed directly from a ScheduleBatch by `ForwardBatch.init_new`.
"""

import copy
import dataclasses
import logging
import re
import sys
from array import array
from concurrent.futures import Future
from enum import Enum, auto
from functools import lru_cache
from http import HTTPStatus
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    TypeAlias,
    Union,
)

import msgspec
import numpy as np
import torch

from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
from sglang.srt.disaggregation.base import BaseKVSender
from sglang.srt.disaggregation.decode_schedule_batch_mixin import (
    ScheduleBatchDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST, DisaggregationMode
from sglang.srt.dllm.mixin.req import ReqDllmMixin
from sglang.srt.environ import envs
from sglang.srt.managers.embed_types import PositionalEmbeds
from sglang.srt.managers.scheduler_components.new_token_ratio_tracker import (
    NewTokenRatioTracker,
)
from sglang.srt.mem_cache.allocation import (
    alloc_for_decode,
    alloc_for_extend,
)
from sglang.srt.mem_cache.allocation_sizing import get_alloc_reserve_per_decode
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    MatchPrefixParams,
    zero_match_result,
)
from sglang.srt.mem_cache.common import (
    RetractionBackup,
    evict_from_tree_cache,
    free_swa_out_of_window_slots,
    release_kv_cache,
    retraction_backup,
)
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.multimodal.transport.cuda_ipc import (
    DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY,
    CudaIpcTensorTransportProxy,
)
from sglang.srt.observability.metrics_collector import (
    DPCooperationInfo,
    SchedulerMetricsCollector,
)
from sglang.srt.observability.req_time_stats import (
    APIServerReqTimeStats,
    DPControllerReqTimeStats,
    SchedulerReqTimeStats,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import flatten_nested_list
from sglang.srt.utils.token_sequence_matcher import TokenSequenceMatcher

if TYPE_CHECKING:
    from typing import Any, Dict

    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
    from sglang.srt.managers.scheduler_components.metrics_reporter import PrefillStats
    from sglang.srt.session.session_controller import Session
    from sglang.srt.speculative.spec_info import SpecInput, SpeculativeAlgorithm

INIT_INCREMENTAL_DETOKENIZATION_OFFSET = 5


MM_PAD_SHIFT_VALUE = 1_000_000
_MM_HASH_MASK = (1 << 64) - 1

logger = logging.getLogger(__name__)


ReturnHiddenStatesMode = Union[bool, Literal["last"]]


def get_return_hidden_states_mode(
    return_hidden_states: ReturnHiddenStatesMode,
) -> CaptureHiddenMode:
    if return_hidden_states is True:
        return CaptureHiddenMode.FULL
    if return_hidden_states == "last":
        return CaptureHiddenMode.LAST
    if return_hidden_states is False:
        return CaptureHiddenMode.NULL
    raise ValueError(
        "return_hidden_states must be a boolean or the string literal 'last'."
    )


def get_request_return_hidden_states_mode(
    return_hidden_states: Union[List[ReturnHiddenStatesMode], ReturnHiddenStatesMode],
) -> CaptureHiddenMode:
    if isinstance(return_hidden_states, list):
        return max(
            (get_return_hidden_states_mode(mode) for mode in return_hidden_states),
            default=CaptureHiddenMode.NULL,
        )
    return get_return_hidden_states_mode(return_hidden_states)


def get_batch_return_hidden_states_mode(reqs: List[Req]) -> CaptureHiddenMode:
    mode = CaptureHiddenMode.NULL
    for req in reqs:
        mode = max(mode, req.return_hidden_states_mode)
    return mode


def need_return_hidden_states(
    return_hidden_states: Union[List[ReturnHiddenStatesMode], ReturnHiddenStatesMode],
) -> bool:
    return get_request_return_hidden_states_mode(return_hidden_states).need_capture()


@lru_cache(maxsize=1)
def sanity_check_mm_pad_shift_value(vocab_size: int) -> None:
    if vocab_size > MM_PAD_SHIFT_VALUE:
        raise ValueError(
            f"Model vocab_size ({vocab_size}) exceeds MM_PAD_SHIFT_VALUE ({MM_PAD_SHIFT_VALUE}). "
            f"MM pad_values may overlap with valid token IDs. "
            f"Please increase MM_PAD_SHIFT_VALUE in schedule_batch.py."
        )


def split_cached_prefix_by_tier(
    prefix_len: int, host_hit_len: int, storage_hit_len: int
) -> tuple[int, int, int]:

    storage = min(host_hit_len, storage_hit_len)
    host = host_hit_len - storage
    device = max(0, prefix_len - host_hit_len)
    return device, host, storage


def _compute_pad_value(hash: int) -> int:

    return MM_PAD_SHIFT_VALUE + (hash % (1 << 30))


class BaseFinishReason:
    def to_json(self):
        raise NotImplementedError()


class FINISH_MATCHED_TOKEN(BaseFinishReason):
    def __init__(self, matched: Union[int, List[int]]):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",
            "matched": self.matched,
        }


class FINISH_MATCHED_STR(BaseFinishReason):
    def __init__(self, matched: str):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",
            "matched": self.matched,
        }


class FINISHED_MATCHED_REGEX(BaseFinishReason):
    def __init__(self, matched: str):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",
            "matched": self.matched,
        }


class FINISH_LENGTH(BaseFinishReason):
    def __init__(self, length: int):
        super().__init__()
        self.length = length

    def to_json(self):
        return {
            "type": "length",
            "length": self.length,
        }


class FINISH_ABORT(BaseFinishReason):
    def __init__(self, message=None, status_code=None, err_type=None):
        super().__init__()
        self.message = message or "Aborted"
        self.status_code = status_code
        self.err_type = err_type

    def to_json(self):
        return {
            "type": "abort",
            "message": self.message,
            "status_code": self.status_code,
            "err_type": self.err_type,
        }


class Modality(Enum):
    IMAGE = auto()
    VIDEO = auto()
    AUDIO = auto()

    @staticmethod
    def from_str(modality_str: str):
        try:
            return Modality[modality_str.upper()]
        except KeyError:
            raise ValueError(
                f"Invalid modality string: {modality_str}. Valid modalities are: {[m.name for m in Modality]}"
            )

    @staticmethod
    def all():
        return [Modality.IMAGE, Modality.VIDEO, Modality.AUDIO]


class MultimodalInputFormat(Enum):
    NORMAL = auto()
    PROCESSOR_OUTPUT = auto()
    PRECOMPUTED_EMBEDDING = auto()


MultimodalDataValue: TypeAlias = object


class MultimodalDataItem(msgspec.Struct, kw_only=True, dict=True, array_like=True):


    modality: Modality
    hash: Optional[int] = None
    pad_value: Optional[int] = None
    offsets: Optional[List[Tuple[int, int]]] = None

    format: MultimodalInputFormat = MultimodalInputFormat.NORMAL


    feature: Optional[MultimodalDataValue] = None


    precomputed_embeddings: Optional[MultimodalDataValue] = None

    keep_device_embedding: bool = False


    model_specific_data: Dict[str, MultimodalDataValue] = msgspec.field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.hash is not None:
            msgspec.Struct.__setattr__(self, "hash", self.hash & _MM_HASH_MASK)

    def __getattr__(self, name: str) -> MultimodalDataValue:
        if name in self.model_specific_data:
            return self.model_specific_data[name]

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value: MultimodalDataValue) -> None:
        if name in self.__struct_fields__:
            if name == "hash" and isinstance(value, int):
                value &= _MM_HASH_MASK
            msgspec.Struct.__setattr__(self, name, value)
        else:
            self.model_specific_data[name] = value

    def __setitem__(self, key: str, value: MultimodalDataValue) -> None:
        setattr(self, key, value)

    def set(self, key: str, value: MultimodalDataValue) -> None:
        self.__setitem__(key, value)

    def set_hash(self, hash_value: int) -> None:
        self.hash = hash_value
        self.pad_value = _compute_pad_value(hash_value)

    @staticmethod
    def is_empty_list(l):
        if l is None:
            return True
        return len([item for item in flatten_nested_list(l) if item is not None]) == 0

    def set_pad_value(self):
        if self.pad_value is not None:
            return

        from sglang.srt.multimodal.cache import resolve_multimodal_item_hash

        self.hash = resolve_multimodal_item_hash(
            existing_hash=self.hash,
            feature=self.feature,
            precomputed_embeddings=self.precomputed_embeddings,
        )
        self.pad_value = _compute_pad_value(self.hash)

    def is_modality(self, modality: Modality) -> bool:
        return self.modality == modality

    def is_audio(self):
        return self.modality == Modality.AUDIO

    def is_image(self):
        return self.modality == Modality.IMAGE

    def is_video(self):
        return self.modality == Modality.VIDEO

    def is_valid(self) -> bool:
        return self.is_image() or self.is_video() or self.is_audio()

    def validate(self):
        ...


    def is_precomputed_embedding(self):
        return self.format == MultimodalInputFormat.PRECOMPUTED_EMBEDDING

    @staticmethod
    def from_dict(obj: dict):
        kwargs = dict(obj)
        modality = kwargs.pop("modality")
        if isinstance(modality, str):
            modality = Modality[modality]
        ret = MultimodalDataItem(modality=modality, **kwargs)
        ret.validate()
        return ret

    def has_cuda_ipc_proxy(self):
        return (
            isinstance(self.feature, CudaIpcTensorTransportProxy)
            or isinstance(self.precomputed_embeddings, CudaIpcTensorTransportProxy)
            or any(
                isinstance(value, CudaIpcTensorTransportProxy)
                for value in self.model_specific_data.values()
            )
        )

    def reconstruct(self, target_device: int, ipc_consumer_count: int = 1):

        if isinstance(self.feature, CudaIpcTensorTransportProxy):
            consumer_count = self._resolve_transport_consumer_count(
                self.feature, ipc_consumer_count
            )
            if consumer_count == 1:
                self.feature = self.feature.reconstruct_on_target_device(target_device)
            else:
                self.feature = self.feature.reconstruct_on_target_device(
                    target_device, consumer_count=consumer_count
                )
        if isinstance(self.precomputed_embeddings, CudaIpcTensorTransportProxy):
            self.precomputed_embeddings = (
                self.precomputed_embeddings.reconstruct_on_target_device(target_device)
            )
        for extra_key in self.model_specific_data:
            if isinstance(
                self.model_specific_data[extra_key], CudaIpcTensorTransportProxy
            ):
                extra_data = self.model_specific_data[
                    extra_key
                ].reconstruct_on_target_device(target_device)
                self.model_specific_data[extra_key] = extra_data

    def can_defer_cuda_ipc_feature_reconstruction(self) -> bool:

        return (
            self.model_specific_data.get(
                DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY, False
            )
            and self.hash is not None
            and self.pad_value is not None
            and isinstance(self.feature, CudaIpcTensorTransportProxy)
            and not isinstance(self.precomputed_embeddings, CudaIpcTensorTransportProxy)
            and not any(
                isinstance(value, CudaIpcTensorTransportProxy)
                for value in self.model_specific_data.values()
            )
        )

    def acknowledge_deferred_cuda_ipc_feature(self, consumer_count: int = 1):

        if isinstance(self.feature, CudaIpcTensorTransportProxy):
            consumer_count = self._resolve_transport_consumer_count(
                self.feature, consumer_count
            )
            self.feature.acknowledge_consumption(consumer_count)

    @staticmethod
    def _resolve_transport_consumer_count(proxy, requested_count: int) -> int:

        proxy_count = getattr(
            proxy,
            "total_consumer_count",
            getattr(proxy, "consumer_count", requested_count),
        )
        return min(requested_count, proxy_count)


class MultimodalProcessorOutput(
    msgspec.Struct, kw_only=True, dict=True, array_like=True, weakref=True
):


    mm_items: List[MultimodalDataItem]
    input_ids: Optional[List[int]] = None
    padded_input_ids: Optional[List[int]] = None


    im_token_id: Optional[int] = None
    im_start_id: Optional[int] = None
    im_end_id: Optional[int] = None
    slice_start_id: Optional[int] = None
    slice_end_id: Optional[int] = None


    video_token_id: Optional[int] = None


    audio_token_id: Optional[int] = None
    audio_start_id: Optional[int] = None
    audio_end_id: Optional[int] = None


    mrope_positions: Optional[torch.Tensor] = None
    mrope_position_delta: Optional[torch.Tensor] = None


    vision_position_ids: Optional[torch.Tensor] = None
    media_nums_per_sample: Optional[List[int]] = None
    visible_frame_counts: Optional[torch.Tensor] = None


    token_type_ids: Optional[torch.Tensor] = None

    @staticmethod
    def from_dict(d: dict) -> MultimodalProcessorOutput:
        return MultimodalProcessorOutput(
            mm_items=d["mm_items"],
            input_ids=d.get("input_ids"),
            padded_input_ids=d.get("padded_input_ids"),
            im_token_id=d.get("im_token_id"),
            im_start_id=d.get("im_start_id"),
            im_end_id=d.get("im_end_id"),
            slice_start_id=d.get("slice_start_id"),
            slice_end_id=d.get("slice_end_id"),
            video_token_id=d.get("video_token_id"),
            audio_token_id=d.get("audio_token_id"),
            audio_start_id=d.get("audio_start_id"),
            audio_end_id=d.get("audio_end_id"),
            mrope_positions=d.get("mrope_positions"),
            mrope_position_delta=d.get("mrope_position_delta"),
            vision_position_ids=d.get("vision_position_ids"),
            media_nums_per_sample=d.get("media_nums_per_sample"),
            visible_frame_counts=d.get("visible_frame_counts"),
        )

    @staticmethod
    def build_padded_input_ids(input_ids, mm_items: List[MultimodalDataItem]):

        if input_ids is None or not mm_items:
            return None

        for item in mm_items:
            if item.pad_value is None or item.offsets is None:
                return None

        if isinstance(input_ids, torch.Tensor):
            padded_input_ids = input_ids.flatten().tolist()
        else:
            padded_input_ids = list(input_ids)

        for item in mm_items:
            for start, end in item.offsets:
                padded_input_ids[start : end + 1] = [item.pad_value] * (end - start + 1)
        return padded_input_ids


@dataclasses.dataclass
class MultimodalInputs:


    mm_items: List[MultimodalDataItem]
    padded_input_ids: Optional[List[int]] = None
    image_pad_len: Optional[list] = None
    num_image_tokens: Optional[int] = None


    im_token_id: Optional[int] = None
    im_start_id: Optional[int] = None
    im_end_id: Optional[int] = None
    slice_start_id: Optional[int] = None
    slice_end_id: Optional[int] = None


    video_token_id: Optional[int] = None


    audio_token_id: Optional[int] = None
    audio_start_id: Optional[int] = None
    audio_end_id: Optional[int] = None


    mrope_positions: Optional[torch.Tensor] = None
    mrope_position_delta: Optional[torch.Tensor] = None
    mrope_position_delta_repeated_cache: Optional[torch.Tensor] = None


    vision_position_ids: Optional[torch.Tensor] = None
    media_nums_per_sample: Optional[List[int]] = None
    visible_frame_counts: Optional[torch.Tensor] = None

    def release_features(self):

        for item in self.mm_items:
            item.feature = None

    @staticmethod
    def from_processor_output(obj: MultimodalProcessorOutput):
        mm_items = obj.mm_items
        assert isinstance(mm_items, list)
        mm_items = [item for item in mm_items if item.is_valid()]


        reconstruct_device = None
        for mm_item in mm_items:
            if (
                mm_item.has_cuda_ipc_proxy()
                and not mm_item.can_defer_cuda_ipc_feature_reconstruction()
            ):
                if reconstruct_device is None:
                    reconstruct_device = torch.cuda.current_device()
                mm_item.reconstruct(reconstruct_device)

        if envs.SGLANG_MM_BUFFER_SIZE_MB.get() > 0:


            from sglang.srt.managers.mm_utils import (
                init_feature_buffer,
                is_feature_buffer_initialized,
                reset_buffer_offset,
                try_add_to_buffer,
            )

            device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
            if not is_feature_buffer_initialized():
                init_feature_buffer(device)
            reset_buffer_offset()
            for item in mm_items:
                if item.feature is not None:
                    if isinstance(item.feature, torch.Tensor):
                        item.feature = try_add_to_buffer(item.feature)

        for item in mm_items:
            item.set_pad_value()

        if envs.SGLANG_MM_BUFFER_SIZE_MB.get() > 0:
            for item in mm_items:
                if item.feature is not None:
                    item.feature = item.feature.to("cpu", non_blocking=True)

        mm_inputs = MultimodalInputs(
            mm_items=mm_items,
            padded_input_ids=obj.padded_input_ids,
        )
        optional_args = [
            "mrope_positions",
            "mrope_position_delta",
            "im_token_id",
            "im_start_id",
            "im_end_id",
            "video_token_id",
            "slice_start_id",
            "slice_end_id",
            "audio_start_id",
            "audio_end_id",
            "audio_token_id",
            "vision_position_ids",
            "media_nums_per_sample",
            "visible_frame_counts",
        ]
        for arg in optional_args:
            val = getattr(obj, arg, None)
            if val is not None:
                setattr(mm_inputs, arg, val)

        return mm_inputs

    def contains_image_inputs(self) -> bool:
        return any(item.is_image() for item in self.mm_items)

    def contains_video_inputs(self) -> bool:
        return any(item.is_video() for item in self.mm_items)

    def contains_audio_inputs(self) -> bool:
        return any(item.is_audio() for item in self.mm_items)

    def contains_mm_input(self) -> bool:
        return any(True for item in self.mm_items if item.is_valid())

    def compute_mm_token_counts(self) -> Tuple[int, int, int]:

        image_tokens = audio_tokens = video_tokens = 0
        for item in self.mm_items:
            if not item.offsets:
                continue
            num_tokens = sum(end - start + 1 for start, end in item.offsets)
            if item.is_image():
                image_tokens += num_tokens
            elif item.is_audio():
                audio_tokens += num_tokens
            elif item.is_video():
                video_tokens += num_tokens
        return image_tokens, audio_tokens, video_tokens

    def merge(self, other: MultimodalInputs):


        optional_args = [
            "mm_items",
            "image_pad_len",
        ]
        for arg in optional_args:
            self_arg = getattr(self, arg, None)
            if self_arg is not None:
                setattr(self, arg, self_arg + getattr(other, arg))

        mrope_positions = self.mrope_positions
        if mrope_positions is not None:
            if other.mrope_positions is None:
                self.mrope_positions = mrope_positions
            else:
                self.mrope_positions = torch.cat(
                    [self.mrope_positions, other.mrope_positions], dim=1
                )

        mrope_position_delta = self.mrope_position_delta
        if mrope_position_delta is not None:
            if other.mrope_position_delta is None:
                self.mrope_position_delta = mrope_position_delta
            else:
                self.mrope_position_delta = torch.cat(
                    [self.mrope_position_delta, other.mrope_position_delta], dim=0
                )

        for key, val in other.__dict__.items():
            if "_id" in key:

                if getattr(self, key, None) is None:
                    setattr(self, key, getattr(other, key, None))


@dataclasses.dataclass(slots=True, kw_only=True)
class ReqLogprob:
    top_logprobs_num: int
    token_ids_logprob: Optional[List[int]]
    input_token_logprobs_val: Optional[List[float]] = None
    input_token_logprobs_idx: Optional[List[int]] = None
    input_top_logprobs_val: Optional[List[List[float]]] = None
    input_top_logprobs_idx: Optional[List[List[int]]] = None


    input_top_logprobs_val_flat: Optional[np.ndarray] = None
    input_top_logprobs_idx_flat: Optional[np.ndarray] = None
    input_top_logprobs_flat_null_prefix: Optional[int] = None
    input_token_ids_logprobs_val: Optional[List[List[float]]] = None
    input_token_ids_logprobs_idx: Optional[List[List[int]]] = None
    output_token_logprobs_val: Optional[list] = None
    output_token_logprobs_idx: Optional[list] = None
    output_top_logprobs_val: Optional[list] = None
    output_top_logprobs_idx: Optional[list] = None

    output_token_ids_logprobs_val: Optional[List[Union[List[float], torch.Tensor]]] = (
        None
    )
    output_token_ids_logprobs_idx: Optional[list] = None


@dataclasses.dataclass(slots=True, kw_only=True)
class ReqKvInfo:
    kv_allocated_len: int


    swa_evicted_seqlen: int


class Req(ReqDllmMixin):


    def __init__(
        self,
        rid: str,
        origin_input_text: str,
        origin_input_ids: array[int],
        sampling_params: SamplingParams,
        return_logprob: bool = False,
        top_logprobs_num: int = 0,
        dllm_config: Optional[DllmConfig] = None,
        token_ids_logprob: List[int] = None,
        return_sampling_mask: bool = False,
        return_flat_raw_top_logprobs: bool = False,
        stream: bool = False,
        origin_input_ids_unpadded: Optional[array[int]] = None,
        lora_id: Optional[str] = None,
        input_embeds: Optional[List[List[float]]] = None,
        positional_embed_overrides: Optional[PositionalEmbeds] = None,
        token_type_ids: List[int] = None,
        session: Optional[Session] = None,
        custom_logit_processor: Optional[str] = None,
        require_reasoning: bool = False,
        return_hidden_states: ReturnHiddenStatesMode = False,
        return_routed_experts: bool = False,
        routed_experts_start_len: int = 0,
        return_indexer_topk: bool = False,
        eos_token_ids: Optional[Set[int]] = None,
        bootstrap_host: Optional[str] = None,
        bootstrap_port: Optional[int] = None,
        bootstrap_room: Optional[int] = None,
        disagg_mode: Optional[DisaggregationMode] = None,
        routed_dp_rank: Optional[int] = None,
        disagg_prefill_dp_rank: Optional[int] = None,
        vocab_size: Optional[int] = None,
        priority: Optional[int] = None,
        metrics_collector: Optional[SchedulerMetricsCollector] = None,
        extra_key: Optional[str] = None,
        routing_key: Optional[str] = None,
        dimensions: Optional[int] = None,
        http_worker_ipc: Optional[str] = None,
        time_stats: Optional[
            Union[APIServerReqTimeStats, DPControllerReqTimeStats]
        ] = None,
        return_pooled_hidden_states: bool = False,
        multi_item_delimiter_indices: Optional[List[int]] = None,
        session_id: Optional[str] = None,
        cache_salt: Optional[str] = None,
    ):

        self.rid = rid
        self.origin_input_ids = origin_input_ids
        self.origin_input_ids_unpadded = (
            origin_input_ids_unpadded
            if origin_input_ids_unpadded
            else self.origin_input_ids
        )


        self.output_ids = array("q")


        self.full_untruncated_fill_ids = array("q")
        self.extend_range: Optional[Range] = None
        self.dllm_initialized: bool = False

        self.session = session
        self.session_id = session_id

        self.session_generation: Optional[int] = None
        self.input_embeds = input_embeds
        self.positional_embed_overrides = positional_embed_overrides
        self.multi_item_delimiter_indices = multi_item_delimiter_indices


        self.kv_committed_len = 0
        self.kv: Optional[ReqKvInfo] = None
        self.retraction_backup: Optional[RetractionBackup] = None


        self.token_type_ids = token_type_ids


        self.swa_evict_floor: int = 0


        self.extend_batch_idx = 0
        self.decode_batch_idx = 0


        self.http_worker_ipc = http_worker_ipc


        self.require_reasoning = require_reasoning


        self._is_reasoning_over = False
        self.reasoning_tokens = 0
        self._think_end_matcher: Optional[TokenSequenceMatcher] = None
        self._think_end_match_len = 0


        if isinstance(sampling_params.custom_params, dict):
            sampling_params = copy.copy(sampling_params)
            sampling_params.custom_params = sampling_params.custom_params | {
                "__req__": self
            }
        self.sampling_params = sampling_params
        self.custom_logit_processor = custom_logit_processor
        self.return_hidden_states = return_hidden_states
        self.return_hidden_states_mode = get_return_hidden_states_mode(
            return_hidden_states
        )


        if lora_id is not None:
            extra_key = (
                extra_key or ""
            ) + lora_id

        self.extra_key = extra_key
        self.cache_salt = cache_salt or None
        self.lora_id = lora_id
        self.routing_key = routing_key


        self.req_pool_idx: Optional[int] = None
        self.mamba_pool_idx: Optional[torch.Tensor] = None
        self.mamba_ping_pong_track_buffer: Optional[torch.Tensor] = None
        self.mamba_next_track_idx: Optional[int] = None
        self.mamba_last_track_idx: Optional[int] = None
        self.mamba_last_track_seqlen: Optional[int] = (
            None
        )


        self.mamba_branching_seqlen: Optional[int] = None

        self.mamba_cow_src_index: Optional[torch.Tensor] = None

        self.mamba_needs_clear: bool = False


        self.mamba_lazy_is_insert: bool = True


        self.tokenizer = None
        self.finished_reason: Optional[BaseFinishReason] = None

        self.finished_len = None

        self.finished_output = None


        self.to_finish: Optional[BaseFinishReason] = None
        self.stream = stream
        self.eos_token_ids = eos_token_ids
        self.vocab_size = vocab_size
        self.priority = priority


        self.surr_offset = None
        self.read_offset = None
        self.decoded_text = ""


        self.multimodal_inputs: Optional[MultimodalInputs] = None


        self.mm_image_tokens: int = 0
        self.mm_audio_tokens: int = 0
        self.mm_video_tokens: int = 0


        self.prefix_indices: torch.Tensor = torch.empty((0,), dtype=torch.int64)

        self.last_node: Any = None
        self.last_host_node: Any = None
        self.best_match_node: Any = None

        self.host_hit_length = 0
        self.swa_host_hit_length = 0
        self.mamba_host_hit_length = 0


        self.num_matched_prefix_tokens = 0

        self.storage_hit_length = 0

        self.swa_uuid_for_lock: Optional[int] = None

        self.swa_prefix_lock_released: bool = False


        self.skip_lock_node_ids: dict = {}

        self.cache_protected_len: int = 0


        self.inflight_middle_chunks = 0


        self.is_retracted = False

        self.retracted_stain = False

        self.weight_version_events: List[WeightVersionEvent] = []


        self.send_token_offset: int = 0
        self.send_decode_id_offset: int = 0


        self.send_output_token_logprobs_offset: int = 0
        self.send_output_sampling_mask_offset: int = 0


        self.return_logprob = return_logprob

        self.logprob_start_len = 0
        self.logprob = ReqLogprob(
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
        )
        self.temp_scaled_logprobs = False
        self.top_p_normalized_logprobs = False
        self.return_sampling_mask = return_sampling_mask
        self.return_flat_raw_top_logprobs = return_flat_raw_top_logprobs


        self.input_logprob_sent: bool = False

        self.input_token_logprobs: Optional[List[Tuple[int]]] = None
        self.temp_input_top_logprobs_val: Optional[List[torch.Tensor]] = None
        self.temp_input_top_logprobs_idx: Optional[List[int]] = None
        self.temp_input_token_ids_logprobs_val: Optional[List[float]] = None
        self.temp_input_token_ids_logprobs_idx: Optional[List[int]] = None

        if return_logprob:

            self.logprob.output_token_logprobs_val = []
            self.logprob.output_token_logprobs_idx = []

            self.logprob.output_top_logprobs_val = []
            self.logprob.output_top_logprobs_idx = []

            self.logprob.output_token_ids_logprobs_val = []
            self.logprob.output_token_ids_logprobs_idx = []
        if return_sampling_mask:
            self.output_token_sampling_mask = []
            self.output_token_sampling_logprobs = []
        else:
            self.output_token_sampling_mask = None
            self.output_token_sampling_logprobs = None
        self.hidden_states: List[List[float]] = []
        self.hidden_states_tensor = None
        self.output_topk_p = None
        self.output_topk_index = None
        self.output_dsa_topk_indices = None


        self.return_routed_experts = return_routed_experts
        self.routed_experts_start_len = routed_experts_start_len
        self.routed_experts: Optional[torch.Tensor] = (
            None
        )

        self.return_indexer_topk = return_indexer_topk
        self.indexer_topk: Optional[torch.Tensor] = (
            None
        )

        self.customized_info: Optional[Dict[str, List[Any]]] = None


        self.embedding = None


        self.grammar_key: Optional[Tuple[str, str]] = None
        self.grammar: Optional[Union[BaseGrammarObject, Future[BaseGrammarObject]]] = (
            None
        )
        self.grammar_wait_ct = 0


        self.cached_tokens = 0
        self.already_computed = 0


        self.cached_tokens_device = 0
        self.cached_tokens_host = 0
        self.cached_tokens_storage = 0
        self._cache_breakdown_computed = (
            False
        )


        self.spec_verify_ct = 0


        self.spec_num_correct_drafts = 0


        self.spec_num_proposed_drafts: Optional[int] = None

        self.spec_num_block_accept_tokens = 0

        self.spec_num_cap_tokens = 0


        self.spec_correct_drafts_histogram: List[int] = []

        self.spec_cap_lens_histogram: List[int] = []


        self.retraction_count = 0
        self.retraction_mb_id = None


        self.metrics_collector = metrics_collector
        if time_stats is not None:
            self.time_stats = SchedulerReqTimeStats.new_from_obj(time_stats)
        else:
            self.time_stats = SchedulerReqTimeStats(disagg_mode=disagg_mode)
        self.time_stats.set_metrics_collector(metrics_collector)
        self.time_stats.set_scheduler_recv_time()
        self.has_log_time_stats: bool = False


        self.bootstrap_host: str = bootstrap_host
        self.bootstrap_port: Optional[int] = bootstrap_port
        self.bootstrap_room: Optional[int] = bootstrap_room


        self.pd_rebootstrap_forced_output_id: Optional[int] = None
        self.skip_radix_cache_insert = bootstrap_host == FAKE_BOOTSTRAP_HOST
        self.disagg_kv_sender: Optional[BaseKVSender] = None

        self.routed_dp_rank: Optional[int] = routed_dp_rank
        self.disagg_prefill_dp_rank: Optional[int] = disagg_prefill_dp_rank


        self.start_send_idx: int = 0
        self.disagg_decode_prefix_len: int = 0


        self.tmp_end_idx: int = -1


        self.disagg_decode_prefix_len: int = 0


        self.early_send_prefix_end: Optional[int] = None
        self.metadata_buffer_index: int = -1


        self.pending_bootstrap = False

        self.prefill_attempt_count = 0


        self.dimensions = dimensions


        self.return_pooled_hidden_states = return_pooled_hidden_states
        self.pooled_hidden_state = None


        self.init_diffusion_llm(dllm_config)


        self.hisparse_staging = False

    @property
    def seqlen(self) -> int:

        return len(self.origin_input_ids) + len(self.output_ids)

    @property
    def is_prefill_only(self) -> bool:


        spec_alg = get_spec().speculative_algorithm
        return self.sampling_params.max_new_tokens == 0 and spec_alg is None

    @property
    def output_ids_through_stop(self) -> array[int]:

        if self.finished_len is not None:
            return self.output_ids[: self.finished_len]
        return self.output_ids

    def needs_host_load_back(self) -> bool:

        return (
            self.host_hit_length > 0
            or self.swa_host_hit_length > 0
            or self.mamba_host_hit_length > 0
        )

    def effective_kv_committed_len(self) -> int:


        if get_serving().strip_thinking_cache and self.reasoning_tokens > 0:
            return min(self.kv_committed_len, len(self.origin_input_ids))
        return self.kv_committed_len

    def update_spec_correct_drafts_histogram(self, num_correct_drafts: int):

        if len(self.spec_correct_drafts_histogram) <= num_correct_drafts:
            self.spec_correct_drafts_histogram.extend(
                [0] * (num_correct_drafts - len(self.spec_correct_drafts_histogram) + 1)
            )
        self.spec_correct_drafts_histogram[num_correct_drafts] += 1

    def update_spec_cap_lens_histogram(self, cap_len: int):
        cap_len = int(cap_len)
        if len(self.spec_cap_lens_histogram) <= cap_len:
            self.spec_cap_lens_histogram.extend(
                [0] * (cap_len - len(self.spec_cap_lens_histogram) + 1)
            )
        self.spec_cap_lens_histogram[cap_len] += 1

    def extend_image_inputs(self, image_inputs):
        if self.multimodal_inputs is None:
            self.multimodal_inputs = image_inputs
        else:
            self.multimodal_inputs.merge(image_inputs)

    def finished(self) -> bool:

        return self.finished_reason is not None

    def set_extend_range(self, start: int, end: int) -> None:
        self.extend_range = Range(start, end)

    def get_fill_ids(self) -> array:
        return self.full_untruncated_fill_ids[: self.extend_range.end]

    def _refresh_fill_ids(self) -> None:

        n_have_output = len(self.full_untruncated_fill_ids) - len(self.origin_input_ids)
        if (
            self.full_untruncated_fill_ids is not self.origin_input_ids
            and 0 <= n_have_output <= len(self.output_ids)
        ):
            self.full_untruncated_fill_ids.extend(self.output_ids[n_have_output:])
        else:
            self.full_untruncated_fill_ids = self.origin_input_ids + self.output_ids

    def init_next_round_input(
        self,
        tree_cache: Optional[BasePrefixCache] = None,
        cow_mamba: Optional[bool] = None,
    ):
        if self.is_dllm():
            self._init_fill_ids_for_dllm()
            self.determine_dllm_phase()
        else:
            self._refresh_fill_ids()

        input_len = len(self.full_untruncated_fill_ids)


        if (
            self.session is not None
            and self.session.streaming
            and self.return_logprob
            and self.logprob_start_len >= 0
        ):
            logger.warning(
                "logprob_start_len=%d is not supported for streaming sessions "
                "and will be ignored (rid=%s). Only new-token logprobs are returned.",
                self.logprob_start_len,
                self.rid,
            )
            self.logprob_start_len = -1


        token_ids_to_match = self.full_untruncated_fill_ids
        key_limit: Optional[int] = self._compute_max_prefix_len(input_len)


        if tree_cache is not None:
            reprefill_tail = tree_cache.swa_reprefill_tail_tokens()
            if reprefill_tail:
                capped = max(0, input_len - reprefill_tail)
                key_limit = capped if key_limit is None else min(key_limit, capped)


        if self.positional_embed_overrides is not None:
            token_ids_to_match = array("q")
            key_limit = None

        if tree_cache is not None:
            if cow_mamba is None:
                cow_mamba = tree_cache.supports_mamba()


            reprefill_tail = tree_cache.swa_reprefill_tail_tokens()
            if reprefill_tail:
                capped = max(0, input_len - reprefill_tail)
                key_limit = capped if key_limit is None else min(key_limit, capped)
            match_result = tree_cache.match_prefix(
                MatchPrefixParams(
                    key=RadixKey(
                        token_ids=token_ids_to_match,
                        extra_key=self.extra_key,
                        limit=key_limit,
                        cache_salt=self.cache_salt,
                    ),
                    req=self,
                    cow_mamba=cow_mamba,
                )
            )
            if envs.SGLANG_RADIX_FORCE_MISS.get():
                match_result = zero_match_result(
                    tree_cache, match_result, extra_key=self.extra_key
                )
            (
                self.prefix_indices,
                self.last_node,
                self.last_host_node,
                self.best_match_node,
                self.host_hit_length,
                self.swa_host_hit_length,
                self.mamba_host_hit_length,
                self.mamba_branching_seqlen,
            ) = (
                match_result.device_indices,
                match_result.last_device_node,
                match_result.last_host_node,
                match_result.best_match_node,
                match_result.host_hit_length,
                match_result.swa_host_hit_length,
                match_result.mamba_host_hit_length,
                match_result.mamba_branching_seqlen,
            )
            if match_result.cache_protected_len is not None:
                self.cache_protected_len = match_result.cache_protected_len
            else:
                self.cache_protected_len = len(self.prefix_indices)

            if self.is_dllm():
                self._update_block_offset_for_dllm()

        if (
            self.is_retracted
            and self.multimodal_inputs is not None
            and self.multimodal_inputs.mrope_positions is not None
        ):
            from sglang.srt.managers.mm_utils import (
                extend_mrope_positions_for_retracted_request,
            )

            self.multimodal_inputs.mrope_positions = (
                extend_mrope_positions_for_retracted_request(
                    self.multimodal_inputs.mrope_positions, len(self.output_ids)
                )
            )

    def _compute_max_prefix_len(self, input_len: int) -> int:

        max_prefix_len = input_len - 1
        if self.return_logprob and self.logprob_start_len >= 0:
            max_prefix_len = min(max_prefix_len, self.logprob_start_len)
        return max(max_prefix_len, 0)


    def init_incremental_detokenize(self):
        first_iter = self.surr_offset is None or self.read_offset is None

        output_ids = self.output_ids_through_stop

        if first_iter:
            self.read_offset = len(self.origin_input_ids_unpadded)
            self.surr_offset = max(
                self.read_offset - INIT_INCREMENTAL_DETOKENIZATION_OFFSET, 0
            )
            self.surr_and_decode_ids = (
                self.origin_input_ids_unpadded[self.surr_offset :] + output_ids
            )
            self.cur_decode_ids_len = len(output_ids)
        else:
            self.surr_and_decode_ids.extend(output_ids[self.cur_decode_ids_len :])
            self.cur_decode_ids_len = len(output_ids)

        return self.surr_and_decode_ids, self.read_offset - self.surr_offset

    def _stop_match_tail_len(self, new_accepted_len: int) -> int:
        max_len_tail_str = max(
            self.sampling_params.stop_str_max_len + 1,
            self.sampling_params.stop_regex_max_len + 1,
        )


        return min(
            max_len_tail_str + max(new_accepted_len - 1, 0), len(self.output_ids)
        )

    def tail_str(self, new_accepted_len: int = 1) -> str:

        if (
            len(self.sampling_params.stop_strs) == 0
            and len(self.sampling_params.stop_regex_strs) == 0
        ):
            return ""

        tail_len = self._stop_match_tail_len(new_accepted_len)
        return self.tokenizer.decode(self.output_ids[-tail_len:])

    def check_match_stop_str_prefix(self) -> bool:
        if not self.sampling_params.stop_strs:
            return False

        tail_str = self.tail_str()


        if not tail_str:
            return False

        for stop_str in self.sampling_params.stop_strs:
            if not stop_str:
                continue

            if stop_str in tail_str:
                return True


            min_len = min(len(tail_str), len(stop_str))
            for i in range(1, min_len + 1):
                if tail_str[-i:] == stop_str[:i]:
                    return True

        return False

    def _check_token_based_finish(self, new_accepted_tokens: List[int]) -> bool:
        if self.sampling_params.ignore_eos:
            return False


        matched_eos = False

        for i, token_id in enumerate(new_accepted_tokens):
            if self.sampling_params.stop_token_ids:
                matched_eos |= token_id in self.sampling_params.stop_token_ids
            if self.eos_token_ids:
                matched_eos |= token_id in self.eos_token_ids
            if self.tokenizer is not None:
                matched_eos |= token_id == self.tokenizer.eos_token_id
                if self.tokenizer.additional_stop_token_ids:
                    matched_eos |= token_id in self.tokenizer.additional_stop_token_ids
            if matched_eos:
                self.finished_reason = FINISH_MATCHED_TOKEN(matched=token_id)
                matched_pos = len(self.output_ids) - len(new_accepted_tokens) + i
                self.finished_len = matched_pos + 1
                return True

        return False

    def _locate_str_stop_finished_len(
        self,
        new_accepted_len: int,
        *,
        stop_str: Optional[str] = None,
        stop_regex: Optional[str] = None,
    ) -> int:


        def matched(text: str) -> bool:
            if stop_str is not None:
                return stop_str in text
            return re.search(stop_regex, text) is not None

        tail_len = self._stop_match_tail_len(new_accepted_len)
        start = len(self.output_ids) - tail_len
        token_window = self.output_ids[start:]


        for token_count in range(
            max(1, len(token_window) - new_accepted_len + 1), len(token_window)
        ):
            if matched(self.tokenizer.decode(token_window[:token_count])):
                return start + token_count


        return len(self.output_ids)

    def _check_str_based_finish(self, new_accepted_len: int = 1):
        if (
            len(self.sampling_params.stop_strs) > 0
            or len(self.sampling_params.stop_regex_strs) > 0
        ):
            tail_str = self.tail_str(new_accepted_len)


            if len(self.sampling_params.stop_strs) > 0:
                for stop_str in self.sampling_params.stop_strs:
                    stop_str_in_tail = stop_str in tail_str
                    if stop_str_in_tail or stop_str in self.decoded_text:
                        self.finished_reason = FINISH_MATCHED_STR(matched=stop_str)
                        if stop_str_in_tail:
                            self.finished_len = self._locate_str_stop_finished_len(
                                new_accepted_len, stop_str=stop_str
                            )
                        return True


            if len(self.sampling_params.stop_regex_strs) > 0:
                for stop_regex_str in self.sampling_params.stop_regex_strs:


                    try:
                        matched = re.search(stop_regex_str, tail_str)
                    except (re.error, RecursionError) as e:
                        logger.warning(
                            "req %s: invalid stop_regex %r (%s); aborting the request",
                            self.rid,
                            stop_regex_str,
                            e,
                        )
                        self.finished_reason = FINISH_ABORT(
                            f"invalid stop_regex {stop_regex_str!r}: {e}",
                            HTTPStatus.BAD_REQUEST,
                            "BadRequestError",
                        )
                        break
                    if matched:
                        self.finished_reason = FINISHED_MATCHED_REGEX(
                            matched=stop_regex_str
                        )
                        self.finished_len = self._locate_str_stop_finished_len(
                            new_accepted_len, stop_regex=stop_regex_str
                        )
                        return True

        return False

    def _check_vocab_boundary_finish(self, new_accepted_tokens: List[int] = None):
        for i, token_id in enumerate(new_accepted_tokens):
            if token_id < 0 or (
                self.vocab_size is not None and token_id >= self.vocab_size
            ):
                offset = len(self.output_ids) - len(new_accepted_tokens) + i
                if self.sampling_params.stop_token_ids:
                    self.output_ids[offset] = next(
                        iter(self.sampling_params.stop_token_ids)
                    )
                if self.eos_token_ids:
                    self.output_ids[offset] = next(iter(self.eos_token_ids))
                self.finished_reason = FINISH_MATCHED_STR(matched="NaN happened")
                self.finished_len = offset + 1
                return True

        return False

    def _cap_finished_len_at_max_new_tokens(self) -> None:

        max_new_tokens = self.sampling_params.max_new_tokens
        if self.finished_len is not None and self.finished_len > max_new_tokens:
            self.finished_reason = FINISH_LENGTH(length=max_new_tokens)
            self.finished_len = max_new_tokens

    def update_finish_state(self, new_accepted_len: int = 1):
        if self.finished():
            return

        if self.to_finish:
            self.finished_reason = self.to_finish
            self.to_finish = None
            return

        new_accepted_tokens = self.output_ids[-new_accepted_len:]


        if self._check_vocab_boundary_finish(new_accepted_tokens):
            self._cap_finished_len_at_max_new_tokens()
            return


        if self._check_str_based_finish(new_accepted_len):
            self._cap_finished_len_at_max_new_tokens()
            return


        if self._check_token_based_finish(new_accepted_tokens):
            self._cap_finished_len_at_max_new_tokens()
            return

        if len(self.output_ids) >= self.sampling_params.max_new_tokens:
            self.finished_reason = FINISH_LENGTH(
                length=self.sampling_params.max_new_tokens
            )
            self.finished_len = self.sampling_params.max_new_tokens
            return

        if self.grammar is not None and self.grammar.is_terminated():
            self.finished_reason = FINISH_MATCHED_TOKEN(matched=self.output_ids[-1])
            return

    def reset_for_retract(self):


        self.retraction_count += 1

        self.prefix_indices = torch.empty((0,), dtype=torch.int64)
        self.routed_experts = None
        self.indexer_topk = None
        self.last_node = None
        self.cache_protected_len = 0
        self.num_matched_prefix_tokens = 0
        self.swa_uuid_for_lock = None
        self.swa_prefix_lock_released = False
        self.skip_lock_node_ids = {}
        self.extend_range = None
        self.dllm_initialized = False
        self.is_retracted = True
        self.retracted_stain = True
        self.input_token_logprobs = None
        self.temp_input_top_logprobs_val = None
        self.temp_input_top_logprobs_idx = None
        self.temp_input_token_ids_logprobs_val = None
        self.temp_input_token_ids_logprobs_idx = None
        self.inflight_middle_chunks = 0
        self.mamba_pool_idx = None
        self.mamba_ping_pong_track_buffer = None
        self.mamba_next_track_idx = None
        self.mamba_last_track_idx = None
        self.mamba_last_track_seqlen = None
        self.mamba_branching_seqlen = None
        self.mamba_cow_src_index = None
        self.mamba_needs_clear = False
        self.already_computed = 0
        assert self.kv is None, "expect it is already released"
        self.kv_committed_len = 0
        self.extend_batch_idx = 0
        self.decode_batch_idx = 0


        if self.input_embeds is not None:
            self.output_ids = array("q")
            self.weight_version_events = truncate_weight_version_events(
                self.weight_version_events, num_kept_tokens=self.send_token_offset
            )

    def _mamba_pool_needing_backup(self, req_to_token_pool, allocator):
        if allocator.get_kvcache().cpu_copy_carries_mamba:
            return None
        if not isinstance(req_to_token_pool, HybridReqToTokenPool):
            return None
        return req_to_token_pool.mamba_pool

    def offload_kv_cache(self, req_to_token_pool, token_to_kv_pool_allocator):
        token_indices = req_to_token_pool.req_to_token[
            self.req_pool_idx, : self.seqlen - 1
        ]

        mamba_pool = self._mamba_pool_needing_backup(
            req_to_token_pool, token_to_kv_pool_allocator
        )
        self.retraction_backup = RetractionBackup(
            cpu_tensors=token_to_kv_pool_allocator.get_cpu_copy(
                token_indices, mamba_indices=self.mamba_pool_idx
            ),
            mamba_cpu=(
                mamba_pool.get_cpu_copy(self.mamba_pool_idx.unsqueeze(0))
                if mamba_pool is not None and self.mamba_pool_idx is not None
                else None
            ),
        )

    def load_kv_cache(self, req_to_token_pool, token_to_kv_pool_allocator):
        assert self.retraction_backup is not None
        token_indices = req_to_token_pool.req_to_token[
            self.req_pool_idx, : self.seqlen - 1
        ]

        mamba_cpu = self.retraction_backup.mamba_cpu
        if mamba_cpu is not None and self.mamba_pool_idx is not None:
            req_to_token_pool.mamba_pool.load_cpu_copy(
                mamba_cpu, self.mamba_pool_idx.unsqueeze(0)
            )
        token_to_kv_pool_allocator.load_cpu_copy(
            self.retraction_backup.cpu_tensors,
            token_indices,
            mamba_indices=self.mamba_pool_idx,
        )
        self.retraction_backup = None

    def build_rebootstrap_payload(self) -> dict:


        sp = self.sampling_params
        return {
            "input_ids": [int(x) for x in self.origin_input_ids]
            + [int(x) for x in self.output_ids],
            "sampling_params": {
                "max_new_tokens": 1,
                "temperature": sp.temperature,
                "top_p": sp.top_p,
                "top_k": sp.top_k,
                "min_p": sp.min_p,
                "frequency_penalty": sp.frequency_penalty,
                "presence_penalty": sp.presence_penalty,
                "repetition_penalty": sp.repetition_penalty,
                "ignore_eos": sp.ignore_eos,
                "skip_special_tokens": sp.skip_special_tokens,
                "spaces_between_special_tokens": sp.spaces_between_special_tokens,
                "no_stop_trim": sp.no_stop_trim,
            },
            "return_logprob": False,
            "stream": False,
            "rid": self.rid,
            "bootstrap_host": self.bootstrap_host,
            "bootstrap_port": self.bootstrap_port,
            "bootstrap_room": self.bootstrap_room,
            "priority": self.priority,
            "extra_key": self.extra_key,
            "cache_salt": self.cache_salt,
            "routing_key": self.routing_key,
            "disagg_prefill_dp_rank": self.disagg_prefill_dp_rank,
        }

    def log_time_stats(self):

        if self.has_log_time_stats:
            return

        bootstrap_info = (
            f", bootstrap_room={self.bootstrap_room}"
            if self.bootstrap_room is not None
            else ""
        )
        prefix = (
            f"ReqTimeStats("
            f"rid={self.rid}{bootstrap_info}, "
            f"input_len={len(self.origin_input_ids)}, "
            f"cached_input_len={self.cached_tokens}, "
            f"output_len={len(self.output_ids)}, "
            f"attempts={self.prefill_attempt_count}, "
            f"type={self.time_stats.disagg_mode_str()})"
        )
        logger.info(f"{prefix}: {self.time_stats.convert_to_duration()}")
        self.has_log_time_stats = True

    def set_finish_with_abort(self, error_msg: str):
        if get_parallel().tp_rank == 0:
            logger.error(f"{error_msg}, {self.rid=}")
        self.multimodal_inputs = None
        self.grammar = None
        self.origin_input_ids = array(
            "q", [0]
        )
        self.return_logprob = False
        self.logprob_start_len = -1
        self.to_finish = FINISH_ABORT(
            error_msg, HTTPStatus.BAD_REQUEST, "BadRequestError"
        )

    def update_reasoning_tokens(self, token_id, think_end_ids):
        if self._is_reasoning_over:
            return

        if not isinstance(token_id, list):
            token_id = [token_id]

        if self._think_end_matcher is None:
            self._think_end_matcher = TokenSequenceMatcher(think_end_ids)

        matched = self._think_end_match_len
        for position, token in enumerate(token_id):
            matched = self._think_end_matcher.advance(matched, token)
            if matched == len(self._think_end_matcher):
                self.reasoning_tokens += position + 1
                self._is_reasoning_over = True
                return

        self._think_end_match_len = matched
        self.reasoning_tokens += len(token_id)

    def __repr__(self):
        return (
            f"Req(rid={self.rid}, "
            f"input_ids={self.origin_input_ids}, output_ids={self.output_ids}, "
            f"{self.grammar=}, "
            f"{self.sampling_params=})"
        )


class _MambaRadixCacheV2TrackEntry(NamedTuple):
    track_mask: bool
    track_index: int
    track_seqlen: int


def mamba_lazy_spec_in_window(
    req, mamba_track_interval: int, max_draft_tokens: int
) -> bool:

    seq_len = req.kv_committed_len
    window = 2 * max_draft_tokens
    return seq_len // mamba_track_interval != (seq_len + window) // mamba_track_interval


def set_mamba_track_indices_from_reqs(
    batch, track_positions: Optional[List[int]] = None
):

    req_to_token_pool = batch.req_to_token_pool
    all_buffers = req_to_token_pool.req_index_to_mamba_ping_pong_track_buffer_mapping[
        batch.req_pool_indices
    ]
    if track_positions is None:


        track_positions = [
            req.mamba_next_track_idx if req.mamba_next_track_idx is not None else 0
            for req in batch.reqs
        ]
    batch.mamba_track_buffer_indices = list(track_positions)
    idx = (
        torch.tensor(
            track_positions,
            dtype=torch.int64,
            pin_memory=True,
        )
        .unsqueeze(1)
        .to(device=all_buffers.device, non_blocking=True)
    )
    batch.mamba_track_indices = (
        torch.gather(all_buffers, 1, idx).squeeze(1).to(torch.int64)
    )


def release_req(
    *,
    req: Req,
    remaing_req_count: int,
    server_args: ServerArgs,
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
    tree_cache: BasePrefixCache,
    hisparse_coordinator: Optional[HiSparseCoordinator],
    offload_kv: bool = True,
) -> bool:

    if hisparse_coordinator is not None and not req.finished():
        hisparse_coordinator.retract_req(req)


    backup_saved = True
    if get_disagg().disaggregation_mode == "decode" and offload_kv:
        backup_saved = retraction_backup(
            req,
            tree_cache,
            req_to_token_pool,
            token_to_kv_pool_allocator,
            get_disagg().disaggregation_decode_retraction_backup,
        )

    release_kv_cache(req, tree_cache, is_insert=False)

    num_tokens = remaing_req_count * envs.SGLANG_RETRACT_DECODE_STEPS.get()
    evict_from_tree_cache(tree_cache, num_tokens)

    req.reset_for_retract()
    return backup_saved


def retract_all(
    *,
    reqs: List[Req],
    server_args: ServerArgs,
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
    tree_cache: BasePrefixCache,
    hisparse_coordinator: Optional[HiSparseCoordinator],
    offload_kv: bool = True,
) -> None:
    for idx in range(len(reqs)):
        release_req(
            req=reqs[idx],
            remaing_req_count=len(reqs) - idx,
            server_args=server_args,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            hisparse_coordinator=hisparse_coordinator,
            offload_kv=offload_kv,
        )


def compute_extend_logprob_start_len(
    *,
    logprob_start_len: int,
    prefix_len: int,
    extend_len: int,
    full_untruncated_fill_len: int,
) -> int:


    if logprob_start_len == -1:
        resolved_start = full_untruncated_fill_len
    else:

        resolved_start = max(logprob_start_len, prefix_len)
    return min(resolved_start - prefix_len, extend_len)


def _compute_chunked_req_next_prompt_token(
    chunked_req: Optional[Req],
    vocab_size: int,
) -> Optional[int]:

    if chunked_req is None:
        return None
    fill_len = chunked_req.extend_range.end
    origin_ids = chunked_req.origin_input_ids
    if fill_len >= len(origin_ids):
        return None
    if origin_ids[fill_len] < vocab_size:
        return int(origin_ids[fill_len])
    return None


@dataclasses.dataclass
class ScheduleBatch(ScheduleBatchDisaggregationDecodeMixin):


    reqs: List[Req]


    req_to_token_pool: ReqToTokenPool = None
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator = None
    tree_cache: BasePrefixCache = None


    model_config: ModelConfig = None
    enable_overlap: bool = False


    device: str = "cuda"


    hisparse_coordinator: Optional[HiSparseCoordinator] = None


    batch_is_full: bool = False


    chunked_req: Optional[Req] = None
    chunked_req_next_prompt_token: Optional[int] = None
    contains_last_prefill_chunk: bool = True


    inner_idle_batch: Optional[ScheduleBatch] = None

    decoding_reqs: List[Req] = None


    split_index: int = 0
    split_prefill_finished: bool = False
    split_forward_count: int = 1
    split_forward_batch: ForwardBatch = None


    req_pool_indices_cpu: torch.Tensor = None


    fpm_start_time: float = 0.0


    hicache_consumer_index: int = -1


    dp_cooperation_info: Optional[DPCooperationInfo] = None
    prefill_stats: Optional[PrefillStats] = None
    forward_iter: Optional[int] = None
    launch_ts: Optional[float] = None
    after_idle_gap: bool = False


    input_ids: torch.Tensor = None

    prefill_input_ids_cpu: Optional[torch.Tensor] = None
    mix_running_indices: Optional[torch.Tensor] = None
    input_embeds: torch.Tensor = None


    replace_embeds: Optional[torch.Tensor] = None
    replace_positions: Optional[torch.Tensor] = None


    ne_token_table: torch.Tensor = None


    ne_skip_token_table_update: torch.Tensor = None

    req_pool_indices: torch.Tensor = None
    seq_lens: torch.Tensor = None


    orig_seq_lens: torch.Tensor = None


    out_cache_loc: torch.Tensor = None


    out_cache_loc_dsv4: Optional[Any] = None


    mamba_track_indices: torch.Tensor = None


    mamba_track_buffer_indices: Optional[List[int]] = None
    mamba_track_mask: torch.Tensor = None
    mamba_track_seqlens: torch.Tensor = None
    mamba_track_mask_cpu: Optional[List[bool]] = None
    mamba_track_mask_next_cpu: Optional[List[bool]] = None
    mamba_decode_batch_idx_cpu: Optional[List[int]] = None


    mamba_lazy_spec_track_positions_cpu: Optional[List[int]] = None

    mamba_cow_src_indices: torch.Tensor = None
    mamba_cow_dst_indices: torch.Tensor = None
    mamba_clear_indices: torch.Tensor = None


    encoder_lens: Optional[torch.Tensor] = None
    encoder_out_cache_loc: Optional[torch.Tensor] = None


    extend_input_logprob_token_ids: Optional[torch.Tensor] = None


    forward_mode: ForwardMode = None
    global_forward_mode: Optional[ForwardMode] = None


    is_extend_in_batch: bool = False
    can_run_decode_cuda_graph: bool = False
    can_run_dp_prefill_cuda_graph: bool = False
    tbo_split_seq_index: Optional[int] = None


    recv_skipper_forward_mode: Optional[ForwardMode] = None
    spec_verify_tier_num_tokens: int = -1


    return_logprob: bool = False


    is_prefill_only: bool = False


    spec_algorithm: SpeculativeAlgorithm = None


    return_hidden_states: bool = False
    return_hidden_states_mode: CaptureHiddenMode = CaptureHiddenMode.NULL


    has_grammar: bool = False


    seq_lens_sum: int = None
    extend_num_tokens: Optional[int] = None


    dllm_config: Optional[DllmConfig] = None


    seq_lens_cpu: torch.Tensor = None


    multimodal_inputs: Optional[List] = None


    top_logprobs_nums: Optional[List[int]] = None
    token_ids_logprobs: Optional[List[List[int]]] = None


    encoder_cached: Optional[List[bool]] = None
    encoder_lens_cpu: Optional[List[int]] = None


    prefix_lens: List[int] = None
    extend_lens: List[int] = None
    extend_logprob_start_lens: List[int] = None


    global_num_tokens: Optional[List[int]] = None
    global_num_tokens_for_logprob: Optional[List[int]] = None
    global_spec_verify_tier_num_tokens: Optional[List[int]] = None


    sampling_info: SamplingBatchInfo = None


    spec_info: Optional[SpecInput] = None

    @classmethod
    def init_new(
        cls,
        reqs: List[Req],
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        tree_cache: BasePrefixCache,
        model_config: ModelConfig,
        enable_overlap: bool,
        spec_algorithm: SpeculativeAlgorithm,
        chunked_req: Optional[Req] = None,
        dllm_config: Optional[DllmConfig] = None,
    ):
        return_logprob = any(req.return_logprob for req in reqs)

        return_hidden_states_mode = get_batch_return_hidden_states_mode(reqs)

        batch = cls(
            reqs=reqs,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=model_config,
            enable_overlap=enable_overlap,
            return_logprob=return_logprob,
            has_grammar=any(req.grammar for req in reqs),
            device=req_to_token_pool.device,
            spec_algorithm=spec_algorithm,
            return_hidden_states=return_hidden_states_mode.need_capture(),
            return_hidden_states_mode=return_hidden_states_mode,
            is_prefill_only=all(req.is_prefill_only for req in reqs),
            chunked_req=chunked_req,
            chunked_req_next_prompt_token=_compute_chunked_req_next_prompt_token(
                chunked_req,
                model_config.vocab_size,
            ),
            dllm_config=dllm_config,
        )
        return batch

    def batch_size(self):
        return len(self.reqs)

    def is_empty(self):
        return len(self.reqs) == 0

    def is_dllm(self):
        return self.dllm_config is not None

    def grammar_needs_sync(self) -> bool:

        return self.has_grammar and not self.spec_algorithm.supports_grammar_overlap()

    def prepare_encoder_info_extend(
        self, input_ids: List[array[int]], seq_lens: List[int]
    ):
        _pin = is_pin_memory_available(self.device)
        encoder_lens_cpu = []
        encoder_cached = []

        for req in self.reqs:
            im = req.multimodal_inputs
            if im is None or im.num_image_tokens is None:

                encoder_lens_cpu.append(0)
                encoder_cached.append(True)
            else:
                encoder_lens_cpu.append(im.num_image_tokens)
                encoder_cached.append(
                    self.forward_mode.is_decode()
                    or len(req.prefix_indices) >= im.num_image_tokens
                )
        self.encoder_lens_cpu = encoder_lens_cpu
        self.encoder_cached = encoder_cached

        self.encoder_lens = torch.tensor(
            self.encoder_lens_cpu, dtype=torch.int64, pin_memory=_pin
        ).to(self.device, non_blocking=True)


        pt = 0
        decoder_out_cache_loc = []
        encoder_out_cache_loc = []
        extend_lens = self.extend_lens[:]
        prefix_lens = self.prefix_lens[:]
        for i, req in enumerate(self.reqs):
            encoder_len = self.encoder_lens_cpu[i]
            seq_lens[i] -= encoder_len

            if len(req.prefix_indices) < encoder_len:

                assert len(req.prefix_indices) == 0
                input_ids[i] = input_ids[i][encoder_len:]
                encoder_out_cache_loc.append(self.out_cache_loc[pt : pt + encoder_len])
                decoder_out_cache_loc.append(
                    self.out_cache_loc[pt + encoder_len : pt + req.extend_range.length]
                )
                extend_lens[i] -= encoder_len
                self.extend_num_tokens = self.extend_num_tokens - encoder_len
            else:
                decoder_out_cache_loc.append(
                    self.out_cache_loc[pt : pt + req.extend_range.length]
                )
                prefix_lens[i] -= encoder_len

            pt += req.extend_range.length
        self.extend_lens = extend_lens
        self.prefix_lens = prefix_lens


        self.prefill_input_ids_cpu = flatten_arrays_to_pinned_cpu(input_ids, _pin)
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int64, pin_memory=_pin).to(
            self.device, non_blocking=True
        )
        self.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)

        if not decoder_out_cache_loc:
            self.out_cache_loc = torch.zeros(0, dtype=torch.int64).to(
                self.device, non_blocking=True
            )
        else:
            self.out_cache_loc = torch.cat(decoder_out_cache_loc)

        if not encoder_out_cache_loc:
            self.encoder_out_cache_loc = torch.zeros(0, dtype=torch.int64).to(
                self.device, non_blocking=True
            )
        else:
            self.encoder_out_cache_loc = torch.cat(encoder_out_cache_loc)

        assert (
            len(self.out_cache_loc) == self.extend_num_tokens
        ), f"Expected {len(self.out_cache_loc)}, got {self.extend_num_tokens}"

        if self.extend_input_logprob_token_ids is not None:
            new_token_ids_parts = []
            offset = 0
            extend_logprob_start_lens = self.extend_logprob_start_lens[:]
            for i, req in enumerate(self.reqs):
                encoder_len = self.encoder_lens_cpu[i]
                old_start_len = extend_logprob_start_lens[i]
                old_contribution = req.extend_range.length - old_start_len

                if len(req.prefix_indices) < encoder_len:
                    tokens_to_strip = max(0, encoder_len - old_start_len)
                    new_token_ids_parts.append(
                        self.extend_input_logprob_token_ids[
                            offset + tokens_to_strip : offset + old_contribution
                        ]
                    )
                    extend_logprob_start_lens[i] = max(0, old_start_len - encoder_len)
                else:
                    new_token_ids_parts.append(
                        self.extend_input_logprob_token_ids[
                            offset : offset + old_contribution
                        ]
                    )

                offset += old_contribution
            self.extend_logprob_start_lens = extend_logprob_start_lens

            if new_token_ids_parts:
                self.extend_input_logprob_token_ids = torch.cat(new_token_ids_parts)
            else:
                self.extend_input_logprob_token_ids = None

        for i, req in enumerate(self.reqs):
            encoder_len = self.encoder_lens_cpu[i]
            if encoder_len == 0:
                continue
            if len(req.prefix_indices) < encoder_len:
                assert len(req.prefix_indices) == 0
                req.extend_range = req.extend_range._replace(
                    start=req.extend_range.start + encoder_len
                )
            req.logprob_start_len = max(req.logprob_start_len, encoder_len)

    def prepare_for_extend(self):
        self.forward_mode = ForwardMode.EXTEND

        if self.is_dllm():

            self.forward_mode = ForwardMode.DLLM_EXTEND


        reqs = self.reqs
        input_ids = [r.get_fill_ids()[len(r.prefix_indices) :] for r in reqs]
        extend_num_tokens = sum(len(ids) for ids in input_ids)
        seq_lens = [r.extend_range.end for r in reqs]
        orig_seq_lens = [max(r.extend_range.end, len(r.origin_input_ids)) for r in reqs]
        prefix_lens = [len(r.prefix_indices) for r in reqs]
        extend_lens = [r.extend_range.length for r in reqs]
        extend_logprob_start_lens = [
            compute_extend_logprob_start_len(
                logprob_start_len=r.logprob_start_len,
                prefix_len=prefix_lens[i],
                extend_len=extend_lens[i],
                full_untruncated_fill_len=len(r.full_untruncated_fill_ids),
            )
            for i, r in enumerate(reqs)
        ]

        _pin = is_pin_memory_available(self.device)


        pinned_input_ids = flatten_arrays_to_pinned_cpu(input_ids, _pin)
        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int64, pin_memory=_pin).to(
            self.device, non_blocking=True
        )
        seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        orig_seq_lens_tensor = torch.tensor(
            orig_seq_lens, dtype=torch.int32, pin_memory=_pin
        ).to(self.device, non_blocking=True)


        self.prefix_lens = prefix_lens
        self.extend_lens = extend_lens
        self.seq_lens = seq_lens_tensor
        self.seq_lens_cpu = seq_lens_cpu
        self.extend_num_tokens = extend_num_tokens


        out_cache_loc, req_pool_indices_tensor, req_pool_indices_cpu = alloc_for_extend(
            self
        )


        input_embeds = []
        all_replace_embeds: List[torch.Tensor] = []
        all_replace_positions: List[int] = []
        has_replace_embeds = False
        input_id_pointer = 0
        input_id_lens = [len(input_id) for input_id in input_ids]
        extend_input_logprob_token_ids = []
        multimodal_inputs = []
        mamba_track_mask_cpu = []
        mamba_track_indices_cpu = []
        mamba_track_seqlens_cpu = []

        for i, (req, seq_len, pre_len) in enumerate(zip(reqs, seq_lens, prefix_lens)):
            assert seq_len - pre_len == req.extend_range.length

            req.extend_batch_idx += 1


            req.kv_committed_len = seq_len


            if req.input_embeds is not None:


                input_embeds.extend(
                    req.input_embeds[pre_len : pre_len + req.extend_range.length]
                )

            if req.positional_embed_overrides is not None:


                embeds_to_add = []
                for embed_idx, pos in enumerate(
                    req.positional_embed_overrides.positions
                ):
                    extend_pos = pos - pre_len
                    if extend_pos < 0 or extend_pos >= req.extend_range.length:
                        continue
                    embeds_to_add.append((embed_idx, input_id_pointer + extend_pos))
                if embeds_to_add:
                    has_replace_embeds = True
                    indices, positions = zip(*embeds_to_add)
                    all_replace_embeds.append(
                        req.positional_embed_overrides.embeds[list(indices)]
                    )
                    all_replace_positions.extend(positions)
            input_id_pointer += input_id_lens[i]

            multimodal_inputs.append(req.multimodal_inputs)


            if not req.retracted_stain:
                new_cached = pre_len - req.already_computed
                req.cached_tokens += new_cached


                if not req._cache_breakdown_computed:


                    (
                        req.cached_tokens_device,
                        req.cached_tokens_host,
                        req.cached_tokens_storage,
                    ) = split_cached_prefix_by_tier(
                        prefix_len=len(req.prefix_indices),
                        host_hit_len=req.host_hit_length,
                        storage_hit_len=req.storage_hit_length,
                    )
                    req._cache_breakdown_computed = True

                req.already_computed = seq_len
            req.is_retracted = False

            if mamba_extra_buffer_enabled():
                track_entry = self._mamba_radix_cache_v2_req_prepare_for_extend(req)
                mamba_track_mask_cpu.append(track_entry.track_mask)
                mamba_track_indices_cpu.append(track_entry.track_index)
                mamba_track_seqlens_cpu.append(track_entry.track_seqlen)

            if self.return_logprob:


                global_start_idx, global_end_idx = (
                    len(req.prefix_indices),
                    req.extend_range.end,
                )
                if req.logprob_start_len == -1:
                    logprob_start_len = len(req.origin_input_ids)
                else:
                    logprob_start_len = req.logprob_start_len

                if global_start_idx < logprob_start_len:
                    global_start_idx = logprob_start_len

                logprob_token_ids = req.origin_input_ids[
                    global_start_idx + 1 : global_end_idx + 1
                ]
                extend_input_logprob_token_ids.extend(logprob_token_ids)


                extend_input_logprob_token_ids.extend(
                    [0]
                    * (
                        req.extend_range.length
                        - extend_logprob_start_lens[i]
                        - len(logprob_token_ids)
                    )
                )

        if self.return_logprob:
            extend_input_logprob_token_ids = torch.tensor(
                extend_input_logprob_token_ids
            )


            extend_input_logprob_token_ids.clamp_(0, self.model_config.vocab_size - 1)
        else:
            extend_input_logprob_token_ids = None

        if has_replace_embeds:
            replace_embeds_tensor = torch.cat(all_replace_embeds, dim=0).to(
                self.device, non_blocking=True
            )
            replace_positions_tensor = torch.tensor(
                all_replace_positions, dtype=torch.long, device=self.device
            )
        else:
            replace_embeds_tensor = None
            replace_positions_tensor = None

        self.input_ids = None
        self.prefill_input_ids_cpu = pinned_input_ids
        self.req_pool_indices = req_pool_indices_tensor
        self.req_pool_indices_cpu = req_pool_indices_cpu
        self.orig_seq_lens = orig_seq_lens_tensor
        self.out_cache_loc = out_cache_loc
        self.input_embeds = (
            torch.tensor(input_embeds, pin_memory=_pin).to(
                self.device, non_blocking=True
            )
            if input_embeds
            else None
        )
        self.replace_embeds = replace_embeds_tensor
        self.replace_positions = replace_positions_tensor
        for mm_input in multimodal_inputs:
            if mm_input is None:
                continue
            if isinstance(mm_input.vision_position_ids, torch.Tensor):
                mm_input.vision_position_ids = mm_input.vision_position_ids.to(
                    self.device, non_blocking=True
                )
            if isinstance(mm_input.visible_frame_counts, torch.Tensor):
                mm_input.visible_frame_counts = mm_input.visible_frame_counts.to(
                    self.device, non_blocking=True
                )
        self.multimodal_inputs = multimodal_inputs
        self.seq_lens_sum = sum(seq_lens)

        if self.return_logprob:
            self.top_logprobs_nums = [r.logprob.top_logprobs_num for r in reqs]
            self.token_ids_logprobs = [r.logprob.token_ids_logprob for r in reqs]

        self.extend_logprob_start_lens = extend_logprob_start_lens
        self.extend_input_logprob_token_ids = extend_input_logprob_token_ids

        if mamba_extra_buffer_enabled():
            self.mamba_track_indices = torch.tensor(
                mamba_track_indices_cpu,
                dtype=torch.int64,
                device=self.device,
            )
            self.mamba_track_mask = torch.tensor(
                mamba_track_mask_cpu,
                dtype=torch.bool,
                device=self.device,
            )
            self.mamba_track_seqlens = torch.tensor(
                mamba_track_seqlens_cpu,
                dtype=torch.int64,
                device=self.device,
            )


        if any(req.mamba_pool_idx is not None for req in reqs):
            self._collect_deferred_mamba_cow_and_clear(reqs)

        if self.model_config.is_encoder_decoder:
            self.prepare_encoder_info_extend(input_ids, seq_lens)


        self.sampling_info = SamplingBatchInfo.from_schedule_batch(
            self,
            self.model_config.vocab_size,
        )

    def _mamba_radix_cache_v2_req_prepare_for_extend(
        self,
        req: Req,
    ) -> _MambaRadixCacheV2TrackEntry:
        chunk_size = mamba_cache_chunk_size()


        checkpoint_grid = mamba_checkpoint_grid(self.tree_cache.page_size)

        def _force_track_h(i: int) -> int:

            assert (i - len(req.prefix_indices)) % chunk_size == 0, (
                f"The force track calculation only handles last-position or "
                f"unaligned seqlens, so it needs a chunk-aligned offset to "
                f"start from. But i={i} prefix_len={len(req.prefix_indices)} "
                f"chunk_size={chunk_size} checkpoint_grid={checkpoint_grid}"
            )


            return i + 1

        mask = req.extend_range.length >= checkpoint_grid
        track_index = req.mamba_ping_pong_track_buffer[req.mamba_next_track_idx].item()
        mamba_track_seqlen = -1
        if mask:


            mamba_track_seqlen = len(req.prefix_indices) + req.extend_range.length


            mamba_track_seqlen_aligned = (
                len(req.prefix_indices)
                + (req.extend_range.length // checkpoint_grid) * checkpoint_grid
            )


            mamba_track_fla_chunk_aligned = (
                len(req.prefix_indices)
                + (req.extend_range.length // chunk_size) * chunk_size
            )
            if mamba_track_fla_chunk_aligned != mamba_track_seqlen_aligned:


                mamba_track_seqlen = _force_track_h(mamba_track_seqlen_aligned)


            req.mamba_last_track_idx = req.mamba_next_track_idx
            if not mamba_extra_buffer_lazy_enabled():
                req.mamba_next_track_idx = (
                    self.req_to_token_pool.get_mamba_ping_pong_other_idx(
                        req.mamba_next_track_idx
                    )
                )
            if req.mamba_branching_seqlen is not None:


                branching_seqlen_aligned_mask = (
                    req.mamba_branching_seqlen - len(req.prefix_indices)
                ) % chunk_size == 0
                if (
                    req.mamba_branching_seqlen > len(req.prefix_indices)
                    and req.mamba_branching_seqlen < mamba_track_seqlen
                    and branching_seqlen_aligned_mask
                ):


                    mamba_track_seqlen = _force_track_h(req.mamba_branching_seqlen)
                    mamba_track_seqlen_aligned = req.mamba_branching_seqlen
            req.mamba_last_track_seqlen = mamba_track_seqlen_aligned

        return _MambaRadixCacheV2TrackEntry(
            track_mask=mask,
            track_index=track_index,
            track_seqlen=mamba_track_seqlen,
        )

    def _collect_deferred_mamba_cow_and_clear(self, reqs):

        cow_src_tensors = []
        cow_dst_tensors = []
        clear_tensors = []
        for req in reqs:
            if req.mamba_cow_src_index is not None:
                cow_src_tensors.append(req.mamba_cow_src_index)
                cow_dst_tensors.append(req.mamba_pool_idx.unsqueeze(0))
                req.mamba_cow_src_index = None
                req.mamba_needs_clear = False
            elif req.mamba_needs_clear:
                clear_tensors.append(req.mamba_pool_idx.unsqueeze(0))
                req.mamba_needs_clear = False
        self.mamba_cow_src_indices = (
            torch.cat(cow_src_tensors) if cow_src_tensors else None
        )
        self.mamba_cow_dst_indices = (
            torch.cat(cow_dst_tensors) if cow_dst_tensors else None
        )
        self.mamba_clear_indices = torch.cat(clear_tensors) if clear_tensors else None

    def prepare_for_split_prefill(self):
        self.prepare_for_extend()

        self.forward_mode = ForwardMode.SPLIT_PREFILL

    def mix_with_running(self, running_batch: ScheduleBatch):
        self.forward_mode = ForwardMode.MIXED
        running_bs = running_batch.batch_size()

        for req in running_batch.reqs:
            req._refresh_fill_ids()
            full_len = len(req.full_untruncated_fill_ids)
            req.set_extend_range(full_len - 1, full_len)


        self.input_ids = None
        self.mix_running_indices = running_batch.req_pool_indices
        out_cache_loc = torch.cat([self.out_cache_loc, running_batch.out_cache_loc])

        self.merge_batch(running_batch)
        self.out_cache_loc = out_cache_loc


        delta = 0 if self.enable_overlap else -1


        self.prefix_lens = self.prefix_lens + [
            len(r.origin_input_ids) + len(r.output_ids) + delta
            for r in running_batch.reqs
        ]
        self.extend_lens = self.extend_lens + [1] * running_bs
        self.extend_num_tokens = self.extend_num_tokens + running_bs

        self.extend_logprob_start_lens = (
            self.extend_logprob_start_lens + [0] * running_bs
        )
        self.is_prefill_only = False

    def new_tokens_required_next_decode(
        self, selected_indices: Optional[List[int]] = None
    ):
        page_size = self.token_to_kv_pool_allocator.page_size
        requests = (
            self.reqs
            if selected_indices is None
            else [self.reqs[i] for i in selected_indices]
        )

        if self.spec_algorithm.is_none():
            new_pages = sum(1 for r in requests if r.kv_committed_len % page_size == 0)
            return new_pages * page_size

        return self._new_tokens_required_next_decode_spec_v2(requests, page_size)

    def _new_tokens_required_next_decode_spec_v2(self, requests, page_size):

        reserve = get_alloc_reserve_per_decode()
        total = 0
        for r in requests:
            x = max(0, r.kv_committed_len + reserve - r.kv.kv_allocated_len)
            cur = r.kv.kv_allocated_len
            nxt = cur + x
            total += ceil_align(nxt, page_size) - ceil_align(cur, page_size)
        return total

    def check_decode_mem(self, selected_indices: Optional[List[int]] = None):

        num_tokens = self.new_tokens_required_next_decode(selected_indices)
        evict_from_tree_cache(self.tree_cache, num_tokens)
        return self.token_to_kv_pool_allocator.available_size() >= num_tokens

    def retract_decode(
        self, server_args: ServerArgs
    ) -> Tuple[List[Req], float, List[Req]]:

        sorted_indices = self._get_decode_retraction_order(self.reqs)

        retracted_reqs = []
        reqs_to_abort: List[Req] = []
        first_iter = True
        while first_iter or (
            not self.check_decode_mem(selected_indices=sorted_indices)
        ):
            if len(sorted_indices) == 1:

                break

            first_iter = False
            idx = sorted_indices.pop()
            req = self.reqs[idx]

            if self.release_req(idx, len(sorted_indices), server_args):
                retracted_reqs.append(req)
            else:


                req.to_finish = FINISH_ABORT(
                    "Retraction host KV pool exhausted. Aborting the request.",
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                reqs_to_abort.append(req)
                logger.warning(
                    "retract_decode: aborted request %s, retraction host pool "
                    "exhausted",
                    req.rid,
                )

        if len(sorted_indices) <= 1 and not self.check_decode_mem(
            selected_indices=sorted_indices
        ):


            last_idx = sorted_indices.pop()
            last_req = self.reqs[last_idx]
            last_req.to_finish = FINISH_ABORT(
                "Out of memory even after retracting all other requests "
                "in the decode batch. Aborting the last request.",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            reqs_to_abort.append(last_req)
            self.release_req(last_idx, 0, server_args, offload_kv=False)
            logger.warning(
                "retract_decode: aborted last request %s due to OOM", last_req.rid
            )

        self.filter_batch(keep_indices=sorted_indices)


        new_estimate_ratio = (
            NewTokenRatioTracker.estimate_new_token_ratio_after_retract(self.reqs)
        )

        return retracted_reqs, new_estimate_ratio, reqs_to_abort

    @staticmethod
    def _get_decode_retraction_order(reqs: List[Req]) -> List[int]:

        sorted_indices = list(range(len(reqs)))


        def length_key(req: Req) -> Tuple[int, int]:
            return (len(req.output_ids), -len(req.origin_input_ids))

        if get_schedule().retraction_policy == "priority":
            priority_sign = (
                1 if get_schedule().schedule_low_priority_values_first else -1
            )

            def retraction_key(req: Req) -> Tuple[int, int, int]:
                priority = req.priority
                if priority is None:
                    priority = (
                        sys.maxsize
                        if get_schedule().schedule_low_priority_values_first
                        else -sys.maxsize - 1
                    )
                return (priority * (-priority_sign), *length_key(req))

            sorted_indices.sort(
                key=lambda i: retraction_key(reqs[i]),
                reverse=True,
            )
            return sorted_indices

        sorted_indices.sort(
            key=lambda i: length_key(reqs[i]),
            reverse=True,
        )
        return sorted_indices

    def release_req(
        self,
        idx: int,
        remaing_req_count: int,
        server_args: ServerArgs,
        offload_kv: bool = True,
    ) -> bool:
        return release_req(
            req=self.reqs[idx],
            remaing_req_count=remaing_req_count,
            server_args=server_args,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            hisparse_coordinator=self.hisparse_coordinator,
            offload_kv=offload_kv,
        )

    def prepare_encoder_info_decode(self):

        self.encoder_cached = [True] * len(self.reqs)

    def prepare_for_idle(self):
        self.forward_mode = ForwardMode.IDLE
        self.input_ids = torch.empty(0, dtype=torch.int64, device=self.device)
        self.seq_lens = torch.empty(0, dtype=torch.int64, device=self.device)
        self.seq_lens_cpu = torch.empty(0, dtype=torch.int64)
        self.orig_seq_lens = torch.empty(0, dtype=torch.int32, device=self.device)
        self.out_cache_loc = torch.empty(0, dtype=torch.int64, device=self.device)
        self.req_pool_indices = torch.empty(0, dtype=torch.int64, device=self.device)
        self.req_pool_indices_cpu = torch.empty(0, dtype=torch.int64)
        self.seq_lens_sum = 0
        self.extend_num_tokens = 0
        self.sampling_info = SamplingBatchInfo.from_schedule_batch(
            self,
            self.model_config.vocab_size,
        )

    def mamba_lazy_prealloc_at_boundary(self, mamba_track_interval: int):

        pool = self.req_to_token_pool
        for i, req in enumerate(self.reqs):
            buf = req.mamba_ping_pong_track_buffer
            assert buf is not None

            if self.seq_lens_cpu[i].item() % mamba_track_interval != 0:
                continue
            other_idx = 1 - req.mamba_next_track_idx
            if buf[other_idx].item() != -1:


                continue
            if envs.SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL.get():
                new_slot = None
            else:


                new_slot = pool.mamba_allocator.alloc(1)
            if new_slot is not None:
                pool.set_mamba_ping_pong_slot(req, other_idx, new_slot[0])
                req.mamba_next_track_idx = other_idx

    def mamba_lazy_spec_prepare(self, mamba_track_interval: int, max_draft_tokens: int):

        pool = self.req_to_token_pool
        track_positions: List[int] = []
        for req in self.reqs:
            buf = req.mamba_ping_pong_track_buffer
            assert buf is not None
            if not mamba_lazy_spec_in_window(
                req, mamba_track_interval, max_draft_tokens
            ):


                track_positions.append(req.mamba_next_track_idx)
                continue
            other_idx = 1 - req.mamba_next_track_idx
            has_pending = buf[other_idx].item() != -1
            if not has_pending:
                if envs.SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL.get():
                    new_slot = None
                else:


                    new_slot = pool.mamba_allocator.alloc(1)
                if new_slot is not None:
                    pool.set_mamba_ping_pong_slot(req, other_idx, new_slot[0])
                    has_pending = True

            track_positions.append(
                other_idx if has_pending else req.mamba_next_track_idx
            )
        self.mamba_lazy_spec_track_positions_cpu = track_positions

    def cumulate_penalty_output_tokens(self):


        last_tokens = [
            req.output_ids[-1] if len(req.output_ids) else req.origin_input_ids[-1]
            for req in self.reqs
        ]


        latest_output_ids = torch.tensor(
            last_tokens,
            dtype=torch.int64,
            pin_memory=is_pin_memory_available(self.device),
        ).to(self.device, non_blocking=True)
        self.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
            latest_output_ids
        )

    def prepare_for_decode(self):
        self.forward_mode = ForwardMode.DECODE


        self.input_embeds = None


        if hasattr(self, "attn_cp_metadata") and self.attn_cp_metadata is not None:
            self.attn_cp_metadata = None

        if not self.spec_algorithm.is_none():

            from sglang.srt.speculative.spec_utils import spec_prepare_for_decode

            self.mamba_track_mask_cpu = None
            self.mamba_track_mask_next_cpu = None
            self.mamba_decode_batch_idx_cpu = None
            spec_prepare_for_decode(self)
            return

        if self.sampling_info.penalizer_orchestrator.is_required:
            self.cumulate_penalty_output_tokens()


        if self.model_config.is_encoder_decoder:
            self.prepare_encoder_info_decode()


        self.out_cache_loc = alloc_for_decode(self, token_per_req=1)


        for req in self.reqs:
            req.decode_batch_idx += 1
            req.kv_committed_len += 1


        self.seq_lens = self.seq_lens + 1
        self.seq_lens_cpu = self.seq_lens_cpu + 1
        self.orig_seq_lens = self.orig_seq_lens + 1

        self.seq_lens_sum = None

        if self.hisparse_coordinator is not None:
            self.hisparse_coordinator.map_last_loc_to_buffer(
                self.seq_lens,
                self.out_cache_loc,
                self.req_pool_indices,
                self.seq_lens_cpu,
                self.req_pool_indices_cpu,
            )

        if mamba_extra_buffer_enabled():
            mamba_track_interval = mamba_track_grid(self.tree_cache.page_size)

            if len(self.reqs) == 0:
                self.mamba_track_indices = torch.empty(
                    (0,), dtype=torch.int64, device=self.device
                )
                self.mamba_track_buffer_indices = []
            else:
                if mamba_extra_buffer_lazy_enabled():
                    self.mamba_lazy_prealloc_at_boundary(mamba_track_interval)
                set_mamba_track_indices_from_reqs(self)

            track_remainders_cpu = self.seq_lens_cpu % mamba_track_interval
            track_mask_cpu = track_remainders_cpu == 0
            self.mamba_track_mask_cpu = track_mask_cpu.tolist()
            self.mamba_track_mask_next_cpu = (
                (track_remainders_cpu == mamba_track_interval - 1).tolist()
                if self.enable_overlap
                else None
            )


            self.mamba_decode_batch_idx_cpu = [
                req.decode_batch_idx for req in self.reqs
            ]

            self.mamba_track_mask = track_mask_cpu.pin_memory().to(
                device=self.device, non_blocking=True
            )

    def filter_batch(
        self,
        chunked_req_to_exclude: Optional[Union[Req, List[Req]]] = None,
        keep_indices: Optional[List[int]] = None,
    ):
        if keep_indices is None:
            if isinstance(chunked_req_to_exclude, Req):
                chunked_req_to_exclude = [chunked_req_to_exclude]
            elif chunked_req_to_exclude is None:
                chunked_req_to_exclude = []
            keep_indices = [
                i
                for i in range(len(self.reqs))
                if not self.reqs[i].finished()
                and self.reqs[i] not in chunked_req_to_exclude
            ]

        if keep_indices is None or len(keep_indices) == 0:


            self.reqs = []
            self.return_hidden_states = False
            self.return_hidden_states_mode = CaptureHiddenMode.NULL
            return

        if len(keep_indices) == len(self.reqs):

            return

        keep_indices_device = torch.tensor(
            keep_indices,
            dtype=torch.int64,
            pin_memory=is_pin_memory_available(self.device),
        ).to(self.device, non_blocking=True)

        if self.model_config.is_encoder_decoder:
            self.encoder_lens = self.encoder_lens[keep_indices_device]
            self.encoder_lens_cpu = [self.encoder_lens_cpu[i] for i in keep_indices]

        self.reqs = [self.reqs[i] for i in keep_indices]
        if self.multimodal_inputs is not None:
            self.multimodal_inputs = [self.multimodal_inputs[i] for i in keep_indices]
        self.req_pool_indices = self.req_pool_indices[keep_indices_device]
        self.req_pool_indices_cpu = self.req_pool_indices_cpu[keep_indices]
        self.seq_lens = self.seq_lens[keep_indices_device]
        self.orig_seq_lens = self.orig_seq_lens[keep_indices_device]
        self.out_cache_loc = None

        self.seq_lens_sum = None

        if self.input_ids is not None:
            self.input_ids = self.input_ids[keep_indices_device]

        if self.seq_lens_cpu is not None:
            self.seq_lens_cpu = self.seq_lens_cpu[keep_indices]

        self.mamba_track_indices = None
        self.mamba_track_buffer_indices = None
        self.mamba_track_mask = None
        self.mamba_track_seqlens = None
        self.mamba_track_mask_cpu = None
        self.mamba_track_mask_next_cpu = None
        self.mamba_decode_batch_idx_cpu = None
        self.mamba_lazy_spec_track_positions_cpu = None
        self.mamba_cow_src_indices = None
        self.mamba_cow_dst_indices = None
        self.mamba_clear_indices = None
        self.return_logprob = any(req.return_logprob for req in self.reqs)
        if self.return_logprob:
            self.top_logprobs_nums = [self.top_logprobs_nums[i] for i in keep_indices]
            self.token_ids_logprobs = [self.token_ids_logprobs[i] for i in keep_indices]
        else:
            self.top_logprobs_nums = None
            self.token_ids_logprobs = None

        self.has_grammar = any(req.grammar for req in self.reqs)
        self.return_hidden_states_mode = get_batch_return_hidden_states_mode(self.reqs)
        self.return_hidden_states = self.return_hidden_states_mode.need_capture()

        self.sampling_info.filter_batch(keep_indices, keep_indices_device)
        if self.spec_info:
            self.spec_info.filter_batch(
                new_indices=keep_indices_device,
                new_indices_cpu=keep_indices,
            )

    def merge_batch(self, other: ScheduleBatch):


        self.sampling_info.merge_batch(other.sampling_info)


        if self.model_config.is_encoder_decoder:
            self.encoder_lens = torch.cat([self.encoder_lens, other.encoder_lens])
            self.encoder_lens_cpu = self.encoder_lens_cpu + other.encoder_lens_cpu
        self.req_pool_indices = torch.cat(
            [self.req_pool_indices, other.req_pool_indices]
        )
        self.req_pool_indices_cpu = torch.cat(
            [self.req_pool_indices_cpu, other.req_pool_indices_cpu]
        )
        self.seq_lens = torch.cat([self.seq_lens, other.seq_lens])
        self.orig_seq_lens = torch.cat([self.orig_seq_lens, other.orig_seq_lens])
        self.out_cache_loc = None

        self.seq_lens_sum = None


        if self.input_ids is not None and other.input_ids is not None:
            self.input_ids = torch.cat([self.input_ids, other.input_ids])
        else:
            self.input_ids = None

        if self.seq_lens_cpu is None or other.seq_lens_cpu is None:
            self.seq_lens_cpu = None
        else:
            self.seq_lens_cpu = torch.cat([self.seq_lens_cpu, other.seq_lens_cpu])
        self.mamba_track_indices = None
        self.mamba_track_buffer_indices = None
        self.mamba_track_mask = None
        self.mamba_track_seqlens = None
        self.mamba_track_mask_cpu = None
        self.mamba_track_mask_next_cpu = None
        self.mamba_decode_batch_idx_cpu = None
        self.mamba_lazy_spec_track_positions_cpu = None
        if self.return_logprob and other.return_logprob:
            self.top_logprobs_nums = self.top_logprobs_nums + other.top_logprobs_nums
            self.token_ids_logprobs = self.token_ids_logprobs + other.token_ids_logprobs
        elif self.return_logprob:
            self.top_logprobs_nums = self.top_logprobs_nums + [0] * len(other.reqs)
            self.token_ids_logprobs = self.token_ids_logprobs + [None] * len(other.reqs)
        elif other.return_logprob:
            self.top_logprobs_nums = [0] * len(self.reqs) + other.top_logprobs_nums
            self.token_ids_logprobs = [None] * len(self.reqs) + other.token_ids_logprobs
        self.reqs = self.reqs + other.reqs
        if self.multimodal_inputs is not None:
            self.multimodal_inputs = self.multimodal_inputs + other.multimodal_inputs

        self.return_logprob = self.return_logprob or other.return_logprob
        self.has_grammar = self.has_grammar or other.has_grammar
        self.return_hidden_states_mode = max(
            self.return_hidden_states_mode, other.return_hidden_states_mode
        )
        self.return_hidden_states = self.return_hidden_states_mode.need_capture()
        self.is_prefill_only = self.is_prefill_only and other.is_prefill_only

        if self.spec_info:
            self.spec_info.merge_batch(other.spec_info)

    def copy(self):


        return ScheduleBatch(
            reqs=self.reqs[:],
            extend_lens=self.extend_lens,
            prefix_lens=self.prefix_lens,
            req_to_token_pool=self.req_to_token_pool,
            req_pool_indices=self.req_pool_indices,
            model_config=self.model_config,
            forward_mode=self.forward_mode,
            out_cache_loc=self.out_cache_loc,
            return_logprob=self.return_logprob,
            has_grammar=self.has_grammar,
            return_hidden_states=self.return_hidden_states,
            return_hidden_states_mode=self.return_hidden_states_mode,
            decoding_reqs=self.decoding_reqs,
            spec_algorithm=self.spec_algorithm,
            spec_info=self.spec_info,
            global_num_tokens=self.global_num_tokens,
            global_num_tokens_for_logprob=self.global_num_tokens_for_logprob,
            can_run_decode_cuda_graph=self.can_run_decode_cuda_graph,
            can_run_dp_prefill_cuda_graph=self.can_run_dp_prefill_cuda_graph,
            is_extend_in_batch=self.is_extend_in_batch,
            is_prefill_only=self.is_prefill_only,
            seq_lens_cpu=self.seq_lens_cpu,
            enable_overlap=self.enable_overlap,
            mamba_track_indices=self.mamba_track_indices,
            mamba_track_buffer_indices=self.mamba_track_buffer_indices,
            mamba_track_mask=self.mamba_track_mask,
            mamba_track_seqlens=self.mamba_track_seqlens,
            mamba_track_mask_cpu=self.mamba_track_mask_cpu,
            mamba_track_mask_next_cpu=self.mamba_track_mask_next_cpu,
            mamba_decode_batch_idx_cpu=self.mamba_decode_batch_idx_cpu,
            mamba_lazy_spec_track_positions_cpu=self.mamba_lazy_spec_track_positions_cpu,
            dp_cooperation_info=self.dp_cooperation_info,
            prefill_stats=self.prefill_stats,
            fpm_start_time=self.fpm_start_time,
            forward_iter=self.forward_iter,
            launch_ts=self.launch_ts,
            after_idle_gap=self.after_idle_gap,
            extend_num_tokens=self.extend_num_tokens,
        )

    def maybe_evict_swa(self):
        if self.tree_cache.supports_swa():
            sliding_window_size = self.tree_cache.sliding_window_size

            release_leaf_lock = (
                envs.SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW.get()
                and hasattr(self.tree_cache, "dec_swa_lock_only")
            )

            eviction_interval = max(1, envs.SGLANG_SWA_EVICTION_INTERVAL.get())
            self.token_to_kv_pool_allocator.free_group_begin()
            for idx, req in enumerate(self.reqs):
                if self.forward_mode.is_decode():


                    if (
                        req.decode_batch_idx >= 1
                        and req.kv is not None
                        and req.seqlen - 1 - sliding_window_size
                        >= req.kv.swa_evicted_seqlen + eviction_interval
                    ):
                        self._evict_swa(req, req.seqlen - 1)


                    if (
                        release_leaf_lock
                        and not req.swa_prefix_lock_released
                        and req.swa_uuid_for_lock is not None
                        and req.last_node is not None
                        and req.decode_batch_idx >= sliding_window_size
                    ):
                        self.tree_cache.dec_swa_lock_only(
                            req.last_node,
                            req.swa_uuid_for_lock,
                            skip_lock_node_ids=req.skip_lock_node_ids,
                        )
                        req.swa_prefix_lock_released = True
                elif self.forward_mode.is_extend() and self.tree_cache.is_chunk_cache():
                    pre_len = self.prefix_lens[idx]
                    if self.enable_overlap:

                        if req.extend_batch_idx < 2:
                            continue
                        else:
                            pre_len = (
                                pre_len - get_schedule().chunked_prefill_size
                                if get_schedule().chunked_prefill_size > 0
                                else pre_len
                            )
                            self._evict_swa(req, pre_len)
                    else:
                        self._evict_swa(req, pre_len)
            self.token_to_kv_pool_allocator.free_group_end()

    def _evict_swa(self, req: Req, pre_len: int):
        assert self.tree_cache.supports_swa(), "prefix cache must support swa"
        free_swa_out_of_window_slots(
            req,
            pre_len,
            sliding_window_size=self.tree_cache.sliding_window_size,
            page_size=self.tree_cache.page_size,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            is_chunk_cache=self.tree_cache.is_chunk_cache(),
            retain_floor=self.tree_cache.swa_retain_floor(req),
        )

    def __str__(self):
        return (
            f"ScheduleBatch(forward_mode={self.forward_mode.name if self.forward_mode else 'None'}, "
            f"#req={(len(self.reqs))})"
        )


class NextBatchPlan(msgspec.Struct):
    batch_to_run: Optional[ScheduleBatch]
    running_batch: ScheduleBatch
