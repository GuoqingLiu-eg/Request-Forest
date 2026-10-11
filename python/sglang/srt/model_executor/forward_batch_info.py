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

import hashlib
import warnings
from dataclasses import dataclass
from enum import IntEnum, auto
from functools import total_ordering
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set, Tuple, Union

import torch

from sglang.kernels.ops.attention.position import compute_position_triton
from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.environ import envs
from sglang.srt.kv_canary.req_to_expected_token_ids_manager import (
    compute_req_all_ids_info,
)
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    set_dp_buffer_len,
    set_is_extend_in_batch,
    world_dp_gather_enabled,
)
from sglang.srt.model_executor.forward_batch_deepseek_mha_mixin import (
    ForwardBatchDeepSeekMHAMixin,
)
from sglang.srt.runtime_context import (
    get_exec,
    get_lora,
    get_parallel,
)
from sglang.srt.utils import (
    is_cpu,
    is_cuda,
    is_hip,
    is_npu,
    support_triton,
)
from sglang.srt.utils.common import ceil_align, is_pin_memory_available

if TYPE_CHECKING:
    from sglang.srt.layers.dcp.metadata import DecodeContextParallelMetadata
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.layers.utils.cp_utils import ContextParallelMetadata
    from sglang.srt.managers.schedule_batch import MultimodalInputs, ScheduleBatch
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
    from sglang.srt.speculative.spec_info import SpecInput, SpeculativeAlgorithm


_skip_attn_backend_init_warned = False

_is_npu = is_npu()
_is_cpu = is_cpu()


def _elastic_should_preserve_local_token_counts(
    *,
    model_runner: ModelRunner,
    dp_padding_mode: DpPaddingMode,
    global_num_tokens: List[int],
) -> bool:
    if not getattr(model_runner, "enable_elastic_ep", False):
        return False
    if not world_dp_gather_enabled():
        return False
    if not dp_padding_mode.is_max_len():
        return False
    if len(global_num_tokens) <= 1:
        return False

    uneven_token_count = len(set(global_num_tokens)) > 1
    return uneven_token_count


class ForwardMode(IntEnum):


    EXTEND = auto()

    DECODE = auto()

    MIXED = auto()

    IDLE = auto()


    TARGET_VERIFY = auto()

    DRAFT_EXTEND_V2 = auto()


    PREBUILT = auto()


    SPLIT_PREFILL = auto()


    DLLM_EXTEND = auto()

    def is_prefill(self, include_draft_extend_v2: bool = False):
        return self.is_extend(include_draft_extend_v2=include_draft_extend_v2)

    def is_extend(self, include_draft_extend_v2: bool = False):
        return (
            self == ForwardMode.EXTEND
            or self == ForwardMode.MIXED
            or (include_draft_extend_v2 and self == ForwardMode.DRAFT_EXTEND_V2)
            or self == ForwardMode.TARGET_VERIFY
            or self == ForwardMode.SPLIT_PREFILL
            or self == ForwardMode.DLLM_EXTEND
        )

    def is_context_parallel_extend(self, include_draft_extend_v2: bool = False):
        return (
            self == ForwardMode.EXTEND
            or self == ForwardMode.MIXED
            or (
                self == ForwardMode.DRAFT_EXTEND_V2
                if include_draft_extend_v2
                else False
            )
        )

    def is_decode(self):
        return self == ForwardMode.DECODE

    def is_mixed(self):
        return self == ForwardMode.MIXED

    def is_idle(self):
        return self == ForwardMode.IDLE

    def is_decode_or_idle(self):
        return self == ForwardMode.DECODE or self == ForwardMode.IDLE

    def is_target_verify(self):
        return self == ForwardMode.TARGET_VERIFY

    def is_draft_extend_v2(self):

        return self == ForwardMode.DRAFT_EXTEND_V2

    def is_extend_or_draft_extend_or_mixed(self, include_draft_extend_v2: bool = False):
        return (
            self == ForwardMode.EXTEND
            or self == ForwardMode.MIXED
            or self == ForwardMode.SPLIT_PREFILL
            or (include_draft_extend_v2 and self == ForwardMode.DRAFT_EXTEND_V2)
        )

    def is_cuda_graph(self):
        return (
            self == ForwardMode.DECODE
            or self == ForwardMode.TARGET_VERIFY
            or self == ForwardMode.IDLE
            or self == ForwardMode.DLLM_EXTEND
        )

    def is_cpu_graph(self):
        return self == ForwardMode.DECODE

    def is_split_prefill(self):
        return self == ForwardMode.SPLIT_PREFILL

    def is_extend_without_speculative(self):
        return self.is_extend() and not self.is_target_verify()

    def is_prebuilt(self):
        return self == ForwardMode.PREBUILT

    def is_dllm_extend(self):
        return self == ForwardMode.DLLM_EXTEND


@total_ordering
class CaptureHiddenMode(IntEnum):

    NULL = 0

    LAST = 1

    FULL = 2

    def need_capture(self):
        return self != CaptureHiddenMode.NULL

    def is_full(self):
        return self == CaptureHiddenMode.FULL

    def is_last(self):
        return self == CaptureHiddenMode.LAST

    def __lt__(self, other):
        return self.value < other.value


_attn_tp_sequence_sharded_predicate: Optional[Callable[[int], bool]] = None


def register_attn_tp_sequence_sharded_predicate(
    predicate: Callable[[int], bool],
) -> None:

    global _attn_tp_sequence_sharded_predicate
    _attn_tp_sequence_sharded_predicate = predicate


def get_server_return_hidden_states_mode() -> CaptureHiddenMode:
    features = get_exec().features
    mode = features.return_hidden_states_mode
    if mode == "last":
        return CaptureHiddenMode.LAST
    if mode == "full" or features.enable_return_hidden_states:
        return CaptureHiddenMode.FULL
    return CaptureHiddenMode.NULL


def get_required_capture_hidden_mode(
    capture_hidden_mode: CaptureHiddenMode,
    spec_info: Optional[SpecInput],
) -> CaptureHiddenMode:
    spec_capture_hidden_mode = (
        getattr(spec_info, "capture_hidden_mode", None) or CaptureHiddenMode.NULL
    )
    return max(capture_hidden_mode, spec_capture_hidden_mode)


def _attn_tp_local_shard_bounds(num_tokens_per_dp: int) -> Tuple[int, int]:

    predicate = _attn_tp_sequence_sharded_predicate
    if predicate is not None and not predicate(num_tokens_per_dp):
        return num_tokens_per_dp, 0
    parallel = get_parallel()
    tokens_per_rank = num_tokens_per_dp // parallel.attn_tp_size
    return tokens_per_rank, tokens_per_rank * parallel.attn_tp_rank


def compute_local_num_token_non_padded(
    global_num_token_non_padded: torch.Tensor,
    num_tokens_per_dp: int,
) -> torch.Tensor:

    tokens_per_rank, rank_offset = _attn_tp_local_shard_bounds(num_tokens_per_dp)
    return torch.clamp(
        global_num_token_non_padded - rank_offset,
        0,
        tokens_per_rank,
    )


def compute_local_num_token_non_padded_cpu(
    global_num_token_non_padded: int,
    num_tokens_per_dp: int,
) -> int:

    tokens_per_rank, rank_offset = _attn_tp_local_shard_bounds(num_tokens_per_dp)
    return min(max(global_num_token_non_padded - rank_offset, 0), tokens_per_rank)


@dataclass
class DSV4OutCacheLoc:


    out_full_loc: torch.Tensor
    out_swa_loc: torch.Tensor
    out_c4_loc: torch.Tensor
    out_c128_loc: torch.Tensor


@dataclass
class NgramEmbeddingInfo:


    token_table: torch.Tensor
    column_starts: torch.Tensor
    req_lens: torch.Tensor
    out_column_starts: torch.Tensor
    out_req_lens: torch.Tensor


    skip_token_table_update: Optional[torch.Tensor] = None

    @classmethod
    def create(
        cls,
        token_table: torch.Tensor,
        batch_size: int,
        device: torch.device,
        column_starts=None,
        req_lens=None,
        skip_token_table_update=None,
    ) -> NgramEmbeddingInfo:
        info = cls(
            token_table=token_table,
            column_starts=torch.empty(batch_size, dtype=torch.int32, device=device),
            req_lens=torch.empty(batch_size, dtype=torch.int32, device=device),
            out_column_starts=torch.empty(batch_size, dtype=torch.int32, device=device),
            out_req_lens=torch.empty(batch_size, dtype=torch.int32, device=device),
            skip_token_table_update=skip_token_table_update,
        )
        if column_starts is not None:
            info.column_starts[:] = column_starts
        if req_lens is not None:
            info.req_lens[:] = req_lens
        return info

    def slice(self, bs: int) -> NgramEmbeddingInfo:
        return NgramEmbeddingInfo(
            token_table=self.token_table,
            column_starts=self.column_starts[:bs],
            req_lens=self.req_lens[:bs],
            out_column_starts=self.out_column_starts[:bs],
            out_req_lens=self.out_req_lens[:bs],
            skip_token_table_update=(
                self.skip_token_table_update[:bs]
                if self.skip_token_table_update is not None
                else None
            ),
        )


@dataclass
class ForwardBatch(ForwardBatchDeepSeekMHAMixin):


    forward_mode: ForwardMode

    batch_size: int

    input_ids: torch.Tensor

    req_pool_indices: torch.Tensor

    seq_lens: torch.Tensor

    out_cache_loc: torch.Tensor

    seq_lens_sum: int


    orig_seq_lens: Optional[torch.Tensor] = None


    out_cache_loc_dsv4: Optional[DSV4OutCacheLoc] = None

    mamba_track_indices: Optional[torch.Tensor] = None

    mamba_track_mask: Optional[torch.Tensor] = None

    mamba_track_seqlens: Optional[torch.Tensor] = None

    mamba_cow_src_indices: Optional[torch.Tensor] = None
    mamba_cow_dst_indices: Optional[torch.Tensor] = None
    mamba_clear_indices: Optional[torch.Tensor] = None


    input_embeds: Optional[torch.Tensor] = None

    replace_embeds: Optional[torch.Tensor] = None
    replace_positions: Optional[torch.Tensor] = None


    token_type_ids: Optional[torch.Tensor] = None


    encoder_lens: Optional[torch.Tensor] = None
    encoder_out_cache_loc: Optional[torch.Tensor] = None


    return_logprob: bool = False

    is_prefill_only: bool = False
    spec_algorithm: SpeculativeAlgorithm = None

    dimensions: Optional[list[int]] = None

    return_pooled_hidden_states: bool = False


    is_extend_in_batch: bool = False
    can_run_decode_cuda_graph: bool = False
    can_run_dp_prefill_cuda_graph: bool = False
    global_forward_mode: Optional[ForwardMode] = None


    tbo_split_seq_index: Optional[int] = None


    seq_lens_cpu: Optional[torch.Tensor] = None


    top_logprobs_nums: Optional[List[int]] = None
    token_ids_logprobs: Optional[List[List[int]]] = None


    mm_inputs: Optional[List[MultimodalInputs]] = None


    encoder_cached: Optional[List[bool]] = None
    encoder_lens_cpu: Optional[List[int]] = None


    multi_item_delimiter_indices: Optional[List[torch.Tensor]] = None


    sampling_info: SamplingBatchInfo = None

    spec_info: Optional[SpecInput] = None


    lora_ids: Optional[List[str]] = None

    rids: Optional[List[str]] = None


    capture_hidden_mode: CaptureHiddenMode = None

    return_hidden_states_before_norm: bool = False


    reuse_dsa_topk_indices: Optional[bool] = False

    minimax_m3_precached_sparse_layers: Optional[Set[int]] = None


    positions: torch.Tensor = None


    extend_num_tokens: Optional[int] = None
    extend_seq_lens: Optional[torch.Tensor] = None
    extend_prefix_lens: Optional[torch.Tensor] = None
    extend_start_loc: Optional[torch.Tensor] = None
    extend_prefix_lens_cpu: Optional[List[int]] = None
    extend_seq_lens_cpu: Optional[List[int]] = None
    extend_logprob_start_lens_cpu: Optional[List[int]] = None
    extend_input_logprob_token_ids_gpu: Optional[torch.Tensor] = None


    original_global_num_tokens_cpu: Optional[List[int]] = None
    _original_batch_size: Optional[int] = None
    _original_forward_mode: Optional[ForwardMode] = None
    _original_num_tokens: Optional[int] = None
    global_num_tokens_cpu: Optional[List[int]] = None
    global_num_tokens_gpu: Optional[torch.Tensor] = None

    global_num_tokens_for_logprob_cpu: Optional[List[int]] = None
    global_num_tokens_for_logprob_gpu: Optional[torch.Tensor] = None

    num_token_non_padded: Optional[torch.Tensor] = None
    num_token_non_padded_cpu: int = None


    _attn_output: Optional[torch.Tensor] = None


    next_token_logits_buffer: torch.Tensor = None
    temperature: torch.Tensor = None
    top_p: torch.Tensor = None


    hidden_states: torch.Tensor = None
    residual: torch.Tensor = None
    model_specific_states: Dict[str, any] = None

    split_index: int = 0


    mm_input_embeds: Optional[torch.Tensor] = None


    cross_attention_custom_mask: Optional[torch.Tensor] = None


    dp_padding_mode: Optional[DpPaddingMode] = None


    dp_local_start_pos: Optional[torch.Tensor] = None
    dp_local_num_tokens: Optional[torch.Tensor] = None
    global_dp_buffer_len: Optional[int] = None


    mrope_positions: torch.Tensor = None


    tbo_parent_token_range: Optional[Tuple[int, int]] = None
    tbo_padded_len: Optional[int] = None
    tbo_children: Optional[List[ForwardBatch]] = None

    attn_cp_metadata: Optional[ContextParallelMetadata] = None


    attn_dcp_metadata: Optional[DecodeContextParallelMetadata] = None


    dcp_kv_mask: Optional[torch.Tensor] = None


    ngram_embedding_info: Optional[NgramEmbeddingInfo] = None


    rids_int: Optional[torch.Tensor] = None
    bootstrap_room_ids_int: Optional[torch.Tensor] = None


    req_all_ids_flat: Optional[torch.Tensor] = None
    req_all_ids_lens: Optional[torch.Tensor] = None


    forward_metadata_ready: bool = False


    forward_metadata_planned_bs: Optional[int] = None
    forward_metadata_planned_num_tokens: Optional[int] = None


    forward_metadata_replan_equivalent: bool = False

    def mark_forward_metadata_ready(self, replan_equivalent: bool = False):

        self.forward_metadata_ready = True
        self.forward_metadata_planned_bs = self.batch_size
        self.forward_metadata_planned_num_tokens = (
            self.input_ids.shape[0] if self.input_ids is not None else 0
        )
        self.forward_metadata_replan_equivalent = replan_equivalent

    def needs_forward_metadata_init(self) -> bool:

        if not self.forward_metadata_ready:
            return True
        if not self.forward_metadata_replan_equivalent:
            return False
        num_tokens = self.input_ids.shape[0] if self.input_ids is not None else 0
        return (
            self.batch_size != self.forward_metadata_planned_bs
            or num_tokens != self.forward_metadata_planned_num_tokens
        )

    def apply_deprecated_skip_attn_backend_init(
        self, skip_attn_backend_init: Optional[bool]
    ) -> None:

        if skip_attn_backend_init is None:
            return
        global _skip_attn_backend_init_warned
        if not _skip_attn_backend_init_warned:
            _skip_attn_backend_init_warned = True
            warnings.warn(
                "skip_attn_backend_init is deprecated and will be removed; "
                "pre-planners should call "
                "ForwardBatch.mark_forward_metadata_ready() after planning "
                "instead. The flag is mapped onto the marker for now.",
                DeprecationWarning,
                stacklevel=3,
            )
        if skip_attn_backend_init:
            self.mark_forward_metadata_ready()

    def init_mlp_sync_metadata(
        self, batch: ScheduleBatch, device: Union[str, torch.device]
    ) -> None:

        if batch.global_num_tokens is None:
            return

        assert batch.global_num_tokens_for_logprob is not None
        if self.spec_info is not None:
            from sglang.srt.speculative.spec_info import spec_scale_global_num_tokens

            global_num_tokens, global_num_tokens_for_logprob = (
                spec_scale_global_num_tokens(
                    self.spec_info,
                    batch.global_num_tokens,
                    batch.global_num_tokens_for_logprob,
                )
            )
        else:
            global_num_tokens = batch.global_num_tokens
            global_num_tokens_for_logprob = batch.global_num_tokens_for_logprob

        self.original_global_num_tokens_cpu = batch.global_num_tokens
        self.global_num_tokens_cpu = global_num_tokens
        self.global_num_tokens_gpu = torch.tensor(
            global_num_tokens, dtype=torch.int64
        ).to(device, non_blocking=True)
        self.global_num_tokens_for_logprob_cpu = global_num_tokens_for_logprob
        self.global_num_tokens_for_logprob_gpu = torch.tensor(
            global_num_tokens_for_logprob, dtype=torch.int64
        ).to(device, non_blocking=True)
        self.can_run_decode_cuda_graph = batch.can_run_decode_cuda_graph

    @classmethod
    def init_new(
        cls,
        batch: ScheduleBatch,
        model_runner: ModelRunner,
        *,
        capture_hidden_mode: Optional[CaptureHiddenMode] = None,
        return_hidden_states_before_norm: bool,
    ):


        if capture_hidden_mode is None:
            request_capture_hidden_mode = (
                CaptureHiddenMode.NULL
                if model_runner.is_draft_worker
                else max(
                    batch.return_hidden_states_mode,
                    get_server_return_hidden_states_mode(),
                )
            )
            capture_hidden_mode = get_required_capture_hidden_mode(
                request_capture_hidden_mode,
                batch.spec_info,
            )


        if batch.forward_mode.is_decode_or_idle():
            extend_seq_lens = extend_prefix_lens = extend_logprob_start_lens = None
        else:
            extend_seq_lens = batch.extend_lens
            extend_prefix_lens = batch.prefix_lens
            extend_logprob_start_lens = batch.extend_logprob_start_lens


        if batch.sampling_info is not None:
            if batch.has_grammar:
                batch.sampling_info.grammars = [req.grammar for req in batch.reqs]
            else:
                batch.sampling_info.grammars = None


        seq_lens_cpu = batch.seq_lens_cpu


        if batch.seq_lens_sum is None and seq_lens_cpu is not None:
            batch.seq_lens_sum = int(seq_lens_cpu.sum())

        ret = cls(

            forward_mode=batch.forward_mode,
            batch_size=len(batch.seq_lens),
            input_ids=batch.input_ids,
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            out_cache_loc=batch.out_cache_loc,
            seq_lens_sum=batch.seq_lens_sum,

            seq_lens_cpu=seq_lens_cpu,
            orig_seq_lens=batch.orig_seq_lens,
            out_cache_loc_dsv4=batch.out_cache_loc_dsv4,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_track_mask=batch.mamba_track_mask,
            mamba_track_seqlens=batch.mamba_track_seqlens,
            mamba_cow_src_indices=batch.mamba_cow_src_indices,
            mamba_cow_dst_indices=batch.mamba_cow_dst_indices,
            mamba_clear_indices=batch.mamba_clear_indices,
            encoder_lens=batch.encoder_lens,
            encoder_out_cache_loc=batch.encoder_out_cache_loc,
            input_embeds=batch.input_embeds,
            replace_embeds=batch.replace_embeds,
            replace_positions=batch.replace_positions,

            return_logprob=batch.return_logprob,
            is_extend_in_batch=batch.is_extend_in_batch,
            can_run_decode_cuda_graph=batch.can_run_decode_cuda_graph,
            can_run_dp_prefill_cuda_graph=batch.can_run_dp_prefill_cuda_graph,
            global_forward_mode=batch.global_forward_mode,
            is_prefill_only=batch.is_prefill_only,
            spec_algorithm=batch.spec_algorithm,
            capture_hidden_mode=capture_hidden_mode,
            return_hidden_states_before_norm=return_hidden_states_before_norm,
            tbo_split_seq_index=batch.tbo_split_seq_index,

            top_logprobs_nums=batch.top_logprobs_nums,
            token_ids_logprobs=batch.token_ids_logprobs,
            mm_inputs=batch.multimodal_inputs,
            encoder_cached=batch.encoder_cached,
            encoder_lens_cpu=batch.encoder_lens_cpu,
            lora_ids=[req.lora_id for req in batch.reqs],
            rids=[req.rid for req in batch.reqs],

            sampling_info=batch.sampling_info,
            spec_info=batch.spec_info,
        )

        ret._maybe_init_non_generation_fields(batch)

        device = model_runner.device

        if envs.SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE.get():
            hashed = _hash_rids_to_tensor(
                rids=[req.rid for req in batch.reqs],
                device=device,
            )
            bootstrap_room_ids = _bootstrap_rooms_to_tensor(
                bootstrap_rooms=[req.bootstrap_room for req in batch.reqs],
                device=device,
            )
            batch.sampling_info.rids_int = hashed
            batch.sampling_info.bootstrap_room_ids_int = bootstrap_room_ids
            ret.rids_int = hashed
            ret.bootstrap_room_ids_int = bootstrap_room_ids

        if envs.SGLANG_KV_CANARY_ENABLE_VERIFY_TOKEN_ASSERT.get():
            ret.req_all_ids_flat, ret.req_all_ids_lens = compute_req_all_ids_info(
                batch.reqs
            )

        if batch.extend_input_logprob_token_ids is not None:
            ret.extend_input_logprob_token_ids_gpu = (
                batch.extend_input_logprob_token_ids.to(device, non_blocking=True)
            )

        num_tokens = len(batch.input_ids) if batch.input_ids is not None else 0
        if enable_num_token_non_padded():
            ret.num_token_non_padded = torch.tensor(num_tokens, dtype=torch.int32).to(
                device, non_blocking=True
            )
        ret.num_token_non_padded_cpu = num_tokens

        ret.init_mlp_sync_metadata(batch, device)

        if ret.forward_mode.is_idle():
            ret.positions = torch.empty((0,), dtype=torch.int64, device=device)
            if model_runner.lora_manager is not None:
                model_runner.lora_manager.reset_lora_batch()
            return ret


        if batch.dllm_config is not None:
            block_size = batch.dllm_config.block_size

            positions_dtype = torch.int64 if is_hip() or _is_npu else torch.int32
            ret.positions = torch.tensor(
                [
                    i
                    for block_offset in (req.dllm_block_offset for req in batch.reqs)
                    for i in range(block_offset, block_offset + block_size)
                ],
                dtype=positions_dtype,
            ).to(device, non_blocking=True)
        elif (
            ret.spec_info is not None
            and getattr(ret.spec_info, "positions", None) is not None
        ):
            ret.positions = ret.spec_info.positions


        if ret.forward_mode.is_decode() or ret.forward_mode.is_target_verify():
            if ret.positions is None:
                ret.positions = clamp_position(batch.seq_lens)
        else:
            if isinstance(extend_seq_lens, list):

                assert isinstance(extend_prefix_lens, list)
                ret.extend_seq_lens = torch.tensor(
                    extend_seq_lens, dtype=torch.int32
                ).to(device, non_blocking=True)
                ret.extend_prefix_lens = torch.tensor(
                    extend_prefix_lens, dtype=torch.int32
                ).to(device, non_blocking=True)
                ret.extend_prefix_lens_cpu = extend_prefix_lens
                ret.extend_seq_lens_cpu = extend_seq_lens
            else:

                assert isinstance(extend_seq_lens, torch.Tensor)
                ret.extend_seq_lens = extend_seq_lens
                ret.extend_prefix_lens = extend_prefix_lens
            ret.extend_num_tokens = batch.extend_num_tokens
            positions, ret.extend_start_loc = compute_position(
                model_runner.prefill_attention_backend_str,
                ret.extend_prefix_lens,
                ret.extend_seq_lens,
                ret.extend_num_tokens,
            )
            if ret.positions is None:
                ret.positions = positions
            ret.extend_logprob_start_lens_cpu = extend_logprob_start_lens

        if model_runner.ngram_embedding_manager.enabled:
            ret._init_ngram_embedding_info(batch, device)

        if model_runner.model_config.model_is_mrope:
            if (
                ret.spec_info is not None
                and getattr(ret.spec_info, "positions", None) is not None
            ):
                ret.compute_spec_mrope_positions(model_runner, batch)
            elif ret.forward_mode.is_draft_extend_v2():


                ret.compute_spec_mrope_positions(
                    model_runner, batch, seq_positions=ret.positions
                )
            else:
                ret._compute_mrope_positions(model_runner, batch)


        if model_runner.lora_manager is not None:


            if not get_lora().enable_lora_overlap_loading:
                model_runner.lora_manager.fetch_new_loras(set(ret.lora_ids))

            model_runner.lora_manager.prepare_lora_batch(ret)

        if (
            model_runner.ps.attn_dcp_size > 1
            and ret.out_cache_loc is not None
            and is_hip()
        ):
            ret.dcp_kv_mask = (
                ret.positions % model_runner.ps.attn_dcp_size
                == model_runner.ps.attn_dcp_rank
            )

        return ret

    def _maybe_init_non_generation_fields(self, batch: ScheduleBatch):

        if self.is_prefill_only:
            if batch.model_config.is_matryoshka and any(
                r.dimensions is not None for r in batch.reqs
            ):
                self.dimensions = [
                    r.dimensions if r.dimensions else batch.model_config.hidden_size
                    for r in batch.reqs
                ]

            self.return_pooled_hidden_states = any(
                r.return_pooled_hidden_states for r in batch.reqs
            )


            if get_exec().features.enable_mis and any(
                r.multi_item_delimiter_indices is not None for r in batch.reqs
            ):
                assert all(
                    r.multi_item_delimiter_indices is not None for r in batch.reqs
                ), "MIS batch must have delimiter indices on every request"
                self.multi_item_delimiter_indices = [
                    torch.tensor(r.multi_item_delimiter_indices, dtype=torch.int64)
                    for r in batch.reqs
                ]

        token_type_ids = [
            r.token_type_ids for r in batch.reqs if r.token_type_ids is not None
        ]
        if token_type_ids:
            self.token_type_ids = torch.tensor(
                sum(token_type_ids, []),
                dtype=torch.int64,
                pin_memory=is_pin_memory_available(batch.device),
            ).to(batch.device, non_blocking=True)

    def adjust_num_token_non_padded_for_attn_tp(self) -> None:

        from sglang.srt.utils.common import require_mlp_tp_gather

        dp_rank = get_parallel().attn_dp_rank
        assert self.global_num_tokens_cpu is not None

        if require_mlp_tp_gather():
            num_tokens_per_dp = self.global_num_tokens_cpu[dp_rank]
        else:
            num_tokens_per_dp = self.global_num_tokens_cpu[0]

        self.num_token_non_padded = compute_local_num_token_non_padded(
            global_num_token_non_padded=self.num_token_non_padded,
            num_tokens_per_dp=num_tokens_per_dp,
        )

    def merge_mm_inputs(self) -> Optional[MultimodalInputs]:

        if not self.mm_inputs or all(x is None for x in self.mm_inputs):
            return None

        valid_inputs = [x for x in self.mm_inputs if x is not None]


        merged = valid_inputs[0].__class__(mm_items=[])


        for mm_input in valid_inputs:
            merged.merge(mm_input)

        return merged

    def contains_image_inputs(self) -> bool:
        if self.mm_inputs is None:
            return False
        return any(
            mm_input is not None and mm_input.contains_image_inputs()
            for mm_input in self.mm_inputs
        )

    def contains_audio_inputs(self) -> bool:
        if self.mm_inputs is None:
            return False
        return any(
            mm_input is not None and mm_input.contains_audio_inputs()
            for mm_input in self.mm_inputs
        )

    def contains_video_inputs(self) -> bool:
        if self.mm_inputs is None:
            return False
        return any(
            mm_input is not None and mm_input.contains_video_inputs()
            for mm_input in self.mm_inputs
        )

    def contains_mm_inputs(self) -> bool:
        return (
            self.contains_audio_inputs()
            or self.contains_video_inputs()
            or self.contains_image_inputs()
        )

    def _init_ngram_embedding_info(self, batch: ScheduleBatch, device: torch.device):
        if self.forward_mode.is_decode():
            column_starts, req_lens = self.seq_lens - 1, 1
        else:
            column_starts, req_lens = self.extend_prefix_lens, self.extend_seq_lens
        self.ngram_embedding_info = NgramEmbeddingInfo.create(
            batch.ne_token_table,
            self.batch_size,
            device,
            column_starts=column_starts,
            req_lens=req_lens,
            skip_token_table_update=batch.ne_skip_token_table_update,
        )

    def compute_spec_mrope_positions(
        self, model_runner: ModelRunner, batch: ScheduleBatch, seq_positions=None
    ):

        batch_size = self.seq_lens.shape[0]
        device = model_runner.device
        mm_inputs = batch.multimodal_inputs


        if seq_positions is None:
            seq_positions = batch.spec_info.positions
        seq_positions = seq_positions.view(batch_size, -1)

        if all(mm_input is None for mm_input in mm_inputs):
            mrope_delta_tensor = torch.zeros(
                (batch_size, 1), dtype=torch.int64, device=device
            )
        else:
            mrope_deltas = [
                (
                    torch.zeros(1, dtype=torch.int64)
                    if mm_inputs[i] is None
                    else mm_inputs[i].mrope_position_delta.squeeze(0)
                )
                for i in range(batch_size)
            ]
            mrope_delta_tensor = torch.stack(mrope_deltas, dim=0).to(device=device)
        next_input_positions = (
            (seq_positions + mrope_delta_tensor).flatten().unsqueeze(0).repeat(3, 1)
        )

        self.mrope_positions = next_input_positions

    def _expand_mrope_from_input(
        self,
        mm_input: MultimodalInputs,
        seq_len: int,
    ) -> torch.Tensor:


        if (
            mm_input.mrope_positions is not None
            and mm_input.mrope_positions.shape[1] >= seq_len
        ):
            pos = mm_input.mrope_positions[:, seq_len - 1 : seq_len]
            return pos


        if mm_input.mrope_position_delta_repeated_cache is None:
            mm_input.mrope_position_delta_repeated_cache = (
                (mm_input.mrope_position_delta - 1).flatten().unsqueeze(0).repeat(3, 1)
            )
        mrope_positions = mm_input.mrope_position_delta_repeated_cache + seq_len
        return mrope_positions

    def _compute_mrope_positions(self, model_runner: ModelRunner, batch: ScheduleBatch):


        forward_mode = self.forward_mode

        if forward_mode.is_decode():
            self._compute_mrope_positions_decode(model_runner, batch)
            return

        if not forward_mode.is_extend(include_draft_extend_v2=True):


            raise RuntimeError(
                f"_compute_mrope_positions called with unsupported forward_mode: {forward_mode}"
            )

        self._compute_mrope_positions_extend(model_runner, batch)

    def _compute_mrope_positions_decode(
        self, model_runner: ModelRunner, batch: ScheduleBatch
    ):
        seq_lens_cpu = self.seq_lens_cpu
        batch_size = seq_lens_cpu.shape[0]
        mm_inputs = batch.multimodal_inputs
        rl_on_policy_target = get_exec().deterministic.rl_on_policy_target
        seq_lens_int64 = self.seq_lens.to(torch.int64)


        has_precomputed_mrope = any(
            mm is not None
            and mm.mrope_positions is not None
            and mm.mrope_positions.shape[1] >= seq_lens_cpu[i]
            for i, mm in enumerate(mm_inputs)
        )
        has_multimodal_input = any(mm is not None for mm in mm_inputs)

        if rl_on_policy_target is not None or not has_multimodal_input:

            positions_1d = seq_lens_int64 - 1
            self.mrope_positions = positions_1d.unsqueeze(0).repeat(3, 1)
            return

        if not has_precomputed_mrope:

            deltas_list = [0] * batch_size
            for i, mm in enumerate(mm_inputs):
                deltas_list[i] = mm.mrope_position_delta.item() if mm is not None else 0
            deltas = torch.tensor(
                deltas_list, dtype=torch.int64, device=model_runner.device
            )
            positions_1d = (deltas - 1) + seq_lens_int64
            self.mrope_positions = positions_1d.unsqueeze(0).repeat(3, 1)
            return


        mrope_positions_list = [None] * batch_size
        for batch_idx in range(batch_size):
            mm_input = mm_inputs[batch_idx]
            if mm_input is None:

                mrope_positions = torch.full(
                    (3, 1),
                    seq_lens_cpu[batch_idx] - 1,
                    dtype=torch.int64,
                )
            else:
                mrope_positions = self._expand_mrope_from_input(
                    mm_input, seq_lens_cpu[batch_idx]
                )
            mrope_positions_list[batch_idx] = mrope_positions

        self.mrope_positions = torch.cat(
            mrope_positions_list,
            dim=1,
        ).to(
            dtype=torch.int64,
            device=model_runner.device,
            non_blocking=True,
        )

    def _compute_mrope_positions_extend(
        self, model_runner: ModelRunner, batch: ScheduleBatch
    ):
        seq_lens_cpu = self.seq_lens_cpu
        batch_size = seq_lens_cpu.shape[0]
        mm_inputs = batch.multimodal_inputs
        rl_on_policy_target = get_exec().deterministic.rl_on_policy_target
        extend_lens = batch.extend_lens
        prefix_lens = batch.prefix_lens


        mrope_positions_list = [None] * batch_size
        for batch_idx in range(batch_size):
            mm_input = mm_inputs[batch_idx]
            extend_seq_len = extend_lens[batch_idx]
            extend_prefix_len = prefix_lens[batch_idx]
            if mm_input is None or rl_on_policy_target is not None:

                mrope_positions = (
                    torch.arange(
                        extend_prefix_len,
                        extend_prefix_len + extend_seq_len,
                        dtype=torch.int64,
                    )
                    .unsqueeze(0)
                    .repeat(3, 1)
                )
            else:
                mrope_positions = mm_input.mrope_positions[
                    :,
                    extend_prefix_len : extend_prefix_len + extend_seq_len,
                ]
                if mrope_positions.numel() == 0:
                    mrope_positions = self._expand_mrope_from_input(
                        mm_input, seq_lens_cpu[batch_idx]
                    )
            mrope_positions_list[batch_idx] = mrope_positions

        self.mrope_positions = torch.cat(
            mrope_positions_list,
            dim=1,
        ).to(dtype=torch.int64, device=model_runner.device, non_blocking=True)

    def _pad_tensor_to_size(self, tensor: torch.Tensor, size: int, *, value: int = 0):
        if value == 0:
            return torch.cat(
                [tensor, tensor.new_zeros(size - tensor.shape[0], *tensor.shape[1:])],
                dim=0,
            )
        else:
            return torch.cat(
                [
                    tensor,
                    tensor.new_full((size - tensor.shape[0], *tensor.shape[1:]), value),
                ],
                dim=0,
            )

    def prepare_mlp_sync_batch(self, model_runner: ModelRunner):
        from sglang.srt.batch_overlap.two_batch_overlap import TboForwardBatchPreparer


        from sglang.srt.layers.cp.padding import get_cp_padding_align_size
        from sglang.srt.layers.cp.utils import enable_cp_v2

        assert self.global_num_tokens_cpu is not None
        assert self.global_num_tokens_for_logprob_cpu is not None

        self._original_batch_size = self.batch_size
        global_num_tokens = list(self.global_num_tokens_cpu)
        sync_group_size = len(global_num_tokens)
        attn_tp_size = get_parallel().attn_tp_size

        for i in range(sync_group_size):


            global_num_tokens[i] = ceil_align(global_num_tokens[i], attn_tp_size)


        if not enable_cp_v2():
            cp_align_size = get_cp_padding_align_size()
            for i in range(sync_group_size):
                global_num_tokens[i] = ceil_align(global_num_tokens[i], cp_align_size)

        dp_padding_mode = DpPaddingMode.get_dp_padding_mode(
            self.is_extend_in_batch, global_num_tokens
        )
        if _elastic_should_preserve_local_token_counts(
            model_runner=model_runner,
            dp_padding_mode=dp_padding_mode,
            global_num_tokens=global_num_tokens,
        ):

            dp_padding_mode = DpPaddingMode.SUM_LEN


        prefill_cg = get_exec().graph.cuda_graph_config.prefill
        if (
            self.can_run_dp_prefill_cuda_graph
            and self.is_extend_in_batch
            and prefill_cg.bs
            and max(global_num_tokens) <= max(prefill_cg.bs)
        ):
            dp_padding_mode = DpPaddingMode.MAX_LEN
        self.dp_padding_mode = dp_padding_mode

        if dp_padding_mode.is_max_len():


            max_num_tokens = max(global_num_tokens)
            global_num_tokens = [max_num_tokens] * sync_group_size
            buffer_len = max_num_tokens * sync_group_size
        else:
            buffer_len = sum(global_num_tokens)

        if len(global_num_tokens) > 1:
            num_tokens = global_num_tokens[get_parallel().attn_dp_rank]
        else:
            num_tokens = global_num_tokens[0]

        self.global_dp_buffer_len = buffer_len
        set_dp_buffer_len(
            buffer_len,
            num_tokens,
            dp_padding_mode.is_max_len(),
            global_num_tokens,
            self.global_num_tokens_gpu,
        )
        set_is_extend_in_batch(self.is_extend_in_batch)

        bs = self.batch_size

        if (
            self.forward_mode.is_decode()
            or self.forward_mode.is_target_verify()
            or self.forward_mode.is_draft_extend_v2()
            or self.forward_mode.is_idle()
        ):


            hybrid_ssm = mambaish_config(model_runner.model_config) is not None or (
                model_runner.is_draft_worker
                and getattr(
                    model_runner.model_config.hf_config,
                    "mtp_hybrid_override_pattern",
                    None,
                )
                is not None
            )
            if (
                hybrid_ssm
                and self.spec_info is not None
                and not self.spec_info.is_draft_input()
            ):
                if self.forward_mode.is_idle():
                    self._original_forward_mode = self.forward_mode
                    self.forward_mode = ForwardMode.TARGET_VERIFY

                bs = self.batch_size = num_tokens // self.spec_info.num_tokens_per_req
            elif self.is_extend_in_batch and dp_padding_mode.is_max_len():
                self._original_forward_mode = self.forward_mode
                self.forward_mode = ForwardMode.EXTEND


                if hybrid_ssm or self.seq_lens.shape[0] == 0:
                    dev = self.seq_lens.device
                    assert (
                        self.seq_lens.shape[0] == 0
                    ), "extend-idle conversion expects an empty rank"
                    self.extend_num_tokens = num_tokens
                    self.extend_seq_lens = torch.tensor(
                        [num_tokens], dtype=torch.int32, device=dev
                    )
                    self.extend_prefix_lens = torch.zeros(
                        1, dtype=self.seq_lens.dtype, device=dev
                    )
                    self.extend_start_loc = torch.zeros(
                        1, dtype=torch.int32, device=dev
                    )
                    self.seq_lens = torch.tensor(
                        [num_tokens], dtype=self.seq_lens.dtype, device=dev
                    )


                    self.orig_seq_lens = torch.tensor(
                        [num_tokens], dtype=self.orig_seq_lens.dtype, device=dev
                    )
                    self.seq_lens_sum = int(num_tokens)
                    if self.seq_lens_cpu is not None:
                        self.seq_lens_cpu = torch.tensor(
                            [num_tokens], dtype=self.seq_lens.dtype
                        )
                    self.extend_prefix_lens_cpu = [0]
                    self.extend_seq_lens_cpu = [int(num_tokens)]
                    self.extend_logprob_start_lens_cpu = [0]
                    bs = self.batch_size = 1


                    if self.num_token_non_padded is not None:
                        self.num_token_non_padded.fill_(num_tokens)
                    self.num_token_non_padded_cpu = num_tokens
                else:
                    self.extend_num_tokens = bs
                    self.extend_seq_lens = torch.full_like(self.seq_lens, 1)
                    self.extend_prefix_lens = self.seq_lens - 1
                    self.extend_start_loc = torch.arange(
                        bs, dtype=torch.int32, device=self.seq_lens.device
                    )
                    self.extend_prefix_lens_cpu = self.extend_prefix_lens.cpu().tolist()
                    self.extend_seq_lens_cpu = self.extend_seq_lens.cpu().tolist()
                    self.extend_logprob_start_lens_cpu = self.extend_prefix_lens_cpu
            else:
                if self.spec_info is not None:

                    bs = self.batch_size = (
                        num_tokens // self.spec_info.num_tokens_per_req
                    )
                else:
                    bs = self.batch_size = num_tokens
        elif self.forward_mode.is_extend():
            self.extend_num_tokens = num_tokens


        self._pad_inputs_to_size(model_runner, num_tokens, bs)
        self.global_num_tokens_cpu = global_num_tokens
        self.use_pin_memory = not _is_cpu
        global_num_tokens_pinned = torch.tensor(
            global_num_tokens, pin_memory=self.use_pin_memory
        )
        self.global_num_tokens_gpu.copy_(
            global_num_tokens_pinned, non_blocking=self.use_pin_memory
        )

        TboForwardBatchPreparer.prepare(
            batch=self, is_draft_worker=model_runner.is_draft_worker
        )


        if self.tbo_children:
            for child in self.tbo_children:
                child._pad_inputs_to_size(
                    model_runner, child.tbo_padded_len, child.batch_size
                )

    def _pad_inputs_to_size(self, model_runner: ModelRunner, num_tokens, bs):

        self._original_num_tokens = self.positions.shape[0]
        self.input_ids = self._pad_tensor_to_size(self.input_ids, num_tokens)
        if self.input_embeds is not None:

            self.input_embeds = self._pad_tensor_to_size(self.input_embeds, num_tokens)
        self.req_pool_indices = self._pad_tensor_to_size(self.req_pool_indices, bs)
        if self.lora_ids is not None:
            self.lora_ids.extend((bs - len(self.lora_ids)) * [None])

        seq_len_fill_value = (
            model_runner.attn_backend.get_cpu_graph_seq_len_fill_value()
            if _is_cpu
            else model_runner.attn_backend.get_cuda_graph_seq_len_fill_value()
        )


        if self.seq_lens_sum is not None:
            self.seq_lens_sum = self.seq_lens_sum + seq_len_fill_value * (
                bs - self.seq_lens.shape[0]
            )
        self.seq_lens = self._pad_tensor_to_size(
            self.seq_lens, bs, value=seq_len_fill_value
        )
        if self.seq_lens_cpu is not None:
            self.seq_lens_cpu = self._pad_tensor_to_size(
                self.seq_lens_cpu, bs, value=seq_len_fill_value
            )

        self.out_cache_loc = self._pad_tensor_to_size(self.out_cache_loc, num_tokens)
        if self.encoder_lens is not None:
            self.encoder_lens = self._pad_tensor_to_size(self.encoder_lens, bs)
        self.positions = self._pad_tensor_to_size(self.positions, num_tokens)
        if self.mamba_track_indices is not None:
            self.mamba_track_indices = self._pad_tensor_to_size(
                self.mamba_track_indices, bs
            )
        if self.mamba_track_mask is not None:
            self.mamba_track_mask = self._pad_tensor_to_size(self.mamba_track_mask, bs)
        if self.mamba_track_seqlens is not None:
            self.mamba_track_seqlens = self._pad_tensor_to_size(
                self.mamba_track_seqlens, bs
            )

        if self.mrope_positions is not None:
            self.mrope_positions = torch.cat(
                [
                    self.mrope_positions,
                    self.mrope_positions.new_zeros(
                        3, num_tokens - self.mrope_positions.shape[1]
                    ),
                ],
                dim=1,
            )


        dummy_extend_len = 0
        if (
            self.spec_info is not None
            and self.forward_mode.is_draft_extend_v2()
            and self.spec_info.num_tokens_per_req > 0
        ):
            dummy_extend_len = self.spec_info.num_tokens_per_req

        if self.extend_seq_lens is not None:
            self.extend_seq_lens = self._pad_tensor_to_size(
                self.extend_seq_lens, bs, value=dummy_extend_len
            )
        if self.extend_prefix_lens is not None:
            self.extend_prefix_lens = self._pad_tensor_to_size(
                self.extend_prefix_lens, bs
            )
        if self.extend_seq_lens_cpu is not None:
            self.extend_seq_lens_cpu.extend(
                [dummy_extend_len] * (bs - len(self.extend_seq_lens_cpu))
            )
        if self.extend_prefix_lens_cpu is not None:
            self.extend_prefix_lens_cpu.extend(
                [0] * (bs - len(self.extend_prefix_lens_cpu))
            )
        if self.extend_logprob_start_lens_cpu is not None:
            self.extend_logprob_start_lens_cpu.extend(
                [0] * (bs - len(self.extend_logprob_start_lens_cpu))
            )

        if self.rids_int is not None:
            self.rids_int = self._pad_tensor_to_size(self.rids_int, bs)
            if self.sampling_info is not None:
                self.sampling_info.rids_int = self.rids_int
        if self.bootstrap_room_ids_int is not None:
            self.bootstrap_room_ids_int = self._pad_tensor_to_size(
                self.bootstrap_room_ids_int, bs, value=-1
            )
            if self.sampling_info is not None:
                self.sampling_info.bootstrap_room_ids_int = self.bootstrap_room_ids_int

        if self.spec_info is not None and self.spec_info.is_draft_input():
            spec_info = self.spec_info
            self.output_cache_loc_backup = self.out_cache_loc
            self.hidden_states_backup = spec_info.hidden_states


            if getattr(spec_info, "topk_p", None) is not None:
                spec_info.topk_p = self._pad_tensor_to_size(spec_info.topk_p, bs)
            if getattr(spec_info, "topk_index", None) is not None:
                spec_info.topk_index = self._pad_tensor_to_size(
                    spec_info.topk_index, bs
                )
            if getattr(spec_info, "draft_probs", None) is not None:
                spec_info.draft_probs = self._pad_tensor_to_size(
                    spec_info.draft_probs, bs
                )
            if getattr(spec_info, "num_correct_drafts", None) is not None:
                spec_info.num_correct_drafts = self._pad_tensor_to_size(
                    spec_info.num_correct_drafts, bs
                )
                spec_info.num_accept_tokens = self._pad_tensor_to_size(
                    spec_info.num_accept_tokens, bs
                )
            if spec_info.hidden_states is not None:
                spec_info.hidden_states = self._pad_tensor_to_size(
                    spec_info.hidden_states, num_tokens
                )

    def prepare_attn_tp_scatter_input(self, model_runner: ModelRunner):
        from sglang.srt.layers.communicator import get_attn_tp_context

        attn_tp_context = get_attn_tp_context()
        input_scattered = attn_tp_context.use_input_scattered(self)
        if not input_scattered:
            return
        assert self.forward_mode.is_extend()
        tokens = self.input_ids.shape[0]
        rank_size = get_parallel().tp_size
        tokens_padded = (tokens + rank_size - 1) // rank_size * rank_size
        self._pad_inputs_to_size(model_runner, tokens_padded, self.batch_size)

    def post_forward_mlp_sync_batch(self, logits_output: LogitsProcessorOutput):
        if self._original_forward_mode is not None:
            self.forward_mode = self._original_forward_mode
        if self._original_batch_size is not None:
            self.batch_size = self._original_batch_size
        bs = self.batch_size


        if self.spec_info is None and self._original_num_tokens is not None:
            self.positions = self.positions[: self._original_num_tokens]
            self.seq_lens = self.seq_lens[:bs]
            self.req_pool_indices = self.req_pool_indices[:bs]
            if self.seq_lens_cpu is not None:
                self.seq_lens_cpu = self.seq_lens_cpu[:bs]

        if self.spec_info is not None:
            if self.forward_mode.is_decode():
                num_tokens = self.hidden_states_backup.shape[0]
                self.positions = self.positions[:num_tokens]
                self.seq_lens = self.seq_lens[:bs]
                self.req_pool_indices = self.req_pool_indices[:bs]
                if self.seq_lens_cpu is not None:
                    self.seq_lens_cpu = self.seq_lens_cpu[:bs]
                if logits_output.next_token_logits is not None:
                    logits_output.next_token_logits = logits_output.next_token_logits[
                        :num_tokens
                    ]
                logits_output.hidden_states = logits_output.hidden_states[:num_tokens]
            elif self.forward_mode.is_target_verify():
                ragged_layout = getattr(
                    self.spec_info, "ragged_verify_layout", None
                )
                num_tokens = (
                    ragged_layout.graph_num_tokens
                    if ragged_layout is not None
                    else bs * self.spec_info.num_tokens_per_req
                )
                if logits_output.next_token_logits is not None:
                    logits_output.next_token_logits = logits_output.next_token_logits[
                        :num_tokens
                    ]
                if logits_output.hidden_states is not None:
                    logits_output.hidden_states = logits_output.hidden_states[
                        :num_tokens
                    ]
            elif self.forward_mode.is_draft_extend_v2():
                bs = bs * self.spec_info.num_tokens_per_req
                if logits_output.next_token_logits is not None:
                    logits_output.next_token_logits = logits_output.next_token_logits[
                        :bs
                    ]
                logits_output.hidden_states = logits_output.hidden_states[:bs]
            elif self.forward_mode.is_extend() or self.forward_mode.is_idle():
                if logits_output.next_token_logits is not None:
                    logits_output.next_token_logits = logits_output.next_token_logits[
                        :bs
                    ]
                logits_output.hidden_states = logits_output.hidden_states[:bs]

            if hasattr(self, "hidden_states_backup"):
                self.spec_info.hidden_states = self.hidden_states_backup
            if hasattr(self, "output_cache_loc_backup"):
                self.out_cache_loc = self.output_cache_loc_backup

        elif self.forward_mode.is_decode() or self.forward_mode.is_idle():
            logits_output.next_token_logits = logits_output.next_token_logits[:bs]
            if logits_output.hidden_states is not None:
                logits_output.hidden_states = logits_output.hidden_states[:bs]
        elif self.forward_mode.is_extend():
            num_tokens = self.seq_lens_sum
            logits_output.next_token_logits = logits_output.next_token_logits[
                :num_tokens
            ]
            if logits_output.hidden_states is not None:
                logits_output.hidden_states = logits_output.hidden_states[:num_tokens]

    @property
    def can_run_tbo(self):
        return self.tbo_split_seq_index is not None


def enable_num_token_non_padded():
    return get_parallel().moe_ep_size > 1


def build_inner_fb_view(
    forward_batch: ForwardBatch,
    *,
    bs: int,
    forward_mode: ForwardMode,
    encoder_lens: Optional[torch.Tensor] = None,
):

    from types import SimpleNamespace

    return SimpleNamespace(
        batch_size=bs,
        forward_mode=forward_mode,
        actual_forward_mode=forward_batch.forward_mode,
        input_ids=getattr(forward_batch, "input_ids", None),
        positions=getattr(forward_batch, "positions", None),
        req_pool_indices=forward_batch.req_pool_indices,
        seq_lens=forward_batch.seq_lens,
        seq_lens_sum=forward_batch.seq_lens_sum,
        seq_lens_cpu=forward_batch.seq_lens_cpu,
        encoder_lens=encoder_lens,
        out_cache_loc=getattr(forward_batch, "out_cache_loc", None),
        out_cache_loc_dsv4=getattr(forward_batch, "out_cache_loc_dsv4", None),
        spec_info=forward_batch.spec_info,
    )


class PPProxyTensors:

    tensors: Dict[str, torch.Tensor]

    def __init__(self, tensors):


        self.tensors = tensors

    def __getitem__(self, key: Union[str, slice]):
        if isinstance(key, str):
            return self.tensors[key]
        elif isinstance(key, slice):
            return self.__class__({k: v[key] for k, v in self.tensors.items()})

    def __setitem__(self, key: str, value: torch.Tensor):
        self.tensors[key] = value

    def __len__(self):
        return len(self.tensors)

    def __eq__(self, other: object):
        return isinstance(other, self.__class__) and self

    def __repr__(self) -> str:
        return f"PPProxyTensors(tensors={self.tensors})"


def compute_position(
    attn_backend: str,
    extend_prefix_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    extend_seq_lens_sum: int,
):
    if support_triton(attn_backend):
        positions, extend_start_loc = compute_position_triton(
            extend_prefix_lens,
            extend_seq_lens,
            extend_seq_lens_sum,
        )
    else:
        positions, extend_start_loc = compute_position_torch(
            extend_prefix_lens, extend_seq_lens
        )
    return positions, extend_start_loc


def compute_position_torch(
    extend_prefix_lens: torch.Tensor, extend_seq_lens: torch.Tensor
):
    positions = torch.cat(
        [
            torch.arange(
                prefix_len, prefix_len + extend_len, device=extend_prefix_lens.device
            )
            for prefix_len, extend_len in zip(extend_prefix_lens, extend_seq_lens)
        ],
        axis=0,
    )
    extend_start_loc = torch.zeros_like(extend_seq_lens)
    extend_start_loc[1:] = torch.cumsum(extend_seq_lens[:-1], dim=0)
    return positions.to(torch.int64), extend_start_loc


def _clamp_position_native(seq_lens):
    return torch.clamp((seq_lens - 1), min=0).to(torch.int64)


if is_cuda() or is_hip():
    from sglang.kernels.ops.attention.clamp_position import clamp_position_cuda

    clamp_position = clamp_position_cuda
else:
    clamp_position = _clamp_position_native


def _hash_rids_to_tensor(*, rids: List[str], device: torch.device) -> torch.Tensor:
    values: List[int] = [_stable_hash_str_to_i64(rid) for rid in rids]
    return torch.tensor(values, dtype=torch.int64, device=device)


def _bootstrap_rooms_to_tensor(
    *, bootstrap_rooms: List[Optional[int]], device: torch.device
) -> torch.Tensor:
    values: List[int] = [room if room is not None else -1 for room in bootstrap_rooms]
    return torch.tensor(values, dtype=torch.int64, device=device)


def _stable_hash_str_to_i64(rid: str) -> int:
    digest = hashlib.blake2b(rid.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=True)
