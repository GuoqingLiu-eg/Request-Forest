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


import dataclasses
import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from sglang.kernels.ops.activation.softcap import (
    softcap_inplace_logits as fused_softcap,
)
from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.device_communicators import triton_symm_mem_ag
from sglang.srt.layers.aux_hidden_states import (
    AuxHiddenStates,
    pack_aux_hidden_states,
)
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    attn_tp_all_gather,
    attn_tp_all_gather_into_tensor,
    dp_gather_replicate,
    dp_scatter,
    get_dp_device,
    get_dp_dtype,
    get_dp_hidden_size,
)
from sglang.srt.layers.logprob_processor import (
    InputLogprobProcessor,
    LogprobStage,
    get_token_ids_logprobs_raw,
    get_top_logprobs_raw,
)
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.runtime_context import get_exec, get_parallel, get_spec
from sglang.srt.sampling.sampling_observer import DeviceAuxiliaryOutput
from sglang.srt.utils.common import (
    is_cpu,
    is_npu,
    is_pin_memory_available,
    require_gathered_buffer,
    use_intel_amx_backend,
)

logger = logging.getLogger(__name__)

_is_npu = is_npu()
_is_cpu = is_cpu()

_UNQUANTIZED_LM_HEAD_METHODS = {
    "UnquantizedEmbeddingMethod",
    "UnquantizedLinearMethod",
    "PackWeightMethod",
}


_autotune_run_lm_head: Optional[bool] = None


def get_in_autotune_dummy_run() -> bool:
    return _autotune_run_lm_head is not None


@contextmanager
def autotune_dummy_run_mode(*, run_lm_head: bool):
    global _autotune_run_lm_head
    _autotune_run_lm_head = run_lm_head
    try:
        yield
    finally:
        _autotune_run_lm_head = None


@dataclasses.dataclass
class LogitsProcessorOutput:


    next_token_logits: Optional[torch.Tensor]


    hidden_states: Optional[torch.Tensor] = None


    next_token_logprobs: Optional[torch.Tensor] = None

    next_token_top_logprobs_val: Optional[List] = None
    next_token_top_logprobs_idx: Optional[List] = None


    next_token_token_ids_logprobs_val: Optional[
        List[Union[List[float], torch.Tensor]]
    ] = None
    next_token_token_ids_logprobs_idx: Optional[List] = None


    next_token_sampling_mask_idx: Optional[List[Optional[List[int]]]] = None
    next_token_sampling_logprobs: Optional[List[Optional[float]]] = None


    input_token_logprobs: Optional[torch.Tensor] = None

    input_top_logprobs_val: Optional[List] = None
    input_top_logprobs_idx: Optional[List] = None


    input_token_ids_logprobs_val: Optional[List[Union[List[float], torch.Tensor]]] = (
        None
    )
    input_token_ids_logprobs_idx: Optional[List] = None


    full_logits: Optional[torch.Tensor] = None


    customized_info: Optional[Dict[str, List[Any]]] = None


    mm_input_embeds: Optional[torch.Tensor] = None


    auxiliary_device_output: Optional[DeviceAuxiliaryOutput] = None


    next_token_top1: Optional[torch.Tensor] = None


def _dflash_native_top1(forward_batch) -> bool:

    spec = forward_batch.spec_info
    if (
        spec is None
        or not forward_batch.forward_mode.is_target_verify()
        or not spec.is_dflash_verify_input()
        or getattr(getattr(spec, "capture_hidden_mode", None), "name", "") != "FULL"
        or forward_batch.return_logprob
    ):
        return False
    info = forward_batch.sampling_info
    if info is None:
        return True
    penalizer = getattr(info, "penalizer_orchestrator", None)
    return bool(
        info.is_all_greedy
        and not getattr(info, "has_custom_logit_processor", False)
        and getattr(info, "logit_bias", None) is None
        and getattr(info, "acc_linear_penalties", None) is None
        and getattr(info, "grammar_mask", None) is None
        and not (penalizer is not None and getattr(penalizer, "is_required", False))
    )


@dataclasses.dataclass
class LogitsMetadata:
    forward_mode: ForwardMode
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL
    next_token_logits_buffer: Optional[torch.Tensor] = None

    extend_return_logprob: bool = False
    extend_return_top_logprob: bool = False
    extend_token_ids_logprob: bool = False
    extend_seq_lens: Optional[torch.Tensor] = None
    extend_seq_lens_cpu: Optional[List[int]] = None
    extend_logprob_start_lens_cpu: Optional[List[int]] = None
    extend_logprob_pruned_lens_cpu: Optional[List[int]] = None
    top_logprobs_nums: Optional[List[int]] = None
    extend_input_logprob_token_ids_gpu: Optional[torch.Tensor] = None
    token_ids_logprobs: Optional[List[List[int]]] = None


    temperature: torch.Tensor = None
    top_p: torch.Tensor = None


    global_num_tokens_gpu: Optional[torch.Tensor] = None

    dp_local_start_pos: Optional[torch.Tensor] = None
    dp_local_num_tokens: Optional[torch.Tensor] = None
    global_dp_buffer_len: Optional[int] = None

    global_num_tokens_for_logprob_cpu: Optional[torch.Tensor] = None
    global_num_tokens_for_logprob_gpu: Optional[torch.Tensor] = None

    dp_padding_mode: Optional[DpPaddingMode] = None


    is_prefill_only: bool = False

    mm_input_embeds: Optional[torch.Tensor] = None


    draft_extend_select_index: Optional[torch.Tensor] = None

    request_forest_verify: bool = False
    dflash_target_top1: bool = False

    @classmethod
    def from_forward_batch(cls, forward_batch: ForwardBatch):
        if (
            forward_batch.forward_mode.is_extend()
            and forward_batch.return_logprob
            and not forward_batch.forward_mode.is_target_verify()
        ):
            extend_return_top_logprob = any(
                x > 0 for x in forward_batch.top_logprobs_nums
            )
            extend_token_ids_logprob = any(
                x is not None for x in forward_batch.token_ids_logprobs
            )
            extend_return_logprob = False
            extend_logprob_pruned_lens_cpu = []
            for extend_len, start_len in zip(
                forward_batch.extend_seq_lens_cpu,
                forward_batch.extend_logprob_start_lens_cpu,
            ):
                if extend_len - start_len > 0:
                    extend_return_logprob = True
                extend_logprob_pruned_lens_cpu.append(extend_len - start_len)
        else:
            extend_return_logprob = extend_return_top_logprob = (
                extend_token_ids_logprob
            ) = extend_logprob_pruned_lens_cpu = False

        if forward_batch.forward_mode.is_draft_extend_v2():
            draft_extend_select_index = forward_batch.spec_info.select_index
        else:
            draft_extend_select_index = None

        return cls(
            forward_mode=forward_batch.forward_mode,
            capture_hidden_mode=forward_batch.capture_hidden_mode,
            next_token_logits_buffer=forward_batch.next_token_logits_buffer,
            extend_return_logprob=extend_return_logprob,
            extend_return_top_logprob=extend_return_top_logprob,
            extend_token_ids_logprob=extend_token_ids_logprob,
            extend_seq_lens=forward_batch.extend_seq_lens,
            extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            extend_logprob_start_lens_cpu=forward_batch.extend_logprob_start_lens_cpu,
            extend_logprob_pruned_lens_cpu=extend_logprob_pruned_lens_cpu,
            top_logprobs_nums=forward_batch.top_logprobs_nums,
            token_ids_logprobs=forward_batch.token_ids_logprobs,
            extend_input_logprob_token_ids_gpu=forward_batch.extend_input_logprob_token_ids_gpu,
            is_prefill_only=forward_batch.is_prefill_only,
            global_num_tokens_gpu=forward_batch.global_num_tokens_gpu,
            dp_local_start_pos=forward_batch.dp_local_start_pos,
            dp_local_num_tokens=forward_batch.dp_local_num_tokens,
            global_dp_buffer_len=forward_batch.global_dp_buffer_len,
            global_num_tokens_for_logprob_cpu=forward_batch.global_num_tokens_for_logprob_cpu,
            global_num_tokens_for_logprob_gpu=forward_batch.global_num_tokens_for_logprob_gpu,
            dp_padding_mode=DpPaddingMode.SUM_LEN,
            mm_input_embeds=forward_batch.mm_input_embeds,
            draft_extend_select_index=draft_extend_select_index,
            request_forest_verify=(
                forward_batch.forward_mode.is_target_verify()
                and forward_batch.spec_info is not None
                and forward_batch.spec_info.is_request_forest_verify_input()
                and not forward_batch.return_logprob
            ),
            dflash_target_top1=_dflash_native_top1(forward_batch),
        )

    def compute_dp_attention_metadata(self):
        cumtokens = torch.cumsum(self.global_num_tokens_for_logprob_gpu, dim=0)
        dp_rank = get_parallel().attn_dp_rank
        if dp_rank == 0:
            dp_local_start_pos = torch.zeros_like(
                self.global_num_tokens_for_logprob_gpu[0]
            )
        else:
            dp_local_start_pos = cumtokens[dp_rank - 1]

        self.dp_local_start_pos = dp_local_start_pos
        self.dp_local_num_tokens = self.global_num_tokens_for_logprob_gpu[dp_rank]

        hidden_size = get_dp_hidden_size()
        dtype = get_dp_dtype()
        device = get_dp_device()

        if self.global_num_tokens_for_logprob_cpu is not None:

            self.global_dp_buffer_len = sum(self.global_num_tokens_for_logprob_cpu)
        else:
            self.global_dp_buffer_len = self.global_dp_buffer_len

        self.gathered_buffer = torch.empty(
            (
                self.global_dp_buffer_len,
                hidden_size,
            ),
            dtype=dtype,
            device=device,
        )


class LogitsProcessor(nn.Module):
    def __init__(
        self,
        config,
        skip_all_gather: bool = False,
        logit_scale: Optional[float] = None,
        return_full_logits: bool = False,
    ):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.logit_scale = logit_scale
        self.use_attn_tp_group = get_parallel().config.enable_dp_lm_head
        self.use_tp_lm_head_all_to_all = (
            get_parallel().config.enable_tp_lm_head_all_to_all
        )
        self.use_fp32_lm_head = get_exec().features.enable_fp32_lm_head
        if self.use_attn_tp_group:
            self.attn_tp_size = get_parallel().attn_tp_size
            self.do_tensor_parallel_all_gather = (
                not skip_all_gather and self.attn_tp_size > 1
            )
            self.do_tensor_parallel_all_gather_dp_attn = False
        else:
            self.do_tensor_parallel_all_gather = (
                not skip_all_gather and get_parallel().tp_size > 1
            )
            self.do_tensor_parallel_all_gather_dp_attn = (
                self.do_tensor_parallel_all_gather and get_parallel().attn_dp_size != 1
            )
        self.final_logit_softcapping = getattr(
            self.config, "final_logit_softcapping", None
        )
        if (
            self.final_logit_softcapping is not None
            and self.final_logit_softcapping < 0
        ):
            self.final_logit_softcapping = None

        self.return_full_logits = return_full_logits
        self.enable_mis = get_exec().features.enable_mis
        self.rl_on_policy_target = get_exec().deterministic.rl_on_policy_target

        self._logits_gatherer = triton_symm_mem_ag.MultimemAllGatherer(
            max_tokens=triton_symm_mem_ag.recommended_max_tokens(
                include_prefill=False, floor=128
            ),
            enabled=self.do_tensor_parallel_all_gather and not self.use_attn_tp_group,
            skip_entry_sync=True,
        )

        self.input_logprob_processor = InputLogprobProcessor()


        self._request_forest_top1_enabled = (
            os.environ.get("SGLANG_REQUEST_FOREST_TARGET_TOP1", "0") == "1"
            and get_spec().speculative_request_forest_branch_label_dir is None
        )


        self._request_forest_top1_check = (
            os.environ.get("SGLANG_REQUEST_FOREST_TARGET_TOP1_CHECK", "0") == "1"
        )
        self._request_forest_top1 = None

    def forward(
        self,
        input_ids,
        hidden_states,
        lm_head: VocabParallelEmbedding,
        logits_metadata: Union[LogitsMetadata, ForwardBatch],
        aux_hidden_states: Optional[AuxHiddenStates] = None,
        hidden_states_before_norm: Optional[torch.Tensor] = None,
    ) -> LogitsProcessorOutput:

        multi_item_delimiter_indices = None
        if isinstance(logits_metadata, ForwardBatch):
            multi_item_delimiter_indices = logits_metadata.multi_item_delimiter_indices
            logits_metadata = LogitsMetadata.from_forward_batch(logits_metadata)


        if _autotune_run_lm_head is False:
            return LogitsProcessorOutput(next_token_logits=None)


        if multi_item_delimiter_indices is not None and logits_metadata.is_prefill_only:
            return self.compute_logprobs_for_multi_item_scoring(
                input_ids,
                hidden_states,
                lm_head,
                logits_metadata,
                multi_item_delimiter_indices,
            )


        if logits_metadata.forward_mode.is_dllm_extend():
            return self._get_dllm_logits(hidden_states, lm_head, logits_metadata)


        (
            pruned_states,
            pruned_states_before_norm,
            aux_pruned_states,
            sample_indices,
            input_logprob_indices,
            token_to_seq_idx,
        ) = self._get_pruned_states(
            hidden_states,
            hidden_states_before_norm,
            aux_hidden_states,
            logits_metadata,
        )

        hidden_states_to_store = self._get_hidden_states_to_store(
            hidden_states,
            hidden_states_before_norm,
            aux_hidden_states,
            pruned_states,
            pruned_states_before_norm,
            aux_pruned_states,
            sample_indices,
            logits_metadata,
        )
        del hidden_states

        if not logits_metadata.extend_return_logprob:
            if self._can_use_request_forest_top1(lm_head, logits_metadata):
                top1 = self._get_request_forest_top1(pruned_states, lm_head)
                reference_logits = None
                if self._request_forest_top1_check:
                    reference_logits = self._get_logits(
                        pruned_states, lm_head, logits_metadata
                    )
                    if sample_indices is not None:
                        reference_logits = reference_logits[sample_indices]
                return LogitsProcessorOutput(
                    next_token_logits=reference_logits,
                    next_token_top1=(
                        top1[sample_indices] if sample_indices is not None else top1
                    ),
                    hidden_states=hidden_states_to_store,
                    mm_input_embeds=logits_metadata.mm_input_embeds,
                )

            logits = self._get_logits(pruned_states, lm_head, logits_metadata)
            sampled_logits = (
                logits[sample_indices] if sample_indices is not None else logits
            )


            return LogitsProcessorOutput(
                next_token_logits=sampled_logits,
                hidden_states=hidden_states_to_store,
                mm_input_embeds=logits_metadata.mm_input_embeds,
            )

        logprobs_result, sampled_logits = self.input_logprob_processor.forward(
            pruned_states=pruned_states,
            sample_indices=sample_indices,
            input_logprob_indices=input_logprob_indices,
            token_to_seq_idx=token_to_seq_idx,
            lm_head=lm_head,
            get_logits_fn=self._get_logits,
            logits_metadata=logits_metadata,
            skip_chunking_for_dp_attn=self.do_tensor_parallel_all_gather_dp_attn,
        )

        logits_output = LogitsProcessorOutput(
            next_token_logits=sampled_logits,
            hidden_states=hidden_states_to_store,
            mm_input_embeds=logits_metadata.mm_input_embeds,
        )
        logprobs_result.write_input_to(logits_output)
        return logits_output

    def _get_pruned_states(
        self,
        hidden_states: torch.Tensor,
        hidden_states_before_norm: Optional[torch.Tensor],
        aux_hidden_states: Optional[AuxHiddenStates],
        logits_metadata: LogitsMetadata,
    ):
        pruned_states_before_norm: Optional[torch.Tensor] = None
        aux_pruned_states = None
        token_to_seq_idx = []

        if (
            logits_metadata.forward_mode.is_decode_or_idle()
            or logits_metadata.forward_mode.is_target_verify()
            or logits_metadata.forward_mode.is_draft_extend_v2()
        ):
            if logits_metadata.draft_extend_select_index is not None:


                pruned_states = hidden_states[logits_metadata.draft_extend_select_index]
            else:
                pruned_states = hidden_states
            pruned_states_before_norm = hidden_states_before_norm
            if aux_hidden_states is not None:
                aux_pruned_states = (
                    aux_hidden_states
                    if isinstance(aux_hidden_states, torch.Tensor)
                    else [hidden for hidden in aux_hidden_states]
                )
            sample_indices = None
            input_logprob_indices = None

        elif (
            logits_metadata.forward_mode.is_extend()
            and not logits_metadata.extend_return_logprob
        ):

            last_index = torch.cumsum(logits_metadata.extend_seq_lens, dim=0) - 1
            pruned_states = hidden_states[last_index]
            if hidden_states_before_norm is not None:
                pruned_states_before_norm = hidden_states_before_norm[last_index]
            if aux_hidden_states is not None:
                aux_pruned_states = (
                    aux_hidden_states[last_index]
                    if isinstance(aux_hidden_states, torch.Tensor)
                    else [hidden[last_index] for hidden in aux_hidden_states]
                )
            sample_indices = None
            input_logprob_indices = None
        else:


            sample_index_pt = -1
            sample_indices = []
            input_logprob_indices_pt = 0
            input_logprob_indices = []
            pt, pruned_states_list, pruned_states_before_norm_list = 0, [], []
            is_packed_aux_hidden_states = isinstance(aux_hidden_states, torch.Tensor)
            aux_pruned_states_lists = None
            if aux_hidden_states is not None:
                aux_pruned_states_lists = (
                    []
                    if is_packed_aux_hidden_states
                    else [[] for _ in aux_hidden_states]
                )

            for idx, (extend_logprob_start_len, extend_len) in enumerate(
                zip(
                    logits_metadata.extend_logprob_start_lens_cpu,
                    logits_metadata.extend_seq_lens_cpu,
                )
            ):


                if extend_len == extend_logprob_start_len:
                    start_len = extend_logprob_start_len - 1
                else:
                    start_len = extend_logprob_start_len


                assert extend_len > start_len
                pruned_states_list.append(
                    hidden_states[pt + start_len : pt + extend_len]
                )
                if hidden_states_before_norm is not None:
                    pruned_states_before_norm_list.append(
                        hidden_states_before_norm[pt + start_len : pt + extend_len]
                    )
                if aux_pruned_states_lists is not None:
                    if is_packed_aux_hidden_states:
                        aux_pruned_states_lists.append(
                            aux_hidden_states[pt + start_len : pt + extend_len]
                        )
                    else:
                        for j, hidden in enumerate(aux_hidden_states):
                            aux_pruned_states_lists[j].append(
                                hidden[pt + start_len : pt + extend_len]
                            )


                token_to_seq_idx.extend([idx] * (extend_len - start_len))
                pt += extend_len
                sample_index_pt += extend_len - start_len
                sample_indices.append(sample_index_pt)
                input_logprob_indices.extend(
                    [
                        input_logprob_indices_pt + i
                        for i in range(extend_len - extend_logprob_start_len)
                    ]
                )
                input_logprob_indices_pt += extend_len - start_len

            pruned_states = torch.cat(pruned_states_list)
            if hidden_states_before_norm is not None:
                pruned_states_before_norm = torch.cat(pruned_states_before_norm_list)
            if aux_pruned_states_lists is not None:
                aux_pruned_states = (
                    torch.cat(aux_pruned_states_lists)
                    if is_packed_aux_hidden_states
                    else [torch.cat(lst) for lst in aux_pruned_states_lists]
                )


            sample_indices = torch.tensor(
                sample_indices,
                dtype=torch.int64,
                pin_memory=is_pin_memory_available(),
            ).to(pruned_states.device, non_blocking=True)
            input_logprob_indices = torch.tensor(
                input_logprob_indices,
                dtype=torch.int64,
                pin_memory=is_pin_memory_available(),
            ).to(pruned_states.device, non_blocking=True)

        return (
            pruned_states,
            pruned_states_before_norm,
            aux_pruned_states,
            sample_indices,
            input_logprob_indices,
            token_to_seq_idx,
        )

    def _get_hidden_states_to_store(
        self,
        hidden_states: torch.Tensor,
        hidden_states_before_norm: Optional[torch.Tensor],
        aux_hidden_states: Optional[AuxHiddenStates],
        pruned_states: torch.Tensor,
        pruned_states_before_norm: Optional[torch.Tensor],
        aux_pruned_states: Optional[AuxHiddenStates],
        sample_indices: Optional[torch.Tensor],
        logits_metadata: LogitsMetadata,
    ) -> Optional[torch.Tensor]:
        hidden_states_to_store: Optional[torch.Tensor] = None
        hidden_states_to_store_before_norm: Optional[torch.Tensor] = None
        if logits_metadata.capture_hidden_mode.need_capture():
            if logits_metadata.capture_hidden_mode.is_full():
                if aux_hidden_states is not None:
                    hidden_states_to_store = pack_aux_hidden_states(aux_hidden_states)
                else:
                    hidden_states_to_store = hidden_states
                hidden_states_to_store_before_norm = hidden_states_before_norm
            elif logits_metadata.capture_hidden_mode.is_last():


                if aux_hidden_states is not None:
                    assert aux_pruned_states is not None
                    aux_pruned_states = pack_aux_hidden_states(aux_pruned_states)
                    hidden_states_to_store = (
                        aux_pruned_states[sample_indices]
                        if sample_indices is not None
                        else aux_pruned_states
                    )
                else:
                    hidden_states_to_store = (
                        pruned_states[sample_indices]
                        if sample_indices is not None
                        else pruned_states
                    )
                    if hidden_states_before_norm is not None:
                        hidden_states_to_store_before_norm = (
                            pruned_states_before_norm[sample_indices]
                            if sample_indices is not None
                            else pruned_states_before_norm
                        )
            else:
                assert False, "Should never reach"

        if hidden_states_to_store_before_norm is not None:


            hidden_states_to_store = hidden_states_to_store_before_norm

        return hidden_states_to_store

    def _can_use_request_forest_top1(
        self, lm_head: VocabParallelEmbedding, logits_metadata: LogitsMetadata
    ) -> bool:


        return (
            self._request_forest_top1_enabled
            and (
                logits_metadata.request_forest_verify
                or logits_metadata.dflash_target_top1
            )
            and not self.return_full_logits
            and not self.final_logit_softcapping
            and self.do_tensor_parallel_all_gather
            and not self.use_attn_tp_group
            and not self.do_tensor_parallel_all_gather_dp_attn


            and not require_gathered_buffer()
            and get_parallel().tp_size == 4
            and isinstance(lm_head, VocabParallelEmbedding)
            and getattr(lm_head, "tp_size", None) == 4
            and getattr(lm_head, "num_added_embeddings", -1) == 0
            and lm_head.org_vocab_size == self.vocab_size
            and hasattr(lm_head, "shard_indices")
            and not hasattr(lm_head, "apply_lora")
            and self.vocab_size > 3 * lm_head.num_embeddings_per_partition
        )

    def _get_request_forest_top1(
        self, hidden_states: torch.Tensor, lm_head: VocabParallelEmbedding
    ) -> torch.Tensor:
        from sglang.srt.layers.vocab_parallel_top1 import VocabParallelTop1


        logits = self._compute_lm_head(hidden_states, lm_head)
        if self.logit_scale is not None:
            logits.mul_(self.logit_scale)


        if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            logits = logits.float()
        if self._request_forest_top1 is None:
            self._request_forest_top1 = VocabParallelTop1(get_tp_group())
        shard = lm_head.shard_indices
        return self._request_forest_top1(
            logits,
            num_valid=shard.num_org_elements,
            vocab_start=shard.org_vocab_start_index,
        )

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        logits_metadata: LogitsMetadata,
        embedding_bias: Optional[torch.Tensor] = None,
        use_logits_buffer: bool = True,
    ) -> torch.Tensor:

        hidden_states, local_hidden_states = self._gather_dp_attn_hidden_states(
            hidden_states, logits_metadata
        )

        logits = self._compute_lm_head(hidden_states, lm_head, embedding_bias)

        if self.logit_scale is not None:
            logits.mul_(self.logit_scale)

        used_tp_lm_head_all_to_all = False
        if self.do_tensor_parallel_all_gather:
            if self.use_attn_tp_group:
                logits = self._gather_attn_tp_logits(logits)
            elif self._can_use_tp_lm_head_all_to_all(
                logits, local_hidden_states, lm_head, logits_metadata
            ):
                logits = self._tp_lm_head_all_to_all(logits)
                used_tp_lm_head_all_to_all = True
            else:
                logits = self._logits_gatherer(logits)

        if not used_tp_lm_head_all_to_all:
            logits = self._scatter_dp_attn_logits(
                logits, local_hidden_states, logits_metadata
            )

        logits = self._copy_logits_to_buffer(
            logits, logits_metadata, use_buffer=use_logits_buffer
        )

        if self.final_logit_softcapping:
            if not (_is_npu or _is_cpu):
                fused_softcap(logits, self.final_logit_softcapping)
            else:
                logits = self.final_logit_softcapping * torch.tanh(
                    logits / self.final_logit_softcapping
                )

        return logits

    def _compute_lm_head(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        quant_method = getattr(lm_head, "quant_method", None)
        if hasattr(lm_head, "set_lora") and hasattr(lm_head, "apply_lora"):

            logits = lm_head(hidden_states)
        elif should_apply_lm_head_quant_method(lm_head, quant_method):
            logits = quant_method.apply(lm_head, hidden_states, embedding_bias)
        elif hasattr(lm_head, "weight"):

            if self.use_fp32_lm_head:


                use_mm_out_dtype = (
                    hidden_states.is_cuda
                    and hidden_states.dtype == lm_head.weight.dtype
                    and hidden_states.dtype in (torch.float16, torch.bfloat16)
                )
                if use_mm_out_dtype:
                    logits = torch.mm(
                        hidden_states,
                        lm_head.weight.T,
                        out_dtype=torch.float32,
                    )
                else:
                    logits = torch.matmul(
                        hidden_states.to(torch.float32),
                        lm_head.weight.to(torch.float32).T,
                    )
            elif use_intel_amx_backend(lm_head):
                logits = torch.ops.sgl_kernel.weight_packed_linear(
                    hidden_states.to(lm_head.weight.dtype),
                    lm_head.weight,
                    None,
                    True,
                )
            elif self.rl_on_policy_target is not None:

                logits = torch.matmul(
                    hidden_states.bfloat16(), lm_head.weight.T.bfloat16()
                )
            else:
                logits = torch.matmul(
                    hidden_states.to(lm_head.weight.dtype), lm_head.weight.T
                )
        else:


            if self.use_fp32_lm_head:
                with torch.cuda.amp.autocast(enabled=False):
                    logits = lm_head.quant_method.apply(
                        lm_head, hidden_states.to(torch.float32), embedding_bias
                    )
            else:
                logits = lm_head.quant_method.apply(
                    lm_head, hidden_states, embedding_bias
                )
        return logits

    def _gather_dp_attn_hidden_states(
        self, hidden_states: torch.Tensor, logits_metadata: LogitsMetadata
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.do_tensor_parallel_all_gather_dp_attn:
            logits_metadata.compute_dp_attention_metadata()
            local_hidden_states = hidden_states
            hidden_states = logits_metadata.gathered_buffer
            dp_gather_replicate(hidden_states, local_hidden_states, logits_metadata)
            return hidden_states, local_hidden_states
        return hidden_states, hidden_states

    def _gather_attn_tp_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.vocab_size % self.attn_tp_size == 0:
            global_logits = torch.empty(
                (
                    self.attn_tp_size,
                    logits.shape[0],
                    self.vocab_size // self.attn_tp_size,
                ),
                device=logits.device,
                dtype=logits.dtype,
            )
            attn_tp_all_gather_into_tensor(global_logits, logits)
            global_logits = global_logits.permute(1, 0, 2).reshape(
                logits.shape[0], self.vocab_size
            )
        else:
            global_logits = torch.empty(
                (self.vocab_size, logits.shape[0]),
                device=logits.device,
                dtype=logits.dtype,
            )
            global_logits = global_logits.T
            attn_tp_all_gather(
                list(global_logits.tensor_split(self.attn_tp_size, dim=-1)),
                logits,
            )
        return global_logits

    def _can_use_tp_lm_head_all_to_all(
        self,
        logits: torch.Tensor,
        local_hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        logits_metadata: LogitsMetadata,
    ) -> bool:
        if not self.use_tp_lm_head_all_to_all:
            return False

        tp_size = get_parallel().tp_size
        base_lm_head = getattr(lm_head, "base_layer", lm_head)
        if getattr(base_lm_head, "tp_size", None) != tp_size:


            return False


        global_counts_cpu = logits_metadata.global_num_tokens_for_logprob_cpu
        is_equal_padded_graph_layout = global_counts_cpu is None and (
            logits_metadata.global_num_tokens_for_logprob_gpu is not None
        )
        is_equal_eager_layout = (
            global_counts_cpu is not None
            and len(global_counts_cpu) == tp_size
            and len(global_counts_cpu) > 0
            and all(count == global_counts_cpu[0] for count in global_counts_cpu)
        )
        if not (is_equal_padded_graph_layout or is_equal_eager_layout):
            return False

        local_rows = local_hidden_states.shape[0]
        return local_rows > 0 and logits.shape[0] == local_rows * tp_size

    def _tp_lm_head_all_to_all(self, logits: torch.Tensor) -> torch.Tensor:

        logits = logits.contiguous()
        all_to_all_output = torch.empty_like(logits)
        get_tp_group().all_to_all_single(all_to_all_output.view(-1), logits.view(-1))
        return _reassemble_tp_lm_head_all_to_all_output(
            all_to_all_output, get_parallel().tp_size
        )

    def _scatter_dp_attn_logits(
        self,
        logits: torch.Tensor,
        local_hidden_states: torch.Tensor,
        logits_metadata: LogitsMetadata,
    ) -> torch.Tensor:
        if self.do_tensor_parallel_all_gather_dp_attn:
            global_logits = logits
            logits = torch.empty(
                (local_hidden_states.shape[0], global_logits.shape[1]),
                device=global_logits.device,
                dtype=global_logits.dtype,
            )
            dp_scatter(logits, global_logits, logits_metadata)
        return logits

    def _copy_logits_to_buffer(
        self,
        logits: torch.Tensor,
        logits_metadata: LogitsMetadata,
        use_buffer: bool = True,
    ) -> torch.Tensor:
        logits_buffer = logits_metadata.next_token_logits_buffer if use_buffer else None
        if logits.shape[-1] > self.vocab_size:
            logits = logits[:, : self.vocab_size]
        logits_width = logits.shape[-1]


        if logits_buffer is not None and tuple(logits_buffer.shape) == tuple(
            logits.shape
        ):
            assert logits_buffer.dtype == torch.float
            logits_buffer.copy_(logits)
            logits = logits_buffer
        else:
            logits = logits.float()
        return logits

    def _get_dllm_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        logits_metadata: LogitsMetadata,
    ) -> LogitsProcessorOutput:
        assert self.return_full_logits
        full_logits = self._get_logits(hidden_states, lm_head, logits_metadata)
        return LogitsProcessorOutput(
            full_logits=full_logits,
            next_token_logits=None,
        )

    def compute_logprobs_for_multi_item_scoring(
        self,
        input_ids,
        hidden_states,
        lm_head: VocabParallelEmbedding,
        logits_metadata: Union[LogitsMetadata, ForwardBatch],
        multi_item_delimiter_indices: List[torch.Tensor],
    ):


        device = input_ids.device
        all_tensors = []
        if logits_metadata.extend_seq_lens_cpu is not None:
            offset = 0
            for req_seq_len, indices_tensor in zip(
                logits_metadata.extend_seq_lens_cpu, multi_item_delimiter_indices
            ):
                if len(indices_tensor) > 0:


                    all_tensors.append(indices_tensor + (offset - 1))
                offset += req_seq_len
        else:
            all_tensors.append(multi_item_delimiter_indices[0] - 1)
        multi_item_indices = torch.cat(all_tensors).to(device, non_blocking=True)


        sliced_hidden = hidden_states[multi_item_indices]

        sliced_logits = self._get_logits(sliced_hidden, lm_head, logits_metadata)
        sliced_logprobs = torch.nn.functional.log_softmax(sliced_logits, dim=-1)


        input_token_ids_logprobs_val = []
        input_token_ids_logprobs_idx = []
        input_top_logprobs_val = None
        input_top_logprobs_idx = None


        if (
            logits_metadata.token_ids_logprobs
            or logits_metadata.extend_return_top_logprob
        ):
            logits_metadata.extend_logprob_pruned_lens_cpu = [
                len(t) for t in multi_item_delimiter_indices
            ]


        if logits_metadata.extend_token_ids_logprob:
            (
                input_token_ids_logprobs_val,
                input_token_ids_logprobs_idx,
            ) = get_token_ids_logprobs_raw(
                sliced_logprobs,
                logits_metadata.token_ids_logprobs,
                stage=LogprobStage.PREFILL,
                extend_logprob_pruned_lens_cpu=logits_metadata.extend_logprob_pruned_lens_cpu,
                no_copy_to_cpu=True,
            )


        if logits_metadata.extend_return_top_logprob:
            (
                input_top_logprobs_val,
                input_top_logprobs_idx,
            ) = get_top_logprobs_raw(
                sliced_logprobs,
                logits_metadata.top_logprobs_nums,
                stage=LogprobStage.PREFILL,
                extend_logprob_pruned_lens_cpu=logits_metadata.extend_logprob_pruned_lens_cpu,
            )


        input_token_logprobs = torch.zeros(multi_item_indices.shape[0], device=device)

        return LogitsProcessorOutput(
            next_token_logits=None,
            input_token_logprobs=input_token_logprobs,
            input_top_logprobs_val=input_top_logprobs_val,
            input_top_logprobs_idx=input_top_logprobs_idx,
            input_token_ids_logprobs_val=input_token_ids_logprobs_val,
            input_token_ids_logprobs_idx=input_token_ids_logprobs_idx,
            mm_input_embeds=logits_metadata.mm_input_embeds,
        )


def _reassemble_tp_lm_head_all_to_all_output(
    all_to_all_output: torch.Tensor, tp_size: int
) -> torch.Tensor:

    assert all_to_all_output.shape[0] % tp_size == 0
    local_rows = all_to_all_output.shape[0] // tp_size
    vocab_shard = all_to_all_output.shape[1]
    return (
        all_to_all_output.view(tp_size, local_rows, vocab_shard)
        .permute(1, 0, 2)
        .reshape(local_rows, tp_size * vocab_shard)
    )


def _has_lm_head_runtime_attrs(lm_head, attr_names: Tuple[str, ...]) -> bool:
    return all(hasattr(lm_head, attr_name) for attr_name in attr_names)


def should_apply_lm_head_quant_method(lm_head, quant_method) -> bool:
    if (
        quant_method is None
        or not hasattr(lm_head, "weight")
        or not callable(getattr(quant_method, "apply", None))
    ):
        return False

    method_name = type(quant_method).__name__
    if method_name in _UNQUANTIZED_LM_HEAD_METHODS:
        return False


    if method_name == "ModelOptFp4LinearMethod":
        if lm_head.weight.dtype == torch.int32 and _has_lm_head_runtime_attrs(
            lm_head,
            (
                "weight_scale",
                "weight_global_scale",
                "workspace",
                "input_size_per_partition",
                "output_size_per_partition",
            ),
        ):
            return True
        return lm_head.weight.dtype == torch.uint8 and _has_lm_head_runtime_attrs(
            lm_head,
            (
                "weight_scale_interleaved",
                "alpha",
                "input_scale_inv",
                "input_size_per_partition",
                "output_size_per_partition",
            ),
        )
    if method_name == "ModelOptNvFp4A16LinearMethod":
        return lm_head.weight.dtype == torch.int32 and _has_lm_head_runtime_attrs(
            lm_head,
            (
                "weight_scale",
                "weight_global_scale",
                "workspace",
                "input_size_per_partition",
                "output_size_per_partition",
            ),
        )
    if method_name == "ModelOptFp8LinearMethod":
        return (
            lm_head.weight.dtype == torch.float8_e4m3fn
            and _has_lm_head_runtime_attrs(lm_head, ("weight_scale", "input_scale"))
        )

    return True
