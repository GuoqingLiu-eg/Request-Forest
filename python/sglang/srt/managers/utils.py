from __future__ import annotations

import dataclasses
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Union

import msgspec
import torch

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.eplb.expert_distribution import ExpertDistributionMetrics
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers import io_struct
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.runtime_context import get_spec, max_speculative_num_draft_tokens
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.state_capturer.base import TopkCaptureOutput

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import GenerationBatchResult
    from sglang.srt.sampling.sampling_observer import HostAuxiliaryOutput
    from sglang.srt.speculative.eagle_info import EagleDraftInput


logger = logging.getLogger(__name__)


def _async_d2h(t: torch.Tensor) -> torch.Tensor:

    if not t.is_cuda:
        return t.to("cpu", non_blocking=True)
    cpu_t = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
    cpu_t.copy_(t, non_blocking=True)
    t.record_stream(torch.cuda.current_stream(t.device))
    return cpu_t


@dataclasses.dataclass
class GenerationBatchResult:
    logits_output: Optional[LogitsProcessorOutput] = None
    pp_hidden_states_proxy_tensors: Optional[PPProxyTensors] = None
    next_token_ids: Optional[
        Union[torch.Tensor, List[torch.Tensor], List[List[int]]]
    ] = None
    num_correct_drafts: int = 0
    num_correct_drafts_per_req_cpu: Optional[List[int]] = None
    num_proposed_drafts_per_req_cpu: Optional[List[int]] = None
    num_block_accept_tokens: int = 0
    num_cap_tokens: int = 0

    accept_length_per_req_cpu: Optional[List[int]] = None
    dllm_algo_state: Optional[List[Any]] = None
    can_run_cuda_graph: bool = False


    skipped_output_comm: bool = False


    extend_input_len_per_req: Optional[List[int]] = None
    extend_logprob_start_len_per_req: Optional[List[int]] = None


    copy_done: Optional[torch.cuda.Event] = None
    delay_sample_func: Optional[callable] = None
    future_indices: Optional[torch.Tensor] = None
    speculative_num_draft_tokens: Optional[int] = None


    adaptive_rf_mode: Optional[str] = None


    grammar_advanced: bool = False
    grammar_retained_tokens: Optional[list] = None


    accept_lens: Optional[torch.Tensor] = None


    accepted_draft_lens: Optional[torch.Tensor] = None


    proposed_draft_lens: Optional[torch.Tensor] = None


    request_forest_committed_slots: Optional[torch.Tensor] = None

    block_accept_lens: Optional[torch.Tensor] = None

    cap_lens: Optional[torch.Tensor] = None


    new_seq_lens: Optional[torch.Tensor] = None


    next_draft_input: Optional[EagleDraftInput] = None


    extra_keep_alive_refs: Optional[List[Any]] = None


    routed_experts_output: Optional[TopkCaptureOutput] = None
    indexer_topk_output: Optional[TopkCaptureOutput] = None


    expert_distribution_metrics: Optional[ExpertDistributionMetrics] = None


    fpm_start_event: Optional[torch.cuda.Event] = None
    fpm_end_event: Optional[torch.cuda.Event] = None

    auxiliary_host_output: Optional[HostAuxiliaryOutput] = None


    accept_lens_cpu: Optional[torch.Tensor] = None
    new_seq_lens_cpu: Optional[torch.Tensor] = None

    @property
    def has_sampled_token_ids(self) -> bool:

        return isinstance(self.next_token_ids, torch.Tensor)

    @torch.profiler.record_function("copy_result_to_cpu")
    def copy_to_cpu(self, return_logprob: bool, return_hidden_states: bool = True):

        if return_logprob:
            if self.logits_output.next_token_logprobs is not None:
                self.logits_output.next_token_logprobs = _async_d2h(
                    self.logits_output.next_token_logprobs
                )
            if self.logits_output.input_token_logprobs is not None:
                self.logits_output.input_token_logprobs = _async_d2h(
                    self.logits_output.input_token_logprobs
                )
            if self.logits_output.next_token_top_logprobs_val is not None:
                self.logits_output.next_token_top_logprobs_val = [
                    _async_d2h(v) if torch.is_tensor(v) else v
                    for v in self.logits_output.next_token_top_logprobs_val
                ]
            if self.logits_output.next_token_top_logprobs_idx is not None:
                self.logits_output.next_token_top_logprobs_idx = [
                    _async_d2h(x) if torch.is_tensor(x) else x
                    for x in self.logits_output.next_token_top_logprobs_idx
                ]
            if self.logits_output.next_token_token_ids_logprobs_val is not None:
                self.logits_output.next_token_token_ids_logprobs_val = [
                    _async_d2h(v) if torch.is_tensor(v) else v
                    for v in self.logits_output.next_token_token_ids_logprobs_val
                ]
        if return_hidden_states and self.logits_output.hidden_states is not None:
            self.logits_output.hidden_states = _async_d2h(
                self.logits_output.hidden_states
            )
        self.next_token_ids = _async_d2h(self.next_token_ids)

        if self.accept_lens is not None:
            self.accept_lens = (
                self.accept_lens_cpu
                if self.accept_lens_cpu is not None
                else _async_d2h(self.accept_lens)
            )

        if self.accepted_draft_lens is not None:
            self.accepted_draft_lens = _async_d2h(self.accepted_draft_lens)

        if self.proposed_draft_lens is not None:
            self.proposed_draft_lens = _async_d2h(self.proposed_draft_lens)

        if self.block_accept_lens is not None:
            self.block_accept_lens = _async_d2h(self.block_accept_lens)

        if self.cap_lens is not None:
            self.cap_lens = _async_d2h(self.cap_lens)


        for holder in (
            self.routed_experts_output,
            self.indexer_topk_output,
            self.expert_distribution_metrics,
        ):
            if holder is not None:
                holder.map_device_tensors(_async_d2h)

        self.copy_auxiliary_output_to_cpu()

        self.copy_done.record()

    def copy_auxiliary_output_to_cpu(self) -> None:
        if self.logits_output is None or self.auxiliary_host_output is not None:
            return
        device_output = self.logits_output.auxiliary_device_output
        if device_output is not None:
            self.auxiliary_host_output = device_output.copy_to_host(_async_d2h)
            self.logits_output.auxiliary_device_output = None

    @classmethod
    def from_pp_proxy(
        cls, logits_output, next_pp_outputs: PPProxyTensors, can_run_cuda_graph
    ):

        proxy_dict = next_pp_outputs.tensors
        return cls(
            logits_output=logits_output,
            pp_hidden_states_proxy_tensors=None,
            next_token_ids=next_pp_outputs["next_token_ids"],
            extend_input_len_per_req=proxy_dict.get("extend_input_len_per_req", None),
            extend_logprob_start_len_per_req=proxy_dict.get(
                "extend_logprob_start_len_per_req", None
            ),
            can_run_cuda_graph=can_run_cuda_graph,
        )


def validate_input_length(
    req: Req, max_req_input_len: int, allow_auto_truncate: bool
) -> Optional[str]:

    if len(req.origin_input_ids) >= max_req_input_len:
        if allow_auto_truncate:
            logger.warning(
                "Request length is longer than the KV cache pool size or "
                "the max context length. Truncated. "
                f"{len(req.origin_input_ids)=}, {max_req_input_len=}."
            )
            req.origin_input_ids = req.origin_input_ids[:max_req_input_len]
            return None
        else:
            error_msg = (
                f"Input length ({len(req.origin_input_ids)} tokens) exceeds "
                f"the maximum allowed length ({max_req_input_len} tokens). "
                f"Use a shorter input or enable --allow-auto-truncate."
            )
            return error_msg

    return None


def get_logprob_dict_from_result(result: GenerationBatchResult) -> dict:

    logits_output = result.logits_output
    assert logits_output is not None

    return {
        "extend_input_len_per_req": result.extend_input_len_per_req,
        "extend_logprob_start_len_per_req": result.extend_logprob_start_len_per_req,
        "next_token_logprobs": result.logits_output.next_token_logprobs,
        "next_token_top_logprobs_val": result.logits_output.next_token_top_logprobs_val,
        "next_token_top_logprobs_idx": result.logits_output.next_token_top_logprobs_idx,
        "next_token_token_ids_logprobs_val": result.logits_output.next_token_token_ids_logprobs_val,
        "next_token_token_ids_logprobs_idx": result.logits_output.next_token_token_ids_logprobs_idx,
        "next_token_sampling_mask_idx": result.logits_output.next_token_sampling_mask_idx,
        "next_token_sampling_logprobs": result.logits_output.next_token_sampling_logprobs,
        "input_token_logprobs": result.logits_output.input_token_logprobs,
        "input_top_logprobs_val": result.logits_output.input_top_logprobs_val,
        "input_top_logprobs_idx": result.logits_output.input_top_logprobs_idx,
        "input_token_ids_logprobs_val": result.logits_output.input_token_ids_logprobs_val,
        "input_token_ids_logprobs_idx": result.logits_output.input_token_ids_logprobs_idx,
    }


def get_logprob_from_pp_outputs(
    next_pp_outputs: PPProxyTensors,
) -> tuple[LogitsProcessorOutput, list[int], list[int]]:
    logits_output = LogitsProcessorOutput(

        next_token_logits=None,
        hidden_states=None,
        next_token_logprobs=next_pp_outputs["next_token_logprobs"],
        next_token_top_logprobs_val=next_pp_outputs["next_token_top_logprobs_val"],
        next_token_top_logprobs_idx=next_pp_outputs["next_token_top_logprobs_idx"],
        next_token_token_ids_logprobs_val=next_pp_outputs[
            "next_token_token_ids_logprobs_val"
        ],
        next_token_token_ids_logprobs_idx=next_pp_outputs[
            "next_token_token_ids_logprobs_idx"
        ],
        next_token_sampling_mask_idx=next_pp_outputs["next_token_sampling_mask_idx"],
        next_token_sampling_logprobs=next_pp_outputs["next_token_sampling_logprobs"],
        input_token_logprobs=next_pp_outputs["input_token_logprobs"],
        input_top_logprobs_val=next_pp_outputs["input_top_logprobs_val"],
        input_top_logprobs_idx=next_pp_outputs["input_top_logprobs_idx"],
        input_token_ids_logprobs_val=next_pp_outputs["input_token_ids_logprobs_val"],
        input_token_ids_logprobs_idx=next_pp_outputs["input_token_ids_logprobs_idx"],
    )
    extend_input_len_per_req = next_pp_outputs["extend_input_len_per_req"]
    extend_logprob_start_len_per_req = next_pp_outputs[
        "extend_logprob_start_len_per_req"
    ]

    return logits_output, extend_input_len_per_req, extend_logprob_start_len_per_req


@dataclass
class EmbeddingBatchResult:


    embeddings: torch.Tensor
    pooled_hidden_states: Optional[torch.Tensor] = None
    copy_done: Optional[torch.cuda.Event] = None
    can_run_cuda_graph: bool = False

    @torch.profiler.record_function("copy_embedding_to_cpu")
    def copy_to_cpu(self):

        if isinstance(self.embeddings, torch.Tensor):
            self.copy_done = torch.get_device_module(self.embeddings.device).Event()
            self.embeddings = _async_d2h(self.embeddings)
        else:
            assert isinstance(self.embeddings, list)
            if len(self.embeddings) == 0:
                return

            self.copy_done = torch.get_device_module(self.embeddings[0].device).Event()
            self.embeddings = [_async_d2h(emb) for emb in self.embeddings]

        if self.pooled_hidden_states is not None:
            if isinstance(self.pooled_hidden_states, list):
                self.pooled_hidden_states = [
                    _async_d2h(t) for t in self.pooled_hidden_states
                ]
            else:
                self.pooled_hidden_states = _async_d2h(self.pooled_hidden_states)

        self.copy_done.record()


def is_health_check_generate_req(recv_req):
    rid = getattr(recv_req, "rid", None)
    return rid is not None and rid.startswith(HEALTH_CHECK_RID_PREFIX)


class MsgpackDecodeError(ValueError):


    def __init__(self, rid: Optional[str], reason: str):
        super().__init__(reason)
        self.rid = rid
        self.reason = reason


def msgpack_decode_explained(data: bytes) -> Any:


    try:
        return io_struct.msgpack_decode(data)
    except Exception as e:
        msg = str(e)
        try:
            arr = msgspec.msgpack.decode(data)
        except Exception:
            arr = None
        if not (isinstance(arr, (list, tuple)) and arr):
            raise MsgpackDecodeError(None, msg) from e


        rid = str(arr[1]) if len(arr) > 1 and arr[1] is not None else None
        tag_to_fields = {
            cls.__struct_config__.tag: cls.__struct_fields__
            for cls in io_struct._all_types
            if isinstance(cls, type) and issubclass(cls, msgspec.Struct)
        }
        fields = tag_to_fields.get(arr[0])
        if fields is not None:


            m = re.search(r"\$\[(\d+)\]", msg)
            if m is not None:
                idx = int(m.group(1))
                if 1 <= idx <= len(fields):
                    msg = f"{msg[:m.start()]}$.{fields[idx - 1]}{msg[m.end():]}"
        raise MsgpackDecodeError(rid, msg) from e


def compute_num_reserved_tokens() -> int:

    spec = get_spec()
    algorithm = SpeculativeAlgorithm.from_string(spec.speculative_algorithm)
    if not algorithm.is_eagle():
        return 0
    return max(
        spec.speculative_eagle_topk * spec.speculative_num_steps,
        max_speculative_num_draft_tokens(),
    )
