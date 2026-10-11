from __future__ import annotations

import dataclasses
import logging
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_exec

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessorOutput
    from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
    from sglang.srt.managers.schedule_batch import ScheduleBatch

logger = logging.getLogger(__name__)


class LogprobStage(Enum):
    PREFILL = auto()
    DECODE = auto()


@dataclasses.dataclass
class LogprobResult:


    token_logprobs: Optional[torch.Tensor] = None
    top_logprobs_val: Optional[List] = None
    top_logprobs_idx: Optional[List] = None
    token_ids_logprobs_val: Optional[List] = None
    token_ids_logprobs_idx: Optional[List] = None

    def write_input_to(self, logits_output: LogitsProcessorOutput) -> None:
        if self.token_logprobs is not None:
            logits_output.input_token_logprobs = self.token_logprobs
        if self.top_logprobs_val is not None:
            logits_output.input_top_logprobs_val = self.top_logprobs_val
            logits_output.input_top_logprobs_idx = self.top_logprobs_idx
        if self.token_ids_logprobs_val is not None:
            logits_output.input_token_ids_logprobs_val = self.token_ids_logprobs_val
            logits_output.input_token_ids_logprobs_idx = self.token_ids_logprobs_idx

    def write_output_to(self, logits_output: LogitsProcessorOutput) -> None:
        if self.token_logprobs is not None:
            logits_output.next_token_logprobs = self.token_logprobs
        if self.top_logprobs_val is not None:
            logits_output.next_token_top_logprobs_val = self.top_logprobs_val
            logits_output.next_token_top_logprobs_idx = self.top_logprobs_idx
        if self.token_ids_logprobs_val is not None:
            logits_output.next_token_token_ids_logprobs_val = (
                self.token_ids_logprobs_val
            )
            logits_output.next_token_token_ids_logprobs_idx = (
                self.token_ids_logprobs_idx
            )


def compute_row_log_normalizer(
    logits: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    if logits.is_cuda:
        from sglang.srt.layers.logsumexp import row_logsumexp

        return row_logsumexp(logits)
    x = logits.float()
    row_max = x.amax(dim=-1)
    row_log_sum = torch.logsumexp(x - row_max[:, None], dim=-1)
    row_log_sum = torch.where(row_max.isinf(), 0.0, row_log_sum)
    return row_max, row_log_sum


def get_top_logprobs_raw(
    logprobs: torch.Tensor,
    top_logprobs_nums: List[int],
    stage: LogprobStage,
    extend_logprob_pruned_lens_cpu: Optional[List[int]] = None,
    no_copy_to_cpu: bool = False,
):
    max_k = max(top_logprobs_nums)
    values, indices = logprobs.topk(max_k, dim=-1)
    if not no_copy_to_cpu:
        values = values.tolist()
        indices = indices.tolist()

    top_logprobs_val = []
    top_logprobs_idx = []

    if stage == LogprobStage.DECODE:
        for i, k in enumerate(top_logprobs_nums):
            top_logprobs_val.append(values[i][:k])
            top_logprobs_idx.append(indices[i][:k])
    else:
        pt = 0
        for k, pruned_len in zip(top_logprobs_nums, extend_logprob_pruned_lens_cpu):
            if pruned_len <= 0:
                top_logprobs_val.append([])
                top_logprobs_idx.append([])
                continue

            top_logprobs_val.append([values[pt + j][:k] for j in range(pruned_len)])
            top_logprobs_idx.append([indices[pt + j][:k] for j in range(pruned_len)])
            pt += pruned_len

    return top_logprobs_val, top_logprobs_idx


def get_top_logprobs(
    logprobs: torch.Tensor,
    top_logprobs_nums: List[int],
    no_copy_to_cpu: bool = False,
):
    return get_top_logprobs_raw(
        logprobs,
        top_logprobs_nums,
        stage=LogprobStage.DECODE,
        no_copy_to_cpu=no_copy_to_cpu,
    )


def get_token_ids_logprobs_raw(
    logprobs: torch.Tensor,
    token_ids_logprobs_list: List[Optional[List[int]]],
    stage: LogprobStage,
    extend_logprob_pruned_lens_cpu: Optional[List[int]] = None,
    no_copy_to_cpu: bool = False,
):
    vals, idxs = [], []
    if stage == LogprobStage.DECODE:
        for i, token_ids in enumerate(token_ids_logprobs_list):
            if token_ids is None:
                vals.append([])
                idxs.append([])
            else:
                token_ids_tensor = torch.tensor(token_ids, dtype=torch.long).to(
                    logprobs.device, non_blocking=True
                )
                row = logprobs[i, token_ids_tensor]
                vals.append(row if no_copy_to_cpu else row.tolist())
                idxs.append(token_ids)
    else:
        pt = 0
        for i, (token_ids, pruned_len) in enumerate(
            zip(token_ids_logprobs_list, extend_logprob_pruned_lens_cpu)
        ):
            if pruned_len <= 0:
                vals.append([])
                idxs.append([])
                continue
            if token_ids is None:

                vals.append([])
                idxs.append([])
                pt += pruned_len
                continue
            token_ids_tensor = torch.tensor(token_ids, dtype=torch.long).to(
                logprobs.device, non_blocking=True
            )
            pos_logprobs = logprobs[pt : pt + pruned_len, token_ids_tensor]
            vals.append(pos_logprobs if no_copy_to_cpu else pos_logprobs.tolist())
            idxs.append([token_ids for _ in range(pruned_len)])
            pt += pruned_len
    return vals, idxs


def get_token_ids_logprobs(logprobs, token_ids_logprobs, no_copy_to_cpu=False):
    return get_token_ids_logprobs_raw(
        logprobs,
        token_ids_logprobs,
        stage=LogprobStage.DECODE,
        no_copy_to_cpu=no_copy_to_cpu,
    )


def get_top_logprobs_chunk(
    logprobs: torch.Tensor,
    top_k_nums: List[int],
    pruned_lens: List[int],
    top_logprobs_val: List,
    top_logprobs_idx: List,
    split_pruned_len: int,
    log_normalizer: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    precomputed_topk: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> int:


    max_k = max(top_k_nums)
    if log_normalizer is not None:
        row_max, row_log_sum = log_normalizer
        if precomputed_topk is not None:
            values_tensor, indices_tensor = precomputed_topk
        else:
            values_tensor, indices_tensor = logprobs.topk(max_k, dim=1)
        values_tensor = (values_tensor.float() - row_max[:, None]) - row_log_sum[
            :, None
        ]
    else:
        values_tensor, indices_tensor = logprobs.topk(max_k, dim=1)
    values = values_tensor.tolist()
    indices = indices_tensor.tolist()

    pt = 0
    next_split_pruned_len = 0
    for n, (k, pruned_len) in enumerate(zip(top_k_nums, pruned_lens)):
        if n == 0:

            pruned_len -= split_pruned_len
        else:

            split_pruned_len = 0

        if pruned_len <= 0:


            top_logprobs_val.append([])
            top_logprobs_idx.append([])
            continue


        val = []
        idx = []
        for j in range(pruned_len):

            if pt + j >= len(values):
                next_split_pruned_len = split_pruned_len + j
                break

            val.append(values[pt + j][:k])
            idx.append(indices[pt + j][:k])


        if split_pruned_len > 0:
            top_logprobs_val[-1].extend(val)
            top_logprobs_idx[-1].extend(idx)
        else:
            top_logprobs_val.append(val)
            top_logprobs_idx.append(idx)

        pt += pruned_len
    return next_split_pruned_len


def get_token_ids_logprobs_chunk(
    logprobs: torch.Tensor,
    token_ids_logprobs: List[int],
    pruned_lens: List[int],
    token_ids_logprobs_val: List,
    token_ids_logprobs_idx: List,
    split_pruned_len: int = 0,
    log_normalizer: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
):


    if log_normalizer is not None:
        row_max, row_log_sum = log_normalizer
    pt = 0
    next_split_pruned_len = 0
    for n, (token_ids, pruned_len) in enumerate(
        zip(
            token_ids_logprobs,
            pruned_lens,
        )
    ):

        if n == 0:
            pruned_len -= split_pruned_len
        else:
            split_pruned_len = 0

        if pruned_len <= 0:


            token_ids_logprobs_val.append([])
            token_ids_logprobs_idx.append([])
            continue


        val = []
        idx = []
        for j in range(pruned_len):

            if pt + j >= logprobs.shape[0]:
                next_split_pruned_len = split_pruned_len + j
                break
            if token_ids is not None:
                row = logprobs[pt + j, token_ids]
                if log_normalizer is not None:
                    row = (row.float() - row_max[pt + j]) - row_log_sum[pt + j]
                val.append(row.tolist())
                idx.append(token_ids)


        if split_pruned_len > 0:
            token_ids_logprobs_val[-1].extend(val)
            token_ids_logprobs_idx[-1].extend(idx)
        else:
            token_ids_logprobs_val.append(val)
            token_ids_logprobs_idx.append(idx)

        pt += pruned_len
    return next_split_pruned_len


def compute_spec_logprobs(
    batch: ScheduleBatch,
    logits_output: LogitsProcessorOutput,
    predict: torch.Tensor,
    *,
    accept_index: Optional[torch.Tensor] = None,
    chain_stride: Optional[int] = None,
):
    assert (accept_index is None) != (
        chain_stride is None
    ), "pass exactly one of accept_index / chain_stride"

    bs = len(batch.seq_lens)
    next_token_logits = logits_output.next_token_logits

    if accept_index is not None:
        max_accept = accept_index.shape[1]
        flat_accept_idx = accept_index.long().reshape(-1)
        gathered_logits = next_token_logits[flat_accept_idx]
        accepted_token_ids = predict[flat_accept_idx]
    else:
        max_accept = chain_stride


        assert next_token_logits.shape[0] == bs * max_accept, (
            f"chain layout expects {bs * max_accept} logits rows, got "
            f"{next_token_logits.shape[0]}"
        )
        gathered_logits = next_token_logits
        accepted_token_ids = predict

    if batch.sampling_info.is_all_greedy or envs.SGLANG_RETURN_ORIGINAL_LOGPROB.get():
        gathered_logprobs = torch.nn.functional.log_softmax(gathered_logits, dim=-1)
    else:
        temperatures = torch.repeat_interleave(
            batch.sampling_info.temperatures,
            max_accept,
            dim=0,
        )
        gathered_logprobs = torch.nn.functional.log_softmax(
            gathered_logits / temperatures, dim=-1
        )
    gathered_logprobs.clamp_(min=torch.finfo(gathered_logprobs.dtype).min)

    logits_output.next_token_logprobs = gathered_logprobs.gather(
        1, accepted_token_ids.long().view(-1, 1)
    ).view(bs, max_accept)

    if batch.top_logprobs_nums and any(x > 0 for x in batch.top_logprobs_nums):
        top_logprobs_nums_expanded = [
            num for num in batch.top_logprobs_nums for _ in range(max_accept)
        ]
        (
            logits_output.next_token_top_logprobs_val,
            logits_output.next_token_top_logprobs_idx,
        ) = get_top_logprobs(
            gathered_logprobs, top_logprobs_nums_expanded, no_copy_to_cpu=True
        )

    if batch.token_ids_logprobs and any(
        x is not None for x in batch.token_ids_logprobs
    ):
        token_ids_logprobs_expanded = [
            ids for ids in batch.token_ids_logprobs for _ in range(max_accept)
        ]
        (
            logits_output.next_token_token_ids_logprobs_val,
            logits_output.next_token_token_ids_logprobs_idx,
        ) = get_token_ids_logprobs(
            gathered_logprobs, token_ids_logprobs_expanded, no_copy_to_cpu=True
        )


def _deterministic_inference_enabled() -> bool:

    try:
        return bool(get_exec().deterministic.enable_deterministic_inference)
    except ValueError:
        return False


class InputLogprobProcessor:


    def __init__(self):

        self.enable_logprobs_chunk = envs.SGLANG_ENABLE_LOGPROB_CHUNK.get()

        self.logprobs_chunk_size = envs.SGLANG_LOGPROB_CHUNK_SIZE.get()


        self.enable_fast_input_logprobs = (
            envs.SGLANG_ENABLE_FAST_INPUT_LOGPROBS.get()
            and not _deterministic_inference_enabled()
        )

    def forward(
        self,
        pruned_states: torch.Tensor,
        sample_indices: Optional[torch.Tensor],
        input_logprob_indices: torch.Tensor,
        token_to_seq_idx: list[int],
        lm_head: VocabParallelEmbedding,
        get_logits_fn: Callable,
        logits_metadata: LogitsMetadata,
        skip_chunking_for_dp_attn: bool = False,
    ) -> Tuple[LogprobResult, torch.Tensor]:


        if (
            not self.enable_logprobs_chunk
            or pruned_states.shape[0] <= self.logprobs_chunk_size
            or skip_chunking_for_dp_attn
        ):
            chunk_size = max(pruned_states.shape[0], 1)
        else:
            chunk_size = self.logprobs_chunk_size

        return self._forward_by_chunk(
            pruned_states,
            sample_indices,
            input_logprob_indices,
            token_to_seq_idx,
            lm_head,
            get_logits_fn,
            logits_metadata,
            chunk_size,
        )

    def _forward_by_chunk(
        self,
        pruned_states: torch.Tensor,
        sample_indices: torch.Tensor,
        input_logprob_indices: torch.Tensor,
        token_to_seq_idx: list[int],
        lm_head: VocabParallelEmbedding,
        get_logits_fn: Callable,
        logits_metadata: LogitsMetadata,
        chunk_size: int,
    ) -> Tuple[LogprobResult, torch.Tensor]:

        total_size = pruned_states.shape[0]
        num_chunks = (total_size + chunk_size - 1) // chunk_size

        token_logprobs = []
        if logits_metadata.extend_return_top_logprob:
            top_logprobs_val = []
            top_logprobs_idx = []
        else:
            top_logprobs_val = None
            top_logprobs_idx = None
        if logits_metadata.extend_token_ids_logprob:
            token_ids_logprobs_val = []
            token_ids_logprobs_idx = []
        else:
            token_ids_logprobs_val = None
            token_ids_logprobs_idx = None


        split_len_topk = 0
        split_len_token_ids = 0

        fused_kernel, fused_max_k = None, 0
        if self.enable_fast_input_logprobs and pruned_states.is_cuda:
            from sglang.srt.layers.logsumexp import (
                FUSED_TOPK_MAX_K,
                row_logsumexp_topk,
            )

            fused_kernel, fused_max_k = row_logsumexp_topk, FUSED_TOPK_MAX_K

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_size)


            if num_chunks > 1 and hasattr(lm_head, "set_lm_head_pass"):
                lm_head.set_lm_head_pass(i)


            chunk_mask = (input_logprob_indices >= start_idx) & (
                input_logprob_indices < end_idx
            )
            global_indices = input_logprob_indices[chunk_mask]
            chunk_indices = global_indices - start_idx


            mask_indices = torch.nonzero(chunk_mask, as_tuple=True)[0]


            chunk_states = pruned_states[start_idx:end_idx]
            chunk_logits = get_logits_fn(
                chunk_states,
                lm_head,
                logits_metadata,
                use_logits_buffer=num_chunks == 1,
            )


            if i == 0:
                sampled_logits = torch.empty(
                    (sample_indices.shape[0], chunk_logits.shape[1]),
                    dtype=chunk_logits.dtype,
                    device=chunk_logits.device,
                )


            chunk_sample_mask = (sample_indices >= start_idx) & (
                sample_indices < end_idx
            )
            if chunk_sample_mask.any():
                chunk_sample_indices = sample_indices[chunk_sample_mask] - start_idx
                sampled_logits[chunk_sample_mask] = chunk_logits[chunk_sample_indices]


            chunk_logprobs = chunk_logits[chunk_indices]
            del chunk_logits


            chunk_slice = slice(
                token_to_seq_idx[start_idx], token_to_seq_idx[end_idx - 1] + 1
            )

            chunk_precomputed_topk = None
            if self.enable_fast_input_logprobs:


                max_k = (
                    max(logits_metadata.top_logprobs_nums[chunk_slice])
                    if logits_metadata.extend_return_top_logprob
                    else 0
                )
                if 0 < max_k <= fused_max_k:
                    row_max, row_log_sum, top_vals, top_idx = fused_kernel(
                        chunk_logprobs, max_k
                    )
                    chunk_log_normalizer = (row_max, row_log_sum)
                    chunk_precomputed_topk = (top_vals, top_idx)
                else:
                    chunk_log_normalizer = compute_row_log_normalizer(chunk_logprobs)
            else:


                chunk_log_normalizer = None
                chunk_logprobs = torch.nn.functional.log_softmax(chunk_logprobs, dim=-1)


            if logits_metadata.extend_return_top_logprob:
                top_k_nums = logits_metadata.top_logprobs_nums[chunk_slice]
                pruned_lens = logits_metadata.extend_logprob_pruned_lens_cpu[
                    chunk_slice
                ]
                split_len_topk = get_top_logprobs_chunk(
                    chunk_logprobs,
                    top_k_nums,
                    pruned_lens,
                    top_logprobs_val,
                    top_logprobs_idx,
                    split_len_topk,
                    log_normalizer=chunk_log_normalizer,
                    precomputed_topk=chunk_precomputed_topk,
                )


            if logits_metadata.extend_token_ids_logprob:
                token_ids_logprobs = logits_metadata.token_ids_logprobs[chunk_slice]
                pruned_lens = logits_metadata.extend_logprob_pruned_lens_cpu[
                    chunk_slice
                ]
                split_len_token_ids = get_token_ids_logprobs_chunk(
                    chunk_logprobs,
                    token_ids_logprobs,
                    pruned_lens,
                    token_ids_logprobs_val,
                    token_ids_logprobs_idx,
                    split_len_token_ids,
                    log_normalizer=chunk_log_normalizer,
                )


            chunk_token_logprobs = chunk_logprobs[
                torch.arange(chunk_logprobs.shape[0], device=chunk_logprobs.device),
                logits_metadata.extend_input_logprob_token_ids_gpu[mask_indices],
            ]
            if chunk_log_normalizer is not None:
                row_max, row_log_sum = chunk_log_normalizer
                chunk_token_logprobs = (
                    chunk_token_logprobs.float() - row_max
                ) - row_log_sum
            token_logprobs.append(chunk_token_logprobs)

            del chunk_logprobs


        if num_chunks > 1 and hasattr(lm_head, "reset_lm_head_pass"):
            assert hasattr(
                lm_head, "set_lm_head_pass"
            ), "lm_head must have set_lm_head_pass method and reset_lm_head_pass method at the same time"
            lm_head.reset_lm_head_pass()


        token_logprobs = torch.cat(token_logprobs, dim=0)

        return (
            LogprobResult(
                token_logprobs=token_logprobs,
                top_logprobs_val=top_logprobs_val,
                top_logprobs_idx=top_logprobs_idx,
                token_ids_logprobs_val=token_ids_logprobs_val,
                token_ids_logprobs_idx=token_ids_logprobs_idx,
            ),
            sampled_logits,
        )


def get_token_ids_logprobs_batch_optimized(
    logprobs: torch.Tensor,
    token_ids_logprobs: List[List[int]],
) -> Tuple[List, List]:

    batch_size = len(token_ids_logprobs)
    device = logprobs.device


    token_lengths = torch.tensor(
        [len(token_ids or []) for token_ids in token_ids_logprobs], device=device
    )
    total_tokens = int(token_lengths.sum().item())


    if total_tokens == 0:
        return [logprobs.new_empty(0) for _ in token_ids_logprobs], [
            [] for _ in token_ids_logprobs
        ]


    row_indices = torch.repeat_interleave(
        torch.arange(batch_size, device=device), token_lengths
    )

    col_indices = torch.tensor(
        [
            token_id
            for token_ids in token_ids_logprobs
            for token_id in (token_ids or [])
        ],
        device=device,
        dtype=torch.long,
    )


    gathered_logprobs = logprobs[row_indices, col_indices]


    split_logprobs = torch.split_with_sizes(
        gathered_logprobs, token_lengths.tolist(), dim=0
    )


    output_token_ids_logprobs_val = []
    output_token_ids_logprobs_idx = []

    for i, token_ids in enumerate(token_ids_logprobs):
        if token_ids is not None and len(token_ids) > 0:
            output_token_ids_logprobs_val.append(split_logprobs[i])
            output_token_ids_logprobs_idx.append(token_ids)
        else:
            output_token_ids_logprobs_val.append(logprobs.new_empty(0))
            output_token_ids_logprobs_idx.append([])

    return output_token_ids_logprobs_val, output_token_ids_logprobs_idx


class OutputLogprobProcessor:


    def compute_logprobs(
        self,
        logprobs: torch.Tensor,
        top_logprobs_nums: List[int],
        token_ids_logprobs: List[List[int]],
        batch_next_token_ids: torch.Tensor,
    ) -> LogprobResult:

        logprobs.clamp_(min=torch.finfo(logprobs.dtype).min)

        result = LogprobResult()
        if any(x > 0 for x in top_logprobs_nums):
            (
                result.top_logprobs_val,
                result.top_logprobs_idx,
            ) = get_top_logprobs(logprobs, top_logprobs_nums, no_copy_to_cpu=True)

        if any(x is not None for x in token_ids_logprobs):
            (
                result.token_ids_logprobs_val,
                result.token_ids_logprobs_idx,
            ) = get_token_ids_logprobs(
                logprobs, token_ids_logprobs, no_copy_to_cpu=True
            )

        result.token_logprobs = logprobs[
            torch.arange(len(batch_next_token_ids), device=batch_next_token_ids.device),
            batch_next_token_ids,
        ]
        return result

    def compute_logprobs_only(
        self,
        next_token_logits: Optional[torch.Tensor],
        top_logprobs_nums: List[int],
        token_ids_logprobs: List[List[int]],
        preprocess_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> Optional[LogprobResult]:


        if next_token_logits is None:
            logger.warning("No logits available for logprob computation")
            return None


        needs_token_ids_logprobs = any(
            token_ids is not None and len(token_ids) > 0
            for token_ids in token_ids_logprobs
        )
        needs_top_logprobs = any(x > 0 for x in top_logprobs_nums)

        if not (needs_token_ids_logprobs or needs_top_logprobs):
            return None


        logits = preprocess_fn(next_token_logits)


        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

        result = LogprobResult()

        if needs_top_logprobs:
            (
                result.top_logprobs_val,
                result.top_logprobs_idx,
            ) = get_top_logprobs(logprobs, top_logprobs_nums, no_copy_to_cpu=True)


        if needs_token_ids_logprobs:
            (
                result.token_ids_logprobs_val,
                result.token_ids_logprobs_idx,
            ) = get_token_ids_logprobs_batch_optimized(logprobs, token_ids_logprobs)
        return result
