import logging
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn

from sglang.kernels.ops.sampling.murmur_hash import murmur_hash32
from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.dp_attention import (
    is_dp_attention_enabled,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.logprob_processor import (
    OutputLogprobProcessor,
)
from sglang.srt.runtime_context import get_exec, get_parallel, get_server_args
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.utils.async_probe import sanitize_nan_logits
from sglang.srt.utils.common import (
    get_bool_env_var,
    is_cuda,
    is_hip,
    is_musa,
    is_npu,
)

if is_cuda():
    from flashinfer.sampling import (
        min_p_sampling_from_probs,
        top_k_top_p_sampling_from_probs,
    )
    from sgl_kernel import (
        top_k_renorm_prob,
        top_p_renorm_prob,
    )

if is_musa():
    from sgl_kernel import (
        min_p_sampling_from_probs,
        top_k_renorm_prob,
        top_k_top_p_sampling_from_probs,
        top_p_renorm_prob,
    )

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()
if _use_aiter:
    from aiter import greedy_sample as _aiter_greedy_sample


_disable_aiter_greedy_sample = get_bool_env_var("SGLANG_DISABLE_AITER_GREEDY_SAMPLE")

if is_npu():
    import torch_npu

logger = logging.getLogger(__name__)

SYNC_TOKEN_IDS_ACROSS_TP = get_bool_env_var("SYNC_TOKEN_IDS_ACROSS_TP")
SGLANG_RETURN_ORIGINAL_LOGPROB = get_bool_env_var("SGLANG_RETURN_ORIGINAL_LOGPROB")
_CUSTOM_SAMPLER_FACTORIES: Dict[str, Callable[[], "Sampler"]] = {}
_BUILT_IN_SAMPLING_BACKENDS = {"flashinfer", "pytorch", "ascend"}


class Sampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.tp_sync_group = get_tp_group().device_group
        if is_dp_attention_enabled():
            self.tp_sync_group = get_parallel().attn_tp_group.device_group

        self.rl_on_policy_target = get_exec().deterministic.rl_on_policy_target

        self.enable_deterministic = (
            get_exec().deterministic.enable_deterministic_inference
        )

        self.use_log_softmax_logprob = self.rl_on_policy_target is not None
        self.use_ascend_backend = get_exec().kernel.sampling_backend == "ascend"

        self.output_logprob_processor = OutputLogprobProcessor()

    def _preprocess_logits(
        self, logits: torch.Tensor, sampling_info: SamplingBatchInfo
    ) -> torch.Tensor:

        if sampling_info.has_custom_logit_processor:
            apply_custom_logit_processor(logits, sampling_info)
        sanitize_nan_logits(logits, "sampler: next_token_logits")
        return logits

    def forward(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        return_logprob: bool,
        top_logprobs_nums: List[int],
        token_ids_logprobs: List[List[int]],
        positions: torch.Tensor,
    ):

        logits = logits_output.next_token_logits

        if _is_hip and logits.shape[0] == 0:
            return torch.empty((0,), dtype=torch.int64, device=logits.device)


        logits = self._preprocess_logits(logits, sampling_info)
        return_sampling_mask = any(sampling_info.return_sampling_masks or [])

        if sampling_info.is_all_greedy:
            if _use_aiter and not _disable_aiter_greedy_sample:
                batch_next_token_ids = torch.empty(
                    logits.shape[0], device=logits.device, dtype=torch.int32
                )
                _aiter_greedy_sample(batch_next_token_ids, logits)
            else:
                batch_next_token_ids = torch.argmax(logits, -1)
            if return_sampling_mask:
                self._attach_greedy_sampling_mask_to_output(
                    logits_output, sampling_info, batch_next_token_ids
                )
            if return_logprob:
                original_logprobs = logprobs = torch.nn.functional.log_softmax(
                    logits, dim=-1
                )
        else:
            simple_sampling_case = (
                not sampling_info.need_top_p_sampling
                and not sampling_info.need_top_k_sampling
                and not sampling_info.need_min_p_sampling
            )


            if return_logprob and SGLANG_RETURN_ORIGINAL_LOGPROB:
                original_logprobs = torch.log_softmax(logits, dim=-1)


            logprobs_via_logsoftmax_kernel = None
            if self.rl_on_policy_target is not None:

                logits_div_temperature = (
                    logits.bfloat16().div(sampling_info.temperatures).bfloat16()
                )
                logprobs_via_logsoftmax_kernel = torch.log_softmax(
                    logits_div_temperature, dim=-1
                )
                del logits_div_temperature

            if self.use_ascend_backend:

                batch_next_token_ids, logprobs = self._forward_ascend_backend(
                    logits,
                    sampling_info,
                    simple_sampling_case,
                    return_logprob,
                    positions,
                )
            elif (
                self.use_log_softmax_logprob
                and self.enable_deterministic
                and simple_sampling_case
            ):

                batch_next_token_ids = self._sample_from_logprobs(
                    logprobs_via_logsoftmax_kernel,
                    sampling_info,
                    positions,
                )
                if return_logprob and not SGLANG_RETURN_ORIGINAL_LOGPROB:
                    logprobs = logprobs_via_logsoftmax_kernel
            else:

                logits.div_(sampling_info.temperatures)


                if (
                    return_logprob
                    and self.enable_deterministic
                    and logprobs_via_logsoftmax_kernel is None
                    and not SGLANG_RETURN_ORIGINAL_LOGPROB
                ):
                    logprobs_via_logsoftmax_kernel = torch.nn.functional.log_softmax(
                        logits, dim=-1
                    )


                logits[:] = torch.softmax(logits, dim=-1)
                probs = logits

                batch_next_token_ids = self._sample_from_probs(
                    probs, sampling_info, positions, simple_sampling_case
                )
                if return_sampling_mask:
                    sampling_mask_data = self._compute_sampling_mask_from_probs(
                        probs, sampling_info
                    )
                    self._attach_sampling_mask_to_output(
                        logits_output,
                        sampling_info,
                        batch_next_token_ids,
                        sampling_mask_data,
                    )
                if return_logprob and not SGLANG_RETURN_ORIGINAL_LOGPROB:
                    logprobs = (
                        logprobs_via_logsoftmax_kernel
                        if logprobs_via_logsoftmax_kernel is not None
                        else torch.log(probs)
                    )
                del probs

        if return_logprob:
            if SGLANG_RETURN_ORIGINAL_LOGPROB:
                logprobs = original_logprobs
            logprob_result = self.output_logprob_processor.compute_logprobs(
                logprobs,
                top_logprobs_nums,
                token_ids_logprobs,
                batch_next_token_ids,
            )
            logprob_result.write_output_to(logits_output)

        self._sync_token_ids_across_tp(batch_next_token_ids, sampling_info)

        return batch_next_token_ids

    def _sample_from_probs(
        self,
        probs: torch.Tensor,
        sampling_info: SamplingBatchInfo,
        positions: torch.Tensor,
        simple_sampling_case: bool,
    ) -> torch.Tensor:

        if simple_sampling_case:
            batch_next_token_ids = sampling_from_probs_torch(
                probs,
                sampling_seed=sampling_info.sampling_seed,
                positions=positions,
            )
        else:
            backend = get_exec().kernel.sampling_backend
            if backend == "flashinfer":
                assert (
                    sampling_info.sampling_seed is None
                ), "Sampling seed is not supported for flashinfer backend"
                if sampling_info.need_min_p_sampling:
                    probs = top_k_renorm_prob(probs, sampling_info.top_ks)
                    probs = top_p_renorm_prob(probs, sampling_info.top_ps)
                    batch_next_token_ids = min_p_sampling_from_probs(
                        probs, sampling_info.min_ps
                    )
                else:
                    batch_next_token_ids = top_k_top_p_sampling_from_probs(
                        probs.contiguous(),
                        sampling_info.top_ks,
                        sampling_info.top_ps,
                        filter_apply_order="joint",
                    )
            elif backend == "pytorch":

                batch_next_token_ids = top_k_top_p_min_p_sampling_from_probs_torch(
                    probs,
                    sampling_info.top_ks,
                    sampling_info.top_ps,
                    sampling_info.min_ps,
                    sampling_info.need_min_p_sampling,
                    sampling_info.sampling_seed,
                    positions,
                )
            else:
                raise ValueError(f"Invalid sampling backend: {backend}")
        return batch_next_token_ids

    def _compute_sampling_mask_from_probs(
        self, probs: torch.Tensor, sampling_info: SamplingBatchInfo
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        vocab_size = probs.shape[-1]
        max_top_k = sampling_info.sampling_mask_max_top_k
        if 0 < max_top_k < vocab_size:
            probs_sort, probs_idx = torch.topk(
                probs,
                k=max_top_k,
                dim=-1,
                largest=True,
                sorted=True,
            )
            positions = torch.arange(max_top_k, device=probs.device).view(1, -1)
        else:
            probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
            positions = torch.arange(vocab_size, device=probs.device).view(1, -1)
        probs_sum = torch.cumsum(probs_sort, dim=-1)

        keep_mask = positions < sampling_info.top_ks.view(-1, 1)
        keep_mask &= (probs_sum - probs_sort) <= sampling_info.top_ps.view(-1, 1)

        if sampling_info.need_min_p_sampling:
            min_p_thresholds = probs_sort[:, 0] * sampling_info.min_ps
            keep_mask &= probs_sort >= min_p_thresholds.view(-1, 1)

        return probs_idx, probs_sort, keep_mask, probs

    def _attach_greedy_sampling_mask_to_output(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        batch_next_token_ids: torch.Tensor,
    ) -> None:
        tokens = batch_next_token_ids.to(torch.int32).cpu().tolist()
        masks = []
        logprobs = []
        for i, should_return in enumerate(sampling_info.return_sampling_masks or []):
            if should_return:
                masks.append([int(tokens[i])])
                logprobs.append(0.0)
            else:
                masks.append(None)
                logprobs.append(None)
        logits_output.next_token_sampling_mask_idx = masks
        logits_output.next_token_sampling_logprobs = logprobs

    def _attach_sampling_mask_to_output(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        batch_next_token_ids: torch.Tensor,
        sampling_mask_data: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> None:
        probs_idx, probs_sort, keep_mask, probs = sampling_mask_data
        return_sampling_masks = sampling_info.return_sampling_masks or []
        if not return_sampling_masks:
            logits_output.next_token_sampling_mask_idx = []
            logits_output.next_token_sampling_logprobs = []
            return

        sampled_tokens = batch_next_token_ids.view(-1, 1)
        sampled_matches_all = probs_idx == sampled_tokens
        sampled_in_idx = sampled_matches_all.any(dim=-1)


        effective_keep_mask = keep_mask | sampled_matches_all
        selected_raw_probs = torch.gather(probs, 1, sampled_tokens).squeeze(1)
        support_mass = torch.where(
            effective_keep_mask, probs_sort, torch.zeros_like(probs_sort)
        ).sum(dim=-1)
        support_mass = support_mass + torch.where(
            sampled_in_idx, torch.zeros_like(selected_raw_probs), selected_raw_probs
        )
        selected_logprobs = torch.log(
            selected_raw_probs.float()
            / support_mass.float().clamp_min(torch.finfo(torch.float32).tiny)
        )

        flat_rows, flat_cols = effective_keep_mask.nonzero(as_tuple=True)
        flat_ids = probs_idx[flat_rows, flat_cols].to(torch.int32)
        mask_lengths = effective_keep_mask.sum(dim=-1, dtype=torch.int32)

        flat_ids_cpu = flat_ids.cpu().tolist()
        mask_lengths_cpu = mask_lengths.cpu().tolist()
        sampled_in_idx_cpu = sampled_in_idx.cpu().tolist()
        sampled_tokens_cpu = batch_next_token_ids.to(torch.int32).cpu().tolist()
        selected_logprobs_cpu = selected_logprobs.cpu().tolist()

        masks = []
        logprobs = []
        cursor = 0
        for i, should_return in enumerate(return_sampling_masks):
            mask_len = int(mask_lengths_cpu[i])
            row_ids = flat_ids_cpu[cursor : cursor + mask_len]
            cursor += mask_len
            if not sampled_in_idx_cpu[i]:
                row_ids.append(int(sampled_tokens_cpu[i]))
            if should_return:
                masks.append(row_ids)
                logprobs.append(float(selected_logprobs_cpu[i]))
            else:
                masks.append(None)
                logprobs.append(None)

        logits_output.next_token_sampling_mask_idx = masks
        logits_output.next_token_sampling_logprobs = logprobs

    def _sample_from_logprobs(
        self,
        logprobs: torch.Tensor,
        sampling_info: SamplingBatchInfo,
        positions: torch.Tensor,
    ) -> torch.Tensor:

        assert (
            sampling_info.sampling_seed is not None
        ), "sampling_seed is required for sampling from logprobs"
        sampled_index = multinomial_with_seed(
            logprobs, sampling_info.sampling_seed, positions
        )
        return sampled_index.view(-1).to(torch.int32)

    def _sample_from_logits(
        self,
        logits: torch.Tensor,
        sampling_info: SamplingBatchInfo,
        simple_sampling_case: bool,
        positions: torch.Tensor,
    ) -> torch.Tensor:

        if simple_sampling_case:
            probs = torch.softmax(logits, dim=-1)
            if sampling_info.sampling_seed is not None:
                probabilities = probs.to(torch.float64).log_()
                batch_next_token_ids = multinomial_with_seed(
                    probabilities, sampling_info.sampling_seed, positions
                ).view(-1)
            else:
                batch_next_token_ids = torch.multinomial(probs, num_samples=1).view(-1)
            return batch_next_token_ids.to(torch.int32)
        else:
            assert (
                self.use_ascend_backend
            ), "Only ascend backend supports sampling from logits"
            batch_next_token_ids = top_k_top_p_min_p_sampling_from_logits_ascend(
                logits,
                sampling_info.top_ks,
                sampling_info.top_ps,
                sampling_info.min_ps,
                sampling_info.need_min_p_sampling,
                sampling_info.sampling_seed,
                positions,
            )
            return batch_next_token_ids.to(torch.int32)

    def _forward_ascend_backend(
        self,
        logits: torch.Tensor,
        sampling_info: SamplingBatchInfo,
        simple_sampling_case: bool,
        return_logprob: bool,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        logits.div_(sampling_info.temperatures)
        batch_next_token_ids = self._sample_from_logits(
            logits, sampling_info, simple_sampling_case, positions
        )
        logprobs = None
        if return_logprob and not SGLANG_RETURN_ORIGINAL_LOGPROB:
            logprobs = torch.log_softmax(logits, dim=-1)
        return batch_next_token_ids, logprobs

    def _sync_token_ids_across_tp(
        self, batch_next_token_ids: torch.Tensor, sampling_info: SamplingBatchInfo
    ):
        if SYNC_TOKEN_IDS_ACROSS_TP or sampling_info.grammars:


            torch.distributed.all_reduce(
                batch_next_token_ids,
                op=dist.ReduceOp.MIN,
                group=self.tp_sync_group,
            )

    def compute_logprobs_only(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        top_logprobs_nums: List[int],
        token_ids_logprobs: List[List[int]],
    ) -> None:
        logprob_result = self.output_logprob_processor.compute_logprobs_only(
            next_token_logits=logits_output.next_token_logits,
            top_logprobs_nums=top_logprobs_nums,
            token_ids_logprobs=token_ids_logprobs,
            preprocess_fn=partial(self._preprocess_logits, sampling_info=sampling_info),
        )
        if logprob_result is not None:
            logprob_result.write_output_to(logits_output)


def register_sampler_backend(backend: str, factory: Callable[[], "Sampler"]) -> None:


    if not backend:
        raise ValueError("backend must be a non-empty string")

    from sglang.srt.server_args import SAMPLING_BACKEND_CHOICES

    if backend in _CUSTOM_SAMPLER_FACTORIES:
        logger.warning("Overriding existing sampler factory for backend '%s'", backend)
    SAMPLING_BACKEND_CHOICES.add(backend)
    _CUSTOM_SAMPLER_FACTORIES[backend] = factory


def create_sampler(backend: Optional[str] = None) -> "Sampler":


    server_args = get_server_args()
    backend = backend or (get_exec().kernel.sampling_backend if server_args else None)

    if backend in _CUSTOM_SAMPLER_FACTORIES:
        sampler = _CUSTOM_SAMPLER_FACTORIES[backend]()
        if not isinstance(sampler, Sampler):
            raise TypeError(
                f"Custom sampler factory for backend '{backend}' must return a Sampler"
            )
        return sampler

    if backend is None or backend in _BUILT_IN_SAMPLING_BACKENDS:
        return Sampler()

    raise ValueError(
        f"Unknown sampling backend '{backend}'. Register it via register_sampler_backend()."
    )


def top_k_top_p_min_p_sampling_from_probs_torch(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    need_min_p_sampling: bool,
    sampling_seed: Optional[torch.Tensor],
    positions: torch.Tensor,
):

    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[
        torch.arange(0, probs.shape[-1], device=probs.device).view(1, -1)
        >= top_ks.view(-1, 1)
    ] = 0.0
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0

    if need_min_p_sampling:

        assert (
            sampling_seed is None
        ), "With sampling seed, multinomial_with_seed will provide wrong results"
        min_p_thresholds = probs_sort[:, 0] * min_ps
        probs_sort[probs_sort < min_p_thresholds.view(-1, 1)] = 0.0

    if sampling_seed is None:
        sampled_index = torch.multinomial(probs_sort, num_samples=1)
    else:


        logprobs = probs_sort.to(torch.float64)
        del probs_sort
        logprobs.log_()
        sampled_index = multinomial_with_seed(logprobs, sampling_seed, positions)


    probs_idx = probs_idx.to(torch.int32)
    batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index).view(-1)
    return batch_next_token_ids


def top_k_top_p_min_p_sampling_from_logits_ascend(
    logits: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    need_min_p_sampling: bool,
    sampling_seed: Optional[torch.Tensor],
    positions: torch.Tensor,
):


    if hasattr(torch_npu, "npu_top_k_top_p") and torch.all(
        (top_ks <= 1024) & (top_ks >= 1)
    ):
        logits_top_k_top_p = torch_npu.npu_top_k_top_p(logits, top_ps, top_ks)
        probs_top_k_top_p = logits_top_k_top_p.softmax(dim=-1)

        if need_min_p_sampling:
            min_p_thresholds = probs_top_k_top_p.max(dim=-1) * min_ps
            min_p_mask = probs_top_k_top_p < min_p_thresholds.view(-1, 1)
            probs_top_k_top_p.masked_fill_(min_p_mask, 0.0)

        if sampling_seed is None:
            batch_next_token_ids = torch.multinomial(probs_top_k_top_p, num_samples=1)
        else:
            logprobs_top_k_top_p = probs_top_k_top_p.to(
                torch.float64
            )
            del probs_top_k_top_p
            logprobs_top_k_top_p.log_()
            batch_next_token_ids = multinomial_with_seed(
                logprobs_top_k_top_p, sampling_seed, positions
            )
    else:
        probs = torch.softmax(logits, dim=-1)
        probs_sort, probs_idx = probs.sort(dim=-1, descending=True)


        topk_all_mask = top_ks == TOP_K_ALL
        top_ks.masked_fill_(topk_all_mask, probs.shape[1])
        top_k_mask = torch.arange(0, probs.shape[-1], device=probs.device).view(
            1, -1
        ) >= top_ks.view(-1, 1)
        probs_sort.masked_fill_(top_k_mask, 0.0)

        probs_sum = torch.cumsum(probs_sort, dim=-1)
        top_p_mask = probs_sum - probs_sort > top_ps.view(-1, 1)
        probs_sort.masked_fill_(top_p_mask, 0.0)

        if need_min_p_sampling:
            min_p_thresholds = probs_sort[:, 0] * min_ps
            min_p_mask = probs_sort < min_p_thresholds.view(-1, 1)
            probs_sort.masked_fill_(min_p_mask, 0.0)

        if sampling_seed is None:
            sampled_index = torch.multinomial(probs_sort, num_samples=1)
        else:
            logprobs = probs_sort.to(
                torch.float64
            )
            del probs_sort
            logprobs.log_()
            sampled_index = multinomial_with_seed(logprobs, sampling_seed, positions)
        probs_idx = probs_idx.to(torch.int32)
        batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index)

    return batch_next_token_ids.view(-1)


@torch.compile(dynamic=True, disable=is_npu())
def multinomial_with_seed(
    logprobs: torch.Tensor, seed: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:

    n, m = logprobs.shape
    seed = seed.to(torch.uint64)
    col_indices = torch.arange(m, device=logprobs.device)
    hashed = murmur_hash32(seed, positions, col_indices)


    x = hashed.to(torch.float64) / torch.iinfo(torch.uint32).max


    x.log_().clamp_(min=torch.finfo(x.dtype).min, max=-(2.0**-32)).neg_()
    x.log_().neg_()


    x.add_(logprobs.to(torch.float64))

    return torch.argmax(x, dim=1, keepdim=True)


def sampling_from_probs_torch(
    probs: torch.Tensor,
    sampling_seed: Optional[torch.Tensor] = None,
    positions: Optional[torch.Tensor] = None,
):

    if sampling_seed is None:
        sampled_index = torch.multinomial(probs, num_samples=1)
    else:

        sampled_index = multinomial_with_seed(
            torch.log(probs), sampling_seed, positions
        )
    batch_next_token_ids = sampled_index.view(-1).to(torch.int32)
    return batch_next_token_ids


def top_p_normalize_probs_torch(
    probs: torch.Tensor,
    top_ps: torch.Tensor,
):

    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    return torch.zeros_like(probs_sort).scatter_(-1, probs_idx, probs_sort)


def apply_custom_logit_processor(
    logits: torch.Tensor,
    sampling_batch_info: SamplingBatchInfo,
    num_tokens_in_batch: int = 1,
):


    assert logits.shape[0] == len(sampling_batch_info) * num_tokens_in_batch, (
        f"The batch size of logits ({logits.shape[0]}) does not match the batch size of "
        f"sampling_batch_info ({len(sampling_batch_info)}) x num_tokens_in_batch "
        f"({num_tokens_in_batch})"
    )

    for _, (
        processor,
        batch_mask,
    ) in sampling_batch_info.custom_logit_processor.items():

        batch_indices = batch_mask.nonzero(as_tuple=True)[0]

        assert batch_mask.shape[0] == len(sampling_batch_info), (
            f"The number of batch mask ({batch_mask.shape[0]}) does not match the number of "
            f"sampling_batch_info ({len(sampling_batch_info)})"
        )
        batch_mask = torch.repeat_interleave(batch_mask, num_tokens_in_batch)


        logits[batch_mask] = processor(
            logits[batch_mask],
            [sampling_batch_info.custom_params[i] for i in batch_indices],
        )

        logger.debug(
            f"Custom logit processor {processor.__class__.__name__} is applied."
        )
