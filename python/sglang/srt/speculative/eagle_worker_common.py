from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.kernels.ops.speculative.cache_locs import (
    assign_draft_cache_locs_contiguous,
)
from sglang.kernels.ops.speculative.eagle import fill_bonus_tokens_func
from sglang.srt.layers.logprob_processor import compute_spec_logprobs
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.eagle_utils import (
    TreeMaskMode,
    build_tree_kernel_efficient,
    eagle_prepare_for_verify,
    eagle_sample,
)
from sglang.srt.speculative.spec_utils import (
    GrammarTree,
    build_grammar_vocab_mask,
    commit_mamba_states_after_verify,
    move_accept_tokens_to_target_kvcache,
    record_stream_each,
    record_stream_for_v2_verify,
)
from sglang.srt.utils import is_cpu
from sglang.srt.utils.async_probe import (
    maybe_detect_inf,
    maybe_detect_nan,
    maybe_detect_oob,
)
from sglang.srt.utils.common import is_npu

_is_cpu = is_cpu()
_is_npu = is_npu()

if _is_cpu:
    from sgl_kernel import assign_draft_cache_locs_contiguous_cpu

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
        EAGLEDraftCudaGraphRunner,
    )
    from sglang.srt.speculative.eagle_info import EagleDraftExtendInput


def duplicate_prefix_tail_to_draft_branches(
    token_to_kv_pool,
    rows: torch.Tensor,
    prefix_base: torch.Tensor,
    last_page: torch.Tensor,
    num_new_pages: torch.Tensor,
    topk: int,
    page_size: int,
) -> None:

    if topk <= 1:
        return
    bs = rows.shape[0]
    page_off = torch.arange(page_size, device=rows.device, dtype=torch.int64)
    branches = torch.arange(1, topk, device=rows.device, dtype=torch.int64).view(
        1, topk - 1, 1
    )

    src_pos = (prefix_base.view(bs, 1, 1) + page_off.view(1, 1, page_size)).expand(
        bs, topk - 1, page_size
    )

    tgt_pos = (
        prefix_base.view(bs, 1, 1)
        + branches * (num_new_pages.view(bs, 1, 1) * page_size)
        + page_off.view(1, 1, page_size)
    )


    vmask = (page_off.view(1, 1, page_size) < last_page.view(bs, 1, 1)).expand(
        bs, topk - 1, page_size
    )
    src_slots = torch.gather(rows, 1, src_pos.reshape(bs, -1)).reshape(
        bs, topk - 1, page_size
    )[vmask]
    tgt_slots = torch.gather(rows, 1, tgt_pos.reshape(bs, -1)).reshape(
        bs, topk - 1, page_size
    )[vmask]
    if src_slots.numel() > 0:
        token_to_kv_pool.move_kv_cache(tgt_slots, src_slots)


def prepare_for_draft_extend(
    draft_extend_input: EagleDraftExtendInput,
    batch: ScheduleBatch,
    predict: torch.Tensor,
    num_draft_tokens: int,
    draft_model_runner: Any,
    cuda_graph_runner: Any,
    *,
    return_hidden_states_before_norm: bool,
    widened_out_cache_loc: Optional[torch.Tensor] = None,
    widened_positions: Optional[torch.Tensor] = None,
):
    bs = len(batch.seq_lens)


    num_front_tokens = draft_extend_input.num_front_tokens
    widen = num_front_tokens > 0 and not batch.forward_mode.is_idle()
    front_offset = num_front_tokens if widen else 0
    num_window_tokens = num_draft_tokens + front_offset
    extend_num_tokens = bs * num_window_tokens

    gpu_only = batch.seq_lens_cpu is None

    batch.spec_info = draft_extend_input


    if widen:
        assert widened_out_cache_loc is not None and widened_positions is not None
        batch.input_ids = predict.new_zeros((extend_num_tokens,))
        batch.out_cache_loc = widened_out_cache_loc

        draft_extend_input.positions = widened_positions

        if draft_extend_input.hidden_states is not None:
            draft_extend_input.hidden_states = (
                draft_extend_input.hidden_states.new_empty(
                    (extend_num_tokens, draft_extend_input.hidden_states.shape[1])
                )
            )
    else:
        batch.input_ids = predict
    maybe_detect_oob(
        batch.input_ids,
        0,
        batch.model_config.vocab_size,
        "v2 prepare_for_draft_extend input_ids",
    )


    if gpu_only:
        batch.prefix_lens = (batch.seq_lens - front_offset).clamp(min=0).to(torch.int32)
        batch.extend_lens = torch.full(
            (bs,), num_window_tokens, dtype=torch.int32, device=batch.seq_lens.device
        )
    else:
        batch.prefix_lens = [
            max(int(x) - front_offset, 0) for x in batch.seq_lens_cpu.tolist()
        ]
        batch.extend_lens = [num_window_tokens] * bs
    batch.extend_num_tokens = extend_num_tokens
    capture_mode = (
        CaptureHiddenMode.NULL
        if draft_model_runner.spec_algorithm.is_standalone()
        else CaptureHiddenMode.FULL
    )
    batch.forward_mode = (
        ForwardMode.IDLE
        if batch.forward_mode.is_idle()
        else ForwardMode.DRAFT_EXTEND_V2
    )
    forward_batch = ForwardBatch.init_new(
        batch,
        draft_model_runner,
        capture_hidden_mode=capture_mode,
        return_hidden_states_before_norm=return_hidden_states_before_norm,
    )


    forward_batch.seq_lens = forward_batch.seq_lens + num_draft_tokens
    if not gpu_only:
        forward_batch.seq_lens_cpu = forward_batch.seq_lens_cpu + num_draft_tokens
        forward_batch.seq_lens_sum = int(forward_batch.seq_lens_cpu.sum())
    else:


        forward_batch.extend_seq_lens_cpu = [num_window_tokens] * bs
    can_run_decode_cuda_graph = cuda_graph_runner and cuda_graph_runner.can_run_graph(
        forward_batch
    )
    if not batch.forward_mode.is_idle() and not can_run_decode_cuda_graph:
        draft_model_runner.attn_backend.init_forward_metadata(forward_batch)


        if not is_npu() or can_run_decode_cuda_graph:
            forward_batch.mark_forward_metadata_ready()
    return forward_batch


def prepare_for_draft(
    draft_input: EagleDraftInput,
    req_to_token_pool: ReqToTokenPool,
    batch: ScheduleBatch,
    cuda_graph_runner: EAGLEDraftCudaGraphRunner,
    draft_model_runner: ModelRunner,
    topk: int,
    num_steps: int,
):

    if not batch.forward_mode.is_idle():
        bs = len(batch.seq_lens)


        page_size = batch.token_to_kv_pool_allocator.page_size
        if page_size == 1 or topk == 1:
            batch.out_cache_loc = torch.empty(
                (bs * topk * num_steps,),
                dtype=torch.int64,
                device=batch.device,
            )
            if _is_cpu:
                assign_draft_cache_locs_contiguous_cpu(
                    batch.req_pool_indices,
                    req_to_token_pool.req_to_token,
                    batch.seq_lens,
                    batch.out_cache_loc,
                    req_to_token_pool.req_to_token.shape[1],
                    topk,
                    num_steps,
                )
            else:

                assign_draft_cache_locs_contiguous[(bs,)](
                    batch.req_pool_indices,
                    req_to_token_pool.req_to_token,
                    batch.seq_lens,
                    batch.out_cache_loc,
                    req_to_token_pool.req_to_token.shape[1],
                    topk,
                    num_steps,
                )
        else:


            rows = req_to_token_pool.req_to_token[batch.req_pool_indices.long()]
            seq_lens = batch.seq_lens.to(torch.int64)
            last_page = seq_lens % page_size
            prefix_base = seq_lens - last_page
            num_new_pages = (last_page + num_steps + page_size - 1) // page_size
            topk_ids = torch.arange(topk, device=rows.device, dtype=torch.int64).view(
                1, topk
            )
            starts = (
                prefix_base.view(bs, 1)
                + topk_ids * (num_new_pages.view(bs, 1) * page_size)
                + last_page.view(bs, 1)
            )
            steps = torch.arange(num_steps, device=rows.device, dtype=torch.int64).view(
                1, 1, num_steps
            )
            pos = (starts.view(bs, topk, 1) + steps).reshape(bs, topk * num_steps)
            batch.out_cache_loc = torch.gather(rows, 1, pos).reshape(-1).contiguous()


            duplicate_prefix_tail_to_draft_branches(
                draft_model_runner.token_to_kv_pool,
                rows,
                prefix_base,
                last_page,
                num_new_pages,
                topk,
                page_size,
            )


    draft_input.num_tokens_per_req = topk
    draft_input.num_tokens_for_logprob_per_req = topk
    capture_mode = (
        CaptureHiddenMode.NULL
        if draft_model_runner.spec_algorithm.is_standalone()
        else CaptureHiddenMode.LAST
    )
    draft_input.positions = batch.seq_lens.repeat_interleave(topk, dim=0)
    forward_batch = ForwardBatch.init_new(
        batch,
        draft_model_runner,
        capture_hidden_mode=capture_mode,
        return_hidden_states_before_norm=False,
    )
    can_run_decode_cuda_graph = cuda_graph_runner and cuda_graph_runner.can_run_graph(
        forward_batch
    )
    return forward_batch, can_run_decode_cuda_graph


def build_eagle_verify_input(
    batch: ScheduleBatch,
    draft_input: EagleDraftInput,
    parent_list: torch.Tensor,
    top_scores_index: torch.Tensor,
    draft_tokens: torch.Tensor,
    draft_probs: Optional[torch.Tensor],
    *,
    target_worker: TpModelWorker,
    topk: int,
    num_steps: int,
    num_draft_tokens: int,
    tree_mask_mode: TreeMaskMode,
    device: str,
) -> EagleVerifyInput:

    if batch.forward_mode.is_idle():
        return EagleVerifyInput.create_idle_input(
            topk,
            num_steps,
            num_draft_tokens,
            device,
        )


    bs = batch.seq_lens.shape[0]
    target_attn_backend = target_worker.model_runner.attn_backend
    verify_mask = target_attn_backend.verify_mask
    if verify_mask is None:
        tree_mask_buf, mask_mode, fill_mask = None, tree_mask_mode, True
    else:
        mask_mode, fill_mask = verify_mask.mode, verify_mask.is_read
        tree_mask_buf = verify_mask.buffer if verify_mask.fits(bs) else None


    seq_lens_sum = batch.seq_lens_sum
    if seq_lens_sum is None:
        if tree_mask_buf is not None or mask_mode == TreeMaskMode.QLEN_ONLY:

            seq_lens_sum = 0
        else:
            seq_lens_sum = bs * target_attn_backend.max_context_len

    (
        tree_mask,
        position,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        draft_tokens,
    ) = build_tree_kernel_efficient(
        draft_input.bonus_tokens,
        parent_list,
        top_scores_index,
        draft_tokens,
        batch.seq_lens,
        seq_lens_sum,
        topk,
        num_steps,
        num_draft_tokens,
        mask_mode,
        tree_mask_buf,
        fill_prefix_mask=fill_mask,
    )

    return EagleVerifyInput(
        draft_token=draft_tokens,
        custom_mask=tree_mask,
        positions=position,
        retrieve_index=retrieve_index,
        retrieve_next_token=retrieve_next_token,
        retrieve_next_sibling=retrieve_next_sibling,
        retrieve_cum_len=None,
        spec_steps=num_steps,
        topk=topk,
        draft_token_num=num_draft_tokens,
        capture_hidden_mode=None,
        seq_lens_sum=None,
        seq_lens_cpu=None,
        draft_probs=draft_probs,
    )


def _finalize_accept_tree_path(
    batch: ScheduleBatch,
    accept_index: torch.Tensor,
    accept_lens: torch.Tensor,
    predict: torch.Tensor,
    logits_output: Any,
    bs: int,
    *,
    token_to_kv_pool_allocator: Any,
    num_draft_tokens: int,
) -> torch.Tensor:

    move_accept_tokens_to_target_kvcache(
        batch, accept_index, accept_lens - 1, token_to_kv_pool_allocator
    )
    predict = _compact_accept_to_front(
        predict, accept_index, bs, num_draft_tokens=num_draft_tokens
    )
    if logits_output.hidden_states is not None:
        logits_output.hidden_states = _compact_accept_to_front(
            logits_output.hidden_states,
            accept_index,
            bs,
            num_draft_tokens=num_draft_tokens,
        )
    return predict


def _compact_accept_to_front(
    x: torch.Tensor,
    accept_index: torch.Tensor,
    bs: int,
    *,
    num_draft_tokens: int,
) -> torch.Tensor:

    nd = num_draft_tokens
    s1 = accept_index.shape[1]
    safe = accept_index.to(torch.int64).clamp(min=0).reshape(-1)
    gathered = x[safe]
    out = x.clone()
    out.view(bs, nd, *x.shape[1:])[:, :s1] = gathered.view(bs, s1, *x.shape[1:])
    return out


def run_eagle_verify(
    batch: ScheduleBatch,
    *,
    target_worker: TpModelWorker,
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: Any,
    plan_stream: Any,
    plan_stream_ctx: Any,
    topk: int,
    num_draft_tokens: int,
    device: str,
    metadata_ready_pre_pad: bool,
    finalize_tree_path: bool,
    grammar_barrier=None,
) -> GenerationBatchResult:

    fwd_stream = torch.get_device_module(device).current_stream()
    verify_input: EagleVerifyInput = batch.spec_info
    record_stream_for_v2_verify(batch, verify_input, fwd_stream)

    bs = len(batch.seq_lens)


    with plan_stream_ctx:
        verify_forward_batch, can_run_cuda_graph = eagle_prepare_for_verify(
            verify_input,
            req_to_token_pool,
            batch,
            target_worker,
        )


    record_stream_each((batch.input_ids, batch.out_cache_loc), fwd_stream)


    if plan_stream:
        torch.get_device_module(device).current_stream().wait_stream(plan_stream)
        if (
            _is_npu
            and target_worker.model_runner.model_config.model_is_mrope
            and batch.spec_info is not None
            and getattr(batch.spec_info, "positions", None) is not None
            and not batch.forward_mode.is_idle()
        ):


            verify_forward_batch.compute_spec_mrope_positions(
                target_worker.model_runner, batch
            )


        target_worker.model_runner.attn_backend.update_verify_buffers_to_fill_after_draft(
            verify_input,
            (
                target_worker.model_runner.decode_cuda_graph_runner.bs
                if can_run_cuda_graph
                else None
            ),
        )


    grammar_tree = (
        GrammarTree.from_device(
            verify_input.retrieve_next_token,
            verify_input.retrieve_next_sibling,
            verify_input.draft_token.view(verify_input.retrieve_next_token.shape),
        )
        if batch.has_grammar
        else None
    )

    if metadata_ready_pre_pad:


        if not _is_npu or can_run_cuda_graph:
            verify_forward_batch.mark_forward_metadata_ready()


    forward_batch_output = target_worker.forward_batch_generation(
        batch=None,
        forward_batch=verify_forward_batch,
        is_verify=True,
    )
    logits_output = forward_batch_output.logits_output


    grammar_mask = None
    if batch.has_grammar:
        grammar_mask = build_grammar_vocab_mask(
            reqs=batch.reqs,
            tree=grammar_tree,
            sampling_info=batch.sampling_info,
            device=verify_input.retrieve_next_token.device,
            barrier=grammar_barrier,
        )


    maybe_detect_nan(logits_output.next_token_logits, "verify: target model logits")
    maybe_detect_inf(logits_output.next_token_logits, "verify: target model logits")
    (
        predict,
        accept_lens,
        accept_index,
    ) = eagle_sample(verify_input, batch, logits_output, grammar_mask)
    new_seq_lens = batch.seq_lens + accept_lens
    clear_unaccepted_c128 = getattr(
        token_to_kv_pool_allocator.get_kvcache(),
        "clear_unaccepted_c128_draft_states",
        None,
    )
    if clear_unaccepted_c128 is not None and not batch.forward_mode.is_idle():
        clear_unaccepted_c128(
            batch.req_pool_indices,
            batch.seq_lens,
            accept_lens,
            num_draft_tokens,
        )


    commit_mamba_states_after_verify(
        target_worker,
        batch,
        accept_lens,
        accept_index,
        num_draft_tokens,
    )

    if not batch.forward_mode.is_idle():
        accept_tokens = predict[accept_index]
        bonus_tokens = torch.empty_like(accept_lens, dtype=torch.int32)


        fill_bonus_tokens_func(
            accept_tokens,
            accept_lens,
            bonus_tokens,
            accept_index.shape[1],
            bs,
        )
    else:
        bonus_tokens = torch.empty((0,), device=device, dtype=torch.int32)

    if batch.return_logprob and not batch.forward_mode.is_idle():
        compute_spec_logprobs(batch, logits_output, predict, accept_index=accept_index)

    if finalize_tree_path and not batch.forward_mode.is_idle() and topk > 1:


        predict = _finalize_accept_tree_path(
            batch,
            accept_index,
            accept_lens,
            predict,
            logits_output,
            bs,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            num_draft_tokens=num_draft_tokens,
        )

    next_draft_input = EagleDraftInput(bonus_tokens=bonus_tokens)


    return GenerationBatchResult(
        logits_output=logits_output,
        next_token_ids=predict,
        can_run_cuda_graph=can_run_cuda_graph,
        speculative_num_draft_tokens=num_draft_tokens,
        next_draft_input=next_draft_input,
        accept_lens=accept_lens,
        new_seq_lens=new_seq_lens,
        routed_experts_output=forward_batch_output.routed_experts_output,
        indexer_topk_output=forward_batch_output.indexer_topk_output,
        extra_keep_alive_refs=[verify_forward_batch],
    )
