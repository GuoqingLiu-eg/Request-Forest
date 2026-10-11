# Copyright 2025 XunhaoLai. All rights reserved.

import logging
from typing import Callable, List, Optional, Tuple

import torch

from sglang.kernels.ops.attention.minimax_sparse.common.index import topk_index_reduce
from sglang.kernels.ops.attention.minimax_sparse.common.utils import get_cu_seqblocks
from sglang.kernels.ops.attention.minimax_sparse.decode.flash_with_topk_idx import (
    flash_decode_with_topk_idx,
)
from sglang.kernels.ops.attention.minimax_sparse.decode.topk_sparse import (
    flash_decode_with_gqa_share_sparse,
)
from sglang.kernels.ops.attention.minimax_sparse.prefill.flash_with_topk_idx import (
    flash_prefill_with_topk_index,
)
from sglang.kernels.ops.attention.minimax_sparse.prefill.topk_sparse import (
    flash_prefill_with_gqa_share_sparse,
)

logger = logging.getLogger(__name__)
_msa_fallback_warned = False


def _warn_msa_fallback(err: Exception) -> None:
    global _msa_fallback_warned
    if _msa_fallback_warned:
        return
    logger.warning(
        "MiniMax MSA backend is unavailable (%s); falling back to Triton sparse attention.",
        err,
    )
    _msa_fallback_warned = True


def minimax_sparse_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    sink: Optional[torch.Tensor],
    idx_q: torch.Tensor,
    idx_k_cache: torch.Tensor,
    idx_v_cache: Optional[
        torch.Tensor
    ],
    idx_sink: Optional[torch.Tensor],
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_size_q: int,
    block_size_k: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    sm_scale: Optional[float] = None,
    idx_sm_scale: Optional[float] = None,
    score_type: str = "max",
    disable_index_value: bool = False,
    use_msa: bool = False,
    cu_seqblocks_q: Optional[torch.Tensor] = None,
    max_seqblock_q: Optional[int] = None,
    all_seqblock_q: Optional[int] = None,
    seqlens_cpu: Optional[List[int]] = None,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    idx_q_scale: Optional[float] = None,
    idx_k_scale: Optional[float] = None,
    idx_v_scale: Optional[float] = None,
):

    if cu_seqblocks_q is None or max_seqblock_q is None or all_seqblock_q is None:
        cu_seqblocks_q, max_seqblock_q, all_seqblock_q, _, _, _ = get_cu_seqblocks(
            cu_seqlens, max_seqlen_q, block_size_q, block_size_k, seqlens_cpu
        )


    idx_o, topk_idx = flash_prefill_with_topk_index(
        q=idx_q,
        k_cache=idx_k_cache,
        v_cache=idx_v_cache,
        sink=idx_sink,
        req_to_token=req_to_token,
        slot_ids=slot_ids,
        cu_seqlens=cu_seqlens,
        seq_lens=seq_lens,
        prefix_lens=prefix_lens,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        block_size_q=block_size_q,
        block_size_k=block_size_k,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        sm_scale=idx_sm_scale,
        score_type=score_type,
        disable_index_value=disable_index_value,
        cu_seqblocks_q=cu_seqblocks_q,
        max_seqblock_q=max_seqblock_q,
        all_seqblock_q=all_seqblock_q,
        q_scale=idx_q_scale,
        k_scale=idx_k_scale,
        v_scale=idx_v_scale,
    )

    num_idx_heads = idx_q.shape[1]
    num_kv_heads = k_cache.shape[1]
    idx_group_size = num_idx_heads // num_kv_heads
    if idx_group_size > 1:
        topk_idx = topk_index_reduce(
            topk_idx.view(num_kv_heads, idx_group_size, -1, topk), dim=1
        )


    if use_msa and sink is None:
        from .msa import MSAUnavailableError, msa_sparse_prefill_main

        try:
            o = msa_sparse_prefill_main(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                topk_idx=topk_idx,
                req_to_token=req_to_token,
                slot_ids=slot_ids,
                cu_seqlens=cu_seqlens,
                seq_lens=seq_lens,
                prefix_lens=prefix_lens,
                block_size_k=block_size_k,
                sm_scale=sm_scale,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
            )
        except MSAUnavailableError as err:
            _warn_msa_fallback(err)
            o = flash_prefill_with_gqa_share_sparse(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                sink=sink,
                req_to_token=req_to_token,
                slot_ids=slot_ids,
                topk_idx=topk_idx,
                block_size_q=block_size_q,
                block_size_k=block_size_k,
                cu_seqlens=cu_seqlens,
                seq_lens=seq_lens,
                prefix_lens=prefix_lens,
                max_seqlen_q=max_seqlen_q,
                sm_scale=sm_scale,
                cu_seqblocks_q=cu_seqblocks_q,
                max_seqblock_q=max_seqblock_q,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
            )
    else:
        o = flash_prefill_with_gqa_share_sparse(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            sink=sink,
            req_to_token=req_to_token,
            slot_ids=slot_ids,
            topk_idx=topk_idx,
            block_size_q=block_size_q,
            block_size_k=block_size_k,
            cu_seqlens=cu_seqlens,
            seq_lens=seq_lens,
            prefix_lens=prefix_lens,
            max_seqlen_q=max_seqlen_q,
            sm_scale=sm_scale,
            cu_seqblocks_q=cu_seqblocks_q,
            max_seqblock_q=max_seqblock_q,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
    return idx_o, o


def minimax_sparse_decode(
    q: torch.Tensor,
    sink: Optional[torch.Tensor],
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    idx_q: torch.Tensor,
    idx_sink: Optional[torch.Tensor],
    idx_k_cache: torch.Tensor,
    idx_v_cache: Optional[
        torch.Tensor
    ],
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seqlen: int,
    block_size_q: int,
    block_size_k: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    sm_scale: Optional[float] = None,
    idx_sm_scale: Optional[float] = None,
    score_type: str = "max",
    disable_index_value: bool = False,
    dense_main_attn_fn: Optional[Callable] = None,
    page_size: int = 1,
    use_msa: bool = False,
    msa_kv_indices: Optional[
        torch.Tensor
    ] = None,
    msa_plan=None,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    idx_q_scale: Optional[float] = None,
    idx_k_scale: Optional[float] = None,
    idx_v_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:


    idx_o, topk_idx, real_seq_lens = flash_decode_with_topk_idx(
        q=idx_q,
        sink=idx_sink,
        k_cache=idx_k_cache,
        v_cache=idx_v_cache,
        req_to_token=req_to_token,
        seq_lens=seq_lens,
        max_seqlen=max_seqlen,
        slot_ids=slot_ids,
        block_size=block_size_k,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        sm_scale=idx_sm_scale,
        score_type=score_type,
        disable_index_value=disable_index_value,
        use_dense_main_attn=dense_main_attn_fn is not None,
        page_size=page_size,
        q_scale=idx_q_scale,
        k_scale=idx_k_scale,
        v_scale=idx_v_scale,
    )
    num_idx_heads = idx_q.shape[1]
    num_kv_heads = k_cache.shape[1]
    idx_group_size = num_idx_heads // num_kv_heads
    if dense_main_attn_fn is not None:

        assert idx_group_size == 1
        o = dense_main_attn_fn(q, topk_idx, real_seq_lens)
    else:

        if idx_group_size > 1:
            topk_idx = topk_index_reduce(
                topk_idx.view(num_kv_heads, idx_group_size, -1, topk), dim=1
            )


        if use_msa and sink is None:
            from .msa import MSAUnavailableError, msa_sparse_decode_main

            try:
                o = msa_sparse_decode_main(
                    q=q,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    topk_idx=topk_idx,
                    req_to_token=req_to_token,
                    slot_ids=slot_ids,
                    seq_lens=seq_lens,
                    block_size_k=block_size_k,
                    sm_scale=sm_scale,
                    kv_indices=msa_kv_indices,
                    plan=msa_plan,
                    q_scale=q_scale,
                    k_scale=k_scale,
                    v_scale=v_scale,
                )
            except MSAUnavailableError as err:
                _warn_msa_fallback(err)
                o = flash_decode_with_gqa_share_sparse(
                    q=q,
                    sink=sink,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    req_to_token=req_to_token,
                    seq_lens=seq_lens,
                    slot_ids=slot_ids,
                    block_size=block_size_k,
                    topk_idx=topk_idx,
                    sm_scale=sm_scale,
                    q_scale=q_scale,
                    k_scale=k_scale,
                    v_scale=v_scale,
                )
        else:
            o = flash_decode_with_gqa_share_sparse(
                q=q,
                sink=sink,
                k_cache=k_cache,
                v_cache=v_cache,
                req_to_token=req_to_token,
                seq_lens=seq_lens,
                slot_ids=slot_ids,
                block_size=block_size_k,
                topk_idx=topk_idx,
                sm_scale=sm_scale,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
            )
    return idx_o, o
