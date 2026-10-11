from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(slots=True)
class FeedbackQuerySelection:


    query_rows: torch.Tensor
    active_mask: torch.Tensor


@dataclass(slots=True)
class ChunkAttentionSummary:


    scores: torch.Tensor
    valid_chunk_mask: torch.Tensor
    chunk_token_counts: torch.Tensor
    normalization: str


@dataclass(slots=True)
class ContextChunkSelection:


    selected_chunk_mask: torch.Tensor
    top_chunk_ids: torch.Tensor


def select_feedback_query_rows(
    accept_index: torch.Tensor,
    accept_lens: torch.Tensor,
    *,
    termination_reasons: torch.Tensor | None = None,
    eos_termination_code: int = 4,
) -> FeedbackQuerySelection:


    if not isinstance(accept_index, torch.Tensor) or accept_index.dim() != 2:
        raise ValueError("accept_index must be a two-dimensional Tensor")
    if not isinstance(accept_lens, torch.Tensor) or accept_lens.dim() != 1:
        raise ValueError("accept_lens must be a one-dimensional Tensor")
    if accept_lens.numel() != accept_index.shape[0]:
        raise ValueError("accept_lens needs one entry per request")
    if accept_lens.device != accept_index.device:
        raise ValueError("accept_index and accept_lens must share one device")
    if accept_index.dtype not in (torch.int32, torch.int64):
        raise TypeError("accept_index must have an integer index dtype")
    if accept_lens.dtype not in (torch.int32, torch.int64):
        raise TypeError("accept_lens must have an integer index dtype")

    active = accept_lens > 0
    if termination_reasons is not None:
        if (
            not isinstance(termination_reasons, torch.Tensor)
            or termination_reasons.dim() != 1
            or termination_reasons.numel() != accept_lens.numel()
            or termination_reasons.device != accept_lens.device
        ):
            raise ValueError(
                "termination_reasons needs one entry per request on the same device"
            )
        active = active & (termination_reasons != eos_termination_code)


    columns = torch.clamp(accept_lens.to(torch.int64) - 1, min=0)
    rows = torch.gather(accept_index, 1, columns.view(-1, 1)).view(-1)
    rows = torch.where(active, rows, torch.full_like(rows, -1))
    return FeedbackQuerySelection(query_rows=rows, active_mask=active)


def gather_feedback_queries(
    packed_queries: torch.Tensor,
    selection: FeedbackQuerySelection,
) -> torch.Tensor:


    if not isinstance(packed_queries, torch.Tensor) or packed_queries.dim() not in (
        2,
        3,
    ):
        raise ValueError("packed_queries must be a two- or three-dimensional Tensor")
    if selection.query_rows.dim() != 1 or selection.active_mask.dim() != 1:
        raise ValueError("feedback query selection must contain vectors")
    if selection.query_rows.numel() != selection.active_mask.numel():
        raise ValueError("feedback query selection vectors must have equal length")
    if (
        selection.query_rows.device != packed_queries.device
        or selection.active_mask.device != packed_queries.device
    ):
        raise ValueError("packed queries and selection must share one device")

    safe_rows = torch.clamp(selection.query_rows.to(torch.int64), min=0)
    gathered = packed_queries.index_select(0, safe_rows)
    mask_shape = (selection.active_mask.numel(),) + (1,) * (gathered.dim() - 1)
    return torch.where(selection.active_mask.view(mask_shape), gathered, 0)


def summarize_historical_chunks(
    queries: torch.Tensor,
    key_cache: torch.Tensor,
    token_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    chunk_size: int,
    softmax_scale: float,
    full_lse: torch.Tensor | None = None,
    active_mask: torch.Tensor | None = None,
    score_mode: str = "mean_attention",
    logit_cap: float = 0.0,
) -> ChunkAttentionSummary:


    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if not math.isfinite(float(softmax_scale)) or float(softmax_scale) < 0:
        raise ValueError("softmax_scale must be finite and non-negative")
    if not math.isfinite(float(logit_cap)) or float(logit_cap) < 0:
        raise ValueError("logit_cap must be finite and non-negative")
    if score_mode not in ("mean_attention", "attention_mass"):
        raise ValueError("score_mode must be 'mean_attention' or 'attention_mass'")
    if queries.dim() != 3 or key_cache.dim() != 3:
        raise ValueError("queries and key_cache must be three-dimensional")
    if token_indices.dim() != 2 or seq_lens.dim() != 1:
        raise ValueError("token_indices must be a matrix and seq_lens a vector")

    request_count, query_heads, head_dim = queries.shape
    if token_indices.shape[0] != request_count or seq_lens.numel() != request_count:
        raise ValueError("history metadata needs one row per request")
    if key_cache.shape[2] != head_dim:
        raise ValueError("query and key head dimensions must match")
    kv_heads = key_cache.shape[1]
    if kv_heads < 1 or query_heads % kv_heads != 0:
        raise ValueError("query_heads must be an integer multiple of kv_heads")
    device = queries.device
    if (
        key_cache.device != device
        or token_indices.device != device
        or seq_lens.device != device
    ):
        raise ValueError("queries, keys, and history metadata must share one device")
    if token_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("token_indices must have an integer index dtype")
    if seq_lens.dtype not in (torch.int32, torch.int64):
        raise TypeError("seq_lens must have an integer index dtype")

    if active_mask is None:
        active_mask = torch.ones(request_count, dtype=torch.bool, device=device)
    elif (
        active_mask.dim() != 1
        or active_mask.numel() != request_count
        or active_mask.dtype != torch.bool
        or active_mask.device != device
    ):
        raise ValueError("active_mask needs one bool entry per request")

    if full_lse is not None:
        if full_lse.shape != (request_count, query_heads):
            raise ValueError("full_lse must have shape [requests, query_heads]")
        if full_lse.device != device:
            raise ValueError("full_lse must share the query device")

    max_prefix_tokens = token_indices.shape[1]
    chunk_count = (max_prefix_tokens + chunk_size - 1) // chunk_size
    if chunk_count == 0:
        empty_scores = torch.empty(
            (request_count, 0), dtype=torch.float32, device=device
        )
        return ChunkAttentionSummary(
            scores=empty_scores,
            valid_chunk_mask=torch.empty(
                (request_count, 0), dtype=torch.bool, device=device
            ),
            chunk_token_counts=torch.empty(
                (request_count, 0), dtype=torch.int32, device=device
            ),
            normalization=(
                "full_attention" if full_lse is not None else "history_only"
            ),
        )

    query_f32 = queries.to(torch.float32)
    gqa_group_size = query_heads // kv_heads
    kv_head_for_query = torch.arange(
        query_heads, dtype=torch.int64, device=device
    ) // gqa_group_size
    partial_lse = torch.full(
        (request_count, chunk_count, query_heads),
        -torch.inf,
        dtype=torch.float32,
        device=device,
    )
    chunk_token_counts = torch.empty(
        (request_count, chunk_count), dtype=torch.int32, device=device
    )

    for chunk in range(chunk_count):
        start = chunk * chunk_size
        end = min(start + chunk_size, max_prefix_tokens)
        token_columns = torch.arange(start, end, dtype=seq_lens.dtype, device=device)
        valid_tokens = token_columns.view(1, -1) < seq_lens.view(-1, 1)
        valid_tokens = valid_tokens & active_mask.view(-1, 1)
        chunk_token_counts[:, chunk] = valid_tokens.sum(dim=1, dtype=torch.int32)


        slots = torch.where(
            valid_tokens,
            token_indices[:, start:end],
            torch.zeros((), dtype=token_indices.dtype, device=device),
        ).to(torch.int64)
        keys = key_cache.index_select(0, slots.reshape(-1)).view(
            request_count, end - start, kv_heads, head_dim
        )
        keys = keys.index_select(2, kv_head_for_query).to(torch.float32)
        logits = torch.einsum("rhd,rthd->rth", query_f32, keys)
        logits.mul_(float(softmax_scale))
        if logit_cap > 0:
            logits = float(logit_cap) * torch.tanh(logits / float(logit_cap))
        logits.masked_fill_(~valid_tokens.unsqueeze(-1), -torch.inf)
        partial_lse[:, chunk, :] = torch.logsumexp(logits, dim=1)

    if full_lse is None:
        denominator = torch.logsumexp(partial_lse, dim=1)
        normalization = "history_only"
    else:
        denominator = full_lse.to(torch.float32)
        normalization = "full_attention"

    chunk_mass_by_head = torch.exp(partial_lse - denominator.unsqueeze(1))
    valid_chunk_mask = chunk_token_counts > 0
    chunk_mass_by_head = torch.where(
        valid_chunk_mask.unsqueeze(-1), chunk_mass_by_head, 0
    )
    scores = chunk_mass_by_head.mean(dim=-1)
    if score_mode == "mean_attention":
        scores = scores / chunk_token_counts.clamp_min(1).to(torch.float32)
    scores = torch.where(valid_chunk_mask, scores, 0)
    return ChunkAttentionSummary(
        scores=scores,
        valid_chunk_mask=valid_chunk_mask,
        chunk_token_counts=chunk_token_counts,
        normalization=normalization,
    )


def select_context_chunks(
    summary: ChunkAttentionSummary,
    seq_lens: torch.Tensor,
    *,
    chunk_size: int,
    top_k: int,
    recent_window_tokens: int,
) -> ContextChunkSelection:


    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if type(top_k) is not int or top_k < 0:
        raise ValueError("top_k must be a non-negative integer")
    if type(recent_window_tokens) is not int or recent_window_tokens < 0:
        raise ValueError("recent_window_tokens must be a non-negative integer")
    scores = summary.scores
    valid = summary.valid_chunk_mask
    if scores.dim() != 2 or valid.shape != scores.shape:
        raise ValueError("chunk summary must contain equally shaped matrices")
    if seq_lens.dim() != 1 or seq_lens.numel() != scores.shape[0]:
        raise ValueError("seq_lens needs one entry per request")
    if seq_lens.device != scores.device or valid.device != scores.device:
        raise ValueError("chunk selection tensors must share one device")

    request_count, chunk_count = scores.shape
    selected = torch.zeros_like(valid)
    top_width = min(top_k, chunk_count)
    if top_width:
        ranked_scores = torch.where(valid, scores, -torch.inf)
        top_chunk_ids = torch.topk(
            ranked_scores, top_width, dim=1, sorted=True
        ).indices
        selected.scatter_(1, top_chunk_ids, True)
        selected &= valid
        top_ids_valid = torch.gather(valid, 1, top_chunk_ids)
        top_chunk_ids = torch.where(
            top_ids_valid, top_chunk_ids, torch.full_like(top_chunk_ids, -1)
        )
    else:
        top_chunk_ids = torch.empty(
            (request_count, 0), dtype=torch.int64, device=scores.device
        )

    if recent_window_tokens and chunk_count:
        chunk_starts = (
            torch.arange(chunk_count, dtype=seq_lens.dtype, device=scores.device)
            * chunk_size
        )
        recent_start = torch.clamp(seq_lens - recent_window_tokens, min=0)


        recent = (
            (chunk_starts.view(1, -1) + chunk_size > recent_start.view(-1, 1))
            & (chunk_starts.view(1, -1) < seq_lens.view(-1, 1))
        )
        selected |= recent & valid

    return ContextChunkSelection(
        selected_chunk_mask=selected,
        top_chunk_ids=top_chunk_ids,
    )


def next_feedback_interval(
    current_interval: int | None,
    measured_overhead_ratio: float,
    *,
    target_overhead_ratio: float = 0.02,
) -> int | None:


    allowed = (1, 2, 4, 8, None)
    if current_interval not in allowed:
        raise ValueError("current_interval must be one of 1, 2, 4, 8, or None")
    if (
        not math.isfinite(float(measured_overhead_ratio))
        or float(measured_overhead_ratio) < 0
    ):
        raise ValueError("measured_overhead_ratio must be finite and non-negative")
    if (
        not math.isfinite(float(target_overhead_ratio))
        or float(target_overhead_ratio) <= 0
    ):
        raise ValueError("target_overhead_ratio must be finite and positive")
    if measured_overhead_ratio <= target_overhead_ratio or current_interval is None:
        return current_interval
    return allowed[allowed.index(current_interval) + 1]


__all__ = [
    "ChunkAttentionSummary",
    "ContextChunkSelection",
    "FeedbackQuerySelection",
    "gather_feedback_queries",
    "next_feedback_interval",
    "select_context_chunks",
    "select_feedback_query_rows",
    "summarize_historical_chunks",
]
