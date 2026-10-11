from __future__ import annotations

import math

import torch

from .context_feedback import ChunkAttentionSummary

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:
    triton = None
    tl = None
    libdevice = None


if triton is not None:

    @triton.jit
    def _request_forest_chunk_lse_kernel(
        queries,
        key_cache,
        token_indices,
        seq_lens,
        active_mask,
        partial_lse,
        softmax_scale,
        REQUEST_COUNT: tl.constexpr,
        MAX_PREFIX_TOKENS: tl.constexpr,
        QUERY_HEADS: tl.constexpr,
        KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        CHUNK_SIZE: tl.constexpr,
        CHUNK_COUNT: tl.constexpr,
        GQA_GROUP_SIZE: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_G: tl.constexpr,
        LOGIT_CAP: tl.constexpr,
    ):
        program = tl.program_id(0)
        kv_head = program % KV_HEADS
        request_chunk = program // KV_HEADS
        chunk = request_chunk % CHUNK_COUNT
        request = request_chunk // CHUNK_COUNT

        chunk_start = chunk * CHUNK_SIZE
        seq_len = tl.load(seq_lens + request)
        is_active = tl.load(active_mask + request)
        chunk_width = tl.minimum(CHUNK_SIZE, MAX_PREFIX_TOKENS - chunk_start)
        valid_count = tl.maximum(0, tl.minimum(chunk_width, seq_len - chunk_start))
        has_tokens = (valid_count > 0) & is_active

        offsets_t = tl.arange(0, BLOCK_T)
        offsets_d = tl.arange(0, BLOCK_D)
        offsets_g = tl.arange(0, BLOCK_G)
        valid_t = (offsets_t < valid_count) & (offsets_t < CHUNK_SIZE) & is_active
        slots = tl.load(
            token_indices
            + request * MAX_PREFIX_TOKENS
            + chunk_start
            + offsets_t,
            mask=valid_t,
            other=0,
        ).to(tl.int64)

        query_heads = kv_head * GQA_GROUP_SIZE + offsets_g
        valid_g = (offsets_g < GQA_GROUP_SIZE) & (query_heads < QUERY_HEADS)
        q = tl.load(
            queries
            + request * QUERY_HEADS * HEAD_DIM
            + query_heads[:, None] * HEAD_DIM
            + offsets_d[None, :],
            mask=valid_g[:, None] & (offsets_d[None, :] < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        k = tl.load(
            key_cache
            + slots[:, None] * KV_HEADS * HEAD_DIM
            + kv_head * HEAD_DIM
            + offsets_d[None, :],
            mask=valid_t[:, None] & (offsets_d[None, :] < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)

        products = q[:, None, :] * k[None, :, :]
        logits = tl.sum(products, axis=2) * softmax_scale
        if LOGIT_CAP > 0.0:
            logits = LOGIT_CAP * libdevice.tanh(logits / LOGIT_CAP)
        logits = tl.where(valid_t[None, :], logits, -float("inf"))
        chunk_max = tl.max(logits, axis=1)
        safe_max = tl.where(has_tokens, chunk_max, 0.0)
        exp_sum = tl.sum(
            tl.where(
                valid_t[None, :], tl.exp(logits - safe_max[:, None]), 0.0
            ),
            axis=1,
        )
        lse = tl.where(has_tokens, safe_max + tl.log(exp_sum), -float("inf"))
        tl.store(
            partial_lse
            + (request * CHUNK_COUNT + chunk) * QUERY_HEADS
            + query_heads,
            lse,
            mask=valid_g,
        )

    @triton.jit
    def _request_forest_history_lse_kernel(
        partial_lse,
        denominator,
        CHUNK_COUNT: tl.constexpr,
        QUERY_HEADS: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        program = tl.program_id(0)
        request = program // QUERY_HEADS
        head = program % QUERY_HEADS
        chunks = tl.arange(0, BLOCK_C)
        values = tl.load(
            partial_lse + (request * CHUNK_COUNT + chunks) * QUERY_HEADS + head,
            mask=chunks < CHUNK_COUNT,
            other=-float("inf"),
        )
        maximum = tl.max(values, axis=0)
        has_value = maximum != -float("inf")
        safe_max = tl.where(has_value, maximum, 0.0)
        exp_sum = tl.sum(
            tl.where(chunks < CHUNK_COUNT, tl.exp(values - safe_max), 0.0),
            axis=0,
        )
        result = tl.where(has_value, safe_max + tl.log(exp_sum), -float("inf"))
        tl.store(denominator + request * QUERY_HEADS + head, result)

    @triton.jit
    def _request_forest_chunk_score_kernel(
        partial_lse,
        denominator,
        chunk_token_counts,
        scores,
        QUERY_HEADS: tl.constexpr,
        CHUNK_COUNT: tl.constexpr,
        BLOCK_H: tl.constexpr,
        DIVIDE_BY_COUNT: tl.constexpr,
    ):
        program = tl.program_id(0)
        request = program // CHUNK_COUNT
        chunk = program % CHUNK_COUNT
        heads = tl.arange(0, BLOCK_H)
        count = tl.load(chunk_token_counts + program)
        has_tokens = count > 0
        numerators = tl.load(
            partial_lse + program * QUERY_HEADS + heads,
            mask=heads < QUERY_HEADS,
            other=-float("inf"),
        )
        normalizers = tl.load(
            denominator + request * QUERY_HEADS + heads,
            mask=heads < QUERY_HEADS,
            other=0.0,
        )
        mass = tl.where(
            (heads < QUERY_HEADS) & has_tokens,
            tl.exp(numerators - normalizers),
            0.0,
        )
        score = tl.sum(mass, axis=0) / QUERY_HEADS
        if DIVIDE_BY_COUNT:
            score = score / tl.maximum(count, 1)
        tl.store(scores + program, tl.where(has_tokens, score, 0.0))


def _require_contiguous(name: str, value: torch.Tensor) -> None:
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def summarize_historical_chunks_triton(
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


    if triton is None:
        raise RuntimeError("Triton is required for the fused context summarizer")
    if queries.device.type != "cuda":
        raise ValueError("the Triton context summarizer requires CUDA tensors")
    if queries.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("queries must use fp16 or bf16")
    if key_cache.dtype != queries.dtype:
        raise TypeError("key_cache must share the query dtype")
    if queries.dim() != 3 or key_cache.dim() != 3:
        raise ValueError("queries and key_cache must be three-dimensional")
    if token_indices.dim() != 2 or seq_lens.dim() != 1:
        raise ValueError("token_indices must be a matrix and seq_lens a vector")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if not math.isfinite(float(softmax_scale)) or float(softmax_scale) < 0:
        raise ValueError("softmax_scale must be finite and non-negative")
    if not math.isfinite(float(logit_cap)) or float(logit_cap) < 0:
        raise ValueError("logit_cap must be finite and non-negative")
    if score_mode not in ("mean_attention", "attention_mass"):
        raise ValueError("score_mode must be 'mean_attention' or 'attention_mass'")

    request_count, query_heads, head_dim = queries.shape
    if token_indices.shape[0] != request_count or seq_lens.numel() != request_count:
        raise ValueError("history metadata needs one row per request")
    if key_cache.shape[2] != head_dim:
        raise ValueError("query and key head dimensions must match")
    kv_heads = key_cache.shape[1]
    if kv_heads < 1 or query_heads % kv_heads:
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
    for name, value in (
        ("queries", queries),
        ("key_cache", key_cache),
        ("token_indices", token_indices),
        ("seq_lens", seq_lens),
    ):
        _require_contiguous(name, value)

    if active_mask is None:
        active_mask = torch.ones(request_count, dtype=torch.bool, device=device)
    elif (
        active_mask.dim() != 1
        or active_mask.numel() != request_count
        or active_mask.dtype != torch.bool
        or active_mask.device != device
    ):
        raise ValueError("active_mask needs one bool entry per request")
    _require_contiguous("active_mask", active_mask)

    if full_lse is not None:
        if full_lse.shape != (request_count, query_heads):
            raise ValueError("full_lse must have shape [requests, query_heads]")
        if full_lse.device != device:
            raise ValueError("full_lse must share the query device")
        _require_contiguous("full_lse", full_lse)

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

    starts = torch.arange(
        chunk_count, dtype=seq_lens.dtype, device=device
    ).view(1, -1) * chunk_size
    chunk_token_counts = torch.clamp(
        seq_lens.view(-1, 1) - starts, min=0, max=chunk_size
    ).to(torch.int32)
    chunk_token_counts = torch.where(
        active_mask.view(-1, 1), chunk_token_counts, 0
    ).contiguous()
    valid_chunk_mask = chunk_token_counts > 0

    partial_lse = torch.empty(
        (request_count, chunk_count, query_heads),
        dtype=torch.float32,
        device=device,
    )
    group_size = query_heads // kv_heads
    block_t = triton.next_power_of_2(chunk_size)
    block_d = triton.next_power_of_2(head_dim)
    block_g = triton.next_power_of_2(group_size)
    _request_forest_chunk_lse_kernel[
        (request_count * chunk_count * kv_heads,)
    ](
        queries,
        key_cache,
        token_indices,
        seq_lens,
        active_mask,
        partial_lse,
        float(softmax_scale),
        REQUEST_COUNT=request_count,
        MAX_PREFIX_TOKENS=max_prefix_tokens,
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        HEAD_DIM=head_dim,
        CHUNK_SIZE=chunk_size,
        CHUNK_COUNT=chunk_count,
        GQA_GROUP_SIZE=group_size,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        BLOCK_G=block_g,
        LOGIT_CAP=float(logit_cap),
        num_warps=4,
    )

    if full_lse is None:
        denominator = torch.empty(
            (request_count, query_heads), dtype=torch.float32, device=device
        )
        _request_forest_history_lse_kernel[(request_count * query_heads,)](
            partial_lse,
            denominator,
            CHUNK_COUNT=chunk_count,
            QUERY_HEADS=query_heads,
            BLOCK_C=triton.next_power_of_2(chunk_count),
            num_warps=4,
        )
        normalization = "history_only"
    else:
        denominator = full_lse
        normalization = "full_attention"

    scores = torch.empty(
        (request_count, chunk_count), dtype=torch.float32, device=device
    )
    _request_forest_chunk_score_kernel[(request_count * chunk_count,)](
        partial_lse,
        denominator,
        chunk_token_counts,
        scores,
        QUERY_HEADS=query_heads,
        CHUNK_COUNT=chunk_count,
        BLOCK_H=triton.next_power_of_2(query_heads),
        DIVIDE_BY_COUNT=score_mode == "mean_attention",
        num_warps=1,
    )
    return ChunkAttentionSummary(
        scores=scores,
        valid_chunk_mask=valid_chunk_mask,
        chunk_token_counts=chunk_token_counts,
        normalization=normalization,
    )


__all__ = ["summarize_historical_chunks_triton"]
