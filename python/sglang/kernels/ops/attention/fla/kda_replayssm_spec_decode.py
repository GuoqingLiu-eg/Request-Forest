# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def kda_replayssm_exact_fold_kernel(
    h0,
    rawv_cache,
    rawk_cache,
    gk_cache,
    beta_cache,
    ssm_state_indices,
    accept_lens,
    mamba_track_indices,
    mamba_steps_to_track,
    stride_state_slot: tl.constexpr,
    stride_rawv_slot: tl.constexpr,
    stride_rawk_slot: tl.constexpr,
    stride_gk_slot: tl.constexpr,
    stride_beta_slot: tl.constexpr,
    stride_state_layer: tl.constexpr,
    stride_rawv_layer: tl.constexpr,
    stride_rawk_layer: tl.constexpr,
    stride_gk_layer: tl.constexpr,
    stride_beta_layer: tl.constexpr,
    stride_indices: tl.constexpr,
    stride_accept: tl.constexpr,
    stride_track: tl.constexpr,
    stride_steps: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    MAX_CACHE_LEN: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    NULL_BLOCK_ID: tl.constexpr,
    HAS_TRACK: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_n = tl.program_id(1)


    i_hvl = tl.program_id(2)


    i_layer = (i_hvl // HV).to(tl.int64)
    i_hv = i_hvl % HV
    i_h = i_hv // (HV // H)

    h0 = h0 + i_layer * stride_state_layer
    rawv_cache = rawv_cache + i_layer * stride_rawv_layer
    rawk_cache = rawk_cache + i_layer * stride_rawk_layer
    gk_cache = gk_cache + i_layer * stride_gk_layer
    beta_cache = beta_cache + i_layer * stride_beta_layer

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices).to(tl.int64)
    if state_idx <= NULL_BLOCK_ID:
        return
    n_commit = tl.load(accept_lens + i_n * stride_accept).to(tl.int32)
    if n_commit <= 0:
        return


    if HAS_TRACK:
        track_idx = tl.load(mamba_track_indices + i_n * stride_track).to(tl.int64)
        track_step = tl.load(mamba_steps_to_track + i_n * stride_steps).to(tl.int32)
    else:
        track_idx = NULL_BLOCK_ID
        track_step = -1

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]


    p_h0 = (
        h0
        + state_idx * stride_state_slot
        + i_hv * V * K
        + o_v[None, :] * K
        + o_k[:, None]
    )
    b_h = tl.load(p_h0, mask=mask_h, other=0.0).to(tl.float32)

    for t in range(0, n_commit):
        phys = t.to(tl.int64)
        b_k = tl.load(
            rawk_cache
            + state_idx * stride_rawk_slot
            + (i_h * MAX_CACHE_LEN + phys) * K
            + o_k,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        b_v = tl.load(
            rawv_cache
            + state_idx * stride_rawv_slot
            + (i_hv * MAX_CACHE_LEN + phys) * V
            + o_v,
            mask=mask_v,
            other=0.0,
        ).to(tl.float32)
        b_gk = tl.load(
            gk_cache
            + state_idx * stride_gk_slot
            + (i_hv * MAX_CACHE_LEN + phys) * K
            + o_k,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        b_beta = tl.load(
            beta_cache + state_idx * stride_beta_slot + i_hv * MAX_CACHE_LEN + phys
        ).to(tl.float32)


        if USE_QK_L2NORM_IN_KERNEL:
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
        b_h *= tl.exp(b_gk[:, None])
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]


        if HAS_TRACK:
            if (t == track_step) and (track_idx > NULL_BLOCK_ID):
                tl.store(
                    h0
                    + track_idx * stride_state_slot
                    + i_hv * V * K
                    + o_v[None, :] * K
                    + o_k[:, None],
                    b_h.to(h0.dtype.element_ty),
                    mask=mask_h,
                )

    tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


def commit_kda_replayssm_spec(
    checkpoint_state: torch.Tensor,
    rawv_cache: torch.Tensor,
    rawk_cache: torch.Tensor,
    gk_cache: torch.Tensor,
    beta_cache: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    accept_lens: torch.Tensor,
    max_cache_len: int,
    num_k_heads: int,
    mamba_track_indices: torch.Tensor | None = None,
    mamba_steps_to_track: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    null_block_id: int = 0,
) -> None:

    num_slots, HV, V, K = checkpoint_state.shape
    B = ssm_state_indices.shape[0]
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 32)
    grid = (triton.cdiv(V, BV), B, HV)
    has_track = mamba_track_indices is not None and mamba_steps_to_track is not None
    if has_track:
        track_idx_t = mamba_track_indices
        steps_t = mamba_steps_to_track
        stride_track = track_idx_t.stride(0)
        stride_steps = steps_t.stride(0)
    else:
        track_idx_t = ssm_state_indices
        steps_t = accept_lens
        stride_track = 0
        stride_steps = 0
    kda_replayssm_exact_fold_kernel[grid](
        checkpoint_state,
        rawv_cache,
        rawk_cache,
        gk_cache,
        beta_cache,
        ssm_state_indices,
        accept_lens,
        track_idx_t,
        steps_t,
        checkpoint_state.stride(0),
        rawv_cache.stride(0),
        rawk_cache.stride(0),
        gk_cache.stride(0),
        beta_cache.stride(0),
        0,
        0,
        0,
        0,
        0,
        ssm_state_indices.stride(0),
        accept_lens.stride(0),
        stride_track,
        stride_steps,
        H=num_k_heads,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        MAX_CACHE_LEN=max_cache_len,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        NULL_BLOCK_ID=null_block_id,
        HAS_TRACK=has_track,
        num_warps=1,
        num_stages=3,
    )


def commit_kda_replayssm_spec_all_layers(
    checkpoint_state: torch.Tensor,
    rawv_cache: torch.Tensor,
    rawk_cache: torch.Tensor,
    gk_cache: torch.Tensor,
    beta_cache: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    accept_lens: torch.Tensor,
    max_cache_len: int,
    num_k_heads: int,
    mamba_track_indices: torch.Tensor | None = None,
    mamba_steps_to_track: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    null_block_id: int = 0,
) -> None:

    num_layers, num_slots, HV, V, K = checkpoint_state.shape
    B = ssm_state_indices.shape[0]
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 32)
    grid = (triton.cdiv(V, BV), B, HV * num_layers)
    has_track = mamba_track_indices is not None and mamba_steps_to_track is not None
    if has_track:
        track_idx_t = mamba_track_indices
        steps_t = mamba_steps_to_track
        stride_track = track_idx_t.stride(0)
        stride_steps = steps_t.stride(0)
    else:
        track_idx_t = ssm_state_indices
        steps_t = accept_lens
        stride_track = 0
        stride_steps = 0
    kda_replayssm_exact_fold_kernel[grid](
        checkpoint_state,
        rawv_cache,
        rawk_cache,
        gk_cache,
        beta_cache,
        ssm_state_indices,
        accept_lens,
        track_idx_t,
        steps_t,
        checkpoint_state.stride(1),
        rawv_cache.stride(1),
        rawk_cache.stride(1),
        gk_cache.stride(1),
        beta_cache.stride(1),
        checkpoint_state.stride(0),
        rawv_cache.stride(0),
        rawk_cache.stride(0),
        gk_cache.stride(0),
        beta_cache.stride(0),
        ssm_state_indices.stride(0),
        accept_lens.stride(0),
        stride_track,
        stride_steps,
        H=num_k_heads,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        MAX_CACHE_LEN=max_cache_len,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        NULL_BLOCK_ID=null_block_id,
        HAS_TRACK=has_track,
        num_warps=1,
        num_stages=3,
    )


def commit_kda_replayssm_after_verify(
    *,
    spec_state,
    state_batch_indices: torch.Tensor,
    accept_lens: torch.Tensor,
    last_correct_step_indices: torch.Tensor,
    mamba_track_indices: torch.Tensor | None = None,
    mamba_steps_to_track: torch.Tensor | None = None,
    null_block_id: int = -1,
) -> None:

    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
        fused_conv_window_scatter_with_mask,
    )

    L = spec_state.replayssm_rawv.shape[-2]
    num_k_heads = spec_state.replayssm_rawk.shape[2]
    commit_kda_replayssm_spec_all_layers(
        checkpoint_state=spec_state.temporal,
        rawv_cache=spec_state.replayssm_rawv,
        rawk_cache=spec_state.replayssm_rawk,
        gk_cache=spec_state.replayssm_g,
        beta_cache=spec_state.replayssm_beta,
        ssm_state_indices=state_batch_indices,
        accept_lens=accept_lens,
        max_cache_len=L,
        num_k_heads=num_k_heads,
        mamba_track_indices=mamba_track_indices,
        mamba_steps_to_track=mamba_steps_to_track,
        null_block_id=null_block_id,
    )


    for conv_states, interm_conv in zip(
        spec_state.conv, spec_state.intermediate_conv_window
    ):
        fused_conv_window_scatter_with_mask(
            conv_states, interm_conv, state_batch_indices, last_correct_step_indices
        )
        if mamba_track_indices is not None and mamba_steps_to_track is not None:
            fused_conv_window_scatter_with_mask(
                conv_states, interm_conv, mamba_track_indices, mamba_steps_to_track
            )
