from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_ar_scattered_sconv_module(
    dtype: torch.dtype,
    world_size: int,
    w: int,
    use_silu: bool,
    use_residual: bool,
) -> Module:
    args = make_cpp_args(dtype, world_size, w, use_silu, use_residual)
    return load_jit(
        "inkling_ar_scattered_sconv",
        *args,
        cuda_files=["inkling/inkling_ar_scattered_sconv.cuh"],
        cuda_wrappers=[
            ("ar_scattered_sconv", f"ArScatteredSconvKernel<{args}>::run"),
            ("ar_banded_sconv", f"ArBandedSconvKernel<{args}>::run"),
            ("ar_ssconv_norm_decode", f"SsconvNormDecodeKernel<{args}>::run"),
            ("ar_col_decode", f"ColDecodeKernel<{args}>::run"),
        ],
    )


def compile_inkling_ar_scattered_sconv(
    dtype: torch.dtype,
    world_size: int,
    w: int,
    use_silu: bool,
    use_residual: bool,
) -> None:

    _jit_ar_scattered_sconv_module(dtype, world_size, w, use_silu, use_residual)


def inkling_ar_scattered_sconv(
    in_buffer: torch.Tensor,
    x_scratch: torch.Tensor,
    sconv_cache: torch.Tensor,
    safe_idx: torch.Tensor,
    cache_mask: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    cu: torch.Tensor,
    si: torch.Tensor,
    weight: torch.Tensor,
    track_rows: torch.Tensor,
    track_mask: torch.Tensor,
    track_dst: torch.Tensor,
    mc_in: int,
    mc_out: int,
    flag_ptrs_dev: int,
    state_ptr: int,
    rank: int,
    world_size: int,
    *,
    activation: str | None,
    use_residual: bool,
    num_blocks: int = 0,
    block_size: int = 0,
    per_block_barrier: bool = False,
    track_from_cache: bool = False,
    out_local: torch.Tensor | None = None,
    norm_gamma: torch.Tensor | None = None,
    norm_residual: torch.Tensor | None = None,
    norm_out: torch.Tensor | None = None,
    norm_eps: float = 0.0,
    need_scratch: bool = True,
    use_stream: bool = False,
    stream_walk: int = 0,
    full_update: bool = False,
    cache_col0: int = 0,
) -> None:

    w = weight.shape[1]
    use_silu = activation in ("silu", "swish")
    module = _jit_ar_scattered_sconv_module(
        in_buffer.dtype, world_size, w, use_silu, use_residual
    )
    do_norm = norm_gamma is not None
    if do_norm:
        assert (
            out_local is not None and norm_residual is not None and norm_out is not None
        )
    else:
        empty = in_buffer.new_empty((0,))
        out_local = norm_gamma = norm_residual = norm_out = empty
    module.ar_scattered_sconv(
        in_buffer,
        x_scratch,
        sconv_cache,
        safe_idx,
        cache_mask,
        cache_indices,
        has_initial_state,
        cu,
        si,
        weight,
        track_rows,
        track_mask,
        track_dst,
        out_local,
        norm_gamma,
        norm_residual,
        norm_out,
        mc_in,
        mc_out,
        flag_ptrs_dev,
        state_ptr,
        rank,
        num_blocks,
        block_size,
        per_block_barrier,
        track_from_cache,
        norm_eps,
        need_scratch,
        use_stream,
        stream_walk,
        full_update,
        cache_col0,
    )


def inkling_ar_ssconv_norm_decode(
    in_partials: torch.Tensor,
    residual_in: torch.Tensor,
    residual_out: torch.Tensor,
    hs_out: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    sconv_cache: torch.Tensor,
    cache_indices: torch.Tensor,
    cache_mask: torch.Tensor,
    conv_weight_full: torch.Tensor,
    track_mask: torch.Tensor,
    track_indices: torch.Tensor,
    mc_stage: int,
    local_stage: int,
    mc_wstage: int,
    local_wstage: int,
    flag_ptrs_dev: int,
    state_ptr: int,
    rank: int,
    world_size: int,
    *,
    activation: str | None,
    use_residual: bool,
    vecs_per_thread: int = 0,
) -> None:

    w = conv_weight_full.shape[1]
    use_silu = activation in ("silu", "swish")
    module = _jit_ar_scattered_sconv_module(
        in_partials.dtype, world_size, w, use_silu, use_residual
    )
    module.ar_ssconv_norm_decode(
        in_partials,
        residual_in,
        residual_out,
        hs_out,
        norm_weight,
        norm_eps,
        sconv_cache,
        cache_indices,
        cache_mask,
        conv_weight_full,
        track_mask,
        track_indices,
        mc_stage,
        local_stage,
        mc_wstage,
        local_wstage,
        flag_ptrs_dev,
        state_ptr,
        rank,
        vecs_per_thread,
    )


def inkling_ar_col_decode(
    in_buffer: torch.Tensor,
    out_local: torch.Tensor,
    residual_in: torch.Tensor,
    residual_out: torch.Tensor,
    hs_out: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    sconv_cache: torch.Tensor,
    cache_indices: torch.Tensor,
    cache_mask: torch.Tensor,
    weight_shard: torch.Tensor,
    track_mask: torch.Tensor,
    track_dst: torch.Tensor,
    mc_in: int,
    mc_out: int,
    flag_ptrs_dev: int,
    state_ptr: int,
    rank: int,
    world_size: int,
    *,
    activation: str | None,
    use_residual: bool,
    vecs_per_thread: int = 0,
) -> None:

    w = weight_shard.shape[1]
    use_silu = activation in ("silu", "swish")
    module = _jit_ar_scattered_sconv_module(
        in_buffer.dtype, world_size, w, use_silu, use_residual
    )
    module.ar_col_decode(
        in_buffer,
        out_local,
        residual_in,
        residual_out,
        hs_out,
        norm_weight,
        norm_eps,
        sconv_cache,
        cache_indices,
        cache_mask,
        weight_shard,
        track_mask,
        track_dst,
        mc_in,
        mc_out,
        flag_ptrs_dev,
        state_ptr,
        rank,
        vecs_per_thread,
    )


def inkling_ar_banded_sconv(
    in_buffer: torch.Tensor,
    scratch: torch.Tensor,
    sconv_cache: torch.Tensor,
    safe_idx: torch.Tensor,
    cache_mask: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    cu: torch.Tensor,
    si: torch.Tensor,
    weight: torch.Tensor,
    track_rows: torch.Tensor,
    track_mask: torch.Tensor,
    track_dst: torch.Tensor,
    mc_in: int,
    mc_out: int,
    flag_ptrs_dev: int,
    state_ptr: int,
    rank: int,
    world_size: int,
    *,
    activation: str | None,
    use_residual: bool,
    num_blocks: int = 0,
    block_size: int = 0,
    per_block_barrier: bool = False,
    debug_phase: int = 0,
    mc_wstage: int = 0,
    local_wstage: int = 0,
) -> None:

    w = weight.shape[1]
    use_silu = activation in ("silu", "swish")
    module = _jit_ar_scattered_sconv_module(
        in_buffer.dtype, world_size, w, use_silu, use_residual
    )
    module.ar_banded_sconv(
        in_buffer,
        scratch,
        sconv_cache,
        safe_idx,
        cache_mask,
        cache_indices,
        has_initial_state,
        cu,
        si,
        weight,
        track_rows,
        track_mask,
        track_dst,
        mc_in,
        mc_out,
        flag_ptrs_dev,
        state_ptr,
        rank,
        num_blocks,
        block_size,
        per_block_barrier,
        debug_phase,
        mc_wstage,
        local_wstage,
    )
