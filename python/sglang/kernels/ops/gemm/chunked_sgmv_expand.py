from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.gemm.lora_tuning_config import get_lora_expand_config
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.srt.utils import cached_triton_kernel


@cached_triton_kernel(
    lambda _, kwargs: (kwargs["NUM_SLICES"], kwargs["BLOCK_M"], kwargs["OUTPUT_DIM"])
)
@triton.jit(do_not_specialize=["num_segs", "output_stride_0", "output_stride_1"])
def _chunked_lora_expand_kernel(

    x,
    weights,
    output,


    output_stride_0,
    output_stride_1,

    seg_indptr,
    weight_indices,
    lora_ranks,
    permutation,
    num_segs,

    scalings,

    slice_offsets,

    NUM_SLICES: tl.constexpr,
    OUTPUT_DIM: tl.constexpr,
    MAX_RANK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):

    x_stride_0: tl.constexpr = NUM_SLICES * MAX_RANK
    x_stride_1: tl.constexpr = 1

    w_stride_0: tl.constexpr = OUTPUT_DIM * MAX_RANK
    w_stride_1: tl.constexpr = MAX_RANK
    w_stride_2: tl.constexpr = 1

    pid_s = tl.program_id(axis=2)
    if pid_s >= num_segs:
        return

    seg_start = tl.load(seg_indptr + pid_s)
    seg_end = tl.load(seg_indptr + pid_s + 1)
    if seg_start == seg_end:
        return


    w_index = tl.load(weight_indices + pid_s)
    cur_rank = tl.load(lora_ranks + w_index)


    if cur_rank == 0:
        return

    slice_id = tl.program_id(axis=1)
    slice_start = tl.load(slice_offsets + slice_id)
    slice_end = tl.load(slice_offsets + slice_id + 1)

    scaling = tl.load(scalings + w_index)

    cur_rank = tl.minimum(MAX_RANK, cur_rank)


    s_offset_logical = tl.arange(0, BLOCK_M) + seg_start
    s_offset_physical = tl.load(
        permutation + s_offset_logical, mask=s_offset_logical < seg_end, other=0
    )


    pid_n = tl.program_id(axis=0)
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N + slice_start
    k_offset = tl.arange(0, BLOCK_K)

    x_ptrs = (
        x
        + slice_id * cur_rank * x_stride_1
        + (s_offset_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1)
    )
    w_ptrs = (weights + w_index * w_stride_0) + (
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
    )


    partial_sum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(cur_rank, BLOCK_K)):
        x_tile = tl.load(
            x_ptrs,
            mask=(s_offset_logical[:, None] < seg_end)
            & (k_offset[None, :] < cur_rank - k * BLOCK_K),
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(k_offset[:, None] < cur_rank - k * BLOCK_K)
            & (n_offset[None, :] < slice_end),
            other=0.0,
        )
        partial_sum += tl.dot(x_tile, w_tile)

        x_ptrs += BLOCK_K * x_stride_1
        w_ptrs += BLOCK_K * w_stride_2


    partial_sum *= scaling
    partial_sum = partial_sum.to(x.dtype.element_ty)
    output_ptr = output + (
        s_offset_physical[:, None] * output_stride_0
        + n_offset[None, :] * output_stride_1
    )
    output_mask = (s_offset_logical[:, None] < seg_end) & (
        n_offset[None, :] < slice_end
    )
    partial_sum += tl.load(output_ptr, mask=output_mask, other=0.0)
    tl.store(output_ptr, partial_sum, mask=output_mask)


def chunked_sgmv_lora_expand_forward(
    x: torch.Tensor,
    weights: torch.Tensor,
    batch_info: LoRABatchInfo,
    slice_offsets: torch.Tensor,
    max_slice_size: int,
    base_output: Optional[torch.Tensor],
) -> torch.Tensor:


    assert x.is_contiguous()
    assert weights.is_contiguous()
    assert len(x.shape) == 2
    assert len(weights.shape) == 3


    M = x.shape[0]
    input_dim = x.shape[1]
    OUTPUT_DIM = weights.shape[1]
    MAX_RANK = weights.shape[2]
    num_slices = len(slice_offsets) - 1
    assert input_dim == num_slices * MAX_RANK


    BLOCK_M = batch_info.max_len
    config = get_lora_expand_config(
        K=OUTPUT_DIM, R=MAX_RANK, num_slices=num_slices, chunk_size=BLOCK_M
    )
    BLOCK_K = config["BLOCK_K"]
    BLOCK_N = config["BLOCK_N"]

    num_segments = batch_info.num_segments
    segment_grid = (
        batch_info.weight_indices.shape[0]
        if batch_info.use_cuda_graph
        else num_segments
    )

    grid = (
        triton.cdiv(max_slice_size, BLOCK_N),
        num_slices,
        segment_grid,
    )

    if base_output is None:
        output = torch.zeros((M, OUTPUT_DIM), device=x.device, dtype=x.dtype)
    else:
        output = base_output


    extra_kwargs = {}
    if "num_warps" in config:
        extra_kwargs["num_warps"] = config["num_warps"]
    if "num_stages" in config:
        extra_kwargs["num_stages"] = config["num_stages"]
    if "maxnreg" in config:
        extra_kwargs["maxnreg"] = config["maxnreg"]

    _chunked_lora_expand_kernel[grid](
        x=x,
        weights=weights,
        output=output,
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        seg_indptr=batch_info.seg_indptr,
        weight_indices=batch_info.weight_indices,
        lora_ranks=batch_info.lora_ranks,
        permutation=batch_info.permutation,
        num_segs=segment_grid,
        scalings=batch_info.scalings,
        slice_offsets=slice_offsets,

        NUM_SLICES=num_slices,
        OUTPUT_DIM=OUTPUT_DIM,
        MAX_RANK=MAX_RANK,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        **extra_kwargs,
    )

    return output
