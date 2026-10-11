import functools

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.gemm.trtllm_lora_temp.kernel_utils import (
    _resolve_token_positions,
    get_pdl_launch_metadata,
)
from sglang.srt.lora.trtllm_lora_temp.environ import lora_envs
from sglang.srt.lora.utils import LoRABatchInfo


@triton.jit
def _sgemm_lora_a_kernel(

    x,
    weights,
    output,

    N,
    K,
    stack_num,

    x_stride_0,
    x_stride_1,
    w_stride_0,
    w_stride_1,
    w_stride_2,
    output_stride_0,
    output_stride_1,

    seg_lens,
    seg_indptr,
    weight_indices,
    lora_ranks,
    sorted_token_ids,

    SORTED_BY_ADAPTER: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr = 1,
    ENABLE_PDL: tl.constexpr = False,
    PADDED_RANK: tl.constexpr = False,
):


    batch_id = tl.program_id(axis=1)
    w_index = tl.load(weight_indices + batch_id)
    rank = tl.load(lora_ranks + w_index)


    if rank == 0:
        return

    pid = tl.program_id(axis=0)

    pid_sk = pid % SPLIT_K
    pid_tile = pid // SPLIT_K

    seg_start = tl.load(seg_indptr + batch_id)
    seg_len = tl.load(seg_lens + batch_id)
    if seg_len == 0:
        return


    if not PADDED_RANK:
        N = tl.minimum(N, rank * stack_num)


    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_s = pid_tile // num_pid_n
    pid_n = pid_tile % num_pid_n
    if pid_s * BLOCK_S >= seg_len:
        return


    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = pid_sk * BLOCK_K + tl.arange(0, BLOCK_K)
    s_physical = _resolve_token_positions(
        sorted_token_ids, seg_start, s_offset, seg_len, SORTED_BY_ADAPTER
    )
    x_ptrs = x + (s_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1)
    w_ptrs = (weights + w_index * w_stride_0) + (
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
    )


    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()


    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        k_remaining = K - k * (BLOCK_K * SPLIT_K)
        x_tile = tl.load(
            x_ptrs,
            mask=(s_offset[:, None] < seg_len) & (k_offset[None, :] < k_remaining),
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(k_offset[:, None] < k_remaining) & (n_offset[None, :] < N),
            other=0.0,
        )
        partial_sum += tl.dot(x_tile, w_tile)

        x_ptrs += BLOCK_K * SPLIT_K * x_stride_1
        w_ptrs += BLOCK_K * SPLIT_K * w_stride_2


    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


    output_mask = (s_offset[:, None] < seg_len) & (n_offset[None, :] < N)
    output_ptr = output + (
        s_physical[:, None] * output_stride_0 + n_offset[None, :] * output_stride_1
    )
    if SPLIT_K == 1:
        tl.store(output_ptr, partial_sum.to(output.dtype.element_ty), mask=output_mask)
    else:
        tl.atomic_add(
            output_ptr,
            partial_sum.to(output.dtype.element_ty),
            mask=output_mask,
            sem="relaxed",
        )


@functools.lru_cache(maxsize=None)
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def sgemm_lora_a_fwd(
    x: torch.Tensor,
    weights: torch.Tensor,
    batch_info: LoRABatchInfo,
    stack_num: int = 1,
    out_alloc_stream=None,
) -> torch.Tensor:


    assert x.is_contiguous()
    assert weights.is_contiguous()
    assert len(x.shape) == 2
    assert len(weights.shape) == 3

    S = x.shape[0]
    R = weights.shape[-2]
    K = weights.shape[-1]
    assert x.shape[-1] == K

    if (
        lora_envs.SGLANG_OPT_LORA_CUBLAS.get()
        or lora_envs.SGLANG_OPT_LORA_CUBLAS_A.get()
    ) and weights.shape[
        0
    ] == 1:


        if out_alloc_stream is not None:
            with torch.cuda.stream(out_alloc_stream):
                output = torch.empty((S, R), device=x.device, dtype=x.dtype)
        else:
            output = torch.empty((S, R), device=x.device, dtype=x.dtype)
        return torch.matmul(x, weights[0].transpose(-2, -1), out=output)


    BLOCK_S = 16
    BLOCK_K = 256
    BLOCK_R = triton.next_power_of_2(R)

    sorted_by_adapter = batch_info.permutation is not None

    num_s_tiles = triton.cdiv(batch_info.max_len, BLOCK_S)
    split_k = 1
    if lora_envs.SGLANG_ENABLE_LORA_SHRINK_SPLIT_K.get() and x.is_cuda:
        num_k_tiles = triton.cdiv(K, BLOCK_K)
        base_grid = batch_info.bs * num_s_tiles
        num_sms = _num_sms(x.device.index)
        if base_grid < num_sms and num_k_tiles >= 16:
            split_k = max(1, min(2 * num_sms // base_grid, num_k_tiles, 16))

    launch_kwargs = {}
    if split_k > 1:
        if out_alloc_stream is not None:
            with torch.cuda.stream(out_alloc_stream):
                output = torch.empty((S, R), device=x.device, dtype=torch.float32)
            output.zero_()
        else:
            output = torch.zeros((S, R), device=x.device, dtype=torch.float32)
        launch_kwargs = {
            "num_warps": 2 if split_k <= 4 else 4,
            "num_stages": 3,
        }
    elif out_alloc_stream is not None:


        with torch.cuda.stream(out_alloc_stream):
            output = torch.empty((S, R), device=x.device, dtype=x.dtype)
    else:
        output = torch.empty((S, R), device=x.device, dtype=x.dtype)

    grid = (
        num_s_tiles * split_k,
        batch_info.bs,
    )

    enable_pdl, pdl_kwargs = get_pdl_launch_metadata()
    _sgemm_lora_a_kernel[grid](
        x,
        weights,
        output,
        R,
        K,
        stack_num,
        x.stride(0),
        x.stride(1),
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        output.stride(0),
        output.stride(1),
        batch_info.seg_lens,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        batch_info.permutation,
        sorted_by_adapter,
        BLOCK_S,
        BLOCK_R,
        BLOCK_K,
        split_k,
        ENABLE_PDL=enable_pdl,
        **launch_kwargs,
        **pdl_kwargs,
    )


    return output


def shared_sink_sgemm_lora_a_fwd(
    x: torch.Tensor,
    weights: torch.Tensor,
    batch_info: LoRABatchInfo,
    *,
    stack_num: int,
    padded_rank: bool,
    out_alloc_stream=None,
) -> torch.Tensor:

    assert x.is_contiguous()
    assert weights.is_contiguous()
    assert x.ndim == 2
    assert weights.ndim == 3

    num_tokens = x.shape[0]
    rank_width = weights.shape[-2]
    input_width = weights.shape[-1]
    assert x.shape[-1] == input_width

    block_s = 16
    block_k = 256
    block_rank = 16
    grid = (
        triton.cdiv(batch_info.max_len, block_s) * triton.cdiv(rank_width, block_rank),
        batch_info.bs,
    )

    if out_alloc_stream is None:
        output = torch.empty((num_tokens, rank_width), device=x.device, dtype=x.dtype)
    else:
        with torch.cuda.stream(out_alloc_stream):
            output = torch.empty(
                (num_tokens, rank_width), device=x.device, dtype=x.dtype
            )

    _sgemm_lora_a_kernel[grid](
        x,
        weights,
        output,
        rank_width,
        input_width,
        stack_num,
        x.stride(0),
        x.stride(1),
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        output.stride(0),
        output.stride(1),
        batch_info.seg_lens,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        batch_info.permutation,
        batch_info.permutation is not None,
        block_s,
        block_rank,
        block_k,
        PADDED_RANK=padded_rank,
    )
    return output
