import torch
import triton
import triton.language as tl

from sglang.kernels.ops.gemm.trtllm_lora_temp.gate_up_lora_b import (
    _CUBLAS_MIN_S_RANK,
)
from sglang.kernels.ops.gemm.trtllm_lora_temp.kernel_utils import (
    _resolve_token_positions,
    get_pdl_launch_metadata,
)
from sglang.srt.lora.trtllm_lora_temp.environ import lora_envs
from sglang.srt.lora.utils import LoRABatchInfo


def _sgemm_lora_b_cublas(
    x: torch.Tensor,
    weights: torch.Tensor,
    batch_info: LoRABatchInfo,
    base_output: torch.Tensor,
) -> torch.Tensor:

    if base_output is None:
        base_output = torch.zeros(
            (x.shape[0], weights.shape[-2]), device=x.device, dtype=x.dtype
        )
    w = weights[batch_info.weight_indices[0]]
    x_scaled = x * batch_info.scalings[0]
    base_output.addmm_(x_scaled, w.t())
    return base_output


@triton.jit
def _sgemm_lora_b_kernel(

    x,
    weights,
    output,

    N,
    K,

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

    scalings,
    ENABLE_PDL: tl.constexpr = False,
    APPLY_SCALING: tl.constexpr = True,
    PADDED_RANK: tl.constexpr = True,
    FLAT_GRID: tl.constexpr = False,
    ATOMIC_ADD: tl.constexpr = True,
):


    if FLAT_GRID:
        pid = tl.program_id(axis=0)
        batch_id = tl.program_id(axis=1)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        pid_s = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        pid_s = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)
        batch_id = tl.program_id(axis=2)
    w_index = tl.load(weight_indices + batch_id)
    rank = tl.load(lora_ranks + w_index)


    if rank == 0:
        return

    seg_len = tl.load(seg_lens + batch_id)
    if pid_s * BLOCK_S >= seg_len:
        return
    seg_start = tl.load(seg_indptr + batch_id)
    scaling = tl.load(scalings + w_index) if APPLY_SCALING else 1.0
    if not PADDED_RANK:
        K = tl.minimum(K, rank)

    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K)
    s_physical = _resolve_token_positions(
        sorted_token_ids, seg_start, s_offset, seg_len, SORTED_BY_ADAPTER
    )
    x_ptrs = x + (s_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1)
    w_ptrs = (weights + w_index * w_stride_0) + (
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
    )


    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    n_mask = n_offset[None, :] < N
    output_ptr = output + (
        s_physical[:, None] * output_stride_0 + n_offset[None, :] * output_stride_1
    )
    output_mask = (s_offset[:, None] < seg_len) & n_mask

    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        x_tile = tl.load(
            x_ptrs,
            mask=(s_offset[:, None] < seg_len) & (k_offset[None, :] < k_remaining),
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(k_offset[:, None] < k_remaining) & n_mask,
            other=0.0,
        )

        partial_sum += tl.dot(x_tile.to(w_tile.dtype), w_tile)
        x_ptrs += BLOCK_K * x_stride_1
        w_ptrs += BLOCK_K * w_stride_2
    partial_sum *= scaling


    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


    partial_sum = partial_sum.to(output.dtype.element_ty)
    if ATOMIC_ADD:
        tl.atomic_add(output_ptr, partial_sum, mask=output_mask, sem="relaxed")
    else:
        partial_sum += tl.load(output_ptr, mask=output_mask, other=0.0)
        tl.store(output_ptr, partial_sum, mask=output_mask)


def sgemm_lora_b_fwd(
    x: torch.Tensor,
    weights: torch.Tensor,
    batch_info: LoRABatchInfo,
    base_output: torch.Tensor = None,
) -> torch.Tensor:


    assert x.is_contiguous()
    assert weights.is_contiguous()
    assert len(x.shape) == 2
    assert len(weights.shape) == 3

    S = x.shape[0]
    N = weights.shape[-2]
    R = weights.shape[-1]
    assert x.shape[-1] == R

    if (
        (
            lora_envs.SGLANG_OPT_LORA_CUBLAS.get()
            or lora_envs.SGLANG_OPT_LORA_CUBLAS_B.get()
        )
        and S * R >= _CUBLAS_MIN_S_RANK
        and weights.shape[0] == 1
        and x.dtype == weights.dtype
    ):
        return _sgemm_lora_b_cublas(x, weights, batch_info, base_output)

    BLOCK_S = 16


    BLOCK_R = max(16, triton.next_power_of_2(R))
    BLOCK_N = 256

    grid = (
        triton.cdiv(batch_info.max_len, BLOCK_S),
        triton.cdiv(N, BLOCK_N),
        batch_info.bs,
    )

    if base_output is None:
        output = torch.zeros((S, N), device=x.device, dtype=x.dtype)
    else:
        output = base_output

    sorted_by_adapter = batch_info.permutation is not None
    enable_pdl, pdl_kwargs = get_pdl_launch_metadata()
    _sgemm_lora_b_kernel[grid](
        x,
        weights,
        output,
        N,
        R,
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
        BLOCK_N,
        BLOCK_R,
        batch_info.scalings,
        ENABLE_PDL=enable_pdl,
        **pdl_kwargs,
    )
    return output


def shared_sink_sgemm_lora_b_fwd(
    x: torch.Tensor,
    weights: torch.Tensor,
    batch_info: LoRABatchInfo,
    base_output: torch.Tensor = None,
    *,
    apply_scaling: bool,
    padded_rank: bool,
) -> torch.Tensor:

    assert x.is_contiguous()
    assert weights.is_contiguous()
    assert x.ndim == 2
    assert weights.ndim == 3

    num_tokens = x.shape[0]
    output_width = weights.shape[-2]
    rank_width = weights.shape[-1]
    assert x.shape[-1] == rank_width

    block_s = 16
    block_rank = 16
    block_n = 256
    grid = (
        triton.cdiv(batch_info.max_len, block_s) * triton.cdiv(output_width, block_n),
        batch_info.bs,
    )
    output = (
        torch.zeros((num_tokens, output_width), device=x.device, dtype=x.dtype)
        if base_output is None
        else base_output
    )
    _sgemm_lora_b_kernel[grid](
        x,
        weights,
        output,
        output_width,
        rank_width,
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
        block_n,
        block_rank,
        batch_info.scalings,
        APPLY_SCALING=apply_scaling,
        PADDED_RANK=padded_rank,
        FLAT_GRID=True,
        ATOMIC_ADD=False,
    )
    return output
