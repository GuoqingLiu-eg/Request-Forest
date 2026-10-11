from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.gemm.trtllm_lora_temp.kernel_utils import (
    _resolve_token_positions,
    get_pdl_launch_metadata,
)
from sglang.srt.lora.trtllm_lora_temp.environ import lora_envs
from sglang.srt.lora.utils import LoRABatchInfo


_CUBLAS_MIN_MAX_LEN = 8


@triton.jit
def _qkv_lora_b_kernel(

    x,
    weights,
    output,

    K,
    max_qkv_out_dim,

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

    n_offs,
    sorted_token_ids,

    SORTED_BY_ADAPTER: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,

    scalings,
    ENABLE_PDL: tl.constexpr = False,
    STORE_WRITEBACK: tl.constexpr = False,
):


    batch_id = tl.program_id(axis=2)
    w_index = tl.load(weight_indices + batch_id)
    rank = tl.load(lora_ranks + w_index)


    if rank == 0:
        return

    qkv_id = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)
    seg_len = tl.load(seg_lens + batch_id)
    if seg_len == 0:
        return
    seg_start = tl.load(seg_indptr + batch_id)
    n_start = tl.load(n_offs + qkv_id)
    n_size = tl.load(n_offs + qkv_id + 1) - n_start
    scaling = tl.load(scalings + w_index)

    K = tl.minimum(K, rank)


    num_pid_n = tl.cdiv(max_qkv_out_dim, BLOCK_N)
    pid_s = pid // num_pid_n
    pid_n = pid % num_pid_n
    if pid_s * BLOCK_S >= seg_len:
        return


    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K)

    s_physical = _resolve_token_positions(
        sorted_token_ids, seg_start, s_offset, seg_len, SORTED_BY_ADAPTER
    )
    x_ptrs = (
        x
        + (qkv_id * K) * x_stride_1
        + (s_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1)
    )
    w_ptrs = (weights + w_index * w_stride_0 + n_start * w_stride_1) + (
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
    )


    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    n_mask = n_offset[None, :] < n_size
    x_tile = tl.load(
        x_ptrs,
        mask=(s_offset[:, None] < seg_len) & (k_offset[None, :] < K),
        other=0.0,
    )
    w_tile = tl.load(
        w_ptrs,
        mask=(k_offset[:, None] < K) & n_mask,
        other=0.0,
    )

    partial_sum = tl.dot(x_tile.to(w_tile.dtype), w_tile)


    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


    partial_sum *= scaling
    partial_sum = partial_sum.to(output.dtype.element_ty)
    output_ptr = (
        output
        + n_start * output_stride_1
        + (s_physical[:, None] * output_stride_0 + n_offset[None, :] * output_stride_1)
    )
    output_mask = (s_offset[:, None] < seg_len) & (n_offset[None, :] < n_size)
    if STORE_WRITEBACK:


        partial_sum += tl.load(output_ptr, mask=output_mask, other=0.0)
        tl.store(output_ptr, partial_sum, mask=output_mask)
    else:
        tl.atomic_add(output_ptr, partial_sum, mask=output_mask, sem="relaxed")


def _qkv_lora_b_cublas(
    x: torch.Tensor,
    qkv_lora_b: torch.Tensor,
    batch_info: LoRABatchInfo,
    output_offset_cpu: torch.Tensor,
    base_output: Optional[torch.Tensor],
    n_slices: int,
) -> torch.Tensor:

    r = qkv_lora_b.shape[-1]
    if base_output is None:
        base_output = torch.zeros(
            (x.shape[0], qkv_lora_b.shape[-2]), device=x.device, dtype=x.dtype
        )
    w = qkv_lora_b[0]
    x_scaled = x[:, : n_slices * r] * batch_info.scalings[0]
    offsets = output_offset_cpu.tolist()
    for i in range(n_slices):
        lo, hi = offsets[i], offsets[i + 1]
        base_output[:, lo:hi].addmm_(x_scaled[:, i * r : (i + 1) * r], w[lo:hi, :r].t())
    return base_output


def qkv_lora_b_fwd(
    x: torch.Tensor,
    qkv_lora_b: torch.Tensor,
    batch_info: LoRABatchInfo,
    output_offset: torch.Tensor,
    max_qkv_out_dim: int,
    base_output: torch.Tensor = None,
    n_slices: int = 3,
    output_offset_cpu: Optional[torch.Tensor] = None,
) -> torch.Tensor:


    s = x.shape[0]
    input_dim = x.shape[1]
    r = qkv_lora_b.shape[-1]
    output_dim = qkv_lora_b.shape[-2]
    assert input_dim == n_slices * r
    assert output_offset.shape[0] == n_slices + 1

    if (
        output_offset_cpu is not None
        and (
            lora_envs.SGLANG_OPT_LORA_CUBLAS.get()
            or lora_envs.SGLANG_OPT_LORA_CUBLAS_QKV.get()
        )
        and batch_info.max_len >= _CUBLAS_MIN_MAX_LEN
        and qkv_lora_b.shape[0]
        == 1
        and x.dtype == qkv_lora_b.dtype
    ):
        return _qkv_lora_b_cublas(
            x, qkv_lora_b, batch_info, output_offset_cpu, base_output, n_slices
        )

    BLOCK_S = 16


    BLOCK_R = max(16, triton.next_power_of_2(r))

    BLOCK_OUT = 64

    grid_b = (
        triton.cdiv(batch_info.max_len, BLOCK_S)
        * triton.cdiv(max_qkv_out_dim, BLOCK_OUT),
        n_slices,
        batch_info.bs,
    )

    if base_output is None:
        output = torch.zeros((s, output_dim), device=x.device, dtype=x.dtype)
    else:
        output = base_output

    sorted_by_adapter = batch_info.permutation is not None
    store_writeback = lora_envs.SGLANG_OPT_LORA_QKV_B_STORE.get()
    enable_pdl, pdl_kwargs = get_pdl_launch_metadata()
    _qkv_lora_b_kernel[grid_b](
        x,
        qkv_lora_b,
        output,
        r,
        max_qkv_out_dim,
        x.stride(0),
        x.stride(1),
        qkv_lora_b.stride(0),
        qkv_lora_b.stride(1),
        qkv_lora_b.stride(2),
        output.stride(0),
        output.stride(1),
        batch_info.seg_lens,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        output_offset,
        batch_info.permutation,
        sorted_by_adapter,
        BLOCK_S,
        BLOCK_OUT,
        BLOCK_R,
        batch_info.scalings,
        ENABLE_PDL=enable_pdl,
        STORE_WRITEBACK=store_writeback,
        **pdl_kwargs,
    )

    return output
