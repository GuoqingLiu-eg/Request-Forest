import torch
import triton
import triton.language as tl

from sglang.kernels.ops.gemm.trtllm_lora_temp.kernel_utils import (
    _resolve_token_positions,
    get_pdl_launch_metadata,
)
from sglang.srt.lora.trtllm_lora_temp.environ import lora_envs
from sglang.srt.lora.utils import LoRABatchInfo


_CUBLAS_MIN_S_RANK = 16384


@triton.jit
def _gate_up_lora_b_kernel(

    x,
    weights,
    output,

    K,
    output_dim,

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
):


    batch_id = tl.program_id(axis=2)
    w_index = tl.load(weight_indices + batch_id)
    rank = tl.load(lora_ranks + w_index)


    if rank == 0:
        return

    gate_up_id = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)
    seg_len = tl.load(seg_lens + batch_id)
    if seg_len == 0:
        return
    seg_start = tl.load(seg_indptr + batch_id)
    n_start = gate_up_id * output_dim
    scaling = tl.load(scalings + w_index)


    K = tl.minimum(K, rank)


    num_pid_n = tl.cdiv(output_dim, BLOCK_N)
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
        + (gate_up_id * K) * x_stride_1
        + (s_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1)
    )
    w_ptrs = (weights + w_index * w_stride_0 + n_start * w_stride_1) + (
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
    )


    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()


    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x_tile = tl.load(
            x_ptrs,
            mask=(s_offset[:, None] < seg_len) & (k_offset[None, :] < K - k * BLOCK_K),
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(k_offset[:, None] < K - k * BLOCK_K)
            & (n_offset[None, :] < output_dim),
            other=0.0,
        )
        partial_sum += tl.dot(
            x_tile.to(w_tile.dtype), w_tile
        )

        x_ptrs += BLOCK_K * x_stride_1
        w_ptrs += BLOCK_K * w_stride_2


    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


    partial_sum *= scaling
    partial_sum = partial_sum.to(output.dtype.element_ty)
    output_ptr = (
        output
        + n_start * output_stride_1
        + (s_physical[:, None] * output_stride_0 + n_offset[None, :] * output_stride_1)
    )
    output_mask = (s_offset[:, None] < seg_len) & (n_offset[None, :] < output_dim)
    partial_sum += tl.load(output_ptr, mask=output_mask)
    tl.store(output_ptr, partial_sum, mask=output_mask)


def _gate_up_lora_b_cublas(
    x: torch.Tensor,
    gate_up_lora_b: torch.Tensor,
    batch_info: LoRABatchInfo,
    output_dim: int,
    base_output: torch.Tensor,
) -> torch.Tensor:

    r = gate_up_lora_b.shape[-1]
    if base_output is None:
        base_output = torch.zeros(
            (x.shape[0], 2 * output_dim), device=x.device, dtype=x.dtype
        )
    w = gate_up_lora_b[0]
    x_scaled = x[:, : 2 * r] * batch_info.scalings[0]
    for i in range(2):
        lo, hi = i * output_dim, (i + 1) * output_dim
        base_output[:, lo:hi].addmm_(x_scaled[:, i * r : (i + 1) * r], w[lo:hi, :r].t())
    return base_output


def gate_up_lora_b_fwd(
    x: torch.Tensor,
    gate_up_lora_b: torch.Tensor,
    batch_info: LoRABatchInfo,
    output_dim: int,
    base_output: torch.Tensor = None,
) -> torch.Tensor:


    s = x.shape[0]
    input_dim = x.shape[1]
    r = gate_up_lora_b.shape[-1]
    assert input_dim == 2 * r

    if (
        (
            lora_envs.SGLANG_OPT_LORA_CUBLAS.get()
            or lora_envs.SGLANG_OPT_LORA_CUBLAS_GATE_UP.get()
        )
        and s * r >= _CUBLAS_MIN_S_RANK
        and gate_up_lora_b.shape[0] == 1
        and x.dtype == gate_up_lora_b.dtype
    ):
        return _gate_up_lora_b_cublas(
            x, gate_up_lora_b, batch_info, output_dim, base_output
        )

    BLOCK_S = 16
    BLOCK_R = 16
    BLOCK_OUT = 64

    grid_b = (
        triton.cdiv(batch_info.max_len, BLOCK_S) * triton.cdiv(output_dim, BLOCK_OUT),
        2,
        batch_info.bs,
    )

    if base_output is None:
        output = torch.zeros((s, 2 * output_dim), device=x.device, dtype=x.dtype)
    else:
        output = base_output

    sorted_by_adapter = batch_info.permutation is not None
    enable_pdl, pdl_kwargs = get_pdl_launch_metadata()
    _gate_up_lora_b_kernel[grid_b](
        x,
        gate_up_lora_b,
        output,
        r,
        output_dim,
        x.stride(0),
        x.stride(1),
        gate_up_lora_b.stride(0),
        gate_up_lora_b.stride(1),
        gate_up_lora_b.stride(2),
        output.stride(0),
        output.stride(1),
        batch_info.seg_lens,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        batch_info.permutation,
        sorted_by_adapter,
        BLOCK_S,
        BLOCK_OUT,
        BLOCK_R,
        batch_info.scalings,
        ENABLE_PDL=enable_pdl,
        **pdl_kwargs,
    )

    return output
