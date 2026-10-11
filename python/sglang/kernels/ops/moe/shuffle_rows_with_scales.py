from typing import Tuple

import torch
import triton
import triton.language as tl


BLOCK_K = 4096
NUM_WARPS = 4


@triton.jit
def _shuffle_rows_with_scales_kernel(
    q_ptr,
    scale_ptr,
    q_out_ptr,
    scale_out_ptr,
    dst2src_ptr,
    k,
    num_groups,
    BLOCK_K: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    dst_row = tl.program_id(0)
    tile = tl.program_id(1)


    src_row = tl.load(dst2src_ptr + dst_row).to(tl.int64)
    dst_row64 = dst_row.to(tl.int64)

    offs_k = tile * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = offs_k < k
    vals = tl.load(q_ptr + src_row * k + offs_k, mask=mask_k)
    tl.store(q_out_ptr + dst_row64 * k + offs_k, vals, mask=mask_k)


    if tile == 0:
        offs_g = tl.arange(0, BLOCK_G)
        mask_g = offs_g < num_groups
        scales = tl.load(scale_ptr + src_row * num_groups + offs_g, mask=mask_g)
        tl.store(scale_out_ptr + dst_row64 * num_groups + offs_g, scales, mask=mask_g)


def shuffle_rows_with_scales(
    q: torch.Tensor,
    scale: torch.Tensor,
    dst2src_map: torch.Tensor,
    num_dst_rows: int,
) -> Tuple[torch.Tensor, torch.Tensor]:

    assert q.dim() == 2 and scale.dim() == 2, "q and scale must be 2D"
    assert q.is_contiguous() and scale.is_contiguous(), "q and scale must be contiguous"
    assert q.element_size() == 1, f"q must be a 1-byte dtype, got {q.dtype}"
    assert (
        q.shape[0] == scale.shape[0]
    ), f"row count mismatch: q {q.shape[0]} vs scale {scale.shape[0]}"
    assert (
        dst2src_map.numel() >= num_dst_rows
    ), f"map holds {dst2src_map.numel()} rows, need {num_dst_rows}"


    assert dst2src_map.dtype in (
        torch.int32,
        torch.int64,
    ), f"dst2src_map must hold integer row ids, got {dst2src_map.dtype}"
    assert q.device == scale.device == dst2src_map.device, (
        f"inputs must share a device: q {q.device}, scale {scale.device}, "
        f"map {dst2src_map.device}"
    )

    k = q.shape[1]
    num_groups = scale.shape[1]
    q_out = torch.empty((num_dst_rows, k), device=q.device, dtype=q.dtype)
    scale_out = torch.empty(
        (num_dst_rows, num_groups), device=scale.device, dtype=scale.dtype
    )
    if num_dst_rows == 0:
        return q_out, scale_out

    _shuffle_rows_with_scales_kernel[(num_dst_rows, triton.cdiv(k, BLOCK_K))](
        q.view(torch.int8),
        scale,
        q_out.view(torch.int8),
        scale_out,
        dst2src_map,
        k,
        num_groups,
        BLOCK_K=BLOCK_K,
        BLOCK_G=triton.next_power_of_2(max(num_groups, 1)),
        num_warps=NUM_WARPS,
    )
    return q_out, scale_out
