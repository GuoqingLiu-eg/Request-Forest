import torch
import triton
import triton.language as tl

from sglang.srt.lora.utils import LoRABatchInfo


@triton.jit(do_not_specialize=["num_segments"])
def _chunked_embedding_lora_a_kernel(

    input_ids,
    weights,
    output,

    vocab_size,
    rank,
    num_loras,

    w_stride_0,
    w_stride_1,
    w_stride_2,
    output_stride_0,
    output_stride_1,

    seg_indptr,
    weight_indices,
    lora_ranks,
    num_segments,
    permutation,

    BLOCK_RANK: tl.constexpr,
):

    chunk_idx = tl.program_id(axis=0)

    if chunk_idx >= num_segments:
        return
    chunk_start = tl.load(seg_indptr + chunk_idx)
    chunk_end = tl.load(seg_indptr + chunk_idx + 1)
    if chunk_start == chunk_end:
        return

    lora_index = tl.load(weight_indices + chunk_idx)
    rank_val = tl.load(lora_ranks + lora_index)

    if rank_val == 0:
        return

    for c in range(chunk_start, chunk_end):
        s_index = tl.load(permutation + c)

        token_id = tl.load(input_ids + s_index)

        num_blocks = tl.cdiv(rank_val, BLOCK_RANK)

        for block_id in range(num_blocks):
            rank_offset = tl.arange(0, BLOCK_RANK) + block_id * BLOCK_RANK
            rank_mask = rank_offset < rank_val


            weight_ptr = (
                weights
                + lora_index * w_stride_0
                + rank_offset * w_stride_1
                + token_id * w_stride_2
            )
            emb_values = tl.load(weight_ptr, mask=rank_mask, other=0.0)


            output_ptr = (
                output + s_index * output_stride_0 + rank_offset * output_stride_1
            )
            tl.store(output_ptr, emb_values, mask=rank_mask)


def chunked_embedding_lora_a_forward(
    input_ids: torch.Tensor,
    weights: torch.Tensor,
    batch_info: LoRABatchInfo,
    vocab_size: int,
) -> torch.Tensor:

    assert input_ids.is_contiguous()
    assert weights.is_contiguous()
    assert len(input_ids.shape) == 1
    assert len(weights.shape) == 3

    S = input_ids.shape[0]
    num_loras = weights.shape[0]
    rank = weights.shape[1]


    BLOCK_RANK = 128
    num_segments = batch_info.num_segments
    segment_grid = (
        batch_info.weight_indices.shape[0]
        if batch_info.use_cuda_graph
        else num_segments
    )

    grid = (segment_grid,)
    output = torch.zeros((S, rank), device=input_ids.device, dtype=weights.dtype)

    _chunked_embedding_lora_a_kernel[grid](
        input_ids,
        weights,
        output,
        vocab_size,
        rank,
        num_loras,
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        output.stride(0),
        output.stride(1),
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        segment_grid,
        batch_info.permutation,
        BLOCK_RANK,
    )

    return output
