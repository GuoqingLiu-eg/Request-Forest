from typing import Optional

import torch
import triton
import triton.language as tl


_MHA_KV_INDEX_BLOCK_TOKENS = 4096

_MHA_KV_INDEX_BLOCK_TOKENS_TL = tl.constexpr(_MHA_KV_INDEX_BLOCK_TOKENS)


def get_num_mha_kv_index_blocks(num_pages: int, page_size: int) -> int:

    pages_per_block = _MHA_KV_INDEX_BLOCK_TOKENS // page_size
    return (num_pages + pages_per_block - 1) // pages_per_block


@triton.jit
def create_trtllm_mha_kv_indices_triton(
    req_to_token_ptr,
    req_pool_indices_ptr,
    seq_lens_ptr,
    full_to_swa_ptr,
    page_table_ptr,
    swa_page_table_ptr,
    full_to_swa_numel,
    req_to_token_stride: tl.constexpr,
    page_table_stride: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HAS_SWA: tl.constexpr,
):

    PAGES_PER_BLOCK: tl.constexpr = _MHA_KV_INDEX_BLOCK_TOKENS_TL // PAGE_SIZE
    pid_req = tl.program_id(0)
    pid_blk = tl.program_id(1)

    seq_len = tl.load(seq_lens_ptr + pid_req)
    num_pages = tl.cdiv(seq_len, PAGE_SIZE)
    num_page_blocks = tl.cdiv(seq_len, _MHA_KV_INDEX_BLOCK_TOKENS_TL)
    if pid_blk >= num_page_blocks:
        return

    req_pool_index = tl.load(req_pool_indices_ptr + pid_req)
    page_idx = tl.arange(0, PAGES_PER_BLOCK) + pid_blk * PAGES_PER_BLOCK
    token_pos = page_idx.to(tl.int64) * PAGE_SIZE
    mask = page_idx < num_pages

    slot = tl.load(
        req_to_token_ptr
        + req_pool_index.to(tl.int64) * req_to_token_stride
        + token_pos,
        mask=mask,
    )
    out_off = pid_req * page_table_stride + page_idx
    tl.store(page_table_ptr + out_off, (slot // PAGE_SIZE).to(tl.int32), mask=mask)
    if HAS_SWA:
        swa_index = tl.minimum(tl.maximum(slot, 0), full_to_swa_numel - 1)
        swa_slot = tl.load(full_to_swa_ptr + swa_index.to(tl.int64), mask=mask)
        tl.store(
            swa_page_table_ptr + out_off,
            (swa_slot // PAGE_SIZE).to(tl.int32),
            mask=mask,
        )


def build_trtllm_mha_page_table(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    cache_seqlens: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
    swa_page_table: Optional[torch.Tensor] = None,
    full_to_swa: Optional[torch.Tensor] = None,
) -> None:

    has_swa = full_to_swa is not None
    assert has_swa == (
        swa_page_table is not None
    ), "full_to_swa and swa_page_table must be provided together"
    assert (
        _MHA_KV_INDEX_BLOCK_TOKENS % page_size == 0
    ), f"page_size={page_size} must divide _MHA_KV_INDEX_BLOCK_TOKENS={_MHA_KV_INDEX_BLOCK_TOKENS}"
    bs, num_pages = page_table.shape
    full_to_swa_numel = full_to_swa.numel() if has_swa else 0
    create_trtllm_mha_kv_indices_triton[
        (bs, get_num_mha_kv_index_blocks(num_pages, page_size))
    ](
        req_to_token,
        req_pool_indices,
        cache_seqlens,
        full_to_swa,
        page_table,
        swa_page_table,
        full_to_swa_numel,
        req_to_token.stride(0),
        page_table.stride(0),
        PAGE_SIZE=page_size,
        HAS_SWA=has_swa,
    )
