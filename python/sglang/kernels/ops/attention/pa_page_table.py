from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _build_pa_page_table_kernel(
    req_to_token_ptr,
    req_pool_indices_ptr,
    seq_lens_ptr,
    prefill_lens_ptr,
    dst_page_table_ptr,
    kv_lens_ptr,
    window_size: tl.constexpr,
    req_to_token_stride,
    dst_stride,
    BLOCK_SIZE: tl.constexpr,
):

    bid = tl.program_id(0)
    req_idx = tl.load(req_pool_indices_ptr + bid)
    sl = tl.load(seq_lens_ptr + bid).to(tl.int32)
    pf = tl.load(prefill_lens_ptr + req_idx).to(tl.int32)

    decode_start = tl.maximum(pf, sl - window_size)
    gap = tl.where(decode_start > pf, decode_start - pf, 0)
    kv_len = sl - gap

    tl.store(kv_lens_ptr + bid, kv_len)

    src_base = req_idx * req_to_token_stride
    dst_base = bid * dst_stride

    for start in tl.range(0, kv_len, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < kv_len
        pos = tl.where(offs < pf, offs, offs + gap)
        kv_loc = tl.load(
            req_to_token_ptr + src_base + pos,
            mask=mask,
            other=0,
        )
        tl.store(dst_page_table_ptr + dst_base + offs, kv_loc.to(tl.int32), mask=mask)


def _build_pa_page_table(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    prefill_lens: torch.Tensor,
    window_size: int,
    bs: int,
    pa_max_len: int,
    device: torch.device,
    dst_page_table: Optional[torch.Tensor] = None,
    dst_kv_lens: Optional[torch.Tensor] = None,
):

    if dst_page_table is None:
        dst_page_table = torch.zeros(bs, pa_max_len, dtype=torch.int32, device=device)
    if dst_kv_lens is None:
        dst_kv_lens = torch.empty(bs, dtype=torch.int32, device=device)
    if bs > 0 and pa_max_len > 0:
        _build_pa_page_table_kernel[(bs,)](
            req_to_token,
            req_pool_indices.contiguous(),
            seq_lens.to(torch.int32),
            prefill_lens,
            dst_page_table,
            dst_kv_lens,
            window_size,
            req_to_token.stride(0),
            dst_page_table.stride(0),
            BLOCK_SIZE=256,
        )
    return dst_page_table, dst_kv_lens
