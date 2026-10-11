from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _zero_pages_kernel(
    buf_ptr,
    pages_ptr,
    page_words,
    BLOCK: tl.constexpr,
):
    m = tl.program_id(0)
    blk = tl.program_id(1)
    pg = tl.load(pages_ptr + m).to(tl.int64)
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < page_words
    tl.store(buf_ptr + pg * page_words + offs, 0, mask=mask)


_BLOCK = 2048


def zero_pages(
    raw: torch.Tensor,
    pages: torch.Tensor,
    num_pages: int,
    page_bytes: int,
) -> None:

    m = int(pages.numel())
    if m == 0:
        return
    assert raw.dtype == torch.uint8, f"expected uint8 pool, got {raw.dtype}"
    assert page_bytes % 8 == 0, f"page_bytes {page_bytes} not int64-aligned"
    page_words = page_bytes // 8
    words = raw[: num_pages * page_bytes].view(torch.int64)
    grid = (m, triton.cdiv(page_words, _BLOCK))
    _zero_pages_kernel[grid](words, pages.to(torch.int64), page_words, BLOCK=_BLOCK)
