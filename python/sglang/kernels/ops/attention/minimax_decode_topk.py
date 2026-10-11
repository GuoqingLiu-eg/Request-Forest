from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_module(seq_dtype: torch.dtype) -> Module:
    args = make_cpp_args(seq_dtype, is_arch_support_pdl())
    return load_jit(
        "minimax_decode_topk",
        *args,
        cuda_files=["minimax/minimax_decode_topk.cuh"],
        cuda_wrappers=[
            ("minimax_decode_topk", f"minimax_decode_topk<{args}>"),
            (
                "minimax_decode_topk_page_table",
                f"minimax_decode_topk_page_table<{args}>",
            ),
        ],
    )


def minimax_decode_topk(
    score: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    topk: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    assert score.is_cuda and score.dtype == torch.float32 and score.dim() == 3
    assert seq_lens.is_cuda and seq_lens.dim() == 1
    assert seq_lens.dtype in (torch.int32, torch.int64)
    num_heads, batch, max_seqblock = score.shape
    assert seq_lens.shape[0] == batch

    if not score.is_contiguous():
        score = score.contiguous()
    if not seq_lens.is_contiguous():
        seq_lens = seq_lens.contiguous()

    if out is None:
        out = torch.empty(
            (num_heads, batch, topk), dtype=torch.int32, device=score.device
        )
    else:
        assert out.shape == (num_heads, batch, topk)
        assert out.dtype == torch.int32 and out.is_cuda
        assert out.is_contiguous()

    module = _jit_module(seq_lens.dtype)
    module.minimax_decode_topk(score, seq_lens, out, int(block_size), int(topk))
    return out


def minimax_decode_topk_page_table(
    score: torch.Tensor,
    seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    block_size: int,
    topk: int,
    page_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:

    assert score.is_cuda and score.dtype == torch.float32 and score.dim() == 3
    num_heads, batch, max_seqblock = score.shape
    assert block_size % page_size == 0
    assert req_to_token.dtype == torch.int32 and slot_ids.dtype == torch.int64
    if not score.is_contiguous():
        score = score.contiguous()
    if not seq_lens.is_contiguous():
        seq_lens = seq_lens.contiguous()
    if not slot_ids.is_contiguous():
        slot_ids = slot_ids.contiguous()

    max_sparse_pages = topk * (block_size // page_size)
    page_table = torch.empty(
        (batch * num_heads, max_sparse_pages), dtype=torch.int32, device=score.device
    )
    real_seq_lens = torch.empty(
        (batch * num_heads,), dtype=torch.int32, device=score.device
    )

    module = _jit_module(seq_lens.dtype)
    module.minimax_decode_topk_page_table(
        score,
        seq_lens,
        req_to_token,
        slot_ids,
        page_table,
        real_seq_lens,
        int(block_size),
        int(topk),
        int(page_size),
    )
    return page_table, real_seq_lens
