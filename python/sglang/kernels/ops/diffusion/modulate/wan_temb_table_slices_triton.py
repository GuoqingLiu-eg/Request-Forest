# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

from sglang.srt.utils.custom_op import register_custom_op


@triton.jit
def _temb_table_slices_kernel(
    out_ptr,
    temb_ptr,
    table_ptr,
    rows,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    NCHUNK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1)
    for i in tl.static_range(NCHUNK):
        cols = i * BLOCK + tl.arange(0, BLOCK)
        mask = cols < D
        t = tl.load(temb_ptr + (row * 6 + j) * D + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        w = tl.load(table_ptr + j * D + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + (j * rows + row) * D + cols, w + t, mask=mask)


def can_use_fused_temb_table_slices(table: torch.Tensor, temb: torch.Tensor) -> bool:
    return (
        temb.is_cuda
        and temb.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and temb.dim() == 4
        and temb.shape[2] == 6
        and temb.is_contiguous()
        and table.is_cuda
        and table.device == temb.device
        and table.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and table.shape == (1, 6, temb.shape[-1])
        and table.is_contiguous()
        and temb.numel() > 0
    )


def _fake_temb_table_slices(table: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
    batch, seq_len, six, hidden = temb.shape
    return temb.new_empty((six, batch, seq_len, hidden), dtype=torch.float32)


@register_custom_op(
    op_name="triton_wan_temb_table_slices",
    mutates_args=[],
    fake_impl=_fake_temb_table_slices,
)
def fused_temb_table_slices(table: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:

    batch, seq_len, _, hidden = temb.shape
    out = temb.new_empty((6, batch, seq_len, hidden), dtype=torch.float32)
    rows = batch * seq_len
    block = min(1024, triton.next_power_of_2(hidden))
    nchunk = (hidden + block - 1) // block
    with torch.cuda.device(temb.device):
        _temb_table_slices_kernel[(rows, 6)](
            out,
            temb,
            table,
            rows,
            D=hidden,
            BLOCK=block,
            NCHUNK=nchunk,
        )
    return out
