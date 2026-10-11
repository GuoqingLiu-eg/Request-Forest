from __future__ import annotations

from typing import Tuple

import torch

MXFP8_BLOCK_SIZE = 32


def _ue8m0_to_fp32(scale_u8: torch.Tensor) -> torch.Tensor:

    return (scale_u8.to(torch.int32) << 23).view(torch.float32)


def dequant_mxfp8_2d_to_bf16(
    weight: torch.Tensor, scale_u8: torch.Tensor
) -> torch.Tensor:

    n, k = weight.shape
    descale = _ue8m0_to_fp32(scale_u8).unsqueeze(-1)
    deq = weight.to(torch.float32).view(n, k // MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE)
    return (deq * descale).view(n, k).to(torch.bfloat16)


def bf16_to_block_fp8_128(
    weight: torch.Tensor, block: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:

    n, k = weight.shape
    pn = ((n + block - 1) // block) * block
    pk = ((k + block - 1) // block) * block
    xp = torch.zeros((pn, pk), dtype=torch.float32, device=weight.device)
    xp[:n, :k] = weight.to(torch.float32)
    xv = xp.view(pn // block, block, pk // block, block)
    amax = xv.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-4)
    sf = amax / 448.0
    xq = (xv / sf).to(torch.float8_e4m3fn)
    qweight = xq.view(pn, pk)[:n, :k].contiguous()
    scale = sf.view(pn // block, pk // block).contiguous()
    return qweight, scale


def convert_mxfp8_weight_to_block_fp8(
    weight: torch.Tensor, scale_u8: torch.Tensor, block: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:

    bf16 = dequant_mxfp8_2d_to_bf16(weight, scale_u8)
    return bf16_to_block_fp8_128(bf16, block=block)
