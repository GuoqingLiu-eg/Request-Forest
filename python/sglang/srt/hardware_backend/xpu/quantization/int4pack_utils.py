# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import torch


AWQ_REVERSE_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


SUPPORTED_GROUP_SIZES = (32, 64, 128, 256)


def pack_int4_to_uint8(q: torch.Tensor) -> torch.Tensor:

    assert q.shape[-1] % 2 == 0, "K must be even to pack into int4 bytes"
    q = q.to(torch.uint8)
    low = q[..., 0::2]
    high = q[..., 1::2]
    return (low | (high << 4)).contiguous()


def unpack_awq_to_codes(packed: torch.Tensor, rows: int) -> torch.Tensor:

    t = packed.contiguous().view(torch.uint8)
    shifter = torch.tensor([0, 4], dtype=torch.uint8, device=t.device)
    t = (t[:, :, None] >> shifter) & 0xF
    t = t.view(-1, 8)[:, AWQ_REVERSE_PACK_ORDER]
    return t.reshape(rows, -1)


def _nibble_shifts(device: torch.device) -> torch.Tensor:
    return torch.arange(0, 32, 4, device=device, dtype=torch.int32)


def unpack_gptq_qweight(qweight: torch.Tensor) -> torch.Tensor:

    n = qweight.shape[1]
    shifts = _nibble_shifts(qweight.device)

    codes = (qweight.unsqueeze(1) >> shifts.view(1, 8, 1)) & 0xF
    return codes.reshape(-1, n)


def unpack_gptq_qzeros(qzeros: torch.Tensor) -> torch.Tensor:

    rows = qzeros.shape[0]
    shifts = _nibble_shifts(qzeros.device)

    codes = (qzeros.unsqueeze(-1) >> shifts.view(1, 1, 8)) & 0xF
    return codes.reshape(rows, -1)


def xpu_int4pack_mm(
    x: torch.Tensor,
    qweight_packed: torch.Tensor,
    group_size: int,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    out_features: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:

    out_shape = x.shape[:-1] + (out_features,)
    reshaped_x = x.reshape(-1, x.shape[-1]).contiguous()
    out = torch.ops.aten._weight_int4pack_mm_with_scales_and_zeros(
        reshaped_x, qweight_packed, group_size, scales, zero_points
    )
    if bias is not None:
        out = out + bias
    return out.reshape(out_shape)
