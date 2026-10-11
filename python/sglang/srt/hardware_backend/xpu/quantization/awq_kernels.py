# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.hardware_backend.xpu.quantization.int4pack_utils import (
    SUPPORTED_GROUP_SIZES,
    pack_int4_to_uint8,
    unpack_awq_to_codes,
    xpu_int4pack_mm,
)
from sglang.srt.layers.quantization.utils import replace_parameter

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig


class AWQXPULinearKernel:
    def __init__(self, quant_config: Optional[QuantizationConfig] = None):
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        group_size = self.quant_config.group_size
        if group_size not in SUPPORTED_GROUP_SIZES:
            raise ValueError(
                f"AWQ on XPU requires group_size in {SUPPORTED_GROUP_SIZES}, "
                f"got {group_size}. The native XPU INT4 operator does not "
                "support this group size (per-channel/-1 is out of scope)."
            )

        qweight = layer.qweight.data
        qzeros = layer.qzeros.data
        scales = layer.scales.data

        k = qweight.shape[0]
        n = scales.shape[1]


        codes = unpack_awq_to_codes(qweight, k)
        codes = codes.t().contiguous()
        qweight_uint8 = pack_int4_to_uint8(codes)
        qweight_packed = torch.ops.aten._convert_weight_to_int4pack(
            qweight_uint8, 8
        )


        zero_points = unpack_awq_to_codes(qzeros, scales.shape[0])

        replace_parameter(layer, "qweight", qweight_packed)
        layer.register_parameter(
            "xpu_scales",
            torch.nn.Parameter(scales.contiguous(), requires_grad=False),
        )
        layer.register_parameter(
            "xpu_zero_points",
            torch.nn.Parameter(zero_points.to(torch.int8), requires_grad=False),
        )
        del layer.qzeros
        del layer.scales

        layer.xpu_out_features = n
        layer.xpu_group_size = group_size

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return xpu_int4pack_mm(
            x,
            layer.qweight,
            layer.xpu_group_size,
            layer.xpu_scales,
            layer.xpu_zero_points,
            layer.xpu_out_features,
            bias,
        )
