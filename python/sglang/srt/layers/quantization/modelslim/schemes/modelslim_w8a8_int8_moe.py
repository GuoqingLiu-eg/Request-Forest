from __future__ import annotations

import logging
from typing import Any, Dict

import torch

from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
    NPUW8A8Int8MoEMethod,
)
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimMoEScheme
from sglang.srt.utils import set_weight_attrs

logger = logging.getLogger(__name__)

__all__ = [
    "ModelSlimW8A8Int8MoE",
]


class ModelSlimW8A8Int8MoE(ModelSlimMoEScheme):


    def __init__(
        self,
        quant_config: Dict[str, Any],
        weight_prefix: str,
    ) -> None:
        self.quant_config = quant_config
        self.kernel = NPUW8A8Int8MoEMethod()
        self.weight_prefix = weight_prefix
        if weight_prefix not in ("w13", "w2"):
            raise ValueError(
                f"weight_prefix must be 'w13' or 'w2', got '{weight_prefix}'"
            )

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        **extra_weight_attrs,
    ) -> None:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        self.num_experts = num_experts
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )


        if self.weight_prefix == "w13":
            a_dim = 2 * intermediate_size_per_partition
            b_dim = hidden_size
        else:
            a_dim = hidden_size
            b_dim = intermediate_size_per_partition

        prefix = self.weight_prefix


        weight_name = f"{prefix}_weight"
        weight = torch.nn.Parameter(
            torch.empty(num_experts, a_dim, b_dim, dtype=torch.int8),
            requires_grad=False,
        )
        layer.register_parameter(weight_name, weight)
        set_weight_attrs(weight, extra_weight_attrs)


        scale_name = f"{prefix}_weight_scale"
        scale = torch.nn.Parameter(
            torch.empty(num_experts, a_dim, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter(scale_name, scale)
        set_weight_attrs(scale, extra_weight_attrs)


        offset_name = f"{prefix}_weight_offset"
        offset = torch.nn.Parameter(
            torch.empty(num_experts, a_dim, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter(offset_name, offset)
        set_weight_attrs(offset, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:

        self.kernel.process_weights_after_loading(layer, self.weight_prefix)
