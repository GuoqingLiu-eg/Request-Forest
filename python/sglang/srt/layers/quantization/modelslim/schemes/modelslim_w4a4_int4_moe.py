from __future__ import annotations

import logging
from typing import Any, Dict

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
    NPUW4A4Int4MoEMethod,
)
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimMoEScheme
from sglang.srt.utils import set_weight_attrs

logger = logging.getLogger(__name__)

__all__ = [
    "ModelSlimW4A4Int4MoE",
]


class ModelSlimW4A4Int4MoE(ModelSlimMoEScheme):


    def __init__(
        self,
        quant_config: Dict[str, Any],
        weight_prefix: str,
        group_size: int = 0,
    ) -> None:
        self.quant_config = quant_config
        self.kernel = NPUW4A4Int4MoEMethod()
        self.weight_prefix = weight_prefix
        self.group_size = group_size
        self.is_per_channel_weight = group_size == 0
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

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )

        if envs.SGLANG_NPU_W4A4_NEW_PACKING.get():
            if self.weight_prefix == "w13":
                out_features = intermediate_size_per_partition
                in_features = hidden_size
            else:
                out_features = hidden_size // 2
                in_features = intermediate_size_per_partition

            weight_shape = (num_experts, out_features, in_features)
            scale_shape = (num_experts, 2 * out_features, 1)
        else:
            if self.weight_prefix == "w13":
                a_dim = 2 * intermediate_size_per_partition
                b_dim = hidden_size
            else:
                a_dim = hidden_size
                b_dim = intermediate_size_per_partition

            weight_shape = (num_experts, a_dim, b_dim)
            scale_shape = (num_experts, a_dim, 1)

        offset_shape = scale_shape

        self._create_weight_params(
            layer,
            self.weight_prefix,
            weight_shape,
            scale_shape,
            offset_shape,
            extra_weight_attrs,
        )

    @staticmethod
    def _create_weight_params(
        layer: torch.nn.Module,
        prefix: str,
        weight_shape: tuple,
        scale_shape: tuple,
        offset_shape: tuple,
        extra_weight_attrs: dict,
    ) -> None:


        weight = torch.nn.Parameter(
            torch.empty(weight_shape, dtype=torch.int8),
            requires_grad=False,
        )
        layer.register_parameter(f"{prefix}_weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)


        scale = torch.nn.Parameter(
            torch.empty(scale_shape, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter(f"{prefix}_weight_scale", scale)
        set_weight_attrs(scale, extra_weight_attrs)


        offset = torch.nn.Parameter(
            torch.empty(offset_shape, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter(f"{prefix}_weight_offset", offset)
        set_weight_attrs(offset, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:

        self.kernel.process_weights_after_loading(layer, self.weight_prefix)
