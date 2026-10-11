from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from compressed_tensors import CompressionFormat

from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsMoEScheme,
)
from sglang.srt.layers.quantization.w4afp8 import interleave_scales
from sglang.srt.utils import set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

logger = logging.getLogger(__name__)

__all__ = ["CompressedTensorsW4AFP8MoE"]


def _unpack_repack_int32_to_cutlass_int8(
    weight_packed: torch.Tensor, num_bits: int
) -> torch.Tensor:

    pack_factor = 32 // num_bits
    mask = (1 << num_bits) - 1
    offset = 1 << (num_bits - 1)
    pair_factor = pack_factor // 2


    out = torch.empty(
        (*weight_packed.shape[:-1], weight_packed.shape[-1], pair_factor),
        dtype=torch.int8,
        device=weight_packed.device,
    )
    for pair_idx in range(pair_factor):
        low_shift = num_bits * (2 * pair_idx)
        high_shift = low_shift + num_bits

        low_nibbles = ((weight_packed >> low_shift) & mask) - offset
        high_nibbles = ((weight_packed >> high_shift) & mask) - offset
        out[..., pair_idx] = ((high_nibbles << 4) | (low_nibbles & 0x0F)).to(torch.int8)

    return out.flatten(-2).contiguous()


class CompressedTensorsW4AFP8MoE(CompressedTensorsMoEScheme):


    def __init__(
        self,
        quant_config: CompressedTensorsConfig,
        weight_quant,
        input_quant,
    ):
        self.quant_config = quant_config
        config = self.quant_config.target_scheme_map["Linear"].get("weights")
        self.num_bits = config.num_bits
        self.packed_factor = 32 // config.num_bits
        self.group_size = config.group_size
        self.weight_quant = weight_quant
        self.input_quant = input_quant

        assert config.symmetric, "Only symmetric quantization is supported"
        assert (
            self.quant_config.quant_format == CompressionFormat.pack_quantized.value
        ), f"W4AFP8MoE requires pack-quantized format, got {self.quant_config.quant_format}"

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):

        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported


        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.packed_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.packed_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)


        num_groups_w13 = hidden_size // self.group_size
        num_groups_w2 = intermediate_size_per_partition // self.group_size

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )

        w13_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                num_groups_w13,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_scale)
        set_weight_attrs(w13_scale, extra_weight_attrs)

        w2_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                num_groups_w2,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_scale)
        set_weight_attrs(w2_scale, extra_weight_attrs)


        for name, shape in [
            ("w13_weight_shape", (num_experts, 2)),
            ("w2_weight_shape", (num_experts, 2)),
        ]:
            p = torch.nn.Parameter(
                torch.empty(shape, dtype=torch.int32), requires_grad=False
            )
            layer.register_parameter(name, p)
            set_weight_attrs(p, extra_weight_attrs)

        self._init_cutlass_buffers(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            layer.w13_weight_packed.device,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:

        if getattr(layer, "is_w4afp8_converted", False):
            return

        dtype = torch.bfloat16
        device = layer.w2_weight_packed.device


        layer.a13_scale = None
        layer.a2_scale = None

        w13 = _unpack_repack_int32_to_cutlass_int8(
            layer.w13_weight_packed.data, self.num_bits
        )
        layer.w13_weight_packed = torch.nn.Parameter(w13, requires_grad=False)

        w2 = _unpack_repack_int32_to_cutlass_int8(
            layer.w2_weight_packed.data, self.num_bits
        )
        layer.w2_weight_packed = torch.nn.Parameter(w2, requires_grad=False)

        w13_weight_scale = layer.w13_weight_scale.to(dtype)
        w13_weight_scale = interleave_scales(w13_weight_scale)
        layer.w13_weight_scale = torch.nn.Parameter(
            w13_weight_scale, requires_grad=False
        )

        w2_weight_scale = layer.w2_weight_scale.to(dtype)
        w2_weight_scale = interleave_scales(w2_weight_scale)
        layer.w2_weight_scale = torch.nn.Parameter(w2_weight_scale, requires_grad=False)

        layer.is_w4afp8_converted = True

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def _init_cutlass_buffers(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        device: torch.device,
    ):

        self.a_strides1 = torch.full(
            (num_experts, 3), hidden_size, device=device, dtype=torch.int64
        )
        self.c_strides1 = torch.full(
            (num_experts, 3),
            2 * intermediate_size,
            device=device,
            dtype=torch.int64,
        )
        self.a_strides2 = torch.full(
            (num_experts, 3), intermediate_size, device=device, dtype=torch.int64
        )
        self.c_strides2 = torch.full(
            (num_experts, 3), hidden_size, device=device, dtype=torch.int64
        )
        self.b_strides1 = self.a_strides1
        self.s_strides13 = self.c_strides1
        self.b_strides2 = self.a_strides2
        self.s_strides2 = self.c_strides2

        self.expert_offsets = torch.empty(
            num_experts + 1, dtype=torch.int32, device=device
        )
        self.problem_sizes1 = torch.empty(
            (num_experts, 3), dtype=torch.int32, device=device
        )
        self.problem_sizes2 = torch.empty(
            (num_experts, 3), dtype=torch.int32, device=device
        )

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:

        return self.apply_weights(layer, dispatch_output)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.cutlass_w4a8_moe import cutlass_w4a8_moe
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output


        output = cutlass_w4a8_moe(
            x,
            layer.w13_weight_packed,
            layer.w2_weight_packed,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            topk_weights,
            topk_ids,
            self.a_strides1,
            self.b_strides1,
            self.c_strides1,
            self.a_strides2,
            self.b_strides2,
            self.c_strides2,
            self.s_strides13,
            self.s_strides2,
            self.expert_offsets,
            self.problem_sizes1,
            self.problem_sizes2,
            layer.a13_scale,
            layer.a2_scale,
            routed_scaling_factor=self.moe_runner_config.routed_scaling_factor or 1.0,
        )
        return StandardCombineInput(hidden_states=output)
