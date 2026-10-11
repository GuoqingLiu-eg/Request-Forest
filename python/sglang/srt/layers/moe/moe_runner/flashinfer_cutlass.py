from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.quantization.fp8_kernel import scaled_fp8_quant
from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    register_fused_func,
)
from sglang.srt.utils import is_flashinfer_available
from sglang.srt.utils.common import next_power_of_2

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
        FlashinferCombineInput,
        FlashinferDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


@dataclass
class FlashInferCutlassMoeQuantInfo(MoeQuantInfo):


    quant_type: str
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    quant_scales: Optional[list[torch.Tensor]] = None
    output_dtype: Optional[torch.dtype] = None
    moe_tp_size: int = 1
    moe_tp_rank: int = 0
    moe_ep_size: int = 1
    moe_ep_rank: int = 0
    apply_routed_scaling_factor: bool = True


@dataclass
class FlashInferCutlassMxfp4MoeQuantInfo(MoeQuantInfo):


    w13_weight: torch.Tensor
    w2_weight: torch.Tensor


    w13_weight_scale: torch.Tensor
    w2_weight_scale: torch.Tensor


    mxfp4_weight_global_scale: Optional[torch.Tensor] = None


    w13_bias: Optional[torch.Tensor] = None
    w2_bias: Optional[torch.Tensor] = None


    swiglu_alpha: Optional[torch.Tensor] = None
    swiglu_beta: Optional[torch.Tensor] = None
    swiglu_limit: Optional[torch.Tensor] = None


    moe_tp_size: int = 1
    moe_tp_rank: int = 0
    moe_ep_size: int = 1
    moe_ep_rank: int = 0


    padded_hidden: Optional[int] = None


def _flashinfer_cutlass_fused_moe():
    if not is_flashinfer_available():
        raise RuntimeError(
            "flashinfer_cutlass MoE runner backend requires flashinfer to be installed."
        )
    from flashinfer.fused_moe import cutlass_fused_moe
    from flashinfer.fused_moe.core import ActivationType

    return cutlass_fused_moe, ActivationType


def _activation_type(runner_config: MoeRunnerConfig):
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import get_activation_type

    _, ActivationType = _flashinfer_cutlass_fused_moe()
    activation = ActivationType(
        get_activation_type(
            runner_config.activation,
            is_gated=runner_config.is_gated,
        )
    )
    supported = {
        ActivationType.Swiglu,
        ActivationType.Geglu,
        ActivationType.Relu2,
        ActivationType.Identity,
    }
    assert activation in supported, (
        f"Activation {runner_config.activation!r} "
        f"(is_gated={runner_config.is_gated}) maps to {activation.name}, "
        "which is not supported by flashinfer cutlass moe."
    )
    return activation


def _maybe_apply_routed_scaling_factor(
    output: torch.Tensor,
    quant_info: FlashInferCutlassMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> torch.Tensor:
    if (
        quant_info.apply_routed_scaling_factor
        and runner_config.routed_scaling_factor is not None
    ):
        output.mul_(runner_config.routed_scaling_factor)
    return output


def _prepare_input(
    dispatch_output,
    quant_info: FlashInferCutlassMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.dtype, int]:
    x = dispatch_output.hidden_states
    x_sf = dispatch_output.hidden_states_scale

    if quant_info.quant_type == "fp8":
        assert quant_info.quant_scales is not None and len(quant_info.quant_scales) == 4
        x, _ = scaled_fp8_quant(x, quant_info.quant_scales[3])
        x_sf = None
        output_dtype = quant_info.output_dtype or dispatch_output.hidden_states.dtype
        output_col = dispatch_output.hidden_states.shape[1]
    elif quant_info.quant_type == "fp4":
        output_dtype = quant_info.output_dtype or torch.bfloat16
        output_col = x.shape[1]
        if x_sf is not None and runner_config.is_gated:
            output_col *= 2
    else:
        assert quant_info.quant_type == "bf16"
        output_dtype = quant_info.output_dtype or x.dtype
        output_col = x.shape[1]

    return x, x_sf, output_dtype, output_col


def _run_flashinfer_cutlass(
    *,
    dispatch_output,
    quant_info: FlashInferCutlassMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    output: Optional[torch.Tensor] = None,
    enable_alltoall: bool = False,
) -> torch.Tensor:
    flashinfer_cutlass_fused_moe, _ = _flashinfer_cutlass_fused_moe()

    topk_output = dispatch_output.topk_output
    topk_weights = topk_output.topk_weights
    topk_ids = topk_output.topk_ids
    x, x_sf, output_dtype, output_col = _prepare_input(
        dispatch_output, quant_info, runner_config
    )

    if output is None:
        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            output = torch.empty(
                x.shape[0],
                output_col,
                dtype=output_dtype,
                device=x.device,
            )

    w13_weight = quant_info.w13_weight
    w2_weight = quant_info.w2_weight
    quant_scales = quant_info.quant_scales
    if quant_info.quant_type == "fp4":
        w13_weight = w13_weight.view(torch.long)
        w2_weight = w2_weight.view(torch.long)
        assert quant_scales is not None and len(quant_scales) == 6
        quant_scales = [
            quant_scales[0],
            quant_scales[1].view(torch.int32),
            quant_scales[2],
            quant_scales[3],
            quant_scales[4].view(torch.int32),
            quant_scales[5],
        ]

    output = flashinfer_cutlass_fused_moe(
        output=output,
        input=x,
        token_selected_experts=topk_ids.to(torch.int),
        token_final_scales=topk_weights,
        fc1_expert_weights=w13_weight,
        fc2_expert_weights=w2_weight,
        output_dtype=output_dtype,
        input_sf=x_sf,
        quant_scales=quant_scales,
        ep_size=quant_info.moe_ep_size,
        ep_rank=quant_info.moe_ep_rank,
        tp_size=quant_info.moe_tp_size,
        tp_rank=quant_info.moe_tp_rank,
        tune_max_num_tokens=next_power_of_2(x.shape[0]),
        activation_type=_activation_type(runner_config),
        enable_alltoall=enable_alltoall,
        use_fused_finalize=envs.SGLANG_FLASHINFER_MOE_FUSED_FINALIZE.get(),
    )[0]

    if quant_info.quant_type in ("bf16", "fp8"):
        _maybe_apply_routed_scaling_factor(output, quant_info, runner_config)
    return output


@register_fused_func("none", "flashinfer_cutlass")
def fused_experts_none_to_flashinfer_cutlass(
    dispatch_output: StandardDispatchOutput,
    quant_info: MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    assert isinstance(
        quant_info, FlashInferCutlassMoeQuantInfo
    ), f"Unexpected quant_info type for flashinfer_cutlass: {type(quant_info)}"
    assert (
        not runner_config.apply_router_weight_on_input
    ), "apply_router_weight_on_input is not supported for FlashInfer CUTLASS"

    output = _run_flashinfer_cutlass(
        dispatch_output=dispatch_output,
        quant_info=quant_info,
        runner_config=runner_config,
    )
    return StandardCombineInput(hidden_states=output)


@register_fused_func("flashinfer", "flashinfer_cutlass")
def fused_experts_flashinfer_to_flashinfer_cutlass(
    dispatch_output: FlashinferDispatchOutput,
    quant_info: MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> FlashinferCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
        FlashinferCombineInput,
    )

    assert isinstance(
        quant_info, FlashInferCutlassMoeQuantInfo
    ), f"Unexpected quant_info type for flashinfer_cutlass: {type(quant_info)}"
    assert (
        not runner_config.apply_router_weight_on_input
    ), "apply_router_weight_on_input is not supported for FlashInfer CUTLASS"

    output = _run_flashinfer_cutlass(
        dispatch_output=dispatch_output,
        quant_info=quant_info,
        runner_config=runner_config,
        output=dispatch_output.moe_output,
        enable_alltoall=True,
    )
    return FlashinferCombineInput(hidden_states=output)


@register_fused_func("none", "flashinfer_mxfp4")
def fused_experts_none_to_flashinfer_mxfp4(
    dispatch_output: StandardDispatchOutput,
    quant_info: MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:

    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    assert isinstance(
        quant_info, FlashInferCutlassMxfp4MoeQuantInfo
    ), f"Unexpected quant_info type for flashinfer_mxfp4: {type(quant_info)}"

    flashinfer_cutlass_fused_moe, ActivationType = _flashinfer_cutlass_fused_moe()

    x = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output


    if TopKOutputChecker.format_is_bypassed(topk_output):
        topk_output = topk_output.to_standard()
    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights


    origin_hidden = x.shape[-1]
    padded_hidden = quant_info.padded_hidden
    do_pad = padded_hidden is not None and padded_hidden != origin_hidden
    if do_pad:
        x = torch.nn.functional.pad(
            x,
            (0, padded_hidden - origin_hidden),
            mode="constant",
            value=0.0,
        )

    weight_global_scale = quant_info.mxfp4_weight_global_scale
    use_mxfp8_act_scaling = weight_global_scale is not None
    input_sf = None
    fc1_expert_weights = quant_info.w13_weight
    fc2_expert_weights = quant_info.w2_weight
    if weight_global_scale is not None:
        from flashinfer import mxfp8_quantize

        x, input_sf = mxfp8_quantize(
            x,
            is_sf_swizzled_layout=True,
            alignment=32,
        )
        fc1_expert_weights = fc1_expert_weights.view(torch.int64)
        fc2_expert_weights = fc2_expert_weights.view(torch.int64)
        quant_scales = [
            quant_info.w13_weight_scale.view(torch.int32),
            weight_global_scale,
            quant_info.w2_weight_scale.view(torch.int32),
            weight_global_scale,
        ]
    else:
        quant_scales = [
            quant_info.w13_weight_scale.view(torch.int32),
            quant_info.w2_weight_scale.view(torch.int32),
        ]

    out_hidden = padded_hidden if do_pad else origin_hidden
    output_dtype = torch.bfloat16
    with use_symmetric_memory(get_tp_group(), disabled=not is_allocation_symmetric()):
        out = torch.empty(x.shape[0], out_hidden, dtype=output_dtype, device=x.device)

    flashinfer_cutlass_fused_moe(
        input=x,
        token_selected_experts=topk_ids.to(torch.int32),
        token_final_scales=topk_weights,
        fc1_expert_weights=fc1_expert_weights,
        fc2_expert_weights=fc2_expert_weights,
        output_dtype=output_dtype,
        quant_scales=quant_scales,
        input_sf=input_sf,
        fc1_expert_biases=quant_info.w13_bias,
        fc2_expert_biases=quant_info.w2_bias,
        swiglu_alpha=quant_info.swiglu_alpha,
        swiglu_beta=quant_info.swiglu_beta,
        swiglu_limit=quant_info.swiglu_limit,
        tp_size=quant_info.moe_tp_size,
        tp_rank=quant_info.moe_tp_rank,
        ep_size=quant_info.moe_ep_size,
        ep_rank=quant_info.moe_ep_rank,
        use_w4_group_scaling=not use_mxfp8_act_scaling,
        use_mxfp8_act_scaling=use_mxfp8_act_scaling,
        activation_type=ActivationType.Swiglu,
        tune_max_num_tokens=next_power_of_2(x.shape[0]),
        output=out,
        use_fused_finalize=envs.SGLANG_FLASHINFER_MOE_FUSED_FINALIZE.get(),
    )

    if do_pad:
        out = out[:, :origin_hidden].contiguous()

    return StandardCombineInput(hidden_states=out)
