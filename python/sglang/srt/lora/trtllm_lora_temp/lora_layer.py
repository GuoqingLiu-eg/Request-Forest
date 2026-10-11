from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput


_SGL_TRTLLM_MODULE_WARMED = False


def _warm_sgl_trtllm_moe_module() -> None:

    global _SGL_TRTLLM_MODULE_WARMED
    if _SGL_TRTLLM_MODULE_WARMED:
        return
    from sglang.kernels.ops.moe.trtllm_lora_temp.core import (
        get_sgl_trtllm_moe_sm100_raw_module,
    )

    get_sgl_trtllm_moe_sm100_raw_module()
    _SGL_TRTLLM_MODULE_WARMED = True


def init_experimental_sgl_trtllm_lora(layer, base_layer) -> None:

    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        FlashInferTrtllmFp8MoeQuantInfo,
        get_activation_type,
    )
    from sglang.srt.layers.moe.utils import RoutingMethodType

    _warm_sgl_trtllm_moe_module()


    if hasattr(base_layer, "g1_scale_c"):
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            FlashInferTrtllmFp4MoeQuantInfo,
        )

        layer._lora_runner = None
        layer._quant_info = FlashInferTrtllmFp4MoeQuantInfo(
            w13_weight=base_layer.w13_weight.data,
            w2_weight=base_layer.w2_weight.data,
            w13_weight_scale=base_layer.w13_weight_scale.data,
            w2_weight_scale=base_layer.w2_weight_scale.data,
            g1_scale_c=base_layer.g1_scale_c.data,
            g1_alphas=base_layer.g1_alphas.data,
            g2_alphas=base_layer.g2_alphas.data,
            w13_input_scale_quant=base_layer.w13_input_scale_quant,
            global_num_experts=int(base_layer.num_experts),
            local_expert_offset=int(base_layer.moe_ep_rank)
            * int(base_layer.num_local_experts),
            local_num_experts=int(base_layer.num_local_experts),
            intermediate_size_per_partition=int(
                base_layer.intermediate_size_per_partition
            ),
            routing_method_type=int(
                getattr(base_layer, "routing_method_type", None)
                or RoutingMethodType.DeepSeekV3
            ),
        )
        return

    quant_method = base_layer.quant_method
    quant_config = getattr(quant_method, "quant_config", None)
    weight_block_size = getattr(quant_config, "weight_block_size", None)
    if weight_block_size is None:
        weight_block_size = getattr(quant_method, "weight_block_size", None)
    use_mxfp8 = bool(getattr(quant_config, "use_mxfp8", False))


    if quant_config is None and not getattr(quant_method, "block_quant", False):
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            FlashInferTrtllmBf16MoeQuantInfo,
        )

        layer._lora_runner = None
        layer._quant_info = FlashInferTrtllmBf16MoeQuantInfo(
            gemm1_weights=base_layer.w13_weight.data,
            gemm2_weights=base_layer.w2_weight.data,
            global_num_experts=int(base_layer.num_experts),
            local_expert_offset=int(base_layer.moe_ep_rank)
            * int(base_layer.num_local_experts),
        )


        _g1 = layer._quant_info.gemm1_weights
        _g2 = layer._quant_info.gemm2_weights
        layer._quant_info.w13_weight = _g1.reshape(_g1.shape[0], _g1.shape[1], -1)
        layer._quant_info.w2_weight = _g2.reshape(_g2.shape[0], _g2.shape[1], -1)
        return

    assert getattr(
        quant_method, "block_quant", False
    ), "experimental_sgl_trtllm LoRA currently requires FP8 block quant."
    assert (
        not use_mxfp8
    ), "experimental_sgl_trtllm LoRA currently targets the non-MX FP8 Qwen path."
    assert (
        weight_block_size is not None
    ), "experimental_sgl_trtllm LoRA needs the FP8 weight block size."
    w13_weight_scale = getattr(base_layer, "w13_weight_scale_inv", None)
    if w13_weight_scale is None:
        w13_weight_scale = getattr(base_layer, "w13_weight_scale", None)
    w2_weight_scale = getattr(base_layer, "w2_weight_scale_inv", None)
    if w2_weight_scale is None:
        w2_weight_scale = getattr(base_layer, "w2_weight_scale", None)
    assert w13_weight_scale is not None and w2_weight_scale is not None

    layer._lora_runner = None
    layer._quant_info = FlashInferTrtllmFp8MoeQuantInfo(
        w13_weight=base_layer.w13_weight,
        w2_weight=base_layer.w2_weight,
        global_num_experts=int(base_layer.num_experts),
        local_expert_offset=int(base_layer.moe_ep_rank)
        * int(base_layer.num_local_experts),
        local_num_experts=int(base_layer.num_local_experts),
        intermediate_size=base_layer.w2_weight.shape[2],
        routing_method_type=int(
            getattr(base_layer, "routing_method_type", None)
            or RoutingMethodType.DeepSeekV3
        ),
        block_quant=True,
        use_mxfp8=False,
        weight_block_k=weight_block_size[1],
        w13_weight_scale_inv=w13_weight_scale,
        w2_weight_scale_inv=w2_weight_scale,
        activation_type=get_activation_type(
            base_layer.moe_runner_config.activation,
            is_gated=base_layer.moe_runner_config.is_gated,
        ),
    )


def dispatch_experimental_sgl_trtllm_lora(
    dispatch_output, quant_info, base_layer, lora_info
) -> StandardCombineInput:

    import sglang.srt.lora.trtllm_lora_temp.lora_dispatch as ft
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        FlashInferTrtllmBf16MoeQuantInfo,
        FlashInferTrtllmFp4MoeQuantInfo,
    )


    if isinstance(quant_info, FlashInferTrtllmFp4MoeQuantInfo):
        fused_fn = ft.fused_experts_none_to_experimental_sgl_trtllm_fp4_lora
    elif isinstance(quant_info, FlashInferTrtllmBf16MoeQuantInfo):
        fused_fn = ft.fused_experts_none_to_experimental_sgl_trtllm_bf16_lora
    else:
        fused_fn = ft.fused_experts_none_to_experimental_sgl_trtllm_fp8_lora

    return fused_fn(
        dispatch_output,
        quant_info,
        base_layer.moe_runner_config,
        lora_info,
    )
