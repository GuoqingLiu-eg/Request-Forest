from sglang.srt.layers.moe.moe_runner.base import register_fused_func


@register_fused_func("none", "experimental_sgl_trtllm")
def fused_experts_none_to_experimental_sgl_trtllm(
    dispatch_output, quant_info, runner_config
):

    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        fused_experts_none_to_flashinfer_trtllm,
    )

    return fused_experts_none_to_flashinfer_trtllm(
        dispatch_output, quant_info, runner_config
    )
