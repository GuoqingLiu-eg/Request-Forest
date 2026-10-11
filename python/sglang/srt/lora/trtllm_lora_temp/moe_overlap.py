import torch

from sglang.srt.lora.trtllm_lora_temp import (
    get_lora_side_stream,
    get_original_bf16_moe_lora_func,
    get_original_fp4_moe_lora_func,
    get_original_moe_lora_func,
    is_two_stream_active,
)


_LORA_OVERLAP_EVENTS: list = []


def fused_experts_none_to_experimental_sgl_trtllm_fp8_lora_two_stream(
    dispatch_output,
    quant_info,
    runner_config,
    lora_info,
):

    hidden_states = dispatch_output.hidden_states

    use_virtual_lora_store = bool(
        lora_info.lora_use_virtual_experts and lora_info.max_lora_rank > 0
    )


    if not (use_virtual_lora_store and is_two_stream_active(hidden_states)):
        return get_original_moe_lora_func()(
            dispatch_output, quant_info, runner_config, lora_info
        )


    from flashinfer.fused_moe import Fp8QuantizationType

    from sglang.kernels.ops.moe.trtllm_lora_temp import (
        trtllm_fp8_block_scale_routed_moe_lora,
    )
    from sglang.kernels.ops.moe.trtllm_lora_temp.topk_pack import fused_pack_topk
    from sglang.kernels.ops.moe.trtllm_lora_temp.virtual_experts import (
        merged_experts_fused_moe_lora_add,
    )
    from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        use_symmetric_memory,
    )
    from sglang.srt.layers.dp_attention import is_allocation_symmetric
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.layers.moe.utils import RoutingMethodType
    from sglang.srt.lora.trtllm_lora_temp.shared_add_overlap import (
        maybe_overlap_staged_shared_add,
    )
    from sglang.srt.utils.common import next_power_of_2

    assert runner_config.activation == "silu" and runner_config.is_gated, (
        "experimental_sgl_trtllm LoRA currently supports the gated SwiGLU FP8 "
        "Qwen path only."
    )
    assert quant_info.block_quant and not quant_info.use_mxfp8, (
        "experimental_sgl_trtllm LoRA currently supports DeepSeekFp8 block-quant "
        "checkpoints only."
    )
    assert quant_info.weight_block_k is not None
    assert quant_info.w13_weight_scale_inv is not None
    assert quant_info.w2_weight_scale_inv is not None

    topk_output = dispatch_output.topk_output
    assert TopKOutputChecker.format_is_standard(topk_output)
    assert runner_config.top_k is not None

    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights
    token_lora_mapping = lora_info.token_lora_mapping
    fused_lora_routing_cache: dict = {}

    side_stream = get_lora_side_stream()


    gate_up_delta_shape = (
        hidden_states.shape[0],
        runner_config.top_k,
        quant_info.w13_weight.shape[1],
    )
    gate_up_delta = hidden_states.new_empty(gate_up_delta_shape)

    def _run_gate_up_lora():
        merged_experts_fused_moe_lora_add(
            output=gate_up_delta,
            hidden_states=hidden_states,
            lora_a=lora_info.gate_up_lora_a_weights,
            lora_b=lora_info.gate_up_lora_b_weights,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            token_lora_mapping=token_lora_mapping,
            mul_routed_weight=False,
            experts_shared_outer_loras_a=lora_info.experts_shared_outer_loras,
            experts_shared_outer_loras_b=False,
            routing_cache=fused_lora_routing_cache,
            fuse_add_to_output=False,
            use_direct_expand_add=lora_info.max_lora_rank <= 64,
            local_expert_offset=quant_info.local_expert_offset,
            local_num_experts=quant_info.local_num_experts,
            intermediate_buffer=gate_up_lora_intermediate,
        )


    lora_event = torch.cuda.Event()


    merged_experts_fused_moe_lora_add(
        output=gate_up_delta,
        hidden_states=hidden_states,
        lora_a=lora_info.gate_up_lora_a_weights,
        lora_b=lora_info.gate_up_lora_b_weights,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        token_lora_mapping=token_lora_mapping,
        mul_routed_weight=False,
        experts_shared_outer_loras_a=lora_info.experts_shared_outer_loras,
        experts_shared_outer_loras_b=False,
        routing_cache=fused_lora_routing_cache,
        stage="routing",
        local_expert_offset=quant_info.local_expert_offset,
        local_num_experts=quant_info.local_num_experts,
    )
    gate_up_lora_intermediate = hidden_states.new_empty(
        (
            hidden_states.shape[0],
            topk_ids.shape[1],
            lora_info.gate_up_lora_a_weights.shape[2],
        )
    )


    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        _run_gate_up_lora()
        lora_event.record()


    a_q, a_sf = per_token_group_quant_fp8(
        hidden_states, quant_info.weight_block_k, column_major_scales=True
    )
    a_sf_t = a_sf.t()

    activation_lora_input = torch.empty(
        (hidden_states.shape[0], runner_config.top_k, quant_info.intermediate_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )


    packed_topk_ids = getattr(topk_output, "packed_topk_ids", None)
    if packed_topk_ids is None:
        packed_topk_ids = fused_pack_topk(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

    with use_symmetric_memory(get_tp_group(), disabled=not is_allocation_symmetric()):
        direct_down_output = torch.empty(
            hidden_states.shape[0],
            hidden_states.shape[1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )


    if torch.cuda.is_current_stream_capturing():
        _LORA_OVERLAP_EVENTS.append(lora_event)
    lora_ready_handle = lora_event.cuda_event


    down_overlap = False
    gemm2_done_handle = 0
    if down_overlap:
        gemm2_done_event = torch.cuda.Event()


        gemm2_done_event.record()
        gemm2_done_handle = gemm2_done_event.cuda_event

    moe_result = trtllm_fp8_block_scale_routed_moe_lora(
        topk_ids=packed_topk_ids,
        routing_bias=None,
        hidden_states=a_q,
        hidden_states_scale=a_sf_t,
        gemm1_weights=quant_info.w13_weight,
        gemm1_weights_scale=quant_info.w13_weight_scale_inv,
        gemm2_weights=quant_info.w2_weight,
        gemm2_weights_scale=quant_info.w2_weight_scale_inv,
        gate_up_lora_delta=gate_up_delta,
        activation_lora_input=activation_lora_input,
        lora_ready_event=lora_ready_handle,
        num_experts=quant_info.global_num_experts,
        top_k=runner_config.top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=quant_info.intermediate_size,
        local_expert_offset=quant_info.local_expert_offset,
        local_num_experts=quant_info.local_num_experts,
        routed_scaling_factor=(
            runner_config.routed_scaling_factor
            if runner_config.routed_scaling_factor is not None
            else 1.0
        ),
        routing_method_type=(
            RoutingMethodType.TopK
            if quant_info.routing_method_type == RoutingMethodType.DeepSeekV3
            else quant_info.routing_method_type
        ),
        use_shuffled_weight=False,
        do_finalize=True,
        output=direct_down_output,
        tune_max_num_tokens=next_power_of_2(a_q.shape[0]),
        fp8_quantization_type=Fp8QuantizationType.DeepSeekFp8,
        activation_type=quant_info.activation_type,
        gemm2_done_event=gemm2_done_handle,
    )

    output = moe_result

    def _run_down_lora(
        out, stage="all", intermediate_buffer=None, expand_wait_event=None
    ):
        return merged_experts_fused_moe_lora_add(
            output=out,
            hidden_states=activation_lora_input.view(-1, quant_info.intermediate_size),
            lora_a=lora_info.down_lora_a_weights,
            lora_b=lora_info.down_lora_b_weights,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            token_lora_mapping=token_lora_mapping,
            mul_routed_weight=True,
            experts_shared_outer_loras_a=False,
            experts_shared_outer_loras_b=lora_info.experts_shared_outer_loras,
            routing_cache=fused_lora_routing_cache,
            fuse_add_to_output=False,
            fuse_sum_all_reduce=True,
            use_direct_expand_add=lora_info.max_lora_rank <= 64,
            local_expert_offset=quant_info.local_expert_offset,
            local_num_experts=quant_info.local_num_experts,
            stage=stage,
            intermediate_buffer=intermediate_buffer,
            expand_wait_event=expand_wait_event,
        )


    shared_add_done = maybe_overlap_staged_shared_add(output)

    if down_overlap:


        down_intermediate = hidden_states.new_empty(
            (
                hidden_states.shape[0],
                topk_ids.shape[1],
                lora_info.down_lora_a_weights.shape[2],
            )
        )
        side_stream.wait_event(gemm2_done_event)
        shrink_done_event = torch.cuda.Event()
        with torch.cuda.stream(side_stream):
            _run_down_lora(
                output, stage="shrink", intermediate_buffer=down_intermediate
            )
            shrink_done_event.record()
        if torch.cuda.is_current_stream_capturing():
            _LORA_OVERLAP_EVENTS.append(gemm2_done_event)
            _LORA_OVERLAP_EVENTS.append(shrink_done_event)
        torch.cuda.current_stream().wait_event(shrink_done_event)
        _run_down_lora(
            output,
            stage="expand",
            intermediate_buffer=down_intermediate,
            expand_wait_event=shared_add_done,
        )
    else:
        _run_down_lora(output, expand_wait_event=shared_add_done)
    return StandardCombineInput(hidden_states=output)


def fused_experts_none_to_experimental_sgl_trtllm_fp4_lora_two_stream(
    dispatch_output,
    quant_info,
    runner_config,
    lora_info,
):

    hidden_states = dispatch_output.hidden_states

    use_virtual_lora_store = bool(
        lora_info.lora_use_virtual_experts and lora_info.max_lora_rank > 0
    )
    if not (use_virtual_lora_store and is_two_stream_active(hidden_states)):
        return get_original_fp4_moe_lora_func()(
            dispatch_output, quant_info, runner_config, lora_info
        )


    from sglang.kernels.ops.moe.trtllm_lora_temp import (
        trtllm_fp4_block_scale_routed_moe_lora,
    )
    from sglang.kernels.ops.moe.trtllm_lora_temp.topk_pack import fused_pack_topk
    from sglang.kernels.ops.moe.trtllm_lora_temp.virtual_experts import (
        merged_experts_fused_moe_lora_add,
    )
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        use_symmetric_memory,
    )
    from sglang.srt.layers.dp_attention import is_allocation_symmetric
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    assert (
        runner_config.activation == "silu" and runner_config.is_gated
    ), "experimental_sgl_trtllm NVFP4 LoRA currently supports the gated SwiGLU path only."
    topk_output = dispatch_output.topk_output
    assert TopKOutputChecker.format_is_standard(topk_output)
    assert runner_config.top_k is not None

    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights
    token_lora_mapping = lora_info.token_lora_mapping
    fused_lora_routing_cache: dict = {}


    inter = quant_info.intermediate_size_per_partition
    side_stream = get_lora_side_stream()

    gate_up_delta = hidden_states.new_empty(
        (hidden_states.shape[0], runner_config.top_k, quant_info.w13_weight.shape[1])
    )

    def _run_gate_up_lora():
        merged_experts_fused_moe_lora_add(
            output=gate_up_delta,
            hidden_states=hidden_states,
            lora_a=lora_info.gate_up_lora_a_weights,
            lora_b=lora_info.gate_up_lora_b_weights,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            token_lora_mapping=token_lora_mapping,
            mul_routed_weight=False,
            experts_shared_outer_loras_a=lora_info.experts_shared_outer_loras,
            experts_shared_outer_loras_b=False,
            routing_cache=fused_lora_routing_cache,
            fuse_add_to_output=False,
            use_direct_expand_add=lora_info.max_lora_rank <= 64,
            local_expert_offset=quant_info.local_expert_offset,
            local_num_experts=quant_info.local_num_experts,
            intermediate_buffer=gate_up_lora_intermediate,
        )


    merged_experts_fused_moe_lora_add(
        output=gate_up_delta,
        hidden_states=hidden_states,
        lora_a=lora_info.gate_up_lora_a_weights,
        lora_b=lora_info.gate_up_lora_b_weights,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        token_lora_mapping=token_lora_mapping,
        mul_routed_weight=False,
        experts_shared_outer_loras_a=lora_info.experts_shared_outer_loras,
        experts_shared_outer_loras_b=False,
        routing_cache=fused_lora_routing_cache,
        stage="routing",
        local_expert_offset=quant_info.local_expert_offset,
        local_num_experts=quant_info.local_num_experts,
    )
    gate_up_lora_intermediate = hidden_states.new_empty(
        (
            hidden_states.shape[0],
            topk_ids.shape[1],
            lora_info.gate_up_lora_a_weights.shape[2],
        )
    )


    lora_event = torch.cuda.Event()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        _run_gate_up_lora()
        lora_event.record()

    activation_lora_input = torch.empty(
        (hidden_states.shape[0], runner_config.top_k, inter),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    packed_topk_ids = fused_pack_topk(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )
    with use_symmetric_memory(get_tp_group(), disabled=not is_allocation_symmetric()):
        direct_down_output = torch.empty(
            hidden_states.shape[0],
            hidden_states.shape[1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )


    if torch.cuda.is_current_stream_capturing():
        _LORA_OVERLAP_EVENTS.append(lora_event)
    lora_ready_handle = lora_event.cuda_event


    down_overlap = False
    gemm2_done_handle = 0
    if down_overlap:
        gemm2_done_event = torch.cuda.Event()


        gemm2_done_event.record()
        gemm2_done_handle = gemm2_done_event.cuda_event

    output = trtllm_fp4_block_scale_routed_moe_lora(
        topk_ids=packed_topk_ids,
        routing_bias=None,
        hidden_states=hidden_states,
        hidden_states_scale=None,
        gemm1_weights=quant_info.w13_weight,
        gemm1_weights_scale=quant_info.w13_weight_scale.view(torch.float8_e4m3fn),
        gemm2_weights=quant_info.w2_weight,
        gemm2_weights_scale=quant_info.w2_weight_scale.view(torch.float8_e4m3fn),
        output1_scales_scalar=quant_info.g1_scale_c,
        output1_scales_gate_scalar=quant_info.g1_alphas,
        output2_scales_scalar=quant_info.g2_alphas,
        gate_up_lora_delta=gate_up_delta,
        activation_lora_input=activation_lora_input,
        lora_ready_event=lora_ready_handle,
        num_experts=quant_info.global_num_experts,
        top_k=runner_config.top_k,
        intermediate_size=inter,
        local_expert_offset=quant_info.local_expert_offset,
        local_num_experts=quant_info.local_num_experts,
        routed_scaling_factor=(
            runner_config.routed_scaling_factor
            if runner_config.routed_scaling_factor is not None
            else 1.0
        ),
        routing_method_type=quant_info.routing_method_type,
        do_finalize=True,
        output=direct_down_output,
        gemm2_done_event=gemm2_done_handle,
    )

    def _run_down_lora(out, stage="all", intermediate_buffer=None):
        return merged_experts_fused_moe_lora_add(
            output=out,
            hidden_states=activation_lora_input.view(-1, inter),
            lora_a=lora_info.down_lora_a_weights,
            lora_b=lora_info.down_lora_b_weights,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            token_lora_mapping=token_lora_mapping,
            mul_routed_weight=True,
            experts_shared_outer_loras_a=False,
            experts_shared_outer_loras_b=lora_info.experts_shared_outer_loras,
            routing_cache=fused_lora_routing_cache,
            fuse_add_to_output=False,
            fuse_sum_all_reduce=True,
            use_direct_expand_add=lora_info.max_lora_rank <= 64,
            local_expert_offset=quant_info.local_expert_offset,
            local_num_experts=quant_info.local_num_experts,
            stage=stage,
            intermediate_buffer=intermediate_buffer,
        )

    if down_overlap:


        down_intermediate = hidden_states.new_empty(
            (
                hidden_states.shape[0],
                topk_ids.shape[1],
                lora_info.down_lora_a_weights.shape[2],
            )
        )
        side_stream.wait_event(gemm2_done_event)
        shrink_done_event = torch.cuda.Event()
        with torch.cuda.stream(side_stream):
            _run_down_lora(
                output, stage="shrink", intermediate_buffer=down_intermediate
            )
            shrink_done_event.record()
        if torch.cuda.is_current_stream_capturing():
            _LORA_OVERLAP_EVENTS.append(gemm2_done_event)
            _LORA_OVERLAP_EVENTS.append(shrink_done_event)
        torch.cuda.current_stream().wait_event(shrink_done_event)
        _run_down_lora(output, stage="expand", intermediate_buffer=down_intermediate)
    else:
        _run_down_lora(output)
    return StandardCombineInput(hidden_states=output)


def fused_experts_none_to_experimental_sgl_trtllm_bf16_lora_two_stream(
    dispatch_output,
    quant_info,
    runner_config,
    lora_info,
):

    hidden_states = dispatch_output.hidden_states

    use_virtual_lora_store = bool(
        lora_info.lora_use_virtual_experts and lora_info.max_lora_rank > 0
    )
    if not (use_virtual_lora_store and is_two_stream_active(hidden_states)):
        return get_original_bf16_moe_lora_func()(
            dispatch_output, quant_info, runner_config, lora_info
        )


    from sglang.kernels.ops.moe.trtllm_lora_temp import trtllm_bf16_routed_moe_lora
    from sglang.kernels.ops.moe.trtllm_lora_temp.topk_pack import fused_pack_topk
    from sglang.kernels.ops.moe.trtllm_lora_temp.virtual_experts import (
        merged_experts_fused_moe_lora_add,
    )
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        use_symmetric_memory,
    )
    from sglang.srt.layers.dp_attention import is_allocation_symmetric
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        get_activation_type,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.layers.moe.utils import RoutingMethodType

    assert (
        runner_config.activation == "silu" and runner_config.is_gated
    ), "experimental_sgl_trtllm BF16 LoRA currently supports the gated SwiGLU path only."
    topk_output = dispatch_output.topk_output
    assert TopKOutputChecker.format_is_standard(topk_output)
    assert runner_config.top_k is not None

    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights
    token_lora_mapping = lora_info.token_lora_mapping
    fused_lora_routing_cache: dict = {}

    inter = runner_config.intermediate_size_per_partition
    side_stream = get_lora_side_stream()

    gate_up_delta = hidden_states.new_empty(
        (hidden_states.shape[0], runner_config.top_k, 2 * inter)
    )

    def _run_gate_up_lora():
        merged_experts_fused_moe_lora_add(
            output=gate_up_delta,
            hidden_states=hidden_states,
            lora_a=lora_info.gate_up_lora_a_weights,
            lora_b=lora_info.gate_up_lora_b_weights,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            token_lora_mapping=token_lora_mapping,
            mul_routed_weight=False,
            experts_shared_outer_loras_a=lora_info.experts_shared_outer_loras,
            experts_shared_outer_loras_b=False,
            routing_cache=fused_lora_routing_cache,
            fuse_add_to_output=False,
            use_direct_expand_add=lora_info.max_lora_rank <= 64,
            local_expert_offset=quant_info.local_expert_offset,
            local_num_experts=runner_config.num_local_experts,
            intermediate_buffer=gate_up_lora_intermediate,
        )


    lora_event = torch.cuda.Event()


    merged_experts_fused_moe_lora_add(
        output=gate_up_delta,
        hidden_states=hidden_states,
        lora_a=lora_info.gate_up_lora_a_weights,
        lora_b=lora_info.gate_up_lora_b_weights,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        token_lora_mapping=token_lora_mapping,
        mul_routed_weight=False,
        experts_shared_outer_loras_a=lora_info.experts_shared_outer_loras,
        experts_shared_outer_loras_b=False,
        routing_cache=fused_lora_routing_cache,
        stage="routing",
        local_expert_offset=quant_info.local_expert_offset,
        local_num_experts=runner_config.num_local_experts,
    )
    gate_up_lora_intermediate = hidden_states.new_empty(
        (
            hidden_states.shape[0],
            topk_ids.shape[1],
            lora_info.gate_up_lora_a_weights.shape[2],
        )
    )
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        _run_gate_up_lora()
        lora_event.record()

    activation_lora_input = torch.empty(
        (hidden_states.shape[0], runner_config.top_k, inter),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    packed_topk_ids = getattr(topk_output, "packed_topk_ids", None)
    if packed_topk_ids is None:
        packed_topk_ids = fused_pack_topk(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

    routing_method_type = runner_config.routing_method_type
    if routing_method_type is None:
        routing_method_type = RoutingMethodType.Default
    elif routing_method_type == RoutingMethodType.DeepSeekV3:
        routing_method_type = RoutingMethodType.TopK

    with use_symmetric_memory(get_tp_group(), disabled=not is_allocation_symmetric()):
        direct_down_output = torch.empty(
            hidden_states.shape[0],
            hidden_states.shape[1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )


    if torch.cuda.is_current_stream_capturing():
        _LORA_OVERLAP_EVENTS.append(lora_event)
    lora_ready_handle = lora_event.cuda_event

    output = trtllm_bf16_routed_moe_lora(
        topk_ids=packed_topk_ids,
        routing_bias=None,
        hidden_states=hidden_states,
        gemm1_weights=quant_info.gemm1_weights,
        gemm2_weights=quant_info.gemm2_weights,
        gate_up_lora_delta=gate_up_delta,
        activation_lora_input=activation_lora_input,
        num_experts=quant_info.global_num_experts,
        top_k=runner_config.top_k,
        intermediate_size=inter,
        local_expert_offset=quant_info.local_expert_offset,
        local_num_experts=runner_config.num_local_experts,
        routed_scaling_factor=(
            runner_config.routed_scaling_factor
            if runner_config.routed_scaling_factor is not None
            else 1.0
        ),
        routing_method_type=routing_method_type,
        do_finalize=True,
        output=direct_down_output,
        activation_type=get_activation_type(
            runner_config.activation, is_gated=runner_config.is_gated
        ),
        lora_ready_event=lora_ready_handle,


        gemm2_done_event=0,
    )

    merged_experts_fused_moe_lora_add(
        output=output,
        hidden_states=activation_lora_input.view(-1, inter),
        lora_a=lora_info.down_lora_a_weights,
        lora_b=lora_info.down_lora_b_weights,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        token_lora_mapping=token_lora_mapping,
        mul_routed_weight=True,
        experts_shared_outer_loras_a=False,
        experts_shared_outer_loras_b=lora_info.experts_shared_outer_loras,
        routing_cache=fused_lora_routing_cache,
        fuse_add_to_output=False,
        fuse_sum_all_reduce=True,
        use_direct_expand_add=lora_info.max_lora_rank <= 64,
        local_expert_offset=quant_info.local_expert_offset,
        local_num_experts=runner_config.num_local_experts,
    )
    return StandardCombineInput(hidden_states=output)
