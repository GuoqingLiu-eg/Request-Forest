from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    register_fused_func,
)
from sglang.srt.model_executor.cuda_graph_config import cuda_graph_fully_disabled
from sglang.srt.runtime_context import (
    cutedsl_moe_max_num_tokens,
    get_parallel,
)
from sglang.srt.utils.common import log_info_on_rank0, print_warning_once

if TYPE_CHECKING:
    from sglang.srt.batch_overlap.single_batch_overlap import DownGemmOverlapArgs
    from sglang.srt.layers.moe.token_dispatcher import (
        DeepEPLLCombineInput,
        DeepEPLLDispatchOutput,
        StandardCombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
        FlashinferCombineInput,
        FlashinferDispatchOutput,
    )

logger = logging.getLogger(__name__)

_FP4_SF_VEC_SIZE = 16
_cutedsl_logged_scalarize: set = set()


def interleave_w13_halves(
    tensor: torch.Tensor, group_size: int = 64, dim: int = 1
) -> torch.Tensor:

    if tensor.shape[dim] % 2 != 0:
        raise ValueError(
            "Expected even size on interleave dimension for W13 half split."
        )
    split = tensor.shape[dim] // 2
    if split % group_size != 0:
        raise ValueError(
            f"Expected split dim divisible by group_size={group_size}, got {split}."
        )
    first_half = tensor.narrow(dim, 0, split)
    second_half = tensor.narrow(dim, split, split)
    first_half_groups = first_half.split(group_size, dim=dim)
    second_half_groups = second_half.split(group_size, dim=dim)
    interleaved = [
        item for pair in zip(first_half_groups, second_half_groups) for item in pair
    ]
    return torch.cat(interleaved, dim=dim)


def cutedsl_quant_scale_to_scalar(
    quant_scale: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:

    quant_scale = quant_scale.to(torch.float32)
    if quant_scale.numel() == 0:
        print_warning_once(
            f"CuteDSL got empty {name}; using 1.0 fallback.",
        )
        return torch.ones(1, device=quant_scale.device, dtype=torch.float32)
    if quant_scale.numel() == 1:
        return quant_scale.reshape(1)
    if name not in _cutedsl_logged_scalarize:
        log_info_on_rank0(
            logger,
            f"CuteDSL: reducing per-expert {name} to scalar via "
            "min(quant_scale) = 1/max(raw_scale), matching TRTLLM convention.",
        )
        _cutedsl_logged_scalarize.add(name)
    return quant_scale.min().reshape(1)


def resolve_cutedsl_standard_scales(
    layer: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:


    def _to_fp32_tensor(x: torch.Tensor | float, ref: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=ref.device)
        return x.to(device=ref.device, dtype=torch.float32)

    def _align_scale_to_alpha(
        scale: torch.Tensor, alpha: torch.Tensor, scale_name: str
    ) -> torch.Tensor:
        scale = scale.to(device=alpha.device, dtype=torch.float32)
        alpha = alpha.to(torch.float32)
        if scale.ndim == 0:
            return scale


        if scale.ndim == 2 and scale.shape[1] <= 2:
            scale = scale[:, 0]
        if scale.numel() == alpha.numel():
            return scale
        if scale.numel() == 1:
            return scale.reshape(())


        num_local_experts = getattr(layer, "num_local_experts", None)
        num_experts = getattr(layer, "num_experts", None)
        moe_ep_rank = getattr(layer, "moe_ep_rank", 0)
        if (
            num_local_experts is not None
            and num_experts is not None
            and scale.numel() == num_experts
            and alpha.numel() == num_local_experts
        ):
            start = moe_ep_rank * num_local_experts
            end = start + num_local_experts
            return scale[start:end]

        raise ValueError(
            f"Unable to align {scale_name} shape={tuple(scale.shape)} "
            f"to alpha shape={tuple(alpha.shape)} for CuteDSL standard scale resolution."
        )

    def _resolve_w1_alpha_from_scalar_input_scale(
        used_input_scale: torch.Tensor,
    ) -> torch.Tensor:

        eps = 1e-12
        scalar = torch.clamp(used_input_scale.to(torch.float32).reshape(()), min=eps)

        if hasattr(layer, "w13_weight_scale_2"):
            w13_weight_scale_2 = _align_scale_to_alpha(
                layer.w13_weight_scale_2, layer.g1_alphas, "w13_weight_scale_2"
            )
            return w13_weight_scale_2.to(torch.float32) / scalar

        w13_isq = _align_scale_to_alpha(
            layer.w13_input_scale_quant, layer.g1_alphas, "w13_input_scale_quant"
        )
        w13_isq = torch.clamp(_to_fp32_tensor(w13_isq, layer.g1_alphas), min=eps)
        return (layer.g1_alphas.to(torch.float32) * w13_isq / scalar).to(torch.float32)

    def _resolve_w2_alpha_from_scalar_fc2_input_scale(
        fc2_input_scale: torch.Tensor,
    ) -> torch.Tensor:

        eps = 1e-12
        fc2_input_scale = fc2_input_scale.to(torch.float32)
        fc2_scalar = torch.clamp(fc2_input_scale.reshape(-1)[:1], min=eps).reshape(())

        if hasattr(layer, "w2_weight_scale_2"):
            w2_weight_scale_2 = _align_scale_to_alpha(
                layer.w2_weight_scale_2, layer.g2_alphas, "w2_weight_scale_2"
            )
            w2_weight_scale_2 = w2_weight_scale_2.to(torch.float32)
            return w2_weight_scale_2 / fc2_scalar

        w2_q_for_w2 = _align_scale_to_alpha(
            layer.w2_input_scale_quant, layer.g2_alphas, "w2_input_scale_quant"
        )
        w2_q_for_w2 = torch.clamp(
            _to_fp32_tensor(w2_q_for_w2, layer.g2_alphas), min=eps
        )
        w2_weight_scale_2 = layer.g2_alphas.to(torch.float32) * w2_q_for_w2
        return w2_weight_scale_2 / fc2_scalar

    fc2_input_scale = cutedsl_quant_scale_to_scalar(
        layer.w2_input_scale_quant,
        name="w2_input_scale_quant",
    )
    w2_alpha = _resolve_w2_alpha_from_scalar_fc2_input_scale(fc2_input_scale)
    used_input_scale = cutedsl_quant_scale_to_scalar(
        layer.w13_input_scale_quant,
        name="w13_input_scale_quant",
    )
    w1_alpha = _resolve_w1_alpha_from_scalar_input_scale(used_input_scale)
    return w1_alpha, fc2_input_scale, w2_alpha, used_input_scale


def _cutedsl_wrapper_activation_type(activation: str, activation_type_cls: Any) -> Any:
    if activation == "silu":
        return activation_type_cls.Swiglu
    if activation == "relu2":
        return activation_type_cls.Relu2
    raise ValueError(
        f"CuteDSL MoE wrapper supports 'silu' (gated) or 'relu2' (non-gated) "
        f"activation, got {activation!r}."
    )


def _make_per_token_global_scale(input_tensor: torch.Tensor) -> torch.Tensor:
    from flashinfer.quantization.nvfp4_quantization_utils import (
        current_nvfp4_4over6_config,
        make_nvfp4_global_scale,
    )

    return make_nvfp4_global_scale(
        input_tensor,
        per_token_activation=True,
        nvfp4_4over6_config=current_nvfp4_4over6_config(),
    )


def refresh_cutedsl_standard_scales_for_weight_update(
    layer: torch.nn.Module,
) -> None:
    w1_alpha, fc2_input_scale, w2_alpha, used_input_scale = (
        resolve_cutedsl_standard_scales(layer)
    )
    if layer.quant_config.use_per_token_activation:
        used_input_scale = _make_per_token_global_scale(used_input_scale)

    new_scales = (w1_alpha, fc2_input_scale, w2_alpha)
    current_scales = layer._cutedsl_scales
    current_input_scale = layer._cutedsl_input_scale


    if (
        not isinstance(current_scales, tuple)
        or len(current_scales) != len(new_scales)
        or not isinstance(current_input_scale, torch.Tensor)
    ):
        raise RuntimeError(
            "CuTe DSL scale metadata changed during weight reload; "
            "CUDA graph recapture is required."
        )
    scale_pairs = (
        *zip(current_scales, new_scales),
        (current_input_scale, used_input_scale),
    )
    for current, new in scale_pairs:
        if (
            not isinstance(current, torch.Tensor)
            or current.shape != new.shape
            or current.dtype != new.dtype
            or current.device != new.device
        ):
            raise RuntimeError(
                "CuTe DSL scale metadata changed during weight reload; "
                "CUDA graph recapture is required."
            )

    with torch.no_grad():
        for current, new in scale_pairs:
            current.copy_(new)


def ensure_cutedsl_wrapper(layer: torch.nn.Module) -> None:

    if layer._cutedsl_wrapper is not None:
        return

    try:
        from flashinfer import ActivationType, CuteDslMoEWrapper
    except ImportError as e:
        raise ImportError(
            "flashinfer_cutedsl backend requires FlashInfer with CuteDSL support. "
            "Install with: pip install flashinfer"
        ) from e

    assert layer.intermediate_size_per_partition > 0, (
        f"CuteDSL MoE: intermediate_size_per_partition must be > 0, "
        f"got {layer.intermediate_size_per_partition}. Check EP/TP configuration."
    )


    use_cuda_graph = not cuda_graph_fully_disabled()


    dispatcher = getattr(layer, "dispatcher", None)
    if hasattr(dispatcher, "max_num_tokens"):

        max_num_tokens = dispatcher.max_num_tokens * getattr(dispatcher, "ep_size", 1)
    else:


        max_num_tokens = get_parallel().config.dp_size * cutedsl_moe_max_num_tokens()
    top_k = layer.top_k if layer.top_k is not None else layer.moe_runner_config.top_k


    with torch.inference_mode(False):
        layer._cutedsl_wrapper = CuteDslMoEWrapper(
            num_experts=layer.num_experts,
            top_k=top_k,
            hidden_size=layer.hidden_size,
            intermediate_size=layer.intermediate_size_per_partition,
            use_cuda_graph=use_cuda_graph,
            max_num_tokens=max_num_tokens,
            num_local_experts=layer.num_local_experts,
            local_expert_offset=layer.moe_ep_rank * layer.num_local_experts,
            output_dtype=layer.moe_runner_config.params_dtype,
            device=str(layer.w13_weight.device),
            use_fused_finalize=envs.SGLANG_FLASHINFER_MOE_FUSED_FINALIZE.get(),
            activation_type=_cutedsl_wrapper_activation_type(
                layer.moe_runner_config.activation, ActivationType
            ),
        )

        w1_alpha, fc2_input_scale, w2_alpha, used_input_scale = (
            resolve_cutedsl_standard_scales(layer)
        )
        if layer.quant_config.use_per_token_activation:
            used_input_scale = _make_per_token_global_scale(used_input_scale)
    layer._cutedsl_scales = (w1_alpha, fc2_input_scale, w2_alpha)
    layer._cutedsl_input_scale = used_input_scale


@dataclass
class CuteDslFp4MoeQuantInfo(MoeQuantInfo):


    w13_weight: torch.Tensor
    w2_weight: torch.Tensor


    w13_weight_sf: torch.Tensor
    w2_weight_sf: torch.Tensor


    w1_alpha: torch.Tensor
    w2_alpha: torch.Tensor


    a1_scale: torch.Tensor
    a2_scale: torch.Tensor


    wrapper: Optional[Any] = None


    use_nvfp4_dispatch: bool = False


    use_per_token_activation: bool = False


    down_gemm_overlap_args: Optional[DownGemmOverlapArgs] = None


@register_fused_func("none", "flashinfer_cutedsl")
def fused_experts_none_to_flashinfer_cutedsl_fp4(
    dispatch_output: StandardDispatchOutput,
    quant_info: CuteDslFp4MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.layers.quantization.fp4_utils import fp4_quantize

    assert runner_config.activation in (
        "silu",
        "relu2",
    ), f"CuteDSL MoE supports 'silu' (gated) or 'relu2' (non-gated), got {runner_config.activation!r}."
    assert quant_info.wrapper is not None, "CuteDSL v2 path requires CuteDslMoEWrapper."

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    assert TopKOutputChecker.format_is_standard(topk_output)

    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights
    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(torch.int32)

    if quant_info.use_per_token_activation:
        from flashinfer import SfLayout, nvfp4_quantize

        x_fp4, x_sf, per_token_scale = nvfp4_quantize(
            hidden_states,
            quant_info.a1_scale,
            sfLayout=SfLayout.layout_linear,
            per_token_activation=True,
            backend="cute-dsl",
        )
    else:
        x_fp4, x_sf = fp4_quantize(
            hidden_states,
            quant_info.a1_scale,
            sf_vec_size=_FP4_SF_VEC_SIZE,
            is_sf_swizzled_layout=False,
        )
        per_token_scale = None

    seq_len, hidden_size = hidden_states.shape
    x_fp4 = x_fp4.reshape(seq_len, hidden_size // 2)
    x_sf = x_sf.view(torch.float8_e4m3fn).reshape(
        seq_len, hidden_size // _FP4_SF_VEC_SIZE
    )

    output = quant_info.wrapper.run(
        x=x_fp4,
        x_sf=x_sf,
        token_selected_experts=topk_ids,
        token_final_scales=topk_weights,
        w1_weight=quant_info.w13_weight,
        w1_weight_sf=quant_info.w13_weight_sf,
        w1_alpha=quant_info.w1_alpha,
        fc2_input_scale=quant_info.a2_scale,
        w2_weight=quant_info.w2_weight,
        w2_weight_sf=quant_info.w2_weight_sf,
        w2_alpha=quant_info.w2_alpha,
        per_token_scale=per_token_scale,
    )

    return StandardCombineInput(hidden_states=output)


@register_fused_func("flashinfer", "flashinfer_cutedsl")
def fused_experts_flashinfer_to_flashinfer_cutedsl_fp4(
    dispatch_output: FlashinferDispatchOutput,
    quant_info: CuteDslFp4MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> FlashinferCombineInput:

    from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
        FlashinferCombineInput,
    )
    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.layers.quantization.fp4_utils import fp4_quantize

    assert runner_config.activation in (
        "silu",
        "relu2",
    ), f"CuteDSL MoE supports 'silu' (gated) or 'relu2' (non-gated), got {runner_config.activation!r}."
    assert quant_info.wrapper is not None, "CuteDSL v2 path requires CuteDslMoEWrapper."

    hidden_states = dispatch_output.hidden_states
    x_sf = dispatch_output.hidden_states_scale
    topk_output = dispatch_output.topk_output
    assert TopKOutputChecker.format_is_standard(topk_output)

    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights
    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(torch.int32)

    if x_sf is not None:
        if quant_info.use_per_token_activation:
            raise ValueError(
                "flashinfer_cutedsl per-token activation requires BF16 dispatch "
                "so the runner can forward per_token_scale to FlashInfer."
            )

        x_fp4 = hidden_states
        per_token_scale = None
    else:
        if quant_info.use_per_token_activation:
            from flashinfer import SfLayout, nvfp4_quantize

            x_fp4, x_sf, per_token_scale = nvfp4_quantize(
                hidden_states,
                quant_info.a1_scale,
                sfLayout=SfLayout.layout_linear,
                per_token_activation=True,
                backend="cute-dsl",
            )
        else:
            x_fp4, x_sf = fp4_quantize(
                hidden_states,
                quant_info.a1_scale,
                sf_vec_size=_FP4_SF_VEC_SIZE,
                is_sf_swizzled_layout=False,
            )
            per_token_scale = None

        seq_len, hidden_size = hidden_states.shape
        x_fp4 = x_fp4.reshape(seq_len, hidden_size // 2)
        x_sf = x_sf.view(torch.float8_e4m3fn).reshape(
            seq_len, hidden_size // _FP4_SF_VEC_SIZE
        )

    output = quant_info.wrapper.run(
        x=x_fp4,
        x_sf=x_sf,
        token_selected_experts=topk_ids,
        token_final_scales=topk_weights,
        w1_weight=quant_info.w13_weight,
        w1_weight_sf=quant_info.w13_weight_sf,
        w1_alpha=quant_info.w1_alpha,
        fc2_input_scale=quant_info.a2_scale,
        w2_weight=quant_info.w2_weight,
        w2_weight_sf=quant_info.w2_weight_sf,
        w2_alpha=quant_info.w2_alpha,
        per_token_scale=per_token_scale,
    )


    if dispatch_output.moe_output is not None:
        dispatch_output.moe_output.copy_(output)
        output = dispatch_output.moe_output

    return FlashinferCombineInput(hidden_states=output)


@register_fused_func("deepep", "flashinfer_cutedsl")
def fused_experts_deepep_to_flashinfer_cutedsl_fp4(
    dispatch_output: DeepEPLLDispatchOutput,
    quant_info: CuteDslFp4MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> DeepEPLLCombineInput:
    from sglang.srt.layers.moe.flashinfer_cutedsl_moe import (
        flashinfer_cutedsl_moe_masked,
    )
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLCombineInput

    assert runner_config.activation in (
        "silu",
        "relu2",
    ), f"CuteDSL masked MoE supports 'silu' or 'relu2', got {runner_config.activation!r}."
    assert (
        not runner_config.apply_router_weight_on_input
    ), "apply_router_weight_on_input is not supported for Flashinfer"

    hidden_states, hidden_states_scale, _, _, masked_m, _ = dispatch_output


    if (
        quant_info.use_nvfp4_dispatch
        and hidden_states_scale is not None
        and hidden_states_scale.element_size() != 1
        and hidden_states_scale.stride(-1) != 1
    ):
        raise AssertionError(
            f"NVFP4 dispatch scale has stride(-1)={hidden_states_scale.stride(-1)}, "
            f"dtype={hidden_states_scale.dtype}; .view(float8_e4m3fn) requires stride(-1)==1. "
            "Try SGLANG_MOE_NVFP4_DISPATCH=0 or check DeepEP version."
        )

    overlap = quant_info.down_gemm_overlap_args
    output = flashinfer_cutedsl_moe_masked(
        hidden_states=(hidden_states, hidden_states_scale),
        input_global_scale=(
            None if quant_info.use_nvfp4_dispatch else quant_info.a1_scale
        ),
        w1=quant_info.w13_weight,
        w1_blockscale=quant_info.w13_weight_sf,
        w1_alpha=quant_info.w1_alpha,
        w2=quant_info.w2_weight,
        a2_global_scale=quant_info.a2_scale,
        w2_blockscale=quant_info.w2_weight_sf,
        w2_alpha=quant_info.w2_alpha,
        masked_m=masked_m,
        activation=runner_config.activation,
        **(
            dict(
                down_sm_count=overlap.num_sms,
                down_signals=overlap.signal,
                down_start_event=overlap.start_event,
            )
            if overlap is not None
            else {}
        ),
    )

    return DeepEPLLCombineInput(
        hidden_states=output,
        topk_ids=dispatch_output.topk_ids,
        topk_weights=dispatch_output.topk_weights,
    )
