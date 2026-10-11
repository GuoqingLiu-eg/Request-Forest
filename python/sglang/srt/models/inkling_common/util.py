from __future__ import annotations

import abc

import torch
from torch import nn

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.runtime_context import get_lora


def lora_compatible_layout_enabled() -> bool:

    return get_lora().enable_lora


def use_inkling_shared_fused_moe(
    *,
    inference_moe_w13_interleaved: bool = True,
    shared_sink_serves_fp4: bool = False,
) -> bool:

    from sglang.srt.environ import envs

    if not inference_moe_w13_interleaved or shared_sink_serves_fp4:
        return True
    if lora_compatible_layout_enabled():
        return False
    if envs.SGLANG_OPT_USE_INKLING_SHARED_FUSED_MOE.is_set():
        return envs.SGLANG_OPT_USE_INKLING_SHARED_FUSED_MOE.get()
    return False


def bf16_routed_uses_stock_fused_moe(
    quant_config: QuantizationConfig | None,
) -> bool:

    if quant_config is not None:
        return False
    from sglang.srt.layers.moe import get_moe_runner_backend

    return get_moe_runner_backend().is_flashinfer_trtllm_routed()


def shared_sink_uses_trtllm_bf16() -> bool:

    from sglang.srt.layers.moe import get_moe_runner_backend

    backend = get_moe_runner_backend()
    if lora_compatible_layout_enabled():
        return False
    return backend.is_flashinfer_trtllm_routed()


def trtllm_bf16_weight_prep_enabled() -> bool:

    from sglang.srt.layers.moe import get_moe_runner_backend

    backend = get_moe_runner_backend()
    return backend.is_flashinfer_trtllm() or backend.is_flashinfer_trtllm_routed()


def deinterleave_gate_up(weight: torch.Tensor, dim: int) -> torch.Tensor:

    dim = dim % weight.dim()
    if weight.shape[dim] % 2 != 0:
        raise ValueError(
            f"Cannot deinterleave odd gate/up dimension {dim}: {tuple(weight.shape)}"
        )
    shape = list(weight.shape)
    half = shape[dim] // 2
    view_shape = shape[:dim] + [half, 2] + shape[dim + 1 :]
    return (
        weight.reshape(view_shape)
        .transpose(dim, dim + 1)
        .reshape_as(weight)
        .contiguous()
    )


class FusedMoELoadingMixin(abc.ABC):
    def __init__(
        self,
        quant_config: QuantizationConfig | None,
        quant_method: UnquantizedFusedMoEMethod,
        moe_runner_config: MoeRunnerConfig,
        moe_tp_rank: int,
        moe_tp_size: int,
    ) -> None:
        super().__init__()
        helper = FusedMoE.__new__(FusedMoE)
        nn.Module.__init__(helper)
        helper.quant_config = quant_config
        helper.quant_method = quant_method
        helper.moe_runner_config = moe_runner_config
        helper.use_triton_kernels = False
        helper.moe_tp_rank = moe_tp_rank
        helper.moe_tp_size = moe_tp_size
        helper.use_presharded_weights = False
        helper.use_flashinfer_trtllm_moe = False


        object.__setattr__(self, "helper", helper)

    def weight_loader_fused(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
    ) -> None:
        return self.helper.weight_loader_fused(
            param, loaded_weight, weight_name, shard_id
        )
