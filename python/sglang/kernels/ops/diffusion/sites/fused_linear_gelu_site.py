from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from sglang.kernels.jit.utils import get_jit_cuda_arch
from sglang.kernels.ops.diffusion.sites.quality_gate import QualityGatedFusion
from sglang.srt.utils.custom_op import register_custom_op

logger = logging.getLogger(__name__)


_HAS_ADDMM_ACTIVATION = hasattr(torch, "_addmm_activation")


_SITE_LINEAR_ATTR = "_sgl_fused_gelu_linear_attr"
_SITE_ENABLED_ATTR = "_sgl_fused_gelu_enabled"
_FUSION = QualityGatedFusion(
    name="fused linear+GELU",
    marker_attr=_SITE_LINEAR_ATTR,
    enabled_attr=_SITE_ENABLED_ATTR,
)


def _fused_linear_gelu_tanh_fake(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


@register_custom_op(
    op_name="diffusion_fused_linear_gelu_tanh",
    mutates_args=[],
    fake_impl=_fused_linear_gelu_tanh_fake,
)
def fused_linear_gelu_tanh(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:

    x2d = x.reshape(-1, x.shape[-1])
    out = torch._addmm_activation(bias, x2d, weight.t(), use_gelu=True)
    return out.view(*x.shape[:-1], weight.shape[0])


def _is_unquantized(linear: Any) -> bool:


    if getattr(linear, "quant_config", None) is not None:
        return False
    quant_method = getattr(linear, "quant_method", None)
    if quant_method is None:

        return True
    from sglang.multimodal_gen.runtime.layers.linear import UnquantizedLinearMethod

    return isinstance(quant_method, UnquantizedLinearMethod)


def _static_reject_reason(linear: Any) -> str | None:

    if not _HAS_ADDMM_ACTIVATION:
        return "torch._addmm_activation unavailable"
    if not _is_unquantized(linear):
        return "quantized linear"
    if getattr(linear, "skip_bias_add", False):
        return "skip_bias_add (bias returned separately)"
    if getattr(linear, "gather_output", False) and getattr(linear, "tp_size", 1) > 1:
        return "multi-rank gather_output"
    weight = getattr(linear, "weight", None)
    bias = getattr(linear, "bias", None)
    if weight is None or bias is None or weight.dim() != 2:
        return "missing bias or non-2D weight"
    if weight.dtype not in (torch.bfloat16, torch.float16):
        return f"non-half weight dtype {weight.dtype}"
    if bias.dtype != weight.dtype:
        return f"bias dtype {bias.dtype} != weight dtype {weight.dtype}"
    return None


def can_use_linear_gelu_static(linear: Any) -> bool:

    return _static_reject_reason(linear) is None


def can_use_linear_gelu(linear: Any, x: torch.Tensor) -> bool:

    if not (x.is_cuda and x.dtype in (torch.bfloat16, torch.float16)):
        return False
    arch = get_jit_cuda_arch()
    if arch.major * 10 + arch.minor >= 120:


        return False
    if getattr(linear, "weight", None) is None or x.dtype != linear.weight.dtype:
        return False
    return can_use_linear_gelu_static(linear)


def mark_fused_gelu_site(module: nn.Module, linear_attr: str) -> None:

    _FUSION.mark(module, linear_attr)


def fused_gelu_active(module: nn.Module) -> bool:

    return _FUSION.is_enabled(module)


def _site_reject_reason(site: nn.Module) -> str | None:
    linear = getattr(site, _FUSION.metadata(site), None)
    return "missing linear" if linear is None else _static_reject_reason(linear)


def mount_fused_linear_gelu(root: nn.Module) -> bool:

    return _FUSION.mount(root, reject_reason=_site_reject_reason, logger=logger)


def unmount_fused_linear_gelu(root: nn.Module) -> None:

    _FUSION.unmount(root)
