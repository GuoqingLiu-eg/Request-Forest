from __future__ import annotations

import logging
from importlib import import_module

import torch
import torch.nn as nn

from sglang.kernels.ops.diffusion.sites.quality_gate import QualityGatedFusion

logger = logging.getLogger(__name__)


_SITE_NORM_ATTRS = "_sgl_fused_gate_rmsnorm_norm_attrs"
_SITE_ENABLED_ATTR = "_sgl_fused_gate_rmsnorm_enabled"
_FUSION = QualityGatedFusion(
    name="fused gate RMSNorm",
    marker_attr=_SITE_NORM_ATTRS,
    enabled_attr=_SITE_ENABLED_ATTR,
)


_MAX_HIDDEN_SIZE = 8192


def fused_rmsnorm_scale(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, eps: float
) -> torch.Tensor | None:

    from sglang.kernels.ops.diffusion.norm.native_bf16_rmsnorm_triton import (
        rmsnorm_scale,
    )

    return rmsnorm_scale(x, weight, scale, eps)


def fused_rmsnorm_tanh_residual(
    x: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor | None:

    from sglang.kernels.ops.diffusion.norm.native_bf16_rmsnorm_triton import (
        rmsnorm_tanh_residual,
    )

    return rmsnorm_tanh_residual(x, gate, residual, weight, eps)


def _static_reject_reason(site: nn.Module) -> str | None:

    try:
        import_module("triton")
    except ImportError:
        return "triton unavailable"
    for attr in _FUSION.metadata(site, ()):
        norm = getattr(site, attr, None)
        weight = getattr(norm, "weight", None)
        if weight is None or weight.dim() != 1:
            return f"{attr}: missing or non-1D norm weight"
        if weight.dtype != torch.bfloat16:
            return f"{attr}: non-bf16 norm weight dtype {weight.dtype}"
        if weight.numel() > _MAX_HIDDEN_SIZE:
            return f"{attr}: hidden size {weight.numel()} above kernel limit"
    return None


def mark_fused_gate_rmsnorm_site(module: nn.Module, norm_attrs: tuple[str, ...]):

    _FUSION.mark(module, tuple(norm_attrs))


def fused_gate_rmsnorm_active(module: nn.Module) -> bool:

    return _FUSION.is_enabled(module)


def mount_fused_gate_rmsnorm(root: nn.Module) -> bool:

    return _FUSION.mount(root, reject_reason=_static_reject_reason, logger=logger)


def unmount_fused_gate_rmsnorm(root: nn.Module) -> None:

    _FUSION.unmount(root)
