from __future__ import annotations

import torch
from torch import nn

from sglang.kernels.ops.diffusion.norm.rmsnorm_scale_shift_bitexact import (
    can_use_fused_rmsnorm_scale_shift,
    fused_rmsnorm_scale_shift_bitexact,
)
from sglang.kernels.ops.diffusion.sites.quality_gate import QualityGatedFusion

_SITE_MARKER_ATTR = "_sgl_ltx2_rms_norm_modulate_site"
_SITE_ENABLED_ATTR = "_sgl_ltx2_rms_norm_modulate_enabled"
_FUSION = QualityGatedFusion(
    name="LTX-2 RMSNorm+modulate",
    marker_attr=_SITE_MARKER_ATTR,
    enabled_attr=_SITE_ENABLED_ATTR,
)


_ONES_WEIGHT_CACHE: dict[tuple[torch.device, int], torch.Tensor] = {}


def mark_ltx2_rms_norm_modulate_site(module: nn.Module) -> None:

    _FUSION.mark(module)


def ltx2_rms_norm_modulate_active(module: nn.Module) -> bool:
    return _FUSION.is_enabled(module)


def mount_ltx2_rms_norm_modulate(root: nn.Module) -> bool:
    return _FUSION.mount(root)


def unmount_ltx2_rms_norm_modulate(root: nn.Module) -> None:
    _FUSION.unmount(root)


def _ones_weight(x: torch.Tensor) -> torch.Tensor:
    key = (x.device, int(x.shape[-1]))
    w = _ONES_WEIGHT_CACHE.get(key)
    if w is None:
        w = torch.ones(x.shape[-1], device=x.device, dtype=torch.bfloat16)
        _ONES_WEIGHT_CACHE[key] = w
    return w


def can_use_ltx2_rms_norm_modulate(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
) -> bool:
    if x.dtype is not torch.bfloat16 or not x.is_cuda:
        return False
    return can_use_fused_rmsnorm_scale_shift(x, _ones_weight(x), scale, shift)


def fused_ltx2_rms_norm_modulate(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float
) -> torch.Tensor:

    return fused_rmsnorm_scale_shift_bitexact(x, _ones_weight(x), scale, shift, eps)
