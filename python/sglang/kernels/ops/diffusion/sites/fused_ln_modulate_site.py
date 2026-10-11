from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from sglang.kernels.ops.diffusion.sites.quality_gate import QualityGatedFusion

_SITE_MARKER_ATTR = "_sgl_fused_ln_modulate_site"
_SITE_ENABLED_ATTR = "_sgl_fused_ln_modulate_enabled"
_FUSION = QualityGatedFusion(
    name="fused LN+modulate",
    marker_attr=_SITE_MARKER_ATTR,
    enabled_attr=_SITE_ENABLED_ATTR,
)

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


def mark_fused_ln_modulate_site(module: nn.Module) -> None:

    _FUSION.mark(module)


def fused_ln_modulate_active(module: nn.Module) -> bool:
    return _FUSION.is_enabled(module)


def mount_fused_ln_modulate(root: nn.Module) -> bool:
    return _FUSION.mount(root)


def unmount_fused_ln_modulate(root: nn.Module) -> None:
    _FUSION.unmount(root)


def can_use_ln_modulate(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
) -> bool:

    return (
        x.is_cuda
        and x.dtype in _SUPPORTED_DTYPES
        and scale.dtype == x.dtype
        and shift.dtype == x.dtype
        and x.dim() == 3
        and x.shape[0] == 1
        and scale.dim() == 2
        and scale.shape == shift.shape
        and scale.shape == (1, x.shape[-1])
        and x.numel() > 0
    )


def fused_ln_modulate(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float
) -> torch.Tensor:

    return F.layer_norm(
        x,
        (x.shape[-1],),
        weight=(1 + scale).reshape(-1),
        bias=shift.reshape(-1),
        eps=eps,
    )
