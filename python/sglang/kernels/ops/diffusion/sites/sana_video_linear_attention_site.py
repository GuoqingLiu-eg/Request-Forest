from __future__ import annotations

import logging

import torch
import torch.nn as nn

from sglang.kernels.ops.diffusion.sites.quality_gate import QualityGatedFusion

logger = logging.getLogger(__name__)

_FUSION = QualityGatedFusion(
    name="SANA-Video BF16-input linear attention",
    marker_attr="_sgl_sana_video_linear_attention_site",
    enabled_attr="_sgl_sana_video_linear_attention_enabled",
)


def mark_sana_video_linear_attention_site(module: nn.Module) -> None:

    _FUSION.mark(module)


def sana_video_linear_attention_active(module: nn.Module) -> bool:

    return _FUSION.is_enabled(module)


def _site_reject_reason(_site: nn.Module) -> str | None:
    if torch.version.cuda is None:
        return "CUDA is unavailable"
    return None


def mount_sana_video_linear_attention(root: nn.Module) -> bool:
    return _FUSION.mount(root, reject_reason=_site_reject_reason, logger=logger)


def unmount_sana_video_linear_attention(root: nn.Module) -> None:
    _FUSION.unmount(root)


def try_sana_video_linear_attention(
    site: nn.Module,
    query_rotate: torch.Tensor,
    key_rotate: torch.Tensor,
    value: torch.Tensor,
    normalizer: torch.Tensor,
) -> torch.Tensor | None:

    if not (
        _FUSION.is_enabled(site)
        and query_rotate.is_cuda
        and query_rotate.dtype == torch.bfloat16
        and key_rotate.dtype == query_rotate.dtype
        and value.dtype == query_rotate.dtype
        and query_rotate.dim() == 4
        and query_rotate.shape == key_rotate.shape == value.shape
        and query_rotate.shape[0] == 1
        and normalizer.is_cuda
    ):
        return None

    batch_size, num_heads, head_dim, _ = value.shape
    scores = torch.bmm(
        value.flatten(0, 1),
        key_rotate.transpose(-1, -2).flatten(0, 1),
        out_dtype=torch.float32,
    ).view(batch_size, num_heads, head_dim, head_dim)
    return (scores @ query_rotate.float()) * normalizer
