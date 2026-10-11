from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.kernels.registry import register_kernel
from sglang.kernels.selector import get_kernel
from sglang.kernels.spec import (
    CapabilityRequirement,
    FormatSignature,
    KernelBackend,
    KernelSpec,
)

if TYPE_CHECKING:
    import torch

_CUDA = frozenset({CapabilityRequirement.CUDA})


register_kernel(
    KernelSpec(
        op="mamba.causal_conv1d_fwd",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.mamba.causal_conv1d:causal_conv1d_fwd",
        capabilities=_CUDA,
        format_signature=FormatSignature(
            in_place=True, description="causal depthwise conv1d forward (prefill)"
        ),
        description="Causal conv1d forward (sglang.kernels.jit).",
    )
)
register_kernel(
    KernelSpec(
        op="mamba.causal_conv1d_update",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.mamba.causal_conv1d:causal_conv1d_update",
        capabilities=_CUDA,
        format_signature=FormatSignature(
            in_place=True, description="causal depthwise conv1d update (decode)"
        ),
        description="Causal conv1d update (sglang.kernels.jit).",
    )
)


def causal_conv1d_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias_: Optional[torch.Tensor],
    conv_states: Optional[torch.Tensor],
    query_start_loc: Optional[torch.Tensor],
    cache_indices: Optional[torch.Tensor],
    has_initial_state: Optional[torch.Tensor],
    silu_activation: bool,
    pad_slot_id: int,
):

    return get_kernel("mamba.causal_conv1d_fwd", KernelBackend.JIT)(
        x,
        weight,
        bias_,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        silu_activation,
        pad_slot_id,
    )


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias_: Optional[torch.Tensor],
    silu_activation: bool,
    cache_seqlens: Optional[torch.Tensor],
    conv_state_indices: Optional[torch.Tensor],
    pad_slot_id: int,
):

    return get_kernel("mamba.causal_conv1d_update", KernelBackend.JIT)(
        x,
        conv_state,
        weight,
        bias_,
        silu_activation,
        cache_seqlens,
        conv_state_indices,
        pad_slot_id,
    )


__all__ = ["causal_conv1d_fwd", "causal_conv1d_update"]


for _mod, _fn in [
    ("triton_ops.ssd_combined", "mamba_chunk_scan_combined"),
    ("triton_ops.mamba_ssm", "selective_state_update"),
    ("causal_conv1d_triton", "causal_conv1d_fn"),
    ("mamba_state_scatter_triton", "fused_mamba_state_scatter_with_mask"),
]:
    register_kernel(
        KernelSpec(
            op=f"mamba.{_fn}",
            backend=KernelBackend.TRITON,
            target=f"sglang.kernels.ops.mamba.{_mod}:{_fn}",
        )
    )
del _mod, _fn
