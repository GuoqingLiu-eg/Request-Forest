from __future__ import annotations

from typing import Optional, Tuple

import msgspec
import torch


class _Handoff(msgspec.Struct):

    request_x: Optional[torch.Tensor] = None

    produced_x: Optional[torch.Tensor] = None
    packed: Optional[torch.Tensor] = None
    x_q: Optional[torch.Tensor] = None
    x_s: Optional[torch.Tensor] = None


_handoff = _Handoff()


def stage(x: torch.Tensor) -> None:

    _handoff.request_x = x
    _handoff.produced_x = None


def clear() -> None:
    _handoff.request_x = None
    _handoff.produced_x = None
    _handoff.packed = None
    _handoff.x_q = None
    _handoff.x_s = None


def try_route_quant_fused(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    *,
    num_fused_shared_experts: int,
    renormalize: bool,
    routed_scaling_factor: Optional[float],
    apply_routed_scaling_factor_on_output: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:

    x = _handoff.request_x
    if x is None or num_fused_shared_experts != 0:
        return None

    from sglang.kernels.ops.moe import moe_route_quant_fused

    if (
        not moe_route_quant_fused.covered(gating_output, correction_bias, topk, x)
        or not moe_route_quant_fused.available()
    ):
        return None

    weights, ids, packed, x_q, x_s = moe_route_quant_fused.route_quant_fused(
        gating_output,
        correction_bias,
        x,
        topk,
        renormalize=renormalize,
        routed_scaling_factor=(
            routed_scaling_factor if routed_scaling_factor is not None else 1.0
        ),
        apply_scale=apply_routed_scaling_factor_on_output,
    )
    _handoff.request_x = None
    _handoff.produced_x = x
    _handoff.packed = packed
    _handoff.x_q = x_q
    _handoff.x_s = x_s
    return weights, ids


def take(
    x: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:

    produced = _handoff.produced_x
    if produced is None:
        return None
    if (
        produced.data_ptr() != x.data_ptr()
        or produced.shape != x.shape
        or produced.dtype != x.dtype
        or produced.stride() != x.stride()
    ):
        return None
    out = (_handoff.packed, _handoff.x_q, _handoff.x_s)
    _handoff.produced_x = None
    _handoff.packed = None
    _handoff.x_q = None
    _handoff.x_s = None
    return out
