from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.ops.moe import moe_route_radix

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_HIDDEN = 3584
_GROUP_SIZE = 32
_NUM_GROUPS = _HIDDEN // _GROUP_SIZE


_MAX_TOKENS = 64


@cache_once
def _jit_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "moe_route_quant_fused",
        *args,
        cuda_files=["moe/route_quant_fused.cuh"],
        cuda_wrappers=[("run", f"RouteQuantFusedKernel<{args}>::run")],


        extra_cuda_cflags=["-O3"],
    )


@cache_once
def available() -> bool:
    import logging

    try:
        _jit_module()
        return True
    except Exception as e:
        logging.getLogger(__name__).warning(
            f"Failed to load the JIT fused route+quant kernel: {e}"
        )
        return False


def covered(
    scores: torch.Tensor, bias: torch.Tensor, topk: int, x: torch.Tensor
) -> bool:

    return (
        moe_route_radix.covered(scores, bias, topk)
        and x.dim() == 2
        and x.shape[0] == scores.shape[0]
        and 0 < x.shape[0] <= _MAX_TOKENS
        and x.shape[1] == _HIDDEN
        and x.dtype in (torch.bfloat16, torch.float32)
        and x.stride(1) == 1
        and x.data_ptr() % 32 == 0
        and (x.stride(0) * x.element_size()) % 32 == 0
    )


def route_quant_fused(
    scores: torch.Tensor,
    bias: torch.Tensor,
    x: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    apply_scale: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    M = scores.shape[0]
    device = scores.device
    out_w = torch.empty((M, topk), dtype=torch.float32, device=device)
    out_i = torch.empty((M, topk), dtype=torch.int32, device=device)
    out_packed = torch.empty((M, topk), dtype=torch.int32, device=device)
    out_q = torch.empty((M, _HIDDEN), dtype=torch.float8_e4m3fn, device=device)
    out_s = torch.empty((M, _NUM_GROUPS // 4), dtype=torch.int32, device=device)
    _jit_module().run(
        scores,
        bias,
        out_w,
        out_i,
        out_packed,
        x,
        out_q,
        out_s,
        topk,
        float(routed_scaling_factor),
        bool(renormalize),
        bool(apply_scale),
    )
    return out_w, out_i, out_packed, out_q, out_s
