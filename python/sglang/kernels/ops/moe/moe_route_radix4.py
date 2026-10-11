from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.jit.utils.common import is_hip_runtime

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_NUM_EXPERTS = 896
_TOPK = 16


_MAX_TOKENS = 1024

logger = logging.getLogger(__name__)


def supported_hardware() -> bool:

    if not is_hip_runtime() or not torch.cuda.is_available():
        return False
    gcn_arch = torch.cuda.get_device_properties(0).gcnArchName
    return any(arch in gcn_arch for arch in ("gfx942", "gfx950"))


@cache_once
def build() -> Module:

    return load_jit(
        "moe_route_radix4",
        cuda_files=["moe/route_radix4_hip.cuh"],
        cuda_wrappers=[("run", "RouteRadix4Kernel::run")],


        extra_cuda_cflags=["-O3"],
    )


@cache_once
def available() -> bool:

    if not supported_hardware():
        return False
    try:
        build()
        return True
    except Exception as e:
        logger.warning(f"Failed to load the JIT ROCm radix router: {e}")
        return False


def covered(
    scores: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    num_expert_group: Optional[int],
    topk_group: Optional[int],
) -> bool:

    return (
        scores.dim() == 2
        and scores.size(0) <= _MAX_TOKENS
        and scores.size(1) == _NUM_EXPERTS
        and int(topk) == _TOPK
        and scores.dtype in (torch.bfloat16, torch.float32)
        and bias.dtype == scores.dtype
        and scores.stride(1) == 1
        and bias.is_contiguous()
        and (num_expert_group or 1) == 1
        and (topk_group or 1) == 1
    )


def route_radix4(
    scores: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
) -> Tuple[torch.Tensor, torch.Tensor]:

    M = scores.shape[0]
    out_w = torch.empty((M, topk), dtype=torch.float32, device=scores.device)
    out_i = torch.empty((M, topk), dtype=torch.int32, device=scores.device)
    build().run(
        scores,
        bias,
        out_w,
        out_i,
        topk,
        float(routed_scaling_factor),
        bool(renormalize),
    )
    return out_w, out_i
