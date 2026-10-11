from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_LOGITS_PAD = 264
_HIDDEN = 6144
_TOPK = 6
_N_SHARED = 2
_FUSED_MAX_TOKENS = 64


@cache_once
def _jit_module() -> Module:
    return load_jit(
        "inkling_gate_topk_renorm",
        "fast_math",
        cuda_files=["moe/inkling_gate_topk_renorm.cuh"],
        cuda_wrappers=[
            ("inkling_gate_topk_renorm", "inkling_gate_topk_renorm"),
            ("inkling_gate_topk_renorm_packed", "inkling_gate_topk_renorm_packed"),
            ("inkling_gate_topk_renorm_v2", "inkling_gate_topk_renorm_v2"),
            (
                "inkling_gate_topk_renorm_v2_packed",
                "inkling_gate_topk_renorm_v2_packed",
            ),
            ("inkling_gate_gemv", "inkling_gate_gemv"),
            ("inkling_gate_gemv_fused", "inkling_gate_gemv_fused"),
            ("inkling_gate_gemv_fused_packed", "inkling_gate_gemv_fused_packed"),
        ],
        extra_cuda_cflags=["-use_fast_math"],
    )


def _launch_inkling_gate_topk_renorm(
    logits: torch.Tensor,
    bias: torch.Tensor,
    global_scale: torch.Tensor,
    routed_w: torch.Tensor,
    shared_w: torch.Tensor,
    indices: torch.Tensor,
    route_scale: float,
) -> None:
    module = _jit_module()
    module.inkling_gate_topk_renorm(
        logits, bias, global_scale, routed_w, shared_w, indices, float(route_scale)
    )


def _check_gate_inputs(
    logits: torch.Tensor, bias: torch.Tensor, global_scale: torch.Tensor
) -> None:
    assert logits.is_cuda and logits.dtype == torch.float32 and logits.dim() == 2
    assert logits.shape[1] == 258 and logits.stride(1) == 1
    assert bias.is_cuda and bias.dtype == torch.float32 and bias.shape == (256,)
    assert global_scale.is_cuda and global_scale.dtype == torch.float32
    assert global_scale.numel() == 1


def inkling_gate_topk_renorm(
    logits: torch.Tensor,
    bias: torch.Tensor,
    global_scale: torch.Tensor,
    route_scale: float,
    *,
    return_packed: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor]
):

    _check_gate_inputs(logits, bias, global_scale)

    tokens = logits.shape[0]
    shared_w = torch.empty(
        (tokens, _N_SHARED), dtype=torch.float32, device=logits.device
    )
    if return_packed:
        packed = torch.empty((tokens, _TOPK), dtype=torch.int32, device=logits.device)
        if tokens == 0:
            return packed, shared_w
        _jit_module().inkling_gate_topk_renorm_packed(
            logits,
            bias.contiguous(),
            global_scale.contiguous(),
            packed,
            shared_w,
            float(route_scale),
        )
        return packed, shared_w

    routed_w = torch.empty((tokens, _TOPK), dtype=torch.float32, device=logits.device)
    indices = torch.empty((tokens, _TOPK), dtype=torch.int64, device=logits.device)
    if tokens == 0:
        return routed_w, shared_w, indices

    _launch_inkling_gate_topk_renorm(
        logits,
        bias.contiguous(),
        global_scale.contiguous(),
        routed_w,
        shared_w,
        indices,
        route_scale,
    )
    return routed_w, shared_w, indices


def inkling_gate_topk_renorm_v2(
    logits: torch.Tensor,
    bias: torch.Tensor,
    global_scale: torch.Tensor,
    route_scale: float,
    *,
    return_packed: bool = False,
    enable_pdl: bool = False,
    warps_per_block: int = 0,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:

    _check_gate_inputs(logits, bias, global_scale)
    assert logits.stride(0) % 8 == 0, f"rows must be 32B-aligned: {logits.stride()=}"

    tokens = logits.shape[0]
    shared_w = torch.empty(
        (tokens, _N_SHARED), dtype=torch.float32, device=logits.device
    )
    if return_packed:
        packed = torch.empty((tokens, _TOPK), dtype=torch.int32, device=logits.device)
        if tokens > 0:
            _jit_module().inkling_gate_topk_renorm_v2_packed(
                logits,
                bias.contiguous(),
                global_scale.contiguous(),
                packed,
                shared_w,
                float(route_scale),
                bool(enable_pdl),
                int(warps_per_block),
            )
        return None, None, shared_w, packed

    routed_w = torch.empty((tokens, _TOPK), dtype=torch.float32, device=logits.device)
    indices = torch.empty((tokens, _TOPK), dtype=torch.int32, device=logits.device)
    if tokens > 0:
        _jit_module().inkling_gate_topk_renorm_v2(
            logits,
            bias.contiguous(),
            global_scale.contiguous(),
            routed_w,
            shared_w,
            indices,
            float(route_scale),
            bool(enable_pdl),
            int(warps_per_block),
        )
    return routed_w, indices, shared_w, None


def _check_gemv_inputs(x: torch.Tensor, weight: torch.Tensor) -> None:
    assert x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 2
    assert x.shape[1] == _HIDDEN and x.stride(1) == 1 and x.stride(0) == _HIDDEN
    assert weight.is_cuda and weight.dtype == torch.bfloat16 and weight.dim() == 2
    assert weight.shape[0] >= 258 and weight.shape[1] == _HIDDEN
    assert weight.stride(1) == 1 and weight.stride(0) == _HIDDEN


def inkling_gate_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    enable_pdl: bool = False,
    experts_per_block: int = 0,
) -> torch.Tensor:

    _check_gemv_inputs(x, weight)
    tokens = x.shape[0]
    logits = torch.empty((tokens, _LOGITS_PAD), dtype=torch.float32, device=x.device)
    if tokens > 0:
        _jit_module().inkling_gate_gemv(
            x, weight, logits, bool(enable_pdl), int(experts_per_block)
        )
    return logits[:, :258]


_fused_scratch: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def _get_fused_scratch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = device.index if device.index is not None else torch.cuda.current_device()
    scratch = _fused_scratch.get(key)
    if scratch is None:


        assert (
            not torch.cuda.is_current_stream_capturing()
        ), "fused gate scratch must be allocated before CUDA graph capture"
        workspace = torch.empty(
            (_FUSED_MAX_TOKENS, _LOGITS_PAD), dtype=torch.float32, device=device
        )
        ticket = torch.zeros((1,), dtype=torch.int32, device=device)
        scratch = (workspace, ticket)
        _fused_scratch[key] = scratch
    return scratch


def ensure_gate_gemv_fused_scratch(device: torch.device) -> None:

    _get_fused_scratch(device)


def inkling_gate_gemv_fused(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    global_scale: torch.Tensor,
    route_scale: float,
    *,
    return_packed: bool = False,
    enable_pdl: bool = False,
    experts_per_block: int = 0,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:

    _check_gemv_inputs(x, weight)
    tokens = x.shape[0]
    assert tokens <= _FUSED_MAX_TOKENS, f"fused gate supports <= 64 tokens: {tokens=}"
    assert bias.is_cuda and bias.dtype == torch.float32 and bias.shape == (256,)
    assert global_scale.is_cuda and global_scale.dtype == torch.float32

    workspace, ticket = _get_fused_scratch(x.device)
    shared_w = torch.empty((tokens, _N_SHARED), dtype=torch.float32, device=x.device)
    if return_packed:
        packed = torch.empty((tokens, _TOPK), dtype=torch.int32, device=x.device)
        if tokens > 0:
            _jit_module().inkling_gate_gemv_fused_packed(
                x,
                weight,
                bias.contiguous(),
                global_scale.contiguous(),
                workspace,
                ticket,
                packed,
                shared_w,
                float(route_scale),
                bool(enable_pdl),
                int(experts_per_block),
            )
        return None, None, shared_w, packed

    routed_w = torch.empty((tokens, _TOPK), dtype=torch.float32, device=x.device)
    indices = torch.empty((tokens, _TOPK), dtype=torch.int32, device=x.device)
    if tokens > 0:
        _jit_module().inkling_gate_gemv_fused(
            x,
            weight,
            bias.contiguous(),
            global_scale.contiguous(),
            workspace,
            ticket,
            routed_w,
            shared_w,
            indices,
            float(route_scale),
            bool(enable_pdl),
            int(experts_per_block),
        )
    return routed_w, indices, shared_w, None
