from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_SUPPORTED_HEADS = {3, 6, 12}
_CONV_STATE_W = 3


@cache_once
def _jit_kda_fused_decode_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "kda_fused_decode",
        *args,
        cuda_files=["attention/kda_fused_decode.cuh"],
        cuda_wrappers=[("run", f"KdaFusedDecodeKernel<{args}>::run")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )


def covered(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    conv_states: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    onorm_g: torch.Tensor,
) -> bool:

    if ssm_states.ndim < 4:
        return False
    H, V, K = ssm_states.shape[-3:]
    if H not in _SUPPORTED_HEADS:
        return False
    seg = H * 128
    conv_dim = 3 * seg
    if mixed_qkv.ndim != 2 or mixed_qkv.shape[-1] != conv_dim:
        return False
    return (
        V == 128
        and K == 128
        and a.ndim == 2
        and a.shape[-1] == seg
        and b.ndim == 2
        and b.shape[-1] == H
        and onorm_g.ndim == 2
        and onorm_g.shape[-1] == seg
        and conv_states.ndim == 3
        and conv_states.shape[-2:] == (_CONV_STATE_W, conv_dim)
        and mixed_qkv.dtype == torch.bfloat16
        and a.dtype == torch.bfloat16
        and b.dtype == torch.bfloat16
        and onorm_g.dtype == torch.bfloat16
        and conv_states.dtype == torch.bfloat16
        and ssm_states.dtype == torch.float32
        and cache_indices.dtype == torch.int32
        and mixed_qkv.stride(-1) == 1
        and a.stride(-1) == 1
        and b.stride(-1) == 1
        and onorm_g.stride(-1) == 1
        and conv_states.stride(-1) == 1


        and ssm_states.stride(-1) == 1
        and ssm_states.stride(-2) == K
        and ssm_states.stride(-3) == V * K
        and cache_indices.is_contiguous()
    )


def kda_fused_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    conv_states: torch.Tensor,
    w_q_t: torch.Tensor,
    w_k_t: torch.Tensor,
    w_v_t: torch.Tensor,
    conv_bias: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    onorm_g: torch.Tensor,
    onorm_weight: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    scale: float,
    onorm_eps: float,
    lower_bound: Optional[float] = None,
) -> torch.Tensor:

    B = mixed_qkv.shape[0]
    H = ssm_states.shape[-3]
    seg = H * 128
    out = torch.empty((B, seg), dtype=torch.bfloat16, device=mixed_qkv.device)
    _jit_kda_fused_decode_module().run(
        mixed_qkv,
        a,
        b,
        conv_states,
        w_q_t,
        w_k_t,
        w_v_t,
        conv_bias,
        A_log,
        dt_bias,
        onorm_g,
        onorm_weight,


        ssm_states,
        cache_indices,
        out,
        float(scale),
        float(onorm_eps),
        float(lower_bound) if lower_bound is not None else 0.0,
        lower_bound is not None,
    )
    return out.view(1, B, H, 128)
