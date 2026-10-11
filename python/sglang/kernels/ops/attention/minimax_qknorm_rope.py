from __future__ import annotations

from typing import TYPE_CHECKING, List, Sequence, Tuple

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_MAX_GROUPS = 4


@cache_once
def _jit_module(pos_dtype, head_dim, rope_dim) -> Module:
    args = make_cpp_args(pos_dtype, head_dim, rope_dim, is_arch_support_pdl())
    return load_jit(
        "fused_gemma_qknorm_rope",
        *args,
        cuda_files=["minimax/fused_gemma_qknorm_rope.cuh"],
        cuda_wrappers=[("fused_gemma_qknorm_rope", f"fused_gemma_qknorm_rope<{args}>")],
    )


@register_custom_op(mutates_args=["qkv"])
def _fused_gemma_qknorm_rope(
    qkv: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    off0: int,
    cnt0: int,
    off1: int,
    cnt1: int,
    off2: int,
    cnt2: int,
    off3: int,
    cnt3: int,
    num_groups: int,
    eps: float,
) -> None:


    module = _jit_module(positions.dtype, 128, 64)
    module.fused_gemma_qknorm_rope(
        qkv,
        w0,
        w1,
        w2,
        w3,
        cos_sin_cache,
        positions,
        off0,
        cnt0,
        off1,
        cnt1,
        off2,
        cnt2,
        off3,
        cnt3,
        num_groups,
        eps,
    )


def minimax_qknorm_rope_grouped(
    qkv: torch.Tensor,
    groups: Sequence[Tuple[torch.Tensor, int, int]],
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
) -> torch.Tensor:

    groups = [(w, off, cnt) for (w, off, cnt) in groups if cnt > 0]
    num_groups = len(groups)
    assert (
        1 <= num_groups <= _MAX_GROUPS
    ), f"need 1..{_MAX_GROUPS} groups, got {num_groups}"

    weights: List[torch.Tensor] = [g[0] for g in groups]
    offsets: List[int] = [int(g[1]) for g in groups]
    counts: List[int] = [int(g[2]) for g in groups]


    while len(weights) < _MAX_GROUPS:
        weights.append(weights[0])
        offsets.append(0)
        counts.append(0)

    _fused_gemma_qknorm_rope(
        qkv,
        weights[0],
        weights[1],
        weights[2],
        weights[3],
        cos_sin_cache,
        positions,
        offsets[0],
        counts[0],
        offsets[1],
        counts[1],
        offsets[2],
        counts[2],
        offsets[3],
        counts[3],
        num_groups,
        eps,
    )
    return qkv


def minimax_qknorm_rope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    nq: int,
    nk: int,
    nv: int,
    eps: float,
) -> torch.Tensor:

    return minimax_qknorm_rope_grouped(
        qkv,
        [(q_weight, 0, nq), (k_weight, nq, nk)],
        cos_sin_cache,
        positions,
        eps,
    )
