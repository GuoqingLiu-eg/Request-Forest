from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

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


@cache_once
def _jit_rotary_embedding_module() -> Module:
    return load_jit(
        "rotary_embedding",
        cuda_files=["elementwise/pos_enc.cuh"],
        cuda_wrappers=[("rotary_embedding", "RotaryEmbeddingKernel::run")],
    )


@cache_once
def _jit_fused_rope_module(
    is_neox: bool, rope_dim: int, dtype: torch.dtype, q_dtype: torch.dtype
) -> Module:


    args = make_cpp_args(is_neox, rope_dim, is_arch_support_pdl(), dtype, q_dtype)
    return load_jit(
        "fused_rope",
        *args,
        cuda_files=["elementwise/rope.cuh"],
        cuda_wrappers=[
            ("run_rope", f"FusedRopeKernel<{args}>::run"),
            ("run_rope_store", f"FusedRopeKernel<{args}>::run_fused"),
        ],
    )


@register_custom_op(
    op_name="rotary_embedding_with_key",
    mutates_args=["query", "key"],
)
def rotary_embedding_with_key(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
) -> None:
    module = _jit_rotary_embedding_module()
    module.rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox)


@register_custom_op(
    op_name="rotary_embedding_without_key",
    mutates_args=["query"],
)
def rotary_embedding_without_key(
    positions: torch.Tensor,
    query: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
) -> None:
    module = _jit_rotary_embedding_module()
    module.rotary_embedding(positions, query, None, head_size, cos_sin_cache, is_neox)


def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: Optional[torch.Tensor],
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
):
    if key is None:
        rotary_embedding_without_key(
            positions, query, head_size, cos_sin_cache, is_neox
        )
    else:
        rotary_embedding_with_key(
            positions, query, key, head_size, cos_sin_cache, is_neox
        )
    return query, key


@dataclass
class FusedSetKVBufferArg:


    value: torch.Tensor
    k_buffer: torch.Tensor
    v_buffer: torch.Tensor
    cache_loc: torch.Tensor


@register_custom_op(mutates_args=["q", "k"])
def apply_rope_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    is_neox: bool,
    rope_dim: int = 0,
) -> None:

    rope_dim = rope_dim or cos_sin_cache.size(-1)
    module = _jit_fused_rope_module(is_neox, rope_dim, k.dtype, q.dtype)
    module.run_rope(q, k, cos_sin_cache, positions)


@register_custom_op(mutates_args=["q", "k", "k_cache", "v_cache"])
def apply_rope_inplace_with_kvcache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    out_loc: torch.Tensor,
    *,
    is_neox: bool,
    rope_dim: int = 0,
) -> None:

    rope_dim = rope_dim or cos_sin_cache.size(-1)
    v = v.view_as(k)
    module = _jit_fused_rope_module(is_neox, rope_dim, k.dtype, q.dtype)
    module.run_rope_store(q, k, v, k_cache, v_cache, cos_sin_cache, positions, out_loc)


def apply_rope_with_cos_sin_cache_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    is_neox: bool,
    rope_dim: int = 0,
    fused_args: Optional[FusedSetKVBufferArg] = None,
) -> None:

    if fused_args is not None:
        apply_rope_inplace_with_kvcache(
            q,
            k,
            fused_args.value,
            fused_args.k_buffer,
            fused_args.v_buffer,
            cos_sin_cache,
            positions,
            fused_args.cache_loc,
            is_neox=is_neox,
            rope_dim=rope_dim,
        )
    else:
        apply_rope_inplace(
            q, k, cos_sin_cache, positions, is_neox=is_neox, rope_dim=rope_dim
        )
