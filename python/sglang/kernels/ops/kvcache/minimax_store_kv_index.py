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


@cache_once
def _jit_module(head_bytes: int) -> Module:


    args = make_cpp_args(head_bytes, is_arch_support_pdl())
    return load_jit(
        "minimax_store_kv_index",
        *args,
        cuda_files=["minimax/fused_store_kv_index.cuh"],
        cuda_wrappers=[("store_kv_index", f"store_kv_index<{args}>")],
    )


def store_kv_index(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    idx_k: torch.Tensor,
    idx_k_cache: torch.Tensor,
    idx_v: Optional[torch.Tensor],
    idx_v_cache: Optional[torch.Tensor],
    indices: torch.Tensor,
    *,
    num_kv_heads: int,
    head_bytes: int,
) -> None:

    has_v = idx_v is not None
    if not has_v:


        idx_v = idx_k
        idx_v_cache = idx_k_cache
    heads_per_token = 2 * num_kv_heads + 1 + (1 if has_v else 0)

    module = _jit_module(head_bytes)
    module.store_kv_index(
        k,
        v,
        k_cache,
        v_cache,
        idx_k,
        idx_k_cache,
        idx_v,
        idx_v_cache,
        indices,
        num_kv_heads,
        heads_per_token,
    )
