from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)


@cache_once
def set_mla_kv_buffer_module(nope_bytes: int, rope_bytes: int, use_pdl: bool) -> Module:
    args = make_cpp_args(nope_bytes, rope_bytes, use_pdl)
    return load_jit(
        f"set_mla_kv_buffer_{nope_bytes}_{rope_bytes}",
        *args,
        cuda_files=["elementwise/set_mla_kv_buffer.cuh"],
        cuda_wrappers=[
            ("set_mla_kv_buffer", f"SetMlaKVBufferKernel<{args}>::run"),
        ],
    )


@cache_once
def can_use_set_mla_kv_buffer(nope_bytes: int, rope_bytes: int) -> bool:

    if nope_bytes % 4 != 0 or rope_bytes % 4 != 0:
        logger.warning(
            "Unsupported nope_bytes=%d rope_bytes=%d for JIT set_mla_kv_buffer:"
            " both must be multiples of 4",
            nope_bytes,
            rope_bytes,
        )
        return False
    if (nope_bytes + rope_bytes) % 16 != 0:
        logger.warning(
            "Unsupported nope_bytes=%d rope_bytes=%d for JIT set_mla_kv_buffer:"
            " (nope_bytes + rope_bytes) must be a multiple of 16 for TMA bulk store",
            nope_bytes,
            rope_bytes,
        )
        return False
    try:
        set_mla_kv_buffer_module(nope_bytes, rope_bytes, is_arch_support_pdl())
        return True
    except Exception as e:
        logger.warning(
            "Failed to load JIT set_mla_kv_buffer kernel "
            "with nope_bytes=%d rope_bytes=%d: %s",
            nope_bytes,
            rope_bytes,
            e,
        )
        return False


def _pick_num_warps(n_loc: int) -> int:


    return 4 if n_loc <= 768 else 8


def set_mla_kv_buffer(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
    num_warps: int = 0,
    *,
    reserved_skip_index: int = 0,
) -> None:

    n_loc = loc.shape[0]
    if n_loc == 0:
        return

    src_nope = cache_k_nope.view(n_loc, -1) if cache_k_nope.dim() != 2 else cache_k_nope
    src_rope = cache_k_rope.view(n_loc, -1) if cache_k_rope.dim() != 2 else cache_k_rope
    buf = kv_buffer.view(kv_buffer.shape[0], -1) if kv_buffer.dim() != 2 else kv_buffer

    nope_bytes = src_nope.shape[-1] * src_nope.element_size()
    rope_bytes = src_rope.shape[-1] * src_rope.element_size()
    if num_warps <= 0:
        num_warps = _pick_num_warps(n_loc)

    module = set_mla_kv_buffer_module(nope_bytes, rope_bytes, is_arch_support_pdl())
    module.set_mla_kv_buffer(
        buf,
        loc,
        src_nope,
        src_rope,
        num_warps,
        reserved_skip_index,
    )
