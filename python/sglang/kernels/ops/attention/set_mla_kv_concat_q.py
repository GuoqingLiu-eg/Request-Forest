from __future__ import annotations

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


@cache_once
def set_mla_kv_concat_q_module(
    nope_bytes: int, rope_bytes: int, use_pdl: bool
) -> Module:
    args = make_cpp_args(nope_bytes, rope_bytes, use_pdl)
    return load_jit(
        f"set_mla_kv_concat_q_{nope_bytes}_{rope_bytes}",
        *args,
        cuda_files=["elementwise/set_mla_kv_concat_q.cuh"],
        cuda_wrappers=[
            ("set_mla_kv_concat_q", f"SetMlaKVConcatQKernel<{args}>::run"),
        ],
    )


@cache_once
def can_use_set_mla_kv_concat_q(nope_bytes: int, rope_bytes: int) -> bool:

    if torch.cuda.get_device_capability()[0] < 9:
        return False
    if nope_bytes % 4 != 0 or rope_bytes % 4 != 0:
        return False
    if (nope_bytes + rope_bytes) % 16 != 0:
        return False


    if (nope_bytes // 2) % (8 * 32) != 0 or rope_bytes // 2 != 2 * 32:
        return False
    try:
        set_mla_kv_concat_q_module(nope_bytes, rope_bytes, is_arch_support_pdl())
        return True
    except Exception:
        return False


def _row_aligned(t: torch.Tensor, align: int) -> bool:

    if t.data_ptr() % align != 0:
        return False
    esize = t.element_size()
    return all(s * esize % align == 0 for s in t.stride()[:-1])


def covered(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
) -> bool:

    if not (
        kv_buffer.dtype == torch.bfloat16
        and k_nope.dtype == torch.bfloat16
        and k_rope.dtype == torch.bfloat16
        and q_nope.dtype == torch.bfloat16
        and q_rope.dtype == torch.bfloat16
    ):
        return False
    if loc.dtype not in (torch.int32, torch.int64):
        return False
    if loc.dim() != 1 or not loc.is_contiguous():
        return False
    if not (
        k_nope.shape[0]
        == k_rope.shape[0]
        == loc.shape[0]
        == q_nope.shape[0]
        == q_rope.shape[0]
    ):
        return False
    if q_nope.shape[1] != q_rope.shape[1]:
        return False
    nope_bytes = k_nope.shape[-1] * 2
    rope_bytes = k_rope.shape[-1] * 2
    if q_nope.shape[-1] * 2 != nope_bytes or q_rope.shape[-1] * 2 != rope_bytes:
        return False
    if not can_use_set_mla_kv_concat_q(nope_bytes, rope_bytes):
        return False

    if any(t.stride(-1) != 1 for t in (kv_buffer, k_nope, k_rope, q_nope, q_rope)):
        return False
    return (
        _row_aligned(kv_buffer, 16)
        and _row_aligned(k_nope, 16)
        and _row_aligned(k_rope, 4)
        and _row_aligned(q_nope, 16)
        and _row_aligned(q_rope, 4)
    )


def _pick_num_warps(total_items: int) -> int:

    return 4 if total_items <= 768 else 8


def set_mla_kv_concat_q(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    num_warps: int = 0,
) -> torch.Tensor:

    n_loc = loc.shape[0]
    src_nope = cache_k_nope.view(n_loc, -1) if cache_k_nope.dim() != 2 else cache_k_nope
    src_rope = cache_k_rope.view(n_loc, -1) if cache_k_rope.dim() != 2 else cache_k_rope
    buf = kv_buffer.view(kv_buffer.shape[0], -1) if kv_buffer.dim() != 2 else kv_buffer

    q_out = torch.empty(
        (*q_nope.shape[:-1], q_nope.shape[-1] + q_rope.shape[-1]),
        dtype=q_nope.dtype,
        device=q_nope.device,
    )

    nope_bytes = src_nope.shape[-1] * src_nope.element_size()
    rope_bytes = src_rope.shape[-1] * src_rope.element_size()
    if num_warps <= 0:
        num_warps = _pick_num_warps(n_loc + q_nope.shape[0] * q_nope.shape[1])

    module = set_mla_kv_concat_q_module(nope_bytes, rope_bytes, is_arch_support_pdl())
    module.set_mla_kv_concat_q(
        buf, loc, src_nope, src_rope, q_nope, q_rope, q_out, num_warps
    )
    return q_out


@cache_once
def set_mla_kv_concat_q_fp8_module(use_pdl: bool) -> Module:
    args = make_cpp_args(use_pdl)
    return load_jit(
        "set_mla_kv_concat_q_fp8",
        *args,
        cuda_files=["elementwise/set_mla_kv_concat_q.cuh"],
        cuda_wrappers=[
            ("set_mla_kv_concat_q_fp8", f"SetMlaKVConcatQFp8Kernel<{args}>::run"),
        ],
    )


@cache_once
def can_use_set_mla_kv_concat_q_fp8() -> bool:

    if torch.cuda.get_device_capability()[0] < 9:
        return False
    try:
        set_mla_kv_concat_q_fp8_module(is_arch_support_pdl())
        return True
    except Exception:
        return False


def covered_fp8(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
) -> bool:

    if not (
        kv_buffer.dtype in (torch.float8_e4m3fn, torch.uint8)
        and k_nope.dtype == torch.bfloat16
        and k_rope.dtype == torch.bfloat16
        and q_nope.dtype == torch.bfloat16
        and q_rope.dtype == torch.bfloat16
    ):
        return False
    if loc.dtype not in (torch.int32, torch.int64):
        return False
    if loc.dim() != 1 or not loc.is_contiguous():
        return False
    if not (
        k_nope.shape[0]
        == k_rope.shape[0]
        == loc.shape[0]
        == q_nope.shape[0]
        == q_rope.shape[0]
    ):
        return False
    if q_nope.shape[1] != q_rope.shape[1]:
        return False
    if (
        k_nope.shape[-1] != 512
        or k_rope.shape[-1] != 64
        or q_nope.shape[-1] != 512
        or q_rope.shape[-1] != 64
        or kv_buffer.shape[-1] < 576
    ):
        return False
    if not can_use_set_mla_kv_concat_q_fp8():
        return False
    if any(t.stride(-1) != 1 for t in (kv_buffer, k_nope, k_rope, q_nope, q_rope)):
        return False
    if kv_buffer.data_ptr() % 16 != 0 or kv_buffer.stride(0) % 16 != 0:
        return False
    return (
        _row_aligned(k_nope, 16)
        and _row_aligned(k_rope, 4)
        and _row_aligned(q_nope, 16)
        and _row_aligned(q_rope, 4)
    )


def set_mla_kv_concat_q_fp8(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    num_warps: int = 0,
    dcp_world_size: int = 1,
    dcp_rank: int = 0,
) -> torch.Tensor:

    n_loc = loc.shape[0]
    src_nope = cache_k_nope.view(n_loc, -1) if cache_k_nope.dim() != 2 else cache_k_nope
    src_rope = cache_k_rope.view(n_loc, -1) if cache_k_rope.dim() != 2 else cache_k_rope
    buf = kv_buffer.view(kv_buffer.shape[0], -1) if kv_buffer.dim() != 2 else kv_buffer

    q_out = torch.empty(
        (q_nope.shape[0], q_nope.shape[1], 576),
        dtype=torch.float8_e4m3fn,
        device=q_nope.device,
    )
    if num_warps <= 0:
        num_warps = _pick_num_warps(n_loc + q_nope.shape[0] * q_nope.shape[1])

    module = set_mla_kv_concat_q_fp8_module(is_arch_support_pdl())
    module.set_mla_kv_concat_q_fp8(
        buf,
        loc,
        src_nope,
        src_rope,
        q_nope,
        q_rope,
        q_out,
        num_warps,
        dcp_world_size,
        dcp_rank,
    )
    return q_out
