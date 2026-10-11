from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, empty_sentinel, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_inkling_all_reduce_module(dtype: torch.dtype, world_size: int) -> Module:
    args = make_cpp_args(dtype, world_size)
    return load_jit(
        "inkling_all_reduce",
        *args,
        cuda_files=["inkling/inkling_all_reduce.cuh"],
        cuda_wrappers=[
            ("two_shot_all_reduce", f"inkling_two_shot_all_reduce<{args}>"),
            ("two_shot_all_reduce_fused", f"inkling_two_shot_all_reduce_fused<{args}>"),
            ("multimem_one_shot_fused", f"inkling_multimem_one_shot_fused<{args}>"),
            ("multimem_full_oneshot", f"inkling_multimem_full_oneshot<{args}>"),
            ("multimem_push_oneshot", f"inkling_multimem_push_oneshot<{args}>"),
        ],
    )


MAX_BARRIER_BLOCKS = 256
STATE_SIZE = 8 + MAX_BARRIER_BLOCKS


def flags_numel(world_size: int) -> int:
    return world_size * (1 + MAX_BARRIER_BLOCKS)


_AR_TUNED_TP4 = {
    1: ("v5", 1, 1024),
    2: ("v5", 1, 1024),
    3: ("v5", 8, 512),
    4: ("v5", 8, 512),
    6: ("v5", 8, 1024),
    8: ("v5", 8, 1024),
    12: ("v5", 8, 512),
    16: ("v5", 8, 512),
    24: ("v5", 8, 1024),
    32: ("v5", 8, 1024),
    48: ("v5", 48, 1024),
    64: ("v5", 48, 1024),
    96: ("v5", 64, 1024),
    128: ("mm", 0, 0),
    192: ("mm", 0, 0),
    256: ("v3b", 64, 1024),
    384: ("v3b", 32, 1024),
    512: ("v3b", 32, 1024),
    768: ("v3b", 48, 512),
    1024: ("v3b", 32, 1024),
    1536: ("v3", 64, 512),
    2048: ("v3", 64, 512),
    3072: ("v3", 96, 512),
    4096: ("v3", 96, 512),
    6144: ("v3", 64, 512),
    8192: ("v3", 32, 1024),
    12288: ("v3", 96, 512),
    16384: ("v3", 96, 512),
}


_AR_TUNED_TP8 = {
    1: ("v4", 1, 1024),
    2: ("v4", 1, 1024),
    3: ("mm", 0, 0),
    4: ("mm", 0, 0),
    6: ("mm", 0, 0),
    8: ("mm", 0, 0),
    12: ("mm", 0, 0),
    16: ("mm", 0, 0),
    24: ("mm", 0, 0),
    32: ("mm", 0, 0),
    48: ("mm", 0, 0),
    64: ("mm", 0, 0),
    96: ("mm", 0, 0),
    128: ("mm", 0, 0),
    192: ("mm", 0, 0),
    256: ("mm", 0, 0),
    384: ("mm", 0, 0),
    512: ("mm", 0, 0),
    768: ("mm", 0, 0),
    1024: ("v3", 32, 512),
    1536: ("v3", 16, 1024),
    2048: ("v3", 32, 512),
    3072: ("v3", 48, 512),
    4096: ("v3", 48, 512),
    6144: ("v3", 64, 512),
    8192: ("v3", 96, 256),
    12288: ("v3", 64, 512),
    16384: ("v3", 64, 512),
}
_AR_TUNED = {4: _AR_TUNED_TP4, 8: _AR_TUNED_TP8}
_AR_TUNED_TOKENS = sorted(_AR_TUNED_TP4)
assert all(
    set(t) == set(_AR_TUNED_TP4) for t in _AR_TUNED.values()
), "all tuned tables must share the same token grid"


def select_ar_config(num_tokens: int, world_size: int = 4):

    table = _AR_TUNED.get(world_size, _AR_TUNED_TP4)
    for t in _AR_TUNED_TOKENS:
        if num_tokens <= t:
            return table[t]
    return table[_AR_TUNED_TOKENS[-1]]


def compile_inkling_all_reduce(dtype: torch.dtype, world_size: int) -> None:

    _jit_inkling_all_reduce_module(dtype, world_size)


def inkling_two_shot_all_reduce(
    buffer: torch.Tensor,
    peer_ptrs_dev: int,
    rank: int,
    world_size: int,
    num_items: int,
) -> None:

    module = _jit_inkling_all_reduce_module(buffer.dtype, world_size)
    module.two_shot_all_reduce(buffer, peer_ptrs_dev, rank, num_items)


def inkling_two_shot_all_reduce_fused(
    buffer: torch.Tensor,
    data_ptrs_dev: int,
    flag_ptrs_dev: int,
    state_ptr: int,
    rank: int,
    world_size: int,
    num_items: int,
    num_blocks: int = 0,
    block_size: int = 0,
    shared: torch.Tensor | None = None,
) -> None:

    module = _jit_inkling_all_reduce_module(buffer.dtype, world_size)
    module.two_shot_all_reduce_fused(
        buffer,
        data_ptrs_dev,
        flag_ptrs_dev,
        state_ptr,
        rank,
        num_items,
        num_blocks,
        block_size,
        shared if shared is not None else empty_sentinel(buffer.device, buffer.dtype),
    )


def inkling_multimem_one_shot_fused(
    buffer: torch.Tensor,
    multicast_ptr: int,
    flag_ptrs_dev: int,
    state_ptr: int,
    rank: int,
    world_size: int,
    num_items: int,
    num_blocks: int = 0,
    block_size: int = 0,
    per_block_barrier: bool = False,
    shared: torch.Tensor | None = None,
) -> None:

    module = _jit_inkling_all_reduce_module(buffer.dtype, world_size)
    module.multimem_one_shot_fused(
        buffer,
        multicast_ptr,
        flag_ptrs_dev,
        state_ptr,
        rank,
        num_items,
        num_blocks,
        block_size,
        int(per_block_barrier),
        shared if shared is not None else empty_sentinel(buffer.device, buffer.dtype),
    )


def inkling_multimem_full_oneshot(
    in_buffer: torch.Tensor,
    out_buffer: torch.Tensor,
    multicast_ptr: int,
    flag_ptrs_dev: int,
    state_ptr: int,
    rank: int,
    world_size: int,
    num_items: int,
    num_blocks: int = 0,
    block_size: int = 0,
    shared: torch.Tensor | None = None,
) -> None:

    module = _jit_inkling_all_reduce_module(in_buffer.dtype, world_size)
    module.multimem_full_oneshot(
        in_buffer,
        out_buffer,
        multicast_ptr,
        flag_ptrs_dev,
        state_ptr,
        rank,
        num_items,
        num_blocks,
        block_size,
        (
            shared
            if shared is not None
            else empty_sentinel(in_buffer.device, in_buffer.dtype)
        ),
    )


def inkling_multimem_push_oneshot(
    in_buffer: torch.Tensor,
    out_buffer: torch.Tensor,
    mc_stage_ptr: int,
    local_stage_ptr: int,
    flag_ptrs_dev: int,
    state_ptr: int,
    rank: int,
    world_size: int,
    num_items: int,
    num_blocks: int = 0,
    block_size: int = 0,
    per_block_barrier: bool = False,
    shared: torch.Tensor | None = None,
) -> None:

    module = _jit_inkling_all_reduce_module(in_buffer.dtype, world_size)
    module.multimem_push_oneshot(
        in_buffer,
        out_buffer,
        mc_stage_ptr,
        local_stage_ptr,
        flag_ptrs_dev,
        state_ptr,
        rank,
        num_items,
        num_blocks,
        block_size,
        int(per_block_barrier),
        (
            shared
            if shared is not None
            else empty_sentinel(in_buffer.device, in_buffer.dtype)
        ),
    )
