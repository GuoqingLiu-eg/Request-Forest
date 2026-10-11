# Copyright 2025 XunhaoLai. All rights reserved.

import functools
from collections import deque
from typing import Any, Callable, List, Optional, Tuple

import torch
import triton
import triton.language as tl

_tma_keep_alive_buf = deque(maxlen=200)


SPARSE_KV_FP8_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
)


def check_sparse_kv_fp8(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: Optional[torch.Tensor],
    *,
    label: str,
) -> bool:

    is_fp8 = k_cache.dtype in SPARSE_KV_FP8_DTYPES
    if q.dtype == torch.float8_e4m3fn:
        assert k_cache.dtype == torch.float8_e4m3fn, (
            f"sparse {label} with fp8 Q requires an fp8_e4m3fn K cache, "
            f"got {k_cache.dtype}"
        )
    else:
        assert q.dtype in (
            torch.bfloat16,
            torch.float16,
        ), f"sparse {label} expects Q dtype bf16/fp16/fp8_e4m3fn, got {q.dtype}"
        assert k_cache.dtype == q.dtype or is_fp8, (
            f"sparse {label} expects K cache dtype == Q dtype ({q.dtype}) "
            f"or fp8, got {k_cache.dtype}"
        )
    if v_cache is not None:
        assert v_cache.dtype == k_cache.dtype
    return is_fp8


def sparse_out_dtype(q: torch.Tensor) -> torch.dtype:

    return torch.bfloat16 if q.dtype == torch.float8_e4m3fn else q.dtype


def unit_scale(scale: Optional[float]) -> float:

    return 1.0 if scale is None else scale


try:
    make_tensor_descriptor = tl.make_tensor_descriptor
except Exception:
    make_tensor_descriptor = tl._experimental_make_tensor_descriptor


def robust_allocator(size: int, alignment: int, stream: int = None):

    tensor = torch.empty(size, device="cuda", dtype=torch.uint8)
    _tma_keep_alive_buf.append(tensor)
    return tensor


def tensor_cache(maxsize: int = 8):


    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:


        _cache: list = []

        def _args_match(args: tuple, cached_args: tuple) -> bool:
            if len(args) != len(cached_args):
                return False
            for i in range(len(args)):
                if args[i] is not cached_args[i]:
                    return False
            return True

        def _kwargs_match(kwargs: dict, cached_kwargs: dict) -> bool:
            if not kwargs and not cached_kwargs:
                return True
            if kwargs.keys() != cached_kwargs.keys():
                return False
            for k, v in kwargs.items():
                if v is not cached_kwargs[k]:
                    return False
            return True

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:

            for i in range(len(_cache) - 1, -1, -1):
                cached_args, cached_kwargs, cached_result = _cache[i]
                if _args_match(args, cached_args) and _kwargs_match(
                    kwargs, cached_kwargs
                ):

                    if i != len(_cache) - 1:
                        _cache.append(_cache.pop(i))
                    return cached_result


            result = fn(*args, **kwargs)


            if len(_cache) >= maxsize:
                _cache.pop(0)
            _cache.append((args, kwargs, result))
            return result


        wrapper.cache_clear = lambda: _cache.clear()
        wrapper.cache_info = lambda: {"size": len(_cache), "maxsize": maxsize}

        return wrapper

    return decorator


@tensor_cache(maxsize=8)
def get_cu_seqblocks(
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    block_size_q: int,
    block_size_k: int,
    seqlens_cpu: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, int, int, torch.Tensor, int, int]:

    cu_seqblocks_q = torch.zeros_like(cu_seqlens)
    cu_seqblocks_k = torch.zeros_like(cu_seqlens)
    seq_lens = torch.diff(cu_seqlens)
    seqblocks_q = (seq_lens + block_size_q - 1) // block_size_q
    seqblocks_k = (seq_lens + block_size_k - 1) // block_size_k
    max_seqblock_q = (max_seqlen + block_size_q - 1) // block_size_q
    max_seqblock_k = (max_seqlen + block_size_k - 1) // block_size_k
    cu_seqblocks_q[1:] = seqblocks_q
    cu_seqblocks_k[1:] = seqblocks_k
    cu_seqblocks_q.cumsum_(0)
    cu_seqblocks_k.cumsum_(0)
    if seqlens_cpu is not None:

        all_seqblock_q = sum(
            (s + block_size_q - 1) // block_size_q for s in seqlens_cpu
        )
        all_seqblock_k = sum(
            (s + block_size_k - 1) // block_size_k for s in seqlens_cpu
        )
    else:
        all_seqblock_q = seqblocks_q.sum().item()
        all_seqblock_k = seqblocks_k.sum().item()
    return (
        cu_seqblocks_q,
        max_seqblock_q,
        all_seqblock_q,
        cu_seqblocks_k,
        max_seqblock_k,
        all_seqblock_k,
    )


@triton.jit
def _compare_and_swap(
    x,
    ids,
    flip,
    i: tl.constexpr,
    n_dims: tl.constexpr,
):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * 2**i, 2, 2 ** (n_dims - i - 1)]
    y = tl.reshape(x, shape)

    mask = tl.arange(0, 2)[None, :, None]
    left = tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape).to(y.dtype)
    right = tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape).to(y.dtype)
    left = tl.reshape(left, x.shape)
    right = tl.reshape(right, x.shape)

    y_idx = tl.reshape(ids, shape)
    left_idx = tl.broadcast_to(tl.sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = tl.broadcast_to(tl.sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = tl.reshape(left_idx, x.shape).to(y_idx.dtype)
    right_idx = tl.reshape(right_idx, x.shape).to(y_idx.dtype)

    idtype = tl.core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    cond = (left > right) != flip
    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))
    new_ids = ids ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(ids))
    return ret.to(x.dtype, bitcast=True), new_ids


@triton.jit
def _bitonic_merge(
    x,
    ids,
    stage: tl.constexpr,
    order: tl.constexpr,
    n_dims: tl.constexpr,
):
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)


    if order == 2:
        shape: tl.constexpr = [
            n_outer * 2 ** (n_dims - 1 - stage),
            2,
            2**stage,
        ]
        flip = tl.reshape(
            tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape
        )
    else:
        flip = order

    for i in tl.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


@triton.jit
def _sort_ids_ascending(ids, valid_n, BLOCK_SIZE_T: tl.constexpr):

    off = tl.arange(0, BLOCK_SIZE_T)
    invalid = (off >= valid_n) | (ids < 0)
    key = tl.where(invalid, 0x40000000 + off, ids)
    rank = tl.sum(tl.where(key[None, :] < key[:, None], 1, 0), axis=1)
    sorted_key = tl.sum(
        tl.where(rank[None, :] == off[:, None], key[None, :], 0), axis=1
    )
    return tl.where(sorted_key >= 0x40000000, -1, sorted_key)
