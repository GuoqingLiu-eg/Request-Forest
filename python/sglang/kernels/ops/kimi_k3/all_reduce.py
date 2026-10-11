from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Optional

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

    from sglang.kernels.ops.communication.all_reduce import Communicator


@cache_once
def _jit_module(world_size: int) -> Module:
    args = make_cpp_args(world_size, is_arch_support_pdl())
    cls = f"AllReduceFusionKernel<{args}>"
    return load_jit(
        "kimi_k3_all_reduce",
        *args,
        cuda_files=["kimi_k3/comm/ar_fusion.cuh"],
        cuda_wrappers=[
            ("push_res", f"{cls}::push_res"),
            ("push_norm", f"{cls}::push_norm"),
            ("pull_res", f"{cls}::pull_res"),
            ("pull_norm", f"{cls}::pull_norm"),
            ("finalize_push_norm", f"{cls}::finalize_push_norm"),
        ],
        extra_cuda_cflags=["-O3"],
    )


_COMM_MAP: dict[int, Communicator] = {}


def register_comm(comm: Communicator) -> None:


    prev = _COMM_MAP.get(comm.world_size)
    assert prev is None or prev is comm, (
        f"a different communicator is already registered for world_size="
        f"{comm.world_size}; these ops key only on world_size, so two groups of "
        f"the same size cannot coexist in one process"
    )
    _COMM_MAP[comm.world_size] = comm


class PullTuning(NamedTuple):
    num_blocks: int
    unroll: int


class PullTuningTable(NamedTuple):
    bands: tuple[tuple[int, PullTuning], ...]
    fallback: PullTuning

    def lookup(self, nbytes: int) -> PullTuning:
        for max_bytes, tuning in self.bands:
            if nbytes <= max_bytes:
                return tuning
        return self.fallback


_KB, _MB = 1024, 1024 * 1024

RES_TUNING = PullTuningTable(
    bands=(
        (128 * _KB, PullTuning(num_blocks=1, unroll=8)),
        (4 * _MB, PullTuning(num_blocks=16, unroll=8)),
    ),
    fallback=PullTuning(num_blocks=4, unroll=8),
)

NORM_TUNING = PullTuningTable(
    bands=(
        (512 * _KB, PullTuning(num_blocks=2, unroll=4)),
        (2 * _MB, PullTuning(num_blocks=12, unroll=2)),
    ),
    fallback=PullTuning(num_blocks=24, unroll=2),
)


def _resolve_tuning(
    table: PullTuningTable,
    *,
    nbytes: int,
    num_blocks: Optional[int],
    unroll: Optional[int],
) -> PullTuning:

    tuned = table.lookup(nbytes)
    return PullTuning(
        num_blocks=num_blocks or tuned.num_blocks,
        unroll=unroll or tuned.unroll,
    )


@register_custom_op(mutates_args=["x"])
def _push_res_op(
    world_size: int,
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
) -> None:
    _jit_module(world_size).push_res(_COMM_MAP[world_size], x.view(-1), residual)


@register_custom_op(mutates_args=["x"])
def _push_norm_op(
    world_size: int,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    num_norm_rows: int,
) -> None:
    _jit_module(world_size).push_norm(
        _COMM_MAP[world_size], x.view(-1), weight, eps, num_norm_rows
    )


@register_custom_op(mutates_args=["out"])
def _finalize_push_norm_op(
    world_size: int,
    out: torch.Tensor,
    gemm2_out: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_weights: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> None:
    _jit_module(world_size).finalize_push_norm(
        _COMM_MAP[world_size],
        out.view(-1),
        gemm2_out,
        expanded_idx_to_permuted_idx,
        expert_weights,
        weight,
        eps,
    )


@register_custom_op(mutates_args=["x"])
def _pull_res_op(
    world_size: int,
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    input_mc_ptr: int,
    num_blocks: int,
    unroll: int,
) -> None:
    _jit_module(world_size).pull_res(
        _COMM_MAP[world_size],
        x.view(-1),
        residual,
        input_mc_ptr,
        num_blocks,
        unroll,
    )


@register_custom_op(mutates_args=["x"])
def _pull_norm_op(
    world_size: int,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    num_norm_rows: int,
    input_mc_ptr: int,
    num_blocks: int,
    unroll: int,
) -> None:
    _jit_module(world_size).pull_norm(
        _COMM_MAP[world_size],
        x.view(-1),
        weight,
        eps,
        num_norm_rows,
        input_mc_ptr,
        num_blocks,
        unroll,
    )


def all_reduce_push_res(
    world_size: int,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    residual_ = residual.view(-1) if residual is not None else None
    _push_res_op(world_size, x, residual_)
    return x


def all_reduce_push_norm(
    world_size: int,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    num_norm_rows: int,
) -> torch.Tensor:

    _push_norm_op(world_size, x, weight, eps, num_norm_rows)
    return x


def finalize_all_reduce_push_norm(
    world_size: int,
    out: torch.Tensor,
    gemm2_out: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_weights: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:

    _finalize_push_norm_op(
        world_size,
        out,
        gemm2_out,
        expanded_idx_to_permuted_idx,
        expert_weights,
        weight,
        eps,
    )
    return out


def all_reduce_pull_res(
    world_size: int,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    *,
    input_mc_ptr: int,
    num_blocks: Optional[int] = None,
    unroll: Optional[int] = None,
) -> torch.Tensor:

    tuning = _resolve_tuning(
        RES_TUNING,
        nbytes=x.numel() * x.element_size(),
        num_blocks=num_blocks,
        unroll=unroll,
    )
    residual_ = residual.view(-1) if residual is not None else None
    _pull_res_op(
        world_size, x, residual_, input_mc_ptr, tuning.num_blocks, tuning.unroll
    )
    return x


def all_reduce_pull_norm(
    world_size: int,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    num_norm_rows: int,
    input_mc_ptr: int,
    num_blocks: Optional[int] = None,
    unroll: Optional[int] = None,
) -> torch.Tensor:

    tuning = _resolve_tuning(
        NORM_TUNING,
        nbytes=x.nbytes,
        num_blocks=num_blocks,
        unroll=unroll,
    )
    _pull_norm_op(
        world_size,
        x,
        weight,
        eps,
        num_norm_rows,
        input_mc_ptr,
        tuning.num_blocks,
        tuning.unroll,
    )
    return x
