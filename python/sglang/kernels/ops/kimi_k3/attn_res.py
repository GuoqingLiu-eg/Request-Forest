from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    load_jit,
    make_cpp_args,
    override_jit_cuda_arch,
)
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

    from sglang.kernels.ops.communication.all_reduce import Communicator

_DIM: int = 7168
_MAX_BANK_ROWS: int = 8


def _make_name(*args):
    return "kimi_k3_attn_res_" + "_".join(str(a) for a in args)


@cache_once
def _jit_fused_tma_module(
    chunk_rows: int, occupancy: int, consumer_regs: int
) -> Module:

    major, minor = torch.cuda.get_device_capability()
    if major < 10 or major == 12:
        raise RuntimeError(
            "attn_res_fused_tma requires SM100+ excluding SM12x; "
            f"SM{major}{minor} is unsupported"
        )
    args = make_cpp_args(
        _DIM,
        _MAX_BANK_ROWS,
        chunk_rows,
        occupancy,
        consumer_regs,
    )
    with override_jit_cuda_arch(major, minor, suffix="a"):
        return load_jit(
            _make_name("fused_tma"),
            *args,
            cuda_files=["kimi_k3/attn_res/fused_tma.cuh"],
            cuda_wrappers=[
                ("run", f"AttnResFusedTmaKernel<{args}>::run"),
                ("run_pull_rs", f"AttnResFusedTmaKernel<{args}>::run_pull_rs"),
                ("run_direct_ag", f"AttnResFusedTmaKernel<{args}>::run_direct_ag"),
            ],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )


_TMA_BEST_CONFIG: dict[int, tuple[int, int, int]] = {
    1: (2, 2, 0),
    2: (4, 1, 200),
    3: (4, 1, 200),
    4: (5, 1, 200),
    5: (3, 1, 200),
    6: (4, 1, 200),
    7: (4, 1, 200),
    8: (5, 1, 200),
}


def _tuning(nvb: int, num_tokens: int) -> tuple[int, int, int]:

    if not 1 <= nvb <= _MAX_BANK_ROWS:
        raise ValueError(f"attn_res: nvb must be in [1, {_MAX_BANK_ROWS}], got {nvb}")
    best = _TMA_BEST_CONFIG[nvb]
    if best[1] > 1 and num_tokens < 128:


        best = (4, 1, 200)
    return best


_COMM_MAP: dict[int, Communicator] = {}


def register_comm(comm: Communicator) -> None:


    prev = _COMM_MAP.get(comm.world_size)
    assert prev is None or prev is comm, (
        f"a different communicator is already registered for world_size="
        f"{comm.world_size}"
    )
    _COMM_MAP[comm.world_size] = comm


@register_custom_op(mutates_args=["out", "prefix_out"])
def _attn_res_fused_pull_rs_op(
    world_size: int,
    input: torch.Tensor,
    residual: torch.Tensor | None,
    bank: torch.Tensor,
    cw: torch.Tensor,
    ow: torch.Tensor,
    out: torch.Tensor,
    prefix_out: torch.Tensor,
    nvb: int,
    eps: float,
    input_mc_ptr: int,
    max_blocks: int,
    chunk_rows: int,
    occupancy: int,
    consumer_regs: int,
) -> None:
    _jit_fused_tma_module(chunk_rows, occupancy, consumer_regs).run_pull_rs(
        _COMM_MAP[world_size],
        input,
        residual,
        bank,
        cw,
        ow,
        out,
        prefix_out,
        nvb,
        eps,
        input_mc_ptr,
        max_blocks,
    )


def attn_res_fused_tma(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    cw: torch.Tensor,
    ow: torch.Tensor,
    out: torch.Tensor,
    nvb: int,
    eps: float,
    *,
    write_prefix: bool = False,
) -> None:

    _jit_fused_tma_module(*_tuning(nvb, prefix_sum.shape[0])).run(
        prefix_sum, bank, cw, ow, out, nvb, eps, write_prefix
    )


@register_custom_op(mutates_args=["bank", "out"])
def _attn_res_fused_direct_ag_op(
    world_size: int,
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    cw: torch.Tensor,
    ow: torch.Tensor,
    out: torch.Tensor,
    nvb: int,
    eps: float,
    output_mc_ptr: int,
    max_blocks: int,
    write_prefix: bool,
    chunk_rows: int,
    occupancy: int,
    consumer_regs: int,
) -> None:
    _jit_fused_tma_module(chunk_rows, occupancy, consumer_regs).run_direct_ag(
        _COMM_MAP[world_size],
        prefix_sum,
        bank,
        cw,
        ow,
        out,
        nvb,
        eps,
        output_mc_ptr,
        max_blocks,
        write_prefix,
    )


def attn_res_fused_direct_ag(
    world_size: int,
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    cw: torch.Tensor,
    ow: torch.Tensor,
    out: torch.Tensor,
    nvb: int,
    eps: float,
    *,
    output_mc_ptr: int,
    max_blocks: int = 128,
    write_prefix: bool = False,
) -> torch.Tensor:

    _attn_res_fused_direct_ag_op(
        world_size,
        prefix_sum,
        bank,
        cw,
        ow,
        out,
        nvb,
        eps,
        output_mc_ptr,
        max_blocks,
        write_prefix,
        *_tuning(nvb, prefix_sum.shape[0]),
    )
    return out


def attn_res_fused_pull_rs(
    world_size: int,
    input: torch.Tensor,
    residual: torch.Tensor | None,
    bank: torch.Tensor,
    cw: torch.Tensor,
    ow: torch.Tensor,
    out: torch.Tensor,
    prefix_out: torch.Tensor,
    nvb: int,
    eps: float,
    *,
    input_mc_ptr: int,
    max_blocks: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:

    _attn_res_fused_pull_rs_op(
        world_size,
        input,
        residual,
        bank,
        cw,
        ow,
        out,
        prefix_out,
        nvb,
        eps,
        input_mc_ptr,
        max_blocks,

        *_tuning(nvb, prefix_out.shape[0]),
    )
    return out, prefix_out
