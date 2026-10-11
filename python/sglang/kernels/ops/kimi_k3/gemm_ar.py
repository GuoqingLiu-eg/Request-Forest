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

N = 7168
MAX_TOKENS = 512


@cache_once
def _jit_module(k: int, world_size: int) -> Module:
    args = make_cpp_args(
        k,
        world_size,
        is_arch_support_pdl(),
    )
    cls = f"GemmArKernel<{args}>"
    return load_jit(
        "kimi_k3_gemm_ar",
        *args,
        cuda_files=["kimi_k3/comm/gemm_ar.cuh"],
        cuda_wrappers=[
            ("run", f"{cls}::run"),
            ("set_bases", f"{cls}::set_bases"),
            ("region_nbytes", f"{cls}::region_nbytes"),
            ("gather_words", f"{cls}::gather_words"),
            ("num_fams", f"{cls}::num_fams"),
        ],
        extra_cuda_cflags=["-O3"],
        extra_dependencies=["cutlass"],
    )


class _State(NamedTuple):
    world_size: int
    rank: int
    region: tuple
    uc_bases: torch.Tensor
    gather: torch.Tensor
    epochs: torch.Tensor


_STATE: Optional[_State] = None


def init(
    *,
    world_size: int,
    rank: int,
    group: torch.distributed.ProcessGroup,
    k: int,
) -> None:

    global _STATE
    if _STATE is not None:
        return


    from torch._C._distributed_c10d import _SymmetricMemory

    mod = _jit_module(k, world_size)
    nbytes = int(mod.region_nbytes())
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.__version__ < "2.11.0":
        import torch.distributed._symmetric_memory as torch_symm_mem

        torch_symm_mem.enable_symm_mem_for_group(group.group_name)
    region = _SymmetricMemory.empty_strided_p2p(
        (nbytes,), [1], torch.uint8, device, group.group_name
    )
    symm = _SymmetricMemory.rendezvous(region)
    region.zero_()
    torch.cuda.synchronize()
    torch.distributed.barrier(group=group)

    peer_bufs = [symm.get_buffer(r, [nbytes], torch.uint8) for r in range(world_size)]
    ptrs = [t.data_ptr() for t in peer_bufs]
    import logging

    logging.getLogger(__name__).info(
        "gemm_ar comm region: rank=%d nbytes=%d uc_bases=%s",
        rank,
        nbytes,
        [hex(p) for p in ptrs],
    )
    assert all(p != 0 for p in ptrs), f"gemm_ar: null peer pointers {ptrs}"


    uc_bases = torch.tensor(ptrs, dtype=torch.int64, device="cpu")
    gather = torch.zeros(int(mod.gather_words()), dtype=torch.int32, device=device)
    epochs = torch.zeros(int(mod.num_fams()), dtype=torch.int32, device=device)
    torch.cuda.synchronize()
    _STATE = _State(
        world_size=world_size,
        rank=rank,
        region=(region, peer_bufs),
        uc_bases=uc_bases,
        gather=gather,
        epochs=epochs,
    )


def initialized() -> bool:
    return _STATE is not None


@cache_once
def _module_with_bases(k: int, world_size: int) -> Module:

    state = _STATE
    assert state is not None
    mod = _jit_module(k, world_size)
    mod.set_bases(state.uc_bases)
    return mod


def fits(x: torch.Tensor) -> bool:

    return (
        _STATE is not None
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and 0 < x.shape[0] <= MAX_TOKENS
        and x.stride(1) == 1
        and x.stride(0) == x.shape[1]
    )


@register_custom_op(mutates_args=["out", "epochs"])
def _gemm_ar_op(
    k: int,
    world_size: int,
    out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    gather: torch.Tensor,
    epochs: torch.Tensor,
    my_rank: int,
) -> None:
    _module_with_bases(k, world_size).run(out, x, weight, gather, epochs, my_rank)


def _cell_of(m: int) -> int:
    for c in (8, 16, 32, 64, 128, 256, 512):
        if m <= c:
            return c
    raise ValueError(f"gemm_ar: M={m} outside [1, {MAX_TOKENS}]")


def o_proj_gemm_ar(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:

    state = _STATE
    assert state is not None
    m = x.shape[0]
    cell = _cell_of(m)
    out = torch.empty((cell, N), dtype=torch.bfloat16, device=x.device)
    _gemm_ar_op(
        weight.shape[1],
        state.world_size,
        out,
        x,
        weight,
        state.gather,
        state.epochs,
        state.rank,
    )
    return out[:m]
