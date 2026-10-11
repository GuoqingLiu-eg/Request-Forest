from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    get_jit_cuda_arch,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

DEFAULT_BLOCK_DIM = 256

DISPATCH_BLOCK_DIM = 256
DEFAULT_NUM_ITERS = 5


def _sm_ver() -> int:
    arch = get_jit_cuda_arch()
    return arch.major * 100 + arch.minor * 10


@cache_once
def _ipm_module(
    nc: int, nv: int, block_dim: int, num_iters: int, sm_ver: int
) -> Module:

    args = make_cpp_args(nc, nv, block_dim, sm_ver, num_iters)


    return load_jit(
        "lplb_ipm",
        *args,
        cuda_files=["lplb/ipm.cuh"],
        cuda_wrappers=[("ipm_solve", f"ipm_solve<{args}>")],
        extra_dependencies=["mathdx"],
    )


def warmup(
    nc: int,
    nv: int,
    num_iters: int = DEFAULT_NUM_ITERS,
    device: str = "cuda",
) -> None:

    module = _ipm_module(nc, nv, DEFAULT_BLOCK_DIM, num_iters, _sm_ver())

    A = torch.zeros(nc, nv, dtype=torch.float32, device=device)
    b = torch.zeros(nc, dtype=torch.float32, device=device)
    c = torch.zeros(nv, dtype=torch.float32, device=device)
    result = torch.empty(nv, dtype=torch.float32, device=device)
    module.ipm_solve(A, b, c, result)
    logger.info(f"LPLB CUDA IPM solver: warmed up for (NC={nc}, NV={nv})")


def solve_ipm(
    A: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    num_iters: int = DEFAULT_NUM_ITERS,
    result: torch.Tensor | None = None,
) -> torch.Tensor:

    assert A.is_cuda and b.is_cuda and c.is_cuda
    assert A.dtype == torch.float32
    nc, nv = A.shape
    assert b.shape == (nc,), f"b shape mismatch: {b.shape} vs ({nc},)"
    assert c.shape == (nv,), f"c shape mismatch: {c.shape} vs ({nv},)"

    module = _ipm_module(nc, nv, DEFAULT_BLOCK_DIM, num_iters, _sm_ver())
    if result is None:
        result = torch.empty(nv, dtype=torch.float32, device=A.device)
    module.ipm_solve(A, b, c, result)
    return result


@cache_once
def _prep_module(
    nc: int,
    nv: int,
    num_single: int,
    num_red_log: int,
    num_gpus: int,
    block_dim: int,
) -> Module:
    args = make_cpp_args(nc, nv, num_single, num_red_log, num_gpus, block_dim)
    return load_jit(
        "lplb_lp_prep",
        *args,
        cuda_files=["lplb/lp_prep.cuh"],
        cuda_wrappers=[("lp_prep", f"lp_prep<{args}>")],
    )


def prep_lp_inputs(
    A_full: torch.Tensor,
    b: torch.Tensor,
    t1: torch.Tensor,
    global_counts: torch.Tensor,
    log_single: torch.Tensor,
    log_replicated: torch.Tensor,
    B1: torch.Tensor,
    A_base_row_sum: torch.Tensor,
) -> None:

    nc, nv = A_full.shape
    num_single = log_single.shape[0]
    num_red_log = log_replicated.shape[0]
    num_gpus, _ns = B1.shape
    module = _prep_module(nc, nv, num_single, num_red_log, num_gpus, DEFAULT_BLOCK_DIM)
    module.lp_prep(
        A_full, b, t1, global_counts, log_single, log_replicated, B1, A_base_row_sum
    )


@cache_once
def _post_module(
    num_logical: int,
    max_copies: int,
    num_single: int,
    num_red_phy: int,
    block_dim: int,
) -> Module:
    args = make_cpp_args(num_logical, max_copies, num_single, num_red_phy, block_dim)
    return load_jit(
        "lplb_lp_post",
        *args,
        cuda_files=["lplb/lp_post.cuh"],
        cuda_wrappers=[("lp_post", f"lp_post<{args}>")],
    )


def extract_log2phy_prob(
    log2phy_prob: torch.Tensor,
    x: torch.Tensor,
    t1: torch.Tensor,
    phy_single: torch.Tensor,
    phy_replicated: torch.Tensor,
    log2phy: torch.Tensor,
) -> None:

    num_logical, max_copies = log2phy_prob.shape
    num_single = phy_single.shape[0]
    num_red_phy = phy_replicated.shape[0]
    module = _post_module(
        num_logical, max_copies, num_single, num_red_phy, DEFAULT_BLOCK_DIM
    )
    module.lp_post(log2phy_prob, x, t1, phy_single, phy_replicated, log2phy)


@cache_once
def _dispatch_module(max_copies: int, block_dim: int) -> Module:
    args = make_cpp_args(max_copies, block_dim)
    return load_jit(
        "lplb_dispatch_probability",
        *args,
        cuda_files=["lplb/dispatch_probability.cuh"],
        cuda_wrappers=[("dispatch_probability", f"dispatch_probability<{args}>")],
    )


def dispatch_probability(
    topk_ids: torch.Tensor,
    log2phy_prob: torch.Tensor,
    log2phy_map: torch.Tensor,
    random_vals: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    original_shape = topk_ids.shape
    flat_ids = topk_ids.reshape(-1).contiguous().to(torch.int32)
    n = flat_ids.shape[0]
    num_logical, max_copies = log2phy_prob.shape
    assert log2phy_map.shape == (num_logical, max_copies)
    map32 = log2phy_map.contiguous().to(torch.int32)

    out = torch.empty(n, dtype=torch.int32, device=topk_ids.device)
    if random_vals is None:
        random_vals = torch.rand(n, dtype=torch.float32, device=topk_ids.device)
    else:
        assert random_vals.shape == (
            n,
        ), f"random_vals must be shape ({n},), got {tuple(random_vals.shape)}"
    module = _dispatch_module(max_copies, DISPATCH_BLOCK_DIM)
    module.dispatch_probability(out, flat_ids, log2phy_prob, map32, random_vals)
    return out.view(original_shape).to(topk_ids.dtype)


def dispatch_probability_torch_reference(
    topk_ids: torch.Tensor,
    log2phy_prob: torch.Tensor,
    log2phy_map: torch.Tensor,
    random_vals: torch.Tensor,
) -> torch.Tensor:

    original_shape = topk_ids.shape
    flat_ids = topk_ids.reshape(-1).long()
    n = flat_ids.shape[0]
    num_logical, max_copies = log2phy_prob.shape
    assert log2phy_map.shape == (num_logical, max_copies)
    assert random_vals.shape == (
        n,
    ), f"random_vals must be shape ({n},), got {tuple(random_vals.shape)}"


    probs = log2phy_prob[flat_ids]
    maps = log2phy_map[flat_ids]


    row_sum = probs.sum(dim=-1, keepdim=True)
    fallback_probs = (maps >= 0).to(probs.dtype)
    probs = torch.where(row_sum > 0, probs, fallback_probs)
    row_sum = probs.sum(dim=-1)


    u = (random_vals * row_sum).unsqueeze(-1)
    cum = probs.cumsum(dim=-1)
    chosen = (cum <= u).sum(dim=-1).clamp(max=max_copies - 1)

    out = maps.gather(1, chosen.unsqueeze(-1)).squeeze(-1)
    return out.view(original_shape).to(topk_ids.dtype)
