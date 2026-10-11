from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    empty_sentinel,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_rel_proj_module(d_rel: int, use_pdl: bool) -> Module:
    args = make_cpp_args(d_rel, use_pdl)
    return load_jit(
        "inkling_rel_proj",
        *args,
        cuda_files=["inkling/inkling_rel_proj.cuh"],
        cuda_wrappers=[("run", f"rel_proj_small_t<{args}>")],
    )


def rel_proj_small_t(
    r: torch.Tensor,
    proj: torch.Tensor,
    tau: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:

    if out is None:
        out = torch.empty(
            (r.shape[0], r.shape[1], proj.shape[1]), dtype=r.dtype, device=r.device
        )
    module = _jit_rel_proj_module(r.shape[2], is_arch_support_pdl())
    sh = tau if tau is not None else empty_sentinel(r.device, torch.float32)
    module.run(r, sh, proj, out)
    return out
