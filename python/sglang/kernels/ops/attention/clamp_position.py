from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_clamp_position_module(dtype: torch.dtype) -> Module:

    args = make_cpp_args(dtype)
    return load_jit(
        "clamp_position",
        *args,
        cuda_files=["elementwise/clamp_position.cuh"],
        cuda_wrappers=[
            ("clamp_position", f"ClampPosition<{args}>::run"),
        ],
    )


def clamp_position_cuda(seq_lens: torch.Tensor) -> torch.Tensor:

    dst = torch.empty_like(seq_lens)
    module = _jit_clamp_position_module(seq_lens.dtype)
    module.clamp_position(dst, seq_lens)
    return dst
