from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module


N_LORA = 512
ROPE_DIM = 64


@cache_once
def _jit_qprep_bf16_fp8_module() -> Module:
    if torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("qprep_bf16_fp8_sm90 requires an SM90 (Hopper) GPU")
    return load_jit(
        "qprep_bf16_fp8_sm90",
        cuda_files=["qprep_bf16_fp8_sm90/entry.cuh"],
        cuda_wrappers=[("dispatch", "qprep_bf16_fp8_dispatch")],


        extra_cuda_cflags=[
            "-O3",
            "-DNDEBUG",
            "-DCUTE_USE_PACKED_TUPLE=1",
            "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
            "--use_fast_math",
        ],
        extra_dependencies=["cutlass"],
    )


_get_current_stream_raw = torch._C._cuda_getCurrentRawStream


@debug_kernel_api
def q8kv8_qprep_fwd(
    q_fp8_pad: torch.Tensor,
    q_nope: torch.Tensor,
    w_kc: torch.Tensor,
    q_rope: torch.Tensor,
    num_heads: int,
) -> None:

    num_tokens, _, k_dim = q_nope.shape
    n_dim = w_kc.shape[-1]
    rope_dim = q_rope.shape[-1]
    assert q_fp8_pad.dtype == torch.float8_e4m3fn
    assert q_nope.dtype == torch.bfloat16 and w_kc.dtype == torch.bfloat16
    assert q_rope.dtype == torch.bfloat16
    assert q_nope.is_cuda and w_kc.is_cuda and q_rope.is_cuda and q_fp8_pad.is_cuda
    assert q_nope.shape[1] == num_heads and q_rope.shape[1] == num_heads
    assert w_kc.shape[0] == num_heads and w_kc.shape[1] == k_dim
    assert q_fp8_pad.shape[0] >= num_tokens and q_fp8_pad.shape[1] >= num_heads
    assert q_fp8_pad.shape[2] == n_dim + rope_dim
    assert k_dim in (128, 192), "CUDA q-prep supports K in {128, 192}"
    assert n_dim == N_LORA and rope_dim == ROPE_DIM

    assert q_nope.stride(2) == 1 and q_rope.stride(2) == 1
    assert q_fp8_pad.stride(2) == 1


    assert w_kc.stride(1) == 1, "w_kc must have K contiguous (N-major layout)"
    assert q_nope.data_ptr() % 16 == 0 and w_kc.data_ptr() % 16 == 0
    assert q_nope.stride(0) % 8 == 0 and q_nope.stride(1) % 8 == 0
    assert w_kc.stride(0) % 8 == 0 and w_kc.stride(2) % 8 == 0
    assert q_fp8_pad.stride(0) % 2 == 0 and q_fp8_pad.stride(1) % 2 == 0

    rope_vec16 = (
        q_rope.data_ptr() % 16 == 0
        and q_rope.stride(0) % 8 == 0
        and q_rope.stride(1) % 8 == 0
    )
    out_vec16 = (
        q_fp8_pad.data_ptr() % 16 == 0
        and q_fp8_pad.stride(0) % 16 == 0
        and q_fp8_pad.stride(1) % 16 == 0
    )

    module = _jit_qprep_bf16_fp8_module()
    module.dispatch(
        q_nope,
        w_kc,
        q_rope,
        q_fp8_pad,
        num_tokens,
        num_heads,
        k_dim,
        q_nope.stride(0),
        q_nope.stride(1),
        w_kc.stride(0),
        w_kc.stride(2),
        q_rope.stride(0),
        q_rope.stride(1),
        q_fp8_pad.stride(0),
        q_fp8_pad.stride(1),
        int(rope_vec16),
        int(out_vec16),
        _get_current_stream_raw(q_nope.device.index),
    )
