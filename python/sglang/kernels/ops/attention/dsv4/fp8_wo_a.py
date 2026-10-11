from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.srt.utils.custom_op import register_custom_op

from .utils import make_name

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_GROUP_SIZE = 128
_UE8M0_SCALES_PER_PACK = 4


@cache_once
def _jit_module(in_dtype: torch.dtype, use_pdl: bool) -> Module:
    args = make_cpp_args(in_dtype, use_pdl)
    return load_jit(
        make_name("fp8_wo_a_group_major_quant_ue8m0"),
        *args,
        cuda_files=["deepseek_v4/fp8_wo_a_group_major_quant.cuh"],
        cuda_wrappers=[
            (
                "fp8_wo_a_group_major_quant_ue8m0",
                f"FP8WoAGroupMajorQuantUE8M0Kernel<{args}>::run",
            )
        ],


        extra_cuda_cflags=["--use_fast_math"],
    )


@register_custom_op(
    op_name="fp8_wo_a_group_major_quant_ue8m0",
    mutates_args=["output_q", "output_s"],
)
def _fp8_wo_a_group_major_quant_ue8m0_custom_op(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
) -> None:

    assert input.dtype in (torch.bfloat16, torch.float16)

    module = _jit_module(input.dtype, is_arch_support_pdl())
    module.fp8_wo_a_group_major_quant_ue8m0(input, output_q, output_s)


@debug_kernel_api
def fp8_wo_a_group_major_quant_ue8m0(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
) -> None:
    _fp8_wo_a_group_major_quant_ue8m0_custom_op(input, output_q, output_s)


def sglang_per_token_group_quant_fp8_dsv4_wo_a(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    num_tokens, num_groups, hidden = x.shape
    hidden_groups = hidden // _GROUP_SIZE
    packed_hidden_groups = (
        hidden_groups + _UE8M0_SCALES_PER_PACK - 1
    ) // _UE8M0_SCALES_PER_PACK
    aligned_num_tokens = (
        (num_tokens + _UE8M0_SCALES_PER_PACK - 1)
        // _UE8M0_SCALES_PER_PACK
        * _UE8M0_SCALES_PER_PACK
    )
    x_q = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    x_s_storage = torch.empty(
        (num_groups, packed_hidden_groups, aligned_num_tokens),
        device=x.device,
        dtype=torch.int32,
    )

    if x.numel() > 0:
        fp8_wo_a_group_major_quant_ue8m0(x, x_q, x_s_storage)


    x_s = x_s_storage.transpose(-1, -2)[:, :num_tokens, :].transpose(0, 1)
    return x_q, x_s
