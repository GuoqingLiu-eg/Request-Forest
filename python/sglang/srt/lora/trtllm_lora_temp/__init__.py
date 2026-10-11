from typing import Callable, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.lora.trtllm_lora_temp.environ import lora_envs


def is_two_stream_active(x: torch.Tensor) -> bool:

    return x.shape[0] <= lora_envs.SGLANG_TWO_STREAM_MAX_TOKENS.get()


def supports_two_stream_dense_lora(lora_a: torch.Tensor, lora_b: torch.Tensor) -> bool:

    return lora_a.shape[-2] <= 128 and lora_b.shape[-1] <= 64


_LORA_SIDE_STREAMS: dict[torch.cuda.Stream, torch.cuda.Stream] = {}


def get_lora_side_stream() -> torch.cuda.Stream:


    consumer = torch.cuda.current_stream()
    if consumer not in _LORA_SIDE_STREAMS:
        _LORA_SIDE_STREAMS[consumer] = torch.cuda.Stream()
    return _LORA_SIDE_STREAMS[consumer]


def init_lora_two_stream_resources(device: Optional[torch.device] = None) -> None:

    if device is not None:
        with torch.cuda.device(device):
            get_lora_side_stream()
    else:
        get_lora_side_stream()


def lora_overlap_alloc_stream() -> Optional[torch.cuda.Stream]:

    if lora_envs.SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC.get():
        return torch.cuda.current_stream()
    return None


_ORIGINAL_QKV_FORWARD: Optional[Callable] = None
_ORIGINAL_ROW_FORWARD: Optional[Callable] = None
_ORIGINAL_MERGED_FORWARD: Optional[Callable] = None
_ORIGINAL_COLUMN_FORWARD: Optional[Callable] = None
_ORIGINAL_REPLICATED_FORWARD: Optional[Callable] = None
_ORIGINAL_MOE_LORA_FUNC: Optional[Callable] = None
_ORIGINAL_FP4_MOE_LORA_FUNC: Optional[Callable] = None
_ORIGINAL_BF16_MOE_LORA_FUNC: Optional[Callable] = None
_INSTALLED: bool = False


def get_original_qkv_forward() -> Callable:
    return _ORIGINAL_QKV_FORWARD


def get_original_row_forward() -> Callable:
    return _ORIGINAL_ROW_FORWARD


def get_original_merged_column_forward() -> Callable:
    return _ORIGINAL_MERGED_FORWARD


def get_original_column_forward() -> Callable:
    return _ORIGINAL_COLUMN_FORWARD


def get_original_replicated_forward() -> Callable:
    return _ORIGINAL_REPLICATED_FORWARD


def get_original_moe_lora_func() -> Callable:
    return _ORIGINAL_MOE_LORA_FUNC


def get_original_fp4_moe_lora_func() -> Callable:
    return _ORIGINAL_FP4_MOE_LORA_FUNC


def get_original_bf16_moe_lora_func() -> Callable:
    return _ORIGINAL_BF16_MOE_LORA_FUNC


def install_two_stream_overrides() -> None:

    global _INSTALLED, _ORIGINAL_QKV_FORWARD, _ORIGINAL_ROW_FORWARD, _ORIGINAL_MERGED_FORWARD, _ORIGINAL_COLUMN_FORWARD, _ORIGINAL_REPLICATED_FORWARD, _ORIGINAL_MOE_LORA_FUNC, _ORIGINAL_FP4_MOE_LORA_FUNC, _ORIGINAL_BF16_MOE_LORA_FUNC

    if _INSTALLED:
        return

    from sglang.srt.lora.layers import (
        ColumnParallelLinearWithLoRA,
        MergedColumnParallelLinearWithLoRA,
        QKVParallelLinearWithLoRA,
        ReplicatedLinearWithLoRA,
        RowParallelLinearWithLoRA,
    )
    from sglang.srt.lora.trtllm_lora_temp.attention import (
        column_parallel_lora_forward,
        qkv_proj_lora_forward,
        replicated_lora_forward,
        row_parallel_lora_forward,
    )
    from sglang.srt.lora.trtllm_lora_temp.merged_column import (
        merged_column_lora_forward,
    )


    _ORIGINAL_QKV_FORWARD = QKVParallelLinearWithLoRA.forward
    _ORIGINAL_ROW_FORWARD = RowParallelLinearWithLoRA.forward
    _ORIGINAL_MERGED_FORWARD = MergedColumnParallelLinearWithLoRA.forward
    _ORIGINAL_COLUMN_FORWARD = ColumnParallelLinearWithLoRA.forward
    _ORIGINAL_REPLICATED_FORWARD = ReplicatedLinearWithLoRA.forward
    QKVParallelLinearWithLoRA.forward = qkv_proj_lora_forward
    RowParallelLinearWithLoRA.forward = row_parallel_lora_forward
    MergedColumnParallelLinearWithLoRA.forward = merged_column_lora_forward

    ColumnParallelLinearWithLoRA.forward = column_parallel_lora_forward
    ReplicatedLinearWithLoRA.forward = replicated_lora_forward

    import sglang.srt.lora.trtllm_lora_temp.lora_dispatch as ft
    from sglang.srt.lora.trtllm_lora_temp.moe_overlap import (
        fused_experts_none_to_experimental_sgl_trtllm_bf16_lora_two_stream,
        fused_experts_none_to_experimental_sgl_trtllm_fp4_lora_two_stream,
        fused_experts_none_to_experimental_sgl_trtllm_fp8_lora_two_stream,
    )


    _ORIGINAL_MOE_LORA_FUNC = ft.fused_experts_none_to_experimental_sgl_trtllm_fp8_lora
    _ORIGINAL_FP4_MOE_LORA_FUNC = (
        ft.fused_experts_none_to_experimental_sgl_trtllm_fp4_lora
    )
    _ORIGINAL_BF16_MOE_LORA_FUNC = (
        ft.fused_experts_none_to_experimental_sgl_trtllm_bf16_lora
    )
    ft.fused_experts_none_to_experimental_sgl_trtllm_fp8_lora = (
        fused_experts_none_to_experimental_sgl_trtllm_fp8_lora_two_stream
    )
    ft.fused_experts_none_to_experimental_sgl_trtllm_fp4_lora = (
        fused_experts_none_to_experimental_sgl_trtllm_fp4_lora_two_stream
    )
    ft.fused_experts_none_to_experimental_sgl_trtllm_bf16_lora = (
        fused_experts_none_to_experimental_sgl_trtllm_bf16_lora_two_stream
    )

    _INSTALLED = True


__all__ = [
    "is_two_stream_active",
    "supports_two_stream_dense_lora",
    "get_lora_side_stream",
    "init_lora_two_stream_resources",
    "get_original_qkv_forward",
    "get_original_row_forward",
    "get_original_merged_column_forward",
    "get_original_column_forward",
    "get_original_replicated_forward",
    "get_original_moe_lora_func",
    "get_original_fp4_moe_lora_func",
    "get_original_bf16_moe_lora_func",
    "install_two_stream_overrides",
]
