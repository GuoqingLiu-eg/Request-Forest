from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch


step_a_q_fwd = step_a_v_fwd = step_b_q_fwd = step_b_v_fwd = None


def _ensure_step_kernels() -> None:
    global step_a_q_fwd, step_a_v_fwd, step_b_q_fwd, step_b_v_fwd
    if step_a_q_fwd is None:
        from sglang.kernels.ops.gemm.trtllm_lora_temp.kv_b_lora_absorbed import (
            step_a_q_fwd as _aq,
        )
        from sglang.kernels.ops.gemm.trtllm_lora_temp.kv_b_lora_absorbed import (
            step_a_v_fwd as _av,
        )
        from sglang.kernels.ops.gemm.trtllm_lora_temp.kv_b_lora_absorbed import (
            step_b_q_fwd as _bq,
        )
        from sglang.kernels.ops.gemm.trtllm_lora_temp.kv_b_lora_absorbed import (
            step_b_v_fwd as _bv,
        )

        step_a_q_fwd, step_a_v_fwd, step_b_q_fwd, step_b_v_fwd = _aq, _av, _bq, _bv


if TYPE_CHECKING:
    from sglang.srt.lora.utils import LoRABatchInfo
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA


def is_kv_b_lora_active(attn_module: DeepseekV2AttentionMLA) -> bool:

    kv_b_proj = getattr(attn_module, "kv_b_proj", None)
    return getattr(kv_b_proj, "set_lora", False)


def _get_state(
    attn_module: DeepseekV2AttentionMLA,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, LoRABatchInfo]]:
    if not is_kv_b_lora_active(attn_module):
        return None
    if not hasattr(attn_module.kv_b_proj, "A_buffer"):
        return None
    lora_backend = attn_module.kv_b_proj.lora_backend
    if not hasattr(lora_backend, "batch_info"):
        return None
    batch_info = lora_backend.batch_info
    if batch_info is None:
        return None


    sgemm_info = getattr(lora_backend, "_sgemm_info", None)
    if callable(sgemm_info):
        batch_info = sgemm_info()


    _ensure_step_kernels()
    return attn_module.kv_b_proj.A_buffer, attn_module.kv_b_proj.B_buffer, batch_info


def apply_q_correction(
    attn_module: DeepseekV2AttentionMLA,
    q_nope: torch.Tensor,
    q_nope_out: torch.Tensor,
) -> torch.Tensor:

    state = _get_state(attn_module)
    if state is None:
        return q_nope_out
    A_buf, B_buf, batch_info = state

    full_K_per_head = attn_module.qk_nope_head_dim + attn_module.v_head_dim
    q_lora_a = step_a_q_fwd(q_nope, B_buf, batch_info, full_K_per_head)
    return step_b_q_fwd(q_lora_a, A_buf, batch_info, q_nope_out)


def apply_v_correction(
    attn_module: DeepseekV2AttentionMLA,
    attn_output: torch.Tensor,
    attn_bmm_flat: torch.Tensor,
) -> torch.Tensor:

    state = _get_state(attn_module)
    if state is None:
        return attn_bmm_flat
    A_buf, B_buf, batch_info = state

    attn_lora_a = step_a_v_fwd(attn_output, A_buf, batch_info)
    base_view = attn_bmm_flat.view(
        -1, attn_module.num_local_heads, attn_module.v_head_dim
    )
    step_b_v_fwd(
        attn_lora_a,
        B_buf,
        batch_info,
        base_view,
        attn_module.qk_nope_head_dim,
        attn_module.v_head_dim,
    )
    return attn_bmm_flat


def _kv_b_two_stream_state(attn_module, x):
    from sglang.srt.lora.trtllm_lora_temp import (
        get_lora_side_stream,
        is_two_stream_active,
    )

    if not is_two_stream_active(x):
        return None
    state = _get_state(attn_module)
    if state is None:
        return None
    A_buf, B_buf, batch_info = state
    return A_buf, B_buf, batch_info, get_lora_side_stream()


def kv_b_lora_q_prepare(attn_module, q_nope):

    st = _kv_b_two_stream_state(attn_module, q_nope)
    if st is None:
        return None
    A_buf, B_buf, batch_info, side_stream = st
    full_K_per_head = attn_module.qk_nope_head_dim + attn_module.v_head_dim
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        q_lora_a = step_a_q_fwd(q_nope, B_buf, batch_info, full_K_per_head)
    return q_lora_a, A_buf, batch_info, side_stream


def kv_b_lora_q_apply(attn_module, q_nope, q_nope_out, handle):

    if handle is not None:
        q_lora_a, A_buf, batch_info, side_stream = handle
        torch.cuda.current_stream().wait_stream(side_stream)
        return step_b_q_fwd(q_lora_a, A_buf, batch_info, q_nope_out)
    if is_kv_b_lora_active(attn_module):
        return apply_q_correction(attn_module, q_nope, q_nope_out)
    return q_nope_out


def kv_b_lora_v_prepare(attn_module, attn_output):

    st = _kv_b_two_stream_state(attn_module, attn_output)
    if st is None:
        return None
    A_buf, B_buf, batch_info, side_stream = st
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        attn_lora_a = step_a_v_fwd(attn_output, A_buf, batch_info)
    return attn_lora_a, B_buf, batch_info, side_stream


def kv_b_lora_v_apply(attn_module, attn_output, attn_bmm_flat, handle):

    if handle is not None:
        attn_lora_a, B_buf, batch_info, side_stream = handle
        torch.cuda.current_stream().wait_stream(side_stream)
        base_view = attn_bmm_flat.view(
            -1, attn_module.num_local_heads, attn_module.v_head_dim
        )
        step_b_v_fwd(
            attn_lora_a,
            B_buf,
            batch_info,
            base_view,
            attn_module.qk_nope_head_dim,
            attn_module.v_head_dim,
        )
        return attn_bmm_flat
    if is_kv_b_lora_active(attn_module):
        return apply_v_correction(attn_module, attn_output, attn_bmm_flat)
    return attn_bmm_flat
