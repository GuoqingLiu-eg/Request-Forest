from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.kernels.ops.gemm.kv_b_lora_absorbed import (
    step_a_q_fwd,
    step_a_v_fwd,
    step_b_q_fwd,
    step_b_v_fwd,
)

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
    batch_info = lora_backend.batch_info
    if batch_info is None:
        return None


    sgemm_info = getattr(lora_backend, "_sgemm_info", None)
    if callable(sgemm_info):
        batch_info = sgemm_info()
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
