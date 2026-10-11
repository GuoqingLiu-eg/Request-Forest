from typing import TYPE_CHECKING, Callable

import torch

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.musa.attention.flashattention_backend import (
        MusaFlashAttentionBackend,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def musa_cp_attn_forward_extend(
    musa_fa_backend: "MusaFlashAttentionBackend",
    forward_batch: "ForwardBatch",
    q: torch.Tensor,
    device: torch.device,
    attn_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor],
) -> torch.Tensor:

    cp_meta = forward_batch.attn_cp_metadata

    q_prev, q_next = torch.chunk(q, 2, dim=0)

    cu_seqlens_q_prev = torch.tensor(
        [0, cp_meta.actual_seq_q_prev], device=device, dtype=torch.int32
    )
    if hasattr(musa_fa_backend, "_current_prefix"):
        musa_fa_backend._current_prefix = "forward_extend_cp_prev"
    result_prev = attn_fn(
        q_prev,
        cu_seqlens_q_prev,
        cp_meta.kv_len_prev_tensor,
        cp_meta.actual_seq_q_prev,
    )

    cu_seqlens_q_next = torch.tensor(
        [0, cp_meta.actual_seq_q_next], device=device, dtype=torch.int32
    )
    if hasattr(musa_fa_backend, "_current_prefix"):
        musa_fa_backend._current_prefix = "forward_extend_cp_next"
    result_next = attn_fn(
        q_next,
        cu_seqlens_q_next,
        cp_meta.kv_len_next_tensor,
        cp_meta.actual_seq_q_next,
    )

    return torch.concat([result_prev, result_next], dim=0)
