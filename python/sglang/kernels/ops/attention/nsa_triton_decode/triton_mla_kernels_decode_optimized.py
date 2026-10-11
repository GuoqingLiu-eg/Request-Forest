from typing import Optional, Tuple

import torch

from .triton_mla_kernels_decode_fused import (
    DSV4_D_QK,
    fused_gather_attn_decode_dsv4,
    fused_gather_attn_decode_dsv4_dual_scope,
    fused_gather_attn_decode_dsv4_dual_scope_low_overhead,
)


def _should_use_fused_splitk(total_tokens: int, h_q: int, total_topk: int) -> bool:

    if total_tokens <= 4:
        return True
    if h_q <= 64 and total_topk <= 800:
        return total_tokens <= 256
    if h_q <= 64 and total_topk >= 1024:
        return total_tokens <= 128

    if h_q > 64:
        if total_topk >= 400:
            return total_tokens <= 32
        else:
            return total_tokens <= 128
    return True


def triton_sparse_attn_decode(
    q: torch.Tensor,
    kv_scope,
    extra_kv_scope,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    d_qk = q.shape[-1]

    if d_qk != DSV4_D_QK:
        raise ValueError(
            f"Unsupported d_qk: {d_qk}. Expected {DSV4_D_QK} (DeepSeek V4)"
        )

    return _triton_sparse_attn_decode_dsv4(
        q, kv_scope, extra_kv_scope, sm_scale, d_v, attn_sink
    )


def _triton_sparse_attn_decode_dsv4(
    q: torch.Tensor,
    kv_scope,
    extra_kv_scope,
    sm_scale: float,
    d_v: int,
    attn_sink: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:

    b, s_q, h_q, d_qk = q.shape
    total_tokens = b * s_q

    topk_main = kv_scope.indices_in_kvcache.shape[-1]
    kv_quantized_main = kv_scope.blocked_k_quantized
    block_size_main = kv_scope.blocked_k.shape[1]


    if extra_kv_scope is None:
        q_reshaped = q.reshape(total_tokens, h_q, d_qk).contiguous()
        indices_main = kv_scope.indices_in_kvcache.reshape(
            total_tokens, topk_main
        ).contiguous()

        output, lse = fused_gather_attn_decode_dsv4(
            q_reshaped,
            kv_quantized_main,
            indices_main,
            block_size_main,
            sm_scale,
            topk_length=kv_scope.topk_length,
            attn_sink=attn_sink,
            s_q=s_q,
        )
        return output.view(b, s_q, h_q, d_v), lse.view(b, s_q, h_q).transpose(1, 2)


    topk_extra = extra_kv_scope.indices_in_kvcache.shape[-1]
    total_topk = topk_main + topk_extra
    block_size_extra = extra_kv_scope.blocked_k.shape[1]

    q_reshaped = q.reshape(total_tokens, h_q, d_qk).contiguous()
    indices_main = kv_scope.indices_in_kvcache.reshape(
        total_tokens, topk_main
    ).contiguous()
    indices_extra = extra_kv_scope.indices_in_kvcache.reshape(
        total_tokens, topk_extra
    ).contiguous()


    if _should_use_fused_splitk(total_tokens, h_q, total_topk):

        output, lse = fused_gather_attn_decode_dsv4_dual_scope_low_overhead(
            q_reshaped,
            kv_quantized_main,
            indices_main,
            block_size_main,
            extra_kv_scope.blocked_k_quantized,
            indices_extra,
            block_size_extra,
            sm_scale,
            topk_length_main=kv_scope.topk_length,
            topk_length_extra=extra_kv_scope.topk_length,
            attn_sink=attn_sink,
            s_q=s_q,
        )
    else:

        output, lse = fused_gather_attn_decode_dsv4_dual_scope(
            q_reshaped,
            kv_quantized_main,
            indices_main,
            block_size_main,
            extra_kv_scope.blocked_k_quantized,
            indices_extra,
            block_size_extra,
            sm_scale,
            topk_length_main=kv_scope.topk_length,
            topk_length_extra=extra_kv_scope.topk_length,
            attn_sink=attn_sink,
            s_q=s_q,
            force_no_splitk=True,
        )

    return output.view(b, s_q, h_q, d_v), lse.view(b, s_q, h_q).transpose(1, 2)
