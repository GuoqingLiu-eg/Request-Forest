from typing import Optional, Tuple

import torch

from sglang.kernels.ops.attention.nsa_triton_decode.triton_mla_kernels_decode_optimized import (
    triton_sparse_attn_decode,
)


class _KVScopeAdapter:


    __slots__ = [
        "blocked_k",
        "blocked_k_quantized",
        "indices_in_kvcache",
        "topk_length",
    ]

    def __init__(
        self,
        k_cache: torch.Tensor,
        indices: torch.Tensor,
        topk_length: Optional[torch.Tensor],
    ):
        self.blocked_k_quantized = k_cache
        self.blocked_k = k_cache
        self.indices_in_kvcache = indices
        self.topk_length = topk_length


def triton_fp8_attention_fwd(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    head_dim_v: int,
    softmax_scale: float,
    indices: torch.Tensor,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    **_unused,
) -> Tuple[torch.Tensor, torch.Tensor]:

    kv_scope = _KVScopeAdapter(k_cache, indices, topk_length)

    extra_kv_scope = None
    if extra_k_cache is not None:
        extra_kv_scope = _KVScopeAdapter(
            extra_k_cache,
            extra_indices_in_kvcache,
            extra_topk_length,
        )

    output, lse = triton_sparse_attn_decode(
        q=q,
        kv_scope=kv_scope,
        extra_kv_scope=extra_kv_scope,
        sm_scale=softmax_scale,
        d_v=head_dim_v,
        attn_sink=attn_sink,
    )


    lse = lse.transpose(1, 2)

    return output, lse
