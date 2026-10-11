# Copyright 2025 XunhaoLai. All rights reserved.

from typing import Optional

import torch


def naive_flash_decode_with_gqa_share_sparse(
    q: torch.Tensor,
    sink: Optional[torch.Tensor],
    kv_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    slot_ids: torch.Tensor,
    block_size: int,
    topk_idx: torch.Tensor,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:

    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5

    original_dtype = q.dtype
    batch_size, num_q_heads, head_dim = q.shape
    max_slots, _, max_kv_len, num_kv_heads, _ = kv_cache.shape
    gqa_group_size = num_q_heads // num_kv_heads

    o = torch.zeros(batch_size, num_q_heads, head_dim, dtype=q.dtype, device=q.device)

    for b in range(batch_size):
        seq_len = seq_lens[b].item()
        sid = slot_ids[b].item() % max_slots

        for kh in range(num_kv_heads):

            block_indices = topk_idx[kh, b, :].tolist()


            selected_k_blocks = []
            selected_v_blocks = []
            for block_idx in block_indices:
                if block_idx < 0:
                    continue
                start = block_idx * block_size
                end = min(start + block_size, seq_len)
                if start >= seq_len:
                    continue

                k_block = kv_cache[sid, 0, start:end, kh, :]

                v_block = kv_cache[sid, 1, start:end, kh, :]
                selected_k_blocks.append(k_block)
                selected_v_blocks.append(v_block)

            if len(selected_k_blocks) == 0:
                continue


            k_selected = torch.cat(selected_k_blocks, dim=0)
            v_selected = torch.cat(selected_v_blocks, dim=0)


            for g in range(gqa_group_size):
                qh = kh * gqa_group_size + g

                q_vec = q[b, qh, :].float()


                scores = torch.matmul(q_vec, k_selected.float().T) * sm_scale


                if sink is not None:
                    sink_vec = sink[qh, :].float()
                    qsink = torch.dot(q_vec, sink_vec) * sm_scale

                    scores_with_sink = torch.cat([qsink.unsqueeze(0), scores], dim=0)
                    attn_weights_with_sink = torch.softmax(scores_with_sink, dim=-1)

                    attn_weights = attn_weights_with_sink[1:]
                else:
                    attn_weights = torch.softmax(scores, dim=-1)


                o[b, qh, :] = torch.matmul(attn_weights.to(original_dtype), v_selected)

    return o
