from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class MarginalProposal:
    candidate_ids: torch.Tensor
    probabilities: torch.Tensor


def marginal_topk(
    logits: torch.Tensor,
    *,
    k: int,
    vocab_start: int = 0,
    gather: Callable | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:

    if logits.ndim != 2 or not logits.is_floating_point():
        raise ValueError("logits must be a floating [rows, local_vocab] tensor")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    if logits.shape[-1] < k:
        raise ValueError("each vocabulary shard must contain at least k tokens")
    values = logits.float()


    shift = values.amax(dim=-1, keepdim=True)
    log_z = shift.double() + torch.logsumexp(values - shift, dim=-1, keepdim=True).double()

    ranks = torch.argsort(values, dim=-1, descending=True, stable=True)[:, :k]
    top = values.gather(-1, ranks)
    ids = ranks + vocab_start
    if gather is not None:
        top = gather(top)
        ids = gather(ids)
        log_z = torch.logsumexp(gather(log_z), dim=-1, keepdim=True)

        id_order = torch.argsort(ids, dim=-1, stable=True)
        ids, top = ids.gather(-1, id_order), top.gather(-1, id_order)
        order = torch.argsort(top, dim=-1, descending=True, stable=True)[:, :k]
        ids, top = ids.gather(-1, order), top.gather(-1, order)
    return ids.long(), torch.exp(top.double() - log_z).float()


def build_marginal_proposal(draft_model, hidden, lm_head, *, k: int, chunk_rows=32):

    from sglang.srt.distributed.communication_op import tensor_model_parallel_all_gather
    from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method
    from sglang.srt.models.dflash import _project_candidate_logits
    from sglang.srt.runtime_context import get_parallel

    batch, depth, width = hidden.shape
    flat = hidden.reshape(-1, width)
    sharded = get_parallel().tp_size > 1
    shard = lm_head.shard_indices if sharded else None
    size = int(shard.num_org_elements if sharded else lm_head.org_vocab_size)
    start = int(shard.org_vocab_start_index) if sharded else 0
    use_quant = should_apply_lm_head_quant_method(lm_head, getattr(lm_head, "quant_method", None))
    gather = (lambda x: tensor_model_parallel_all_gather(x, dim=-1)) if sharded else None
    transform = getattr(draft_model, "_transform_unary_logits", None)
    all_ids, all_probs = [], []
    for offset in range(0, flat.shape[0], chunk_rows):
        logits = _project_candidate_logits(flat[offset:offset + chunk_rows], lm_head,
                                           num_org=size, use_quant_head=use_quant)
        if transform is not None:
            logits = transform(logits)
        ids, probs = marginal_topk(logits, k=k, vocab_start=start, gather=gather)
        all_ids.append(ids)
        all_probs.append(probs)
    return MarginalProposal(torch.cat(all_ids).view(batch, depth, k),
                            torch.cat(all_probs).view(batch, depth, k))
