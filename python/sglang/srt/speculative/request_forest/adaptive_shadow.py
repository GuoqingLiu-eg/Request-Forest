import dataclasses
import json
import logging

import torch

from sglang.srt.speculative.dflash_info import DFlashVerifyInput

logger = logging.getLogger(__name__)


def compare_native_on_forest_prefix(worker, batch, prepared, forest, result, acceptance):
    index = 65
    counts = acceptance.accept_lens.tolist()
    interested = [(i, index - len(req.output_ids))
                  for i, req in enumerate(prepared.requests)
                  if 0 <= index - len(req.output_ids) < counts[i]]
    if not interested:
        return
    width = worker.block_size
    rows = acceptance.accept_index

    interested = [(i, j) for i, j in interested if j < width]
    if not interested:
        return
    bs = len(prepared.requests)
    device = forest.draft_token.device
    candidates = torch.zeros((bs, width), dtype=torch.int64, device=device)
    for i, count in enumerate(counts):
        n = min(count, width)
        candidates[i, :n] = forest.draft_token[rows[i, :n].long()]
    prefixes = torch.tensor(prepared.committed_lens, dtype=torch.int64, device=device)
    positions = (prefixes[:, None] + torch.arange(width, device=device)).reshape(-1)

    reference = result.logits_output.next_token_logits.detach().clone()
    result.logits_output.next_token_logits = reference
    if result.logits_output.hidden_states is not None:
        result.logits_output.hidden_states = result.logits_output.hidden_states.clone()
    table = batch.req_to_token_pool.req_to_token
    backups = [table[row, prefix:prefix + width].clone()
               for row, prefix in zip(prepared.request_rows, prepared.committed_lens)]
    if any(x.numel() != width for x in backups):
        raise RuntimeError("Native shadow probe has insufficient request-table headroom")
    snapshot = {f.name: getattr(batch, f.name) for f in dataclasses.fields(batch)}
    allocator = batch.token_to_kv_pool_allocator
    slots = allocator.alloc(bs * width)
    if slots is None:
        raise RuntimeError("Native shadow probe cannot allocate scratch KV")
    try:
        for i, (row, prefix) in enumerate(zip(prepared.request_rows, prepared.committed_lens)):
            table[row, prefix:prefix + width] = slots[i * width:(i + 1) * width]
        batch.seq_lens = prefixes
        batch.seq_lens_cpu = prefixes.cpu() + width
        batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
        batch.out_cache_loc = slots
        native = DFlashVerifyInput(candidates.reshape(-1), positions, width)
        forward, _ = native.prepare_for_verify(batch, worker.target_worker)
        shadow = worker.target_worker.forward_batch_generation(
            batch=None, forward_batch=forward, is_verify=True,
            skip_attn_backend_init=True,
        )
        logits = shadow.logits_output.next_token_logits
        for i, j in interested:
            forest_row = int(rows[i, j])
            native_row = i * width + j
            lhs, rhs = reference[forest_row].float(), logits[native_row].float()
            def summary(value):
                values, ids = torch.topk(value, 5)
                return dict(top5_ids=ids.tolist(), top5_logits=values.tolist(),
                            argmax=int(value.argmax()),
                            comma_logit=float(value[11]), backtick_logit=float(value[1565]))
            logger.info("Adaptive RF shadow %s", json.dumps(dict(
                output_index=index, request=i, batch=bs, prefix_len=prepared.committed_lens[i],
                local_output_index=j, same_committed_kv=True,
                same_causal_token_ids=candidates[i, :j + 1].tolist(),
                forest=summary(lhs), native=summary(rhs),
                max_abs_logit_difference=float((lhs - rhs).abs().max()),
            )))
    finally:


        torch.cuda.synchronize(device)
        for (row, prefix), backup in zip(
            zip(prepared.request_rows, prepared.committed_lens), backups
        ):
            table[row, prefix:prefix + width] = backup
        for name, value in snapshot.items():
            setattr(batch, name, value)
        allocator.free(slots)
