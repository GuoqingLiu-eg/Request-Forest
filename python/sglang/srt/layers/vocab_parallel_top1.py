from __future__ import annotations

from typing import Any

import torch


class VocabParallelTop1:


    def __init__(self, tp_group: Any):
        self.tp_group = tp_group
        self.tp_size = int(tp_group.world_size)
        self._buffers: dict[tuple, tuple[torch.Tensor, ...]] = {}

    def __call__(
        self, logits: torch.Tensor, *, num_valid: int, vocab_start: int
    ) -> torch.Tensor:
        if logits.ndim != 2 or not 0 < num_valid <= logits.shape[1]:
            raise ValueError("top-1 requires a nonempty valid local vocabulary")
        rows = logits.shape[0]
        key = (rows, logits.dtype, logits.device)
        buffers = self._buffers.get(key)
        if buffers is None:
            score_options = dict(device=logits.device, dtype=logits.dtype)
            id_options = dict(device=logits.device, dtype=torch.int64)
            buffers = (
                torch.empty(rows, **score_options),
                torch.empty(rows, **id_options),
                torch.empty(self.tp_size * rows, **score_options),
                torch.empty(self.tp_size * rows, **id_options),
                torch.empty((1, rows), **id_options),
                torch.empty((1, rows), **id_options),
            )
            self._buffers[key] = buffers
        local_max, local_id, all_max, all_ids, best_rank, selected = buffers
        torch.max(logits[:, :num_valid], dim=-1, out=(local_max, local_id))
        local_id.add_(vocab_start)
        self.tp_group.all_gather_into_tensor(all_max, local_max)
        self.tp_group.all_gather_into_tensor(all_ids, local_id)
        torch.argmax(all_max.view(self.tp_size, rows), dim=0, out=best_rank[0])
        torch.gather(all_ids.view(self.tp_size, rows), 0, best_rank, out=selected)
        return selected.view(-1)
