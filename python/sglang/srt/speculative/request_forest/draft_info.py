from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.speculative.spec_info import SpecInput, SpecInputType

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch


@dataclass
class RequestForestDraftInput(SpecInput):


    bonus_tokens: torch.Tensor
    new_seq_lens: torch.Tensor
    future_indices: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        super().__init__(spec_input_type=SpecInputType.REQUEST_FOREST_DRAFT)
        self.num_tokens_per_req = 1
        self.num_tokens_for_logprob_per_req = 1

    def prepare_for_decode(self, batch: ScheduleBatch) -> None:


        for req in batch.reqs:
            req.decode_batch_idx += 1

    def filter_batch(
        self,
        new_indices: torch.Tensor,
        new_indices_cpu: Optional[List[int]] = None,
    ) -> None:
        del new_indices_cpu
        if self.future_indices is not None:
            self.future_indices = self.future_indices[new_indices]
            return
        self.bonus_tokens = self.bonus_tokens[new_indices]
        self.new_seq_lens = self.new_seq_lens[new_indices]

    def merge_batch(self, spec_info: RequestForestDraftInput) -> None:
        if not isinstance(spec_info, RequestForestDraftInput):
            raise TypeError("Request Forest batches require matching draft input state")
        if (self.future_indices is None) != (spec_info.future_indices is None):
            raise ValueError(
                "Request Forest draft inputs must agree on future-index state"
            )

        if self.future_indices is not None:
            self.future_indices = torch.cat(
                (self.future_indices, spec_info.future_indices), dim=0
            )
            return

        self.bonus_tokens = torch.cat(
            (self.bonus_tokens, spec_info.bonus_tokens), dim=0
        )
        self.new_seq_lens = torch.cat(
            (self.new_seq_lens, spec_info.new_seq_lens), dim=0
        )


__all__ = ["RequestForestDraftInput"]
