from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass(frozen=True, slots=True, kw_only=True)
class PlanInput:


    req_pool_indices: torch.Tensor
    prefix_lens: torch.Tensor
    extend_seq_lens: torch.Tensor
    req_to_verify_expected_tokens_valid_lens: torch.Tensor

    def zero_(self) -> None:
        self.req_pool_indices.zero_()
        self.prefix_lens.zero_()
        self.extend_seq_lens.zero_()
        self.req_to_verify_expected_tokens_valid_lens.zero_()

    @classmethod
    def allocate(
        cls,
        *,
        bs_capacity: int,
        device: torch.device,
    ) -> PlanInput:
        return cls(
            req_pool_indices=torch.zeros(bs_capacity, dtype=torch.int64, device=device),
            prefix_lens=torch.zeros(bs_capacity, dtype=torch.int64, device=device),
            extend_seq_lens=torch.zeros(bs_capacity, dtype=torch.int64, device=device),
            req_to_verify_expected_tokens_valid_lens=torch.zeros(
                bs_capacity, dtype=torch.int64, device=device
            ),
        )

    def fill_from_forward_batch(self, *, forward_batch: ForwardBatch) -> None:
        req_pool_indices = forward_batch.req_pool_indices
        bs = int(req_pool_indices.shape[0])
        capacity = int(self.req_pool_indices.shape[0])
        if bs > capacity:
            raise RuntimeError(
                f"kv-canary: per-forward batch size {bs} exceeds static capacity {capacity}; "
                "raise the buffer size in CanaryLaunchCapacities"
            )

        self.zero_()
        self.req_pool_indices[:bs].copy_(req_pool_indices)

        _extract_prefix_lens_and_extend_seq_lens(
            forward_batch=forward_batch,
            out_prefix_lens=self.prefix_lens[:bs],
            out_extend_seq_lens=self.extend_seq_lens[:bs],
            bs=bs,
        )

        req_all_ids_lens = forward_batch.req_all_ids_lens
        if req_all_ids_lens is not None:
            self.req_to_verify_expected_tokens_valid_lens[:bs].copy_(
                req_all_ids_lens.to(torch.int64), non_blocking=True
            )


def _extract_prefix_lens_and_extend_seq_lens(
    *,
    forward_batch: ForwardBatch,
    out_prefix_lens: torch.Tensor,
    out_extend_seq_lens: torch.Tensor,
    bs: int,
) -> None:


    forward_mode = forward_batch.forward_mode
    spec_info = forward_batch.spec_info
    if forward_mode.is_decode_or_idle():


        positions = forward_batch.positions
        out_prefix_lens[: positions.shape[0]].copy_(positions.to(torch.int64))
        out_extend_seq_lens.fill_(1)
    elif forward_mode.is_target_verify():


        out_prefix_lens.copy_(forward_batch.seq_lens[:bs].to(torch.int64))
        out_extend_seq_lens.fill_(int(spec_info.draft_token_num))
    elif forward_mode.is_draft_extend_v2():


        extend_seq_lens = forward_batch.extend_seq_lens[:bs].to(torch.int64)
        out_extend_seq_lens.copy_(extend_seq_lens)
        out_prefix_lens.copy_(
            forward_batch.seq_lens[:bs].to(torch.int64) - extend_seq_lens
        )
    elif forward_mode.is_extend():


        out_prefix_lens.copy_(forward_batch.extend_prefix_lens[:bs].to(torch.int64))
        out_extend_seq_lens.copy_(forward_batch.extend_seq_lens[:bs].to(torch.int64))
    else:
        raise NotImplementedError(
            f"Unsupported forward mode for kv-canary: {forward_mode}"
        )
