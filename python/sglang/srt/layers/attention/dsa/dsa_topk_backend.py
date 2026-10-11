from __future__ import annotations

from enum import Enum, IntEnum, auto
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_exec, get_spec

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

_FLASHINFER_TIE_BREAK_VALUES = {
    "small": 1,
    "large": 2,
}


class TopkTransformMethod(IntEnum):

    PAGED = auto()

    RAGGED = auto()


class DSATopKBackend(Enum):
    SGL_KERNEL = "sgl-kernel"
    TORCH = "torch"
    FLASHINFER = "flashinfer"

    @classmethod
    def resolve(cls, model_runner: ModelRunner) -> DSATopKBackend:

        if model_runner.is_draft_worker:
            return cls(get_spec().speculative_dsa_topk_backend)
        return cls(get_exec().kernel.dsa_topk_backend)

    def is_sgl_kernel(self) -> bool:
        return self == DSATopKBackend.SGL_KERNEL

    def is_torch(self) -> bool:
        return self == DSATopKBackend.TORCH

    def is_flashinfer(self) -> bool:
        return self == DSATopKBackend.FLASHINFER

    def should_use_topk_v2(self) -> bool:
        return self.is_sgl_kernel() and envs.SGLANG_OPT_USE_TOPK_V2.get()

    def topk_func(
        self,
        score: torch.Tensor,
        lengths: torch.Tensor,
        topk: int,
        row_starts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.is_sgl_kernel():
            from sgl_kernel import fast_topk_v2

            return fast_topk_v2(score, lengths, topk, row_starts=row_starts)
        if self.is_torch():
            return _topk_unfused(
                score,
                lengths,
                topk,
                row_starts=row_starts,
                topk_op=torch.topk,
                topk_op_kwargs={"dim": -1},
            )
        if self.is_flashinfer():
            import flashinfer

            return _topk_unfused(
                score,
                lengths,
                topk,
                row_starts=row_starts,
                topk_op=flashinfer.top_k,
                topk_op_kwargs={
                    "sorted": False,
                    "deterministic": envs.SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC.get(),
                    "tie_break": _flashinfer_tie_break_value(),
                    "dsa_graph_safe": True,
                },
            )
        raise RuntimeError(f"Unsupported {self = }.")

    def topk_transform(
        self,
        logits: torch.Tensor,
        lengths: torch.Tensor,
        topk: int,
        topk_transform_method: TopkTransformMethod,
        attn_metadata,
        cu_seqlens_q_topk: Optional[torch.Tensor] = None,
        topk_indices_offset: Optional[torch.Tensor] = None,
        row_starts: Optional[torch.Tensor] = None,
        batch_idx_list: Optional[List[int]] = None,
        force_unfused_topk: bool = False,
    ) -> torch.Tensor:
        if not envs.SGLANG_DSA_FUSE_TOPK.get() or force_unfused_topk:
            return self.topk_func(logits, lengths, topk, row_starts=row_starts)


        if (
            self.should_use_topk_v2()
            and topk_transform_method == TopkTransformMethod.PAGED
            and row_starts is None
            and batch_idx_list is None
            and 0 < topk <= 2048
            and lengths.shape[0]
            == logits.shape[0]
            == attn_metadata.real_page_table.shape[0]
        ):
            return _topk_transform_v2_paged(logits, lengths, topk, attn_metadata)


        if (
            self.should_use_topk_v2()
            and topk_transform_method == TopkTransformMethod.RAGGED
            and topk_indices_offset is not None
            and batch_idx_list is None
            and 0 < topk <= 2048
            and lengths.shape[0] == logits.shape[0] == topk_indices_offset.shape[0]
        ):
            return _topk_transform_v2_ragged(
                logits, lengths, topk, topk_indices_offset, row_starts
            )


        assert attn_metadata.page_table_1 is not None

        if self.is_sgl_kernel():
            from sgl_kernel import (
                fast_topk_transform_fused,
                fast_topk_transform_ragged_fused,
            )

            if topk_transform_method == TopkTransformMethod.PAGED:
                page_table_size_1 = (
                    attn_metadata.page_table_1[batch_idx_list]
                    if batch_idx_list is not None
                    else attn_metadata.page_table_1
                )
                return fast_topk_transform_fused(
                    score=logits,
                    lengths=lengths,
                    page_table_size_1=page_table_size_1,
                    cu_seqlens_q=cu_seqlens_q_topk,
                    topk=topk,
                    row_starts=row_starts,
                )
            if topk_transform_method == TopkTransformMethod.RAGGED:
                if topk_indices_offset is None:
                    raise RuntimeError(
                        "RAGGED topk_transform requires topk_indices_offset; "
                        "expected extend-without-speculative metadata."
                    )
                return fast_topk_transform_ragged_fused(
                    score=logits,
                    lengths=lengths,
                    topk_indices_offset=topk_indices_offset,
                    topk=topk,
                    row_starts=row_starts,
                )
            raise RuntimeError(f"Unsupported {topk_transform_method = }.")

        if self.is_flashinfer():
            import flashinfer

            if topk_transform_method == TopkTransformMethod.PAGED:
                row_to_batch, page_table_row_starts = _build_flashinfer_paged_args(
                    attn_metadata=attn_metadata,
                    row_starts=row_starts,
                    cu_seqlens_q_topk=cu_seqlens_q_topk,
                    batch_idx_list=batch_idx_list,
                    device=logits.device,
                    num_rows=logits.shape[0],
                )
                return flashinfer.top_k_page_table_transform(
                    logits.contiguous(),
                    attn_metadata.page_table_1.contiguous(),
                    lengths.contiguous(),
                    topk,
                    row_to_batch=row_to_batch,
                    deterministic=envs.SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC.get(),
                    tie_break=_flashinfer_tie_break_value(),
                    dsa_graph_safe=True,
                    row_starts=row_starts,
                    page_table_row_starts=page_table_row_starts,
                )
            if topk_transform_method == TopkTransformMethod.RAGGED:
                if topk_indices_offset is None:
                    raise RuntimeError(
                        "RAGGED topk_transform requires topk_indices_offset; "
                        "expected extend-without-speculative metadata."
                    )
                return flashinfer.top_k_ragged_transform(
                    logits.contiguous(),
                    topk_indices_offset.contiguous(),
                    lengths.contiguous(),
                    topk,
                    deterministic=envs.SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC.get(),
                    tie_break=_flashinfer_tie_break_value(),
                    dsa_graph_safe=True,
                    row_starts=row_starts,
                )
            raise RuntimeError(f"Unsupported {topk_transform_method = }.")

        raise RuntimeError(f"Unsupported {self = } for SGLANG_DSA_FUSE_TOPK.")


def _topk_unfused(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
    topk_op: Callable[..., Tuple[torch.Tensor, torch.Tensor]] = torch.topk,
    topk_op_kwargs: Optional[Dict[str, object]] = None,
) -> torch.Tensor:
    batch_size, max_score_len = score.shape
    topk_indices = score.new_full((batch_size, topk), -1, dtype=torch.int32)
    if batch_size == 0 or topk == 0 or max_score_len == 0:
        return topk_indices

    if row_starts is None:
        row_starts = torch.zeros_like(lengths, dtype=torch.int32, device=score.device)
    else:
        row_starts = row_starts.to(dtype=torch.int32, device=score.device)
    lengths = lengths.to(dtype=torch.int32, device=score.device)

    col_indices = torch.arange(max_score_len, dtype=torch.int32, device=score.device)
    col_indices = col_indices.unsqueeze(0)
    row_starts_unsqueezed = row_starts.unsqueeze(1)
    row_ends_unsqueezed = (row_starts + lengths).unsqueeze(1)
    valid_mask = (col_indices >= row_starts_unsqueezed) & (
        col_indices < row_ends_unsqueezed
    )

    masked_logits = score.masked_fill(~valid_mask, float("-inf"))
    valid_topk = min(topk, max_score_len)
    topk_kwargs = topk_op_kwargs or {}
    topk_scores, topk_col_indices = topk_op(masked_logits, valid_topk, **topk_kwargs)
    topk_local_indices = topk_col_indices.to(torch.int32) - row_starts_unsqueezed
    topk_local_indices = topk_local_indices.masked_fill(
        topk_scores == float("-inf"), -1
    )
    topk_indices[:, :valid_topk] = topk_local_indices

    return topk_indices


def _topk_transform_v2_paged(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    attn_metadata,
) -> torch.Tensor:

    from sglang.kernels.ops.attention.dsv4.topk import topk_transform_512_v2

    num_rows = logits.shape[0]


    assert (
        logits.dtype == torch.float32
        and logits.stride(1) == 1
        and logits.stride(0) % 4 == 0
    ), f"v2 top-k expects fp32 scores with unit row stride and 16B-aligned score_stride, got {logits.dtype=} {logits.stride()=}"
    assert 0 < topk <= 2048, f"v2 top-k supports 0 < topk <= 2048, got {topk=}"

    page_table = attn_metadata.real_page_table


    plan = attn_metadata.topk_v2_plan
    assert (
        plan is not None and plan.shape[0] == num_rows + 1
    ), "topk_v2_plan must be preprocessed per forward (see DSAMetadata.topk_v2_plan)"

    page_size = attn_metadata.page_size
    out = logits.new_empty((num_rows, topk), dtype=torch.int32)
    topk_transform_512_v2(logits, lengths, page_table, out, page_size, plan)
    return out


def _topk_transform_v2_ragged(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    topk_indices_offset: torch.Tensor,
    row_starts: Optional[torch.Tensor],
) -> torch.Tensor:

    from sglang.kernels.ops.attention.dsv4.topk import topk_transform_ragged_v2

    out = logits.new_empty((logits.shape[0], topk), dtype=torch.int32)
    topk_transform_ragged_v2(
        logits,
        lengths,
        out_offsets=topk_indices_offset,
        out_indices=out,
        row_starts=row_starts,
    )
    return out


def _build_flashinfer_paged_args(
    attn_metadata,
    row_starts: Optional[torch.Tensor],
    cu_seqlens_q_topk: Optional[torch.Tensor],
    batch_idx_list: Optional[List[int]],
    device: torch.device,
    num_rows: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    row_to_batch = (
        torch.as_tensor(batch_idx_list, dtype=torch.int32, device=device)
        if batch_idx_list is not None
        else None
    )


    if (
        row_to_batch is not None
        and cu_seqlens_q_topk is not None
        and row_to_batch.shape[0] != num_rows
    ):
        q_lens = (cu_seqlens_q_topk[1:] - cu_seqlens_q_topk[:-1]).to(
            dtype=torch.int32, device=device
        )
        row_to_batch = torch.repeat_interleave(
            row_to_batch, q_lens, output_size=num_rows
        )

    if row_to_batch is None and cu_seqlens_q_topk is not None:


        num_batches = cu_seqlens_q_topk.shape[0] - 1
        if not (row_starts is None and num_rows == num_batches):
            q_lens = (cu_seqlens_q_topk[1:] - cu_seqlens_q_topk[:-1]).to(
                dtype=torch.int32, device=device
            )
            row_to_batch = torch.repeat_interleave(
                torch.arange(q_lens.shape[0], dtype=torch.int32, device=device),
                q_lens,
                output_size=num_rows,
            )

    if row_starts is not None and row_to_batch is None:
        raise RuntimeError(
            "PAGED topk_transform with row_starts requires cu_seqlens_q metadata."
        )

    page_table_row_starts = row_starts
    if page_table_row_starts is not None and row_to_batch is not None:
        page_table_row_starts = (
            page_table_row_starts - attn_metadata.cu_seqlens_k[:-1][row_to_batch]
        )

    return row_to_batch, page_table_row_starts


def _flashinfer_tie_break_value() -> int:
    mode = envs.SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK.get()
    if mode is None:
        return 0
    mode = mode.lower()
    if mode not in _FLASHINFER_TIE_BREAK_VALUES:
        raise RuntimeError(
            "SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK must be one of "
            f"{tuple(_FLASHINFER_TIE_BREAK_VALUES)} or unset, got {mode!r}."
        )
    return _FLASHINFER_TIE_BREAK_VALUES[mode]
