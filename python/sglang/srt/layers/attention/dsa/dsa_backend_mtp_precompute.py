from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.attention.utils import seqlens_expand_triton
from sglang.srt.layers.attention.dsa.utils import compute_dsa_seqlens
from sglang.srt.utils import is_cuda, is_hip

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

_is_cuda = is_cuda()
_is_hip = is_hip()


@dataclass
class PrecomputedMetadata:


    cache_seqlens: torch.Tensor
    cu_seqlens_k: torch.Tensor


    page_indices: torch.Tensor
    real_page_table: Optional[torch.Tensor]


    seqlens_expanded: torch.Tensor
    dsa_cache_seqlens: torch.Tensor
    dsa_cu_seqlens_k: torch.Tensor
    seqlens_expanded_size: int


    max_len: int
    max_seqlen_k: int


    flashmla_metadata: Optional[torch.Tensor] = None


def compute_cu_seqlens(seqlens: torch.Tensor) -> torch.Tensor:

    assert seqlens.dtype == torch.int32
    return torch.nn.functional.pad(
        torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0)
    )


class DeepseekSparseAttnBackendMTPPrecomputeMixin:


    def _precompute_replay_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        forward_mode: ForwardMode,
    ) -> PrecomputedMetadata:


        seq_lens = seq_lens[:bs]
        if seq_lens_cpu is not None:
            seq_lens_cpu = seq_lens_cpu[:bs]
        req_pool_indices = req_pool_indices[:bs]


        if forward_mode.is_decode_or_idle():
            return self._precompute_decode_mode(
                bs, req_pool_indices, seq_lens, seq_lens_cpu
            )
        elif forward_mode.is_target_verify():
            return self._precompute_target_verify_mode(
                bs, req_pool_indices, seq_lens, seq_lens_cpu
            )
        else:
            raise ValueError(f"Unsupported forward mode: {forward_mode}")

    def _precompute_decode_mode(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
    ) -> PrecomputedMetadata:

        max_len = self.decode_cuda_graph_metadata[bs].page_table_1.shape[1]

        if _is_cuda and not _is_hip:
            from sglang.kernels.ops.attention.dsa_metadata import (
                fused_dsa_decode_metadata,
            )

            cache_seqlens = torch.empty(bs, dtype=torch.int32, device=self.device)
            cu_seqlens_k = torch.empty(bs + 1, dtype=torch.int32, device=self.device)
            page_indices = torch.empty(
                (bs, max_len), dtype=torch.int32, device=self.device
            )
            dsa_cache_seqlens = torch.empty(bs, dtype=torch.int32, device=self.device)
            dsa_cu_seqlens_k = torch.empty(
                bs + 1, dtype=torch.int32, device=self.device
            )
            if self.real_page_size > 1:
                real_cols = (max_len + self.real_page_size - 1) // self.real_page_size
                real_page_table = torch.empty(
                    (bs, real_cols), dtype=torch.int32, device=self.device
                )
                real_page_table_arg = real_page_table
            else:
                real_page_table = None
                real_page_table_arg = page_indices

            fused_dsa_decode_metadata(
                seq_lens=seq_lens,
                req_pool_indices=req_pool_indices,
                req_to_token=self.req_to_token,
                cache_seqlens=cache_seqlens,
                cu_seqlens_k=cu_seqlens_k,
                page_table_1=page_indices,
                dsa_cache_seqlens=dsa_cache_seqlens,
                dsa_cu_seqlens_k=dsa_cu_seqlens_k,
                real_page_table=real_page_table_arg,
                bs=bs,
                max_len=max_len,
                dsa_index_topk=self.dsa_index_topk,
                real_page_size=self.real_page_size,
            )
            seqlens_expanded = cache_seqlens
            seqlens_expanded_size = bs

            flashmla_metadata = None
            if self.dsa_decode_impl == "flashmla_kv":
                flashmla_metadata = self._compute_flashmla_metadata(
                    cache_seqlens=dsa_cache_seqlens,
                    seq_len_q=1,
                )

            return PrecomputedMetadata(
                cache_seqlens=cache_seqlens,
                cu_seqlens_k=cu_seqlens_k,
                page_indices=page_indices,
                real_page_table=real_page_table,
                seqlens_expanded=seqlens_expanded,
                dsa_cache_seqlens=dsa_cache_seqlens,
                dsa_cu_seqlens_k=dsa_cu_seqlens_k,
                seqlens_expanded_size=seqlens_expanded_size,
                max_len=max_len,
                max_seqlen_k=max_len,
                flashmla_metadata=flashmla_metadata,
            )


        cache_seqlens = seq_lens.to(torch.int32)
        cu_seqlens_k = compute_cu_seqlens(cache_seqlens)


        page_indices = self.req_to_token[req_pool_indices, :max_len].contiguous()


        dsa_cache_seqlens = compute_dsa_seqlens(
            cache_seqlens, dsa_index_topk=self.dsa_index_topk
        )
        seqlens_expanded = cache_seqlens
        seqlens_expanded_size = seqlens_expanded.shape[0]


        dsa_cu_seqlens_k = compute_cu_seqlens(dsa_cache_seqlens)


        if self.real_page_size > 1:
            real_page_table = self._transform_table_1_to_real(page_indices)
        else:
            real_page_table = None


        flashmla_metadata = None
        if self.dsa_decode_impl == "flashmla_kv":
            flashmla_metadata = self._compute_flashmla_metadata(
                cache_seqlens=dsa_cache_seqlens,
                seq_len_q=1,
            )

        return PrecomputedMetadata(
            cache_seqlens=cache_seqlens,
            cu_seqlens_k=cu_seqlens_k,
            page_indices=page_indices,
            real_page_table=real_page_table,
            seqlens_expanded=seqlens_expanded,
            dsa_cache_seqlens=dsa_cache_seqlens,
            dsa_cu_seqlens_k=dsa_cu_seqlens_k,
            seqlens_expanded_size=seqlens_expanded_size,
            max_len=max_len,
            max_seqlen_k=max_len,
            flashmla_metadata=flashmla_metadata,
        )

    def _precompute_target_verify_mode(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
    ) -> PrecomputedMetadata:

        max_seqlen_k = self.decode_cuda_graph_metadata[bs].page_table_1.shape[1]
        seqlens_expanded_size = bs * self.speculative_num_draft_tokens

        if _is_cuda and not _is_hip:
            from sglang.kernels.ops.attention.dsa_metadata import (
                fused_dsa_target_verify_metadata,
            )

            cache_seqlens = torch.empty(bs, dtype=torch.int32, device=self.device)
            cu_seqlens_k = torch.empty(bs + 1, dtype=torch.int32, device=self.device)
            page_indices = torch.empty(
                (seqlens_expanded_size, max_seqlen_k),
                dtype=torch.int32,
                device=self.device,
            )
            seqlens_expanded = torch.empty(
                seqlens_expanded_size, dtype=torch.int32, device=self.device
            )
            dsa_cache_seqlens = torch.empty(
                seqlens_expanded_size, dtype=torch.int32, device=self.device
            )
            dsa_cu_seqlens_k = torch.empty(
                seqlens_expanded_size + 1,
                dtype=torch.int32,
                device=self.device,
            )
            if self.real_page_size > 1:
                real_cols = (
                    max_seqlen_k + self.real_page_size - 1
                ) // self.real_page_size
                real_page_table = torch.empty(
                    (seqlens_expanded_size, real_cols),
                    dtype=torch.int32,
                    device=self.device,
                )
                real_page_table_arg = real_page_table
            else:
                real_page_table = None
                real_page_table_arg = page_indices

            fused_dsa_target_verify_metadata(
                seq_lens=seq_lens,
                req_pool_indices=req_pool_indices,
                req_to_token=self.req_to_token,
                cache_seqlens=cache_seqlens,
                cu_seqlens_k=cu_seqlens_k,
                page_table_1=page_indices,
                seqlens_expanded=seqlens_expanded,
                dsa_cache_seqlens=dsa_cache_seqlens,
                dsa_cu_seqlens_k=dsa_cu_seqlens_k,
                real_page_table=real_page_table_arg,
                bs=bs,
                max_seqlen_k=max_seqlen_k,
                dsa_index_topk=self.dsa_index_topk,
                real_page_size=self.real_page_size,
                next_n=self.speculative_num_draft_tokens,
            )

            flashmla_metadata = None
            if self.dsa_decode_impl == "flashmla_kv":
                flashmla_metadata = self._compute_flashmla_metadata(
                    cache_seqlens=dsa_cache_seqlens,
                    seq_len_q=1,
                )

            return PrecomputedMetadata(
                cache_seqlens=cache_seqlens,
                cu_seqlens_k=cu_seqlens_k,
                page_indices=page_indices,
                real_page_table=real_page_table,
                seqlens_expanded=seqlens_expanded,
                dsa_cache_seqlens=dsa_cache_seqlens,
                dsa_cu_seqlens_k=dsa_cu_seqlens_k,
                seqlens_expanded_size=seqlens_expanded_size,
                max_len=-1,
                max_seqlen_k=max_seqlen_k,
                flashmla_metadata=flashmla_metadata,
            )


        cache_seqlens = (seq_lens + self.speculative_num_draft_tokens).to(torch.int32)
        cu_seqlens_k = compute_cu_seqlens(cache_seqlens)


        page_indices = self.req_to_token[req_pool_indices, :max_seqlen_k]
        page_indices = torch.repeat_interleave(
            page_indices, repeats=self.speculative_num_draft_tokens, dim=0
        ).contiguous()


        extend_seq_lens = torch.full(
            (bs,),
            self.speculative_num_draft_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        seqlens_expanded = seqlens_expand_triton(
            extend_seq_lens,
            cache_seqlens,
            bs * self.speculative_num_draft_tokens,
            self.speculative_num_draft_tokens,
        )


        dsa_cache_seqlens = compute_dsa_seqlens(seqlens_expanded, self.dsa_index_topk)
        seqlens_expanded_size = seqlens_expanded.shape[0]


        dsa_cu_seqlens_k = compute_cu_seqlens(dsa_cache_seqlens)


        if self.real_page_size > 1:
            real_page_table = self._transform_table_1_to_real(page_indices)
        else:
            real_page_table = None


        flashmla_metadata = None
        if self.dsa_decode_impl == "flashmla_kv":
            flashmla_metadata = self._compute_flashmla_metadata(
                cache_seqlens=dsa_cache_seqlens,
                seq_len_q=1,
            )

        return PrecomputedMetadata(
            cache_seqlens=cache_seqlens,
            cu_seqlens_k=cu_seqlens_k,
            page_indices=page_indices,
            real_page_table=real_page_table,
            seqlens_expanded=seqlens_expanded,
            dsa_cache_seqlens=dsa_cache_seqlens,
            dsa_cu_seqlens_k=dsa_cu_seqlens_k,
            seqlens_expanded_size=seqlens_expanded_size,
            max_len=-1,
            max_seqlen_k=max_seqlen_k,
            flashmla_metadata=flashmla_metadata,
        )


DeepseekSparseAttnBackendMTPPrecomputeMixin = (
    DeepseekSparseAttnBackendMTPPrecomputeMixin
)
