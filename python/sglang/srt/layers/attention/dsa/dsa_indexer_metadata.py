from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from sglang.srt.layers.attention.dsa.dsa_backend_mtp_precompute import (
    compute_cu_seqlens,
)
from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
    DSATopKBackend,
    TopkTransformMethod,
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.dsa_backend import DSAMetadata


class BaseIndexerMetadata(ABC):
    @abstractmethod
    def get_seqlens_int32(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get_page_table_64(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get_page_table_1(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get_seqlens_expanded(self) -> torch.Tensor:
        pass

    def get_indexer_kvcache_range(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def get_indexer_seq_len_cpu(self) -> torch.Tensor:
        pass

    def get_indexer_seq_len(self) -> torch.Tensor:
        pass

    def get_dsa_extend_len_cpu(self) -> List[int]:
        pass

    def get_token_to_batch_idx(self) -> torch.Tensor:
        pass

    @abstractmethod
    def topk_transform(
        self,
        logits: torch.Tensor,
        topk: int,
        **kwargs,
    ) -> torch.Tensor:
        pass


@dataclass(frozen=True)
class DSAIndexerMetadata(BaseIndexerMetadata):
    attn_metadata: DSAMetadata
    topk_transform_method: TopkTransformMethod
    topk_backend: DSATopKBackend = DSATopKBackend.SGL_KERNEL
    paged_mqa_schedule_metadata: Optional[torch.Tensor] = None
    paged_mqa_ctx_lens_2d: Optional[torch.Tensor] = None
    force_unfused_topk: bool = False

    def get_seqlens_int32(self) -> torch.Tensor:
        return self.attn_metadata.cache_seqlens_int32

    def get_page_table_64(self) -> torch.Tensor:
        return self.attn_metadata.real_page_table

    def get_page_table_1(self) -> torch.Tensor:
        return self.attn_metadata.page_table_1

    def get_seqlens_expanded(self) -> torch.Tensor:
        return self.attn_metadata.dsa_seqlens_expanded

    def get_indexer_kvcache_range(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.attn_metadata.indexer_k_start_end

    def get_indexer_seq_len(self) -> torch.Tensor:
        return self.attn_metadata.indexer_seq_lens

    def get_indexer_seq_len_cpu(self) -> torch.Tensor:
        return self.attn_metadata.indexer_seq_lens_cpu

    def get_dsa_extend_len_cpu(self) -> List[int]:
        return self.attn_metadata.dsa_extend_seq_lens_list

    def get_token_to_batch_idx(self) -> torch.Tensor:
        return self.attn_metadata.token_to_batch_idx

    def topk_transform(
        self,
        logits: torch.Tensor,
        topk: int,
        ks: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        ke_offset: Optional[torch.Tensor] = None,
        batch_idx_list: Optional[List[int]] = None,
        topk_indices_offset_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if topk_indices_offset_override is not None:
            cu_topk_indices_offset = topk_indices_offset_override
            cu_seqlens_q_topk = None
        elif cu_seqlens_q is not None:
            cu_seqlens_q = cu_seqlens_q.to(torch.int32)
            cu_seqlens_q_topk = compute_cu_seqlens(cu_seqlens_q)
            cu_topk_indices_offset = torch.repeat_interleave(
                cu_seqlens_q_topk[:-1],
                cu_seqlens_q,

                output_size=logits.shape[0],
            )
        else:
            cu_seqlens_q_topk = self.attn_metadata.cu_seqlens_q
            cu_topk_indices_offset = self.attn_metadata.topk_indices_offset
        if ke_offset is not None:
            seq_lens_topk = ke_offset
        else:
            seq_lens_topk = self.get_seqlens_expanded()
        return self.topk_backend.topk_transform(
            logits=logits,
            lengths=seq_lens_topk,
            topk=topk,
            topk_transform_method=self.topk_transform_method,
            attn_metadata=self.attn_metadata,
            cu_seqlens_q_topk=cu_seqlens_q_topk,
            topk_indices_offset=cu_topk_indices_offset,
            row_starts=ks,
            batch_idx_list=batch_idx_list,
            force_unfused_topk=self.force_unfused_topk,
        )
