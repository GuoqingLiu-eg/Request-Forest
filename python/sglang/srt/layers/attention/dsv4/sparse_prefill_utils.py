import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import triton

from sglang.kernels.ops.attention.dsv4.dequant_k_cache import DIM_NOPE, DIM_ROPE
from sglang.srt.utils import ceil_align


SPARSE_PREFILL_TOPK_ALIGNMENT = 128


WORKSPACE_DIM = DIM_NOPE + DIM_ROPE
DSV4_Q8KV8_PREFILL_ENV = "SGLANG_DSV4_Q8KV8_PREFILL"
DSV4_Q8KV8_PREFILL_LOG_ENV = "SGLANG_DSV4_Q8KV8_PREFILL_LOG"

from sglang.kernels.ops.attention.dsv4.sparse_prefill_kernels import (
    _build_swa_token_ids_kernel,
    _combine_topk_swa_indices_kernel,
)


def use_dsv4_q8kv8_sparse_prefill(dsv4_prefill_backend: str = "auto") -> bool:

    env_value = os.getenv(DSV4_Q8KV8_PREFILL_ENV)
    if env_value is not None:
        return env_value.lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    return dsv4_prefill_backend == "flashmla_sparse_q8"


class SparsePrefillWorkspace:


    def __init__(self, device: torch.device):
        self.device = device
        self._buffer: Optional[torch.Tensor] = None

    def get(
        self,
        num_tokens: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        assert num_tokens > 0
        current_capacity = self._buffer.shape[0] if self._buffer is not None else 0
        current_dtype = self._buffer.dtype if self._buffer is not None else None
        if num_tokens > current_capacity or dtype != current_dtype:
            self._buffer = torch.empty(
                (num_tokens, 1, WORKSPACE_DIM),
                dtype=dtype,
                device=self.device,
            )
        return self._buffer[:num_tokens]


def combined_topk_width(topk: int, window_size: int) -> int:

    return ceil_align(topk + window_size, SPARSE_PREFILL_TOPK_ALIGNMENT)


def combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    compressed_base: torch.Tensor,
    swa_base: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    out_indices: Optional[torch.Tensor] = None,
    out_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:

    assert topk_indices.dtype == torch.int32
    assert query_start_loc.dtype == torch.int32
    assert seq_lens.dtype == torch.int32
    assert gather_lens.dtype == torch.int32
    assert compressed_base.dtype == torch.int32
    assert swa_base.dtype == torch.int32
    assert compress_ratio >= 1, "compress_ratio must be >= 1 (use topk=0 for SWA-only)"
    assert (
        topk_indices.shape[-1] >= topk
    ), f"topk_indices width {topk_indices.shape[-1]} must be >= topk {topk}"

    num_tokens = topk_indices.shape[0]
    num_reqs = seq_lens.shape[0]
    combined_topk = combined_topk_width(topk, window_size)
    if out_indices is None:
        combined_indices = torch.full(
            (num_tokens, combined_topk),
            -1,
            dtype=torch.int32,
            device=topk_indices.device,
        )
    else:
        assert out_indices.shape == (num_tokens, combined_topk)
        assert out_indices.dtype == torch.int32
        combined_indices = out_indices
    if out_lens is None:
        combined_lens = torch.zeros(
            num_tokens, dtype=torch.int32, device=topk_indices.device
        )
    else:
        assert out_lens.shape == (num_tokens,)
        assert out_lens.dtype == torch.int32
        combined_lens = out_lens

    NUM_WORKERS = 128
    _combine_topk_swa_indices_kernel[(num_reqs, NUM_WORKERS)](
        combined_indices,
        combined_indices.stride(0),
        combined_lens,
        topk_indices,
        topk_indices.stride(0),
        query_start_loc,
        seq_lens,
        gather_lens,
        compressed_base,
        swa_base,
        top_k=topk,
        COMPRESS_RATIO=compress_ratio,
        WINDOW_SIZE=window_size,
        PADDED_TOP_K=triton.next_power_of_2(topk_indices.shape[-1]),
    )
    return combined_indices, combined_lens


def build_swa_token_ids(
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    full_to_swa: torch.Tensor,
    swa_window: int,
    total_swa: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    assert seq_lens.dtype == torch.int32
    assert extend_seq_lens.dtype == torch.int32
    assert req_pool_indices.dtype == torch.int32
    assert req_to_token.dtype == torch.int32
    assert full_to_swa.dtype == torch.int64

    num_reqs = seq_lens.shape[0]
    device = seq_lens.device

    swa_gather_lens = torch.minimum(seq_lens, extend_seq_lens + (swa_window - 1)).to(
        torch.int32
    )
    swa_first_pos = (seq_lens - swa_gather_lens).to(torch.int32)
    swa_offsets = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
    swa_offsets[1:] = torch.cumsum(swa_gather_lens, dim=0).to(torch.int32)
    swa_token_ids = torch.empty(total_swa, dtype=torch.int32, device=device)
    if total_swa == 0:
        return swa_token_ids, swa_first_pos, swa_gather_lens, swa_offsets

    NUM_WORKERS = 128
    _build_swa_token_ids_kernel[(num_reqs, NUM_WORKERS)](
        swa_token_ids,
        swa_first_pos,
        swa_gather_lens,
        swa_offsets,
        req_pool_indices,
        req_to_token,
        req_to_token.stride(0),
        full_to_swa,
    )
    return swa_token_ids, swa_first_pos, swa_gather_lens, swa_offsets


@dataclass
class SparsePrefillChunkCache:


    num_reqs: int
    num_qo_tokens: int


    max_seq_len: int


    swa_window_size: int


    swa_page_size: int
    seq_lens: torch.Tensor
    query_start_loc: torch.Tensor


    swa_token_ids: torch.Tensor
    swa_first_pos: torch.Tensor
    swa_gather_lens: torch.Tensor
    swa_offsets: torch.Tensor


    c0_combined_indices: torch.Tensor = field(default=None)
    c0_combined_lens: torch.Tensor = field(default=None)

    c128_flat_token_ids: Optional[torch.Tensor] = None
    c128_combined_indices: Optional[torch.Tensor] = None
    c128_combined_lens: Optional[torch.Tensor] = None


    c4_flat_token_ids: Optional[torch.Tensor] = None
    c4_page_size: Optional[int] = None
    c4_compressed_base: Optional[torch.Tensor] = None
    c4_swa_base: Optional[torch.Tensor] = None


    c4_combined_indices: Optional[torch.Tensor] = None
    c4_combined_lens: Optional[torch.Tensor] = None

    @classmethod
    def build(
        cls,
        seq_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        req_to_token: torch.Tensor,
        full_to_swa: torch.Tensor,
        swa_window_size: int,
        swa_page_size: int,
        num_qo_tokens: int,
        max_seq_len: int,
        total_swa: int,
    ) -> "SparsePrefillChunkCache":
        device = seq_lens.device
        num_reqs = seq_lens.shape[0]

        query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
        query_start_loc[1:] = torch.cumsum(extend_seq_lens, dim=0).to(torch.int32)

        swa_token_ids, swa_first_pos, swa_gather_lens, swa_offsets = (
            build_swa_token_ids(
                seq_lens=seq_lens,
                extend_seq_lens=extend_seq_lens,
                req_pool_indices=req_pool_indices,
                req_to_token=req_to_token,
                full_to_swa=full_to_swa,
                swa_window=swa_window_size,
                total_swa=total_swa,
            )
        )

        cache = cls(
            num_reqs=num_reqs,
            num_qo_tokens=num_qo_tokens,
            max_seq_len=max_seq_len,
            swa_window_size=swa_window_size,
            swa_page_size=swa_page_size,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            swa_token_ids=swa_token_ids,
            swa_first_pos=swa_first_pos,
            swa_gather_lens=swa_gather_lens,
            swa_offsets=swa_offsets,
        )


        zero_topk = torch.zeros((num_qo_tokens, 1), dtype=torch.int32, device=device)
        zero_compressed_base = torch.zeros(num_reqs, dtype=torch.int32, device=device)
        c0_swa_base = swa_offsets[:-1].to(torch.int32)
        cache.c0_combined_indices, cache.c0_combined_lens = combine_topk_swa_indices(
            topk_indices=zero_topk,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            gather_lens=swa_gather_lens,
            compressed_base=zero_compressed_base,
            swa_base=c0_swa_base,
            window_size=swa_window_size,
            compress_ratio=1,
            topk=0,
        )
        return cache

    def ensure_c128(self, c128_page_indices: torch.Tensor) -> None:

        if self.c128_flat_token_ids is not None:
            return
        device = self.seq_lens.device
        c128_max = max(self.max_seq_len // 128, 1)
        assert c128_max <= c128_page_indices.shape[-1], (
            f"live c128 extent {c128_max} exceeds metadata capacity "
            f"{c128_page_indices.shape[-1]}"
        )
        last_q_per_req = (self.query_start_loc[1:] - 1).long()
        per_req_c128 = c128_page_indices.narrow(1, 0, c128_max).index_select(
            0, last_q_per_req
        )


        flat_c128_ids = per_req_c128.reshape(-1).clamp_min(0).to(torch.int32)
        compressed_base = (
            torch.arange(self.num_reqs, dtype=torch.int32, device=device) * c128_max
        ).to(torch.int32)
        total_compressed = self.num_reqs * c128_max


        topk_indices = (
            torch.arange(c128_max, dtype=torch.int32, device=device)[None, :]
            .expand(self.num_qo_tokens, -1)
            .contiguous()
        )
        swa_base = (total_compressed + self.swa_offsets[:-1]).to(torch.int32)
        combined_indices, combined_lens = combine_topk_swa_indices(
            topk_indices=topk_indices,
            query_start_loc=self.query_start_loc,
            seq_lens=self.seq_lens,
            gather_lens=self.swa_gather_lens,
            compressed_base=compressed_base,
            swa_base=swa_base,
            window_size=self.swa_window_size,
            compress_ratio=128,
            topk=c128_max,
        )

        self.c128_flat_token_ids = flat_c128_ids
        self.c128_combined_indices = combined_indices
        self.c128_combined_lens = combined_lens

    def ensure_c4(
        self,
        page_table: torch.Tensor,
        c4_page_size: int,
    ) -> None:

        if self.c4_flat_token_ids is not None:
            return
        device = self.seq_lens.device
        c4_max = max(self.max_seq_len // 4, 1)
        c4_capacity = page_table.shape[-1] * c4_page_size
        assert (
            c4_max <= c4_capacity
        ), f"live c4 extent {c4_max} exceeds metadata capacity {c4_capacity}"
        first_q_per_req = self.query_start_loc[:-1].long()
        num_blocks = (c4_max + c4_page_size - 1) // c4_page_size
        assert num_blocks <= page_table.shape[1]
        per_req_page_table = page_table.narrow(1, 0, num_blocks).index_select(
            0, first_q_per_req
        )

        k_arange = torch.arange(c4_max, dtype=torch.int32, device=device)
        block_idx = (k_arange // c4_page_size).long()
        in_page = (k_arange % c4_page_size).to(torch.int32)
        c4_token_ids_2d = (
            per_req_page_table.index_select(1, block_idx) * c4_page_size + in_page
        ).to(torch.int32)
        flat_c4_ids = c4_token_ids_2d.reshape(-1).clamp_min(0)
        total_compressed = self.num_reqs * c4_max
        compressed_base = (
            torch.arange(self.num_reqs, dtype=torch.int32, device=device) * c4_max
        ).to(torch.int32)
        swa_base = (total_compressed + self.swa_offsets[:-1]).to(torch.int32)

        self.c4_flat_token_ids = flat_c4_ids
        self.c4_page_size = c4_page_size
        self.c4_compressed_base = compressed_base
        self.c4_swa_base = swa_base

    def combine_c4_layer(
        self,
        c4_sparse_raw_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        topk = c4_sparse_raw_indices.shape[-1]
        if self.c4_combined_indices is None:
            device = self.seq_lens.device
            self.c4_combined_indices = torch.full(
                (self.num_qo_tokens, combined_topk_width(topk, self.swa_window_size)),
                -1,
                dtype=torch.int32,
                device=device,
            )
            self.c4_combined_lens = torch.zeros(
                self.num_qo_tokens, dtype=torch.int32, device=device
            )
        return combine_topk_swa_indices(
            topk_indices=c4_sparse_raw_indices,
            query_start_loc=self.query_start_loc,
            seq_lens=self.seq_lens,
            gather_lens=self.swa_gather_lens,
            compressed_base=self.c4_compressed_base,
            swa_base=self.c4_swa_base,
            window_size=self.swa_window_size,
            compress_ratio=4,
            topk=topk,
            out_indices=self.c4_combined_indices,
            out_lens=self.c4_combined_lens,
        )
