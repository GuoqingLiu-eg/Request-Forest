from __future__ import annotations

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.disaggregation.decode import DecodeReqToTokenPool
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.runtime_context import get_schedule
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter


class DSV4ReqToTokenTablesMixin:


    def _init_dsv4_tables(
        self,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        c128_page_size: int,
    ) -> None:
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )


        self._dsv4_allocator = None
        self.c128_page_size = c128_page_size

        group_tokens = 128 * c128_page_size
        with memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.req_to_c128_sidecar = torch.zeros(
                (
                    self._alloc_size,
                    max(1, (max_context_len + group_tokens - 1) // group_tokens),
                ),
                dtype=torch.int32,
                device=device,
            )

    def write_c128(self, indices, values: torch.Tensor) -> None:
        req_pool_idx, token_slice = indices
        page_size = self.c128_page_size
        first_group = (token_slice.start + page_size - 1) // page_size
        end_group = (token_slice.stop + page_size - 1) // page_size
        if first_group == end_group:
            return
        groups = torch.arange(first_group, end_group, device=values.device)
        pages = values[groups * page_size - token_slice.start] // page_size
        prefix_pages = self.req_to_c128_sidecar[req_pool_idx, :end_group].clone()
        prefix_pages[groups] = pages
        self._dsv4_allocator.replace_req_c128_prefix(req_pool_idx, prefix_pages, self)

    def register_dsv4_allocator(self, allocator) -> None:

        self._dsv4_allocator = allocator

    def set_c128_prefix_pages(self, req, page_ids: torch.Tensor) -> None:

        if req.req_pool_idx is None:
            req.c128_prefix_page_ids = page_ids
            return
        self._dsv4_allocator.replace_req_c128_prefix(
            int(req.req_pool_idx), page_ids, self
        )

    def alloc(self, reqs):
        fresh = [req.req_pool_idx is None for req in reqs]
        indices = super().alloc(reqs)
        if indices is None:
            return None
        for is_fresh, req, req_pool_idx in zip(fresh, reqs, indices):
            if is_fresh:
                self.req_to_c128_sidecar[int(req_pool_idx)].zero_()
            pages = getattr(req, "c128_prefix_page_ids", None)
            if pages is not None:
                self._dsv4_allocator.replace_req_c128_prefix(
                    int(req_pool_idx), pages, self
                )
                req.c128_prefix_page_ids = None
        return indices

    def _dsv4_free(self, req) -> None:


        if self._dsv4_allocator is not None:
            self._dsv4_allocator.free(req=req, req_to_token_pool=self)


class DSV4NPUReqToTokenPool(DSV4ReqToTokenTablesMixin, ReqToTokenPool):


    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
    ):
        super().__init__(size, max_context_len, device, enable_memory_saver)
        self._init_dsv4_tables(
            max_context_len,
            device,
            enable_memory_saver,
            get_schedule().c128_page_size,
        )

    def free(self, req):
        self._dsv4_free(req)
        super().free(req)


class DSV4NPUDecodeReqToTokenPool(DSV4ReqToTokenTablesMixin, DecodeReqToTokenPool):


    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        pre_alloc_size: int,
    ):
        super().__init__(
            size=size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
            pre_alloc_size=pre_alloc_size,
        )
        self._init_dsv4_tables(
            max_context_len,
            device,
            enable_memory_saver,
            get_schedule().c128_page_size,
        )

    def free(self, req):
        self._dsv4_free(req)
        super().free(req)
