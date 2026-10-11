# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from __future__ import annotations

import inspect
import logging
import os
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch
from torch.profiler import record_function

from sglang.kernels.ops.memory.virtual_slot import alloc_bind_inplace
from sglang.srt.environ import envs
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import (
    alloc_decode_kernel,
    alloc_extend_kernel,
)
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.unified_memory_pool import (
    UnifiedKVPool,
    UnifiedMLATokenToKVPool,
)
from sglang.srt.utils.common import get_num_new_pages, next_power_of_2

logger = logging.getLogger(__name__)


_SORT_FREE_LIST_AFTER_MERGE = envs.SGLANG_SORT_FREE_LIST_AFTER_MERGE.get()


import atexit
import signal
import time as _time_mod
import weakref

_LAZY_COMPACTION_STATS_ENABLED = envs.SGLANG_LOG_LAZY_COMPACTION_STATS.get()
_LAZY_COMPACTION_STATS_INTERVAL_SEC = float(
    envs.SGLANG_LOG_LAZY_COMPACTION_STATS_INTERVAL_SEC.get()
)

_STATS_INSTANCES: weakref.WeakSet[MultiEndedAllocator] = weakref.WeakSet()
_SIGNAL_HANDLERS_INSTALLED = False


def _emit_all_final_stats(reason: str) -> None:
    for inst in list(_STATS_INSTANCES):
        try:
            inst._emit_stats_final(reason=reason)
        except Exception:
            pass


def _signal_handler(signum, frame):
    try:
        sig_name = signal.Signals(signum).name
    except (ValueError, AttributeError):
        sig_name = str(signum)
    _emit_all_final_stats(reason=sig_name)
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def _install_signal_handlers_once() -> None:
    global _SIGNAL_HANDLERS_INSTALLED
    if _SIGNAL_HANDLERS_INSTALLED:
        return
    _SIGNAL_HANDLERS_INSTALLED = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            prev = signal.getsignal(sig)
            if prev in (signal.SIG_DFL, signal.SIG_IGN, None):
                signal.signal(sig, _signal_handler)
        except (ValueError, OSError):

            pass


class MultiEndedAllocator(BaseTokenToKVPoolAllocator):


    def __init__(
        self,
        *,
        kvcache,
        unified_buffer: UnifiedKVPool,
        sub_pool_name: str,
        device: str,
        is_id_owner: bool,
        page_size: int = 1,
        need_sort: bool = False,
        forward_stream: Optional[torch.cuda.Stream] = None,
        lazy_compaction: bool = False,
        kernel_page_multiplier: int = 1,
    ):
        spec = unified_buffer.spec(sub_pool_name)
        max_slots = unified_buffer.max_slots(sub_pool_name)
        super().__init__(
            size=max_slots,
            page_size=page_size,
            dtype=spec.get_dtype(),
            device=device,
            kvcache=kvcache,
            need_sort=need_sort,
        )
        self.unified_buffer = unified_buffer
        self.sub_pool_name = sub_pool_name
        self.spec = spec
        self.max_slots = max_slots
        self.grow_direction = spec.grow_direction
        self.entry_bytes = spec.entry_bytes()
        self.min_slot_index = unified_buffer.min_slot_index(sub_pool_name)
        self.is_id_owner = is_id_owner


        self.kernel_page_multiplier = kernel_page_multiplier

        self._zero_pages_on_alloc = isinstance(kvcache, UnifiedMLATokenToKVPool)


        self.forward_stream = forward_stream


        self.page_size = page_size
        self.num_pages = max_slots // page_size
        self.min_page_index = (self.min_slot_index + page_size - 1) // page_size
        self.entry_bytes_per_page = self.entry_bytes * page_size


        self.virtual_to_physical = torch.full(
            (self.num_pages + 1,),
            -1,
            dtype=torch.int64,
            device=device,
        )
        self.physical_to_virtual = torch.full(
            (self.num_pages + 1,),
            -1,
            dtype=torch.int64,
            device=device,
        )

        self.num_virtual_ids = self.num_pages

        self._peer: Optional[MultiEndedAllocator] = None


        self._inverse_history: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = (
            []
        )


        self.lazy_compaction = lazy_compaction
        self._free_phys_pages: torch.Tensor = torch.empty(
            0, dtype=torch.int64, device=device
        )


        self._pending_reuse: Dict[
            torch.cuda.Event,
            Tuple[List[int], torch.Tensor],
        ] = {}

        self._pending_reuse_pages_cpu: Set[int] = set()

        self._stats_n_free_lazy: int = 0
        self._stats_n_release_batch: int = 0
        self._stats_n_drain_calls: int = 0
        self._stats_n_drain_did_work: int = 0
        self._stats_n_drained_pages_total: int = 0
        self._stats_n_flush_calls: int = 0
        self._stats_n_flush_did_work: int = 0
        self._stats_n_flush_moves: int = 0
        self._stats_n_pages_absorbed: int = 0
        self._stats_peak_free_list_len: int = 0
        self._stats_peak_pending_pages: int = 0
        self._stats_n_emits: int = 0
        self._stats_last_emit_ts: float = _time_mod.monotonic()
        self._stats_final_emitted: bool = False
        if _LAZY_COMPACTION_STATS_ENABLED:
            atexit.register(self._emit_stats_final, reason="atexit")
            _STATS_INSTANCES.add(self)
            _install_signal_handlers_once()
        self.live_page_count = 0

        self.disagg_move_gate: Optional[Callable[[], bool]] = None
        self._latest_forward_done_event: Optional[torch.cuda.Event] = None


        self._inflight_forward: Optional[Tuple[torch.cuda.Event, torch.Tensor]] = None


        self._lazy_max_moves_per_call = int(
            os.environ.get("SGLANG_LAZY_COMPACTION_MAX_MOVES_PER_CALL", "4096")
        )

        self.clear()

        logger.info(
            "[unified-memory-pool] MultiEndedAllocator(%r) ready: grow=%s, max_slots=%d, "
            "min_slot_index=%d, page_size=%d, num_pages=%d, min_page_index=%d, "
            "entry_bytes=%d, entry_bytes_per_page=%d, is_id_owner=%s, "
            "initial_watermark_page=%d, allocatable_pages=%d",
            self.sub_pool_name,
            self.grow_direction,
            self.max_slots,
            self.min_slot_index,
            self.page_size,
            self.num_pages,
            self.min_page_index,
            self.entry_bytes,
            self.entry_bytes_per_page,
            self.is_id_owner,
            self.watermark_physical,
            self.num_pages - self.min_page_index,
        )


    def bind_peer(self, peer: MultiEndedAllocator) -> None:
        self._peer = peer

    @property
    def peer(self) -> Optional[MultiEndedAllocator]:
        return self._peer


    def clear(self) -> None:

        if self.grow_direction == "up":
            self.watermark_physical = self.min_page_index
        else:
            self.watermark_physical = self.num_pages - 1
        self.virtual_to_physical.fill_(-1)

        self.virtual_to_physical[0] = 0
        self.virtual_to_physical[-1] = -1
        self.physical_to_virtual.fill_(-1)
        self.physical_to_virtual[0] = 0
        self.physical_to_virtual[-1] = -1
        if self.is_id_owner:
            self.free_virtual_ids = torch.arange(
                self.min_page_index,
                self.num_pages,
                dtype=torch.int64,
                device=self.device,
            )
        else:
            self.free_virtual_ids = None
        self.is_not_in_free_group = True
        self.free_group: List[torch.Tensor] = []
        self._inverse_history.clear()
        self._free_phys_pages = torch.empty(0, dtype=torch.int64, device=self.device)
        self._pending_reuse.clear()
        self._pending_reuse_pages_cpu.clear()
        self.live_page_count = 0
        self._inflight_forward = None
        self._latest_forward_done_event = None

    def clear_inverse_history(self) -> None:
        self._inverse_history.clear()


    def _allocated_pages(self) -> int:

        if self.grow_direction == "up":
            return max(0, self.watermark_physical - self.min_page_index)
        return max(0, self.num_pages - 1 - self.watermark_physical)

    def allocated_count(self) -> int:

        if self.lazy_compaction:
            return self.live_page_count * self.page_size
        return self._allocated_pages() * self.page_size

    def is_slot_allocated(self, slot: int) -> bool:

        virt_page = slot // self.page_size
        if virt_page < 0 or virt_page >= self.num_pages:
            return False
        return int(self.virtual_to_physical[virt_page].item()) != -1

    def allocator_state_str(self) -> str:
        return (
            f"sub_pool={self.sub_pool_name!r}, grow_direction={self.grow_direction}, "
            f"is_id_owner={self.is_id_owner}, page_size={self.page_size}, "
            f"min_page_index={self.min_page_index}, "
            f"num_pages={self.num_pages}, "
            f"watermark_physical={self.watermark_physical}, "
            f"allocated_pages={self._allocated_pages()}"
        )

    def _byte_high_frontier(self) -> int:

        if self.grow_direction == "up":
            return self.watermark_physical * self.entry_bytes_per_page
        return self.num_pages * self.entry_bytes_per_page

    def _byte_low_frontier(self) -> int:

        if self.grow_direction == "up":
            return self.min_page_index * self.entry_bytes_per_page
        return (self.watermark_physical + 1) * self.entry_bytes_per_page

    def _current_gap_bytes(self) -> int:

        if self.grow_direction == "up":
            my_high = self._byte_high_frontier()
            peer_low = (
                self._peer._byte_low_frontier()
                if self._peer is not None
                else self.unified_buffer.total_bytes
            )
            return max(0, peer_low - my_high)
        my_low = self._byte_low_frontier()
        peer_high = self._peer._byte_high_frontier() if self._peer is not None else 0
        return max(0, my_low - peer_high)

    def _available_tokens(self, extra_gap_bytes: int = 0) -> int:

        gap_bytes = self._current_gap_bytes() + extra_gap_bytes
        pages_by_bytes = gap_bytes // self.entry_bytes_per_page
        pages_by_index_space = (
            self.num_pages - self.min_page_index - self._allocated_pages()
        )
        pages_extend = min(pages_by_bytes, pages_by_index_space)

        pages_drain = len(self._free_phys_pages) if self.lazy_compaction else 0
        return (pages_extend + pages_drain) * self.page_size

    def available_size(self) -> int:

        return self._available_tokens()

    def _peer_drainable_hole_bytes(self) -> int:

        peer = self._peer
        if peer is None or not peer.lazy_compaction:
            return 0
        if peer.disagg_move_gate is not None and not peer.disagg_move_gate():


            return 0
        return len(peer._free_phys_pages) * peer.entry_bytes_per_page

    def schedulable_available_size(self) -> int:

        return self._available_tokens(extra_gap_bytes=self._peer_drainable_hole_bytes())

    def _flush_peer_for_alloc(self, need_tokens: int) -> bool:

        if not (self.lazy_compaction and self._peer is not None):
            return False
        self._peer._flush(urgent=True)
        return need_tokens <= self.available_size()


    def take_physical(self, need_size: int) -> Optional[torch.Tensor]:

        with record_function("MultiEndedAlloc.take_physical"):
            if need_size <= 0:
                return torch.empty(0, dtype=torch.int64, device=self.device)
            assert need_size % self.page_size == 0, (
                f"take_physical: need_size={need_size} must be a multiple of "
                f"page_size={self.page_size}"
            )
            num_pages = need_size // self.page_size

            if not self.lazy_compaction:
                return self._take_physical_eager(num_pages)


            n_drain = min(num_pages, int(self._free_phys_pages.shape[0]))
            need_more = num_pages - n_drain


            if need_more > 0:
                if not self._extend_watermark(need_more):
                    return None

            if n_drain > 0:
                if _SORT_FREE_LIST_AFTER_MERGE:
                    if self.grow_direction == "up":
                        drained_t = self._free_phys_pages[:n_drain]
                        self._free_phys_pages = self._free_phys_pages[n_drain:]
                    else:
                        drained_t = self._free_phys_pages[-n_drain:].flip(0)
                        self._free_phys_pages = self._free_phys_pages[:-n_drain]
                else:
                    drained_t = self._free_phys_pages[:n_drain]
                    self._free_phys_pages = self._free_phys_pages[n_drain:]
            else:
                drained_t = None

            self.live_page_count += num_pages

            if drained_t is None:
                return self._take_physical_arange(num_pages)


            if need_more == 0:
                return drained_t.clone()


            if self.grow_direction == "up":
                new_wm = self.watermark_physical
                extended_t = torch.arange(
                    new_wm - need_more,
                    new_wm,
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                new_wm = self.watermark_physical
                extended_t = torch.arange(
                    new_wm + need_more,
                    new_wm,
                    -1,
                    dtype=torch.int64,
                    device=self.device,
                )
            return torch.cat([drained_t, extended_t])

    def _take_physical_eager(self, num_pages: int) -> Optional[torch.Tensor]:

        if self.grow_direction == "up":
            start = self.watermark_physical
            end_exclusive = start + num_pages
            if end_exclusive > self.num_pages:
                return None
            phys_pages = torch.arange(
                start, end_exclusive, dtype=torch.int64, device=self.device
            )
            self.watermark_physical = end_exclusive
            return phys_pages
        else:
            end = self.watermark_physical
            start = end - num_pages + 1
            if start < self.min_page_index:
                return None
            phys_pages = torch.arange(
                start, end + 1, dtype=torch.int64, device=self.device
            )
            self.watermark_physical -= num_pages
            return phys_pages

    def _extend_watermark(self, num_pages: int) -> bool:

        if self.grow_direction == "up":
            new_wm = self.watermark_physical + num_pages
            if new_wm > self.num_pages:
                return False

            if self._peer is not None:
                peer_low_pages = (
                    self._peer._byte_low_frontier() // self.entry_bytes_per_page
                )
                if new_wm > peer_low_pages:
                    return False
            self.watermark_physical = new_wm
        else:
            new_wm = self.watermark_physical - num_pages
            if new_wm < self.min_page_index - 1:
                return False


            if self._peer is not None:
                peer_high_pages = (
                    self._peer._byte_high_frontier() // self.entry_bytes_per_page
                )
                if new_wm + 1 < peer_high_pages:
                    return False
            self.watermark_physical = new_wm
        return True

    def _take_physical_arange(self, num_pages: int) -> torch.Tensor:

        if self.grow_direction == "up":
            return torch.arange(
                self.watermark_physical - num_pages,
                self.watermark_physical,
                dtype=torch.int64,
                device=self.device,
            )
        return torch.arange(
            self.watermark_physical + 1,
            self.watermark_physical + num_pages + 1,
            dtype=torch.int64,
            device=self.device,
        )

    def take_physical_pages(self, num_pages: int) -> Optional[torch.Tensor]:

        with record_function("MultiEndedAlloc.take_physical_pages"):
            return self.take_physical(num_pages * self.page_size)

    def bind(self, virtual_ids: torch.Tensor, physical_ids: torch.Tensor) -> None:

        with record_function("MultiEndedAlloc.bind"):
            self.virtual_to_physical[virtual_ids] = physical_ids
            self.physical_to_virtual[physical_ids] = virtual_ids

    def bind_pages(
        self, virtual_pages: torch.Tensor, physical_pages: torch.Tensor
    ) -> None:

        with record_function("MultiEndedAlloc.bind_pages"):
            self.bind(virtual_pages, physical_pages)


    def _alloc_bind_fast_or_slow(
        self, v_pages: torch.Tensor, N: int
    ) -> Optional[torch.Tensor]:

        with record_function("MultiEndedAlloc._alloc_bind_fast_or_slow"):
            if N == 0:
                return torch.empty(0, dtype=torch.int64, device=self.device)


            if not self.lazy_compaction or self._free_phys_pages.numel() == 0:
                start_wm = self.watermark_physical


                if self.lazy_compaction:
                    if not self._extend_watermark(N):
                        return None
                else:
                    if self.grow_direction == "up":
                        new_wm = start_wm + N
                        if new_wm > self.num_pages:
                            return None
                        self.watermark_physical = new_wm
                    else:
                        new_wm = start_wm - N
                        if new_wm < self.min_page_index - 1:
                            return None
                        self.watermark_physical = new_wm


                if self.grow_direction == "up":
                    start_phys = start_wm
                else:
                    start_phys = start_wm - N + 1

                phys_pages = alloc_bind_inplace(
                    v_pages,
                    self.virtual_to_physical,
                    self.physical_to_virtual,
                    start_phys,
                )

                if self.lazy_compaction:
                    self.live_page_count += N
                self._maybe_zero_pages(phys_pages)
                return phys_pages


            phys_pages = self.take_physical_pages(N)
            if phys_pages is None:
                return None
            self.bind(v_pages, phys_pages)
            self._maybe_zero_pages(phys_pages)
            return phys_pages

    def _maybe_zero_pages(self, phys_pages: torch.Tensor) -> None:

        if not self._zero_pages_on_alloc or phys_pages.numel() == 0:
            return
        with record_function("MultiEndedAlloc._maybe_zero_pages"):
            self._kvcache.zero_physical_pages(phys_pages)


    def translate_kv_loc(
        self,
        virt_tokens: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if out is not None:
            assert out.dtype == torch.int64, (
                f"translate_kv_loc: out= dtype must be int64 (matches v2p), "
                f"got {out.dtype}"
            )
            assert out.shape == virt_tokens.shape, (
                f"translate_kv_loc: out= shape {tuple(out.shape)} must match "
                f"virt_tokens shape {tuple(virt_tokens.shape)}"
            )
        with record_function("MultiEndedAlloc.translate_kv_loc"):
            return self._translate_kv_loc_impl(virt_tokens, out)

    def _translate_kv_loc_impl(
        self,
        virt_tokens: torch.Tensor,
        out: Optional[torch.Tensor],
    ) -> torch.Tensor:


        if self.page_size == 1:
            if out is not None:


                tmp = torch.index_select(self.virtual_to_physical, 0, virt_tokens)
                tmp = torch.clamp_min(tmp, 0)
                out.copy_(tmp)
                return out
            result = torch.index_select(self.virtual_to_physical, 0, virt_tokens)
            return torch.clamp_min(result, 0)


        virt_pages = virt_tokens // self.page_size
        offsets = virt_tokens % self.page_size
        if out is not None:
            torch.index_select(self.virtual_to_physical, 0, virt_pages, out=out)
            out.mul_(self.page_size)
            out.add_(offsets)
            out.clamp_(min=0)
            return out
        phys_pages = self.virtual_to_physical[virt_pages]
        result = phys_pages * self.page_size + offsets
        return torch.clamp_min(result, 0)

    def translate_kv_loc_dense(
        self,
        virt_tokens: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if self.kernel_page_multiplier == 1:
            return self.translate_kv_loc(virt_tokens, out=out)
        if out is not None:
            assert out.dtype == torch.int64, (
                f"translate_kv_loc_dense: out= dtype must be int64 (matches v2p), "
                f"got {out.dtype}"
            )
            assert out.shape == virt_tokens.shape, (
                f"translate_kv_loc_dense: out= shape {tuple(out.shape)} must "
                f"match virt_tokens shape {tuple(virt_tokens.shape)}"
            )
        with record_function("MultiEndedAlloc.translate_kv_loc_dense"):
            dense_page_stride = self.page_size * self.kernel_page_multiplier
            if self.page_size == 1:

                if out is not None:
                    tmp = torch.index_select(self.virtual_to_physical, 0, virt_tokens)
                    tmp = torch.clamp_min(tmp * dense_page_stride, 0)
                    out.copy_(tmp)
                    return out
                result = torch.index_select(self.virtual_to_physical, 0, virt_tokens)
                return torch.clamp_min(result * dense_page_stride, 0)
            virt_pages = virt_tokens // self.page_size
            offsets = virt_tokens % self.page_size
            if out is not None:
                torch.index_select(self.virtual_to_physical, 0, virt_pages, out=out)
                out.mul_(dense_page_stride)
                out.add_(offsets)

                out.clamp_(min=0)
                return out
            phys_pages = self.virtual_to_physical[virt_pages]
            result = phys_pages * dense_page_stride + offsets
            return torch.clamp_min(result, 0)


    def alloc(self, need_size: int) -> Optional[torch.Tensor]:

        with record_function("MultiEndedAlloc.alloc"):
            assert self.is_id_owner, (
                f"MultiEndedAllocator({self.sub_pool_name!r}).alloc called on a "
                "non-id-owner allocator; use alloc_with_virtual instead"
            )
            if need_size <= 0:
                return torch.empty(0, dtype=torch.int64, device=self.device)
            assert need_size % self.page_size == 0, (
                f"MultiEndedAllocator({self.sub_pool_name!r}).alloc: need_size="
                f"{need_size} must be a multiple of page_size={self.page_size}"
            )
            if need_size > self.available_size():


                if not self._flush_peer_for_alloc(need_size):
                    return None
            num_pages = need_size // self.page_size
            v_pages = self.free_virtual_ids[:num_pages]
            self.free_virtual_ids = self.free_virtual_ids[num_pages:]
            phys_pages = self._alloc_bind_fast_or_slow(v_pages, num_pages)
            if phys_pages is None:
                self.free_virtual_ids = torch.cat([v_pages, self.free_virtual_ids])
                return None
            if self.page_size == 1:
                return v_pages

            return (
                v_pages[:, None] * self.page_size
                + torch.arange(self.page_size, device=self.device)
            ).reshape(-1)

    def alloc_with_virtual(self, virtual_pages: torch.Tensor) -> None:

        with record_function("MultiEndedAlloc.alloc_with_virtual"):
            if virtual_pages.numel() == 0:
                return
            phys_pages = self._alloc_bind_fast_or_slow(
                virtual_pages, int(virtual_pages.numel())
            )
            assert phys_pages is not None, (
                f"MultiEndedAllocator({self.sub_pool_name!r}).alloc_with_virtual: out of "
                "physical room (the composite's byte-budget check should have caught this)"
            )


    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        num_new_pages: Optional[int] = None,
    ) -> Optional[torch.Tensor]:

        with record_function("MultiEndedAlloc.alloc_extend"):
            assert (
                self.is_id_owner
            ), f"alloc_extend on a non-id-owner allocator ({self.sub_pool_name!r})"
            if num_new_pages is None:
                num_new_pages = get_num_new_pages(
                    seq_lens=seq_lens_cpu,
                    page_size=self.page_size,
                    prefix_lens=prefix_lens_cpu,
                )
            if num_new_pages > len(self.free_virtual_ids):
                return None


            need_tokens = num_new_pages * self.page_size
            if need_tokens > self.available_size():
                if not self._flush_peer_for_alloc(need_tokens):
                    return None
            bs = len(prefix_lens)
            if self.need_sort and extend_num_tokens // self.page_size + bs + 1 > len(
                self.free_virtual_ids
            ):
                self.merge_and_sort_free()


            if num_new_pages > 0:
                new_virtual_pages = self.free_virtual_ids[:num_new_pages].clone()
            else:
                new_virtual_pages = None

            out_indices = torch.empty(
                (extend_num_tokens,), dtype=torch.int64, device=self.device
            )


            with record_function("MultiEndedAlloc.alloc_extend.kernel"):
                alloc_extend_kernel[(bs,)](
                    prefix_lens,
                    seq_lens,
                    last_loc,
                    self.free_virtual_ids,
                    out_indices,
                    next_power_of_2(bs),
                    self.page_size,
                )


            if new_virtual_pages is not None:
                phys_pages = self._alloc_bind_fast_or_slow(
                    new_virtual_pages, num_new_pages
                )
                if phys_pages is None:
                    return None

            self.free_virtual_ids = self.free_virtual_ids[num_new_pages:]
            return out_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ) -> Optional[torch.Tensor]:

        with record_function("MultiEndedAlloc.alloc_decode"):
            assert (
                self.is_id_owner
            ), f"alloc_decode on a non-id-owner allocator ({self.sub_pool_name!r})"
            bs = len(seq_lens)


            num_new_pages = get_num_new_pages(
                seq_lens=seq_lens_cpu, page_size=self.page_size, decode=True
            )
            if num_new_pages > len(self.free_virtual_ids):
                return None

            need_tokens = num_new_pages * self.page_size
            if need_tokens > self.available_size():
                if not self._flush_peer_for_alloc(need_tokens):
                    return None
            if self.need_sort and bs > len(self.free_virtual_ids):
                self.merge_and_sort_free()


            if num_new_pages > 0:
                new_virtual_pages = self.free_virtual_ids[:num_new_pages].clone()
            else:
                new_virtual_pages = None

            out_indices = torch.empty((bs,), dtype=torch.int64, device=self.device)
            with record_function("MultiEndedAlloc.alloc_decode.kernel"):
                alloc_decode_kernel[(bs,)](
                    seq_lens,
                    last_loc,
                    self.free_virtual_ids,
                    out_indices,
                    next_power_of_2(bs),
                    self.page_size,
                )

            if new_virtual_pages is not None:
                phys_pages = self._alloc_bind_fast_or_slow(
                    new_virtual_pages, num_new_pages
                )
                if phys_pages is None:
                    return None

            self.free_virtual_ids = self.free_virtual_ids[num_new_pages:]
            return out_indices


    def free(self, free_index: torch.Tensor) -> None:

        with record_function("MultiEndedAlloc.free"):
            if free_index is None or free_index.numel() == 0:
                return
            if not self.is_not_in_free_group:
                self.free_group.append(self._copy_for_free_group(free_index))
                return
            if self.lazy_compaction:
                self._free_lazy(free_index)
                return


            if self.forward_stream is not None:
                with record_function("MultiEndedAlloc.free.wait_stream"):
                    torch.cuda.current_stream().wait_stream(self.forward_stream)
            with record_function("MultiEndedAlloc.free.v2p_lookup"):
                free_v_pages = torch.unique(
                    free_index.detach().to(torch.int64) // self.page_size
                )
                freed_p_pages = self.virtual_to_physical[free_v_pages]
            with record_function("MultiEndedAlloc.free.sync_check"):

                if bool((freed_p_pages < 0).any().item()):
                    self._raise_stale_slot_assertion(
                        free_v=free_v_pages, freed_p=freed_p_pages
                    )
            self.virtual_to_physical[free_v_pages] = -1
            if self.is_id_owner:
                self.free_virtual_ids = torch.cat([self.free_virtual_ids, free_v_pages])
            self._compact_pending(freed_p_pages)

    def _free_lazy(self, free_index: torch.Tensor) -> None:

        self._stats_n_free_lazy += 1
        with record_function("MultiEndedAlloc._free_lazy"):
            with record_function("MultiEndedAlloc._free_lazy.v2p_lookup"):
                free_v_pages_raw = free_index.detach().to(torch.int64)
                if self.page_size == 1:
                    free_v_pages = free_v_pages_raw
                else:
                    free_v_pages = torch.unique(free_v_pages_raw // self.page_size)
                freed_p_pages = self.virtual_to_physical[free_v_pages]


            self.virtual_to_physical[free_v_pages] = -1
            self.physical_to_virtual[freed_p_pages] = -1
            if self.is_id_owner:
                self.free_virtual_ids = torch.cat([self.free_virtual_ids, free_v_pages])
            self._free_phys_pages = torch.cat([self._free_phys_pages, freed_p_pages])
            if _SORT_FREE_LIST_AFTER_MERGE:
                self._free_phys_pages, _ = torch.sort(self._free_phys_pages)
            self.live_page_count -= int(freed_p_pages.shape[0])

    def _release_phys_pages_batch(self, pages: torch.Tensor) -> None:

        if pages.numel() == 0:
            return
        self._stats_n_release_batch += 1
        with record_function("MultiEndedAlloc._release_phys_pages_batch"):
            self._free_phys_pages = torch.cat([self._free_phys_pages, pages])
            if _SORT_FREE_LIST_AFTER_MERGE:
                self._free_phys_pages, _ = torch.sort(self._free_phys_pages)

    def _compact_pending(self, freed_physical_pages: torch.Tensor) -> None:

        with record_function("MultiEndedAlloc._compact_pending"):
            self._compact_pending_impl(freed_physical_pages)

    def _compact_pending_impl(self, freed_physical_pages: torch.Tensor) -> None:
        assert self.disagg_move_gate is None, (
            f"_compact_pending({self.sub_pool_name!r}): eager compaction ran with "
            "a PD-disaggregation move gate installed; PD requires lazy_compaction."
        )
        freed_set = set(int(x) for x in freed_physical_pages.tolist())
        if not freed_set:
            return
        K = len(freed_set)
        if self.grow_direction == "up":

            old_wm = self.watermark_physical
            new_wm = old_wm - K
            assert new_wm >= self.min_page_index, (
                f"_compact_pending({self.sub_pool_name!r}): freeing {K} pages "
                f"would push the watermark below min_page_index "
                f"({new_wm} < {self.min_page_index})"
            )
            assert all(self.min_page_index <= h < old_wm for h in freed_set), (
                f"_compact_pending({self.sub_pool_name!r}): freed physical pages "
                f"{sorted(freed_set)} not all within allocated range "
                f"[{self.min_page_index}, {old_wm})"
            )

            src_list = [s for s in range(new_wm, old_wm) if s not in freed_set]
            dst_list = sorted(h for h in freed_set if h < new_wm)
            self.watermark_physical = new_wm
            vacated_lo, vacated_hi = new_wm, old_wm
        else:

            old_wm = self.watermark_physical
            new_wm = old_wm + K
            assert new_wm <= self.num_pages - 1, (
                f"_compact_pending({self.sub_pool_name!r}): freeing {K} pages "
                f"would push the watermark above num_pages "
                f"({new_wm} > {self.num_pages - 1})"
            )
            assert all(old_wm < h < self.num_pages for h in freed_set), (
                f"_compact_pending({self.sub_pool_name!r}): freed physical pages "
                f"{sorted(freed_set)} not all within allocated range "
                f"({old_wm}, {self.num_pages})"
            )

            src_list = [s for s in range(old_wm + 1, new_wm + 1) if s not in freed_set]
            dst_list = sorted(h for h in freed_set if h > new_wm)
            self.watermark_physical = new_wm
            vacated_lo, vacated_hi = old_wm + 1, new_wm + 1

        assert len(src_list) == len(dst_list), (
            f"_compact_pending({self.sub_pool_name!r}): {len(src_list)} survivors vs "
            f"{len(dst_list)} holes — corrupt allocator state"
        )

        if src_list:
            src_pages = torch.tensor(src_list, dtype=torch.int64, device=self.device)
            dst_pages = torch.tensor(dst_list, dtype=torch.int64, device=self.device)
            v_moved = self.physical_to_virtual[
                src_pages
            ].clone()


            if self.page_size == 1:
                src_t, dst_t = src_pages, dst_pages
            else:
                offsets = torch.arange(
                    self.page_size, dtype=torch.int64, device=self.device
                )
                src_t = (src_pages[:, None] * self.page_size + offsets).reshape(-1)
                dst_t = (dst_pages[:, None] * self.page_size + offsets).reshape(-1)


            move_fn = getattr(self._kvcache, "move_kv_cache", None)
            if move_fn is not None:
                move_fn(dst_t, src_t)
            else:
                copy_phys = getattr(self._kvcache, "_copy_from_physical", None)
                assert copy_phys is not None, (
                    f"sub-pool {self.sub_pool_name!r} supports neither move_kv_cache "
                    "nor _copy_from_physical"
                )
                copy_phys(src_t, dst_t)

            self.physical_to_virtual[vacated_lo:vacated_hi] = -1
            self.virtual_to_physical[v_moved] = dst_pages
            self.physical_to_virtual[dst_pages] = v_moved
            self._inverse_history.append((src_pages, dst_pages, v_moved))
        else:
            self.physical_to_virtual[vacated_lo:vacated_hi] = -1


    def set_latest_forward_done_event(self, event: Optional[torch.cuda.Event]) -> None:

        with record_function("MultiEndedAlloc.set_latest_forward_done_event"):
            self._latest_forward_done_event = event

    def set_inflight_forward(
        self,
        forward_done: torch.cuda.Event,
        out_cache_loc_virtual: Optional[torch.Tensor],
    ) -> None:

        with record_function("MultiEndedAlloc.set_inflight_forward"):
            if not self.lazy_compaction:
                return
            if out_cache_loc_virtual is None or out_cache_loc_virtual.numel() == 0:


                self._inflight_forward = None
                return
            self._inflight_forward = (forward_done, out_cache_loc_virtual)

    def _materialize_inflight_write_set(self) -> Optional[Set[int]]:

        inflight = self._inflight_forward
        if inflight is None:
            return None
        event, oclv = inflight


        if event.query():
            self._inflight_forward = None
            return None

        with record_function("MultiEndedAlloc._materialize_inflight_write_set"):
            phys_tokens = self.translate_kv_loc(oclv)
            if self.page_size > 1:
                phys_pages = (phys_tokens // self.page_size).unique()
            else:
                phys_pages = phys_tokens
            return set(phys_pages.tolist())

    def _maybe_emit_stats(self) -> None:

        if not _LAZY_COMPACTION_STATS_ENABLED:
            return
        now = _time_mod.monotonic()
        if now - self._stats_last_emit_ts < _LAZY_COMPACTION_STATS_INTERVAL_SEC:
            return
        self._stats_last_emit_ts = now
        self._stats_n_emits += 1
        cur_holes = int(self._free_phys_pages.shape[0])
        cur_pending = len(self._pending_reuse_pages_cpu)
        self._stats_peak_free_list_len = max(self._stats_peak_free_list_len, cur_holes)
        self._stats_peak_pending_pages = max(
            self._stats_peak_pending_pages, cur_pending
        )
        sort_tag = "ON" if _SORT_FREE_LIST_AFTER_MERGE else "OFF"
        logger.info(
            f"[lazy-stats sub={self.sub_pool_name!r}] "
            f"free_lazy={self._stats_n_free_lazy} "
            f"flush={self._stats_n_flush_calls} "
            f"(work={self._stats_n_flush_did_work} "
            f"moves={self._stats_n_flush_moves} "
            f"abs={self._stats_n_pages_absorbed}) "
            f"drain={self._stats_n_drain_did_work}/{self._stats_n_drain_calls} "
            f"sort={sort_tag} "
            f"peak_holes={self._stats_peak_free_list_len} "
            f"peak_pending={self._stats_peak_pending_pages} "
            f"cur_holes={cur_holes} cur_pending={cur_pending} "
            f"live={self.live_page_count} wm={self.watermark_physical}"
        )

    def _emit_stats_final(self, reason: str = "exit") -> None:

        if not _LAZY_COMPACTION_STATS_ENABLED:
            return
        if self._stats_final_emitted:
            return
        try:
            cur_holes = int(self._free_phys_pages.shape[0])
            cur_pending = len(self._pending_reuse_pages_cpu)
            self._stats_peak_free_list_len = max(
                self._stats_peak_free_list_len, cur_holes
            )
            self._stats_peak_pending_pages = max(
                self._stats_peak_pending_pages, cur_pending
            )
            sort_tag = "ON" if _SORT_FREE_LIST_AFTER_MERGE else "OFF"
            self._stats_final_emitted = True
            logger.info(
                f"[lazy-stats FINAL sub={self.sub_pool_name!r} reason={reason}] "
                f"free_lazy={self._stats_n_free_lazy} "
                f"flush={self._stats_n_flush_calls} "
                f"(work={self._stats_n_flush_did_work} "
                f"moves={self._stats_n_flush_moves} "
                f"abs={self._stats_n_pages_absorbed}) "
                f"drain={self._stats_n_drain_did_work}/{self._stats_n_drain_calls} "
                f"sort={sort_tag} "
                f"peak_holes={self._stats_peak_free_list_len} "
                f"peak_pending={self._stats_peak_pending_pages} "
                f"cur_holes={cur_holes} cur_pending={cur_pending} "
                f"live={self.live_page_count} wm={self.watermark_physical} "
                f"n_emits={self._stats_n_emits}"
            )
        except Exception:
            pass

    def _drain_pending_reuse(self, *, urgent: bool) -> None:

        self._stats_n_drain_calls += 1
        if not self._pending_reuse:
            return
        with record_function("MultiEndedAlloc._drain_pending_reuse"):
            ready_tensors: List[torch.Tensor] = []
            ready_entries: List[Tuple[torch.cuda.Event, List[int]]] = []
            for event, (cpu_list, gpu_tensor) in self._pending_reuse.items():
                if event is None or event.query():
                    ready_tensors.append(gpu_tensor)
                    ready_entries.append((event, cpu_list))
                elif urgent:
                    torch.cuda.current_stream().wait_event(event)
                    ready_tensors.append(gpu_tensor)
                    ready_entries.append((event, cpu_list))

            for event, cpu_list in ready_entries:
                del self._pending_reuse[event]
                self._pending_reuse_pages_cpu.difference_update(cpu_list)

            if ready_tensors:
                self._free_phys_pages = torch.cat(
                    [self._free_phys_pages] + ready_tensors
                )
                self._stats_n_drain_did_work += 1
                self._stats_n_drained_pages_total += sum(
                    t.numel() for t in ready_tensors
                )
                if _SORT_FREE_LIST_AFTER_MERGE:
                    self._free_phys_pages, _ = torch.sort(self._free_phys_pages)

    def maybe_drain_pending_reuse(self) -> None:

        if not self.lazy_compaction:
            return
        if not self._pending_reuse:
            return
        self._drain_pending_reuse(urgent=False)

    def _topmost_survivor(
        self,
        start_hint: Optional[int] = None,
        *,
        holes_cpu: Optional[List[int]] = None,
        j_in: Optional[int] = None,
    ) -> Tuple[Optional[int], Optional[int]]:

        if holes_cpu is None:
            holes_cpu = self._free_phys_pages.tolist()
        if self.grow_direction == "up":
            if start_hint is None or start_hint >= self.watermark_physical:
                p = self.watermark_physical - 1
            else:
                p = start_hint
            j = j_in if j_in is not None else len(holes_cpu) - 1
            while p >= self.min_page_index:
                while j >= 0 and holes_cpu[j] > p:
                    j -= 1
                is_hole = j >= 0 and holes_cpu[j] == p
                if is_hole or p in self._pending_reuse_pages_cpu:
                    if is_hole:
                        j -= 1
                    p -= 1
                    continue
                return p, j
            return None, j
        else:
            if start_hint is None or start_hint <= self.watermark_physical:
                p = self.watermark_physical + 1
            else:
                p = start_hint
            j = j_in if j_in is not None else 0
            while p < self.num_pages:
                while j < len(holes_cpu) and holes_cpu[j] < p:
                    j += 1
                is_hole = j < len(holes_cpu) and holes_cpu[j] == p
                if is_hole or p in self._pending_reuse_pages_cpu:
                    if is_hole:
                        j += 1
                    p += 1
                    continue
                return p, j
            return None, j

    def _absorb_boundary_holes(self, all_cpu: List[int]) -> Tuple[int, List[int]]:

        M = len(all_cpu)
        wm = self.watermark_physical
        n = 0
        if self.grow_direction == "up":
            while n < M and all_cpu[M - 1 - n] == wm - 1 - n:
                n += 1
            new_wm = wm - n
            holes_cpu = all_cpu[: M - n]
            self._free_phys_pages = self._free_phys_pages[: M - n]
        else:
            while n < M and all_cpu[n] == wm + 1 + n:
                n += 1
            new_wm = wm + n
            holes_cpu = all_cpu[n:]
            self._free_phys_pages = self._free_phys_pages[n:]
        self.watermark_physical = new_wm
        self._stats_n_pages_absorbed += n
        return new_wm, holes_cpu

    def _settle_inflight_forward(self) -> None:

        ev = self._latest_forward_done_event
        if ev is not None:
            torch.cuda.current_stream().wait_event(ev)
            self._inflight_forward = None

    def _flush(self, *, urgent: bool) -> int:

        if not self.lazy_compaction:
            return 0
        if self.disagg_move_gate is not None and not self.disagg_move_gate():

            return 0
        self._stats_n_flush_calls += 1
        with record_function("MultiEndedAlloc._flush"):
            self._drain_pending_reuse(urgent=urgent)


            if not _SORT_FREE_LIST_AFTER_MERGE and self._free_phys_pages.numel() > 1:
                self._free_phys_pages, _ = torch.sort(self._free_phys_pages)

            all_cpu = self._free_phys_pages.tolist()


            new_wm, holes_cpu = self._absorb_boundary_holes(all_cpu)

            latest_event = self._latest_forward_done_event


            single_pass_absorb = urgent and len(holes_cpu) > 0
            if single_pass_absorb:
                self._settle_inflight_forward()
                latest_event = None


            write_set: Optional[Set[int]] = set() if single_pass_absorb else None

            srcs: List[int] = []
            dsts: List[int] = []


            released_fired: List[torch.Tensor] = []

            cursor: Optional[int] = None
            j_cursor: Optional[int] = None


            if self.grow_direction == "up":
                dst_cursor = 0
            else:
                dst_cursor = len(holes_cpu) - 1
            n_dst_consumed = 0

            move_cap = self._lazy_max_moves_per_call if not urgent else None

            n_moves = 0
            while n_dst_consumed < len(holes_cpu):
                src, j_cursor = self._topmost_survivor(
                    start_hint=cursor,
                    holes_cpu=holes_cpu,
                    j_in=j_cursor,
                )
                if src is None:
                    break


                if write_set is None:
                    materialized = self._materialize_inflight_write_set()
                    write_set = materialized if materialized is not None else set()
                if write_set and src in write_set:
                    if urgent:


                        self._commit_move_batch(
                            srcs, dsts, latest_event, released_fired
                        )
                        n_moves += len(srcs)
                        srcs.clear()
                        dsts.clear()
                        inflight = self._inflight_forward
                        if inflight is not None:
                            torch.cuda.current_stream().wait_event(inflight[0])
                            self._inflight_forward = None
                        write_set = set()
                        latest_event = None


                        continue
                    else:
                        break


                dst = holes_cpu[dst_cursor]


                if (self.grow_direction == "up" and src < dst) or (
                    self.grow_direction == "down" and src > dst
                ):
                    break
                if self.grow_direction == "up":
                    dst_cursor += 1
                else:
                    dst_cursor -= 1
                n_dst_consumed += 1

                srcs.append(src)
                dsts.append(dst)


                if self.grow_direction == "up":
                    cursor = src - 1
                else:
                    cursor = src + 1

                if move_cap is not None and len(srcs) >= move_cap:
                    break

            self._commit_move_batch(srcs, dsts, latest_event, released_fired)
            n_moves += len(srcs)

            if single_pass_absorb:


                n_reclaimed = len(holes_cpu)
                if self.grow_direction == "up":
                    self.watermark_physical = new_wm - n_reclaimed
                else:
                    self.watermark_physical = new_wm + n_reclaimed
                self._stats_n_pages_absorbed += n_reclaimed
                self._free_phys_pages = self._free_phys_pages[:0]
            else:


                if n_dst_consumed > 0:
                    if self.grow_direction == "up":
                        self._free_phys_pages = self._free_phys_pages[n_dst_consumed:]
                    else:
                        self._free_phys_pages = self._free_phys_pages[:-n_dst_consumed]
                if released_fired:
                    self._release_phys_pages_batch(
                        released_fired[0]
                        if len(released_fired) == 1
                        else torch.cat(released_fired)
                    )
            if n_moves > 0:
                self._stats_n_flush_did_work += 1
                self._stats_n_flush_moves += n_moves
            self._maybe_emit_stats()
            return n_moves

    def _commit_move_batch(
        self,
        srcs: List[int],
        dsts: List[int],
        latest_event: Optional[torch.cuda.Event],
        released_fired: List[torch.Tensor],
    ) -> None:

        if not srcs:
            return
        with record_function("MultiEndedAlloc._commit_move_batch"):
            src_pages_t = torch.tensor(srcs, dtype=torch.int64, device=self.device)
            dst_pages_t = torch.tensor(dsts, dtype=torch.int64, device=self.device)
            v_moveds_t = self.physical_to_virtual[src_pages_t]
            torch._assert_async(
                (v_moveds_t >= 0).all(),
                "invalid p2v mapping in MultiEndedAllocator._flush",
            )

            if self.page_size == 1:
                src_t, dst_t = src_pages_t, dst_pages_t
            else:
                offsets = torch.arange(
                    self.page_size,
                    dtype=torch.int64,
                    device=self.device,
                )
                src_t = (src_pages_t[:, None] * self.page_size + offsets).reshape(-1)
                dst_t = (dst_pages_t[:, None] * self.page_size + offsets).reshape(-1)
            move_fn = getattr(self._kvcache, "move_kv_cache", None)
            if move_fn is not None:
                move_fn(dst_t, src_t)
            else:
                copy_phys = getattr(self._kvcache, "_copy_from_physical", None)
                assert copy_phys is not None, (
                    f"sub-pool {self.sub_pool_name!r} supports neither "
                    "move_kv_cache nor _copy_from_physical"
                )
                copy_phys(src_t, dst_t)

            self.virtual_to_physical[v_moveds_t] = dst_pages_t
            self.physical_to_virtual[dst_pages_t] = v_moveds_t
            self.physical_to_virtual[src_pages_t] = -1
            self._inverse_history.append((src_pages_t, dst_pages_t, v_moveds_t))


            event_fired = latest_event is None or latest_event.query()
            if event_fired:
                released_fired.append(src_pages_t)
            else:
                srcs_copy: List[int] = list(srcs)
                self._pending_reuse[latest_event] = (srcs_copy, src_pages_t)
                self._pending_reuse_pages_cpu.update(srcs_copy)

    def flush_opportunistic(self) -> int:

        with record_function("MultiEndedAlloc.flush_opportunistic"):
            if not self.lazy_compaction:
                return 0
            if self._free_phys_pages.numel() == 0 and not self._pending_reuse:
                return 0
            return self._flush(urgent=False)

    def _raise_stale_slot_assertion(self, *, free_v, freed_p) -> None:
        bad = free_v[freed_p < 0].tolist()
        frames = inspect.stack()[1:9]
        callers = " <- ".join(f"{f.filename.split('/')[-1]}:{f.lineno}" for f in frames)
        raise AssertionError(
            f"MultiEndedAllocator({self.sub_pool_name!r}).free: virtual id(s) {bad} have "
            f"virtual_to_physical == -1 (double-free or never-allocated). "
            f"State: {self.allocator_state_str()}. free_index unique={free_v.tolist()}. "
            f"recent _inverse_history (last 3): "
            f"{[(s.tolist(), d.tolist()) for s, d, _ in self._inverse_history[-3:]]}. "
            f"Caller: {callers}."
        )


    def free_group_begin(self) -> None:
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self) -> None:
        self.is_not_in_free_group = True
        if self.free_group:
            merged = torch.cat(self.free_group)
            self.free_group = []
            self.free(merged)


class UnifiedMambaTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):


    def __init__(
        self,
        *,
        unified_buffer: UnifiedKVPool,
        kvcache,
        device: str,
        page_size: int = 1,
        need_sort: bool = False,
        forward_stream: Optional[torch.cuda.Stream] = None,
        lazy_compaction: bool = False,
        full_kernel_page_multiplier: int = 1,
    ):
        full_max = unified_buffer.max_slots("full")
        super().__init__(
            size=full_max - 1,
            page_size=page_size,
            dtype=unified_buffer.spec("full").get_dtype(),
            device=device,
            kvcache=kvcache,
            need_sort=need_sort,
        )
        self.unified_buffer = unified_buffer
        self._kvcache = kvcache
        self.page_size = page_size
        self.lazy_compaction = lazy_compaction


        self.full_attn_allocator = MultiEndedAllocator(
            kvcache=kvcache.full_kv_pool,
            unified_buffer=unified_buffer,
            sub_pool_name="full",
            device=device,
            is_id_owner=True,
            page_size=page_size,
            need_sort=need_sort,
            forward_stream=forward_stream,
            lazy_compaction=lazy_compaction,
            kernel_page_multiplier=full_kernel_page_multiplier,
        )
        self.mamba_allocator = MultiEndedAllocator(
            kvcache=kvcache.mamba_pool,
            unified_buffer=unified_buffer,
            sub_pool_name="mamba",
            device=device,
            is_id_owner=True,
            page_size=1,
            need_sort=need_sort,
            forward_stream=forward_stream,
            lazy_compaction=lazy_compaction,
        )
        self.full_attn_allocator.bind_peer(self.mamba_allocator)
        self.mamba_allocator.bind_peer(self.full_attn_allocator)


        self.is_not_in_free_group = True
        self.free_group: List[torch.Tensor] = []

        self.free_pages = torch.empty(0, dtype=torch.int64, device=device)
        self.release_pages = torch.empty(0, dtype=torch.int64, device=device)

        logger.info(
            "[unified-memory-pool] UnifiedMambaTokenToKVPoolAllocator ready: "
            "full max_slots=%d (min_slot_index=%d, page_size=%d, "
            "num_pages=%d), mamba max_slots=%d (min_slot_index=%d), "
            "full_available=%d, mamba_available=%d",
            self.full_attn_allocator.max_slots,
            self.full_attn_allocator.min_slot_index,
            self.full_attn_allocator.page_size,
            self.full_attn_allocator.num_pages,
            self.mamba_allocator.max_slots,
            self.mamba_allocator.min_slot_index,
            self.full_attn_allocator.available_size(),
            self.mamba_allocator.available_size(),
        )


    @property
    def size(self) -> int:


        return (
            self.full_attn_allocator.schedulable_available_size()
            + self.full_attn_allocator.allocated_count()
        )

    @size.setter
    def size(self, value) -> None:
        pass


    def available_size(self) -> int:
        return self.full_attn_allocator.schedulable_available_size()

    def full_available_size(self) -> int:
        return self.full_attn_allocator.schedulable_available_size()

    def mamba_slot_full_token_cost(self) -> int:

        return -(
            -self.mamba_allocator.entry_bytes_per_page
            // self.full_attn_allocator.entry_bytes
        )

    @property
    def size_full(self) -> int:
        return self.full_attn_allocator.max_slots - 1

    @property
    def size_mamba(self) -> int:
        return self.mamba_allocator.max_slots - 1

    def debug_print(self) -> str:
        return (
            f"#full-available={self.full_attn_allocator.available_size()}, "
            f"#mamba-available={self.mamba_allocator.available_size()}"
        )

    def get_kvcache(self):
        return self._kvcache

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        with record_function("UnifiedMambaAlloc.alloc"):
            return self.full_attn_allocator.alloc(need_size)

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        num_new_pages: Optional[int] = None,
    ) -> Optional[torch.Tensor]:

        with record_function("UnifiedMambaAlloc.alloc_extend"):
            return self.full_attn_allocator.alloc_extend(
                prefix_lens,
                prefix_lens_cpu,
                seq_lens,
                seq_lens_cpu,
                last_loc,
                extend_num_tokens,
                num_new_pages=num_new_pages,
            )

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ) -> Optional[torch.Tensor]:

        with record_function("UnifiedMambaAlloc.alloc_decode"):
            return self.full_attn_allocator.alloc_decode(
                seq_lens, seq_lens_cpu, last_loc
            )

    def translate_kv_loc(
        self,
        loc: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        result = self.full_attn_allocator.translate_kv_loc(loc, out=out)
        return result

    @property
    def kernel_page_multiplier(self) -> int:
        return self.full_attn_allocator.kernel_page_multiplier

    @property
    def full_v2p_page_table(self) -> torch.Tensor:

        return self.full_attn_allocator.virtual_to_physical

    def translate_kv_loc_dense(
        self,
        loc: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        return self.full_attn_allocator.translate_kv_loc_dense(loc, out=out)

    def translate_kv_indices_for_transfer(
        self, kv_indices: torch.Tensor
    ) -> torch.Tensor:

        return self.full_attn_allocator.translate_kv_loc(kv_indices.to(torch.int64))

    def set_disagg_move_gate(self, gate: Callable[[], bool]) -> None:

        assert self.lazy_compaction, (
            "PD disaggregation with the unified memory pool requires lazy "
            "compaction (eager free-path compaction moves pages under "
            "in-flight transfers)."
        )
        self.full_attn_allocator.disagg_move_gate = gate
        self.mamba_allocator.disagg_move_gate = gate

    def is_slot_allocated(self, slot: int) -> bool:
        return self.full_attn_allocator.is_slot_allocated(slot)

    def allocator_state_str(self) -> str:
        return self.full_attn_allocator.allocator_state_str()

    def free(self, free_index: torch.Tensor) -> None:
        with record_function("UnifiedMambaAlloc.free"):
            if free_index is None or free_index.numel() == 0:
                return
            if not self.is_not_in_free_group:
                self.free_group.append(self._copy_for_free_group(free_index))
                return
            self.full_attn_allocator.free(free_index)
            self.full_attn_allocator.clear_inverse_history()
            self.mamba_allocator.clear_inverse_history()

    def free_group_begin(self) -> None:
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self) -> None:
        self.is_not_in_free_group = True
        if self.free_group:
            merged = torch.cat(self.free_group)
            self.free_group = []
            self.full_attn_allocator.free(merged)
            self.full_attn_allocator.clear_inverse_history()
            self.mamba_allocator.clear_inverse_history()

    def clear(self) -> None:
        self.full_attn_allocator.clear()
        self.mamba_allocator.clear()
        self.is_not_in_free_group = True
        self.free_group = []


    def set_latest_forward_done_event(self, event: Optional[torch.cuda.Event]) -> None:

        with record_function("UnifiedMambaAlloc.set_latest_forward_done_event"):
            self.full_attn_allocator.set_latest_forward_done_event(event)
            self.mamba_allocator.set_latest_forward_done_event(event)

    def set_inflight_forward(
        self,
        forward_done: torch.cuda.Event,
        out_cache_loc_virtual: Optional[torch.Tensor],
    ) -> None:

        with record_function("UnifiedMambaAlloc.set_inflight_forward"):
            self.full_attn_allocator.set_inflight_forward(
                forward_done, out_cache_loc_virtual
            )
            self.mamba_allocator.set_inflight_forward(forward_done, None)

    def flush_opportunistic(self) -> int:

        with record_function("UnifiedMambaAlloc.flush_opportunistic"):
            fa = self.full_attn_allocator
            ma = self.mamba_allocator
            if (
                fa._free_phys_pages.numel() == 0
                and not fa._pending_reuse
                and ma._free_phys_pages.numel() == 0
                and not ma._pending_reuse
            ):
                return 0
            return fa.flush_opportunistic() + ma.flush_opportunistic()


class UnifiedSWATokenToKVPoolAllocator(SWATokenToKVPoolAllocator):


    @property
    def size(self) -> int:
        return min(self._size_full, self._size_swa)

    @size.setter
    def size(self, value) -> None:
        pass

    def __init__(
        self,
        *,
        unified_buffer: UnifiedKVPool,
        kvcache,
        device: str,
        full_max_total_num_tokens: int,
        swa_max_total_num_tokens: int,
        page_size: int = 1,
        need_sort: bool = False,
        forward_stream: Optional[torch.cuda.Stream] = None,
        lazy_compaction: bool = False,
    ):


        self._size_full = full_max_total_num_tokens
        self._size_swa = swa_max_total_num_tokens
        self._full_max_total_num_tokens = full_max_total_num_tokens
        self._swa_max_total_num_tokens = swa_max_total_num_tokens
        self.page_size = page_size


        BaseTokenToKVPoolAllocator.__init__(
            self,
            size=full_max_total_num_tokens,
            page_size=page_size,
            dtype=unified_buffer.mha_spec("full").store_dtype,
            device=device,
            kvcache=kvcache,
            need_sort=need_sort,
        )
        self.unified_buffer = unified_buffer
        self._kvcache = kvcache
        self.lazy_compaction = lazy_compaction

        self.full_attn_allocator = MultiEndedAllocator(
            kvcache=kvcache.full_kv_pool,
            unified_buffer=unified_buffer,
            sub_pool_name="full",
            device=device,
            is_id_owner=True,
            page_size=page_size,
            need_sort=need_sort,
            forward_stream=forward_stream,
            lazy_compaction=lazy_compaction,
        )
        self.swa_attn_allocator = MultiEndedAllocator(
            kvcache=kvcache.swa_kv_pool,
            unified_buffer=unified_buffer,
            sub_pool_name="swa",
            device=device,
            is_id_owner=False,
            page_size=page_size,
            need_sort=need_sort,
            forward_stream=forward_stream,
            lazy_compaction=lazy_compaction,
        )
        self.full_attn_allocator.bind_peer(self.swa_attn_allocator)
        self.swa_attn_allocator.bind_peer(self.full_attn_allocator)


        kvcache.attach_allocators(
            full_allocator=self.full_attn_allocator,
            swa_allocator=self.swa_attn_allocator,
        )

        self.is_not_in_free_group = True
        self.free_group: List[torch.Tensor] = []

        self.free_pages = torch.empty(0, dtype=torch.int64, device=device)
        self.release_pages = torch.empty(0, dtype=torch.int64, device=device)

        logger.info(
            "[unified-memory-pool] UnifiedSWATokenToKVPoolAllocator ready: "
            "full max_slots=%d (min_slot_index=%d, entry_bytes=%d), "
            "swa max_slots=%d (min_slot_index=%d, entry_bytes=%d), "
            "static caps full=%d swa=%d, joint available=%d",
            self.full_attn_allocator.max_slots,
            self.full_attn_allocator.min_slot_index,
            self.full_attn_allocator.entry_bytes,
            self.swa_attn_allocator.max_slots,
            self.swa_attn_allocator.min_slot_index,
            self.swa_attn_allocator.entry_bytes,
            self._full_max_total_num_tokens,
            self._swa_max_total_num_tokens,
            self.available_size(),
        )


    def available_size(self) -> int:

        fa, sa = self.full_attn_allocator, self.swa_attn_allocator
        e_f = fa.entry_bytes_per_page
        e_s = sa.entry_bytes_per_page

        if fa.grow_direction == "up":
            gap_bytes = max(0, sa._byte_low_frontier() - fa._byte_high_frontier())
        else:
            gap_bytes = max(0, fa._byte_low_frontier() - sa._byte_high_frontier())
        R_f = fa.num_pages - fa.min_page_index - fa._allocated_pages()
        R_s = sa.num_pages - sa.min_page_index - sa._allocated_pages()

        if not self.lazy_compaction:
            pages_by_bytes = gap_bytes // (e_f + e_s)
            return min(pages_by_bytes, R_f, R_s) * self.page_size

        H_f = len(fa._free_phys_pages)
        H_s = len(sa._free_phys_pages)

        K1 = min(H_f, H_s)


        if H_f <= H_s:
            e_phase2 = e_f
            K_phase2_max = H_s
        else:
            e_phase2 = e_s
            K_phase2_max = H_f
        K2_room = K_phase2_max - K1
        K2 = min(K2_room, gap_bytes // e_phase2) if e_phase2 > 0 else K2_room
        gap_bytes -= K2 * e_phase2

        K3 = gap_bytes // (e_f + e_s)

        K_total = K1 + K2 + K3
        K_total = min(K_total, H_f + R_f, H_s + R_s)
        return K_total * self.page_size


    def _conserve_full_available_size(self) -> int:
        return (
            self._full_max_total_num_tokens - self.full_attn_allocator.allocated_count()
        )

    def _conserve_swa_available_size(self) -> int:
        return (
            self._swa_max_total_num_tokens - self.swa_attn_allocator.allocated_count()
        )


    def full_available_size(self) -> int:
        return min(
            self._conserve_full_available_size(),
            self.schedulable_full_available_size(),
        )

    def swa_available_size(self) -> int:
        return min(
            self._conserve_swa_available_size(),
            self.schedulable_swa_available_size(),
        )


    def schedulable_full_available_size(self) -> int:
        return self.full_attn_allocator.schedulable_available_size()

    def schedulable_swa_available_size(self) -> int:
        return self.swa_attn_allocator.schedulable_available_size()

    def _flush_both_for_alloc(self, need_tokens: int) -> bool:

        if not self.lazy_compaction:
            return need_tokens <= self.available_size()
        self.full_attn_allocator._flush(urgent=True)
        self.swa_attn_allocator._flush(urgent=True)
        return need_tokens <= self.available_size()


    def debug_print(self) -> str:
        return (
            f"#full-available={self.full_attn_allocator.available_size()}, "
            f"#swa-available={self.swa_attn_allocator.available_size()}, "
            f"#joint-available={self.available_size()}"
        )

    def get_kvcache(self):
        return self._kvcache

    def translate_kv_loc(
        self,
        loc: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        result = self.full_attn_allocator.translate_kv_loc(loc, out=out)
        return result

    def translate_loc_from_full_to_swa(
        self,
        kv_indices: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if out is not None:
            assert out.dtype == torch.int32, (
                f"translate_loc_from_full_to_swa: out= dtype must be int32 "
                f"(matches SWA Triton kernel contract), got {out.dtype}"
            )
            assert out.shape == kv_indices.shape, (
                f"translate_loc_from_full_to_swa: out= shape "
                f"{tuple(out.shape)} must match kv_indices shape "
                f"{tuple(kv_indices.shape)}"
            )


        if self.swa_attn_allocator.page_size == 1:
            if out is not None:

                tmp = torch.index_select(
                    self.swa_attn_allocator.virtual_to_physical, 0, kv_indices
                )
                tmp = torch.clamp_min(tmp, 0)
                out.copy_(tmp.to(torch.int32))
                return out
            result = self.swa_attn_allocator.virtual_to_physical[kv_indices]
            result = torch.clamp_min(result, 0)
            return result.to(torch.int32)
        ps = self.swa_attn_allocator.page_size
        virt_pages = kv_indices // ps
        offsets = kv_indices % ps
        swa_phys_pages = self.swa_attn_allocator.virtual_to_physical[virt_pages]
        result = (swa_phys_pages * ps + offsets).to(torch.int32)
        result = torch.clamp_min(result, 0)
        if out is not None:
            out.copy_(result)
            return out
        return result


    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        with record_function("UnifiedSWAAlloc.alloc"):


            if need_size > self.available_size():
                if not self._flush_both_for_alloc(need_size):
                    return None

            num_pages = need_size // self.page_size
            fa = self.full_attn_allocator
            new_virtual_pages = fa.free_virtual_ids[:num_pages].clone()

            v_tokens = fa.alloc(need_size)

            assert v_tokens is not None, (
                "UnifiedSWA.alloc: full.alloc returned None after joint "
                "pre-check passed — internal-state inconsistency"
            )
            self.swa_attn_allocator.alloc_with_virtual(new_virtual_pages)
            return v_tokens

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ) -> Optional[torch.Tensor]:

        with record_function("UnifiedSWAAlloc.alloc_extend"):
            num_new_pages = get_num_new_pages(
                seq_lens=seq_lens_cpu,
                page_size=self.page_size,
                prefix_lens=prefix_lens_cpu,
            )
            need_tokens = num_new_pages * self.page_size
            if need_tokens > self.available_size():
                if not self._flush_both_for_alloc(need_tokens):
                    return None


            fa = self.full_attn_allocator
            new_virtual_pages = fa.free_virtual_ids[:num_new_pages].clone()

            out_indices = fa.alloc_extend(
                prefix_lens,
                prefix_lens_cpu,
                seq_lens,
                seq_lens_cpu,
                last_loc,
                extend_num_tokens,
                num_new_pages=num_new_pages,
            )
            assert out_indices is not None, (
                "UnifiedSWA.alloc_extend: full.alloc_extend returned None "
                "after joint pre-check passed — internal-state inconsistency"
            )
            self.swa_attn_allocator.alloc_with_virtual(new_virtual_pages)
            return out_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ) -> Optional[torch.Tensor]:

        with record_function("UnifiedSWAAlloc.alloc_decode"):
            num_new_pages = get_num_new_pages(
                seq_lens=seq_lens_cpu, page_size=self.page_size, decode=True
            )
            need_tokens = num_new_pages * self.page_size
            if need_tokens > self.available_size():
                if not self._flush_both_for_alloc(need_tokens):
                    return None

            fa = self.full_attn_allocator
            new_virtual_pages = fa.free_virtual_ids[:num_new_pages].clone()

            out_indices = fa.alloc_decode(seq_lens, seq_lens_cpu, last_loc)
            assert out_indices is not None, (
                "UnifiedSWA.alloc_decode: full.alloc_decode returned None "
                "after joint pre-check passed — internal-state inconsistency"
            )

            if new_virtual_pages.numel() > 0:
                self.swa_attn_allocator.alloc_with_virtual(new_virtual_pages)

            return out_indices

    def is_slot_allocated(self, slot: int) -> bool:

        return self.full_attn_allocator.is_slot_allocated(slot)

    def allocator_state_str(self) -> str:
        return self.full_attn_allocator.allocator_state_str()


    def free(self, free_index: torch.Tensor) -> None:
        with record_function("UnifiedSWAAlloc.free"):
            if free_index is None or free_index.numel() == 0:
                return
            if not self.is_not_in_free_group:
                self.free_group.append(self._copy_for_free_group(free_index))
                return


            v = free_index.detach().to(torch.int64)
            v_pages = v // self.page_size
            swa_v2p_pages = self.swa_attn_allocator.virtual_to_physical[v_pages]

            live_token_mask = swa_v2p_pages > 0
            live_tokens = v[live_token_mask]
            if live_tokens.numel() > 0:
                self.swa_attn_allocator.free(live_tokens)
            self.full_attn_allocator.free(v)
            self.full_attn_allocator.clear_inverse_history()
            self.swa_attn_allocator.clear_inverse_history()

    def free_swa(self, free_index: torch.Tensor) -> None:

        if free_index is None or free_index.numel() == 0:
            return


        v = free_index.detach().to(torch.int64)
        v_pages = v // self.page_size

        swa_v2p_pages = self.swa_attn_allocator.virtual_to_physical[v_pages]
        live = v[swa_v2p_pages > 0]
        if live.numel() == 0:
            return
        self.swa_attn_allocator.free(live)
        self.swa_attn_allocator.clear_inverse_history()

    def set_full_to_swa_mapping(
        self, full_indices: torch.Tensor, swa_indices: torch.Tensor
    ) -> None:

        return

    def clear_full_to_swa_mapping(self, full_indices: torch.Tensor) -> None:

        return


    def free_group_begin(self) -> None:
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self) -> None:
        self.is_not_in_free_group = True
        if self.free_group:
            merged = torch.cat(self.free_group)
            self.free_group = []
            self.free(merged)

    def clear(self) -> None:
        self.full_attn_allocator.clear()
        self.swa_attn_allocator.clear()
        self.is_not_in_free_group = True
        self.free_group = []


    def set_latest_forward_done_event(self, event: Optional[torch.cuda.Event]) -> None:

        with record_function("UnifiedSWAAlloc.set_latest_forward_done_event"):
            self.full_attn_allocator.set_latest_forward_done_event(event)
            self.swa_attn_allocator.set_latest_forward_done_event(event)

    def set_inflight_forward(
        self,
        forward_done: torch.cuda.Event,
        out_cache_loc_virtual: Optional[torch.Tensor],
    ) -> None:

        with record_function("UnifiedSWAAlloc.set_inflight_forward"):
            self.full_attn_allocator.set_inflight_forward(
                forward_done, out_cache_loc_virtual
            )
            self.swa_attn_allocator.set_inflight_forward(
                forward_done, out_cache_loc_virtual
            )

    def flush_opportunistic(self) -> int:

        with record_function("UnifiedSWAAlloc.flush_opportunistic"):
            fa = self.full_attn_allocator
            sa = self.swa_attn_allocator
            if (
                fa._free_phys_pages.numel() == 0
                and not fa._pending_reuse
                and sa._free_phys_pages.numel() == 0
                and not sa._pending_reuse
            ):
                return 0
            return fa.flush_opportunistic() + sa.flush_opportunistic()
