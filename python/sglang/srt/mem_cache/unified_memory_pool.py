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

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
import triton
from torch.profiler import record_function

from sglang.kernels.ops.kvcache.cache_move import store_cache_4d_kernel
from sglang.kernels.ops.kvcache.zero_pages import zero_pages
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.mem_cache.layout.page_major import (
    build_dense_mla_views,
    build_page_major_mamba_views,
    build_page_major_mha_views,
)
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    HybridReqToTokenPool,
    MambaPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
    move_kv_cache_native,
    unwrap_write_loc,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024


def _prod(iterable) -> int:
    out = 1
    for x in iterable:
        out *= int(x)
    return out


def _store_dtype_for(kv_cache_dtype: torch.dtype) -> torch.dtype:
    if kv_cache_dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
        return torch.uint8
    return kv_cache_dtype


@dataclass(frozen=True, kw_only=True)
class SubPoolSpec(ABC):


    name: str
    layer_num: int
    grow_direction: str

    def __post_init__(self):
        assert self.grow_direction in (
            "up",
            "down",
        ), f"grow_direction must be 'up' or 'down'; got {self.grow_direction!r}"
        assert self.layer_num > 0, f"layer_num must be positive; got {self.layer_num}"

    @abstractmethod
    def entry_bytes(self) -> int:

        raise NotImplementedError

    @abstractmethod
    def get_dtype(self) -> torch.dtype:

        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class MHASubPoolSpec(SubPoolSpec):


    head_num: int
    head_dim: int
    store_dtype: torch.dtype
    v_head_dim: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        assert self.head_num > 0, f"head_num must be positive; got {self.head_num}"
        assert self.head_dim > 0, f"head_dim must be positive; got {self.head_dim}"
        if self.v_head_dim is None:
            object.__setattr__(self, "v_head_dim", self.head_dim)
        assert (
            self.v_head_dim > 0
        ), f"v_head_dim must be positive; got {self.v_head_dim}"

    def k_row_bytes(self) -> int:
        return self.head_num * self.head_dim * self.store_dtype.itemsize

    def v_row_bytes(self) -> int:
        return self.head_num * self.v_head_dim * self.store_dtype.itemsize

    def entry_bytes(self) -> int:
        return self.layer_num * (self.k_row_bytes() + self.v_row_bytes())


    def page_bytes(self, page_size: int) -> int:
        return page_size * self.entry_bytes()

    def layer_k_offset_in_page(self, layer_id: int, page_size: int) -> int:
        return layer_id * page_size * (self.k_row_bytes() + self.v_row_bytes())

    def layer_v_offset_in_page(self, layer_id: int, page_size: int) -> int:
        return (
            self.layer_k_offset_in_page(layer_id, page_size)
            + page_size * self.k_row_bytes()
        )

    def get_dtype(self) -> torch.dtype:
        return self.store_dtype


@dataclass(frozen=True, kw_only=True)
class MLASubPoolSpec(SubPoolSpec):


    kv_lora_rank: int
    qk_rope_head_dim: int
    store_dtype: torch.dtype

    def __post_init__(self):
        super().__post_init__()
        assert (
            self.kv_lora_rank > 0
        ), f"kv_lora_rank must be positive; got {self.kv_lora_rank}"
        assert (
            self.qk_rope_head_dim > 0
        ), f"qk_rope_head_dim must be positive; got {self.qk_rope_head_dim}"

    @property
    def kv_cache_dim(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

    def entry_bytes(self) -> int:
        return self.layer_num * self.kv_cache_dim * self.store_dtype.itemsize

    def get_dtype(self) -> torch.dtype:
        return self.store_dtype


@dataclass(frozen=True, kw_only=True)
class MambaSubPoolSpec(SubPoolSpec):


    conv_state_shapes: Tuple[Tuple[int, ...], ...]
    conv_dtype: torch.dtype
    temporal_state_shape: Tuple[int, ...]
    temporal_dtype: torch.dtype
    conv_slice_axis: int = 0

    def __post_init__(self):
        super().__post_init__()
        assert len(self.conv_state_shapes) > 0, "conv_state_shapes must be non-empty"

    def conv_row_bytes(self, idx: int) -> int:
        return _prod(self.conv_state_shapes[idx]) * self.conv_dtype.itemsize

    def temporal_row_bytes(self) -> int:
        return _prod(self.temporal_state_shape) * self.temporal_dtype.itemsize

    def entry_bytes(self) -> int:
        total = 0
        for i in range(len(self.conv_state_shapes)):
            total += self.layer_num * self.conv_row_bytes(i)
        total += self.layer_num * self.temporal_row_bytes()
        return total

    def get_dtype(self) -> torch.dtype:
        return self.conv_dtype


class UnifiedKVPool:


    def __init__(
        self,
        *,
        total_bytes: int,
        sub_pool_specs: List[SubPoolSpec],
        device: str,
        enable_memory_saver: bool,
        page_size: int = 1,
        view_tail_pad_bytes: int = 0,
    ):
        assert page_size >= 1, f"page_size must be >= 1; got {page_size}"
        assert len(sub_pool_specs) == 2, (
            f"UnifiedKVPool currently supports exactly 2 sub-pools; got "
            f"{len(sub_pool_specs)} (N>2 is not yet implemented)"
        )
        names = [s.name for s in sub_pool_specs]
        assert len(set(names)) == 2, f"sub-pool names must be unique; got {names}"
        directions = sorted(s.grow_direction for s in sub_pool_specs)
        assert directions == ["down", "up"], (
            f"UnifiedKVPool needs one grow-up and one grow-down sub-pool; "
            f"got {directions}"
        )

        self.device = device
        self.total_bytes = total_bytes
        self.sub_pool_specs = sub_pool_specs
        self._page_size = page_size
        self._specs_by_name: Dict[str, SubPoolSpec] = {
            s.name: s for s in sub_pool_specs
        }

        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )


        self.view_tail_pad_bytes = view_tail_pad_bytes
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self._raw = torch.empty(
                total_bytes + view_tail_pad_bytes, dtype=torch.uint8, device=device
            )
        if envs.SGLANG_DEBUG_POISON_POOL.get():


            self._raw.view(torch.int16).fill_(0x7FC1)
            logger.warning(
                "[unified-memory-pool] POISONED: pool filled with bf16-NaN "
                "patterns (SGLANG_DEBUG_POISON_POOL)"
            )
        else:
            self._raw.zero_()

        self._max_slots: Dict[str, int] = {}
        self._anchor_bytes: Dict[str, int] = {}
        self._min_slot_index: Dict[str, int] = {}


        self._mha_views: Dict[str, Tuple[List[torch.Tensor], List[torch.Tensor]]] = {}
        self._mla_views: Dict[str, List[torch.Tensor]] = {}
        self._mamba_views: Dict[str, Tuple[List[torch.Tensor], torch.Tensor]] = {}


        entry_max = max(s.entry_bytes() for s in sub_pool_specs)
        reserved_floor = max(
            [entry_max]
            + [
                page_size * s.entry_bytes()
                for s in sub_pool_specs
                if not isinstance(s, MambaSubPoolSpec)
            ]
        )

        for spec in sub_pool_specs:
            entry_bytes = spec.entry_bytes()
            max_slots = total_bytes // entry_bytes
            min_slot_index = (reserved_floor + entry_bytes - 1) // entry_bytes
            if max_slots <= min_slot_index:
                raise RuntimeError(
                    f"UnifiedKVPool: sub-pool {spec.name!r} fits only {max_slots} "
                    f"slots in {total_bytes} bytes, but min_slot_index={min_slot_index} "
                    f"leaves no room for real data. Increase total_bytes."
                )
            anchor = 0
            self._max_slots[spec.name] = max_slots
            self._anchor_bytes[spec.name] = anchor
            self._min_slot_index[spec.name] = min_slot_index
            if isinstance(spec, MHASubPoolSpec):
                self._mha_views[spec.name] = self._build_mha_views(
                    spec,
                    anchor,
                    max_slots,
                    page_size=page_size,
                )
            elif isinstance(spec, MLASubPoolSpec):
                self._mla_views[spec.name] = self._build_mla_views(
                    spec,
                    anchor,
                    max_slots,
                    page_size=page_size,
                )
            elif isinstance(spec, MambaSubPoolSpec):
                self._mamba_views[spec.name] = self._build_mamba_views(
                    spec, anchor, max_slots
                )
            else:
                raise TypeError(f"unsupported SubPoolSpec type: {type(spec)}")

        logger.info(
            "[unified-memory-pool] UnifiedKVPool allocated: total_bytes=%.2f GB (=%d B), "
            "%d sub-pool(s)",
            total_bytes / GB,
            total_bytes,
            len(sub_pool_specs),
        )
        for s in sub_pool_specs:
            logger.info(
                "[unified-memory-pool]   sub-pool %r: kind=%s, layer_num=%d, grow=%s, "
                "entry_bytes=%d, max_slots=%d, min_slot_index=%d (slots [0,%d) reserved)",
                s.name,
                type(s).__name__,
                s.layer_num,
                s.grow_direction,
                s.entry_bytes(),
                self._max_slots[s.name],
                self._min_slot_index[s.name],
                self._min_slot_index[s.name],
            )


    def spec(self, name: str) -> SubPoolSpec:
        return self._specs_by_name[name]

    def mha_spec(self, name: str) -> MHASubPoolSpec:
        s = self._specs_by_name[name]
        assert isinstance(
            s, MHASubPoolSpec
        ), f"sub-pool {name!r} is {type(s).__name__}, expected MHASubPoolSpec"
        return s

    def mla_spec(self, name: str) -> MLASubPoolSpec:
        s = self._specs_by_name[name]
        assert isinstance(
            s, MLASubPoolSpec
        ), f"sub-pool {name!r} is {type(s).__name__}, expected MLASubPoolSpec"
        return s

    def mamba_spec(self, name: str) -> MambaSubPoolSpec:
        s = self._specs_by_name[name]
        assert isinstance(
            s, MambaSubPoolSpec
        ), f"sub-pool {name!r} is {type(s).__name__}, expected MambaSubPoolSpec"
        return s

    def max_slots(self, name: str) -> int:
        return self._max_slots[name]

    def min_slot_index(self, name: str) -> int:
        return self._min_slot_index[name]

    def anchor_bytes(self, name: str) -> int:
        anchor = self._anchor_bytes[name]
        assert anchor == 0, f"current design assumes all anchors are 0; got {anchor}"
        return anchor

    def mha_views_for(self, name: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        return self._mha_views[name]

    def mla_views_for(self, name: str) -> List[torch.Tensor]:
        return self._mla_views[name]

    def mamba_views_for(self, name: str) -> Tuple[List[torch.Tensor], torch.Tensor]:
        return self._mamba_views[name]

    def _build_mha_views(
        self,
        spec: MHASubPoolSpec,
        anchor_bytes: int,
        max_slots: int,
        page_size: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        return build_page_major_mha_views(
            self._raw,
            layer_num=spec.layer_num,
            head_num=spec.head_num,
            head_dim=spec.head_dim,
            v_head_dim=spec.v_head_dim,
            store_dtype=spec.store_dtype,
            page_size=page_size,
            num_pages=max_slots // page_size,
            anchor_bytes=anchor_bytes,
        )

    def _build_mla_views(
        self,
        spec: MLASubPoolSpec,
        anchor_bytes: int,
        max_slots: int,
        page_size: int,
    ) -> List[torch.Tensor]:
        return build_dense_mla_views(
            self._raw,
            layer_num=spec.layer_num,
            kv_cache_dim=spec.kv_cache_dim,
            store_dtype=spec.store_dtype,
            page_size=page_size,
            num_pages=max_slots // page_size,
            anchor_bytes=anchor_bytes,
        )

    def _build_mamba_views(
        self, spec: MambaSubPoolSpec, anchor_bytes: int, max_slots: int
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        return build_page_major_mamba_views(
            self._raw,
            layer_num=spec.layer_num,
            conv_state_shapes=spec.conv_state_shapes,
            conv_dtype=spec.conv_dtype,
            temporal_state_shape=spec.temporal_state_shape,
            temporal_dtype=spec.temporal_dtype,
            max_slots=max_slots,
            anchor_bytes=anchor_bytes,
        )


class UnifiedMHATokenToKVPool(MHATokenToKVPool):


    def __init__(
        self,
        *,
        unified_buffer: UnifiedKVPool,
        sub_pool_name: str,
        page_size: int = 1,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,
    ):
        spec = unified_buffer.mha_spec(sub_pool_name)
        k_buffer, v_buffer = unified_buffer.mha_views_for(sub_pool_name)
        max_slots = unified_buffer.max_slots(sub_pool_name)

        self._unified_buffer = unified_buffer
        self._sub_pool_name = sub_pool_name
        self._k_views = k_buffer
        self._v_views = v_buffer
        self._page_size = page_size

        super().__init__(
            size=max_slots - 1,
            page_size=page_size,
            dtype=spec.store_dtype,
            head_num=spec.head_num,
            head_dim=spec.head_dim,
            layer_num=spec.layer_num,
            device=unified_buffer.device,
            enable_memory_saver=False,
            v_head_dim=spec.v_head_dim,
            start_layer=start_layer,
            end_layer=end_layer,
            enable_alt_stream=enable_alt_stream,
            enable_kv_cache_copy=False,
        )

    def _create_buffers(self):
        self.k_buffer = self._k_views
        self.v_buffer = self._v_views

        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs], dim=0)
        self.data_strides = torch.tensor(
            [x.stride(0) * x.dtype.itemsize for x in (self.k_buffer + self.v_buffer)],
            device=self.device,
        )

    def _clear_buffers(self):

        pass

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):

        if tgt_loc.numel() == 0:
            return
        with record_function("UnifiedMHA.move_kv_cache"):
            move_kv_cache_native(
                self.k_buffer,
                self.v_buffer,
                tgt_loc,
                src_loc,
                page_size=self._page_size,
            )

    def get_kv_size_bytes(self):
        return 0, 0

    def set_kv_buffer(
        self,
        layer,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale=None,
        v_scale=None,
        layer_id_override: Optional[int] = None,
        dcp_kv_mask: Optional[torch.Tensor] = None,
    ):

        assert dcp_kv_mask is None, (
            "UnifiedMHATokenToKVPool.set_kv_buffer: decode context parallel "
            "(dcp_kv_mask) is not supported with --enable-unified-memory."
        )


        with record_function("UnifiedMHA.set_kv_buffer"):
            if cache_k.dtype != self.dtype:
                if k_scale is not None:
                    cache_k.div_(k_scale)
                if v_scale is not None:
                    cache_v.div_(v_scale)
                cache_k = cache_k.to(self.dtype)
                cache_v = cache_v.to(self.dtype)
            if self.store_dtype != self.dtype:
                cache_k = cache_k.view(self.store_dtype)
                cache_v = cache_v.view(self.store_dtype)

            layer_id = (
                layer.layer_id if layer_id_override is None else layer_id_override
            ) - self.start_layer
            k_view = self.k_buffer[layer_id]
            v_view = self.v_buffer[layer_id]
            ps = self._page_size
            N = loc.numel()
            if N == 0:
                return
            head_num = k_view.shape[2]
            head_dim = k_view.shape[3]
            v_head_dim = v_view.shape[3]
            K_ROW_DIM = head_num * head_dim
            V_ROW_DIM = head_num * v_head_dim
            BLOCK = 128
            row_dim_max = K_ROW_DIM if K_ROW_DIM > V_ROW_DIM else V_ROW_DIM
            store_cache_4d_kernel[(N, triton.cdiv(row_dim_max, BLOCK), 2)](
                k_view,
                v_view,
                cache_k,
                cache_v,
                loc,
                k_view.stride(0),
                k_view.stride(1),
                v_view.stride(0),
                v_view.stride(1),
                cache_k.stride(0),
                cache_v.stride(0),
                K_ROW_DIM=K_ROW_DIM,
                V_ROW_DIM=V_ROW_DIM,
                PAGE_SIZE=ps,
                BLOCK=BLOCK,
                num_warps=4,
            )


class UnifiedMLATokenToKVPool(MLATokenToKVPool):


    def __init__(
        self,
        *,
        unified_buffer: UnifiedKVPool,
        sub_pool_name: str,
        kv_cache_dtype: torch.dtype,
        page_size: int = 1,
    ):
        spec = unified_buffer.mla_spec(sub_pool_name)
        store_dtype = _store_dtype_for(kv_cache_dtype)
        assert spec.store_dtype == store_dtype, (
            f"sub-pool {sub_pool_name!r} store dtype {spec.store_dtype} does not "
            f"match kv cache dtype {kv_cache_dtype} (store {store_dtype})"
        )

        self._unified_buffer = unified_buffer
        self._sub_pool_name = sub_pool_name
        self._kv_views = unified_buffer.mla_views_for(sub_pool_name)
        max_slots = unified_buffer.max_slots(sub_pool_name)
        self._num_pages = max_slots // page_size
        self._page_bytes = page_size * spec.entry_bytes()

        self._dense_size = self._num_pages * spec.layer_num * page_size

        super().__init__(


            size=self._dense_size - page_size,
            page_size=page_size,
            dtype=kv_cache_dtype,
            kv_lora_rank=spec.kv_lora_rank,
            qk_rope_head_dim=spec.qk_rope_head_dim,
            layer_num=spec.layer_num,
            device=unified_buffer.device,
            enable_memory_saver=False,
        )

    def _create_buffers(self):
        self.kv_buffer = self._kv_views

    def _clear_buffers(self):

        pass

    def get_kv_size_bytes(self):
        return 0

    def get_contiguous_buf_infos(self):


        assert self._unified_buffer.anchor_bytes(self._sub_pool_name) == 0
        raw = self._unified_buffer._raw
        return [raw.data_ptr()], [raw.numel()], [self._page_bytes]

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):

        if tgt_loc.numel() == 0:
            return
        ps = self.page_size
        tgt_pages = tgt_loc.view(-1, ps)[:, 0] // ps
        src_pages = src_loc.view(-1, ps)[:, 0] // ps
        with record_function("UnifiedMLA.move_kv_cache"):
            env = self._unified_buffer._raw[: self._num_pages * self._page_bytes].view(
                self._num_pages, self._page_bytes
            )
            env[tgt_pages] = env[src_pages]

    def zero_physical_pages(self, phys_pages: torch.Tensor) -> None:

        zero_pages(
            self._unified_buffer._raw,
            phys_pages,
            self._num_pages,
            self._page_bytes,
        )


class UnifiedMambaPool(MambaPool):


    def __init__(
        self,
        *,
        unified_buffer: UnifiedKVPool,
        sub_pool_name: str,
        spec_state_size: int,
        mamba_layer_ids: List[int],
        enable_memory_saver: bool = False,
        speculative_num_draft_tokens: Optional[int] = None,
    ):
        spec = unified_buffer.mamba_spec(sub_pool_name)
        assert spec.layer_num == len(mamba_layer_ids)

        self.mamba_layer_ids = list(mamba_layer_ids)
        conv_views, temporal_view = unified_buffer.mamba_views_for(sub_pool_name)
        max_slots = unified_buffer.max_slots(sub_pool_name)

        self._unified_buffer = unified_buffer
        self._sub_pool_name = sub_pool_name


        self._max_size = max_slots - 1
        self.size = self._max_size
        self.device = unified_buffer.device
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        self.enable_custom_mem_pool = False
        self.custom_mem_pool = None
        self.num_mamba_layers = spec.layer_num


        self.enable_linear_replayssm = False
        self.linear_replayssm_cache_len = 16
        self.replayssm_write_pos = None
        self.replayssm_is_kda = False
        self.enable_linear_replayssm_spec = False
        self.replayssm_spec_fold = False
        self.replayssm_cache_base = None
        self.replayssm_is_flush = None
        self.debug_memory_pool = False
        self.conv_shard_groups = None
        self.conv_slice_axis = spec.conv_slice_axis

        assert (
            conv_views[0].shape[0] == self.num_mamba_layers
        ), f"conv_views layers={conv_views[0].shape[0]} vs expected {self.num_mamba_layers}"
        assert (
            conv_views[0].shape[1] == self._max_size + 1
        ), f"conv_views slots={conv_views[0].shape[1]} vs expected {self._max_size + 1}"


        temporal_state_shape = spec.temporal_state_shape
        conv_state_shape = spec.conv_state_shapes
        conv_dtype = spec.conv_dtype
        ssm_dtype = spec.temporal_dtype
        if speculative_num_draft_tokens is not None:
            with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
                intermediate_ssm_state_cache = torch.zeros(
                    size=(
                        self.num_mamba_layers,
                        spec_state_size + 1,
                        speculative_num_draft_tokens,
                        temporal_state_shape[0],
                        temporal_state_shape[1],
                        temporal_state_shape[2],
                    ),
                    dtype=ssm_dtype,
                    device=unified_buffer.device,
                )
                intermediate_conv_window_cache = [
                    torch.zeros(
                        size=(
                            self.num_mamba_layers,
                            spec_state_size + 1,
                            speculative_num_draft_tokens,
                            cshape[0],
                            cshape[1],
                        ),
                        dtype=conv_dtype,
                        device=unified_buffer.device,
                    )
                    for cshape in conv_state_shape
                ]
            self.mamba_cache = self.SpeculativeState(
                conv=list(conv_views),
                temporal=temporal_view,
                intermediate_ssm=intermediate_ssm_state_cache,
                intermediate_conv_window=intermediate_conv_window_cache,
            )
        else:
            self.mamba_cache = self.State(conv=list(conv_views), temporal=temporal_view)

        self.mem_usage = unified_buffer.total_bytes / GB
        logger.info(
            "[unified-memory-pool] UnifiedMambaPool(%s) wrapped unified buffer: max_slots=%d, "
            "num_mamba_layers=%d",
            sub_pool_name,
            max_slots,
            self.num_mamba_layers,
        )


    def _copy_from_physical(self, src_index: torch.Tensor, dst_index: torch.Tensor):

        MambaPool.copy_from(self, src_index, dst_index)


    def get_contiguous_buf_infos(self):

        assert self._unified_buffer.anchor_bytes(self._sub_pool_name) == 0
        spec = self._unified_buffer.mamba_spec(self._sub_pool_name)
        raw = self._unified_buffer._raw
        return [raw.data_ptr()], [raw.numel()], [spec.entry_bytes()]

    def get_state_dim_per_tensor(self):
        return []

    def get_state_layer_ids(self):
        return []

    def get_state_slice_outer_counts(self):
        return []

    def get_state_conv_shard_groups(self):
        return []


class UnifiedMambaSlotAllocator:


    def __init__(self, mea, max_size: int, device: str):
        self._multi_ended_allocator = mea
        self._max_size = max_size
        self._device = device
        self._alloc_iter = None


    def translate(self, virtual_ids: torch.Tensor) -> torch.Tensor:

        return self._multi_ended_allocator.virtual_to_physical[virtual_ids]

    @property
    def virtual_to_physical(self) -> torch.Tensor:
        return self._multi_ended_allocator.virtual_to_physical


    @property
    def size(self) -> int:
        return self._max_size

    def available_size(self) -> int:


        return self._max_size - self._multi_ended_allocator.allocated_count()

    def schedulable_available_size(self) -> int:


        return self._multi_ended_allocator.schedulable_available_size()

    @property
    def free_slots(self) -> torch.Tensor:

        a = self._multi_ended_allocator
        assert a.page_size == 1, (
            "UnifiedMambaSlotAllocator.free_slots assumes page_size==1; got "
            f"{a.page_size}. Mamba state is per-request, orthogonal to paging."
        )
        if a.grow_direction == "up":
            start, end = a.watermark_physical, a.num_pages
        else:
            start, end = a.min_page_index, a.watermark_physical + 1
        if start >= end:
            return torch.empty((0,), dtype=torch.int64, device=self._device)
        return torch.arange(start, end, dtype=torch.int64, device=self._device)


    def alloc(self, need_size: int):

        if self._alloc_iter is not None and need_size == 1:
            slot = next(self._alloc_iter, None)
            if slot is not None:
                return slot
        return self._multi_ended_allocator.alloc(need_size)

    def free(self, free_index: torch.Tensor):
        return self._multi_ended_allocator.free(free_index)

    def clear(self):
        self._alloc_iter = None
        return self._multi_ended_allocator.clear()

    def alloc_group_begin(self, num_reqs: int):

        self._alloc_iter = None
        if num_reqs > 0:
            result = self._multi_ended_allocator.alloc(num_reqs)
            if result is not None:
                self._alloc_iter = iter(result.split(1))

    def alloc_group_end(self):

        if self._alloc_iter is not None:
            remaining = list(self._alloc_iter)
            if remaining:
                self._multi_ended_allocator.free(torch.cat(remaining))
        self._alloc_iter = None

    def is_slot_allocated(self, slot) -> bool:
        return self._multi_ended_allocator.is_slot_allocated(int(slot))

    def allocator_state_str(self) -> str:
        return self._multi_ended_allocator.allocator_state_str()


class UnifiedHybridReqToTokenPool(HybridReqToTokenPool):


    def __init__(
        self,
        *,
        unified_buffer: UnifiedKVPool,
        mamba_sub_pool_name: str,
        size: int,
        mamba_spec_state_size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        cache_params,
        mamba_layer_ids: List[int],
        enable_mamba_extra_buffer: bool,
        speculative_num_draft_tokens: Optional[int] = None,
        enable_overlap_schedule: bool = True,
        start_layer: Optional[int] = None,
        pre_alloc_size: int = 0,
    ):
        self._unified_buffer = unified_buffer
        self._mamba_sub_pool_name = mamba_sub_pool_name
        self._shared_mamba_size = (
            unified_buffer.max_slots(mamba_sub_pool_name) - 1
        )
        super().__init__(


            size=size + pre_alloc_size,
            mamba_size=self._shared_mamba_size,
            mamba_spec_state_size=mamba_spec_state_size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            enable_mamba_extra_buffer=enable_mamba_extra_buffer,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
            enable_overlap_schedule=enable_overlap_schedule,
            start_layer=start_layer,
        )
        self.size = size
        self.pre_alloc_size = pre_alloc_size

    def _init_mamba_pool(
        self,
        mamba_size: int,
        mamba_spec_state_size: int,
        cache_params,
        mamba_layer_ids: List[int],
        device: str,
        enable_mamba_extra_buffer: bool,
        speculative_num_draft_tokens: Optional[int] = None,
        speculative_eagle_topk: Optional[int] = None,
        mamba_envelope_layout: bool = False,
        enable_linear_replayssm: bool = False,
        linear_replayssm_cache_len: int = 16,
        enable_linear_replayssm_spec: bool = False,
    ):


        assert mamba_size == self._shared_mamba_size, (
            f"UnifiedHybridReqToTokenPool._init_mamba_pool: mamba_size={mamba_size} "
            f"!= unified_buffer.max_slots({self._mamba_sub_pool_name!r}) - 1 "
            f"= {self._shared_mamba_size}"
        )
        assert len(cache_params.layers) >= len(mamba_layer_ids), (
            f"cache_params.layers ({len(cache_params.layers)}) cannot supply "
            f"{len(mamba_layer_ids)} mamba layer ids"
        )
        self.mamba_pool = UnifiedMambaPool(
            unified_buffer=self._unified_buffer,
            sub_pool_name=self._mamba_sub_pool_name,
            spec_state_size=mamba_spec_state_size,
            mamba_layer_ids=mamba_layer_ids,
            enable_memory_saver=self.enable_memory_saver,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )

        self.mamba_allocator = None
        self.mamba_map = {layer_id: i for i, layer_id in enumerate(mamba_layer_ids)}
        self.mamba_ckpt_pool = None
        self.device = device


        req_pool_size = self.req_to_token.shape[0]
        self.req_index_to_mamba_index_mapping: torch.Tensor = torch.zeros(
            req_pool_size, dtype=torch.int32, device=self.device
        )
        if enable_mamba_extra_buffer:
            self.req_index_to_mamba_ping_pong_track_buffer_mapping: torch.Tensor = (
                torch.zeros(
                    (req_pool_size, self.mamba_ping_pong_track_buffer_size),


                    dtype=torch.int64,
                    device=self.device,
                )
            )

    def translate_mamba_indices(self, virtual_ids: torch.Tensor) -> torch.Tensor:

        return self.mamba_allocator.translate(virtual_ids).to(torch.int32)


class UnifiedHybridLinearKVPool(HybridLinearKVPool):


    def get_kv_layer_ids(self):


        return []


class UnifiedPoolBundle(NamedTuple):
    unified_memory_pool: UnifiedKVPool
    token_to_kv_pool: object
    token_to_kv_pool_allocator: object
    req_to_token_pool: object


def init_unified_mamba_pools(
    *,
    device: str,
    kv_cache_dtype: torch.dtype,
    head_num: int,
    head_dim: int,
    page_size: int,
    start_layer: int,
    end_layer: int,
    is_draft_worker: bool,
    use_mla_backend: bool,
    kv_lora_rank: Optional[int] = None,
    qk_rope_head_dim: Optional[int] = None,
    mamba_layer_ids: List[int],
    full_attention_layer_ids: List[int],
    mamba2_cache_params,
    model_context_len: int,
    extra_max_context_len: int,
    max_total_num_tokens: int,
    max_mamba_cache_size: int,
    max_num_reqs: int,
    enable_memory_saver: bool,
    enable_mamba_extra_buffer: bool,
    speculative_num_draft_tokens: Optional[int],
    disable_overlap_schedule: bool,
    need_sort: bool,
    mamba_full_memory_ratio: Optional[float] = None,
    forward_stream: Optional[torch.cuda.Stream] = None,
    lazy_compaction: bool = False,
    decode_pre_alloc_size: int = 0,
) -> UnifiedPoolBundle:

    from sglang.srt.mem_cache.multi_ended_allocator import (
        UnifiedMambaTokenToKVPoolAllocator,
    )


    assert page_size >= 1, f"page_size must be >= 1, got {page_size}"

    store_dtype = _store_dtype_for(kv_cache_dtype)

    if use_mla_backend:
        assert kv_lora_rank and qk_rope_head_dim, (
            "init_unified_mamba_pools: MLA-hybrid-Mamba needs kv_lora_rank and "
            f"qk_rope_head_dim; got {kv_lora_rank} / {qk_rope_head_dim}"
        )
        assert not is_draft_worker, (
            "init_unified_mamba_pools: draft workers (speculative decoding) are "
            "not supported with the MLA unified pool"
        )
        full_spec = MLASubPoolSpec(
            name="full",
            layer_num=len(full_attention_layer_ids),
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            store_dtype=store_dtype,
            grow_direction="down",
        )
    else:
        full_spec = MHASubPoolSpec(
            name="full",
            layer_num=len(full_attention_layer_ids),
            head_num=head_num,
            head_dim=head_dim,
            store_dtype=store_dtype,
            grow_direction="down",
        )
    cp = mamba2_cache_params
    mamba_spec = MambaSubPoolSpec(
        name="mamba",
        layer_num=len(mamba_layer_ids),
        conv_state_shapes=tuple(tuple(int(x) for x in s) for s in cp.shape.conv),
        conv_dtype=cp.dtype.conv,
        temporal_state_shape=tuple(int(x) for x in cp.shape.temporal),
        temporal_dtype=cp.dtype.temporal,
        conv_slice_axis=getattr(cp.shape, "conv_slice_axis", 0),
        grow_direction="up",
    )
    total_bytes = (
        max_total_num_tokens * full_spec.entry_bytes()
        + max_mamba_cache_size * mamba_spec.entry_bytes()
    )


    view_tail_pad_bytes = page_size * full_spec.entry_bytes() if use_mla_backend else 0
    shared_pool = UnifiedKVPool(
        total_bytes=total_bytes,
        sub_pool_specs=[full_spec, mamba_spec],
        device=device,
        enable_memory_saver=enable_memory_saver,
        page_size=page_size,
        view_tail_pad_bytes=view_tail_pad_bytes,
    )
    req_to_token_pool = UnifiedHybridReqToTokenPool(
        unified_buffer=shared_pool,
        mamba_sub_pool_name="mamba",
        size=max_num_reqs,
        mamba_spec_state_size=max_num_reqs,
        max_context_len=model_context_len + extra_max_context_len,
        device=device,
        enable_memory_saver=enable_memory_saver,
        cache_params=mamba2_cache_params,
        mamba_layer_ids=mamba_layer_ids,
        enable_mamba_extra_buffer=enable_mamba_extra_buffer,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
        enable_overlap_schedule=not disable_overlap_schedule,
        start_layer=start_layer,
        pre_alloc_size=decode_pre_alloc_size,
    )
    if use_mla_backend:


        unified_full_kv_pool = UnifiedMLATokenToKVPool(
            unified_buffer=shared_pool,
            sub_pool_name="full",
            kv_cache_dtype=kv_cache_dtype,
            page_size=page_size,
        )
    else:
        unified_full_kv_pool = UnifiedMHATokenToKVPool(
            unified_buffer=shared_pool,
            sub_pool_name="full",
            page_size=page_size,
            start_layer=start_layer,
            end_layer=end_layer,
        )
    full_attn_layer_ids_for_pool = (
        [0] if is_draft_worker else list(full_attention_layer_ids)
    )
    token_to_kv_pool = UnifiedHybridLinearKVPool(
        page_size=page_size,
        size=max_total_num_tokens,
        dtype=kv_cache_dtype,
        head_num=head_num,
        head_dim=head_dim,
        full_attention_layer_ids=full_attn_layer_ids_for_pool,
        device=device,
        mamba_pool=req_to_token_pool.mamba_pool,
        enable_memory_saver=enable_memory_saver,
        use_mla=use_mla_backend,
        start_layer=start_layer,
        full_kv_pool=unified_full_kv_pool,
    )
    allocator = UnifiedMambaTokenToKVPoolAllocator(
        unified_buffer=shared_pool,
        kvcache=token_to_kv_pool,
        device=device,
        page_size=page_size,
        need_sort=need_sort,
        forward_stream=forward_stream,
        lazy_compaction=lazy_compaction,
        full_kernel_page_multiplier=(
            len(full_attention_layer_ids) if use_mla_backend else 1
        ),
    )


    mamba_slot_allocator = UnifiedMambaSlotAllocator(
        allocator.mamba_allocator,
        max_size=req_to_token_pool._shared_mamba_size,
        device=device,
    )

    req_to_token_pool.mamba_allocator = mamba_slot_allocator
    token_to_kv_pool._mamba_translate = mamba_slot_allocator.translate
    if use_mla_backend:


        token_to_kv_pool._full_translate = allocator.translate_kv_loc_dense

    logger.info(
        "[unified-memory-pool] ============================================================"
    )
    logger.info(
        "[unified-memory-pool] UNIFIED MEMORY POOL ENABLED -- path=Mamba hybrid (%s full side)",
        "MLA" if use_mla_backend else "MHA",
    )
    if use_mla_backend:
        logger.info(
            "[unified-memory-pool]   full_layers=%d, mamba_layers=%d, kv_lora_rank=%d, "
            "qk_rope_head_dim=%d, page_size=%d (dense views, kernel_page_multiplier=%d, "
            "view_tail_pad=%d B)",
            len(full_attention_layer_ids),
            len(mamba_layer_ids),
            kv_lora_rank,
            qk_rope_head_dim,
            page_size,
            len(full_attention_layer_ids),
            view_tail_pad_bytes,
        )
    else:
        logger.info(
            "[unified-memory-pool]   full_layers=%d, mamba_layers=%d, head_num=%d, head_dim=%d, "
            "page_size=%d, is_draft_worker=%s",
            len(full_attention_layer_ids),
            len(mamba_layer_ids),
            head_num,
            head_dim,
            page_size,
            is_draft_worker,
        )
    logger.info(
        "[unified-memory-pool]   total_bytes=%d, max_total_num_tokens=%d, max_mamba_cache_size=%d, "
        "max_num_reqs=%d, speculative_num_draft_tokens=%s",
        total_bytes,
        max_total_num_tokens,
        max_mamba_cache_size,
        max_num_reqs,
        speculative_num_draft_tokens,
    )
    if mamba_full_memory_ratio is not None:
        logger.info(
            "[unified-memory-pool]   mamba_full_memory_ratio=%s governs the total budget only, "
            "not the runtime split.",
            mamba_full_memory_ratio,
        )
    logger.info(
        "[unified-memory-pool] ============================================================"
    )
    return UnifiedPoolBundle(
        unified_memory_pool=shared_pool,
        token_to_kv_pool=token_to_kv_pool,
        token_to_kv_pool_allocator=allocator,
        req_to_token_pool=req_to_token_pool,
    )


class UnifiedSWAKVPool(SWAKVPool):


    def __init__(
        self,
        *,
        unified_buffer: UnifiedKVPool,
        swa_attention_layer_ids: List[int],
        full_attention_layer_ids: List[int],
        page_size: int = 1,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_memory_saver: bool = False,
    ):

        self.unified_buffer = unified_buffer
        self.swa_layer_nums = len(swa_attention_layer_ids)
        self.full_layer_nums = len(full_attention_layer_ids)
        self.layer_num = self.full_layer_nums + self.swa_layer_nums
        self.start_layer = start_layer if start_layer is not None else 0
        self.page_size = page_size
        self.layer_transfer_counter = None

        self.size = unified_buffer.max_slots("full") - 1
        self.size_swa = unified_buffer.max_slots("swa") - 1

        full_spec = unified_buffer.mha_spec("full")
        swa_spec = unified_buffer.mha_spec("swa")
        assert full_spec.store_dtype == swa_spec.store_dtype, (
            "UnifiedSWAKVPool: full and swa sub-pools must share store_dtype; got "
            f"full={full_spec.store_dtype}, swa={swa_spec.store_dtype}"
        )
        self.dtype = full_spec.store_dtype
        self.head_num = full_spec.head_num
        self.head_dim = full_spec.head_dim
        self.device = unified_buffer.device

        self.full_kv_pool = UnifiedMHATokenToKVPool(
            unified_buffer=unified_buffer,
            sub_pool_name="full",
            page_size=page_size,
            start_layer=start_layer,
            end_layer=end_layer,
        )
        self.swa_kv_pool = UnifiedMHATokenToKVPool(
            unified_buffer=unified_buffer,
            sub_pool_name="swa",
            page_size=page_size,
            start_layer=start_layer,
            end_layer=end_layer,
        )


        self.enable_custom_mem_pool = False
        self.custom_mem_pool = None


        self.layers_mapping: Dict[int, Tuple[int, bool]] = {}
        for idx, gid in enumerate(full_attention_layer_ids):
            self.layers_mapping[gid] = (idx, False)
        for idx, gid in enumerate(swa_attention_layer_ids):
            self.layers_mapping[gid] = (idx, True)


        self.full_to_swa_index_mapping: Optional[torch.Tensor] = None

        self.mem_usage = 0.0


        self._full_allocator = None
        self._swa_allocator = None

        logger.info(
            "[unified-memory-pool] UnifiedSWAKVPool wrapped unified buffer: "
            "full_layers=%d (max_slots=%d), swa_layers=%d (max_slots=%d), "
            "head_num=%d, head_dim=%d",
            self.full_layer_nums,
            unified_buffer.max_slots("full"),
            self.swa_layer_nums,
            unified_buffer.max_slots("swa"),
            self.head_num,
            self.head_dim,
        )


    def attach_allocators(self, *, full_allocator, swa_allocator) -> None:

        self._full_allocator = full_allocator
        self._swa_allocator = swa_allocator


    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor) -> None:
        return

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):

        assert self._swa_allocator is not None, (
            "UnifiedSWAKVPool.translate_loc_from_full_to_swa called before "
            "attach_allocators"
        )
        ps = self._swa_allocator.page_size


        if ps == 1:
            swa_locs = self._swa_allocator.virtual_to_physical[kv_indices]
        else:
            virt_pages = kv_indices // ps
            offsets = kv_indices % ps
            swa_phys_pages = self._swa_allocator.virtual_to_physical[virt_pages]

            swa_locs = swa_phys_pages * ps + offsets
        return swa_locs.clamp(min=0).to(torch.int32)

    def get_state_buf_infos(self):
        return self.swa_kv_pool.get_contiguous_buf_infos()


    def get_kv_size_bytes(self):
        return 0, 0

    def get_contiguous_buf_infos(self):
        return self.full_kv_pool.get_contiguous_buf_infos()


    def get_key_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        pool_layer_id, is_swa = self.layers_mapping[layer_id]
        pool = self.swa_kv_pool if is_swa else self.full_kv_pool
        return pool.get_key_buffer(pool_layer_id)

    def get_value_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        pool_layer_id, is_swa = self.layers_mapping[layer_id]
        pool = self.swa_kv_pool if is_swa else self.full_kv_pool
        return pool.get_value_buffer(pool_layer_id)

    def get_kv_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        pool_layer_id, is_swa = self.layers_mapping[layer_id]
        pool = self.swa_kv_pool if is_swa else self.full_kv_pool
        return pool.get_kv_buffer(pool_layer_id)


    def set_kv_buffer(
        self,
        layer,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ):

        _, swa_loc, full_loc = unwrap_write_loc(loc_info)
        layer_id = layer.layer_id
        pool_layer_id, is_swa = self.layers_mapping[layer_id]
        if is_swa:


            assert swa_loc is not None, (
                "UnifiedSWAKVPool.set_kv_buffer: SWA layer received no swa_loc; the "
                "attention backend must bundle forward_metadata.swa_out_cache_loc."
            )
            self.swa_kv_pool.set_kv_buffer(
                None,
                swa_loc,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=pool_layer_id,
            )
            return

        assert full_loc is not None, (
            "UnifiedSWAKVPool.set_kv_buffer: full layer received no full_loc; "
            "ForwardMetadata.out_cache_loc_full_physical must be precomputed for "
            "the unified memory pool."
        )
        self.full_kv_pool.set_kv_buffer(
            None,
            full_loc,
            cache_k,
            cache_v,
            k_scale,
            v_scale,
            layer_id_override=pool_layer_id,
        )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):


        raise NotImplementedError(
            "UnifiedSWAKVPool.move_kv_cache should not be called; compaction "
            "operates per-sub-pool via UnifiedMHATokenToKVPool.move_kv_cache."
        )


    @staticmethod
    def _virt_tokens_to_phys_tokens(
        virt_tokens: torch.Tensor, allocator
    ) -> torch.Tensor:

        ps = allocator.page_size
        if ps == 1:
            return allocator.virtual_to_physical[virt_tokens]
        virt_pages = virt_tokens // ps
        offsets = virt_tokens % ps
        phys_pages = allocator.virtual_to_physical[virt_pages]
        return phys_pages * ps + offsets

    def get_cpu_copy(self, indices, mamba_indices=None):
        assert self._full_allocator is not None
        assert self._swa_allocator is not None

        full_phys = self._virt_tokens_to_phys_tokens(indices, self._full_allocator)
        swa_phys = self._virt_tokens_to_phys_tokens(indices, self._swa_allocator)
        full_cpu = self.full_kv_pool.get_cpu_copy(full_phys)
        valid = swa_phys >= 0
        swa_cpu = None
        if bool(valid.any().item()):
            swa_cpu = self.swa_kv_pool.get_cpu_copy(swa_phys[valid])
        return {"full": full_cpu, "swa": swa_cpu}

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        assert self._full_allocator is not None
        full_phys = self._virt_tokens_to_phys_tokens(indices, self._full_allocator)
        self.full_kv_pool.load_cpu_copy(kv_cache_cpu["full"], full_phys)
        if kv_cache_cpu.get("swa") is not None:
            assert self._swa_allocator is not None
            swa_phys = self._virt_tokens_to_phys_tokens(indices, self._swa_allocator)
            self.swa_kv_pool.load_cpu_copy(kv_cache_cpu["swa"], swa_phys)


class UnifiedSWAPoolBundle(NamedTuple):
    unified_memory_pool: UnifiedKVPool
    token_to_kv_pool: object
    token_to_kv_pool_allocator: object


def init_unified_swa_pools(
    *,
    device: str,
    kv_cache_dtype: torch.dtype,
    head_num: int,
    head_dim: int,
    v_head_dim: int,
    swa_head_num: int,
    swa_head_dim: int,
    swa_v_head_dim: int,
    page_size: int,
    start_layer: int,
    end_layer: int,
    swa_attention_layer_ids: List[int],
    full_attention_layer_ids: List[int],
    full_max_total_num_tokens: int,
    swa_max_total_num_tokens: int,
    enable_memory_saver: bool,
    need_sort: bool,
    forward_stream: Optional[torch.cuda.Stream] = None,
    lazy_compaction: bool = False,
) -> UnifiedSWAPoolBundle:

    from sglang.srt.mem_cache.multi_ended_allocator import (
        UnifiedSWATokenToKVPoolAllocator,
    )


    assert page_size >= 1, f"page_size must be >= 1, got {page_size}"
    assert (
        len(full_attention_layer_ids) > 0
    ), "SWA-hybrid with zero full-attention layers is degenerate"
    assert (
        len(swa_attention_layer_ids) > 0
    ), "SWA-hybrid with zero SWA-attention layers is degenerate"

    store_dtype = _store_dtype_for(kv_cache_dtype)

    full_spec = MHASubPoolSpec(
        name="full",
        layer_num=len(full_attention_layer_ids),
        head_num=head_num,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        store_dtype=store_dtype,
        grow_direction="down",
    )
    swa_spec = MHASubPoolSpec(
        name="swa",
        layer_num=len(swa_attention_layer_ids),
        head_num=swa_head_num,
        head_dim=swa_head_dim,
        v_head_dim=swa_v_head_dim,
        store_dtype=store_dtype,
        grow_direction="up",
    )
    total_bytes = (
        full_max_total_num_tokens * full_spec.entry_bytes()
        + swa_max_total_num_tokens * swa_spec.entry_bytes()
    )
    shared_pool = UnifiedKVPool(
        total_bytes=total_bytes,
        sub_pool_specs=[full_spec, swa_spec],
        device=device,
        enable_memory_saver=enable_memory_saver,
        page_size=page_size,
    )
    token_to_kv_pool = UnifiedSWAKVPool(
        unified_buffer=shared_pool,
        swa_attention_layer_ids=swa_attention_layer_ids,
        full_attention_layer_ids=full_attention_layer_ids,
        page_size=page_size,
        start_layer=start_layer,
        end_layer=end_layer,
        enable_memory_saver=enable_memory_saver,
    )
    allocator = UnifiedSWATokenToKVPoolAllocator(
        unified_buffer=shared_pool,
        kvcache=token_to_kv_pool,
        device=device,
        full_max_total_num_tokens=full_max_total_num_tokens,
        swa_max_total_num_tokens=swa_max_total_num_tokens,
        page_size=page_size,
        need_sort=need_sort,
        forward_stream=forward_stream,
        lazy_compaction=lazy_compaction,
    )

    logger.info(
        "[unified-memory-pool] ============================================================"
    )
    logger.info("[unified-memory-pool] UNIFIED MEMORY POOL ENABLED -- path=SWA hybrid")
    logger.info(
        "[unified-memory-pool]   full_layers=%d, swa_layers=%d, head_num=%d, head_dim=%d, "
        "v_head_dim=%d, swa_head_num=%d, swa_head_dim=%d, swa_v_head_dim=%d, "
        "page_size=%d",
        len(full_attention_layer_ids),
        len(swa_attention_layer_ids),
        head_num,
        head_dim,
        v_head_dim,
        swa_head_num,
        swa_head_dim,
        swa_v_head_dim,
        page_size,
    )
    logger.info(
        "[unified-memory-pool]   total_bytes=%d (=%.2f GB), full_max_total_num_tokens=%d, "
        "swa_max_total_num_tokens=%d, joint_available=%d slots",
        total_bytes,
        total_bytes / GB,
        full_max_total_num_tokens,
        swa_max_total_num_tokens,
        allocator.available_size(),
    )
    logger.info(
        "[unified-memory-pool] ============================================================"
    )
    return UnifiedSWAPoolBundle(
        unified_memory_pool=shared_pool,
        token_to_kv_pool=token_to_kv_pool,
        token_to_kv_pool_allocator=allocator,
    )
