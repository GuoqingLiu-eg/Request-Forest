# Copyright 2023-2024 SGLang Team
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
#
# Memory pool.
#
# SGLang has two levels of memory pool.
# ReqToTokenPool maps a request to its token locations.
# TokenToKVPoolAllocator manages the indices to kv cache data.
# KVCache actually holds the physical kv cache.

from __future__ import annotations

import abc
import copy
import dataclasses
import logging
import math
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, fields
from functools import cached_property
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import numpy as np
import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsa.quant_k_cache import (
    quantize_k_cache,
    quantize_k_cache_separate,
)
from sglang.kernels.ops.kvcache.cache_move import (
    copy_all_layer_kv_cache_func,
    set_kv_buffer_prefix_valid_tiled,
    store_cache_4d,
)
from sglang.kernels.ops.kvcache.kvcache import can_use_store_cache, store_cache
from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype, is_fp8_fnuz
from sglang.srt.configs.mamba_utils import BaseLinearStateParams
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.utils import aiter_can_use_preshuffle_paged_mqa
from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
    UnquantizedKVCacheMethod,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.mem_cache.index_key_cache import IndexKeyCache
from sglang.srt.mem_cache.kv_vmm_backing import KvVmmBufferOwner
from sglang.srt.mem_cache.layout.page_major import (
    build_page_major_mamba_views,
    build_page_major_mha_views,
    mamba_entry_bytes,
    mha_entry_bytes,
)
from sglang.srt.mem_cache.utils import (
    get_mla_kv_buffer_triton,
    maybe_init_custom_mem_pool,
    set_mla_kv_buffer_triton,
    set_mla_kv_buffer_triton_fp8_quant,
    set_mla_kv_scale_buffer_triton,
)
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_parallel, get_spec
from sglang.srt.utils import (
    cpu_has_amx_support,
    is_cpu,
    is_cuda,
    is_float4_e2m1fn_x2,
    is_hip,
    is_npu,
    next_power_of_2,
)
from sglang.srt.utils.async_probe import maybe_detect_oob
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import LayerDoneCounter
    from sglang.srt.managers.schedule_batch import Req


logger = logging.getLogger(__name__)


_MAMBA_DEBUG_ASSERTS = os.environ.get("SGLANG_MAMBA_DEBUG_ASSERTS", "0") == "1"

GB = 1024 * 1024 * 1024
_is_cuda = is_cuda()
_is_npu = is_npu()
_is_cpu = is_cpu()
_cpu_has_amx_support = cpu_has_amx_support()
_is_hip = is_hip()
_is_fp8_fnuz = is_fp8_fnuz()


_use_aiter = bool(envs.SGLANG_USE_AITER.get()) and _is_hip


def conv_window_dedup_enabled(
    is_npu: bool,
    is_cpu: bool,
    speculative_eagle_topk: Optional[int],
    is_kda: bool,
    *,
    is_tree_verify: bool = False,
) -> bool:

    return (
        not is_npu
        and not is_cpu
        and not is_kda
        and not is_tree_verify
        and (speculative_eagle_topk is None or speculative_eagle_topk <= 1)
    )


def get_tensor_size_bytes(t: Union[torch.Tensor, List[torch.Tensor]]):
    if isinstance(t, list):
        return sum(get_tensor_size_bytes(x) for x in t)
    return np.prod(t.shape) * t.dtype.itemsize


def _set_kv_buffer_impl(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    row_dim: int,
    store_dtype: torch.dtype,
    device_module: Any,
    size_limit: int,
    alt_stream: Optional[torch.cuda.Stream] = None,
    v_row_dim: Optional[int] = None,
) -> None:
    v_row_dim = row_dim if v_row_dim is None else v_row_dim
    row_bytes = row_dim * store_dtype.itemsize
    v_row_bytes = v_row_dim * store_dtype.itemsize
    if (_is_cuda or _is_hip) and can_use_store_cache(row_bytes, v_row_bytes):
        return store_cache(
            k.view(-1, row_dim),
            v.view(-1, v_row_dim),
            k_cache.view(-1, row_dim),
            v_cache.view(-1, v_row_dim),
            indices,
            row_bytes=row_bytes,
            v_row_bytes=v_row_bytes,
            size_limit=size_limit,
        )


    if _is_cpu and _cpu_has_amx_support and v_row_dim == row_dim:
        return torch.ops.sgl_kernel.store_cache_cpu(
            k,
            v,
            k_cache,
            v_cache,
            indices,
            row_dim,
        )

    from sglang.srt.model_executor.runner import get_is_capture_mode

    if get_is_capture_mode() and alt_stream is not None:
        current_stream = device_module.current_stream()
        alt_stream.wait_stream(current_stream)
        k_cache[indices] = k
        with device_module.stream(alt_stream):
            v_cache[indices] = v
        current_stream.wait_stream(alt_stream)
    else:
        k_cache[indices] = k
        v_cache[indices] = v


def _set_kv_buffer_prefix_valid_impl(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    loc_2d: torch.Tensor,
    commit_lens: torch.Tensor,
    row_dim: int,
    store_dtype: torch.dtype,
) -> None:
    if k.numel() == 0 or loc_2d.numel() == 0 or commit_lens.numel() == 0:
        return

    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()
    if not loc_2d.is_contiguous():
        loc_2d = loc_2d.contiguous()
    if not commit_lens.is_contiguous():
        commit_lens = commit_lens.contiguous()

    row_bytes = row_dim * store_dtype.itemsize
    if row_bytes <= 0:
        return

    if row_bytes >= 8192:
        bytes_per_tile = 512
        num_warps = 8
    elif row_bytes >= 4096:
        bytes_per_tile = 256
        num_warps = 4
    else:
        bytes_per_tile = 128
        num_warps = 4

    grid = (
        int(loc_2d.shape[0]),
        int(loc_2d.shape[1]),
        triton.cdiv(row_bytes, bytes_per_tile),
    )

    set_kv_buffer_prefix_valid_tiled[grid](
        k,
        v,
        k_cache,
        v_cache,
        loc_2d,
        commit_lens,
        int(k.stride(0) * k.element_size()),
        int(v.stride(0) * v.element_size()),
        int(k_cache.stride(0) * k_cache.element_size()),
        int(v_cache.stride(0) * v_cache.element_size()),
        int(loc_2d.shape[1]),
        ROW_BYTES=row_bytes,
        BYTES_PER_TILE=bytes_per_tile,
        num_warps=num_warps,
        num_stages=2,
    )


class ReqToTokenPool:


    enable_mamba_extra_buffer_lazy: bool = False

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
    ):
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self.size = size


        self._alloc_size = size + 1
        self.max_context_len = max_context_len
        self.device = device
        with memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.req_to_token = torch.zeros(
                (self._alloc_size, max_context_len), dtype=torch.int32, device=device
            )
        self.free_slots = list(range(1, self._alloc_size))
        self.req_generation = torch.zeros(self._alloc_size, dtype=torch.int64)
        self._aux_cache: Any = None

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, reqs: list[Req]) -> Optional[List[int]]:


        reusing = [i for i, r in enumerate(reqs) if r.req_pool_idx is not None]


        assert all(
            reqs[i].inflight_middle_chunks > 0 or reqs[i].kv_committed_len > 0
            for i in reusing
        ), "reusing request must be chunked or have committed KV"

        need_size = len(reqs) - len(reusing)
        if need_size > len(self.free_slots):
            return None
        if need_size > 0:


            select_index = self.free_slots[-need_size:]
            del self.free_slots[-need_size:]
        else:

            select_index = []
        offset = 0
        for r in reqs:
            if r.req_pool_idx is None:
                r.req_pool_idx = select_index[offset]
                self.req_generation[r.req_pool_idx] += 1
                offset += 1
        return [r.req_pool_idx for r in reqs]

    def free(self, req: Req):
        assert req.req_pool_idx is not None, "request must have req_pool_idx"
        if self._aux_cache is not None:
            self._aux_cache.free(req.req_pool_idx)
        self.free_slots.append(req.req_pool_idx)
        req.req_pool_idx = None

    def clear(self):
        self.free_slots = list(range(1, self._alloc_size))
        self.req_generation.zero_()
        if self._aux_cache is not None:
            self._aux_cache.clear()

    def attach_aux_cache(self, aux_cache: Any) -> None:
        assert self._aux_cache is None
        self._aux_cache = aux_cache

    def reset_aux_cache_allocator(self) -> None:
        if self._aux_cache is not None:
            self._aux_cache.reset_allocator()

    def schedulable_token_capacity(self, physical_capacity: int) -> int:
        if self._aux_cache is None:
            return physical_capacity
        return self._aux_cache.dense_capacity

    def alloc_aux_to_lengths(
        self,
        *,
        req_pool_indices_cpu: torch.Tensor,
        target_seq_lens_cpu: torch.Tensor,
    ) -> None:
        if self._aux_cache is not None:
            self._aux_cache.alloc_to_lengths(
                req_pool_indices_cpu=req_pool_indices_cpu,
                target_seq_lens_cpu=target_seq_lens_cpu,
            )


class MambaPool:


    conv_window_axis = -1

    @dataclass(frozen=True, kw_only=True)
    class State:
        conv: List[torch.Tensor]
        temporal: torch.Tensor


        replayssm_d: Optional[torch.Tensor] = None
        replayssm_k: Optional[torch.Tensor] = None
        replayssm_g: Optional[torch.Tensor] = None
        replayssm_rawv: Optional[torch.Tensor] = None
        replayssm_rawk: Optional[torch.Tensor] = None
        replayssm_beta: Optional[torch.Tensor] = None

        def at_layer_idx(self, layer: int):
            kwargs = {}

            for f in fields(self):
                name = f.name
                v = getattr(self, name)
                if v is None:
                    kwargs[name] = None
                elif name in ("conv", "intermediate_conv_window"):
                    kwargs[name] = [conv[layer] for conv in v]
                else:
                    kwargs[name] = v[layer]

            return type(self)(**kwargs)

        def mem_usage_bytes(self):
            return sum(
                get_tensor_size_bytes(getattr(self, f.name))
                for f in dataclasses.fields(self)
                if getattr(self, f.name) is not None
            )

    @dataclass(frozen=True, kw_only=True)
    class SpeculativeState(State):


        intermediate_ssm: Optional[torch.Tensor]
        intermediate_conv_window: List[torch.Tensor]

    def _detect_conv_window_axis(
        self, conv_state_shape: List[Tuple[int, int]], win_len: int
    ) -> int:

        axis = None
        for conv_shape in conv_state_shape:
            if conv_shape[-1] == win_len:
                shape_axis = len(conv_shape) - 1
            elif conv_shape[0] == win_len:
                shape_axis = 0
            else:
                raise ValueError(
                    f"conv_state shape {conv_shape} has no axis of length "
                    f"conv_kernel-1={win_len}; cannot build the deduplicated "
                    "sliding-window conv-intermediate view."
                )
            if axis is None:
                axis = shape_axis
            elif axis != shape_axis:
                raise ValueError(
                    "inconsistent conv-window axis across conv shapes "
                    f"{conv_state_shape}; a single conv_window_axis cannot serve "
                    "mixed layouts."
                )
        return axis

    def _allocate_deduplicated_conv_window(
        self,
        *,
        conv_shape: Tuple[int, int],
        num_mamba_layers: int,
        spec_state_size: int,
        speculative_num_draft_tokens: int,
        conv_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        window_axis = self.conv_window_axis % len(conv_shape)
        win = conv_shape[window_axis]
        physical_conv_shape = list(conv_shape)
        physical_conv_shape[window_axis] = speculative_num_draft_tokens + win - 1
        phys = torch.zeros(
            (
                num_mamba_layers,
                spec_state_size + 1,
                *physical_conv_shape,
            ),
            dtype=conv_dtype,
            device="cuda",
        )
        physical_conv_strides = phys.stride()[2:]
        window_stride = physical_conv_strides[window_axis]
        view = phys.as_strided(
            (
                phys.shape[0],
                phys.shape[1],
                speculative_num_draft_tokens,
                *conv_shape,
            ),
            (
                phys.stride(0),
                phys.stride(1),
                window_stride,
                *physical_conv_strides,
            ),
        )
        return phys, view

    def __init__(
        self,
        *,
        size: int,
        spec_state_size: int,
        cache_params: BaseLinearStateParams,
        mamba_layer_ids: List[int],
        device: str,
        enable_memory_saver: bool = False,
        speculative_num_draft_tokens: Optional[int] = None,
        speculative_eagle_topk: Optional[int] = None,
        enable_linear_replayssm: bool = False,
        linear_replayssm_cache_len: int = 16,
        envelope_layout: bool = False,
        enable_linear_replayssm_spec: bool = False,
        spec_ssm_row_capacity: Optional[int] = None,
    ):
        conv_state_shape = cache_params.shape.conv
        temporal_state_shape = cache_params.shape.temporal
        conv_dtype = cache_params.dtype.conv
        ssm_dtype = cache_params.dtype.temporal
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        num_mamba_layers = len(mamba_layer_ids)
        self.mamba_layer_ids = list(mamba_layer_ids)

        self.size = size
        self.device = device
        self.spec_ssm_row_capacity = spec_ssm_row_capacity
        self.debug_memory_pool = envs.SGLANG_DEBUG_MEMORY_POOL.get()
        self.enable_linear_replayssm = enable_linear_replayssm
        self.linear_replayssm_cache_len = linear_replayssm_cache_len


        self.enable_linear_replayssm_spec = enable_linear_replayssm_spec
        self.replayssm_spec_fold = bool(enable_linear_replayssm_spec)
        _replayssm_on = enable_linear_replayssm or enable_linear_replayssm_spec


        self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
            maybe_init_custom_mem_pool(device=self.device)
        )

        with (
            self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE),
            (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ),
        ):
            if envelope_layout:


                assert not _is_npu and not (
                    _is_cpu and _cpu_has_amx_support
                ), "envelope_layout mamba is only supported on the CUDA path"
                max_slots = size + 1
                entry_bytes = mamba_entry_bytes(
                    layer_num=num_mamba_layers,
                    conv_state_shapes=conv_state_shape,
                    conv_dtype=conv_dtype,
                    temporal_state_shape=temporal_state_shape,
                    temporal_dtype=ssm_dtype,
                )
                self._raw = torch.zeros(
                    max_slots * entry_bytes, dtype=torch.uint8, device=device
                )
                conv_state, temporal_state = build_page_major_mamba_views(
                    self._raw,
                    layer_num=num_mamba_layers,
                    conv_state_shapes=conv_state_shape,
                    conv_dtype=conv_dtype,
                    temporal_state_shape=temporal_state_shape,
                    temporal_dtype=ssm_dtype,
                    max_slots=max_slots,
                )
            else:
                conv_state = [
                    torch.zeros(
                        size=(num_mamba_layers, size + 1) + conv_shape,
                        dtype=conv_dtype,
                        device=device,
                    )
                    for conv_shape in conv_state_shape
                ]

                if _is_npu:
                    from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                        _init_npu_conv_state,
                    )

                    conv_state = _init_npu_conv_state(
                        conv_state[0],
                        conv_state_shape,
                        speculative_num_draft_tokens,
                        is_kda=cache_params.is_kda,
                    )

                if _is_cpu and _cpu_has_amx_support:
                    from sglang.srt.layers.amx_utils import _init_amx_conv_state


                    conv_state = _init_amx_conv_state(conv_state)

                temporal_state = torch.zeros(
                    size=(num_mamba_layers, size + 1) + temporal_state_shape,
                    dtype=ssm_dtype,
                    device=device,
                )


            replayssm_d = replayssm_k = replayssm_g = None
            replayssm_rawv = replayssm_rawk = replayssm_beta = None
            if _replayssm_on:
                hv, v_dim, k_dim = temporal_state_shape
                h_k = getattr(cache_params.shape, "num_k_heads_per_tp", hv)
                L = linear_replayssm_cache_len
                num_slots = size + 1


                ring_dtype = conv_dtype if enable_linear_replayssm_spec else ssm_dtype


                if self.replayssm_spec_fold and not cache_params.is_kda:
                    record_len = (
                        speculative_num_draft_tokens
                        if speculative_num_draft_tokens is not None
                        else L
                    )
                else:
                    record_len = L
                if not self.replayssm_spec_fold or cache_params.is_kda:
                    replayssm_d = torch.zeros(
                        size=(num_mamba_layers, num_slots, hv, L, v_dim),
                        dtype=ring_dtype,
                        device=device,
                    )
                    replayssm_k = torch.zeros(
                        size=(num_mamba_layers, num_slots, h_k, L, k_dim),
                        dtype=ring_dtype,
                        device=device,
                    )


                g_shape = (
                    (num_mamba_layers, num_slots, hv, record_len, k_dim)
                    if cache_params.is_kda
                    else (num_mamba_layers, num_slots, hv, record_len)
                )
                replayssm_g = torch.zeros(
                    size=g_shape,
                    dtype=torch.float32,
                    device=device,
                )


                if enable_linear_replayssm_spec:
                    if cache_params.is_kda:


                        if L & (L - 1) != 0:
                            raise ValueError(
                                f"spec-verify ring length must be a power of two, got {L}"
                            )
                        if (
                            speculative_num_draft_tokens is not None
                            and L < 2 * speculative_num_draft_tokens
                        ):
                            raise ValueError(
                                f"spec-verify ring too small: {L} < "
                                f"2 * {speculative_num_draft_tokens} (early-flush margin)"
                            )
                    replayssm_rawv = torch.zeros(
                        size=(num_mamba_layers, num_slots, hv, record_len, v_dim),
                        dtype=conv_dtype,
                        device=device,
                    )
                    replayssm_rawk = torch.zeros(
                        size=(num_mamba_layers, num_slots, h_k, record_len, k_dim),
                        dtype=conv_dtype,
                        device=device,
                    )
                    replayssm_beta = torch.zeros(
                        size=(num_mamba_layers, num_slots, hv, record_len),
                        dtype=torch.float32,
                        device=device,
                    )

            if speculative_num_draft_tokens is not None:
                if _is_npu:
                    temporal_state = temporal_state.transpose(-1, -2)
                    temporal_state_shape = (
                        *temporal_state_shape[:-2],
                        temporal_state_shape[-1],
                        temporal_state_shape[-2],
                    )


                if enable_linear_replayssm_spec:
                    intermediate_ssm_state_cache = None
                else:


                    intermediate_ssm_state_cache = torch.zeros(
                        size=(
                            num_mamba_layers,
                            (
                                spec_state_size + 1
                                if spec_ssm_row_capacity is None
                                else spec_ssm_row_capacity
                            ),
                            (
                                speculative_num_draft_tokens
                                if spec_ssm_row_capacity is None
                                else 1
                            ),
                            temporal_state_shape[0],
                            temporal_state_shape[1],
                            temporal_state_shape[2],
                        ),
                        dtype=ssm_dtype,
                        device="cuda",
                    )


                spec_algorithm = get_spec().speculative_algorithm
                dedup_conv_window = (
                    not cache_params.shape.disable_conv_window_dedup
                    and conv_window_dedup_enabled(
                        _is_npu,
                        _is_cpu,
                        speculative_eagle_topk,
                        cache_params.is_kda,
                        is_tree_verify=(spec_algorithm or "").upper()
                        == "REQUEST_FOREST",
                    )
                )
                self._intermediate_conv_window_phys = []
                if dedup_conv_window:
                    win_len = cache_params.shape.conv_kernel - 1
                    self.conv_window_axis = self._detect_conv_window_axis(
                        conv_state_shape, win_len
                    )
                    intermediate_conv_window_cache = []
                    for conv_shape in conv_state_shape:
                        phys, view = self._allocate_deduplicated_conv_window(
                            conv_shape=conv_shape,
                            num_mamba_layers=num_mamba_layers,
                            spec_state_size=spec_state_size,
                            speculative_num_draft_tokens=speculative_num_draft_tokens,
                            conv_dtype=conv_dtype,
                        )
                        self._intermediate_conv_window_phys.append(phys)
                        intermediate_conv_window_cache.append(view)
                else:


                    dense_conv_shapes = [
                        (
                            (conv_shape[1], conv_shape[0])
                            if _is_npu and cache_params.is_kda
                            else conv_shape
                        )
                        for conv_shape in conv_state_shape
                    ]
                    intermediate_conv_window_cache = [
                        torch.zeros(
                            size=(
                                num_mamba_layers,
                                spec_state_size + 1,
                                speculative_num_draft_tokens,
                                conv_shape[0],
                                conv_shape[1],
                            ),
                            dtype=conv_dtype,
                            device="cuda",
                        )
                        for conv_shape in dense_conv_shapes
                    ]
                    self._intermediate_conv_window_phys = intermediate_conv_window_cache
                self.mamba_cache = self.SpeculativeState(
                    conv=conv_state,
                    temporal=temporal_state,
                    intermediate_ssm=intermediate_ssm_state_cache,
                    intermediate_conv_window=intermediate_conv_window_cache,
                    replayssm_d=replayssm_d,
                    replayssm_k=replayssm_k,
                    replayssm_g=replayssm_g,
                    replayssm_rawv=replayssm_rawv,
                    replayssm_rawk=replayssm_rawk,
                    replayssm_beta=replayssm_beta,
                )
                intermediate_ssm_gb = (
                    get_tensor_size_bytes(intermediate_ssm_state_cache) / GB
                    if intermediate_ssm_state_cache is not None
                    else 0.0
                )
                logger.info(
                    f"Mamba Cache is allocated. "
                    f"max_mamba_cache_size: {size}, "
                    f"conv_state size: {get_tensor_size_bytes(conv_state) / GB:.2f}GB, "
                    f"ssm_state size: {get_tensor_size_bytes(temporal_state) / GB:.2f}GB "
                    f"intermediate_ssm_state_cache size: {intermediate_ssm_gb:.2f}GB "


                    f"intermediate_conv_window_cache size: {get_tensor_size_bytes(self._intermediate_conv_window_phys) / GB:.2f}GB "
                )
            else:
                self.mamba_cache = self.State(
                    conv=conv_state,
                    temporal=temporal_state,
                    replayssm_d=replayssm_d,
                    replayssm_k=replayssm_k,
                    replayssm_g=replayssm_g,
                    replayssm_rawv=replayssm_rawv,
                    replayssm_rawk=replayssm_rawk,
                    replayssm_beta=replayssm_beta,
                )
                logger.info(
                    f"Mamba Cache is allocated. "
                    f"max_mamba_cache_size: {size}, "
                    f"conv_state size: {get_tensor_size_bytes(conv_state) / GB:.2f}GB, "
                    f"ssm_state size: {get_tensor_size_bytes(temporal_state) / GB:.2f}GB "
                )
            if _replayssm_on:
                logger.info(
                    f"GDN ReplaySSM ring buffers allocated "
                    f"(record_len={record_len}, fold={self.replayssm_spec_fold}): "
                    f"d={get_tensor_size_bytes(replayssm_d) / GB if replayssm_d is not None else 0.0:.3f}GB, "
                    f"k={get_tensor_size_bytes(replayssm_k) / GB if replayssm_k is not None else 0.0:.3f}GB, "
                    f"g={get_tensor_size_bytes(replayssm_g) / GB:.3f}GB "
                    + (
                        f"rawv={get_tensor_size_bytes(replayssm_rawv) / GB:.3f}GB, "
                        f"rawk={get_tensor_size_bytes(replayssm_rawk) / GB:.3f}GB, "
                        f"beta={get_tensor_size_bytes(replayssm_beta) / GB:.3f}GB "
                        if enable_linear_replayssm_spec
                        else ""
                    )
                )


            self.replayssm_is_kda = bool(_replayssm_on and cache_params.is_kda)


            self.replayssm_write_pos = (
                torch.zeros((size + 1,), dtype=torch.int32, device=device)
                if _replayssm_on and not self.replayssm_spec_fold
                else None
            )


            self.replayssm_cache_base = (
                torch.zeros((size + 1,), dtype=torch.int32, device=device)
                if enable_linear_replayssm_spec and not self.replayssm_spec_fold
                else None
            )
            self.replayssm_is_flush = (
                torch.zeros((size + 1,), dtype=torch.int8, device=device)
                if enable_linear_replayssm_spec and not self.replayssm_spec_fold
                else None
            )
            mem_usage_bytes = self.mamba_cache.mem_usage_bytes()
            if isinstance(self.mamba_cache, self.SpeculativeState):


                mem_usage_bytes -= get_tensor_size_bytes(
                    self.mamba_cache.intermediate_conv_window
                )
                mem_usage_bytes += get_tensor_size_bytes(
                    self._intermediate_conv_window_phys
                )
            self.mem_usage = mem_usage_bytes / GB
            self.num_mamba_layers = num_mamba_layers


        self.conv_shard_groups = getattr(cache_params.shape, "conv_shard_groups", None)
        self.conv_slice_axis = getattr(cache_params.shape, "conv_slice_axis", 0)

    def get_speculative_mamba2_params_all_layers(self) -> SpeculativeState:
        assert isinstance(self.mamba_cache, self.SpeculativeState)
        return self.mamba_cache

    def mamba2_layer_cache(self, layer_id: int):
        return self.mamba_cache.at_layer_idx(layer_id)


    @cached_property
    def _conv_fuse_ok(self) -> bool:

        convs = self.mamba_cache.conv
        return (
            not _is_npu
            and len(convs) > 0
            and convs[0].shape[0] > 0
            and convs[0].is_cuda
            and all(c.dtype == torch.bfloat16 and c.is_contiguous() for c in convs)
        )

    @cached_property
    def _conv_slot_desc(self):
        from sglang.srt.mem_cache.mamba_slot_fused import build_conv_slot_descriptor

        return build_conv_slot_descriptor(self.mamba_cache.conv)

    def _should_fuse_slot_ops(self) -> bool:
        return self._conv_fuse_ok and not envs.SGLANG_DISABLE_FUSED_MAMBA_SLOT_OPS.get()

    def clear_slots(self, indices: torch.Tensor):

        if self._should_fuse_slot_ops():
            from sglang.srt.mem_cache.mamba_slot_fused import fused_clear_conv_slots

            fused_clear_conv_slots(self._conv_slot_desc, indices)
            temporal = self.mamba_cache.temporal
            if temporal.numel() > 0:
                temporal[:, indices] = 0
            return
        if not _is_npu:
            need_size = len(indices)
            for i in range(len(self.mamba_cache.conv)):
                t = self.mamba_cache.conv[i]
                z = torch.zeros(1, dtype=t.dtype, device=t.device).expand(
                    t.shape[0], need_size, *t.shape[2:]
                )
                t[:, indices] = z
            t = self.mamba_cache.temporal
            z = torch.zeros(1, dtype=t.dtype, device=t.device).expand(
                t.shape[0], need_size, *t.shape[2:]
            )
            t[:, indices] = z
        else:
            for i in range(len(self.mamba_cache.conv)):
                t = self.mamba_cache.conv[i]
                t[:, indices] = 0
            t = self.mamba_cache.temporal
            t[:, indices] = 0

    def copy_from(self, src_indices: torch.Tensor, dst_indices: torch.Tensor):

        if self.replayssm_write_pos is not None and self.debug_memory_pool:

            src_wp = self.replayssm_write_pos[src_indices]
            assert bool((src_wp == 0).all().item()), (
                "copy_from requires a fully-flushed ReplaySSM source "
                f"(write_pos==0), got {src_wp.tolist()} for src "
                f"{src_indices.tolist()}"
            )
        if self._should_fuse_slot_ops():
            from sglang.srt.mem_cache.mamba_slot_fused import fused_copy_conv_slots

            if envs.SGLANG_DEBUG_MEMORY_POOL.get():
                overlap = set(src_indices.tolist()) & set(dst_indices.tolist())
                assert not overlap, (
                    "fused copy_from requires disjoint src/dst slots; "
                    f"overlap={sorted(overlap)}"
                )
            fused_copy_conv_slots(self._conv_slot_desc, src_indices, dst_indices)
            temporal = self.mamba_cache.temporal
            if temporal.numel() > 0:
                temporal[:, dst_indices] = temporal[:, src_indices]
        else:
            for i in range(len(self.mamba_cache.conv)):
                self.mamba_cache.conv[i][:, dst_indices] = self.mamba_cache.conv[i][
                    :, src_indices
                ]
            self.mamba_cache.temporal[:, dst_indices] = self.mamba_cache.temporal[
                :, src_indices
            ]
        if self.replayssm_write_pos is not None:
            self.replayssm_write_pos[dst_indices] = 0


        if self.replayssm_cache_base is not None:
            self.replayssm_cache_base[dst_indices] = 0
        if self.replayssm_is_flush is not None:
            self.replayssm_is_flush[dst_indices] = 0

    def get_cpu_copy(self, indices):
        current_platform.synchronize()
        conv_cpu = [
            conv[:, indices].to("cpu", non_blocking=True)
            for conv in self.mamba_cache.conv
        ]
        temporal_cpu = self.mamba_cache.temporal[:, indices].to(
            "cpu", non_blocking=True
        )


        if self.replayssm_cache_base is not None:
            cursors_cpu = (
                self.replayssm_write_pos[indices].to("cpu", non_blocking=True),
                self.replayssm_cache_base[indices].to("cpu", non_blocking=True),
                self.replayssm_is_flush[indices].to("cpu", non_blocking=True),
            )
            current_platform.synchronize()
            return conv_cpu, temporal_cpu, cursors_cpu
        current_platform.synchronize()
        return conv_cpu, temporal_cpu

    def load_cpu_copy(self, mamba_cache_cpu, indices):


        if len(mamba_cache_cpu) == 3:
            conv_cpu, temporal_cpu, cursors_cpu = mamba_cache_cpu
        else:
            conv_cpu, temporal_cpu = mamba_cache_cpu
            cursors_cpu = None
        current_platform.synchronize()
        for i, conv in enumerate(self.mamba_cache.conv):
            conv[:, indices] = conv_cpu[i].to(conv.device, non_blocking=True)
        self.mamba_cache.temporal[:, indices] = temporal_cpu.to(
            self.mamba_cache.temporal.device, non_blocking=True
        )
        if cursors_cpu is not None and self.replayssm_cache_base is not None:
            wp_cpu, cb_cpu, fl_cpu = cursors_cpu
            self.replayssm_write_pos[indices] = wp_cpu.to(
                self.replayssm_write_pos.device, non_blocking=True
            )
            self.replayssm_cache_base[indices] = cb_cpu.to(
                self.replayssm_cache_base.device, non_blocking=True
            )
            self.replayssm_is_flush[indices] = fl_cpu.to(
                self.replayssm_is_flush.device, non_blocking=True
            )
        current_platform.synchronize()

    _NON_TRANSFER_STATE_FIELDS = frozenset(
        {
            "intermediate_ssm",
            "intermediate_conv_window",
            "replayssm_d",
            "replayssm_k",
            "replayssm_g",
            "replayssm_rawv",
            "replayssm_rawk",
            "replayssm_beta",
        }
    )

    def _iter_transfer_state_tensors(self):

        for field, value in vars(self.mamba_cache).items():
            if field in self._NON_TRANSFER_STATE_FIELDS or value is None:
                continue
            tensors = value if isinstance(value, list) else [value]
            slice_axis = self.conv_slice_axis if field == "conv" else 0
            for state_tensor in tensors:


                if state_tensor.numel() == 0:
                    continue
                yield field, state_tensor, slice_axis

    def get_contiguous_buf_infos(self):

        data_ptrs, data_lens, item_lens = [], [], []

        for _, state_tensor, _ in self._iter_transfer_state_tensors():
            data_ptrs += [
                state_tensor[i].data_ptr() for i in range(self.num_mamba_layers)
            ]
            data_lens += [state_tensor[i].nbytes for i in range(self.num_mamba_layers)]
            item_lens += [
                state_tensor[i][0].nbytes for i in range(self.num_mamba_layers)
            ]
        return data_ptrs, data_lens, item_lens

    def get_state_dim_per_tensor(self):

        dim_per_tensor = []
        for _, state_tensor, slice_axis in self._iter_transfer_state_tensors():


            axis = 2 + slice_axis
            sliceable_dim = state_tensor.shape[axis]

            dim_per_tensor += [sliceable_dim] * self.num_mamba_layers
        return dim_per_tensor

    def get_state_layer_ids(self):

        state_tensor_count = sum(1 for _ in self._iter_transfer_state_tensors())
        return list(self.mamba_layer_ids) * state_tensor_count

    def get_state_slice_outer_counts(self):

        outer_counts = []
        for _, state_tensor, slice_axis in self._iter_transfer_state_tensors():
            outer_count = math.prod(state_tensor.shape[2 : 2 + slice_axis])
            outer_counts += [outer_count] * self.num_mamba_layers
        return outer_counts

    def get_state_conv_shard_groups(self):

        subdims_per_tensor = []
        for field, _, _ in self._iter_transfer_state_tensors():

            subdims = (
                list(self.conv_shard_groups)
                if field == "conv" and self.conv_shard_groups is not None
                else None
            )
            subdims_per_tensor += [subdims] * self.num_mamba_layers
        return subdims_per_tensor

    def get_kv_size_bytes(self):
        return self.mamba_cache.mem_usage_bytes()


class HybridReqToTokenPool(ReqToTokenPool):


    mamba_pool_cls = MambaPool

    def __init__(
        self,
        *,
        size: int,
        mamba_size: int,
        mamba_spec_state_size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        cache_params: BaseLinearStateParams,
        mamba_layer_ids: List[int],
        enable_mamba_extra_buffer: bool,
        enable_mamba_extra_buffer_lazy: bool = False,
        speculative_num_draft_tokens: int = None,
        speculative_eagle_topk: Optional[int] = None,
        enable_overlap_schedule: bool = True,
        start_layer: Optional[int] = None,
        enable_linear_replayssm: bool = False,
        linear_replayssm_cache_len: int = 16,
        mamba_envelope_layout: bool = False,
        enable_linear_replayssm_spec: bool = False,
        spec_ssm_row_capacity: Optional[int] = None,
    ):
        super().__init__(
            size=size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
        )

        self.mamba_ping_pong_track_buffer_size = 2 if enable_overlap_schedule else 1
        self.enable_mamba_extra_buffer = enable_mamba_extra_buffer
        self.enable_mamba_extra_buffer_lazy = enable_mamba_extra_buffer_lazy
        self.enable_memory_saver = enable_memory_saver
        self.start_layer = start_layer if start_layer is not None else 0
        self.layer_transfer_counter = None
        self._init_mamba_pool(
            mamba_size=mamba_size,
            mamba_spec_state_size=mamba_spec_state_size,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            device=device,
            enable_mamba_extra_buffer=enable_mamba_extra_buffer,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
            speculative_eagle_topk=speculative_eagle_topk,
            enable_linear_replayssm=enable_linear_replayssm,
            linear_replayssm_cache_len=linear_replayssm_cache_len,
            mamba_envelope_layout=mamba_envelope_layout,
            enable_linear_replayssm_spec=enable_linear_replayssm_spec,
            **(
                {"spec_ssm_row_capacity": spec_ssm_row_capacity}
                if spec_ssm_row_capacity is not None
                else {}
            ),
        )

    def _init_mamba_pool(
        self,
        mamba_size: int,
        mamba_spec_state_size: int,
        cache_params: BaseLinearStateParams,
        mamba_layer_ids: List[int],
        device: str,
        enable_mamba_extra_buffer: bool,
        speculative_num_draft_tokens: int = None,
        speculative_eagle_topk: Optional[int] = None,
        enable_linear_replayssm: bool = False,
        linear_replayssm_cache_len: int = 16,
        mamba_envelope_layout: bool = False,
        enable_linear_replayssm_spec: bool = False,
        spec_ssm_row_capacity: Optional[int] = None,
    ):
        self.mamba_pool = self.mamba_pool_cls(
            size=mamba_size,
            spec_state_size=mamba_spec_state_size,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            device=device,
            enable_memory_saver=self.enable_memory_saver,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
            speculative_eagle_topk=speculative_eagle_topk,
            enable_linear_replayssm=enable_linear_replayssm,
            linear_replayssm_cache_len=linear_replayssm_cache_len,
            envelope_layout=mamba_envelope_layout,
            enable_linear_replayssm_spec=enable_linear_replayssm_spec,
            **(
                {"spec_ssm_row_capacity": spec_ssm_row_capacity}
                if spec_ssm_row_capacity is not None
                else {}
            ),
        )
        self.mamba_allocator = MambaSlotAllocator(
            size=mamba_size,
            device=device,
        )
        self.mamba_map = {layer_id: i for i, layer_id in enumerate(mamba_layer_ids)}


        from sglang.srt.mem_cache.mamba_checkpoint_pool import (
            maybe_init_int8_mamba_checkpoint_pool,
        )

        self.mamba_ckpt_pool = maybe_init_int8_mamba_checkpoint_pool(
            mamba_size=mamba_size,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            device=device,
        )

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

    def clone_with_new_mamba(
        self,
        *,
        mamba_size: int,
        mamba_spec_state_size: int,
        cache_params: BaseLinearStateParams,
        device: str,
        enable_mamba_extra_buffer: bool,
        draft_model_idx: int,
        speculative_num_draft_tokens: int = None,
        speculative_eagle_topk: Optional[int] = None,
    ) -> HybridReqToTokenPool:

        clone = copy.copy(self)
        clone._init_mamba_pool(
            mamba_size=mamba_size,
            mamba_spec_state_size=mamba_spec_state_size,
            cache_params=cache_params,
            mamba_layer_ids=[draft_model_idx],
            device=device,
            enable_mamba_extra_buffer=enable_mamba_extra_buffer,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
            speculative_eagle_topk=speculative_eagle_topk,
        )
        clone.req_index_to_mamba_index_mapping = self.req_index_to_mamba_index_mapping
        if enable_mamba_extra_buffer:
            clone.req_index_to_mamba_ping_pong_track_buffer_mapping = (
                self.req_index_to_mamba_ping_pong_track_buffer_mapping
            )
        return clone

    def register_layer_transfer_counter(self, layer_transfer_counter: LayerDoneCounter):
        self.layer_transfer_counter = layer_transfer_counter


    def alloc(self, reqs: List[Req]) -> Optional[List[int]]:
        select_index = super().alloc(reqs)
        if select_index is None:
            return None

        mamba_indices: list[torch.Tensor] = []
        mamba_ping_pong_track_buffers: list[torch.Tensor] = []
        for req in reqs:
            if req.mamba_pool_idx is not None:
                pass
            else:
                mid = self.mamba_allocator.alloc(1)
                assert (
                    mid is not None
                ), f"Not enough space for mamba cache, try to increase --mamba-full-memory-ratio or --max-mamba-cache-size. {mid=}, {self.mamba_pool.size=}, {self.mamba_allocator.available_size()=}, {len(reqs)=}"
                req.mamba_pool_idx = mid[0]
                req.mamba_needs_clear = True


                if self.mamba_pool.replayssm_write_pos is not None:
                    self.mamba_pool.replayssm_write_pos[req.mamba_pool_idx] = 0


                if self.mamba_pool.replayssm_cache_base is not None:
                    self.mamba_pool.replayssm_cache_base[req.mamba_pool_idx] = 0
                    self.mamba_pool.replayssm_is_flush[req.mamba_pool_idx] = 0
            mamba_indices.append(req.mamba_pool_idx)
            if self.enable_mamba_extra_buffer:
                if req.mamba_ping_pong_track_buffer is None:
                    self._alloc_ping_pong_buffer(req)
                mamba_ping_pong_track_buffers.append(req.mamba_ping_pong_track_buffer)
        assert len(select_index) == len(
            mamba_indices
        ), "Not enough space for mamba cache, try to increase --mamba-full-memory-ratio or --max-mamba-cache-size."
        if self.enable_mamba_extra_buffer:
            assert len(select_index) == len(
                mamba_ping_pong_track_buffers
            ), "Not enough space for mamba ping pong idx, try to increase --mamba-full-memory-ratio."
        mamba_index_tensor = torch.stack(mamba_indices).to(dtype=torch.int32)
        self.req_index_to_mamba_index_mapping[select_index] = mamba_index_tensor
        if self.enable_mamba_extra_buffer:
            ping_pong_tensor = torch.stack(mamba_ping_pong_track_buffers)
            self.req_index_to_mamba_ping_pong_track_buffer_mapping[select_index] = (
                ping_pong_tensor
            )
        return select_index

    def get_mamba_indices(self, req_indices: torch.Tensor) -> torch.Tensor:
        return self.req_index_to_mamba_index_mapping[req_indices]

    def translate_mamba_indices(self, mamba_indices: torch.Tensor) -> torch.Tensor:

        return mamba_indices

    def mamba2_layer_index(self, layer_id: int) -> int:

        assert layer_id in self.mamba_map
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.mamba_map[layer_id]

    def mamba2_layer_cache(self, layer_id: int):
        return self.mamba_pool.mamba2_layer_cache(self.mamba2_layer_index(layer_id))

    def get_speculative_mamba2_params_all_layers(self) -> MambaPool.SpeculativeState:
        return self.mamba_pool.get_speculative_mamba2_params_all_layers()

    def get_state_buf_infos(self):
        return self.mamba_pool.get_contiguous_buf_infos()

    def get_state_dim_per_tensor(self):
        return self.mamba_pool.get_state_dim_per_tensor()

    def get_state_slice_outer_counts(self):
        return self.mamba_pool.get_state_slice_outer_counts()

    def get_state_conv_shard_groups(self):
        return self.mamba_pool.get_state_conv_shard_groups()

    def get_mamba_ping_pong_other_idx(self, mamba_next_track_idx: int) -> int:
        if self.mamba_ping_pong_track_buffer_size == 2:
            return 1 - mamba_next_track_idx
        else:
            return mamba_next_track_idx

    def get_mamba_ping_pong_keep_idx(self, req: Req) -> int:

        return req.mamba_last_track_idx

    def _alloc_ping_pong_buffer(self, req: Req):

        n = (
            1
            if self.enable_mamba_extra_buffer_lazy
            else self.mamba_ping_pong_track_buffer_size
        )
        slots = self.mamba_allocator.alloc(n)
        assert slots is not None, (
            "Not enough space for mamba ping pong idx, "
            "try to increase --mamba-full-memory-ratio."
        )
        buf = torch.full(
            (self.mamba_ping_pong_track_buffer_size,),
            -1,
            dtype=slots.dtype,
            device=slots.device,
        )
        buf[:n] = slots
        req.mamba_ping_pong_track_buffer = buf
        req.mamba_next_track_idx = 0
        req.mamba_last_track_idx = (
            0
            if self.enable_mamba_extra_buffer_lazy
            else self.get_mamba_ping_pong_other_idx(0)
        )

    def set_mamba_ping_pong_slot(self, req: Req, idx: int, value):

        req.mamba_ping_pong_track_buffer[idx] = value
        self.req_index_to_mamba_ping_pong_track_buffer_mapping[req.req_pool_idx] = (
            req.mamba_ping_pong_track_buffer
        )

    def donate_mamba_ping_pong_slot(
        self, req: Req, new_slot: torch.Tensor
    ) -> torch.Tensor:

        donate_idx = self.get_mamba_ping_pong_keep_idx(req)
        mamba_value_donated = (
            req.mamba_ping_pong_track_buffer[donate_idx].unsqueeze(-1).clone()
        )
        if _MAMBA_DEBUG_ASSERTS:

            assert mamba_value_donated.item() != -1, (
                f"Donated mamba slot is -1: donate_idx={donate_idx}, "
                f"buf={req.mamba_ping_pong_track_buffer.tolist()}, "
                f"next_track_idx={req.mamba_next_track_idx}, "
                f"rid={req.rid}"
            )
        self.set_mamba_ping_pong_slot(req, donate_idx, new_slot[0])
        return mamba_value_donated

    def free_mamba_cache(
        self, req: Req, mamba_ping_pong_track_buffer_to_keep: Optional[int] = None
    ):
        mamba_index = req.mamba_pool_idx
        assert mamba_index is not None, "double free? mamba_index is None"
        self.mamba_allocator.free(mamba_index.unsqueeze(0))
        req.mamba_pool_idx = None

        if self.enable_mamba_extra_buffer:
            mamba_ping_pong_track_buffer_to_free = (
                self.req_index_to_mamba_ping_pong_track_buffer_mapping[req.req_pool_idx]
            )
            if mamba_ping_pong_track_buffer_to_keep is not None:
                assert mamba_ping_pong_track_buffer_to_keep in [
                    0,
                    1,
                ], f"mamba_ping_pong_track_buffer_to_keep must be 0 or 1, {mamba_ping_pong_track_buffer_to_keep=}"


                if self.mamba_ping_pong_track_buffer_size == 2:
                    idx_to_free = 1 - mamba_ping_pong_track_buffer_to_keep
                    mamba_ping_pong_track_buffer_to_free = (
                        mamba_ping_pong_track_buffer_to_free[
                            idx_to_free : idx_to_free + 1
                        ]
                    )
                else:
                    assert self.mamba_ping_pong_track_buffer_size == 1, (
                        f"Unexpected mamba_ping_pong_track_buffer_size="
                        f"{self.mamba_ping_pong_track_buffer_size}"
                    )
                    assert mamba_ping_pong_track_buffer_to_keep == 0, (
                        "mamba_ping_pong_track_buffer_to_keep must be 0 when "
                        "mamba_ping_pong_track_buffer_size is 1"
                    )

                    mamba_ping_pong_track_buffer_to_free = (
                        mamba_ping_pong_track_buffer_to_free[0:0]
                    )
            if self.enable_mamba_extra_buffer_lazy:
                mamba_ping_pong_track_buffer_to_free = (
                    mamba_ping_pong_track_buffer_to_free[
                        mamba_ping_pong_track_buffer_to_free != -1
                    ]
                )
            self.mamba_allocator.free(mamba_ping_pong_track_buffer_to_free)


            req.mamba_ping_pong_track_buffer = None
            req.mamba_next_track_idx = None
            req.mamba_last_track_idx = None

    def clear(self):
        logger.info("Reset HybridReqToTokenPool")
        super().clear()
        self.mamba_allocator.clear()


        if self.mamba_ckpt_pool is not None:
            self.mamba_ckpt_pool.clear()
        self.req_index_to_mamba_index_mapping.zero_()
        if self.enable_mamba_extra_buffer:
            self.req_index_to_mamba_ping_pong_track_buffer_mapping.zero_()


@dataclass
class KVWriteLoc:


    loc: torch.Tensor
    swa_loc: Optional[torch.Tensor] = None
    full_loc: Optional[torch.Tensor] = None

    def __post_init__(self):


        if self.swa_loc is not None and self.swa_loc.shape[0] != self.loc.shape[0]:
            self.swa_loc = self.swa_loc[: self.loc.shape[0]]
        if self.full_loc is not None and self.full_loc.shape[0] != self.loc.shape[0]:
            self.full_loc = self.full_loc[: self.loc.shape[0]]


def unwrap_write_loc(loc_info):

    if isinstance(loc_info, KVWriteLoc):
        return loc_info.loc, loc_info.swa_loc, loc_info.full_loc
    return loc_info, None, None


class KvBufferDesc:


    __slots__ = ("name", "shape", "row_bytes", "tokens_per_row")

    def __init__(self, name: str, shape: tuple, *, row_bytes: int, tokens_per_row: int):
        self.name = name
        self.shape = tuple(shape)
        self.row_bytes = int(row_bytes)
        self.tokens_per_row = int(tokens_per_row)

    def _rows(self, num_tokens: int) -> int:
        n = max(int(num_tokens), 0)
        return (n + self.tokens_per_row - 1) // self.tokens_per_row

    def reserved_span_bytes(self, itemsize: int) -> int:

        return math.prod(self.shape) * itemsize

    def prefix_span_bytes(self, num_tokens: int, page_size: int) -> int:

        return self._rows(num_tokens) * self.row_bytes

    def final_span_bytes(self, num_tokens: int, page_size: int) -> int:

        return self._rows(max(int(num_tokens), 0) + page_size) * self.row_bytes

    def item_len_bytes(self, page_size: int) -> int:

        return (page_size // self.tokens_per_row) * self.row_bytes


class KVCache(abc.ABC):
    layer_shard_enabled: bool = False
    post_capture_active: bool = False


    cpu_copy_carries_mamba: bool = False

    @abc.abstractmethod
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        allocation_label: Optional[str] = None,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn, torch.float8_e4m3fnuz):

            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self.layer_num = layer_num
        self.start_layer = start_layer or 0
        self.end_layer = end_layer or layer_num - 1
        self.allocation_label = allocation_label
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        self.mem_usage = 0


        self.cpu_offloading_chunk_size = 8192


        self.layer_transfer_counter = None


        self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
            maybe_init_custom_mem_pool(device=self.device)
        )

    def _finalize_allocation_log(self, num_tokens: int):

        cache_name = (
            f"{self.allocation_label} KV Cache"
            if self.allocation_label is not None
            else "KV Cache"
        )
        kv_size_bytes = self.get_kv_size_bytes()
        if isinstance(kv_size_bytes, tuple):
            k_size, v_size = kv_size_bytes
            k_size_GB = k_size / GB
            v_size_GB = v_size / GB
            logger.info(
                f"{cache_name} {'VA upper bound' if self.post_capture_active else 'is allocated'}. dtype: {self.dtype}, "
                f"#tokens: {num_tokens}, K size: {k_size_GB:.2f} GB, "
                f"V size: {v_size_GB:.2f} GB"
            )
            self.mem_usage = k_size_GB + v_size_GB
        else:
            kv_size_GB = kv_size_bytes / GB
            logger.info(
                f"{cache_name} {'VA upper bound' if self.post_capture_active else 'is allocated'}. dtype: {self.dtype}, "
                f"#tokens: {num_tokens}, KV size: {kv_size_GB:.2f} GB"
            )
            self.mem_usage = kv_size_GB

    def get_kv_buffer_shape(self) -> Tuple[torch.Size, torch.Size]:
        k_buffer, v_buffer = self.get_kv_buffer(self.start_layer)
        return k_buffer.shape, v_buffer.shape

    @abc.abstractmethod
    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    @abc.abstractmethod
    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        raise NotImplementedError()

    def register_layer_transfer_counter(self, layer_transfer_counter: LayerDoneCounter):
        self.layer_transfer_counter = layer_transfer_counter

    def get_cpu_copy(self, indices, mamba_indices=None):
        raise NotImplementedError()

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        raise NotImplementedError()

    def get_kv_cache_quant_method(self) -> Any:

        fallback = None
        for pool in (
            self,
            getattr(self, "full_kv_pool", None),
            getattr(self, "swa_kv_pool", None),
        ):
            if pool is None:
                continue
            quant_method = getattr(pool, "quant_method", None)
            if quant_method is None:
                continue
            if getattr(quant_method, "name", None) != "unquantized":
                return quant_method
            fallback = quant_method
        return fallback

    def maybe_get_custom_mem_pool(self):
        return self.custom_mem_pool


class MHATokenToKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        v_head_dim: Optional[int] = None,
        swa_head_num: Optional[int] = None,
        swa_head_dim: Optional[int] = None,
        swa_v_head_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,
        enable_kv_cache_copy: bool = False,
        kv_cache_layout: Optional[str] = None,
        quant_method=None,
        post_capture_active: bool = False,
        allocation_label: Optional[str] = None,
    ):
        self.k_buffer = None
        self.v_buffer = None
        if post_capture_active:


            size = (size + page_size - 1) // page_size * page_size
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
            allocation_label,
        )
        self.post_capture_active = post_capture_active
        self._post_capture_owner = None
        self.head_num = swa_head_num if swa_head_num is not None else head_num
        self.head_dim = swa_head_dim if swa_head_dim is not None else head_dim
        self.v_head_dim = (
            swa_v_head_dim
            if swa_v_head_dim is not None
            else v_head_dim if v_head_dim is not None else head_dim
        )


        self.use_hnd = envs.SGLANG_USE_HND_KVCACHE.get()
        self.use_native_move_kv_cache = envs.SGLANG_NATIVE_MOVE_KV_CACHE.get()
        if kv_cache_layout is not None:


            self.use_hnd = False
            self.kv_cache_layout = kv_cache_layout
        elif self.use_hnd:
            total_slots = self.size + self.page_size
            assert total_slots % self.page_size == 0, (
                f"HND KV cache needs (size+page_size) divisible by page_size, got "
                f"size={self.size}, page_size={self.page_size}"
            )
            self.num_pages = total_slots // self.page_size
            self.kv_cache_layout = "hnd"
        else:
            self.kv_cache_layout = "nhd"
            if _use_aiter:
                layout = envs.SGLANG_AITER_KV_CACHE_LAYOUT.get().lower()
                if layout not in ("nhd", "vectorized_5d"):
                    raise ValueError(
                        f"Unsupported SGLANG_AITER_KV_CACHE_LAYOUT={layout!r}; "
                        "expected 'nhd' or 'vectorized_5d'."
                    )
                self.kv_cache_layout = layout
                if layout == "vectorized_5d":


                    self._kv_vector_x = 16 // self.store_dtype.itemsize
                    assert (self.size + self.page_size) % self.page_size == 0
                    assert self.page_size % self._kv_vector_x == 0, (
                        f"page_size={self.page_size} must be divisible by "
                        f"X={self._kv_vector_x} for vectorized_5d layout"
                    )
                    assert self.head_dim % self._kv_vector_x == 0
                    assert self.v_head_dim % self._kv_vector_x == 0

        self.quant_method = (
            quant_method if quant_method is not None else UnquantizedKVCacheMethod()
        )

        self._create_buffers()

        self.device_module = torch.get_device_module(self.device)

        _use_alt_stream = _is_cuda or current_platform.is_cuda_alike()
        self.alt_stream = (
            self.device_module.Stream()
            if _use_alt_stream and enable_alt_stream
            else None
        )

        if enable_kv_cache_copy and not self.use_hnd:


            self._init_kv_copy_and_warmup()
        else:
            self._kv_copy_config = None

        self._finalize_allocation_log(size)


        self.row_dim = self.head_num * self.head_dim
        self.v_row_dim = self.head_num * self.v_head_dim

    def _init_kv_copy_and_warmup(self):

        if self.layer_num == 0:
            self._kv_copy_config = None
            return


        _KV_COPY_STRIDE_THRESHOLD_LARGE = 8192
        _KV_COPY_STRIDE_THRESHOLD_MEDIUM = 4096
        _KV_COPY_TILE_SIZE_LARGE = 512
        _KV_COPY_TILE_SIZE_MEDIUM = 256
        _KV_COPY_TILE_SIZE_SMALL = 128
        _KV_COPY_NUM_WARPS_LARGE_TILE = 8
        _KV_COPY_NUM_WARPS_SMALL_TILE = 4

        stride_bytes = int(self.data_strides[0].item())
        if stride_bytes >= _KV_COPY_STRIDE_THRESHOLD_LARGE:
            bytes_per_tile = _KV_COPY_TILE_SIZE_LARGE
        elif stride_bytes >= _KV_COPY_STRIDE_THRESHOLD_MEDIUM:
            bytes_per_tile = _KV_COPY_TILE_SIZE_MEDIUM
        else:
            bytes_per_tile = _KV_COPY_TILE_SIZE_SMALL


        chunk_upper = 128 if bytes_per_tile >= _KV_COPY_TILE_SIZE_LARGE else 256

        self._kv_copy_config = {
            "bytes_per_tile": bytes_per_tile,
            "byte_tiles": (stride_bytes + bytes_per_tile - 1) // bytes_per_tile,
            "num_warps": (
                _KV_COPY_NUM_WARPS_SMALL_TILE
                if bytes_per_tile <= _KV_COPY_TILE_SIZE_MEDIUM
                else _KV_COPY_NUM_WARPS_LARGE_TILE
            ),
            "num_locs_upper": chunk_upper,
        }

        dummy_loc = torch.zeros(chunk_upper, dtype=torch.int64, device=self.device)
        copy_all_layer_kv_cache_func(
            self.data_ptrs,
            self.data_strides,
            dummy_loc,
            dummy_loc,
            1,
            chunk_upper,
            self._kv_copy_config,
        )

    @property
    def is_quantized_kv_cache(self) -> bool:
        return not isinstance(self.quant_method, UnquantizedKVCacheMethod)

    def _create_buffers(self):
        if self.is_quantized_kv_cache:
            if self.post_capture_active:
                raise NotImplementedError(
                    "Post-capture KV backing is not supported for quantized KV cache."
                )
            self._create_quantized_buffers()
        else:
            self.k_scale_buffer = None
            self.v_scale_buffer = None
            self.dq_k_buffer = None
            self.dq_v_buffer = None
            if self.post_capture_active:
                self._alloc_post_capture_buffers()
            else:
                self._create_buffers_normal()
        self._kv_buffer_descs = self._build_kv_buffer_descs()
        self._init_data_ptrs_and_strides()

    def _create_quantized_buffers(self):

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                buf = self.quant_method.create_buffers(
                    self.size + self.page_size,
                    self.head_num,
                    self.head_dim,
                    self.layer_num,
                    self.device,
                )
        self.k_buffer = buf["k_buffer"]
        self.v_buffer = buf["v_buffer"]
        self.k_scale_buffer = buf.get("k_scale_buffer")
        self.v_scale_buffer = buf.get("v_scale_buffer")
        self.dq_k_buffer = buf.get("dq_k_buffer")
        self.dq_v_buffer = buf.get("dq_v_buffer")
        self.store_dtype = buf.get("store_dtype", torch.uint8)
        self._check_quantized_buffer_access_requirements()

    def _check_quantized_buffer_access_requirements(self):
        expected_workspace_dtype = self.quant_method.dequant_workspace_dtype()
        has_k_workspace = self.dq_k_buffer is not None
        has_v_workspace = self.dq_v_buffer is not None
        if has_k_workspace != has_v_workspace:
            raise RuntimeError(
                f"KV cache method {self.quant_method.name!r} created only one "
                "dequant workspace buffer."
            )

        if expected_workspace_dtype is None:
            if has_k_workspace:
                raise RuntimeError(
                    f"KV cache method {self.quant_method.name!r} does not declare "
                    "DEQUANT_WORKSPACE access but created dequant buffers."
                )
            return

        if not has_k_workspace:
            raise RuntimeError(
                f"KV cache method {self.quant_method.name!r} declares "
                "DEQUANT_WORKSPACE access but did not create dequant buffers."
            )

        if (
            self.dq_k_buffer.dtype != expected_workspace_dtype
            or self.dq_v_buffer.dtype != expected_workspace_dtype
        ):
            raise RuntimeError(
                f"KV cache method {self.quant_method.name!r} declares dequant "
                f"workspace dtype {expected_workspace_dtype}, but created "
                f"{self.dq_k_buffer.dtype}/{self.dq_v_buffer.dtype}."
            )

    def _slot_move_pointer_buffers(self):

        buffers = [*self.k_buffer, *self.v_buffer]
        if getattr(self, "k_scale_buffer", None) is not None:
            buffers.extend([*self.k_scale_buffer, *self.v_scale_buffer])
        return buffers

    def _init_data_ptrs_and_strides(self):
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
        slot_move_pointer_buffers = self._slot_move_pointer_buffers()
        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in slot_move_pointer_buffers],
            dtype=torch.uint64,
            device=self.device,
        )
        self.data_strides = torch.tensor(
            [
                np.prod(x.shape[1:]) * x.dtype.itemsize
                for x in slot_move_pointer_buffers
            ],
            device=self.device,
        )

    def _kv_buffer_shapes(self):

        if self.use_hnd:
            return (
                (self.num_pages, self.head_num, self.page_size, self.head_dim),
                (self.num_pages, self.head_num, self.page_size, self.v_head_dim),
            )
        rows = self.size + self.page_size
        return (
            (rows, self.head_num, self.head_dim),
            (rows, self.head_num, self.v_head_dim),
        )

    def _create_buffers_normal(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):

                if self.kv_cache_layout == "vectorized_5d":
                    total_slots = self.size + self.page_size
                    num_blocks = total_slots // self.page_size
                    x = self._kv_vector_x

                    self.k_buffer = [
                        torch.zeros(
                            (
                                num_blocks,
                                self.head_num,
                                self.head_dim // x,
                                self.page_size,
                                x,
                            ),
                            dtype=self.store_dtype,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]

                    self.v_buffer = [
                        torch.zeros(
                            (
                                num_blocks,
                                self.head_num,
                                self.page_size // x,
                                self.v_head_dim,
                                x,
                            ),
                            dtype=self.store_dtype,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                else:
                    k_shape, v_shape = self._kv_buffer_shapes()
                    self.k_buffer = [
                        torch.zeros(k_shape, dtype=self.store_dtype, device=self.device)
                        for _ in range(self.layer_num)
                    ]
                    self.v_buffer = [
                        torch.zeros(v_shape, dtype=self.store_dtype, device=self.device)
                        for _ in range(self.layer_num)
                    ]


    def _build_kv_buffer_descs(self):

        itemsize = self.store_dtype.itemsize


        if self.k_buffer and self.v_buffer:
            k_shape = tuple(self.k_buffer[0].shape)
            v_shape = tuple(self.v_buffer[0].shape)
        else:
            k_shape, v_shape = self._kv_buffer_shapes()


        num_slots = self.size + self.page_size
        tokens_per_row = (
            self.page_size if k_shape[0] * self.page_size == num_slots else 1
        )
        descs = []
        for prefix, shape in (("k", k_shape), ("v", v_shape)):
            row_bytes = int(np.prod(shape[1:])) * itemsize
            for layer in range(self.layer_num):
                descs.append(
                    KvBufferDesc(
                        f"{prefix}{layer}",
                        shape,
                        row_bytes=row_bytes,
                        tokens_per_row=tokens_per_row,
                    )
                )
        return descs

    def _assign_post_capture_tensors(self, tensors):

        self.k_buffer = tensors[: self.layer_num]
        self.v_buffer = tensors[self.layer_num :]

    def _alloc_post_capture_buffers(self):
        dev = torch.device(self.device)
        device_id = dev.index if dev.index is not None else torch.cuda.current_device()
        self._post_capture_owner = KvVmmBufferOwner(
            device=self.device,
            device_id=device_id,
            store_dtype=self.store_dtype,
            page_size=self.page_size,
            reserved_num_tokens=self.size,
            buffer_descs=self._build_kv_buffer_descs(),
        )
        self._assign_post_capture_tensors(self._post_capture_owner.tensors)

    def finalize_backing(self, config) -> None:

        self._finalize_backing_tokens(config.max_total_num_tokens)

    def _finalize_backing_tokens(self, final_num_tokens: int) -> None:

        self._post_capture_owner.finalize(final_num_tokens)
        self.size = int(final_num_tokens)

    @property
    def post_capture_backed_bytes(self) -> int:
        return self._post_capture_owner.backed_bytes if self._post_capture_owner else 0

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer
        if hasattr(self, "k_scale_buffer") and self.k_scale_buffer is not None:
            del self.k_scale_buffer
        if hasattr(self, "v_scale_buffer") and self.v_scale_buffer is not None:
            del self.v_scale_buffer
        if hasattr(self, "dq_k_buffer") and self.dq_k_buffer is not None:
            del self.dq_k_buffer
        if hasattr(self, "dq_v_buffer") and self.dq_v_buffer is not None:
            del self.dq_v_buffer
        if self._post_capture_owner is not None:
            self._post_capture_owner.close()
            self._post_capture_owner = None

    def get_kv_size_bytes(self):
        assert hasattr(self, "k_buffer")
        assert hasattr(self, "v_buffer")
        k_size_bytes = get_tensor_size_bytes(self.k_buffer)
        v_size_bytes = get_tensor_size_bytes(self.v_buffer)
        if getattr(self, "k_scale_buffer", None) is not None:
            k_size_bytes += get_tensor_size_bytes(self.k_scale_buffer)
            v_size_bytes += get_tensor_size_bytes(self.v_scale_buffer)
        if getattr(self, "dq_k_buffer", None) is not None:
            k_size_bytes += get_tensor_size_bytes(self.dq_k_buffer)
            v_size_bytes += get_tensor_size_bytes(self.dq_v_buffer)
        return k_size_bytes, v_size_bytes


    def _pd_registerable_tensors(self):

        return self.k_buffer + self.v_buffer

    def get_contiguous_buf_infos(self):

        assert not self.use_hnd, (
            "PD-disaggregation KV transfer assumes NHD slot-row layout; "
            "HND KV cache (SGLANG_USE_HND_KVCACHE) is not supported with disagg yet."
        )
        tensors = self._pd_registerable_tensors()
        ptrs = [t.data_ptr() for t in tensors]
        lens = [
            d.final_span_bytes(self.size, self.page_size) for d in self._kv_buffer_descs
        ]
        item_lens = [d.item_len_bytes(self.page_size) for d in self._kv_buffer_descs]
        return ptrs, lens, item_lens

    def get_cpu_copy(self, indices, mamba_indices=None):
        assert not self.use_hnd, (
            "CPU KV offload indexes by slot (NHD); HND KV cache "
            "(SGLANG_USE_HND_KVCACHE) is not supported with CPU offload yet."
        )
        current_platform.synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu = self.k_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                v_cpu = self.v_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                kv_cache_cpu[-1].append([k_cpu, v_cpu])
        current_platform.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        assert not self.use_hnd, (
            "CPU KV offload indexes by slot (NHD); HND KV cache "
            "(SGLANG_USE_HND_KVCACHE) is not supported with CPU offload yet."
        )
        current_platform.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu, v_cpu = (
                    kv_cache_cpu[layer_id][i // chunk_size][0],
                    kv_cache_cpu[layer_id][i // chunk_size][1],
                )
                assert k_cpu.shape[0] == v_cpu.shape[0] == len(chunk_indices)
                k_chunk = k_cpu.to(self.k_buffer[0].device, non_blocking=True)
                v_chunk = v_cpu.to(self.v_buffer[0].device, non_blocking=True)
                self.k_buffer[layer_id][chunk_indices] = k_chunk
                self.v_buffer[layer_id][chunk_indices] = v_chunk
        current_platform.synchronize()

    def _get_key_buffer(self, layer_id: int):

        local_layer_id = layer_id - self.start_layer
        if (
            self.is_quantized_kv_cache
            and self.quant_method.needs_plain_kv_dequant_read()
        ):
            return self.quant_method.dequantize_kv_tensor(
                self.k_buffer[local_layer_id],
                self.k_scale_buffer[local_layer_id],
                layer_id,
            )
        if self.store_dtype != self.dtype:
            return self.k_buffer[local_layer_id].view(self.dtype)
        return self.k_buffer[local_layer_id]

    def get_key_buffer(self, layer_id: int):


        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_key_buffer(layer_id)

    def _get_value_buffer(self, layer_id: int):

        local_layer_id = layer_id - self.start_layer
        if (
            self.is_quantized_kv_cache
            and self.quant_method.needs_plain_kv_dequant_read()
        ):
            return self.quant_method.dequantize_kv_tensor(
                self.v_buffer[local_layer_id],
                self.v_scale_buffer[local_layer_id],
                layer_id,
            )
        if self.store_dtype != self.dtype:
            return self.v_buffer[local_layer_id].view(self.dtype)
        return self.v_buffer[local_layer_id]

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_value_buffer(layer_id)

    def get_v_head_dim(self):


        return self.v_head_dim

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
        dcp_kv_mask: Optional[torch.Tensor] = None,
    ):
        loc, _, _ = unwrap_write_loc(loc_info)


        maybe_detect_oob(loc, 0, self.size + self.page_size, "set_kv_buffer (MHA)")
        layer_id = (
            layer_id_override if layer_id_override is not None else layer.layer_id
        )
        global_layer_id = layer.layer_id if layer is not None else layer_id

        if self.is_quantized_kv_cache:
            if dcp_kv_mask is not None:
                raise RuntimeError("dcp_kv_mask is not supported for FP4 KV cache.")
            self._set_quantized_kv_buffer(
                layer_id,
                global_layer_id,
                loc,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
            )
            return

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

        if dcp_kv_mask is not None:
            N, H, D = cache_k.shape
            masked_set_kv_buffer_kernel[(N,)](
                cache_k,
                cache_v,
                self.k_buffer[layer_id - self.start_layer],
                self.v_buffer[layer_id - self.start_layer],
                loc,
                dcp_kv_mask,
                N,
                H,
                D,
                128,
                cache_k.stride(0),
                cache_k.stride(1),
                cache_v.stride(0),
                cache_v.stride(1),
            )
            return

        if self.use_hnd:

            k_buf = self.k_buffer[layer_id - self.start_layer]
            v_buf = self.v_buffer[layer_id - self.start_layer]
            pages = loc // self.page_size
            offs = loc % self.page_size
            k_buf[pages, :, offs, :] = cache_k
            v_buf[pages, :, offs, :] = cache_v
            return

        self._store_kv_layer(layer_id - self.start_layer, loc, cache_k, cache_v)

    def _store_kv_layer(
        self,
        layer_idx: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):


        if self.kv_cache_layout == "vectorized_5d":

            from sglang.kernels.ops.attention.utils import (
                launch_reshape_and_cache_shuffle_5d,
            )


            launch_reshape_and_cache_shuffle_5d(
                cache_k,
                cache_v,
                self.k_buffer[layer_idx],
                self.v_buffer[layer_idx],
                loc,
            )
            return

        _set_kv_buffer_impl(
            cache_k,
            cache_v,
            self.k_buffer[layer_idx],
            self.v_buffer[layer_idx],
            loc,
            row_dim=self.row_dim,
            store_dtype=self.store_dtype,
            device_module=self.device_module,


            size_limit=self.size + self.page_size,
            alt_stream=self.alt_stream,
            v_row_dim=self.v_row_dim,
        )

    def _quantized_scales(self, global_layer_id: int, k_scale, v_scale):
        if k_scale is None and hasattr(self.quant_method, "k_scales_gpu"):
            k_scale = self.quant_method.k_scales_gpu[
                global_layer_id : global_layer_id + 1
            ]
            v_scale = self.quant_method.v_scales_gpu[
                global_layer_id : global_layer_id + 1
            ]
        return k_scale, v_scale

    def _set_quantized_kv_buffer(
        self,
        layer_id: int,
        global_layer_id: int,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale=None,
        v_scale=None,
    ) -> None:
        loc, _, _ = unwrap_write_loc(loc_info)
        local_layer_id = layer_id - self.start_layer
        k_scale, v_scale = self._quantized_scales(global_layer_id, k_scale, v_scale)
        self.quant_method.quantize_and_store(
            self.k_buffer[local_layer_id],
            self.v_buffer[local_layer_id],
            (
                self.k_scale_buffer[local_layer_id]
                if self.k_scale_buffer is not None
                else None
            ),
            (
                self.v_scale_buffer[local_layer_id]
                if self.v_scale_buffer is not None
                else None
            ),
            loc,
            cache_k,
            cache_v,
            k_scale,
            v_scale,
        )

    def get_raw_kv_buffer(
        self, layer_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        local_layer_id = layer_id - self.start_layer
        if self.k_scale_buffer is None or self.v_scale_buffer is None:
            raise RuntimeError("Raw FP4 KV cache requested from a non-FP4 KV pool.")
        k_scale = self.k_scale_buffer[local_layer_id]
        v_scale = self.v_scale_buffer[local_layer_id]
        scale_view_dtype = self.quant_method.scale_buffer_view_dtype()
        if scale_view_dtype is not None:
            k_scale = k_scale.view(scale_view_dtype)
            v_scale = v_scale.view(scale_view_dtype)
        return (
            self.k_buffer[local_layer_id],
            self.v_buffer[local_layer_id],
            k_scale,
            v_scale,
        )

    def get_dequant_workspace(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.dq_k_buffer is None or self.dq_v_buffer is None:
            raise RuntimeError(
                "Dequant workspace requested from a KV pool without FP4 dequant buffers."
            )
        return self.dq_k_buffer, self.dq_v_buffer

    def get_flashinfer_dequant_workspace_kv_buffer(
        self,
        layer: RadixAttention,
        req_to_token: torch.Tensor,
        req_pool_indices_cpu,
        extend_prefix_lens_cpu,
        extend_seq_lens_cpu,
        page_size: int,
        *,
        prepare_workspace: bool,
        use_ragged: bool,
        k_cur: Optional[torch.Tensor] = None,
        v_cur: Optional[torch.Tensor] = None,
        layer_id_override: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if not self.is_quantized_kv_cache:
            raise RuntimeError(
                "FlashInfer quantized KV buffer requested from a non-quantized KV pool."
            )

        if prepare_workspace:
            transfer_cur_kv = not use_ragged
            k_cur_fp8 = (
                k_cur.to(torch.float8_e4m3fn)
                if k_cur is not None and transfer_cur_kv
                else None
            )
            v_cur_fp8 = (
                v_cur.to(torch.float8_e4m3fn)
                if v_cur is not None and transfer_cur_kv
                else None
            )
            self._prepare_dequant_extend_workspace(
                layer.layer_id if layer_id_override is None else layer_id_override,
                layer.layer_id,
                req_to_token,
                req_pool_indices_cpu,
                extend_prefix_lens_cpu,
                extend_seq_lens_cpu,
                page_size,
                k_cur_fp8=k_cur_fp8,
                v_cur_fp8=v_cur_fp8,
            )

        k_buffer_dq, v_buffer_dq = self.get_dequant_workspace()
        return (
            k_buffer_dq.view(-1, layer.tp_k_head_num, layer.head_dim),
            v_buffer_dq.view(-1, layer.tp_v_head_num, layer.head_dim),
        )

    def get_flashinfer_decode_dequant_workspace_kv_buffer(
        self,
        layer: RadixAttention,
        req_to_token: torch.Tensor,
        req_pool_indices,
        seq_lens,
        *,
        layer_id_override: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_quantized_kv_cache:
            raise RuntimeError(
                "FlashInfer dequant workspace requested from a non-quantized KV pool."
            )

        self._prepare_dequant_decode_workspace(
            layer.layer_id if layer_id_override is None else layer_id_override,
            layer.layer_id,
            req_to_token,
            req_pool_indices,
            seq_lens,
        )
        k_buffer_dq, v_buffer_dq = self.get_dequant_workspace()
        return (
            k_buffer_dq.view(-1, layer.tp_k_head_num, layer.head_dim),
            v_buffer_dq.view(-1, layer.tp_v_head_num, layer.head_dim),
        )

    @staticmethod
    def _to_cpu_int_list(values) -> list[int]:
        if isinstance(values, list):
            return [int(value) for value in values]
        if isinstance(values, torch.Tensor):
            return [int(value) for value in values.cpu().tolist()]
        return [int(value) for value in values]

    def _prepare_dequant_extend_workspace(
        self,
        layer_id: int,
        global_layer_id: int,
        req_to_token: torch.Tensor,
        req_pool_indices_cpu,
        extend_prefix_lens_cpu,
        extend_seq_lens_cpu,
        page_size: int,
        k_cur_fp8: Optional[torch.Tensor] = None,
        v_cur_fp8: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        k_fp4, v_fp4, k_scales, v_scales = self.get_raw_kv_buffer(layer_id)
        dq_k, dq_v = self.get_dequant_workspace()

        cur_batch_start_loc_cpu = 0
        cur_token_idx_dq = page_size

        for i in range(len(req_pool_indices_cpu)):
            req_idx = int(req_pool_indices_cpu[i])
            prev_len = int(extend_prefix_lens_cpu[i])
            extend_len = int(extend_seq_lens_cpu[i])

            if prev_len > 0:
                prev_indices = req_to_token[req_idx, :prev_len]
                k_prev_fp8, v_prev_fp8 = self.quant_method.dequantize_prev_kv(
                    k_fp4[prev_indices],
                    k_scales[prev_indices],
                    v_fp4[prev_indices],
                    v_scales[prev_indices],
                    global_layer_id,
                )
                dq_k[cur_token_idx_dq : cur_token_idx_dq + prev_len] = k_prev_fp8
                dq_v[cur_token_idx_dq : cur_token_idx_dq + prev_len] = v_prev_fp8

            if k_cur_fp8 is not None:
                cur_end = cur_batch_start_loc_cpu + extend_len
                dst_start = cur_token_idx_dq + prev_len
                dst_end = dst_start + extend_len
                dq_k[dst_start:dst_end] = k_cur_fp8[cur_batch_start_loc_cpu:cur_end]
                dq_v[dst_start:dst_end] = v_cur_fp8[cur_batch_start_loc_cpu:cur_end]
                cur_batch_start_loc_cpu = cur_end

            workspace_len = prev_len + (extend_len if k_cur_fp8 is not None else 0)
            cur_token_idx_dq = (
                (cur_token_idx_dq + workspace_len + page_size - 1)
                // page_size
                * page_size
            )

        return dq_k, dq_v

    def _prepare_dequant_decode_workspace(
        self,
        layer_id: int,
        global_layer_id: int,
        req_to_token: torch.Tensor,
        req_pool_indices,
        seq_lens,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k_fp4, v_fp4, k_scales, v_scales = self.get_raw_kv_buffer(layer_id)
        dq_k, dq_v = self.get_dequant_workspace()

        req_pool_indices_cpu = self._to_cpu_int_list(req_pool_indices)
        seq_lens_cpu = self._to_cpu_int_list(seq_lens)

        for req_idx, seq_len in zip(req_pool_indices_cpu, seq_lens_cpu):
            if seq_len <= 0:
                continue
            kv_indices = req_to_token[req_idx, :seq_len]
            k_prev_fp8, v_prev_fp8 = self.quant_method.dequantize_prev_kv(
                k_fp4[kv_indices],
                k_scales[kv_indices],
                v_fp4[kv_indices],
                v_scales[kv_indices],
                global_layer_id,
            )
            dq_k[kv_indices] = k_prev_fp8
            dq_v[kv_indices] = v_prev_fp8

        return dq_k, dq_v

    def set_kv_buffer_prefix_valid(
        self,
        layer: RadixAttention,
        loc_2d: torch.Tensor,
        commit_lens: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id

        if loc_2d.ndim != 2:
            raise ValueError(f"loc_2d must be rank-2, got shape={tuple(loc_2d.shape)}.")
        if commit_lens.ndim != 1 or commit_lens.shape[0] != loc_2d.shape[0]:
            raise ValueError(
                "commit_lens must match loc_2d batch size: "
                f"{tuple(commit_lens.shape)=} {tuple(loc_2d.shape)=}."
            )

        num_rows = int(loc_2d.numel())
        if cache_k.shape[0] != num_rows or cache_v.shape[0] != num_rows:
            raise ValueError(
                "dense KV rows must match loc_2d size: "
                f"{tuple(cache_k.shape)=} {tuple(cache_v.shape)=} {tuple(loc_2d.shape)=}."
            )

        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.contiguous().view(self.store_dtype)
            cache_v = cache_v.contiguous().view(self.store_dtype)
        else:
            cache_k = cache_k.contiguous()
            cache_v = cache_v.contiguous()

        if loc_2d.device != self.k_buffer[0].device:
            loc_2d = loc_2d.to(device=self.k_buffer[0].device, non_blocking=True)
        if commit_lens.device != self.k_buffer[0].device:
            commit_lens = commit_lens.to(
                device=self.k_buffer[0].device, non_blocking=True
            )
        if loc_2d.dtype != torch.int64:
            loc_2d = loc_2d.to(torch.int64)
        if commit_lens.dtype != torch.int32:
            commit_lens = commit_lens.to(torch.int32)

        if not (_is_cuda or _is_hip):
            row_offsets = torch.arange(loc_2d.shape[1], device=loc_2d.device)
            valid_mask = row_offsets[None, :] < commit_lens.to(torch.int64)[:, None]
            valid_idx = torch.nonzero(valid_mask.reshape(-1), as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                return
            self.set_kv_buffer(
                layer,
                loc_2d.reshape(-1).index_select(0, valid_idx),
                cache_k.index_select(0, valid_idx),
                cache_v.index_select(0, valid_idx),
                k_scale,
                v_scale,
                layer_id_override=layer_id,
            )
            return


        if self.v_row_dim != self.row_dim:
            raise NotImplementedError(
                "prefix-valid commit requires equal-width K/V rows, got "
                f"head_dim={self.head_dim} v_head_dim={self.v_head_dim}."
            )

        _set_kv_buffer_prefix_valid_impl(
            cache_k,
            cache_v,
            self.k_buffer[layer_id - self.start_layer],
            self.v_buffer[layer_id - self.start_layer],
            loc_2d,
            commit_lens,
            row_dim=self.row_dim,
            store_dtype=self.store_dtype,
        )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):

        if self.layer_num == 0:
            return


        size_limit = self.size + self.page_size
        maybe_detect_oob(tgt_loc, 0, size_limit, "move_kv_cache tgt_loc")
        maybe_detect_oob(src_loc, 0, size_limit, "move_kv_cache src_loc")

        if self.use_hnd:
            pages_t, offs_t = tgt_loc // self.page_size, tgt_loc % self.page_size
            pages_s, offs_s = src_loc // self.page_size, src_loc % self.page_size
            for kb, vb in zip(self.k_buffer, self.v_buffer):
                kb[pages_t, :, offs_t, :] = kb[pages_s, :, offs_s, :]
                vb[pages_t, :, offs_t, :] = vb[pages_s, :, offs_s, :]
            return

        self._move_kv_cache_impl(tgt_loc, src_loc)

    def _move_kv_cache_impl(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):


        if self.use_native_move_kv_cache:
            move_kv_cache_native(self.k_buffer, self.v_buffer, tgt_loc, src_loc)
            if getattr(self, "k_scale_buffer", None) is not None:
                move_kv_cache_native(
                    self.k_scale_buffer, self.v_scale_buffer, tgt_loc, src_loc
                )
            return

        N = tgt_loc.numel()
        if N == 0:
            return

        assert (
            self._kv_copy_config is not None
        ), "KV copy not initialized. Set enable_kv_cache_copy=True in __init__"

        cfg = self._kv_copy_config
        cap = int(cfg.get("num_locs_upper", 256))

        if N <= cap:
            copy_all_layer_kv_cache_func(
                self.data_ptrs,
                self.data_strides,
                tgt_loc,
                src_loc,
                N,
                next_power_of_2(N),
                cfg,
            )
            return


        for start in range(0, N, cap):
            end = min(start + cap, N)
            chunk_len = end - start
            copy_all_layer_kv_cache_func(
                self.data_ptrs,
                self.data_strides,
                tgt_loc[start:end],
                src_loc[start:end],
                chunk_len,
                next_power_of_2(chunk_len),
                cfg,
            )


class NoOpMHATokenToKVPool(MHATokenToKVPool):


    def _create_buffers(self):


        self.use_hnd = False
        self.kv_cache_layout = "nhd"


        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.k_buffer = [
                torch.zeros(
                    (self.page_size, self.head_num, self.head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]
            self.v_buffer = [
                torch.zeros(
                    (self.page_size, self.head_num, self.v_head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]

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
            [
                np.prod(x.shape[1:]) * x.dtype.itemsize
                for x in self.k_buffer + self.v_buffer
            ],
            device=self.device,
        )

    def _finalize_allocation_log(self, num_tokens: int):
        self.mem_usage = 0.0
        placeholder_bytes = (
            2
            * self.layer_num
            * self.page_size
            * self.head_num
            * max(self.head_dim, self.v_head_dim)
            * self.store_dtype.itemsize
        )
        logger.info(
            f"KV Cache skipped (no-op pool). Logical #tokens: {num_tokens}, "
            f"physical K/V size: ~{placeholder_bytes / 1024:.1f} KB placeholder"
        )

    def get_kv_size_bytes(self):

        return (0, 0)

    def set_kv_buffer(self, *args, **kwargs):
        raise RuntimeError(
            "NoOpMHATokenToKVPool.set_kv_buffer was called. This pool is only "
            "valid in prefill-only modes (e.g. --is-embedding, scoring) with "
            "the FA backend's fa_skip_kv_cache path active; the attention "
            "backend must never write to it. Check that the workload truly "
            "performs no decode and that the FA backend's fa_skip_kv_cache "
            "preconditions are met."
        )

    def get_key_buffer(self, layer_id: int):


        return self.k_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        return self.v_buffer[layer_id - self.start_layer]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):

        return


class MHATokenToKVPoolFP4(MHATokenToKVPool):
    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):


                m = self.size + self.page_size
                n = self.head_num
                k = self.head_dim

                scale_block_size = 16
                self.store_dtype = torch.uint8
                self.k_buffer = [
                    torch.zeros(
                        (m, n, k // 2),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_buffer = [
                    torch.zeros(
                        (m, n, k // 2),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

                self.k_scale_buffer = [
                    torch.zeros(
                        (m, (n * k) // scale_block_size),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_scale_buffer = [
                    torch.zeros(
                        (m, (n * k) // scale_block_size),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer
        del self.k_scale_buffer
        del self.v_scale_buffer

    def _get_key_buffer(self, layer_id: int):

        if self.store_dtype != self.dtype:
            cache_k_nope_fp4 = self.k_buffer[layer_id - self.start_layer].view(
                torch.uint8
            )
            cache_k_nope_fp4_sf = self.k_scale_buffer[layer_id - self.start_layer]

            from sglang.srt.layers.quantization.kvfp4_tensor import (
                FP4MXBlock16KVQuantizeUtil,
            )

            cache_k_nope_fp4_dequant = FP4MXBlock16KVQuantizeUtil.batched_dequantize(
                cache_k_nope_fp4, cache_k_nope_fp4_sf
            )
            return cache_k_nope_fp4_dequant
        return self.k_buffer[layer_id - self.start_layer]

    def _get_value_buffer(self, layer_id: int):

        if self.store_dtype != self.dtype:
            cache_v_nope_fp4 = self.v_buffer[layer_id - self.start_layer].view(
                torch.uint8
            )
            cache_v_nope_fp4_sf = self.v_scale_buffer[layer_id - self.start_layer]

            from sglang.srt.layers.quantization.kvfp4_tensor import (
                FP4MXBlock16KVQuantizeUtil,
            )

            cache_v_nope_fp4_dequant = FP4MXBlock16KVQuantizeUtil.batched_dequantize(
                cache_v_nope_fp4, cache_v_nope_fp4_sf
            )
            return cache_v_nope_fp4_dequant
        return self.v_buffer[layer_id - self.start_layer]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        loc, _, _ = unwrap_write_loc(loc_info)
        maybe_detect_oob(loc, 0, self.size + self.page_size, "set_kv_buffer (MHA-FP4)")
        from sglang.srt.model_executor.runner import get_is_capture_mode

        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)

            from sglang.srt.layers.quantization.kvfp4_tensor import (
                FP4MXBlock16KVQuantizeUtil,
            )

            cache_k, cache_k_fp4_sf = FP4MXBlock16KVQuantizeUtil.batched_quantize(
                cache_k
            )
            cache_v, cache_v_fp4_sf = FP4MXBlock16KVQuantizeUtil.batched_quantize(
                cache_v
            )

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

            cache_k_fp4_sf = cache_k_fp4_sf.view(self.store_dtype)
            cache_v_fp4_sf = cache_v_fp4_sf.view(self.store_dtype)

        if get_is_capture_mode() and self.alt_stream is not None:

            current_stream = self.device_module.current_stream()
            self.alt_stream.wait_stream(current_stream)
            self.k_buffer[layer_id - self.start_layer][loc] = cache_k

            self.k_scale_buffer[layer_id - self.start_layer][loc] = cache_k_fp4_sf
            with self.device_module.stream(self.alt_stream):
                self.v_buffer[layer_id - self.start_layer][loc] = cache_v

                self.v_scale_buffer[layer_id - self.start_layer][loc] = cache_v_fp4_sf
            current_stream.wait_stream(self.alt_stream)
        else:
            self.k_buffer[layer_id - self.start_layer][loc] = cache_k
            self.v_buffer[layer_id - self.start_layer][loc] = cache_v

            self.k_scale_buffer[layer_id - self.start_layer][loc] = cache_k_fp4_sf
            self.v_scale_buffer[layer_id - self.start_layer][loc] = cache_v_fp4_sf


class PageMajorMHATokenToKVPool(MHATokenToKVPool):


    def __init__(
        self,
        *args,
        kv_cache_layout: Optional[str] = None,
        enable_kv_cache_copy: bool = False,
        **kwargs,
    ):
        assert kv_cache_layout in (
            None,
            "page_major_layer_major",
        ), f"PageMajorMHATokenToKVPool fixes its layout; got {kv_cache_layout!r}"


        super().__init__(
            *args,
            kv_cache_layout="page_major_layer_major",
            enable_kv_cache_copy=False,
            **kwargs,
        )

    def _create_buffers(self):


        total_slots = self.size + self.page_size
        assert total_slots % self.page_size == 0, (
            f"page_major_layer_major needs (size + page_size) divisible by "
            f"page_size; got size={self.size}, page_size={self.page_size}"
        )
        num_pages = total_slots // self.page_size
        entry_bytes = mha_entry_bytes(
            layer_num=self.layer_num,
            head_num=self.head_num,
            head_dim=self.head_dim,
            v_head_dim=self.v_head_dim,
            itemsize=self.store_dtype.itemsize,
        )
        total_bytes = num_pages * self.page_size * entry_bytes
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):

                self._raw = torch.zeros(
                    total_bytes, dtype=torch.uint8, device=self.device
                )
        self.k_buffer, self.v_buffer = build_page_major_mha_views(
            self._raw,
            layer_num=self.layer_num,
            head_num=self.head_num,
            head_dim=self.head_dim,
            v_head_dim=self.v_head_dim,
            store_dtype=self.store_dtype,
            page_size=self.page_size,
            num_pages=num_pages,
        )


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

    def _store_kv_layer(
        self,
        layer_idx: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):


        store_cache_4d(
            self.k_buffer[layer_idx],
            self.v_buffer[layer_idx],
            cache_k,
            cache_v,
            loc,
            page_size=self.page_size,
        )

    def _move_kv_cache_impl(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):


        move_kv_cache_native(
            self.k_buffer,
            self.v_buffer,
            tgt_loc,
            src_loc,
            page_size=self.page_size,
        )


    def get_contiguous_buf_infos(self):
        raise NotImplementedError(
            "page-major layout has no per-layer contiguous regions; KV transfer / "
            "disaggregation is unsupported (TODO: expose the single _raw buffer "
            "with a page-aware transfer scheme)."
        )

    def get_cpu_copy(self, indices, mamba_indices=None):
        raise NotImplementedError(
            "CPU offloading is unsupported under the page-major layout "
            "(TODO: split token ids into page/slot for the 4-D index)."
        )

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        raise NotImplementedError(
            "CPU offloading is unsupported under the page-major layout "
            "(TODO: split token ids into page/slot for the 4-D index)."
        )

    def set_kv_buffer_prefix_valid(self, *args, **kwargs):
        raise NotImplementedError(
            "prefix-valid commit is unsupported under the page-major layout "
            "(_set_kv_buffer_prefix_valid_impl assumes 3-D contiguous + row_dim)."
        )


class MHATokenToKVPoolMXFP8(MHATokenToKVPool):


    MXFP8_SCALE_BLOCK_SIZE = 32

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                m = self.size + self.page_size
                n = self.head_num
                k = self.head_dim
                v = self.v_head_dim

                if k % self.MXFP8_SCALE_BLOCK_SIZE != 0:
                    raise ValueError(
                        f"MXFP8 KV cache requires head_dim divisible by "
                        f"{self.MXFP8_SCALE_BLOCK_SIZE}, got {k}."
                    )
                if v % self.MXFP8_SCALE_BLOCK_SIZE != 0:
                    raise ValueError(
                        f"MXFP8 KV cache requires v_head_dim divisible by "
                        f"{self.MXFP8_SCALE_BLOCK_SIZE}, got {v}."
                    )
                if not hasattr(torch, "float8_e8m0fnu"):
                    raise RuntimeError(
                        "MXFP8 KV cache requires torch.float8_e8m0fnu support."
                    )
                if self.use_hnd:


                    raise ValueError(
                        "MXFP8 KV cache does not support SGLANG_USE_HND_KVCACHE."
                    )

                self.store_dtype = torch.float8_e4m3fn
                self.k_buffer = [
                    torch.zeros((m, n, k), dtype=self.store_dtype, device=self.device)
                    for _ in range(self.layer_num)
                ]
                self.v_buffer = [
                    torch.zeros((m, n, v), dtype=self.store_dtype, device=self.device)
                    for _ in range(self.layer_num)
                ]


                k_sf_dim = k // self.MXFP8_SCALE_BLOCK_SIZE
                v_sf_dim = v // self.MXFP8_SCALE_BLOCK_SIZE
                self.mxfp8_sf_interleaved = self.page_size == 128
                if self.mxfp8_sf_interleaved:
                    assert m % self.page_size == 0
                    num_pages = m // self.page_size
                    chunk = self.page_size // self.MXFP8_SCALE_BLOCK_SIZE
                    k_sf_shape = (
                        num_pages,
                        n,
                        self.MXFP8_SCALE_BLOCK_SIZE,
                        chunk,
                        k_sf_dim,
                    )
                    v_sf_shape = (
                        num_pages,
                        n,
                        self.MXFP8_SCALE_BLOCK_SIZE,
                        chunk,
                        v_sf_dim,
                    )
                else:
                    k_sf_shape = (m, n, k_sf_dim)
                    v_sf_shape = (m, n, v_sf_dim)
                self.k_scale_buffer = [
                    torch.zeros(
                        k_sf_shape, dtype=torch.float8_e8m0fnu, device=self.device
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_scale_buffer = [
                    torch.zeros(
                        v_sf_shape, dtype=torch.float8_e8m0fnu, device=self.device
                    )
                    for _ in range(self.layer_num)
                ]

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
            [
                np.prod(x.shape[1:]) * x.dtype.itemsize
                for x in self.k_buffer + self.v_buffer
            ],
            device=self.device,
        )


        self._kv_buffer_descs = self._build_kv_buffer_descs()

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer
        del self.k_scale_buffer
        del self.v_scale_buffer

    def _get_key_buffer(self, layer_id: int):
        return self.k_buffer[layer_id - self.start_layer]

    def _get_value_buffer(self, layer_id: int):
        return self.v_buffer[layer_id - self.start_layer]

    def get_kv_scale_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = layer_id - self.start_layer
        return self.k_scale_buffer[idx], self.v_scale_buffer[idx]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
        layer_id_override: Optional[int] = None,
        dcp_kv_mask: Optional[torch.Tensor] = None,
    ):
        if dcp_kv_mask is not None:
            raise NotImplementedError("MXFP8 KV cache does not support DCP KV masks.")
        loc, _, _ = unwrap_write_loc(loc_info)
        maybe_detect_oob(
            loc, 0, self.size + self.page_size, "set_kv_buffer (MHA-MXFP8)"
        )
        layer_id = (
            layer_id_override if layer_id_override is not None else layer.layer_id
        )
        idx = layer_id - self.start_layer

        if k_scale is None or v_scale is None:


            if not self.mxfp8_sf_interleaved or cache_k.dtype == self.store_dtype:
                raise ValueError("MXFP8 KV cache requires K and V scale tensors.")
            from sglang.kernels.ops.quantization.mxfp8_quant import quant_store_kv_mxfp8

            quant_store_kv_mxfp8(
                cache_k,
                cache_v,
                loc,
                self.k_buffer[idx],
                self.v_buffer[idx],
                self.k_scale_buffer[idx],
                self.v_scale_buffer[idx],
                page_size=self.page_size,
            )
            return

        from sglang.srt.model_executor.runner import get_is_capture_mode

        if get_is_capture_mode() and self.alt_stream is not None:
            current_stream = self.device_module.current_stream()
            self.alt_stream.wait_stream(current_stream)
            self.k_buffer[idx][loc] = cache_k
            self._write_scales(idx, loc, k_scale, v_scale)
            with self.device_module.stream(self.alt_stream):
                self.v_buffer[idx][loc] = cache_v
            current_stream.wait_stream(self.alt_stream)
        else:
            self.k_buffer[idx][loc] = cache_k
            self.v_buffer[idx][loc] = cache_v
            self._write_scales(idx, loc, k_scale, v_scale)

    def _write_scales(self, idx, loc, k_scale, v_scale):

        if self.mxfp8_sf_interleaved:
            from sglang.kernels.ops.quantization.mxfp8_interleave_sf import (
                store_sf_interleaved,
            )

            store_sf_interleaved(
                k_scale, self.k_scale_buffer[idx], loc, page_size=self.page_size
            )
            store_sf_interleaved(
                v_scale, self.v_scale_buffer[idx], loc, page_size=self.page_size
            )
        else:
            self.k_scale_buffer[idx][loc] = k_scale
            self.v_scale_buffer[idx][loc] = v_scale

    def _read_sf_interleaved(self, sf_buf: torch.Tensor, loc: torch.Tensor):

        num_pages, n = sf_buf.shape[0], sf_buf.shape[1]
        sf_dim = sf_buf.shape[-1]

        buf_u32 = sf_buf.reshape(num_pages, n, -1).view(torch.int32)
        off = loc % self.page_size
        page = (loc // self.page_size).long()
        chunk = self.page_size // self.MXFP8_SCALE_BLOCK_SIZE
        ipos = (
            (off % self.MXFP8_SCALE_BLOCK_SIZE) * chunk
            + (off // self.MXFP8_SCALE_BLOCK_SIZE)
        ).long()
        heads = torch.arange(n, device=loc.device)
        gathered = buf_u32[page[:, None], heads[None, :], ipos[:, None]]
        return (
            gathered.reshape(loc.shape[0], n, 1)
            .view(torch.uint8)
            .reshape(loc.shape[0], n, sf_dim)
            .view(torch.float8_e8m0fnu)
        )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):


        if self.mxfp8_sf_interleaved:
            from sglang.kernels.ops.quantization.mxfp8_interleave_sf import (
                store_sf_interleaved,
            )

            for idx in range(self.layer_num):
                self.k_buffer[idx][tgt_loc] = self.k_buffer[idx][src_loc]
                self.v_buffer[idx][tgt_loc] = self.v_buffer[idx][src_loc]
                k_sf = self._read_sf_interleaved(self.k_scale_buffer[idx], src_loc)
                v_sf = self._read_sf_interleaved(self.v_scale_buffer[idx], src_loc)
                store_sf_interleaved(
                    k_sf, self.k_scale_buffer[idx], tgt_loc, page_size=self.page_size
                )
                store_sf_interleaved(
                    v_sf, self.v_scale_buffer[idx], tgt_loc, page_size=self.page_size
                )
        else:
            super().move_kv_cache(tgt_loc, src_loc)
            for idx in range(self.layer_num):
                self.k_scale_buffer[idx][tgt_loc] = self.k_scale_buffer[idx][src_loc]
                self.v_scale_buffer[idx][tgt_loc] = self.v_scale_buffer[idx][src_loc]

    def _read_scales(self, idx, loc):

        if self.mxfp8_sf_interleaved:
            return (
                self._read_sf_interleaved(self.k_scale_buffer[idx], loc),
                self._read_sf_interleaved(self.v_scale_buffer[idx], loc),
            )
        return self.k_scale_buffer[idx][loc], self.v_scale_buffer[idx][loc]

    def get_cpu_copy(self, indices, mamba_indices=None):


        assert not self.use_hnd, (
            "CPU KV offload indexes by slot (NHD); HND KV cache "
            "(SGLANG_USE_HND_KVCACHE) is not supported with CPU offload yet."
        )
        current_platform.synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_scale, v_scale = self._read_scales(layer_id, chunk_indices)
                kv_cache_cpu[-1].append(
                    [
                        self.k_buffer[layer_id][chunk_indices].to(
                            "cpu", non_blocking=True
                        ),
                        self.v_buffer[layer_id][chunk_indices].to(
                            "cpu", non_blocking=True
                        ),
                        k_scale.to("cpu", non_blocking=True),
                        v_scale.to("cpu", non_blocking=True),
                    ]
                )
        current_platform.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        assert not self.use_hnd, (
            "CPU KV offload indexes by slot (NHD); HND KV cache "
            "(SGLANG_USE_HND_KVCACHE) is not supported with CPU offload yet."
        )
        current_platform.synchronize()
        device = self.k_buffer[0].device
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu, v_cpu, k_scale_cpu, v_scale_cpu = kv_cache_cpu[layer_id][
                    i // chunk_size
                ]
                assert k_cpu.shape[0] == v_cpu.shape[0] == len(chunk_indices)
                self.k_buffer[layer_id][chunk_indices] = k_cpu.to(
                    device, non_blocking=True
                )
                self.v_buffer[layer_id][chunk_indices] = v_cpu.to(
                    device, non_blocking=True
                )
                self._write_scales(
                    layer_id,
                    chunk_indices,
                    k_scale_cpu.to(device, non_blocking=True),
                    v_scale_cpu.to(device, non_blocking=True),
                )
        current_platform.synchronize()

    def get_kv_scale_buf_infos(self):

        tensors = self.k_scale_buffer + self.v_scale_buffer
        ptrs = [t.data_ptr() for t in tensors]
        lens = [t.nbytes for t in tensors]
        row_bytes = [t[0].nbytes for t in tensors]
        if self.mxfp8_sf_interleaved:
            item_lens = row_bytes
        else:
            item_lens = [rb * self.page_size for rb in row_bytes]
        return ptrs, lens, item_lens

    def set_kv_buffer_prefix_valid(self, *args, **kwargs):
        raise NotImplementedError(
            "prefix-valid commit is unsupported for MXFP8 KV cache "
            "(it does not carry the scale buffers)."
        )

    def get_kv_size_bytes(self):
        k_size_bytes = 0
        v_size_bytes = 0
        for k_cache in self.k_buffer:
            k_size_bytes += get_tensor_size_bytes(k_cache)
        for k_scale in self.k_scale_buffer:
            k_size_bytes += get_tensor_size_bytes(k_scale)
        for v_cache in self.v_buffer:
            v_size_bytes += get_tensor_size_bytes(v_cache)
        for v_scale in self.v_scale_buffer:
            v_size_bytes += get_tensor_size_bytes(v_scale)
        return k_size_bytes, v_size_bytes


class HybridLinearKVPool(KVCache):


    cpu_copy_carries_mamba = True

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        page_size: int,
        head_num: int,
        head_dim: int,
        full_attention_layer_ids: List[int],
        device: str,
        mamba_pool: MambaPool,
        enable_memory_saver: bool = False,
        enable_kv_cache_copy: bool = False,

        use_mla: bool = False,
        kv_lora_rank: int = None,
        qk_rope_head_dim: int = None,
        start_layer: Optional[int] = None,
        full_kv_pool_class: Optional[type] = None,
        quant_method=None,


        full_kv_pool: Optional[KVCache] = None,
        post_capture_active: bool = False,
    ):
        self.size = size
        self.dtype = dtype
        self.device = device
        self.full_layer_nums = len(full_attention_layer_ids)
        self.page_size = page_size
        self.start_layer = start_layer if start_layer is not None else 0
        self.layer_transfer_counter = None
        self.head_num = head_num
        self.head_dim = head_dim
        self.mamba_pool = mamba_pool


        self._mamba_translate = lambda ids: ids


        self._full_translate = lambda ids: ids
        self.use_mla = use_mla
        if full_kv_pool is not None:


            self.full_kv_pool = full_kv_pool
        elif not use_mla:
            TokenToKVPoolClass = MHATokenToKVPool
            quant_method_kwarg = {"quant_method": quant_method}

            if current_platform.is_out_of_tree():
                TokenToKVPoolClass = current_platform.get_mha_kv_pool_cls()
                quant_method_kwarg = {}
            elif _is_npu:
                assert not is_float4_e2m1fn_x2(
                    dtype
                ), "FP4 is not supported on NPU yet."
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMHATokenToKVPool,
                )

                TokenToKVPoolClass = NPUMHATokenToKVPool
                quant_method_kwarg = {}
            elif full_kv_pool_class is not None:


                TokenToKVPoolClass = full_kv_pool_class
            else:
                TokenToKVPoolClass = MHATokenToKVPool

            post_capture_kwargs = (
                {"post_capture_active": True} if post_capture_active else {}
            )
            self.full_kv_pool = TokenToKVPoolClass(
                size=size,
                page_size=self.page_size,
                dtype=dtype,
                head_num=head_num,
                head_dim=head_dim,
                layer_num=self.full_layer_nums,
                device=device,
                enable_memory_saver=enable_memory_saver,
                enable_kv_cache_copy=enable_kv_cache_copy,
                **quant_method_kwarg,
                **post_capture_kwargs,
            )
        else:
            TokenToKVPoolClass = MLATokenToKVPool

            if current_platform.is_out_of_tree():
                TokenToKVPoolClass = current_platform.get_mla_kv_pool_cls()
            elif _is_npu:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMLATokenToKVPool,
                )

                TokenToKVPoolClass = NPUMLATokenToKVPool

            self.full_kv_pool = TokenToKVPoolClass(
                size=size,
                page_size=self.page_size,
                dtype=dtype,
                layer_num=self.full_layer_nums,
                device=device,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                enable_memory_saver=enable_memory_saver,
            )
        self.full_attention_layer_id_mapping = {
            id: i for i, id in enumerate(full_attention_layer_ids)
        }
        if use_mla:
            self.mem_usage = self.get_kv_size_bytes() / GB
        else:
            k_size, v_size = self.get_kv_size_bytes()
            self.mem_usage = (k_size + v_size) / GB

    @property
    def post_capture_active(self) -> bool:
        return self.full_kv_pool.post_capture_active

    @property
    def post_capture_backed_bytes(self) -> int:
        return self.full_kv_pool.post_capture_backed_bytes

    def finalize_backing(self, config) -> None:

        self.full_kv_pool._finalize_backing_tokens(config.max_total_num_tokens)
        self.size = int(config.max_total_num_tokens)

    def get_kv_size_bytes(self):
        return self.full_kv_pool.get_kv_size_bytes()

    def get_kv_buffer_shape(self) -> Tuple[torch.Size, torch.Size]:


        return self.full_kv_pool.get_kv_buffer_shape()

    def get_contiguous_buf_infos(self):
        return self.full_kv_pool.get_contiguous_buf_infos()

    def get_kv_layer_ids(self):

        layer_ids = list(self.full_attention_layer_id_mapping)
        return layer_ids if self.use_mla else layer_ids * 2

    def get_state_buf_infos(self):
        mamba_data_ptrs, mamba_data_lens, mamba_item_lens = (
            self.mamba_pool.get_contiguous_buf_infos()
        )
        return mamba_data_ptrs, mamba_data_lens, mamba_item_lens

    def get_state_dim_per_tensor(self):

        return self.mamba_pool.get_state_dim_per_tensor()

    def get_state_layer_ids(self):

        return self.mamba_pool.get_state_layer_ids()

    def get_state_slice_outer_counts(self):

        return self.mamba_pool.get_state_slice_outer_counts()

    def get_state_conv_shard_groups(self):

        return self.mamba_pool.get_state_conv_shard_groups()

    def maybe_get_custom_mem_pool(self):
        return self.full_kv_pool.maybe_get_custom_mem_pool()

    def _transfer_full_attention_id(self, layer_id: int):
        if layer_id not in self.full_attention_layer_id_mapping:
            raise ValueError(
                f"{layer_id=} not in full attention layers: {self.full_attention_layer_id_mapping.keys()}"
            )
        return self.full_attention_layer_id_mapping[layer_id]

    def register_layer_transfer_counter(self, layer_transfer_counter: LayerDoneCounter):
        self.layer_transfer_counter = layer_transfer_counter


        self.full_kv_pool.register_layer_transfer_counter(None)

    def _wait_for_layer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

    def get_key_buffer(self, layer_id: int, scale: Optional[float] = None):
        self._wait_for_layer(layer_id)
        layer_id = self._transfer_full_attention_id(layer_id)
        if scale is not None:
            return self.full_kv_pool.get_key_buffer(layer_id, scale)
        return self.full_kv_pool.get_key_buffer(layer_id)

    def get_value_buffer(self, layer_id: int, scale: Optional[float] = None):
        self._wait_for_layer(layer_id)
        layer_id = self._transfer_full_attention_id(layer_id)
        if scale is not None:
            return self.full_kv_pool.get_value_buffer(layer_id, scale)
        return self.full_kv_pool.get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_kv_buffer(layer_id)

    def get_raw_kv_buffer(
        self, layer_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._wait_for_layer(layer_id)
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_raw_kv_buffer(layer_id)

    def get_dequant_workspace(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.full_kv_pool.get_dequant_workspace()

    def get_flashinfer_dequant_workspace_kv_buffer(self, layer, *args, **kwargs):
        self._wait_for_layer(layer.layer_id)
        local_layer_id = self._transfer_full_attention_id(layer.layer_id)
        return self.full_kv_pool.get_flashinfer_dequant_workspace_kv_buffer(
            layer, *args, layer_id_override=local_layer_id, **kwargs
        )

    def get_flashinfer_decode_dequant_workspace_kv_buffer(self, layer, *args, **kwargs):
        self._wait_for_layer(layer.layer_id)
        local_layer_id = self._transfer_full_attention_id(layer.layer_id)
        return self.full_kv_pool.get_flashinfer_decode_dequant_workspace_kv_buffer(
            layer, *args, layer_id_override=local_layer_id, **kwargs
        )

    def get_kv_scale_buffer(self, layer_id: int):

        self._wait_for_layer(layer_id)
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_kv_scale_buffer(layer_id)

    @contextmanager
    def _transfer_id_context(self, layer: RadixAttention):
        @contextmanager
        def _patch_layer_id(layer):
            original_layer_id = layer.layer_id
            layer.layer_id = self._transfer_full_attention_id(layer.layer_id)
            try:
                yield
            finally:
                layer.layer_id = original_layer_id

        with _patch_layer_id(layer):
            yield

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        dcp_kv_mask: Optional[torch.Tensor] = None,
    ):


        loc, _, full_loc = unwrap_write_loc(loc)
        layer_id = self._transfer_full_attention_id(layer.layer_id)
        if not self.use_mla:
            write_loc = full_loc if full_loc is not None else loc
            self.full_kv_pool.set_kv_buffer(
                layer,
                write_loc,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=layer_id,
                dcp_kv_mask=dcp_kv_mask,
            )
        else:


            write_loc = full_loc if full_loc is not None else loc
            with self._transfer_id_context(layer):
                self.full_kv_pool.set_kv_buffer(
                    layer,
                    write_loc,
                    cache_k,
                    cache_v,
                )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        self.full_kv_pool.move_kv_cache(tgt_loc, src_loc)

    def get_cpu_copy(self, indices, mamba_indices=None):
        kv_cpu = self.full_kv_pool.get_cpu_copy(indices)

        mamba_cpu = (
            self.mamba_pool.get_cpu_copy(self._mamba_translate(mamba_indices))
            if mamba_indices is not None
            else None
        )
        return kv_cpu, mamba_cpu

    def load_cpu_copy(self, cache_cpu, indices, mamba_indices=None):
        kv_cpu, mamba_cpu = cache_cpu
        self.full_kv_pool.load_cpu_copy(kv_cpu, indices)
        if mamba_cpu is not None and mamba_indices is not None:
            self.mamba_pool.load_cpu_copy(
                mamba_cpu, self._mamba_translate(mamba_indices)
            )

    def get_v_head_dim(self):


        return self.full_kv_pool.get_value_buffer(self.full_kv_pool.start_layer).shape[
            -1
        ]

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
        loc_is_dense: bool = False,
    ):
        assert self.use_mla, "set_mla_kv_buffer called when use_mla is False"


        if not loc_is_dense:
            loc = self._full_translate(loc)
        with self._transfer_id_context(layer):
            self.full_kv_pool.set_mla_kv_buffer(layer, loc, cache_k_nope, cache_k_rope)

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        assert self.use_mla, "get_mla_kv_buffer called when use_mla is False"
        loc = self._full_translate(loc)
        with self._transfer_id_context(layer):
            return self.full_kv_pool.get_mla_kv_buffer(layer, loc, dst_dtype)


class MLATokenToKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        use_dsa: bool = False,
        override_kv_cache_dim: Optional[int] = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )

        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.use_dsa = use_dsa
        self.dsa_kv_cache_store_fp8 = (
            use_dsa
            and dtype == torch.float8_e4m3fn
            and override_kv_cache_dim is not None
        )


        self.kv_cache_dim = (
            override_kv_cache_dim
            if self.dsa_kv_cache_store_fp8
            else (kv_lora_rank + qk_rope_head_dim)
        )

        self._create_buffers()

        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.kv_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        if not use_dsa:

            self._finalize_allocation_log(size)

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):

                self.kv_buffer = [
                    torch.zeros(
                        (self.size + self.page_size, 1, self.kv_cache_dim),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def _clear_buffers(self):
        del self.kv_buffer

    def get_kv_size_bytes(self):
        assert hasattr(self, "kv_buffer")
        kv_size_bytes = 0
        for kv_cache in self.kv_buffer:
            kv_size_bytes += get_tensor_size_bytes(kv_cache)
        return kv_size_bytes


    def get_contiguous_buf_infos(self):

        kv_data_ptrs = [self.kv_buffer[i].data_ptr() for i in range(self.layer_num)]
        kv_data_lens = [self.kv_buffer[i].nbytes for i in range(self.layer_num)]
        kv_item_lens = [
            self.kv_buffer[i][0].nbytes * self.page_size for i in range(self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id - self.start_layer].view(self.dtype)

        return self.kv_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id - self.start_layer][
                ..., : self.kv_lora_rank
            ].view(self.dtype)
        return self.kv_buffer[layer_id - self.start_layer][..., : self.kv_lora_rank]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        layer_id_override: Optional[int] = None,
    ):
        loc, _, _ = unwrap_write_loc(loc_info)
        maybe_detect_oob(loc, 0, self.size + self.page_size, "set_kv_buffer (MLA)")
        layer_id = (
            layer_id_override if layer_id_override is not None else layer.layer_id
        )
        assert not self.dsa_kv_cache_store_fp8
        parallel = get_parallel()
        if parallel.dcp_enabled:
            valid_mask = loc % parallel.attn_dcp_size == parallel.attn_dcp_rank
            if not valid_mask.all():
                loc = loc[valid_mask]
                cache_k = cache_k[valid_mask]
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)

        if self.store_dtype != self.dtype:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k.view(
                self.store_dtype
            )
        else:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k

    def _write_mla_kv_buffer(
        self,
        dst_buffer: torch.Tensor,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ) -> None:
        if _is_hip and self.use_dsa and self.dtype == fp8_dtype:


            set_mla_kv_buffer_triton_fp8_quant(
                dst_buffer,
                loc,
                cache_k_nope,
                cache_k_rope,
                fp8_dtype,
            )
        elif self.dsa_kv_cache_store_fp8:


            cache_k_nope_fp8, cache_k_rope_fp8 = quantize_k_cache_separate(
                cache_k_nope, cache_k_rope
            )


            set_mla_kv_buffer_triton(
                dst_buffer,
                loc,
                cache_k_nope_fp8,
                cache_k_rope_fp8,
            )
        else:
            if cache_k_nope.dtype != self.dtype:
                cache_k_nope = cache_k_nope.to(self.dtype)
                cache_k_rope = cache_k_rope.to(self.dtype)
            if self.store_dtype != self.dtype:
                cache_k_nope = cache_k_nope.view(self.store_dtype)
                cache_k_rope = cache_k_rope.view(self.store_dtype)

            set_mla_kv_buffer_triton(
                dst_buffer,
                loc,
                cache_k_nope,
                cache_k_rope,
            )

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
        layer_id_override: Optional[int] = None,
    ):

        maybe_detect_oob(
            loc,
            0,
            (self.size + self.page_size) * get_parallel().attn_dcp_size,
            "set_mla_kv_buffer (MLA)",
        )
        layer_id = (
            layer_id_override if layer_id_override is not None else layer.layer_id
        )
        self._write_mla_kv_buffer(
            self.kv_buffer[layer_id - self.start_layer],
            loc,
            cache_k_nope,
            cache_k_rope,
        )

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):

        layer_id = layer.layer_id
        kv_buffer = self.get_key_buffer(layer_id)
        dst_dtype = dst_dtype or self.dtype
        cache_k_nope = torch.empty(
            (loc.shape[0], 1, self.kv_lora_rank),
            dtype=dst_dtype,
            device=kv_buffer.device,
        )
        cache_k_rope = torch.empty(
            (loc.shape[0], 1, self.qk_rope_head_dim),
            dtype=dst_dtype,
            device=kv_buffer.device,
        )
        get_mla_kv_buffer_triton(kv_buffer, loc, cache_k_nope, cache_k_rope)
        return cache_k_nope, cache_k_rope

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):

        size_limit = self.size + self.page_size
        maybe_detect_oob(tgt_loc, 0, size_limit, "move_kv_cache tgt_loc")
        maybe_detect_oob(src_loc, 0, size_limit, "move_kv_cache src_loc")

        if tgt_loc.numel() == 0:
            return

        tgt_loc_flat = tgt_loc.view(-1).long()
        src_loc_flat = src_loc.view(-1).long()
        for kv_cache in self.kv_buffer:
            kv_cache[tgt_loc_flat] = kv_cache[src_loc_flat]

    def get_cpu_copy(self, indices, mamba_indices=None):
        current_platform.synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                kv_cpu = self.kv_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                kv_cache_cpu[-1].append(kv_cpu)
        current_platform.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        current_platform.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                kv_cpu = kv_cache_cpu[layer_id][i // chunk_size]
                assert kv_cpu.shape[0] == len(chunk_indices)
                kv_chunk = kv_cpu.to(self.kv_buffer[0].device, non_blocking=True)
                self.kv_buffer[layer_id][chunk_indices] = kv_chunk
        current_platform.synchronize()


class MLATokenToKVPoolFP4(MLATokenToKVPool):
    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):

                m = self.size + self.page_size
                n = 1
                k = self.kv_cache_dim

                scale_block_size = 16
                self.store_dtype = torch.uint8

                self.kv_buffer = [
                    torch.zeros(
                        (m, n, k // 2),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

                self.kv_scale_buffer = [
                    torch.zeros(
                        (m, k // scale_block_size),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def _clear_buffers(self):
        del self.kv_buffer
        del self.kv_scale_buffer

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            cache_k_nope_fp4 = self.kv_buffer[layer_id - self.start_layer].view(
                torch.uint8
            )
            cache_k_nope_fp4_sf = self.kv_scale_buffer[layer_id - self.start_layer]

            from sglang.srt.layers.quantization.kvfp4_tensor import (
                FP4MXBlock16KVQuantizeUtil,
            )

            cache_k_nope_fp4_dequant = FP4MXBlock16KVQuantizeUtil.batched_dequantize(
                cache_k_nope_fp4, cache_k_nope_fp4_sf
            )
            return cache_k_nope_fp4_dequant

        return self.kv_buffer[layer_id - self.start_layer]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):

        loc, _, _ = unwrap_write_loc(loc_info)
        maybe_detect_oob(loc, 0, self.size + self.page_size, "set_kv_buffer (MLA-FP4)")
        layer_id = layer.layer_id
        assert not self.dsa_kv_cache_store_fp8
        if cache_k.dtype != self.dtype:
            from sglang.srt.layers.quantization.kvfp4_tensor import (
                FP4MXBlock16KVQuantizeUtil,
            )

            cache_k_fp4, cache_k_fp4_sf = FP4MXBlock16KVQuantizeUtil.batched_quantize(
                cache_k
            )

        if self.store_dtype != self.dtype:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k_fp4.view(
                self.store_dtype
            )
            self.kv_scale_buffer[layer_id - self.start_layer][loc] = (
                cache_k_fp4_sf.view(self.store_dtype)
            )
        else:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        maybe_detect_oob(
            loc, 0, self.size + self.page_size, "set_mla_kv_buffer (MLA-FP4)"
        )
        layer_id = layer.layer_id

        if self.dsa_kv_cache_store_fp8:


            cache_k = torch.cat([cache_k_nope, cache_k_rope], dim=-1)
            cache_k = quantize_k_cache(cache_k.unsqueeze(1)).squeeze(1)
            cache_k = cache_k.view(self.store_dtype)
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k
        else:
            if cache_k_nope.dtype != self.dtype:
                from sglang.srt.layers.quantization.kvfp4_tensor import (
                    FP4MXBlock16KVQuantizeUtil,
                )

                cache_k_nope_fp4, cache_k_nope_fp4_sf = (
                    FP4MXBlock16KVQuantizeUtil.batched_quantize(cache_k_nope)
                )
                cache_k_rope_fp4, cache_k_rope_fp4_sf = (
                    FP4MXBlock16KVQuantizeUtil.batched_quantize(cache_k_rope)
                )

            if self.store_dtype != self.dtype:
                cache_k_nope = cache_k_nope.view(self.store_dtype)
                cache_k_rope = cache_k_rope.view(self.store_dtype)

            set_mla_kv_buffer_triton(
                self.kv_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope_fp4,
                cache_k_rope_fp4,
            )
            set_mla_kv_scale_buffer_triton(
                self.kv_scale_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope_fp4_sf,
                cache_k_rope_fp4_sf,
            )


class DSATokenToKVPool(MLATokenToKVPool):
    quant_block_size = 128
    index_k_with_scale_buffer_dtype = torch.uint8
    rope_storage_dtype = torch.bfloat16

    def __init__(
        self,
        size: int,
        page_size: int,
        kv_lora_rank: int,
        dtype: torch.dtype,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        index_head_dim: int,
        enable_memory_saver: bool,
        kv_cache_dim: int,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        index_buf_size: Optional[int] = None,
        skip_topk_layers: Optional[List[bool]] = None,
    ):
        override_dim = (
            kv_cache_dim if kv_cache_dim != kv_lora_rank + qk_rope_head_dim else None
        )

        super().__init__(
            size,
            page_size,
            dtype,
            kv_lora_rank,
            qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
            use_dsa=True,
            override_kv_cache_dim=override_dim,
        )


        self.index_head_dim = index_head_dim
        if index_buf_size is None:
            index_buf_size = size
        self.index_buf_size = index_buf_size

        assert index_head_dim == 128

        self.skip_topk_layers = (
            list(skip_topk_layers)
            if skip_topk_layers is not None
            else [False] * layer_num
        )
        assert len(self.skip_topk_layers) == layer_num

        if _is_hip:
            if aiter_can_use_preshuffle_paged_mqa():
                assert (
                    self.page_size % 16 == 0
                ), f"HIP preshuffle requires page_size to be a multiple of 16, got {self.page_size}"
            else:
                assert (
                    self.page_size == 1
                ), f"HIP legacy DSA path requires page_size == 1, got {self.page_size}"
        else:
            assert self.page_size == 64
        self.index_key_cache = self._create_index_key_cache()
        self._finalize_allocation_log(size)

    def _create_index_key_cache(self) -> IndexKeyCache:
        return IndexKeyCache(self, self.index_buf_size)

    @property
    def index_k_with_scale_buffer(self):

        return self.index_key_cache.buffer

    def _clear_buffers(self):
        super()._clear_buffers()
        self.index_key_cache.clear()

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):

        super().move_kv_cache(tgt_loc, src_loc)
        self.index_key_cache.move(tgt_loc, src_loc)

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        return self.index_key_cache.get_local_buffer(layer_id)

    def get_index_k_continuous(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        return self.index_key_cache.get_k_continuous(layer_id, seq_len, page_indices)

    def get_index_k_scale_continuous(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        return self.index_key_cache.get_k_scale_continuous(
            layer_id, seq_len, page_indices
        )

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len_tensor: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
    ):
        return self.index_key_cache.get_k_and_scale(
            layer_id, seq_len_tensor, page_indices, seq_len_sum, max_seq_len
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        self.index_key_cache.store_quantized(layer_id, loc, index_k, index_k_scale)

    def get_cpu_copy(self, indices, mamba_indices=None):
        kv_cache_cpu = super().get_cpu_copy(indices, mamba_indices=mamba_indices)
        return {"kv": kv_cache_cpu, "index_k": self.index_key_cache.cpu_copy(indices)}

    def load_cpu_copy(self, kv_cache_cpu_dict, indices, mamba_indices=None):
        super().load_cpu_copy(
            kv_cache_cpu_dict["kv"], indices, mamba_indices=mamba_indices
        )
        self.index_key_cache.load_cpu_copy(kv_cache_cpu_dict["index_k"], indices)

    def get_state_buf_infos(self):
        return self.index_key_cache.state_buf_infos()

    def get_kv_size_bytes(self):
        kv_size_bytes = super().get_kv_size_bytes()
        for index_k_cache in self.index_k_with_scale_buffer:
            kv_size_bytes += get_tensor_size_bytes(index_k_cache)
        return kv_size_bytes


def move_kv_cache_native(
    k_buffer: List[torch.Tensor],
    v_buffer: List[torch.Tensor],
    tgt_loc: torch.Tensor,
    src_loc: torch.Tensor,
    page_size: int = 1,
):

    if tgt_loc.numel() == 0:
        return

    tgt_loc_flat = tgt_loc.view(-1).long()
    src_loc_flat = src_loc.view(-1).long()
    for k_cache, v_cache in zip(k_buffer, v_buffer):
        if k_cache.ndim == 4:
            if page_size == 1:

                k_cache[tgt_loc_flat, 0] = k_cache[src_loc_flat, 0]
                v_cache[tgt_loc_flat, 0] = v_cache[src_loc_flat, 0]
            else:
                tgt_page = tgt_loc_flat // page_size
                tgt_tok = tgt_loc_flat % page_size
                src_page = src_loc_flat // page_size
                src_tok = src_loc_flat % page_size
                k_cache[tgt_page, tgt_tok] = k_cache[src_page, src_tok]
                v_cache[tgt_page, tgt_tok] = v_cache[src_page, src_tok]
        else:
            k_cache[tgt_loc_flat] = k_cache[src_loc_flat]
            v_cache[tgt_loc_flat] = v_cache[src_loc_flat]


@triton.jit
def masked_set_kv_buffer_kernel(
    k_ptr,
    v_ptr,
    k_buffer_ptr,
    v_buffer_ptr,
    loc_ptr,
    mask_ptr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    CHUNK: tl.constexpr,
    k_stride_B: tl.constexpr,
    k_stride_H: tl.constexpr,
    v_stride_B: tl.constexpr,
    v_stride_H: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N:
        return

    do_write = tl.load(mask_ptr + pid) != 0
    if not do_write:
        return

    loc = tl.load(loc_ptr + pid)
    total = H * D
    num_chunks = tl.cdiv(total, CHUNK)

    for c in range(num_chunks):
        offs = tl.arange(0, CHUNK)
        idx = c * CHUNK + offs
        mask = idx < total
        row = idx // D
        col = idx % D

        key = tl.load(k_ptr + pid * k_stride_B + row * k_stride_H + col, mask=mask)
        tl.store(k_buffer_ptr + loc * H * D + idx, key, mask=mask)

        value = tl.load(v_ptr + pid * v_stride_B + row * v_stride_H + col, mask=mask)
        tl.store(v_buffer_ptr + loc * H * D + idx, value, mask=mask)


class MHATokenToKOnlyPool(KVCache):


    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.head_num = head_num
        self.head_dim = head_dim
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                self.k_buffer = [
                    torch.zeros(
                        (size + page_size, head_num, head_dim),
                        dtype=self.store_dtype,
                        device=device,
                    )
                    for _ in range(layer_num)
                ]
        self._finalize_allocation_log(size)

    def _get_key_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.k_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.k_buffer[layer_id - self.start_layer]

    def register_layer_transfer_counter(
        self, layer_transfer_counter: LayerDoneCounter
    ) -> None:
        self.layer_transfer_counter = layer_transfer_counter

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_key_buffer(layer_id)

    def set_k_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
        self.k_buffer[layer_id][loc] = cache_k

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError("MHATokenToKOnlyPool does not allocate V")

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("MHATokenToKOnlyPool does not allocate V")

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ) -> None:

        raise NotImplementedError(
            "MHATokenToKOnlyPool: use set_index_k_buffer on the parent "
            "MiniMaxSparseKVPool — this pool does not store V"
        )

    def get_kv_size_bytes(self):
        k_size_bytes = sum(get_tensor_size_bytes(k) for k in self.k_buffer)
        return k_size_bytes, 0


class MiniMaxSparseKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        idx_head_dim: int,
        dense_layer_ids: List[int],
        sparse_layer_ids: List[int],
        device: str,
        disable_value_sparse_layer_ids: Optional[List[int]] = None,
        enable_memory_saver: bool = False,
        index_dtype: Optional[torch.dtype] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        main_pool_cls=MHATokenToKVPool,
        index_kv_pool_cls=MHATokenToKVPool,
        index_k_pool_cls=MHATokenToKOnlyPool,
    ):

        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self.use_minimax_fused_kv_index_store = (
            envs.SGLANG_OPT_USE_MINIMAX_FUSED_KV_INDEX_STORE.get()
        )

        local_dense_layer_ids = [
            lid for lid in dense_layer_ids if start_layer <= lid < end_layer
        ]
        local_sparse_layer_ids = [
            lid for lid in sparse_layer_ids if start_layer <= lid < end_layer
        ]

        index_dtype = index_dtype if index_dtype is not None else dtype


        disable_set = set(disable_value_sparse_layer_ids or [])
        local_kv_sparse_layer_ids = [
            g for g in local_sparse_layer_ids if g not in disable_set
        ]
        local_k_only_sparse_layer_ids = [
            g for g in local_sparse_layer_ids if g in disable_set
        ]


        self.sparse_layer_id_mapping: dict[int, int] = {
            gid: i for i, gid in enumerate(local_sparse_layer_ids)
        }

        self.index_kv_layer_id_mapping: dict[int, int] = {
            gid: i for i, gid in enumerate(local_kv_sparse_layer_ids)
        }
        self.index_k_layer_id_mapping: dict[int, int] = {
            gid: i for i, gid in enumerate(local_k_only_sparse_layer_ids)
        }

        self.main_pool = main_pool_cls(
            size=size,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=len(local_dense_layer_ids) + len(local_sparse_layer_ids),
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
            end_layer=end_layer,
        )

        self.index_kv_pool: Optional[MHATokenToKVPool] = (
            index_kv_pool_cls(
                size=size,
                page_size=page_size,
                dtype=index_dtype,
                head_num=1,
                head_dim=idx_head_dim,
                layer_num=len(local_kv_sparse_layer_ids),
                device=device,
                enable_memory_saver=enable_memory_saver,
            )
            if local_kv_sparse_layer_ids
            else None
        )

        self.index_k_pool: Optional[MHATokenToKOnlyPool] = (
            index_k_pool_cls(
                size=size,
                page_size=page_size,
                dtype=index_dtype,
                head_num=1,
                head_dim=idx_head_dim,
                layer_num=len(local_k_only_sparse_layer_ids),
                device=device,
                enable_memory_saver=enable_memory_saver,
            )
            if local_k_only_sparse_layer_ids
            else None
        )

        self.mem_usage = self.main_pool.mem_usage
        if self.index_kv_pool is not None:
            self.mem_usage += self.index_kv_pool.mem_usage
        if self.index_k_pool is not None:
            self.mem_usage += self.index_k_pool.mem_usage


        self.layer_num = self.main_pool.layer_num
        self.start_layer = self.main_pool.start_layer
        self.end_layer = self.main_pool.end_layer

        self.head_num = self.main_pool.head_num
        self.head_dim = self.main_pool.head_dim
        self.layer_transfer_counter = None

    def register_layer_transfer_counter(
        self, layer_transfer_counter: LayerDoneCounter
    ) -> None:
        self.layer_transfer_counter = layer_transfer_counter

    def get_kv_cache_quant_method(self) -> Any:


        return self.main_pool.get_kv_cache_quant_method()

    def _wait_for_layer(self, layer_id: int) -> None:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        self._wait_for_layer(layer_id)
        return self.main_pool.get_key_buffer(layer_id)

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        self._wait_for_layer(layer_id)
        return self.main_pool.get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._wait_for_layer(layer_id)
        return self.main_pool.get_kv_buffer(layer_id)

    def get_index_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._wait_for_layer(layer_id)
        mapped_id = self.index_kv_layer_id_mapping.get(layer_id)
        if mapped_id is None:
            raise ValueError(
                f"layer_id={layer_id} does not have an index V cache "
                f"(either dense, or in the K-only group). "
                f"index_kv layers: {list(self.index_kv_layer_id_mapping.keys())}"
            )
        return self.index_kv_pool.get_kv_buffer(mapped_id)

    def get_index_k_buffer(self, layer_id: int) -> torch.Tensor:
        self._wait_for_layer(layer_id)


        mapped_id = self.index_k_layer_id_mapping.get(layer_id)
        if mapped_id is not None:
            return self.index_k_pool.get_key_buffer(mapped_id)
        mapped_id = self.index_kv_layer_id_mapping.get(layer_id)
        if mapped_id is not None:
            return self.index_kv_pool.get_key_buffer(mapped_id)
        raise ValueError(
            f"layer_id={layer_id} is not a sparse attention layer; "
            f"sparse layers: {list(self.sparse_layer_id_mapping.keys())}"
        )

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
    ) -> None:

        self.main_pool.set_kv_buffer(
            layer,
            loc,
            cache_k,
            cache_v,
            k_scale,
            v_scale,
        )

    def set_index_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_idx_k: torch.Tensor,
        cache_idx_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
    ) -> None:
        mapped_id = self.index_kv_layer_id_mapping.get(layer.layer_id)
        if mapped_id is None:
            raise ValueError(
                f"layer.layer_id={layer.layer_id} does not have an index V "
                f"cache (either dense, or in the K-only group). "
                f"index_kv layers: {list(self.index_kv_layer_id_mapping.keys())}"
            )
        self.index_kv_pool.set_kv_buffer(
            layer,
            loc,
            cache_idx_k,
            cache_idx_v,
            k_scale,
            v_scale,
            layer_id_override=mapped_id,
        )

    def set_index_k_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_idx_k: torch.Tensor,
        k_scale: Optional[float] = None,
    ) -> None:
        mapped_id = self.index_k_layer_id_mapping.get(layer.layer_id)
        if mapped_id is None:
            raise ValueError(
                f"layer.layer_id={layer.layer_id} is not in the K-only "
                f"sparse group. K-only layers: "
                f"{list(self.index_k_layer_id_mapping.keys())}"
            )
        sub_pool = self.index_k_pool
        if cache_idx_k.dtype != sub_pool.dtype:
            if k_scale is not None:
                cache_idx_k = cache_idx_k / k_scale
        sub_pool.set_k_buffer(mapped_id, loc, cache_idx_k)

    def _can_fuse_kv_index_store(
        self,
        index_pool: MHATokenToKVPool,
        cache_k: torch.Tensor,
        cache_idx_k: torch.Tensor,
    ) -> bool:

        main = self.main_pool
        return (
            self.use_minimax_fused_kv_index_store
            and _is_cuda


            and main.store_dtype == main.dtype
            and index_pool.store_dtype == index_pool.dtype
            and cache_k.dtype == main.dtype
            and cache_idx_k.dtype == index_pool.dtype

            and main.dtype == index_pool.dtype
            and main.head_dim == index_pool.head_dim

            and (main.head_dim * main.dtype.itemsize) % 16 == 0
        )

    def set_fused_kv_index_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        cache_idx_k: torch.Tensor,
        cache_idx_v: Optional[torch.Tensor],
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        idx_k_scale: Optional[float] = None,
        idx_v_scale: Optional[float] = None,
    ) -> None:

        disable_value = cache_idx_v is None
        index_pool = self.index_k_pool if disable_value else self.index_kv_pool

        if index_pool is not None and self._can_fuse_kv_index_store(
            index_pool, cache_k, cache_idx_k
        ):
            from sglang.kernels.ops.kvcache.minimax_store_kv_index import store_kv_index

            main = self.main_pool
            head_bytes = main.head_dim * main.dtype.itemsize
            if disable_value:
                idx_k_cache = self.get_index_k_buffer(layer.layer_id).flatten(1)
                idx_v_cache = None
            else:
                ik, iv = self.get_index_kv_buffer(layer.layer_id)
                idx_k_cache, idx_v_cache = ik.flatten(1), iv.flatten(1)
            store_kv_index(
                cache_k.flatten(1),
                cache_v.flatten(1),
                main.get_key_buffer(layer.layer_id).flatten(1),
                main.get_value_buffer(layer.layer_id).flatten(1),
                cache_idx_k.flatten(1),
                idx_k_cache,
                None if disable_value else cache_idx_v.flatten(1),
                idx_v_cache,
                loc,
                num_kv_heads=main.head_num,
                head_bytes=head_bytes,
            )
            return


        self.set_kv_buffer(layer, loc, cache_k, cache_v, k_scale, v_scale)
        if disable_value:
            self.set_index_k_buffer(layer, loc, cache_idx_k, idx_k_scale)
        else:
            self.set_index_kv_buffer(
                layer,
                loc,
                cache_idx_k,
                cache_idx_v,
                idx_k_scale,
                idx_v_scale,
            )

    def get_kv_size_bytes(self):
        sub_pools = [self.main_pool, self.index_kv_pool, self.index_k_pool]
        sizes = [p.get_kv_size_bytes() for p in sub_pools if p is not None]
        return sum(k for k, _ in sizes), sum(v for _, v in sizes)

    def get_contiguous_buf_infos(self):

        return self.main_pool.get_contiguous_buf_infos()

    def get_index_k_state_buf_infos(self):


        pool = self.index_k_pool
        n = pool.layer_num
        data_ptrs = [pool.k_buffer[i].data_ptr() for i in range(n)]
        data_lens = [pool.k_buffer[i].nbytes for i in range(n)]
        item_lens = [pool.k_buffer[i][0].nbytes * pool.page_size for i in range(n)]
        return data_ptrs, data_lens, item_lens

    def maybe_get_custom_mem_pool(self):
        return self.main_pool.maybe_get_custom_mem_pool()

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):


        raise NotImplementedError(
            "move_kv_cache is not yet supported for MiniMaxSparseKVPool: "
            "sub-pools must be built with enable_kv_cache_copy=True first."
        )

    def get_v_head_dim(self):


        return self.main_pool.get_value_buffer(self.main_pool.start_layer).shape[-1]
