from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch_npu

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    ONLINE_C128,
    DeepSeekV4IndexerPool,
    DeepSeekV4SingleKVPool,
    DeepSeekV4TokenToKVPool,
)
from sglang.srt.runtime_context import get_schedule


class NPUDeepSeekV4SingleKVPool(DeepSeekV4SingleKVPool):


    def __init__(self, *args, kernel_page_size: int, **kwargs):


        self.kernel_page_size = kernel_page_size
        super().__init__(*args, **kwargs)

    def create_buffer(self, *, num_pages: int):

        if self.store_dtype != torch.bfloat16:
            return super().create_buffer(num_pages=num_pages)
        kv_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.kv_cache_total_dim = kv_dim


        npu_num_pages = (self.size + self.kernel_page_size + 1) // self.kernel_page_size
        return torch.zeros(
            npu_num_pages,
            self.kernel_page_size,
            1,
            kv_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )


class NPUCompressStatePool(CompressStatePool):


    def __init__(
        self,
        *,
        size: int,
        overlap: bool,
        head_dim: int,
        dtype: torch.dtype,
        device: str,
        enable_memory_saver: bool,
        ratio: int,
        ring_size: int,
        swa_page_size: int,
    ):
        assert ratio in (
            4,
            128,
        ), f"NPUCompressStatePool only supports ratio in (4, 128); got {ratio}"
        assert dtype == torch.float32, (
            "Atlas A3 npu.compressor requires FP32 state_cache, "
            f"but NPUCompressStatePool got {dtype}."
        )
        assert ring_size > 0, f"ring_size must be positive, got {ring_size}"
        super().__init__(
            size=size,
            ring_size=ring_size,
            overlap=overlap,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
            enable_memory_saver=enable_memory_saver,
            ratio=ratio,
            online=False,
            swa_page_size=swa_page_size,
            state_cache_page_size=ring_size,
        )
        self.dummy_state_loc = self._size - 1


        if ratio == 128:
            self.kv_score_buffer.clear()

    def _replace_invalid_with_dummy(self, state_loc: torch.Tensor) -> torch.Tensor:
        return torch.where(
            state_loc < 0,
            torch.full_like(state_loc, self.dummy_state_loc),
            state_loc,
        )

    def translate_from_swa_loc_to_state_loc(
        self, swa_loc: torch.Tensor
    ) -> torch.Tensor:
        return self._replace_invalid_with_dummy(
            super().translate_from_swa_loc_to_state_loc(swa_loc)
        )

    def translate_from_req_position_to_state_loc(
        self, req_pool_indices: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        return self._replace_invalid_with_dummy(
            super().translate_from_req_position_to_state_loc(
                req_pool_indices, positions
            )
        )


class NPUDeepSeekV4IndexerPool(DeepSeekV4IndexerPool):


    def __init__(self, *args, kernel_page_size: int, **kwargs):

        self._kernel_page_size = kernel_page_size
        super().__init__(*args, **kwargs)

    def _create_buffer(self):


        super()._create_buffer()
        kp = self._kernel_page_size
        npu_num_pages = (self.size + kp + 1) // kp
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.index_k_buffer = [
                torch.zeros(
                    npu_num_pages,
                    kp,
                    1,
                    self.index_head_dim,
                    dtype=torch.int8,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]
            self.index_scale_buffer = [
                torch.zeros(
                    npu_num_pages,
                    kp,
                    1,
                    1,
                    dtype=torch.float16,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]

    @property
    def has_npu_storage(self) -> bool:
        return True

    def get_index_k(self, layer_id: int) -> torch.Tensor:
        return self.index_k_buffer[layer_id]

    def get_index_scale(self, layer_id: int) -> torch.Tensor:
        return self.index_scale_buffer[layer_id]

    def set_index_k_scale(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: Optional[torch.Tensor],
    ) -> None:


        d = self.index_head_dim
        loc_long = loc.view(-1, 1).long()
        torch_npu.npu_scatter_nd_update_(
            self.index_k_buffer[layer_id].view(-1, 1, d),
            loc_long,
            index_k.to(torch.int8).view(-1, 1, d),
        )
        if index_k_scale is not None:
            torch_npu.npu_scatter_nd_update_(
                self.index_scale_buffer[layer_id].view(-1, 1, 1),
                loc_long,
                index_k_scale.to(torch.float16).view(-1, 1, 1),
            )


class DSV4NPUTokenToKVPool(DeepSeekV4TokenToKVPool):


    def __init__(self, *args, **kwargs):
        c128_page_size = get_schedule().c128_page_size
        if c128_page_size <= 0 or c128_page_size % 16 != 0:
            raise ValueError(
                "c128_page_size must be a positive multiple of 16 for the NPU "
                f"sparse-attention operator, got {c128_page_size}"
            )
        self.c128_page_size = c128_page_size
        super().__init__(*args, **kwargs)

    def _make_kv_pool(
        self,
        *,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        global_page_size: int,
        cls: type = DeepSeekV4SingleKVPool,
    ) -> NPUDeepSeekV4SingleKVPool:


        assert cls is DeepSeekV4SingleKVPool, (
            "enable_hisparse is not supported on the NPU DSV4 KV pool "
            f"(got c4 pool class {cls.__name__})."
        )


        is_c4_pool = page_size * 4 == global_page_size
        is_c128_pool = page_size * 128 == global_page_size
        if is_c4_pool:
            kernel_page_size = page_size
        elif is_c128_pool:
            kernel_page_size = self.c128_page_size
        else:
            kernel_page_size = global_page_size
        return NPUDeepSeekV4SingleKVPool(
            size,
            page_size,
            dtype,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            kernel_page_size=kernel_page_size,
        )

    def _get_state_pool(self, layer_id: int, from_indexer: bool) -> CompressStatePool:

        if from_indexer:
            return self.get_indexer_compress_states(layer_id)
        return self.get_attention_compress_states(layer_id)

    def _make_attn_state_pool(
        self, ratio: int, enable_memory_saver: bool
    ) -> NPUCompressStatePool:


        assert not (ratio == 128 and ONLINE_C128), (
            "SGLANG_OPT_USE_ONLINE_COMPRESS is incompatible with the "
            "NPU fused compressor (no online mode in the kernel)."
        )
        return NPUCompressStatePool(
            size=self._state_pool_size(ratio),
            ring_size=self.get_ring_size(ratio),
            overlap=ratio == 4,
            head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,
            dtype=self.c4_state_dtype if ratio == 4 else self.c128_state_dtype,
            device=self.device,
            enable_memory_saver=enable_memory_saver,
            ratio=ratio,
            swa_page_size=self.swa_page_size,
        )

    def _make_indexer_state_pool(
        self, ratio: int, enable_memory_saver: bool
    ) -> NPUCompressStatePool:


        return NPUCompressStatePool(
            size=self.c4_state_pool_size,
            ring_size=self.get_ring_size(ratio),
            overlap=ratio == 4,
            head_dim=self.indexer_head_dim,
            device=self.device,
            dtype=self.c4_state_dtype,
            enable_memory_saver=enable_memory_saver,
            ratio=ratio,
            swa_page_size=self.swa_page_size,
        )

    def _make_indexer_pool(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        index_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
    ) -> NPUDeepSeekV4IndexerPool:

        return NPUDeepSeekV4IndexerPool(
            size,
            page_size,
            dtype,
            index_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            kernel_page_size=page_size,
        )

    def get_contiguous_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:

        buffers = (
            self.c4_kv_pool.kv_buffer
            + self.c4_indexer_kv_pool.index_k_buffer
            + self.c4_indexer_kv_pool.index_scale_buffer
        )
        return (
            [buf.data_ptr() for buf in buffers],
            [buf.nbytes for buf in buffers],
            [buf[0].nbytes for buf in buffers],
        )

    def get_state_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:

        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []

        for buf in self.swa_kv_pool.kv_buffer:
            data_ptrs.append(buf.data_ptr())
            data_lens.append(buf.nbytes)
            item_lens.append(buf[0].nbytes)

        for pools in (self.compress_state_pools, self.indexer_compress_state_pools):
            for pool in pools:
                if pool is None or pool.ratio != 4:
                    continue
                state = pool.kv_score_buffer.kv_score
                data_ptrs.append(state.data_ptr())
                data_lens.append(state.nbytes)
                item_lens.append(state[0].nbytes * pool.ring_size)

        return data_ptrs, data_lens, item_lens

    def get_c128_kv_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        buffers = self.c128_kv_pool.kv_buffer
        return (
            [buf.data_ptr() for buf in buffers],
            [buf.nbytes for buf in buffers],
            [buf[0].nbytes for buf in buffers],
        )

    def get_state_cache(self, layer_id: int, from_indexer: bool) -> torch.Tensor:

        return self._get_state_pool(layer_id, from_indexer).state_cache_3d


    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        item = self.layer_mapping[layer_id]
        ratio = item.compress_ratio
        if ratio == 0:
            return self.swa_kv_pool.kv_buffer[item.compress_layer_id]
        if ratio == 4:
            return self.c4_kv_pool.kv_buffer[item.compress_layer_id]
        if ratio == 128:
            return self.c128_kv_pool.kv_buffer[item.compress_layer_id]
        raise ValueError(f"unsupported compress_ratio={ratio} for get_key_buffer")

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:

        return self.get_key_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        buf = self.get_key_buffer(layer_id)
        return buf, buf

    def get_swa_buffer(
        self, layer_id: int, loc: Optional[torch.Tensor] = None
    ) -> torch.Tensor:


        kv = self.swa_kv_pool.kv_buffer[layer_id]
        if loc is not None:
            kv = kv.flatten(0, 1)[loc]
        return kv

    def get_compress_buffer(
        self,
        layer_id: int,
        from_indexer: bool = False,
        loc: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:

        item = self.layer_mapping[layer_id]
        if item.compress_ratio == 4:
            if from_indexer:
                kv = self.c4_indexer_kv_pool.get_index_k(item.compress_layer_id)
            else:
                kv = self.c4_kv_pool.kv_buffer[item.compress_layer_id]
        elif item.compress_ratio == 128:
            assert not from_indexer, "c128 has no indexer pool"
            kv = self.c128_kv_pool.kv_buffer[item.compress_layer_id]
        else:
            return None
        if loc is not None:
            kv = kv.flatten(0, 1)[loc]
        return kv

    def set_swa_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache: torch.Tensor,
    ) -> None:


        buf = self.swa_kv_pool.kv_buffer[layer_id]
        buf_flat = buf.flatten(0, 1)


        if cache.ndim == buf_flat.ndim - 1:
            cache = cache.unsqueeze(1)
        buf_flat[loc] = cache.to(buf_flat.dtype)

    def set_swa_key_buffer_radix_fused_norm_rope(
        self,
        layer_id: int,
        swa_loc: torch.Tensor,
        kv: torch.Tensor,
        kv_weight: torch.Tensor,
        eps: float,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        kv_out = torch_npu.npu_rms_norm(kv, kv_weight, eps)[0]

        rope_dim = freqs_cis.shape[-1] * 2

        from sglang.srt.hardware_backend.npu.dsv4.dsv4_rope import Dsv4NpuRoPE

        cos, sin = Dsv4NpuRoPE.for_freqs(freqs_cis).get_cos_sin(
            positions,
            kv_out.dtype,
            view_4d=True,
            allow_build=True,
            cache_dtype=torch.float32,
        )
        Dsv4NpuRoPE.apply_rotary_mul_inplace(
            kv_out.reshape(kv_out.shape[0], -1, kv_out.shape[-1]),
            None,
            cos,
            sin,
            qk_nope_dim=kv_out.shape[-1] - rope_dim,
        )

        safe_swa_loc = swa_loc.clamp_min(0).to(torch.int64)
        self.set_swa_buffer(
            layer_id,
            safe_swa_loc,
            kv_out,
        )

    def set_compress_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        kv: torch.Tensor,
        kv_scale: Optional[torch.Tensor],
        from_indexer: bool,
    ) -> None:


        ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        device_type = kv.device.type
        if from_indexer:
            assert ratio == 4, f"indexer only on c4 layers, got ratio={ratio}"
            if device_type == "npu":
                assert (
                    self.c4_indexer_kv_pool.has_npu_storage
                ), "NPU index buffers not allocated — pool was init'd on CUDA?"
                self.c4_indexer_kv_pool.set_index_k_scale(
                    compress_layer_id, loc, kv, kv_scale
                )
                return
            if kv_scale is None:
                self.c4_indexer_kv_pool.set_index_fused(compress_layer_id, loc, kv)
                return
            self.c4_indexer_kv_pool.set_index_k_scale_buffer(
                compress_layer_id, loc, kv, kv_scale
            )
            return
        compress_pool = self.c4_kv_pool if ratio == 4 else self.c128_kv_pool
        if device_type == "npu":


            buf = compress_pool.kv_buffer[compress_layer_id]
            buf_flat = buf.flatten(0, 1)
            kv_view = kv.to(buf_flat.dtype)
            if kv_view.ndim == buf_flat.ndim - 1:
                kv_view = kv_view.unsqueeze(1)
            buf_flat[loc] = kv_view
            return
        compress_pool.set_key_buffer_fused(compress_layer_id, loc, kv)

    def get_compress_dequant_scale_buffer(
        self,
        layer_id: int,
        from_indexer: bool,
    ) -> torch.Tensor:


        assert from_indexer, "only indexer compress pool has dequant scale"
        compress_layer_id = self.layer_mapping[layer_id].compress_layer_id
        return self.c4_indexer_kv_pool.get_index_scale(compress_layer_id)
