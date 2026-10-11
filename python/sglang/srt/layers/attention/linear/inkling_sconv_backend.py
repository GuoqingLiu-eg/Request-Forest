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

from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
    scatter_mamba_states_after_mtp_verify,
)
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    ShortConvHybridAttnBackend,
)
from sglang.srt.layers.attention.linear.short_conv_backend import ShortConvAttnBackend
from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.inkling_common.kernels.sconv import (
    HIS_ONES,
    HIS_PREFIX,
    HIS_SEQ_MINUS_EXT,
    HIS_ZEROS,
    SconvDecodeMetadata,
    SconvExtendMetadata,
    SconvMetadataOut,
    fused_decode_sconv_metadata,
    fused_extend_sconv_metadata,
    precompute_helion_extend_metadata,
)
from sglang.srt.runtime_context import (
    get_exec,
    get_spec,
    mamba_cache_chunk_size,
)
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


class InklingShortConvMetadata(msgspec.Struct):


    cache_indices: Optional[torch.Tensor] = None
    query_start_loc: Optional[torch.Tensor] = None
    has_initial_state: Optional[torch.Tensor] = None
    precomputed: Optional[SconvExtendMetadata | SconvDecodeMetadata] = None


    track_conv_indices: Optional[torch.Tensor] = None


class InklingShortConvAttnBackend(ShortConvAttnBackend):


    cache_indices_dtype: torch.dtype = torch.int32


    needs_extend_host_mirrors: bool = False

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)


        self._mamba_cache = self.req_to_token_pool.mamba_pool.mamba_cache
        self.conv_state_len: int = self.conv_states_shape[2]
        self.mamba_cache_chunk_size = mamba_cache_chunk_size()


        self._slot_gather_recordable = (
            type(self.req_to_token_pool).translate_mamba_indices
            is HybridReqToTokenPool.translate_mamba_indices
        )

        self.sconv_metadata = InklingShortConvMetadata()

        self._alloc_graph_buffers()

    def _alloc_graph_buffers(self):

        cuda_graph_config = get_exec().graph.cuda_graph_config
        decode_bs: list[int] = []
        prefill_tokens: list[int] = []
        decode_max_bs = 0
        if cuda_graph_config is not None:
            decode_bs = list(cuda_graph_config.decode.bs or [])
            prefill_tokens = list(cuda_graph_config.prefill.bs or [])
            decode_max_bs = cuda_graph_config.decode.max_bs or 0
        draft_token_num = get_spec().speculative_num_draft_tokens or 1

        max_bs = max([self.req_to_token_pool.size, decode_max_bs, *decode_bs])
        max_tokens = max([max_bs, *prefill_tokens, max_bs * draft_token_num])

        dev = self.device
        self._graph_bufs = SconvMetadataOut(
            query_start_loc=torch.empty(max_bs + 1, dtype=torch.int32, device=dev),
            has_initial_state=torch.empty(max_bs, dtype=torch.bool, device=dev),
            cache_mask=torch.empty((max_bs, 1, 1), dtype=torch.bool, device=dev),
            safe_idx=torch.empty(max_bs, dtype=torch.int64, device=dev),
            cu=torch.empty(max_bs + 1, dtype=torch.int64, device=dev),
            si=torch.empty(max_tokens, dtype=torch.int32, device=dev),
        )
        self._graph_track_conv_indices = torch.zeros(
            (max_bs, self.conv_state_len), dtype=torch.int64, device=dev
        )


        self._graph_track_inert_mask = torch.zeros(max_bs, dtype=torch.bool, device=dev)
        self._graph_track_inert_indices = torch.zeros(
            max_bs, dtype=torch.int64, device=dev
        )
        self._graph_track_inert_seqlens = torch.zeros(
            max_bs, dtype=torch.int64, device=dev
        )


        self._alloc_cache_indices_buf(max_bs)
        self._track_window_offsets = torch.arange(
            self.conv_state_len, dtype=torch.int64, device=dev
        )
        self._track_index_floor = torch.zeros((1,), dtype=torch.int64, device=dev)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        super().init_cuda_graph_state(max_bs, max_num_tokens)

        self._graph_metadata_out(B=max_bs, T=max_num_tokens)

    def _graph_metadata_out(self, *, B: int, T: int) -> SconvMetadataOut:

        bufs = self._graph_bufs
        assert B + 1 <= bufs["query_start_loc"].shape[0] and T <= bufs["si"].shape[0], (
            f"short-conv metadata buffers too small for a captured shape: "
            f"B={B}, T={T} vs bs bound {bufs['query_start_loc'].shape[0] - 1}, "
            f"token bound {bufs['si'].shape[0]}"
        )
        return SconvMetadataOut(
            query_start_loc=bufs["query_start_loc"][: B + 1],
            has_initial_state=bufs["has_initial_state"][:B],
            cache_mask=bufs["cache_mask"][:B],
            safe_idx=bufs["safe_idx"][:B],
            cu=bufs["cu"][: B + 1],
            si=bufs["si"][:T],
        )

    def _forward_metadata(self, forward_batch: ForwardBatch) -> ForwardMetadata:

        return ForwardMetadata(
            query_start_loc=None,
            mamba_cache_indices=self._translate_mamba_indices(
                self.req_to_token_pool.get_mamba_indices(forward_batch.req_pool_indices)
            ),
        )

    def _reset_step_state(self):
        super()._reset_step_state()
        self.sconv_metadata = InklingShortConvMetadata()

    @staticmethod
    def _phase_records_metadata(forward_batch: ForwardBatch) -> bool:

        mode = forward_batch.forward_mode
        return (
            mode.is_decode_or_idle()
            or mode.is_target_verify()
            or mode.is_draft_extend_v2()
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):

        self._prepare_slot_indices(forward_batch)
        self._refresh_sconv_metadata(forward_batch, on_graph_path=False)

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):

        del in_capture
        if self._phase_records_metadata(forward_batch):
            if not self._slot_gather_recordable:
                self._prepare_slot_indices(forward_batch)
            return
        self._prepare_slot_indices(forward_batch)
        self._refresh_sconv_metadata(forward_batch, on_graph_path=True)

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):

        if not self._phase_records_metadata(forward_batch):
            return
        if self._slot_gather_recordable:
            self._prepare_slot_indices(forward_batch)
        self._refresh_sconv_metadata(forward_batch, on_graph_path=True)

    def init_forward_metadata_capture_cpu_graph(self, *args, **kwargs):
        raise NotImplementedError(
            "Inkling's short-conv backend has no CPU-graph path; its conv "
            "kernels are CUDA/Triton only."
        )

    def _prepare_slot_indices(self, forward_batch: ForwardBatch):
        self._reset_step_state()
        req_pool_indices = forward_batch.req_pool_indices
        n = req_pool_indices.shape[0]
        buf = self._cache_indices_buf
        if self._slot_gather_recordable and n <= buf.shape[0]:


            torch.index_select(
                self.req_to_token_pool.req_index_to_mamba_index_mapping,
                0,
                req_pool_indices,
                out=buf[:n],
            )
            self._cache_indices = buf[:n]
            self.forward_metadata = ForwardMetadata(
                query_start_loc=None, mamba_cache_indices=self._cache_indices
            )
            return
        self.forward_metadata = self._forward_metadata(forward_batch)
        self._refresh_cache_indices()

    def _refresh_sconv_metadata(
        self, forward_batch: ForwardBatch, *, on_graph_path: bool
    ):
        if self._cache_indices is None:
            return
        self.sconv_metadata.cache_indices = self._cache_indices
        mode = forward_batch.forward_mode
        if mode.is_decode_or_idle():
            self._refresh_decode_metadata(forward_batch, on_graph_path)
        elif mode.is_target_verify():
            self._refresh_extend_metadata(forward_batch, on_graph_path)
        elif mode.is_extend(include_draft_extend_v2=True):
            self._refresh_extend_metadata(forward_batch, on_graph_path)
            self._refresh_track_conv_indices(forward_batch, on_graph_path)
        else:
            raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode=}")

    def _refresh_decode_metadata(
        self, forward_batch: ForwardBatch, on_graph_path: bool
    ):
        B = forward_batch.batch_size
        md = self.sconv_metadata
        (
            md.query_start_loc,
            md.has_initial_state,
            md.precomputed,
        ) = fused_decode_sconv_metadata(
            B=B,
            cache_indices=self._cache_indices,
            out=self._graph_metadata_out(B=B, T=B) if on_graph_path else None,
        )

    def _refresh_extend_metadata(
        self, forward_batch: ForwardBatch, on_graph_path: bool
    ):

        B = forward_batch.batch_size
        if forward_batch.forward_mode.is_target_verify():


            draft_token_num = forward_batch.spec_info.draft_token_num
            T = B * draft_token_num
            his_kwargs = dict(his_mode=HIS_ONES, draft_token_num=draft_token_num)
        else:
            T = forward_batch.extend_num_tokens
            spec_info = forward_batch.spec_info
            if (
                isinstance(spec_info, EagleDraftExtendInput)
                and spec_info.num_front_tokens > 0
            ):

                his_mode, his_src = HIS_ZEROS, None
            elif forward_batch.extend_prefix_lens is not None:
                his_mode, his_src = HIS_PREFIX, forward_batch.extend_prefix_lens
            else:

                his_mode, his_src = HIS_SEQ_MINUS_EXT, forward_batch.seq_lens
            his_kwargs = dict(
                his_mode=his_mode,
                extend_seq_lens=forward_batch.extend_seq_lens,
                his_src=his_src,
            )


        fill_T = T
        out = None
        if on_graph_path:
            if not forward_batch.forward_mode.is_target_verify():
                fill_T = self._graph_bufs["si"].shape[0]
            out = self._graph_metadata_out(B=B, T=fill_T)

        fused = fused_extend_sconv_metadata(
            B=B,
            T=fill_T,
            cache_indices=self._cache_indices,
            out=out,
            **his_kwargs,
        )
        if fused is not None:
            query_start_loc, has_initial_state, precomputed = fused
        else:

            assert not on_graph_path, (
                "the fused extend metadata kernel declined a captured shape "
                f"(B={B}); its unfused fallback is not cuda-graph safe"
            )
            query_start_loc, has_initial_state = self._unfused_extend_metadata(
                forward_batch
            )
            precomputed = precompute_helion_extend_metadata(
                B=B,
                T=T,
                W=self.conv_state_len + 1,
                cache_indices=self._cache_indices,
                has_initial_state=has_initial_state,
                query_start_loc=query_start_loc,
            )
        if fill_T != T:

            precomputed = SconvExtendMetadata(
                cache_mask=precomputed["cache_mask"],
                safe_idx=precomputed["safe_idx"],
                cu=precomputed["cu"],
                si=precomputed["si"][:T],
            )
        md = self.sconv_metadata
        md.query_start_loc = query_start_loc
        md.has_initial_state = has_initial_state
        md.precomputed = precomputed

    def _unfused_extend_metadata(self, forward_batch: ForwardBatch):

        device = forward_batch.req_pool_indices.device
        if forward_batch.forward_mode.is_target_verify():
            draft_token_num = forward_batch.spec_info.draft_token_num
            query_start_loc = torch.arange(
                0,
                (forward_batch.batch_size + 1) * draft_token_num,
                draft_token_num,
                dtype=torch.int32,
                device=device,
            )
            has_initial_state = torch.ones(
                forward_batch.batch_size, dtype=torch.bool, device=device
            )
            return query_start_loc, has_initial_state
        query_start_loc = torch.zeros(
            forward_batch.batch_size + 1,
            dtype=torch.int32,
            device=device,
        )
        query_start_loc[1:] = forward_batch.extend_seq_lens.cumsum(dim=0)
        spec_info = forward_batch.spec_info
        if (
            isinstance(spec_info, EagleDraftExtendInput)
            and spec_info.num_front_tokens > 0
        ):
            has_initial_state = torch.zeros(
                forward_batch.batch_size, dtype=torch.bool, device=device
            )
        elif forward_batch.extend_prefix_lens is not None:
            has_initial_state = forward_batch.extend_prefix_lens > 0
        else:
            has_initial_state = (
                forward_batch.seq_lens[: forward_batch.batch_size]
                - forward_batch.extend_seq_lens
            ) > 0
        return query_start_loc, has_initial_state

    def _refresh_track_conv_indices(
        self, forward_batch: ForwardBatch, on_graph_path: bool
    ):

        if forward_batch.mamba_track_mask is None:
            if not on_graph_path:
                self.sconv_metadata.track_conv_indices = None
                return


            rows = forward_batch.batch_size
            forward_batch.mamba_track_mask = self._graph_track_inert_mask[:rows]
            forward_batch.mamba_track_indices = self._graph_track_inert_indices[:rows]
            forward_batch.mamba_track_seqlens = self._graph_track_inert_seqlens[:rows]
        rows = forward_batch.batch_size
        query_start_loc = self.sconv_metadata.query_start_loc


        prefix_lens = forward_batch.extend_prefix_lens
        if prefix_lens is None and on_graph_path:
            prefix_lens = torch.zeros_like(forward_batch.mamba_track_seqlens)
        live = min(
            rows,
            forward_batch.mamba_track_seqlens.shape[0],
            prefix_lens.shape[0],
        )

        lens_to_track = forward_batch.mamba_track_seqlens[:live] - prefix_lens[:live]
        chunk_aligned = (
            lens_to_track // self.mamba_cache_chunk_size
        ) * self.mamba_cache_chunk_size
        start_indices = query_start_loc[:live] + chunk_aligned - self.conv_state_len

        if on_graph_path:
            assert rows <= self._graph_track_conv_indices.shape[0], (
                f"track-index buffer too small for a captured shape: rows={rows} "
                f"vs bound {self._graph_track_conv_indices.shape[0]}"
            )
            out = self._graph_track_conv_indices[:rows]
        else:
            out = torch.empty(
                (rows, self.conv_state_len),
                dtype=torch.int64,
                device=start_indices.device,
            )
        torch.add(
            start_indices.unsqueeze(-1).to(torch.int64),
            self._track_window_offsets,
            out=out[:live],
        )

        torch.clamp(
            out[:live],
            min=self._track_index_floor,
            max=query_start_loc[-1:].to(torch.int64) - 1,
            out=out[:live],
        )
        if live < rows:
            out[live:].zero_()
        self.sconv_metadata.track_conv_indices = out

    def commit_conv_state_after_mtp_verify(
        self,
        *,
        req_pool_indices: torch.Tensor,
        last_correct_step_indices: torch.Tensor,
        mamba_track_indices: Optional[torch.Tensor],
        mamba_steps_to_track: Optional[torch.Tensor],
    ) -> None:

        pool = self.req_to_token_pool
        scatter_mamba_states_after_mtp_verify(
            pool.get_speculative_mamba2_params_all_layers(),
            self._translate_mamba_indices(pool.get_mamba_indices(req_pool_indices)),
            last_correct_step_indices,
            mamba_track_indices,
            mamba_steps_to_track,
        )

    def conv_state_metadata(
        self, layer_id: int, forward_batch: ForwardBatch
    ) -> InklingShortConvMetadata:

        del layer_id, forward_batch
        return self.sconv_metadata

    def sconv_state(self, *, layer_id: int, stream: int) -> torch.Tensor:

        pool_layer = self.req_to_token_pool.mamba2_layer_index(layer_id)
        return self._mamba_cache.conv[stream][pool_layer]

    def sconv_intermediate_window(self, *, layer_id: int, stream: int) -> torch.Tensor:

        pool_layer = self.req_to_token_pool.mamba2_layer_index(layer_id)
        return self._mamba_cache.intermediate_conv_window[stream][pool_layer]


class InklingShortConvHybridAttnBackend(ShortConvHybridAttnBackend):


    def sconv_state(self, *, layer_id: int, stream: int) -> torch.Tensor:
        return self.short_conv_backend.sconv_state(layer_id=layer_id, stream=stream)

    def sconv_intermediate_window(self, *, layer_id: int, stream: int) -> torch.Tensor:
        return self.short_conv_backend.sconv_intermediate_window(
            layer_id=layer_id, stream=stream
        )

    def _is_full_attn(self, layer=None, layer_id: Optional[int] = None) -> bool:
        del layer, layer_id
        return True

    def update_mamba_state_after_mtp_verify(
        self,
        last_correct_step_indices: torch.Tensor,
        mamba_track_indices: Optional[torch.Tensor],
        mamba_steps_to_track: Optional[torch.Tensor],
        model=None,
        req_pool_indices: Optional[torch.Tensor] = None,
    ):

        del model
        assert req_pool_indices is not None, (
            "Inkling's conv-state commit needs req_pool_indices; the caller must "
            "pass the verify batch's request slots."
        )
        self.short_conv_backend.commit_conv_state_after_mtp_verify(
            req_pool_indices=req_pool_indices,
            last_correct_step_indices=last_correct_step_indices,
            mamba_track_indices=mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata(forward_batch)

    @property
    def forward_metadata(self):


        return self.full_attn_backend.forward_metadata

    @forward_metadata.setter
    def forward_metadata(self, value):
        self.full_attn_backend.forward_metadata = value

    @property
    def supports_ragged_verify_graph(self) -> bool:
        return self.full_attn_backend.supports_ragged_verify_graph

    @property
    def supports_full_cuda_graph_chunked_prefix(self) -> bool:
        return self.full_attn_backend.supports_full_cuda_graph_chunked_prefix

    def prepare_full_cuda_graph_chunked_prefix(self, *args, **kwargs):
        return self.full_attn_backend.prepare_full_cuda_graph_chunked_prefix(
            *args, **kwargs
        )

    def draft_extend_metadata_captured_in_graph(self) -> bool:
        return self.full_attn_backend.draft_extend_metadata_captured_in_graph()
