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

from typing import TYPE_CHECKING, Any, List, NamedTuple, Optional

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    MambaAttnBackendBase,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


class ShortConvMetadata(NamedTuple):


    layer_cache: Any
    cache_indices: torch.Tensor

    query_start_loc: Optional[torch.Tensor] = None

    has_initial_state: Optional[torch.Tensor] = None

    slot_ids_cpu: Optional[List[int]] = None

    has_prefix_cpu: Optional[List[bool]] = None


class ShortConvAttnBackend(MambaAttnBackendBase):


    needs_cpu_seq_lens: bool = False


    cache_indices_dtype: torch.dtype = torch.int64


    needs_extend_host_mirrors: bool = True

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        mamba_cache = self.req_to_token_pool.mamba_pool.mamba_cache

        self.conv_states_shape = mamba_cache.conv[0].shape


        self._has_initial_state: Optional[torch.Tensor] = None
        self._slot_ids_cpu: Optional[List[int]] = None
        self._has_prefix_cpu: Optional[List[bool]] = None
        self._cache_indices: Optional[torch.Tensor] = None
        self._cache_indices_buf: Optional[torch.Tensor] = None

    def _reset_step_state(self):
        self._has_initial_state = None
        self._slot_ids_cpu = None
        self._has_prefix_cpu = None

    def _alloc_cache_indices_buf(self, max_bs: int):


        buf = self._cache_indices_buf
        if buf is not None and buf.shape[0] >= max_bs:
            return
        assert buf is None, (
            f"cache-indices buffer must be sized before any graph capture: have "
            f"{buf.shape[0]}, need {max_bs}"
        )
        self._cache_indices_buf = torch.empty(
            max_bs, dtype=self.cache_indices_dtype, device=self.device
        )

    def _refresh_cache_indices(self):


        md = self.forward_metadata
        idx = md.mamba_cache_indices if md is not None else None
        buf = self._cache_indices_buf
        if idx is None:
            self._cache_indices = None
        elif buf is not None and idx.shape[0] <= buf.shape[0]:
            n = idx.shape[0]
            buf[:n].copy_(idx)
            self._cache_indices = buf[:n]
        else:
            self._cache_indices = idx.to(self.cache_indices_dtype)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        super().init_cuda_graph_state(max_bs, max_num_tokens)
        self._alloc_cache_indices_buf(max_bs)

    def init_cpu_graph_state(self, max_bs: int, max_num_tokens: int):
        super().init_cpu_graph_state(max_bs, max_num_tokens)
        self._alloc_cache_indices_buf(max_bs)

    def init_forward_metadata(self, forward_batch: ForwardBatch):


        super().init_forward_metadata(forward_batch)
        self._reset_step_state()
        self._refresh_cache_indices()
        mode = forward_batch.forward_mode
        if (
            mode.is_extend()
            and not mode.is_target_verify()
            and not mode.is_draft_extend_v2()
        ):
            self._has_initial_state = forward_batch.extend_prefix_lens > 0
            if self.needs_extend_host_mirrors and self._cache_indices is not None:
                self._slot_ids_cpu = self._cache_indices.tolist()
                self._has_prefix_cpu = [
                    int(p) > 0 for p in forward_batch.extend_prefix_lens_cpu
                ]

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):

        super().init_forward_metadata_out_graph(forward_batch, in_capture)
        self._reset_step_state()
        self._refresh_cache_indices()

    def init_forward_metadata_capture_cpu_graph(self, *args, **kwargs):


        super().init_forward_metadata_capture_cpu_graph(*args, **kwargs)
        self._reset_step_state()
        self._refresh_cache_indices()

    def conv_state_metadata(
        self, layer_id: int, forward_batch: ForwardBatch
    ) -> ShortConvMetadata:

        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        md = self.forward_metadata


        return ShortConvMetadata(
            layer_cache=layer_cache,
            cache_indices=self._cache_indices,
            query_start_loc=md.query_start_loc,
            has_initial_state=self._has_initial_state,
            slot_ids_cpu=self._slot_ids_cpu,
            has_prefix_cpu=self._has_prefix_cpu,
        )


    def forward_decode(self, *args, **kwargs):
        raise NotImplementedError(
            "ShortConvAttnBackend is invoked via conv_state_metadata; "
            "it does not run through forward_decode."
        )

    def forward_extend(self, *args, **kwargs):
        raise NotImplementedError(
            "ShortConvAttnBackend is invoked via conv_state_metadata; "
            "it does not run through forward_extend."
        )
