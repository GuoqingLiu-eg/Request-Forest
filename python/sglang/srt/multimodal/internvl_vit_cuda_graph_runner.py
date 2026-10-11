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

from typing import Dict, Hashable, Tuple

import torch
import torch.nn as nn

from sglang.srt.layers.attention.vision import VisionAttention


class InternViTCudaGraphRunner:


    def __init__(self, encoder: nn.Module) -> None:
        self.encoder = encoder


        self.graphs: Dict[Hashable, torch.cuda.CUDAGraph] = {}
        self.inp: Dict[Hashable, torch.Tensor] = {}
        self.ws: Dict[Hashable, torch.Tensor] = {}
        self.out: Dict[Hashable, torch.Tensor] = {}


        self.cu: Dict[Hashable, torch.Tensor] = {}
        self.cu_kk: Dict[Hashable, torch.Tensor] = {}


        first_layer = encoder.layers[0]

        self._attn: VisionAttention = first_layer.attn.attn  # type: ignore
        self._attn_backend: str | None = getattr(self._attn, "qkv_backend_name", None)

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.encoder.parameters()).dtype

    def _graph_key(self, x: torch.Tensor) -> Tuple[int, int]:

        return (x.shape[0], x.shape[1])

    def _build_cu(self, B: int, S: int, device: torch.device) -> torch.Tensor:

        return torch.arange(0, (B + 1) * S, step=S, device=device, dtype=torch.int32)

    def _alloc_ws(
        self, B: int, S: int, H: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:

        tokens = B * S

        num_heads = getattr(self._attn, "num_attention_heads_per_partition", None)
        if num_heads is None:
            num_heads = getattr(self._attn, "num_heads", None)
        if num_heads is None:
            raise RuntimeError("Cannot infer num_heads from VisionAttention")

        head_dim = getattr(self._attn, "head_size", None)
        if head_dim is None:

            head_dim = H // int(num_heads)

        return torch.empty(
            tokens,
            int(num_heads),
            int(head_dim),
            device=device,
            dtype=dtype,
        )

    def _warmup_once(self, key: Hashable) -> None:

        backend = self._attn_backend
        cu = self.cu[key]
        cu_kk = self.cu_kk[key]
        max_len = int(cu_kk.max().item()) if cu_kk.numel() else 0

        if backend == "triton_attn":
            cu_ws = [cu, cu_kk, max_len]
        elif backend == "fa3":
            cu_ws = [cu, max_len]
        else:
            raise RuntimeError(
                f"InternVL ViT CUDA graph does not support attention backend: {backend}"
            )

        x = self.inp[key]
        y = x
        with torch.no_grad():
            for blk in self.encoder.layers:
                y = blk(y, cu_seqlens=cu_ws, output_ws=self.ws[key])

    def _capture_graph(self, key: Hashable) -> None:
        g = torch.cuda.CUDAGraph()
        backend = self._attn_backend

        cu = self.cu[key]
        cu_kk = self.cu_kk[key]
        max_len = int(cu_kk.max().item()) if cu_kk.numel() else 0

        if backend == "triton_attn":
            cu_ws = [cu, cu_kk, max_len]
        elif backend == "fa3":
            cu_ws = [cu, max_len]
        else:
            raise RuntimeError(
                f"InternVL ViT CUDA graph does not support attention backend: {backend}"
            )

        torch.cuda.synchronize()

        with torch.cuda.graph(g):
            y = self.inp[key]
            for blk in self.encoder.layers:
                y = blk(y, cu_seqlens=cu_ws, output_ws=self.ws[key])

            self.out[key] = y

        self.graphs[key] = g

    def create_graph(self, x: torch.Tensor) -> Hashable:

        x = x.contiguous()
        key = self._graph_key(x)
        if key in self.graphs:
            return key

        B, S, H = x.shape
        device = x.device
        dtype = x.dtype


        self.inp[key] = torch.empty_like(x, device=device).contiguous()


        cu = self._build_cu(B, S, device=device)
        self.cu[key] = cu
        self.cu_kk[key] = cu[1:] - cu[:-1]


        self.ws[key] = self._alloc_ws(B, S, H, device=device, dtype=dtype)

        self.inp[key].copy_(x)
        self._warmup_once(key)


        self._capture_graph(key)
        return key

    def run(self, x: torch.Tensor) -> torch.Tensor:

        x = x.contiguous()
        key = self._graph_key(x)
        if key not in self.graphs:
            self.create_graph(x)


        self.inp[key].copy_(x)


        self.graphs[key].replay()

        return self.out[key]
