from __future__ import annotations

from abc import ABC
from enum import Enum
from typing import TYPE_CHECKING, Iterable, Optional

import torch

from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.srt.utils.common import is_npu

if TYPE_CHECKING:
    from sglang.srt.layers.attention.dsa.dsa_indexer_metadata import (
        BaseIndexerMetadata,
    )
    from sglang.srt.layers.attention.verify_mask import VerifyMask
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.speculative.spec_info import SpecInput


class SharedReadEnds(Enum):


    PRE_REPLAY = 1
    IN_REPLAY = 2
    POST_REPLAY = 3
    UNKNOWN = 4

    @staticmethod
    def max_of(items: Iterable[SharedReadEnds]) -> SharedReadEnds:

        return max(items, key=lambda x: x.value)


class AttentionBackend(ABC):


    prefill_attention_backend_str: Optional[str] = None
    decode_attention_backend_str: Optional[str] = None

    supports_ragged_verify_graph: bool = False

    def init_forward_metadata(self, forward_batch: ForwardBatch):

        self.init_forward_metadata_out_graph(forward_batch)
        self.init_forward_metadata_in_graph(forward_batch)

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        pass

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        pass

    def draft_extend_metadata_captured_in_graph(self) -> bool:

        return False


    needs_cpu_seq_lens: bool = True


    extend_dummy_seqs_capped_by_req_pool: bool = False


    use_captured_forward_metadata_for_breakable_cuda_graph: bool = False

    def shared_read_ends(self, fm: ForwardMode) -> SharedReadEnds:

        if fm.is_decode() or fm.is_target_verify():
            return SharedReadEnds.IN_REPLAY
        return SharedReadEnds.UNKNOWN


    supports_full_cuda_graph_chunked_prefix: bool = False

    def prepare_full_cuda_graph_chunked_prefix(
        self,
        forward_batch: ForwardBatch,
        *,
        in_capture: bool,
    ) -> None:
        pass

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):

        raise NotImplementedError()

    def init_forward_metadata_for_breakable_cuda_graph_capture(
        self,
        forward_batch: ForwardBatch,
    ):

        raise NotImplementedError()

    def prepare_forward_metadata_for_breakable_cuda_graph_replay(
        self,
        capture_metadata,
        forward_batch: ForwardBatch,
        *,
        static_forward_batch: Optional[ForwardBatch] = None,
    ) -> None:

        raise NotImplementedError()

    def get_cuda_graph_seq_len_fill_value(self):

        raise NotImplementedError()

    def on_after_cuda_graph_warmup(self):

        pass

    @property
    def verify_mask(self) -> Optional[VerifyMask]:

        return None

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):

        raise NotImplementedError()

    @debug_kernel_api
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):

        if forward_batch.forward_mode.is_idle():
            return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
        elif forward_batch.forward_mode.is_decode():
            return self.forward_decode(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )
        elif forward_batch.forward_mode.is_mixed() and is_npu():
            return self.forward_mixed(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )
        else:
            return self.forward_extend(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):

        raise NotImplementedError()

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):

        raise NotImplementedError()

    def forward_mixed(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):

        raise NotImplementedError()

    def support_triton(self):

        return True

    def get_indexer_metadata(
        self,
        layer_id: int,
        forward_batch: ForwardBatch,
    ) -> Optional[BaseIndexerMetadata]:

        return None
