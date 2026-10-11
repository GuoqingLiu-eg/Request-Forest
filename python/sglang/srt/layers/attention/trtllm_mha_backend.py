from __future__ import annotations

"""
Support attention backend for TRTLLM MHA kernels from flashinfer.

Prefill dispatch:
  - SM90 / SM120: trtllm_fmha_v2_prefill (Q_PAGED_KV_NHD layout)
  - SM100:        trtllm_batch_context_with_kv_cache (HND layout)

Decode: uses XQA on SM90 and SM120, TRTLLM-GEN on SM100.

Sliding window and attention sink features are supported.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.attention.utils import canonicalize_stride
from sglang.kernels.ops.kvcache.trtllm_mha_graph_metadata import (
    Q_MODE_NONE,
    Q_MODE_STRIDED,
    update_trtllm_mha_graph_metadata,
)
from sglang.kernels.ops.kvcache.trtllm_mha_page_table import (
    build_trtllm_mha_page_table,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import SharedReadEnds
from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferAttnBackend,
    FlashInferMultiStepDraftBackend,
)
from sglang.srt.layers.attention.trtllm_mla_backend import (
    make_persistent_multi_ctas_kv_counter_buffer,
)
from sglang.srt.layers.cp.base import CPAttentionBackendKind, get_cp_strategy
from sglang.srt.layers.cp.utils import is_cp_v2_active
from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
    KVCacheAttentionAccessKind,
)
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import get_buffer, get_spec
from sglang.srt.speculative.ragged_verify import (
    build_ragged_target_verify_geometry,
    resolve_ragged_verify_layout,
)
from sglang.srt.utils import is_flashinfer_available
from sglang.srt.utils.common import is_sm90_supported, is_sm120_supported

logger = logging.getLogger(__name__)

if is_flashinfer_available():
    import flashinfer

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
    from sglang.srt.speculative.spec_info import SpecInput


DEFAULT_WORKSPACE_SIZE_MB = 512


@dataclass
class TRTLLMMHAMetadata:

    cache_seqlens_int32: torch.Tensor = None

    max_seq_len_q: int = 1

    cu_seqlens_q: torch.Tensor = None

    cu_seqlens_k: torch.Tensor = None

    page_table: torch.Tensor = None

    swa_page_table: torch.Tensor = None

    zigzag_page_table: torch.Tensor = None
    zigzag_swa_page_table: torch.Tensor = None

    swa_out_cache_loc: torch.Tensor = None
    is_ragged_verify: bool = False


    encoder_cache_seqlens: torch.Tensor = None
    encoder_page_table: torch.Tensor = None
    encoder_row_map: torch.Tensor = None


class TRTLLMHAAttnBackend(FlashInferAttnBackend):


    needs_cpu_seq_lens: bool = False

    supports_ragged_verify_graph: bool = True

    def shared_read_ends(self, fm: ForwardMode) -> SharedReadEnds:

        if fm == ForwardMode.EXTEND:
            return SharedReadEnds.PRE_REPLAY
        return super().shared_read_ends(fm)

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        kv_last_page_len_buf: Optional[torch.Tensor] = None,
        speculative_step_id: int = 0,
    ):


        env_var = envs.SGLANG_FLASHINFER_WORKSPACE_SIZE
        workspace_size_bytes = (
            env_var.get()
            if env_var.is_set()
            else DEFAULT_WORKSPACE_SIZE_MB * 1024 * 1024
        )

        super().__init__(
            model_runner, skip_prefill, kv_indptr_buf, kv_last_page_len_buf
        )
        self.decode_kv_access = self.kv_cache_quant_method.resolve_attention_access(
            "decode", "trtllm_mha"
        )
        self._check_decode_kv_access()
        self.decode_uses_native_fp4 = (
            self.decode_kv_access.kind == KVCacheAttentionAccessKind.NATIVE_FP4
        )
        self.is_nvfp4_kvcache = (
            self.decode_uses_native_fp4
            and self.decode_kv_access.scale_recipe == "nvfp4"
        )

        config = model_runner.model_config


        self.max_context_len = model_runner.model_config.context_len
        self.hidden_size = config.hidden_size


        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.page_size = model_runner.page_size
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.device = model_runner.device


        self.workspace_size = workspace_size_bytes

        self.workspace_buffer = get_buffer(
            "trtllm_mha_zero_workspace",
            lambda: torch.zeros(
                self.workspace_size,
                dtype=torch.uint8,
                device=model_runner.device,
            ),
        )


        self.decode_cuda_graph_metadata = {}


        self.topk = get_spec().speculative_eagle_topk or 0
        self.speculative_step_id = speculative_step_id
        self.target_verify_metadata = {}

        self.speculative_num_draft_tokens = get_spec().speculative_num_draft_tokens


        self.expand_encoder_only_verify = any(
            getattr(module, "attn_type", None) == AttentionType.ENCODER_ONLY
            for module in model_runner.model.modules()
        )


        self._swa_kv_pool: Optional[SWAKVPool] = self._resolve_swa_kv_pool(model_runner)


        if self._swa_kv_pool is not None:
            self._swa_full_to_swa_mapping = self._swa_kv_pool.full_to_swa_index_mapping
            assert self._swa_full_to_swa_mapping is not None, (
                "SWA pool must register full_to_swa_index_mapping before "
                "TRTLLMHAAttnBackend is constructed"
            )
        else:
            self._swa_full_to_swa_mapping = None


        self.max_num_pages = (
            self.max_context_len + self.page_size - 1
        ) // self.page_size


        self.forward_metadata: Optional[TRTLLMMHAMetadata] = None


        self.is_xqa_impl = is_sm90_supported() or is_sm120_supported()


        self.use_fmha_v2 = is_sm90_supported() or is_sm120_supported()


        if self.page_size >= 128 and not self.is_xqa_impl:
            from sglang.srt.runtime_context import get_parallel

            attn_tp_size = get_parallel().attn_tp_size
            num_q_heads = config.num_attention_heads // attn_tp_size
            num_kv_heads = config.get_num_kv_heads(attn_tp_size)
            if (
                num_q_heads // num_kv_heads <= 1
                or config.head_dim != config.v_head_dim
                or self.page_size & (self.page_size - 1) != 0
            ):
                raise ValueError(
                    f"trtllm_mha with page_size={self.page_size} requires "
                    f"trtllm-gen's dynamic tokens-per-page kernels, which only "
                    f"support GQA (q heads per kv head > 1, got "
                    f"{num_q_heads}/{num_kv_heads}) with equal QK/V head dims "
                    f"(got {config.head_dim}/{config.v_head_dim}) and a "
                    f"power-of-2 page size. Use --page-size 64 instead."
                )


        self._multi_ctas_kv_counter_buffer = (
            make_persistent_multi_ctas_kv_counter_buffer(
                torch.device(self.device),
                num_q_heads=config.num_attention_heads,
                max_batch_size=(model_runner.max_running_requests + 1)
                * max(1, self.speculative_num_draft_tokens or 1),
            )
        )


        self._default_kv_scale = get_buffer(
            "trtllm_mha_default_kv_scale",
            lambda: torch.ones(1, dtype=torch.float32, device=self.device),
        )
        self.decode_seq_len_splits = envs.SGLANG_TRTLLM_MHA_DECODE_SEQ_LEN_SPLITS.get()
        if self.decode_seq_len_splits < 1:
            raise ValueError(
                "SGLANG_TRTLLM_MHA_DECODE_SEQ_LEN_SPLITS must be at least 1, "
                f"got {self.decode_seq_len_splits}"
            )

    def _check_decode_kv_access(self) -> None:
        supported_kinds = {
            KVCacheAttentionAccessKind.PLAIN,
            KVCacheAttentionAccessKind.NATIVE_FP4,
        }
        if (
            self.decode_kv_access is not None
            and self.decode_kv_access.kind in supported_kinds
        ):
            return

        method_name = getattr(self.kv_cache_quant_method, "name", "unknown")
        available = self.kv_cache_quant_method.describe_attention_accesses("decode")
        raise ValueError(
            f"KV cache method {method_name!r} does not support decode with "
            f"trtllm_mha. Available decode accesses: {available}."
        )

    @staticmethod
    def _resolve_swa_kv_pool(model_runner: ModelRunner) -> Optional[SWAKVPool]:

        active_pool = model_runner.token_to_kv_pool
        if isinstance(active_pool, SWAKVPool):
            return active_pool

        if model_runner.is_draft_worker:
            if not model_runner.spec_algorithm.is_frozen_kv_mtp():
                return None

        allocator = model_runner.token_to_kv_pool_allocator
        kvcache = allocator.get_kvcache()
        return kvcache if isinstance(kvcache, SWAKVPool) else None

    def _alloc_swa_page_table(
        self, max_bs: int, max_num_pages: int
    ) -> Optional[torch.Tensor]:

        if self._swa_kv_pool is None:
            return None
        return torch.zeros(max_bs, max_num_pages, dtype=torch.int32, device=self.device)

    def _fill_page_table_device(
        self,
        metadata: TRTLLMMHAMetadata,
        req_pool_indices: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ):

        has_swa = self._swa_kv_pool is not None
        build_trtllm_mha_page_table(
            req_to_token=self.req_to_token,
            req_pool_indices=req_pool_indices,
            cache_seqlens=cache_seqlens,
            page_table=metadata.page_table,
            page_size=self.page_size,
            swa_page_table=metadata.swa_page_table if has_swa else None,
            full_to_swa=(
                self._swa_kv_pool.full_to_swa_index_mapping if has_swa else None
            ),
        )

    def _get_layer_cache_loc(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:

        if self._swa_kv_pool is not None:
            _, is_swa = self._swa_kv_pool.layers_mapping[layer.layer_id]
            if is_swa:
                swa_loc = self.forward_metadata.swa_out_cache_loc
                assert (
                    swa_loc is not None
                    and swa_loc.shape[0] >= forward_batch.out_cache_loc.shape[0]
                ), (
                    "SWA write locs missing or too short: init_forward_metadata "
                    "must translate out_cache_loc once; a per-layer gather of "
                    "the live mapping would race the scheduler after the WAR "
                    "fence releases"
                )


                return swa_loc[: forward_batch.out_cache_loc.shape[0]]
        return forward_batch.out_cache_loc

    def _bind_swa_page_table(
        self, metadata: TRTLLMMHAMetadata, source: dict, key: str, bs: int
    ):

        buf = source.get(key)
        if buf is not None:
            metadata.swa_page_table = buf[:bs, :]

    def _get_layer_page_table(
        self, layer: RadixAttention, forward_batch: ForwardBatch
    ) -> torch.Tensor:

        swa_pt = self.forward_metadata.swa_page_table
        if swa_pt is not None:
            _, is_swa = self._swa_kv_pool.layers_mapping[layer.layer_id]
            if is_swa:
                return swa_pt
        return self.forward_metadata.page_table

    def _maybe_build_cp_zigzag_page_tables(
        self,
        metadata: TRTLLMMHAMetadata,
        forward_batch: ForwardBatch,
    ) -> None:

        if not is_cp_v2_active(forward_batch):
            return


        metadata.zigzag_page_table = torch.cat(
            (metadata.page_table, metadata.page_table), dim=0
        )
        if metadata.swa_page_table is not None:
            metadata.zigzag_swa_page_table = torch.cat(
                (metadata.swa_page_table, metadata.swa_page_table), dim=0
            )

    @staticmethod
    def _get_scalar_scale(
        layer: RadixAttention,
        float_attr: str,
        scale_attr: str,
    ) -> float:
        scale = getattr(layer, float_attr, None)
        if scale is None:
            scale = getattr(layer, scale_attr, None)
        if scale is None:
            return 1.0
        if isinstance(scale, torch.Tensor):
            logger.warning_once(
                "Ignoring tensor %s for TRT-LLM MHA FP8 KV cache scale. "
                "Expected %s to be populated with a Python scalar.",
                scale_attr,
                float_attr,
            )
            return 1.0
        scale = float(scale)
        return scale if scale > 0.0 else 1.0

    def _get_bmm_scales(
        self, layer: RadixAttention, q_scale: float | torch.Tensor = 1.0
    ) -> tuple[float | torch.Tensor, float]:

        if self.data_type != torch.float8_e4m3fn:
            return layer.scaling, 1.0

        k_scale = self._get_scalar_scale(layer, "k_scale_float", "k_scale")
        v_scale = self._get_scalar_scale(layer, "v_scale_float", "v_scale")
        return q_scale * k_scale * layer.scaling, v_scale

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):

        max_num_pages = self.max_num_pages
        self.decode_cuda_graph_metadata = {
            "cache_seqlens": torch.zeros(max_bs, dtype=torch.int32, device=self.device),
            "page_table": torch.zeros(
                max_bs,
                max_num_pages,
                dtype=torch.int32,
                device=self.device,
            ),
            "swa_page_table": self._alloc_swa_page_table(max_bs, max_num_pages),
        }


        self.cuda_graph_swa_out_cache_loc = (
            torch.zeros(max_num_tokens, dtype=torch.int64, device=self.device)
            if self.use_sliding_window_kv_pool
            else None
        )

        if (
            self.speculative_num_draft_tokens is not None
            and self.speculative_num_draft_tokens > 0
        ):
            self.decode_cuda_graph_metadata["cu_seqlens_q"] = torch.arange(
                0, max_bs + 1, dtype=torch.int32, device=self.device
            )
            self.decode_cuda_graph_metadata["cu_seqlens_k"] = torch.zeros(
                max_bs + 1, dtype=torch.int32, device=self.device
            )
            self.decode_cuda_graph_metadata["page_table_draft_decode"] = torch.zeros(
                max_bs,
                max_num_pages,
                dtype=torch.int32,
                device=self.device,
            )
            self.decode_cuda_graph_metadata["swa_page_table_draft_decode"] = (
                self._alloc_swa_page_table(max_bs, max_num_pages)
            )

            self.target_verify_metadata = {
                "cache_seqlens": torch.zeros(
                    max_bs, dtype=torch.int32, device=self.device
                ),


                "cu_seqlens_q": torch.arange(
                    0,
                    max_bs * self.speculative_num_draft_tokens + 1,
                    step=self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_k": torch.zeros(
                    max_bs + 1, dtype=torch.int32, device=self.device
                ),
                "page_table": torch.zeros(
                    max_bs,
                    max_num_pages,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "swa_page_table": self._alloc_swa_page_table(max_bs, max_num_pages),
            }
            if self.expand_encoder_only_verify:
                max_verify_rows = max_bs * self.speculative_num_draft_tokens
                self.target_verify_metadata["encoder_cache_seqlens"] = torch.zeros(
                    max_verify_rows, dtype=torch.int32, device=self.device
                )
                self.target_verify_metadata["encoder_page_table"] = torch.zeros(
                    max_verify_rows,
                    max_num_pages,
                    dtype=torch.int32,
                    device=self.device,
                )

            self.draft_extend_metadata = {
                "cache_seqlens": torch.zeros(
                    max_bs, dtype=torch.int32, device=self.device
                ),
                "cu_seqlens_q": torch.zeros(
                    max_bs + 1,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_k": torch.zeros(
                    max_bs + 1, dtype=torch.int32, device=self.device
                ),
                "page_table": torch.zeros(
                    max_bs,
                    max_num_pages,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "swa_page_table": self._alloc_swa_page_table(max_bs, max_num_pages),
            }

    def _build_cuda_graph_metadata(
        self,
        bs: int,
        num_tokens: int,
        forward_mode: ForwardMode,
        spec_info,
        device: torch.device,
    ) -> TRTLLMMHAMetadata:

        metadata = TRTLLMMHAMetadata()

        if forward_mode.is_decode_or_idle():
            if spec_info is not None:

                metadata.cache_seqlens_int32 = self.decode_cuda_graph_metadata[
                    "cache_seqlens"
                ][:bs]
                metadata.cu_seqlens_q = self.decode_cuda_graph_metadata["cu_seqlens_q"][
                    : bs + 1
                ]
                metadata.cu_seqlens_k = self.decode_cuda_graph_metadata["cu_seqlens_k"][
                    : bs + 1
                ]
                metadata.page_table = self.decode_cuda_graph_metadata[
                    "page_table_draft_decode"
                ][:bs, :]
                self._bind_swa_page_table(
                    metadata,
                    self.decode_cuda_graph_metadata,
                    "swa_page_table_draft_decode",
                    bs,
                )
                self.decode_cuda_graph_metadata[bs] = metadata
            else:

                metadata.cache_seqlens_int32 = self.decode_cuda_graph_metadata[
                    "cache_seqlens"
                ][:bs]
                metadata.cu_seqlens_q = torch.arange(
                    0, bs + 1, dtype=torch.int32, device=device
                )
                metadata.cu_seqlens_k = torch.zeros(
                    bs + 1, dtype=torch.int32, device=device
                )
                metadata.page_table = self.decode_cuda_graph_metadata["page_table"][
                    :bs, :
                ]
                self._bind_swa_page_table(
                    metadata,
                    self.decode_cuda_graph_metadata,
                    "swa_page_table",
                    bs,
                )
                self.decode_cuda_graph_metadata[bs] = metadata
        elif forward_mode.is_target_verify():

            metadata.cache_seqlens_int32 = self.target_verify_metadata["cache_seqlens"][
                :bs
            ]
            metadata.cu_seqlens_q = self.target_verify_metadata["cu_seqlens_q"][
                : bs + 1
            ]
            metadata.cu_seqlens_k = self.target_verify_metadata["cu_seqlens_k"][
                : bs + 1
            ]
            metadata.is_ragged_verify = (
                spec_info is not None and spec_info.ragged_verify_layout is not None
            )
            metadata.max_seq_len_q = (
                self.speculative_num_draft_tokens
                if metadata.is_ragged_verify
                else num_tokens // bs
            )
            metadata.page_table = self.target_verify_metadata["page_table"][:bs, :]
            self._bind_swa_page_table(
                metadata,
                self.target_verify_metadata,
                "swa_page_table",
                bs,
            )
            if self._needs_encoder_only_expand(forward_mode, metadata):
                verify_rows = bs * metadata.max_seq_len_q


                metadata.encoder_row_map = (
                    torch.arange(verify_rows, device=self.device)
                    // metadata.max_seq_len_q
                )
                metadata.encoder_cache_seqlens = self.target_verify_metadata[
                    "encoder_cache_seqlens"
                ][:verify_rows]
                metadata.encoder_page_table = self.target_verify_metadata[
                    "encoder_page_table"
                ][:verify_rows, :]
            self.target_verify_metadata[bs] = metadata
        elif forward_mode.is_draft_extend_v2():
            num_tokens_per_req = spec_info.num_tokens_per_req
            metadata.cache_seqlens_int32 = self.draft_extend_metadata["cache_seqlens"][
                :bs
            ]
            metadata.cu_seqlens_q = self.draft_extend_metadata["cu_seqlens_q"][: bs + 1]
            metadata.cu_seqlens_k = self.draft_extend_metadata["cu_seqlens_k"][: bs + 1]
            metadata.max_seq_len_q = num_tokens_per_req
            metadata.page_table = self.draft_extend_metadata["page_table"][:bs, :]
            self._bind_swa_page_table(
                metadata,
                self.draft_extend_metadata,
                "swa_page_table",
                bs,
            )
            self.draft_extend_metadata[bs] = metadata


        if self.use_sliding_window_kv_pool:
            metadata.swa_out_cache_loc = self.cuda_graph_swa_out_cache_loc[:num_tokens]

        return metadata

    def _needs_encoder_only_expand(
        self, forward_mode: ForwardMode, metadata: TRTLLMMHAMetadata
    ) -> bool:


        return (
            self.expand_encoder_only_verify
            and forward_mode.is_target_verify()
            and not metadata.is_ragged_verify
        )

    def _apply_cuda_graph_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        out_cache_loc: Optional[torch.Tensor] = None,
    ):

        seq_lens = seq_lens[:bs]
        req_pool_indices = req_pool_indices[:bs]

        metadata = None
        seqlen_offset = 0
        cu_seqlens_q = None
        qlens = None
        q_stride = 0
        q_mode = Q_MODE_NONE
        if forward_mode.is_decode_or_idle():
            if spec_info is not None:


                metadata = self.decode_cuda_graph_metadata[bs]
                seqlen_offset = self.speculative_step_id + 1
            else:

                metadata = self.decode_cuda_graph_metadata[bs]
        elif forward_mode.is_target_verify():

            metadata = self.target_verify_metadata[bs]
            if spec_info is not None and spec_info.ragged_verify_layout is not None:


                return
            seqlen_offset = metadata.max_seq_len_q
        elif forward_mode.is_draft_extend_v2():
            metadata = self.draft_extend_metadata[bs]


            num_tokens_per_req = metadata.max_seq_len_q
            cu_seqlens_q = metadata.cu_seqlens_q
            q_stride = num_tokens_per_req
            q_mode = Q_MODE_STRIDED
        else:
            raise ValueError(
                "TRTLLM-MHA CUDA graph metadata build got an unsupported forward "
                f"mode: {forward_mode}"
            )

        assert metadata is not None


        max_seq_pages = self.max_num_pages
        update_trtllm_mha_graph_metadata(
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            req_to_token=self.req_to_token,
            cache_seqlens=metadata.cache_seqlens_int32,
            cu_seqlens_k=metadata.cu_seqlens_k,
            page_table=metadata.page_table,
            bs=bs,
            seqlen_offset=seqlen_offset,
            max_seq_pages=max_seq_pages,
            page_size=self.page_size,
            swa_mapping=self._swa_full_to_swa_mapping,
            swa_page_table=metadata.swa_page_table,
            out_cache_loc=out_cache_loc,
            swa_out_cache_loc=metadata.swa_out_cache_loc,
            cu_seqlens_q=cu_seqlens_q,
            qlens=qlens,
            q_stride=q_stride,
            q_mode=q_mode,
        )

        if self._needs_encoder_only_expand(forward_mode, metadata):


            metadata.encoder_cache_seqlens.copy_(
                metadata.cache_seqlens_int32[metadata.encoder_row_map]
            )
            metadata.encoder_page_table.copy_(
                metadata.page_table[metadata.encoder_row_map]
            )

        self.forward_metadata = metadata

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        pass

    def get_cuda_graph_seq_len_fill_value(self) -> int:

        return 1

    def _should_use_fused_fp8_path(
        self, save_kv_cache: bool, k: torch.Tensor, forward_batch: ForwardBatch
    ) -> bool:

        return (
            not is_cp_v2_active(forward_batch)
            and save_kv_cache
            and k is not None
            and self.data_type == torch.float8_e4m3fn
        )

    def _fused_fp8_qkv_kv_cache(
        self,
        q: torch.Tensor | None,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor | None:
        from sglang.kernels.ops.kvcache.fused_fp8_qkv_kv_cache import (
            fused_fp8_qkv_kv_cache,
        )

        cache_loc = self._get_layer_cache_loc(layer, forward_batch)
        k_cache, v_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)


        return fused_fp8_qkv_kv_cache(
            q=q,
            k=k,
            v=v,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_loc=cache_loc,
            k_scale=(
                layer.k_scale if layer.k_scale is not None else self._default_kv_scale
            ),
            v_scale=(
                layer.v_scale if layer.v_scale is not None else self._default_kv_scale
            ),
        )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        bs = forward_batch.batch_size
        forward_mode = forward_batch.forward_mode
        spec_info = forward_batch.spec_info

        if (
            forward_mode.is_target_verify()
            and resolve_ragged_verify_layout(forward_batch) is not None
        ):
            self._assert_ragged_verify_supported()

        if in_capture:
            num_tokens = forward_batch.positions.numel()
            self._build_cuda_graph_metadata(
                bs, num_tokens, forward_mode, spec_info, forward_batch.seq_lens.device
            )

        if forward_mode.is_decode_or_idle():
            self.forward_metadata = self.decode_cuda_graph_metadata[bs]
        elif forward_mode.is_target_verify():
            self.forward_metadata = self.target_verify_metadata[bs]
            ragged_layout = resolve_ragged_verify_layout(forward_batch)
            if ragged_layout is not None:
                self._write_ragged_verify_graph_metadata(
                    self.forward_metadata, forward_batch, ragged_layout, bs
                )
        elif forward_mode.is_draft_extend_v2():
            self.forward_metadata = self.draft_extend_metadata[bs]
        else:
            raise ValueError(
                f"Invalid forward mode: {forward_mode=} for CUDA Graph replay."
            )

    def _assert_ragged_verify_supported(self) -> None:
        if self.is_xqa_impl:
            raise NotImplementedError(
                "Compact ragged verify (variable-length cum_seq_lens_q) "
                "requires the trtllm-gen decode kernel; the xqa impl (sm90 / sm120) "
                "rejects it. Disable SGLANG_RAGGED_VERIFY_MODE for this configuration."
            )

    def _write_ragged_verify_graph_metadata(
        self,
        metadata: TRTLLMMHAMetadata,
        forward_batch: ForwardBatch,
        ragged_layout: RaggedVerifyLayout,
        bs: int,
    ) -> None:

        seq_lens = forward_batch.seq_lens[:bs]
        req_pool_indices = forward_batch.req_pool_indices[:bs]
        padded_layout = ragged_layout.padded_to_bucket(padded_bs=bs)
        geometry = build_ragged_target_verify_geometry(
            seq_lens=seq_lens, layout=padded_layout
        )
        metadata.cache_seqlens_int32.copy_(geometry.cache_seqlens_int32)
        metadata.cu_seqlens_q.copy_(geometry.cu_seqlens_q)
        metadata.cu_seqlens_k[1:].copy_(
            torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32)
        )
        self._fill_page_table_device(
            metadata, req_pool_indices, metadata.cache_seqlens_int32
        )


        if self.use_sliding_window_kv_pool and forward_batch.out_cache_loc is not None:
            n = forward_batch.out_cache_loc.shape[0]
            self.cuda_graph_swa_out_cache_loc[n:].zero_()
            self.cuda_graph_swa_out_cache_loc[:n].copy_(
                self.token_to_kv_pool.translate_loc_from_full_to_swa(
                    forward_batch.out_cache_loc
                )
            )

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        self._apply_cuda_graph_metadata(
            bs=forward_batch.batch_size,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            forward_mode=forward_batch.forward_mode,
            spec_info=forward_batch.spec_info,
            out_cache_loc=forward_batch.out_cache_loc,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):


        metadata = TRTLLMMHAMetadata()
        seqlens_in_batch = forward_batch.seq_lens
        batch_size = forward_batch.batch_size
        device = seqlens_in_batch.device

        if forward_batch.forward_mode.is_decode_or_idle():
            if forward_batch.spec_info is not None:


                metadata.cache_seqlens_int32 = (
                    seqlens_in_batch + (self.speculative_step_id + 1)
                ).to(torch.int32)
                metadata.cu_seqlens_q = torch.arange(
                    0, batch_size + 1, dtype=torch.int32, device=device
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(
                        metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
            else:

                metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
                metadata.cu_seqlens_q = torch.arange(
                    0, batch_size + 1, dtype=torch.int32, device=device
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
                )
        elif forward_batch.forward_mode.is_target_verify():
            ragged_layout = resolve_ragged_verify_layout(forward_batch)
            if ragged_layout is not None:
                self._assert_ragged_verify_supported()
                geometry = build_ragged_target_verify_geometry(
                    seq_lens=seqlens_in_batch, layout=ragged_layout
                )
                metadata.cache_seqlens_int32 = geometry.cache_seqlens_int32


                metadata.max_seq_len_q = (
                    geometry.max_seq_len_q
                    if geometry.max_seq_len_q is not None
                    else self.speculative_num_draft_tokens
                )
                metadata.cu_seqlens_q = geometry.cu_seqlens_q
                metadata.cu_seqlens_k = geometry.cu_seqlens_k
                metadata.is_ragged_verify = True
            else:
                tokens_per_req = forward_batch.input_ids.shape[0] // batch_size
                metadata.cache_seqlens_int32 = (
                    forward_batch.seq_lens + tokens_per_req
                ).to(torch.int32)
                metadata.max_seq_len_q = tokens_per_req
                metadata.cu_seqlens_q = torch.arange(
                    0,
                    batch_size * tokens_per_req + 1,
                    tokens_per_req,
                    dtype=torch.int32,
                    device=device,
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(
                        metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )

        else:
            metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
            )


            metadata.max_seq_len_q = int(max(forward_batch.extend_seq_lens_cpu))
            if (
                forward_batch.extend_prefix_lens_cpu is not None
                and any(forward_batch.extend_prefix_lens_cpu)
            ) or forward_batch.forward_mode.is_draft_extend_v2():
                extend_seq_lens = forward_batch.extend_seq_lens
                metadata.cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )
            else:
                metadata.cu_seqlens_q = metadata.cu_seqlens_k

        has_swa = self._swa_kv_pool is not None
        metadata.page_table = torch.empty(
            (batch_size, self.max_num_pages), dtype=torch.int32, device=device
        )
        metadata.swa_page_table = (
            torch.empty(
                (batch_size, self.max_num_pages), dtype=torch.int32, device=device
            )
            if has_swa
            else None
        )
        self._fill_page_table_device(
            metadata, forward_batch.req_pool_indices, metadata.cache_seqlens_int32
        )
        self._maybe_build_cp_zigzag_page_tables(metadata, forward_batch)

        if self._needs_encoder_only_expand(forward_batch.forward_mode, metadata):
            row_map = (
                torch.arange(batch_size * metadata.max_seq_len_q, device=device)
                // metadata.max_seq_len_q
            )
            metadata.encoder_row_map = row_map
            metadata.encoder_cache_seqlens = metadata.cache_seqlens_int32[row_map]
            metadata.encoder_page_table = metadata.page_table[row_map]


        if self.use_sliding_window_kv_pool and forward_batch.out_cache_loc is not None:
            metadata.swa_out_cache_loc = (
                self.token_to_kv_pool.translate_loc_from_full_to_swa(
                    forward_batch.out_cache_loc
                )
            )

        self.forward_metadata = metadata

    def _reshape_paged_kv_cache(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        layer: RadixAttention,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k_cache = k_cache.view(
            -1, self.page_size, layer.tp_k_head_num, head_dim
        ).permute(0, 2, 1, 3)
        v_cache = v_cache.view(
            -1, self.page_size, layer.tp_v_head_num, head_dim
        ).permute(0, 2, 1, 3)
        if layer.tp_k_head_num == 1:
            k_cache = canonicalize_stride(k_cache)
        if layer.tp_v_head_num == 1:
            v_cache = canonicalize_stride(v_cache)
        return k_cache, v_cache

    def _get_nvfp4_bmm_scales(self, layer: RadixAttention) -> tuple[float, float]:
        assert self.is_nvfp4_kvcache
        return self.kv_cache_quant_method.get_bmm_scales(layer.layer_id)

    def _run_fixed_q_len_decode(
        self,
        query: torch.Tensor,
        kv_cache,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        bmm1_scale,
        bmm2_scale,
        window_left: int,
        sinks: Optional[torch.Tensor],
        q_len_per_req: int = 1,
        kv_cache_sf=None,
    ) -> torch.Tensor:


        def run_group(group_query, group_block_tables, group_seq_lens):
            kwargs = {}
            if q_len_per_req != 1:
                kwargs["q_len_per_req"] = q_len_per_req
            return flashinfer.decode.trtllm_batch_decode_with_kv_cache(
                query=group_query,
                kv_cache=kv_cache,
                workspace_buffer=self.workspace_buffer,
                block_tables=group_block_tables,
                seq_lens=group_seq_lens,
                max_seq_len=self.max_context_len,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                window_left=window_left,
                sinks=sinks,
                skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),
                out_dtype=self.q_data_type,
                kv_cache_sf=kv_cache_sf,
                multi_ctas_kv_counter_buffer=self._multi_ctas_kv_counter_buffer,
                **kwargs,
            )

        num_requests = seq_lens.shape[0]
        num_splits = min(self.decode_seq_len_splits, num_requests)
        if num_splits == 1:
            return run_group(query, block_tables, seq_lens)

        order = torch.argsort(seq_lens)
        query_by_request = query.view(
            num_requests, q_len_per_req, query.shape[-2], query.shape[-1]
        )
        output_by_request = torch.empty(
            query_by_request.shape,
            dtype=self.q_data_type,
            device=query.device,
        )
        for indices in torch.tensor_split(order, num_splits):
            group_output = run_group(
                query_by_request.index_select(0, indices).reshape(
                    -1, query.shape[-2], query.shape[-1]
                ),
                block_tables.index_select(0, indices),
                seq_lens.index_select(0, indices),
            )
            output_by_request.index_copy_(
                0,
                indices,
                group_output.view(-1, q_len_per_req, query.shape[-2], query.shape[-1]),
            )
        return output_by_request.view(-1, query.shape[-2], query.shape[-1])

    def _get_nvfp4_decode_kv_cache(self, layer: RadixAttention) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        assert self.is_nvfp4_kvcache
        k_fp4, v_fp4, k_scale, v_scale = self.token_to_kv_pool.get_raw_kv_buffer(
            layer.layer_id
        )
        kv_cache = self._reshape_paged_kv_cache(
            k_fp4, v_fp4, layer, layer.head_dim // 2
        )
        kv_cache_block_scales = self._reshape_paged_kv_cache(
            k_scale, v_scale, layer, layer.head_dim // 16
        )
        return kv_cache, kv_cache_block_scales

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:

        cache_loc = forward_batch.out_cache_loc

        use_fused_fp8_path = self._should_use_fused_fp8_path(
            save_kv_cache, k, forward_batch
        )
        use_fused_qkv = use_fused_fp8_path and not self.is_xqa_impl
        pool = self.token_to_kv_pool

        if use_fused_fp8_path:
            fused_q = self._fused_fp8_qkv_kv_cache(
                q if use_fused_qkv else None, k, v, layer, forward_batch
            )
            if fused_q is not None:
                q = fused_q
            k = None
            v = None
        else:
            if save_kv_cache and k is not None:
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    KVWriteLoc(cache_loc, self.forward_metadata.swa_out_cache_loc),
                    k,
                    v,
                    *self._kv_write_scales(layer),
                )


        q_scale = 1.0
        if (
            self.data_type == torch.float8_e4m3fn
            and not self.is_xqa_impl
            and not use_fused_qkv
        ):
            q = q.to(torch.float8_e4m3fn)
        if self.is_xqa_impl:
            q = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
        else:
            q = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)

        if self.is_nvfp4_kvcache:
            kv_cache, kv_cache_block_scales = self._get_nvfp4_decode_kv_cache(layer)
        else:
            k_cache, v_cache = pool.get_kv_buffer(layer.layer_id)
            kv_cache = self._reshape_paged_kv_cache(
                k_cache, v_cache, layer, layer.head_dim
            )
            kv_cache_block_scales = None

        if self.is_nvfp4_kvcache:
            k_scale, v_scale = self._get_nvfp4_bmm_scales(layer)
            bmm1_scale = q_scale * k_scale * layer.scaling
            bmm2_scale = v_scale
        else:
            bmm1_scale, bmm2_scale = self._get_bmm_scales(layer, q_scale)
        attention_sink = kwargs.get("sinks", None)

        page_table = self._get_layer_page_table(layer, forward_batch)

        o = self._run_fixed_q_len_decode(
            q,
            kv_cache,
            page_table,
            self.forward_metadata.cache_seqlens_int32,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            window_left=layer.sliding_window_size,
            sinks=attention_sink,
            kv_cache_sf=kv_cache_block_scales,
        )
        if self.is_nvfp4_kvcache and o.dtype != self.q_data_type:
            o = o.to(self.q_data_type)

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        **kwargs,
    ):
        if self.decode_uses_native_fp4:
            raise RuntimeError(
                "TRTLLM MHA with native FP4 KV cache supports decode only; "
                "use a separate prefill backend such as flashinfer or triton."
            )

        cache_loc = forward_batch.out_cache_loc
        cp_v2_active = is_cp_v2_active(forward_batch)


        use_fused_fp8_path = self._should_use_fused_fp8_path(
            save_kv_cache, k, forward_batch
        )
        use_fused_qkv = use_fused_fp8_path and not self.is_xqa_impl

        if use_fused_fp8_path:
            fused_q = self._fused_fp8_qkv_kv_cache(
                q if use_fused_qkv else None, k, v, layer, forward_batch
            )
            if fused_q is not None:
                q = fused_q
            k = None
            v = None
        else:
            if save_kv_cache and k is not None:
                if cp_v2_active:
                    cp_strategy = get_cp_strategy()
                    assert cp_strategy is not None
                    cp_strategy.materialize_full_kv(
                        forward_batch,
                        layer,
                        k,
                        v,
                        swa_loc=self.forward_metadata.swa_out_cache_loc,
                    )
                else:
                    self.token_to_kv_pool.set_kv_buffer(
                        layer,
                        KVWriteLoc(cache_loc, self.forward_metadata.swa_out_cache_loc),
                        k,
                        v,
                        layer.k_scale,
                        layer.v_scale,
                    )

        q_scale = 1.0
        if (
            self.data_type == torch.float8_e4m3fn
            and (
                not self.is_xqa_impl
                or not forward_batch.forward_mode.is_target_verify()
            )
            and not use_fused_qkv
        ):
            q = q.to(torch.float8_e4m3fn)

        if self.use_fmha_v2:
            q = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
        else:
            q = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)


        k_cache_raw, v_cache_raw = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)

        is_decode_mode = (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        )

        if not self.use_fmha_v2 or is_decode_mode:

            k_cache, v_cache = self._reshape_paged_kv_cache(
                k_cache_raw, v_cache_raw, layer, layer.head_dim
            )
        else:
            k_cache = k_cache_raw.view(
                -1, self.page_size, layer.tp_k_head_num, layer.head_dim
            )
            v_cache = v_cache_raw.view(
                -1, self.page_size, layer.tp_v_head_num, layer.head_dim
            )

        kv_cache = (k_cache, v_cache)

        attention_sink = kwargs.get("sinks", None)
        bmm1_scale, bmm2_scale = self._get_bmm_scales(layer, q_scale)

        page_table = self._get_layer_page_table(layer, forward_batch)

        if is_decode_mode:
            if (
                forward_batch.forward_mode.is_target_verify()
                and layer.attn_type == AttentionType.ENCODER_ONLY
            ):


                assert not self.forward_metadata.is_ragged_verify, (
                    "ENCODER_ONLY target_verify does not support ragged "
                    "verify layouts"
                )
                assert self.forward_metadata.encoder_cache_seqlens is not None, (
                    "ENCODER_ONLY target_verify requires the expanded decode "
                    "metadata (built only on the draft worker)"
                )
                o = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
                    query=q,
                    kv_cache=kv_cache,
                    workspace_buffer=self.workspace_buffer,
                    block_tables=self.forward_metadata.encoder_page_table,
                    seq_lens=self.forward_metadata.encoder_cache_seqlens,
                    max_seq_len=self.max_context_len,
                    bmm1_scale=bmm1_scale,
                    bmm2_scale=bmm2_scale,
                    window_left=layer.sliding_window_size,
                    sinks=attention_sink,
                    skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),
                    out_dtype=self.q_data_type,
                    q_len_per_req=1,
                    multi_ctas_kv_counter_buffer=self._multi_ctas_kv_counter_buffer,
                )
            elif self.forward_metadata.is_ragged_verify:
                o = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
                    query=q,
                    kv_cache=kv_cache,
                    workspace_buffer=self.workspace_buffer,
                    block_tables=page_table,
                    seq_lens=self.forward_metadata.cache_seqlens_int32,
                    max_seq_len=self.max_context_len,
                    bmm1_scale=bmm1_scale,
                    bmm2_scale=bmm2_scale,
                    window_left=layer.sliding_window_size,
                    sinks=attention_sink,
                    skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),
                    out_dtype=self.q_data_type,
                    q_len_per_req=None,
                    max_q_len=self.forward_metadata.max_seq_len_q,
                    cum_seq_lens_q=self.forward_metadata.cu_seqlens_q,
                    multi_ctas_kv_counter_buffer=self._multi_ctas_kv_counter_buffer,
                )
            else:
                o = self._run_fixed_q_len_decode(
                    q,
                    kv_cache,
                    page_table,
                    self.forward_metadata.cache_seqlens_int32,
                    bmm1_scale=bmm1_scale,
                    bmm2_scale=bmm2_scale,
                    window_left=layer.sliding_window_size,
                    sinks=attention_sink,
                    q_len_per_req=self.forward_metadata.max_seq_len_q,
                )
        elif self.use_fmha_v2 and not cp_v2_active:


            paged_kv = torch.stack([k_cache, v_cache], dim=1)
            o = flashinfer.prefill.trtllm_fmha_v2_prefill(
                (q, paged_kv),
                input_layout="Q_PAGED_KV_NHD",
                workspace_buffer=self.workspace_buffer,
                seq_lens=self.forward_metadata.cache_seqlens_int32,
                max_q_len=self.forward_metadata.max_seq_len_q,
                max_kv_len=self.max_context_len,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                batch_size=forward_batch.batch_size,
                cum_seq_lens_q=self.forward_metadata.cu_seqlens_q,
                cum_seq_lens_kv=self.forward_metadata.cu_seqlens_k,
                block_tables=page_table,
                out_dtype=self.q_data_type,
                mask_mode="causal",
                window_left=layer.sliding_window_size,
                sinks=attention_sink,
                skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_PREFILL_THRESHOLD_SCALE_FACTOR.get()
                or 0.0,
            )
        else:

            def _trtllm_context_attn(
                q_chunk,
                cu_seqlens_q,
                cache_seqlens,
                max_seqlen_q,
                *,
                cu_seqlens_kv,
                use_zigzag_page_table=False,
                out=None,
            ):
                block_tables = page_table
                if use_zigzag_page_table:
                    block_tables = self.forward_metadata.zigzag_page_table
                    zigzag_swa_pt = self.forward_metadata.zigzag_swa_page_table
                    if zigzag_swa_pt is not None:
                        _, is_swa = self._swa_kv_pool.layers_mapping[layer.layer_id]
                        if is_swa:
                            block_tables = zigzag_swa_pt
                return flashinfer.prefill.trtllm_batch_context_with_kv_cache(
                    query=q_chunk,
                    kv_cache=kv_cache,
                    workspace_buffer=self.workspace_buffer,
                    block_tables=block_tables,
                    seq_lens=cache_seqlens,
                    max_q_len=max_seqlen_q,
                    max_kv_len=self.max_context_len,
                    bmm1_scale=bmm1_scale,
                    bmm2_scale=bmm2_scale,
                    batch_size=cu_seqlens_q.shape[0] - 1,
                    cum_seq_lens_q=cu_seqlens_q,
                    cum_seq_lens_kv=cu_seqlens_kv,
                    window_left=layer.sliding_window_size,
                    sinks=attention_sink,
                    skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_PREFILL_THRESHOLD_SCALE_FACTOR.get(),
                    out=out,
                    out_dtype=self.q_data_type,
                )

            if cp_v2_active:
                cp_strategy = get_cp_strategy()
                assert cp_strategy is not None
                o = cp_strategy.run_attention(
                    q,
                    forward_batch,
                    self.device,
                    _trtllm_context_attn,
                    attention_backend=CPAttentionBackendKind.TRTLLM_MHA,
                )
            else:
                out = forward_batch._attn_output
                if out is not None:
                    out = out.view_as(q)
                o = _trtllm_context_attn(
                    q,
                    self.forward_metadata.cu_seqlens_q,
                    self.forward_metadata.cache_seqlens_int32,
                    self.forward_metadata.max_seq_len_q,
                    cu_seqlens_kv=self.forward_metadata.cu_seqlens_k,
                    out=out,
                )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)


class TRTLLMHAAttnMultiStepDraftBackend(FlashInferMultiStepDraftBackend):


    needs_cpu_seq_lens: bool = False

    def __init__(
        self, model_runner: ModelRunner, topk: int, speculative_num_steps: int
    ):
        super().__init__(model_runner, topk, speculative_num_steps)
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i] = TRTLLMHAAttnBackend(
                model_runner,
                skip_prefill=True,
                kv_indptr_buf=self.kv_indptr[i],
                kv_last_page_len_buf=self.kv_last_page_len,
                speculative_step_id=i,
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        from sglang.srt.model_executor.forward_batch_info import build_inner_fb_view

        assert forward_batch.spec_info is not None
        assert forward_batch.spec_info.is_draft_input()


        inner_fb = build_inner_fb_view(
            forward_batch,
            bs=forward_batch.batch_size,
            forward_mode=ForwardMode.DECODE,
            encoder_lens=forward_batch.encoder_lens,
        )
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata_out_graph(
                inner_fb, in_capture=in_capture
            )

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch) -> None:
        from sglang.srt.model_executor.forward_batch_info import build_inner_fb_view

        assert forward_batch.spec_info is not None
        assert forward_batch.spec_info.is_draft_input()

        inner_fb = build_inner_fb_view(
            forward_batch,
            bs=forward_batch.batch_size,
            forward_mode=ForwardMode.DECODE,
            encoder_lens=forward_batch.encoder_lens,
        )
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata_in_graph(inner_fb)
