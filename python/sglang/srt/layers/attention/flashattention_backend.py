from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
import triton
import triton.language as tl
from sglang.kernels.ops.attention.metadata import (
    draft_extend_set_metadata,
    normal_decode_set_metadata,
    prepare_swa_spec_page_table_triton,
)
from sglang.kernels.ops.attention.pa_page_table import _build_pa_page_table
from sglang.kernels.ops.attention.utils import assert_buffer_fits
from sglang.kernels.ops.kvcache.trtllm_mha_page_table import (
    build_trtllm_mha_page_table,
)
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.unified_mem_hooks import unified_mla_hooks
from sglang.srt.layers.attention.verify_mask import VerifyMask, maybe_create_verify_mask
from sglang.srt.layers.cp.base import CPAttentionBackendKind, get_cp_strategy
from sglang.srt.layers.cp.utils import is_cp_v2_active
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.layers.utils.cp_utils import (
    cp_allgather_and_save_kv_cache,
    cp_attn_forward_extend,
)
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import (
    get_exec,
    get_memory,
    get_model,
    get_parallel,
    get_schedule,
    get_spec,
)
from sglang.srt.speculative.ragged_verify import build_ragged_target_verify_geometry
from sglang.srt.speculative.spec_info import SpecInput, SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import resolve_num_tokens_per_req
from sglang.srt.utils import get_compiler_backend
from sglang.srt.utils.common import get_device_capability
from sglang.srt.utils.nvtx_utils import profile_range

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

from sgl_kernel import merge_state_v2
from sglang.kernels.ops.attention.flash_attention import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)


@triton.jit
def _zero_request_forest_ghost_rows_kernel(
    output,
    tree_lengths,
    row_stride: tl.constexpr,
    head_stride: tl.constexpr,
    dim_stride: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROW_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    if tl.load(tree_lengths + row) == 0:
        columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        offsets = (
            row * row_stride
            + (columns // HEAD_DIM) * head_stride
            + (columns % HEAD_DIM) * dim_stride
        )
        tl.store(output + offsets, 0, columns < ROW_WIDTH)


def zero_request_forest_ghost_rows_(output, tree_lengths):

    width = output.shape[1] * output.shape[2]
    _zero_request_forest_ghost_rows_kernel[(output.shape[0], triton.cdiv(width, 256))](
        output,
        tree_lengths,
        *output.stride(),
        output.shape[2],
        width,
        256,
    )


def _should_disable_scheduler_metadata_precompute() -> bool:
    return bool(
        get_parallel().config.enable_prefill_cp
        or get_parallel().config.enable_dp_attention
    )


@dataclass
class FlashAttentionMetadata:


    cache_seqlens_int32: torch.Tensor = None

    max_seq_len_q: int = 1

    max_seq_len_k: int = 0

    cu_seqlens_q: torch.Tensor = None

    cu_seqlens_k: torch.Tensor = None


    fa_skip_cu_seqlens_q: torch.Tensor = None
    fa_skip_max_seqlen_q: int = None

    window_size: tuple = (-1, -1)

    page_table: torch.Tensor = None

    swa_page_table: torch.Tensor = None

    pa_swa_page_table: torch.Tensor = None
    pa_swa_cache_seqlens: torch.Tensor = None

    swa_out_cache_loc: torch.Tensor = None

    scheduler_metadata: torch.Tensor = None


    rel_bias_prep_cache: Optional[dict] = None


    encoder_cu_seqlens_k: torch.Tensor = None

    encoder_max_seq_len_k: int = 0

    encoder_lens_int32: torch.Tensor = None

    encoder_page_table: torch.Tensor = None

    @dataclass
    class LocalAttentionMetadata:
        local_query_start_loc: torch.Tensor = None
        local_seqused_k: torch.Tensor = None
        local_block_table: torch.Tensor = None
        local_max_query_len: int = 0
        local_max_seq_len: int = 0

    local_attn_metadata: Optional[LocalAttentionMetadata] = None


    swa_spec_metadata: Optional[FlashAttentionMetadata] = None


class FlashAttentionBackend(AttentionBackend):


    needs_cpu_seq_lens: bool = False
    supports_ragged_verify_graph: bool = True


    supports_full_cuda_graph_chunked_prefix = True

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        speculative_step_id=0,
        topk=0,
        speculative_num_steps=0,
        fa_impl_ver=3,
    ):
        super().__init__()

        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        self.is_encoder_decoder = model_runner.model_config.is_encoder_decoder
        self.forward_metadata: FlashAttentionMetadata = None

        self.forward_metadata_spec_decode_expand: FlashAttentionMetadata = None
        self.max_context_len = model_runner.model_config.context_len
        self.device = model_runner.device
        self.decode_cuda_graph_metadata = {}
        self.target_verify_metadata = {}
        self._request_forest_graph_max_tokens = 0
        self._request_forest_graph_depth = 0
        self._request_forest_tree_workspace = None
        self._request_forest_tree_cu_seqlens_k = None
        self._request_forest_tree_metadata = {}
        self._request_forest_graph_ragged_layouts = {}
        self._request_forest_prefix_workspace = None


        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.kv_cache_dtype = model_runner.kv_cache_dtype
        self.kv_cache_dtype_str = model_runner.kv_cache_dtype_str
        self.kv_cache_is_mxfp8 = self.kv_cache_dtype_str == "mxfp8"
        self.page_size = model_runner.page_size


        self.max_num_pages = (
            self.max_context_len + self.page_size - 1
        ) // self.page_size


        self.needs_cpu_seq_lens = False
        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA


        self._unified_hooks = unified_mla_hooks(model_runner.token_to_kv_pool_allocator)
        self._unified_dense = self._unified_hooks.enabled and self.use_mla
        self.skip_prefill = skip_prefill
        self.attn_cp_size = model_runner.ps.attn_cp_size
        self._verify_mask = None


        self.is_draft_runner = model_runner.is_draft_worker

        self.use_sliding_window_kv_pool = (
            isinstance(model_runner.token_to_kv_pool, SWAKVPool)
            and model_runner.token_to_kv_pool.swa_layer_nums > 0
        )

        self.topk = get_spec().speculative_eagle_topk or 0
        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens = get_spec().speculative_num_draft_tokens
        if (
            self.speculative_num_draft_tokens is not None
            and model_runner.is_draft_worker
        ):

            self.speculative_num_draft_tokens = resolve_num_tokens_per_req(
                phase="target_verify",
                spec_algorithm=SpeculativeAlgorithm.from_string(
                    get_spec().speculative_algorithm
                ),
                is_draft_worker=True,
                num_draft_tokens=int(self.speculative_num_draft_tokens),
            )
        self.speculative_step_id = speculative_step_id


        self.has_local_attention = model_runner.model_config.is_local_attention_model
        if self.has_local_attention:
            assert (
                model_runner.attention_chunk_size is not None
            ), "Attention chunk size is required for local attention"
            self.attention_chunk_size = model_runner.attention_chunk_size


        self.sliding_window_size = model_runner.sliding_window_size
        self.has_swa = (
            self.sliding_window_size is not None and self.sliding_window_size > -1
        )

        self.is_prefill_aware_swa = getattr(model_runner, "prefill_aware_swa", False)
        if self.is_prefill_aware_swa:
            assert self.page_size == 1, (
                "Prefill-aware SWA requires page_size=1, "
                f"got page_size={self.page_size}"
            )


            max_req_pool_idx = model_runner.req_to_token_pool.size
            self._pa_swa_prefill_lens = torch.zeros(
                max_req_pool_idx + 1, dtype=torch.int32, device=model_runner.device
            )
            self._pa_swa_max_prefill_len = 0


        self.fa_impl_ver = fa_impl_ver
        device_capability = get_device_capability()
        if self.fa_impl_ver == 3:
            from sgl_kernel.flash_attn import (
                flash_attn_varlen_func,
                flash_attn_with_kvcache,
                get_scheduler_metadata,
            )

            self._get_scheduler_metadata = get_scheduler_metadata
            self._get_fa_runtime_policy = None
        elif self.fa_impl_ver == 4:
            if device_capability[0] == 12:
                from sglang.kernels.ops.attention.flash_attention_v4_sm120 import (
                    flash_attn_varlen_func,
                    flash_attn_with_kvcache,
                    get_flash_attention_v4_sm120_runtime_policy,
                )

                self._get_fa_runtime_policy = (
                    get_flash_attention_v4_sm120_runtime_policy
                )
            else:
                from sglang.kernels.ops.attention.flash_attention_v4 import (
                    flash_attn_varlen_func,
                    flash_attn_with_kvcache,
                )

                self._get_fa_runtime_policy = None

            self._get_scheduler_metadata = None
            if get_exec().deterministic.enable_deterministic_inference:

                from sglang.kernels.ops.attention.flash_attn.cute.batch_invariance import (
                    set_batch_invariant,
                )

                set_batch_invariant(True)
        else:
            raise ValueError(f"Invalid version: {self.fa_impl_ver=}")

        self.flash_attn_varlen_func = flash_attn_varlen_func
        self.flash_attn_with_kvcache = flash_attn_with_kvcache


        self.head_dim = model_runner.model_config.head_dim
        self.num_attention_heads = (
            model_runner.model_config.hf_text_config.num_attention_heads
            // model_runner.ps.tp_size
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            model_runner.ps.tp_size
        )
        _softcapping = getattr(
            model_runner.model_config.hf_text_config, "attn_logit_softcapping", None
        )
        self.has_softcap = _softcapping is not None and _softcapping > 0.0


        deterministic = get_exec().deterministic.enable_deterministic_inference
        if self._get_fa_runtime_policy is None:
            self.num_splits = 1 if deterministic else 0
            self.decode_num_splits = self.num_splits
            self._decode_uses_static_max_seqlen_k = False
        else:
            runtime_policy = self._get_fa_runtime_policy(
                device_capability=device_capability,
                deterministic=deterministic,
            )
            self.num_splits = runtime_policy.num_splits
            self.decode_num_splits = runtime_policy.decode_num_splits
            self._decode_uses_static_max_seqlen_k = (
                runtime_policy.decode_uses_static_max_seqlen_k
            )


        self.full_cg_prefill_metadata = None


        self.fa_skip_kv_cache = (
            get_model().is_embedding
            and get_schedule().chunked_prefill_size == -1
            and get_memory().disable_radix_cache
            and not self.use_mla
        )


        self._disable_scheduler_metadata_precompute = (
            _should_disable_scheduler_metadata_precompute()
        )

    def _compute_scheduler_metadata(
        self, batch_size, max_seq_len_k, cache_seqlens, cu_seqlens_q
    ):

        if self._get_scheduler_metadata is None or self.use_mla:
            return None
        if self._disable_scheduler_metadata_precompute:
            return None


        return self._get_scheduler_metadata(
            batch_size=batch_size,
            max_seqlen_q=1,
            max_seqlen_k=max_seq_len_k,
            num_heads=self.num_attention_heads,
            num_heads_k=self.num_kv_heads,
            headdim=self.head_dim,
            cache_seqlens=cache_seqlens,
            qkv_dtype=self.kv_cache_dtype,
            cu_seqlens_q=cu_seqlens_q,
            page_size=self.page_size,
            causal=True,
            has_softcap=self.has_softcap,
            num_splits=self.num_splits,
        )

    def _cuda_graph_scheduler_max_seq_len_k(
        self, metadata: FlashAttentionMetadata
    ) -> int:


        return int(metadata.page_table.shape[1]) * int(self.page_size)

    def _mxfp8_sf_kwargs(self, layer, forward_batch, q_descale=None):

        if not self.kv_cache_is_mxfp8:
            return {}
        if self.fa_impl_ver != 4:
            raise RuntimeError("MXFP8 KV cache requires the FA4 backend.")
        if q_descale is None:
            raise RuntimeError(
                "MXFP8 KV cache requires per-token Q scales (q_descale) from "
                "the attention layer for the block-scaled QK^T path."
            )


        k_sf, v_sf = self.token_to_kv_pool.get_kv_scale_buffer(layer.layer_id)
        return {"sfq": q_descale, "sfk": k_sf, "sfv": v_sf}

    @staticmethod
    def _is_request_forest_verify(forward_batch: ForwardBatch) -> bool:
        spec_info = forward_batch.spec_info
        return (
            forward_batch.forward_mode.is_target_verify()
            and spec_info is not None
            and spec_info.is_request_forest_verify_input()
        )

    def _validate_request_forest_verify_support(self) -> None:


        unsupported = []
        if self.use_mla:
            unsupported.append("MLA")
        if self.has_swa:
            unsupported.append("sliding-window attention")
        if self.has_local_attention:
            unsupported.append("local attention")
        if self.is_encoder_decoder:
            unsupported.append("cross attention")
        if self.attn_cp_size > 1:
            unsupported.append("context parallel attention")
        if unsupported:
            raise NotImplementedError(
                "Request Forest Target verification does not support "
                + ", ".join(unsupported)
            )

    def _init_request_forest_verify_metadata(
        self,
        forward_batch: ForwardBatch,
        metadata: FlashAttentionMetadata,
        *,
        eager_max_k: int,
    ) -> None:


        self._validate_request_forest_verify_support()
        spec_info = forward_batch.spec_info
        if spec_info.request_count != forward_batch.batch_size:
            raise ValueError(
                "Request Forest request_offsets must match ForwardBatch.batch_size"
            )

        prefix_lens = forward_batch.seq_lens.to(torch.int32)


        metadata.cache_seqlens_int32 = torch.nn.functional.pad(prefix_lens, (0, 1))
        metadata.max_seq_len_q = spec_info.physical_rows
        metadata.max_seq_len_k = eager_max_k
        metadata.cu_seqlens_q = torch.cat(
            (
                spec_info.request_offsets,
                spec_info.request_offsets.new_tensor([spec_info.physical_rows]),
            )
        )
        metadata.cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32), (1, 0)
        )
        metadata.page_table = torch.nn.functional.pad(
            self.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, :eager_max_k
            ],
            (0, 0, 0, 1),
        )

        tree_layout = spec_info.build_tree_attention_layout(forward_batch.out_cache_loc)
        self.forward_metadata_spec_decode_expand = FlashAttentionMetadata(
            cache_seqlens_int32=tree_layout.cache_seqlens_int32,
            max_seq_len_q=tree_layout.max_seq_len_q,
            max_seq_len_k=tree_layout.max_seq_len_k,
            cu_seqlens_q=tree_layout.cu_seqlens_q,
            cu_seqlens_k=tree_layout.cu_seqlens_k,
            page_table=tree_layout.page_table,
        )

    def _ensure_request_forest_graph_tree_buffers(self) -> None:


        if self._request_forest_tree_workspace is not None:
            return
        from sglang.srt.speculative.request_forest.forest_verify_input import (
            ForestTreeAttentionWorkspace,
        )

        depth = int(self.speculative_num_draft_tokens)
        if self._request_forest_graph_max_tokens < 1 or depth < 1:
            raise RuntimeError("invalid Request Forest CUDA Graph capacity")
        workspace = ForestTreeAttentionWorkspace()
        workspace.views(
            logical_rows=self._request_forest_graph_max_tokens,
            max_tree_depth=depth,
            device=self.req_to_token.device,
        )
        self._request_forest_tree_workspace = workspace
        self._request_forest_graph_depth = depth
        self._request_forest_tree_cu_seqlens_k = torch.zeros(
            self._request_forest_graph_max_tokens + 1,
            dtype=torch.int32,
            device=self.device,
        )

    def _bind_request_forest_graph_tree_metadata(
        self, num_tokens: int
    ) -> FlashAttentionMetadata:
        self._ensure_request_forest_graph_tree_buffers()
        metadata = self._request_forest_tree_metadata.get(num_tokens)
        if metadata is not None:
            return metadata
        page_table, cache_seqlens, cu_seqlens_q = (
            self._request_forest_tree_workspace.views(
                logical_rows=num_tokens,
                max_tree_depth=self._request_forest_graph_depth,
                device=self.req_to_token.device,
            )
        )
        metadata = FlashAttentionMetadata(
            cache_seqlens_int32=cache_seqlens,
            max_seq_len_q=1,
            max_seq_len_k=self._request_forest_graph_depth,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=self._request_forest_tree_cu_seqlens_k[: num_tokens + 1],
            page_table=page_table,
        )
        self._request_forest_tree_metadata[num_tokens] = metadata
        return metadata

    def _refill_request_forest_graph_metadata(
        self,
        *,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        spec_info,
        out_cache_loc: torch.Tensor,
    ) -> tuple[FlashAttentionMetadata, FlashAttentionMetadata]:


        live_layout = spec_info.ragged_verify_layout
        if live_layout is None or live_layout.graph_num_tokens != num_tokens:
            raise ValueError("Request Forest graph layout does not match its tier")
        key = (num_tokens, bs)
        captured_layout = self._request_forest_graph_ragged_layouts.get(key)
        if captured_layout is None or captured_layout.bs != bs:
            raise RuntimeError("Request Forest graph tier has no staged ragged layout")

        with profile_range("request_forest.stage_prefix_metadata"):
            prefix = self.target_verify_metadata[key]
            prefix.max_seq_len_q = num_tokens
            live_bs = spec_info.request_count
            prefix.cache_seqlens_int32.zero_()
            prefix.cache_seqlens_int32[:live_bs].copy_(seq_lens[:live_bs])
            prefix.cu_seqlens_q.fill_(spec_info.logical_rows)
            prefix.cu_seqlens_q[: live_bs + 1].copy_(
                spec_info.request_offsets[: live_bs + 1]
            )
            prefix.cu_seqlens_q[-1] = num_tokens
            prefix.page_table[live_bs:].zero_()
            torch.cumsum(
                prefix.cache_seqlens_int32,
                dim=0,
                dtype=torch.int32,
                out=prefix.cu_seqlens_k[1:],
            )
            build_trtllm_mha_page_table(
                req_to_token=self.req_to_token,
                req_pool_indices=req_pool_indices[:live_bs],
                cache_seqlens=prefix.cache_seqlens_int32[:live_bs],
                page_table=prefix.page_table[:live_bs],
                page_size=self.page_size,
            )

        with profile_range("request_forest.stage_tree_metadata"):
            tree = self._bind_request_forest_graph_tree_metadata(num_tokens)
            staged_tree = spec_info.build_tree_attention_layout(
                out_cache_loc,
                workspace=self._request_forest_tree_workspace,
                max_tree_depth=self._request_forest_graph_depth,
            )
            tree.cu_seqlens_k.copy_(staged_tree.cu_seqlens_k)
        return prefix, tree

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch) -> None:


        if not forward_batch.forward_mode.is_draft_extend_v2():
            return
        bs = forward_batch.batch_size
        metadata = self.draft_extend_metadata[bs]
        mapping = self._in_graph_full_to_swa_index_mapping()
        draft_extend_set_metadata(
            cache_seqlens_int32=metadata.cache_seqlens_int32,
            cu_seqlens_k=metadata.cu_seqlens_k,
            cu_seqlens_q=metadata.cu_seqlens_q,
            page_table=metadata.page_table,
            req_to_token=self.req_to_token,
            req_pool_indices=forward_batch.req_pool_indices[:bs],
            seq_lens=forward_batch.seq_lens[:bs],
            extend_seq_lens=forward_batch.extend_seq_lens[:bs],
            page_size=self.page_size,
            full_to_swa_index_mapping=mapping,
            swa_page_table=metadata.swa_page_table if mapping is not None else None,
            out_cache_loc=(
                forward_batch.out_cache_loc if mapping is not None else None
            ),
            swa_out_cache_loc=(
                metadata.swa_out_cache_loc if mapping is not None else None
            ),
        )

    def _in_graph_full_to_swa_index_mapping(self) -> Optional[torch.Tensor]:


        if not self.use_sliding_window_kv_pool:
            return None
        return self.token_to_kv_pool.full_to_swa_index_mapping

    def draft_extend_metadata_captured_in_graph(self) -> bool:
        return (
            not self.use_sliding_window_kv_pool
            or self.token_to_kv_pool.full_to_swa_index_mapping is not None
        )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):

        forward_mode = forward_batch.forward_mode
        if forward_mode.is_extend() and not (
            forward_mode.is_target_verify()
            or forward_mode.is_draft_extend_v2()
            or forward_mode.is_dllm_extend()
        ):
            self._init_full_cg_prefill_metadata(forward_batch, in_capture)
        else:
            self._init_full_cg_decode_metadata(forward_batch, in_capture)

    def _init_full_cg_decode_metadata(
        self, forward_batch: ForwardBatch, in_capture: bool
    ):

        bs = forward_batch.batch_size
        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens
        encoder_lens = forward_batch.encoder_lens
        forward_mode = forward_batch.forward_mode
        spec_info = forward_batch.spec_info
        out_cache_loc = getattr(forward_batch, "out_cache_loc", None)

        if in_capture:
            num_tokens = forward_batch.positions.numel()
            seq_lens_cpu = seq_lens.cpu()
            self._bind_metadata_buffers(
                bs,
                num_tokens,
                encoder_lens,
                forward_mode,
                spec_info,
                seq_lens.device,
            )

            if (
                forward_mode.is_decode_or_idle()
                and spec_info is not None
                and self.topk > 1
            ):


                self.forward_metadata = self.draft_decode_metadata_topk_normal[bs]
                self.forward_metadata_spec_decode_expand = (
                    self.draft_decode_metadata_topk_expand[bs]
                )
                return

            if forward_mode.is_target_verify() and self.topk > 1:


                self.forward_metadata = self.target_verify_metadata_topk_normal[bs]
                self.forward_metadata_spec_decode_expand = (
                    self.target_verify_metadata_topk_expand[bs]
                )
                return

            self._apply_cuda_graph_metadata(
                bs=bs,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_sum=None,
                encoder_lens=encoder_lens,
                forward_mode=forward_mode,
                spec_info=spec_info,
                seq_lens_cpu=seq_lens_cpu,
                out_cache_loc=out_cache_loc,
            )

            if forward_mode.is_decode_or_idle() and spec_info is None:


                metadata = self.decode_cuda_graph_metadata[bs]
                if self._decode_uses_static_max_seqlen_k:


                    metadata.max_seq_len_k = self.max_context_len
                self._maybe_update_local_attn_metadata_for_capture(metadata, bs)
                if self._sched_meta_buf is not None:
                    sched = self._compute_scheduler_metadata(
                        bs,
                        self._cuda_graph_scheduler_max_seq_len_k(metadata),
                        metadata.cache_seqlens_int32,
                        metadata.cu_seqlens_q,
                    )
                    if sched is not None:
                        n = sched.shape[0]
                        self._sched_meta_buf[:n] = sched
                        self._sched_meta_buf[n:] = 0
                        metadata.scheduler_metadata = self._sched_meta_buf[:n]

            if forward_mode.is_draft_extend_v2():


                self.forward_metadata.max_seq_len_q = num_tokens // bs
        else:


            self._apply_cuda_graph_metadata(
                bs=bs,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_sum=forward_batch.seq_lens_sum,
                encoder_lens=encoder_lens,
                forward_mode=forward_mode,
                spec_info=spec_info,
                seq_lens_cpu=(
                    forward_batch.seq_lens_cpu if self.needs_cpu_seq_lens else None
                ),
                out_cache_loc=out_cache_loc,
            )

    def _init_full_cg_prefill_metadata(
        self, forward_batch: ForwardBatch, in_capture: bool
    ):

        bs = forward_batch.batch_size
        if in_capture and getattr(self, "full_cg_prefill_metadata", None) is None:
            device = forward_batch.seq_lens.device
            m = FlashAttentionMetadata()
            m.cache_seqlens_int32 = torch.zeros((bs,), dtype=torch.int32, device=device)
            m.cu_seqlens_q = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
            m.cu_seqlens_k = torch.zeros((bs + 1,), dtype=torch.int32, device=device)


            m.page_table = torch.zeros(
                (bs, self.max_num_pages), dtype=torch.int32, device=device
            )


            self.full_cg_prefill_strided_indices = torch.arange(
                0, self.max_context_len, self.page_size, device=device
            )


            if self.use_sliding_window_kv_pool:
                assert forward_batch.out_cache_loc is not None
                m.swa_page_table = torch.zeros(
                    (bs, self.max_num_pages), dtype=torch.int32, device=device
                )
                self.full_cg_prefill_swa_out_cache_loc = torch.zeros(
                    (forward_batch.out_cache_loc.shape[0],),
                    dtype=torch.int64,
                    device=device,
                )
            self.full_cg_prefill_metadata = m
        m = self.full_cg_prefill_metadata
        assert m is not None and bs == m.cache_seqlens_int32.shape[0], (
            "full-CG prefill metadata must be created at capture with the same "
            "fixed request-slot count used at replay"
        )

        seq_lens = forward_batch.seq_lens[:bs]
        m.cache_seqlens_int32.copy_(seq_lens)
        m.cu_seqlens_k[1:].copy_(torch.cumsum(seq_lens, dim=0))
        m.cu_seqlens_q[1:].copy_(
            torch.cumsum(forward_batch.extend_seq_lens[:bs], dim=0)
        )
        max_seq_len_k = int(forward_batch.seq_lens_cpu[:bs].max().item())
        if max_seq_len_k > 0:


            max_seq_pages = (max_seq_len_k + self.page_size - 1) // self.page_size
            page_indices = self.req_to_token[
                forward_batch.req_pool_indices[:bs, None],
                self.full_cg_prefill_strided_indices[:max_seq_pages],
            ]
            m.page_table[:, :max_seq_pages].copy_(page_indices // self.page_size)
            if self.use_sliding_window_kv_pool:


                swa_starts = self.token_to_kv_pool.translate_loc_from_full_to_swa(
                    page_indices
                )
                m.swa_page_table[:, :max_seq_pages].copy_(swa_starts // self.page_size)
        if self.use_sliding_window_kv_pool and forward_batch.out_cache_loc is not None:


            num_out = forward_batch.out_cache_loc.shape[0]
            assert_buffer_fits(
                num_out,
                self.full_cg_prefill_swa_out_cache_loc.shape[0],
                "full-CG prefill SWA write-location buffer",
            )
            self.full_cg_prefill_swa_out_cache_loc[:num_out].copy_(
                self.token_to_kv_pool.translate_loc_from_full_to_swa(
                    forward_batch.out_cache_loc
                )
            )


            self.full_cg_prefill_swa_out_cache_loc[num_out:].zero_()
            m.swa_out_cache_loc = self.full_cg_prefill_swa_out_cache_loc[:num_out]
        if in_capture:

            m.max_seq_len_q = forward_batch.positions.numel()
            m.max_seq_len_k = self.max_context_len
        self.forward_metadata = m

    def init_forward_metadata(self, forward_batch: ForwardBatch):

        metadata = FlashAttentionMetadata()
        seqlens_in_batch = forward_batch.seq_lens
        batch_size = forward_batch.batch_size
        device = seqlens_in_batch.device


        seq_lens_cpu = forward_batch.seq_lens_cpu
        eager_max_k = (
            seq_lens_cpu.max().item()
            if seq_lens_cpu is not None and seq_lens_cpu.numel() > 0
            else self.max_context_len
        )

        if forward_batch.forward_mode.is_decode_or_idle():

            if forward_batch.spec_info is not None:
                if self.topk <= 1:
                    metadata.cache_seqlens_int32 = (
                        seqlens_in_batch + (self.speculative_step_id + 1)
                    ).to(torch.int32)
                    metadata.max_seq_len_k = eager_max_k + (
                        self.speculative_step_id + 1
                    )
                    metadata.cu_seqlens_q = torch.arange(
                        0, batch_size + 1, dtype=torch.int32, device=device
                    )
                    metadata.cu_seqlens_k = torch.nn.functional.pad(
                        torch.cumsum(
                            metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                        ),
                        (1, 0),
                    )
                    metadata.page_table = self.req_to_token_pool.req_to_token[
                        forward_batch.req_pool_indices, : metadata.max_seq_len_k
                    ]
                elif self.speculative_num_steps == 0:


                    metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
                    metadata.max_seq_len_k = eager_max_k
                    metadata.cu_seqlens_q = torch.arange(
                        0, batch_size + 1, dtype=torch.int32, device=device
                    )
                    metadata.cu_seqlens_k = torch.nn.functional.pad(
                        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
                    )
                    metadata.page_table = self.req_to_token_pool.req_to_token[
                        forward_batch.req_pool_indices, : metadata.max_seq_len_k
                    ]
                else:
                    metadata.cache_seqlens_int32 = (seqlens_in_batch).to(torch.int32)
                    metadata.max_seq_len_q = self.topk
                    metadata.max_seq_len_k = eager_max_k
                    metadata.cu_seqlens_q = torch.arange(
                        0,
                        batch_size * self.topk + 1,
                        step=self.topk,
                        dtype=torch.int32,
                        device=device,
                    )
                    metadata.cu_seqlens_k = torch.nn.functional.pad(
                        torch.cumsum(
                            metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                        ),
                        (1, 0),
                    )
                    metadata.page_table = self.req_to_token_pool.req_to_token[
                        forward_batch.req_pool_indices, : metadata.max_seq_len_k
                    ]
                    metadata_expand = FlashAttentionMetadata()
                    decode_length = self.speculative_step_id + 1
                    metadata_expand.cache_seqlens_int32 = torch.full(
                        (seqlens_in_batch.numel() * self.topk,),
                        decode_length,
                        device=device,
                        dtype=torch.int32,
                    )
                    metadata_expand.max_seq_len_q = 1
                    metadata_expand.cu_seqlens_q = torch.arange(
                        0,
                        metadata_expand.cache_seqlens_int32.numel() + 1,
                        dtype=torch.int32,
                        device=device,
                    )
                    metadata_expand.cu_seqlens_k = torch.arange(
                        0,
                        metadata_expand.cache_seqlens_int32.numel() * decode_length + 1,
                        step=decode_length,
                        dtype=torch.int32,
                        device=device,
                    )

                    cache_loc = forward_batch.out_cache_loc.view(
                        -1, self.speculative_num_steps
                    )
                    metadata_expand.page_table = (
                        cache_loc[:, :decode_length].contiguous().to(torch.int32)
                    )
                    self.forward_metadata_spec_decode_expand = metadata_expand
            else:

                metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
                metadata.max_seq_len_k = eager_max_k
                metadata.cu_seqlens_q = torch.arange(
                    0, batch_size + 1, dtype=torch.int32, device=device
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
                )
                metadata.page_table = self.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k
                ]

                if self.is_prefill_aware_swa and self.has_swa:
                    pa_max_len = min(
                        self._pa_swa_max_prefill_len + self.sliding_window_size,
                        metadata.max_seq_len_k,
                    )

                    pa_page_table, pa_kv_lens = _build_pa_page_table(
                        self.req_to_token,
                        forward_batch.req_pool_indices[:batch_size],
                        forward_batch.seq_lens,
                        self._pa_swa_prefill_lens,
                        self.sliding_window_size,
                        batch_size,
                        pa_max_len,
                        device,
                    )
                    metadata.pa_swa_page_table = pa_page_table
                    metadata.pa_swa_cache_seqlens = pa_kv_lens


                metadata.scheduler_metadata = self._compute_scheduler_metadata(
                    batch_size,
                    metadata.max_seq_len_k,
                    metadata.cache_seqlens_int32,
                    metadata.cu_seqlens_q,
                )

            self._maybe_init_local_attn_metadata(forward_batch, metadata, device)
        elif forward_batch.forward_mode.is_target_verify():
            if self._is_request_forest_verify(forward_batch):
                self._init_request_forest_verify_metadata(
                    forward_batch,
                    metadata,
                    eager_max_k=eager_max_k,
                )
            elif self.topk <= 1:
                ragged_layout = getattr(
                    forward_batch.spec_info, "ragged_verify_layout", None
                )
                if ragged_layout is not None:
                    geometry = build_ragged_target_verify_geometry(
                        seq_lens=forward_batch.seq_lens, layout=ragged_layout
                    )
                    metadata.cache_seqlens_int32 = geometry.cache_seqlens_int32


                    metadata.max_seq_len_q = (
                        geometry.max_seq_len_q
                        if geometry.max_seq_len_q is not None
                        else self.speculative_num_draft_tokens
                    )
                    metadata.max_seq_len_k = int(
                        metadata.cache_seqlens_int32.max().item()
                    )
                    metadata.cu_seqlens_q = geometry.cu_seqlens_q
                    metadata.cu_seqlens_k = geometry.cu_seqlens_k
                else:
                    metadata.cache_seqlens_int32 = (
                        forward_batch.seq_lens + self.speculative_num_draft_tokens
                    ).to(torch.int32)
                    metadata.max_seq_len_q = self.speculative_num_draft_tokens
                    metadata.max_seq_len_k = (
                        eager_max_k + self.speculative_num_draft_tokens
                    )
                    metadata.cu_seqlens_q = torch.arange(
                        0,
                        batch_size * self.speculative_num_draft_tokens + 1,
                        self.speculative_num_draft_tokens,
                        dtype=torch.int32,
                        device=device,
                    )
                    metadata.cu_seqlens_k = torch.nn.functional.pad(
                        torch.cumsum(
                            metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                        ),
                        (1, 0),
                    )
                metadata.page_table = self.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k
                ]

                self._maybe_init_local_attn_metadata(forward_batch, metadata, device)
            else:
                metadata.cache_seqlens_int32 = forward_batch.seq_lens.to(torch.int32)
                metadata.max_seq_len_q = self.speculative_num_draft_tokens
                metadata.max_seq_len_k = eager_max_k
                metadata.cu_seqlens_q = torch.arange(
                    0,
                    batch_size * self.speculative_num_draft_tokens + 1,
                    step=self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=device,
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(
                        metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                metadata.page_table = self.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k
                ]

                metadata_expand = FlashAttentionMetadata()

                metadata_expand.max_seq_len_q = 1
                metadata_expand.cu_seqlens_q = torch.arange(
                    0,
                    forward_batch.seq_lens.numel() * self.speculative_num_draft_tokens
                    + 1,
                    dtype=torch.int32,
                    device=device,
                )


                offsets = torch.arange(
                    self.speculative_num_draft_tokens, device=device
                ).unsqueeze(
                    0
                )
                cols = offsets.expand(
                    forward_batch.seq_lens.numel(), -1
                ) + forward_batch.seq_lens.unsqueeze(1)
                cum_len = torch.nn.functional.pad(
                    torch.cumsum(
                        (
                            forward_batch.seq_lens + self.speculative_num_draft_tokens
                        ).repeat_interleave(self.speculative_num_draft_tokens),
                        dim=0,
                    ),
                    (1, 0),
                )[:-1]
                mask_extraction_indices = (
                    cols.repeat_interleave(self.speculative_num_draft_tokens, dim=0)
                    + cum_len[:, None]
                ).view(1, -1)
                mask = forward_batch.spec_info.custom_mask[
                    mask_extraction_indices
                ].view(
                    -1, self.speculative_num_draft_tokens
                )


                col_indices = offsets.expand(
                    mask.shape[0], self.speculative_num_draft_tokens
                )


                keys = torch.where(
                    mask, col_indices, col_indices + self.speculative_num_draft_tokens
                )
                _, sort_order = torch.sort(keys, dim=1)
                non_masked_page_table = (
                    self.req_to_token_pool.req_to_token[
                        forward_batch.req_pool_indices, :
                    ]
                    .gather(1, cols)
                    .repeat_interleave(self.speculative_num_draft_tokens, dim=0)
                )
                metadata_expand.page_table = non_masked_page_table.gather(1, sort_order)
                metadata_expand.cache_seqlens_int32 = mask.sum(dim=1).to(torch.int32)
                metadata_expand.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(
                        metadata_expand.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                self.forward_metadata_spec_decode_expand = metadata_expand

                if self.has_swa:
                    self._init_sliding_window_attn_spec_metadata(
                        metadata, metadata_expand
                    )

        elif forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed(
            include_draft_extend_v2=True
        ):
            metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
            metadata.max_seq_len_k = eager_max_k
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
            )


            if (
                not is_cp_v2_active(forward_batch)
                and self.attn_cp_size > 1
                and forward_batch.global_num_tokens_cpu is not None
                and forward_batch.extend_num_tokens is not None
                and forward_batch.extend_seq_lens_cpu is not None
            ):
                padded_extend = int(forward_batch.extend_num_tokens)
                real_extend = int(sum(forward_batch.extend_seq_lens_cpu))
                pad_delta = padded_extend - real_extend
                if pad_delta > 0:
                    metadata.max_seq_len_k += pad_delta

            metadata.page_table = self.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : metadata.max_seq_len_k
            ]

            if forward_batch.forward_mode.is_draft_extend_v2():


                extend_seq_lens = forward_batch.extend_seq_lens
                metadata.max_seq_len_q = (
                    max(forward_batch.extend_seq_lens_cpu)
                    if forward_batch.extend_seq_lens_cpu is not None
                    else self.speculative_num_draft_tokens
                )
                metadata.cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )
            elif any(forward_batch.extend_prefix_lens_cpu):
                extend_seq_lens = forward_batch.extend_seq_lens
                metadata.max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)
                metadata.cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )
            else:


                metadata.max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)
                metadata.cu_seqlens_q = metadata.cu_seqlens_k


            if forward_batch.forward_mode == ForwardMode.EXTEND:
                self._maybe_init_local_attn_metadata(forward_batch, metadata, device)

            if self.is_prefill_aware_swa:
                self._pa_swa_prefill_lens[
                    forward_batch.req_pool_indices[:batch_size]
                ] = forward_batch.seq_lens[:batch_size].to(torch.int32)
                if seq_lens_cpu is not None:
                    max_pf = int(seq_lens_cpu[:batch_size].max().item())
                else:


                    max_pf = int(forward_batch.seq_lens[:batch_size].max().item())
                if max_pf > self._pa_swa_max_prefill_len:
                    self._pa_swa_max_prefill_len = max_pf


        if forward_batch.encoder_lens is not None:
            metadata.encoder_lens_int32 = forward_batch.encoder_lens.to(torch.int32)
            metadata.encoder_cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(metadata.encoder_lens_int32, dim=0, dtype=torch.int32),
                (1, 0),
            )
            metadata.encoder_max_seq_len_k = metadata.encoder_lens_int32.max().item()


            metadata.encoder_page_table = self.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : metadata.encoder_max_seq_len_k
            ]


            text_max = metadata.max_seq_len_k
            arange_text = torch.arange(
                text_max, device=forward_batch.req_pool_indices.device
            )
            text_col = forward_batch.encoder_lens.long().unsqueeze(
                1
            ) + arange_text.unsqueeze(
                0
            )
            text_row = forward_batch.req_pool_indices.unsqueeze(1).expand(-1, text_max)
            metadata.page_table = self.req_to_token_pool.req_to_token[
                text_row, text_col
            ]

        if self.use_sliding_window_kv_pool:

            metadata.swa_page_table = (
                self.token_to_kv_pool.translate_loc_from_full_to_swa(
                    metadata.page_table
                ).to(torch.int32)
            )
            if forward_batch.out_cache_loc is not None:
                metadata.swa_out_cache_loc = (
                    self.token_to_kv_pool.translate_loc_from_full_to_swa(
                        forward_batch.out_cache_loc
                    )
                )


        if self._unified_dense and metadata.page_table is not None:


            pt = metadata.page_table
            metadata.page_table = (
                self._unified_hooks.translate_kv_loc_dense(pt.reshape(-1))
                .to(torch.int32)
                .view(pt.shape)
            )


        if self.page_size > 1:
            self.strided_indices = torch.arange(
                0, metadata.page_table.shape[1], self.page_size, device=self.device
            )

            if self.use_sliding_window_kv_pool:
                metadata.swa_page_table = (
                    metadata.swa_page_table[:, self.strided_indices] // self.page_size
                )

            metadata.page_table = (
                metadata.page_table[:, self.strided_indices] // self.page_size
            )

            if (
                self.topk > 1
                and forward_batch.forward_mode.is_decode_or_idle()
                and forward_batch.spec_info is not None
            ):

                last_page_lens = forward_batch.seq_lens % self.page_size

                metadata.cache_seqlens_int32 -= last_page_lens


                expanded_last_page_lens = last_page_lens.repeat_interleave(self.topk)
                self.forward_metadata_spec_decode_expand.cache_seqlens_int32 += (
                    expanded_last_page_lens
                )


                expand_page_table = torch.zeros(
                    forward_batch.batch_size * self.topk,
                    self.speculative_num_steps,
                    dtype=torch.int32,
                    device=self.device,
                )

                cache_loc = forward_batch.out_cache_loc.view(
                    -1, self.speculative_num_steps
                )
                draft_decode_set_expand_metadata(
                    cache_seqlens_int32=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                    page_table=expand_page_table,
                    last_page_lens=last_page_lens,
                    decode_length=decode_length,
                    cache_loc=cache_loc,
                    topk=self.topk,
                    page_size=self.page_size,
                )
                self.forward_metadata_spec_decode_expand.page_table = expand_page_table

        self.forward_metadata = metadata

    def get_paged_mha_kv_cache(
        self,
        layer: RadixAttention,
        *,
        head_group_num: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_cache, value_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        return (
            key_cache.view(
                -1,
                self.page_size,
                layer.tp_k_head_num // head_group_num,
                layer.head_dim,
            ),
            value_cache.view(
                -1,
                self.page_size,
                layer.tp_v_head_num // head_group_num,
                layer.v_head_dim,
            ),
        )

    def prepare_paged_mha_query(
        self,
        q: torch.Tensor,
        q_rope: Optional[torch.Tensor],
        k_rope: Optional[torch.Tensor],
        layer: RadixAttention,
        *,
        logical_batch_size: int,
        kv_head_num: int,
        is_prefill: bool,
    ) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        k_descale = v_descale = None
        if (
            self.kv_cache_dtype_str != "auto"
            and layer.head_dim <= 256
            and not self.kv_cache_is_mxfp8
            and (not is_prefill or self.fa_impl_ver != 4)
        ):
            if layer.k_scale is not None:
                descale_shape = (logical_batch_size, kv_head_num)
                k_descale = layer.k_scale.expand(descale_shape)
                v_descale = layer.v_scale.expand(descale_shape)
            q = q.to(self.kv_cache_dtype)
            q_rope = q_rope.to(self.kv_cache_dtype) if q_rope is not None else None
            k_rope = k_rope.to(self.kv_cache_dtype) if k_rope is not None else None
        return q, q_rope, k_rope, k_descale, v_descale

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,

        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
        q_descale: Optional[torch.Tensor] = None,
        k_descale: Optional[torch.Tensor] = None,
        v_descale: Optional[torch.Tensor] = None,
        score_mod=None,
        aux_tensors=None,
        rel_bias=None,
        rel_bias_event=None,
    ):
        if score_mod is not None and self.fa_impl_ver != 4:
            raise RuntimeError("score_mod is only supported by the FA4 backend.")
        is_cp_mode = (
            forward_batch.forward_mode.is_context_parallel_extend()
            and forward_batch.attn_cp_metadata is not None
            and self.attn_cp_size > 1
        )

        if k is not None:
            assert v is not None

            if save_kv_cache and not self.fa_skip_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                if self.use_mla:
                    if is_cp_v2_active(forward_batch):


                        cp_strategy = get_cp_strategy()
                        assert cp_strategy is not None
                        cp_strategy.materialize_full_mla_kv(
                            forward_batch, layer, k, k_rope
                        )
                    else:


                        self.token_to_kv_pool.set_mla_kv_buffer(
                            layer,
                            cache_loc,
                            k,
                            k_rope,
                        )
                elif is_cp_mode:


                    swa_loc = (
                        self.forward_metadata.swa_out_cache_loc
                        if self.use_sliding_window_kv_pool
                        else None
                    )
                    if is_cp_v2_active(forward_batch):
                        cp_strategy = get_cp_strategy()
                        assert cp_strategy is not None
                        cp_strategy.materialize_full_kv(
                            forward_batch, layer, k, v, swa_loc=swa_loc
                        )
                    else:
                        cp_allgather_and_save_kv_cache(
                            forward_batch,
                            layer,
                            k,
                            v,
                            self.attn_cp_size,
                            swa_loc=swa_loc,
                        )
                else:
                    k_scale = k_descale if self.kv_cache_is_mxfp8 else layer.k_scale
                    v_scale = v_descale if self.kv_cache_is_mxfp8 else layer.v_scale
                    self.token_to_kv_pool.set_kv_buffer(
                        layer,
                        KVWriteLoc(cache_loc, self.forward_metadata.swa_out_cache_loc),
                        k,
                        v,
                        k_scale,
                        v_scale,
                    )


        metadata = self.forward_metadata


        is_swa_layer = (
            layer.sliding_window_size is not None and layer.sliding_window_size > -1
        )
        causal = not (
            layer.is_cross_attention
            or layer.attn_type
            in (AttentionType.ENCODER_ONLY, AttentionType.DECODER_BIDIRECTIONAL)
        )


        window_size = (
            (layer.sliding_window_size, 0 if causal else layer.sliding_window_size)
            if is_swa_layer
            else (-1, -1)
        )
        q, q_rope, k_rope, fa_k_descale, fa_v_descale = self.prepare_paged_mha_query(
            q,
            q_rope,
            k_rope,
            layer,
            logical_batch_size=forward_batch.batch_size,
            kv_head_num=layer.tp_k_head_num,
            is_prefill=True,
        )

        use_local_attn = (
            self.has_local_attention
            and self.attention_chunk_size is not None
            and metadata.local_attn_metadata is not None
            and (hasattr(layer, "use_irope") and layer.use_irope)
        )


        use_cascade_attn = (
            forward_batch.forward_mode.is_target_verify()
            and (self.topk > 1 or self._is_request_forest_verify(forward_batch))
            and not is_swa_layer
        )

        kwargs = {}
        if sinks is not None:
            kwargs["sinks"] = sinks
        if score_mod is not None:
            kwargs["score_mod"] = score_mod
            kwargs["aux_tensors"] = aux_tensors
        kwargs.update(self._mxfp8_sf_kwargs(layer, forward_batch, q_descale))
        if fa_k_descale is not None:
            kwargs["k_descale"] = fa_k_descale
            kwargs["v_descale"] = fa_v_descale
        if rel_bias is not None:
            if self.fa_impl_ver != 4:
                raise RuntimeError(
                    "rel_bias (sheared bias) is only supported by the FA4 backend."
                )
            if rel_bias_event is not None:


                rel_bias_event.wait()
            kwargs["rel_bias"] = rel_bias
            if metadata is self.full_cg_prefill_metadata:


                kwargs["rel_bias_prep_cache"] = None
            else:
                if metadata.rel_bias_prep_cache is None:
                    metadata.rel_bias_prep_cache = {}
                kwargs["rel_bias_prep_cache"] = metadata.rel_bias_prep_cache

        _fa_out = (
            forward_batch._attn_output.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            if getattr(forward_batch, "_attn_output", None) is not None
            else None
        )


        if use_local_attn:
            local_metadata = metadata.local_attn_metadata
            page_table = local_metadata.local_block_table
            cu_seqlens_q = local_metadata.local_query_start_loc
            cache_seqlens = local_metadata.local_seqused_k
            max_seqlen_q = local_metadata.local_max_query_len
        elif is_swa_layer and metadata.swa_spec_metadata is not None:
            swa_spec_metadata = metadata.swa_spec_metadata
            page_table = swa_spec_metadata.page_table
            cu_seqlens_q = swa_spec_metadata.cu_seqlens_q
            cache_seqlens = swa_spec_metadata.cache_seqlens_int32
            max_seqlen_q = swa_spec_metadata.max_seq_len_q
            cu_seqlens_k = swa_spec_metadata.cu_seqlens_k
        else:
            page_table = metadata.page_table
            if is_swa_layer and self.use_sliding_window_kv_pool:
                if metadata.swa_page_table is not None:
                    page_table = metadata.swa_page_table
                else:
                    page_table = self.token_to_kv_pool.translate_loc_from_full_to_swa(
                        metadata.page_table
                    ).to(torch.int32)
            cu_seqlens_q = metadata.cu_seqlens_q
            cache_seqlens = metadata.cache_seqlens_int32
            max_seqlen_q = metadata.max_seq_len_q
            cu_seqlens_k = metadata.cu_seqlens_k


        if not self.use_mla:

            key_cache, value_cache = self.get_paged_mha_kv_cache(
                layer,
            )
            if layer.is_cross_attention:
                page_table = metadata.encoder_page_table
                cache_seqlens = metadata.encoder_lens_int32
                cu_seqlens_k = metadata.encoder_cu_seqlens_k
                window_size = (-1, -1)

            if (
                forward_batch.forward_mode.is_context_parallel_extend()
                and forward_batch.attn_cp_metadata is not None
                and self.attn_cp_size > 1
            ):

                def _fa_cp_attn(
                    q_chunk, cu_seqlens_q_cp, cache_seqlens_cp, max_seqlen_q_cp
                ):
                    return flash_attn_with_kvcache(
                        q=q_chunk,
                        k_cache=key_cache,
                        v_cache=value_cache,
                        page_table=page_table,
                        cache_seqlens=cache_seqlens_cp,
                        cu_seqlens_q=cu_seqlens_q_cp,
                        cu_seqlens_k_new=cu_seqlens_k if not use_local_attn else None,
                        max_seqlen_q=max_seqlen_q_cp,
                        softmax_scale=layer.scaling,
                        causal=False if use_cascade_attn else causal,
                        window_size=window_size,
                        softcap=layer.logit_cap,
                        return_softmax_lse=use_cascade_attn,
                        num_splits=self.num_splits,
                        ver=self.fa_impl_ver,
                        **kwargs,
                    )

                q_cp = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
                if is_cp_v2_active(forward_batch):
                    cp_strategy = get_cp_strategy()
                    assert cp_strategy is not None
                    result = cp_strategy.run_attention(
                        q_cp,
                        forward_batch,
                        self.device,
                        _fa_cp_attn,
                        attention_backend=CPAttentionBackendKind.FLASH_ATTENTION,
                    )
                else:
                    result = cp_attn_forward_extend(
                        forward_batch,
                        q_cp,
                        self.device,
                        _fa_cp_attn,
                    )
            elif self.fa_skip_kv_cache:


                assert k is not None, "fa_skip_kv_cache requires k to be provided"
                assert k_descale is None and v_descale is None, (
                    "fa_skip_kv_cache uses raw K/V tensors, "
                    "FP8 KV cache descaling is not supported in this mode"
                )


                if metadata.fa_skip_cu_seqlens_q is None:
                    num_real_tokens = forward_batch.extend_num_tokens
                    num_padded_tokens = q.shape[0]
                    if (
                        num_real_tokens is not None
                        and num_real_tokens < num_padded_tokens
                    ):
                        metadata.fa_skip_cu_seqlens_q = torch.cat(
                            [cu_seqlens_q, cu_seqlens_q.new_tensor([num_padded_tokens])]
                        )
                        metadata.fa_skip_max_seqlen_q = max(
                            int(max_seqlen_q), num_padded_tokens - num_real_tokens
                        )
                    else:
                        metadata.fa_skip_cu_seqlens_q = cu_seqlens_q
                        metadata.fa_skip_max_seqlen_q = max_seqlen_q
                result = flash_attn_varlen_func(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k=k.view(-1, layer.tp_k_head_num, layer.head_dim),
                    v=v.view(-1, layer.tp_v_head_num, layer.v_head_dim),
                    cu_seqlens_q=metadata.fa_skip_cu_seqlens_q,
                    cu_seqlens_k=metadata.fa_skip_cu_seqlens_q,
                    max_seqlen_q=metadata.fa_skip_max_seqlen_q,
                    max_seqlen_k=metadata.fa_skip_max_seqlen_q,
                    softmax_scale=layer.scaling,
                    causal=causal,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    num_splits=self.num_splits,
                    out=_fa_out,
                    **kwargs,
                )
            else:
                result = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k_new=cu_seqlens_k if not use_local_attn else None,
                    max_seqlen_q=max_seqlen_q,
                    softmax_scale=layer.scaling,
                    causal=False if use_cascade_attn else causal,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    return_softmax_lse=use_cascade_attn,
                    num_splits=self.num_splits,
                    out=_fa_out,
                    ver=self.fa_impl_ver,
                    **kwargs,
                )

            if use_cascade_attn:
                o, softmax_lse, *rest = result
                o_expand, softmax_lse_expand, *rest_expand = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),


                    k_cache=key_cache.view(-1, 1, layer.tp_k_head_num, layer.head_dim),
                    v_cache=value_cache.view(
                        -1, 1, layer.tp_v_head_num, layer.head_dim
                    ),
                    page_table=self.forward_metadata_spec_decode_expand.page_table,
                    cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                    cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                    cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                    max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                    softmax_scale=layer.scaling,
                    causal=False,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    return_softmax_lse=True,
                    num_splits=self.num_splits,
                    ver=self.fa_impl_ver,
                    **kwargs,
                )
                o, _ = merge_state_v2_wrapper(
                    o,
                    softmax_lse.T.contiguous(),
                    o_expand,
                    softmax_lse_expand.T.contiguous(),
                )
                if self._is_request_forest_verify(forward_batch):
                    zero_request_forest_ghost_rows_(
                        o, self.forward_metadata_spec_decode_expand.cache_seqlens_int32
                    )
            else:
                o = result
        else:
            if (
                forward_batch.attn_attend_prefix_cache is not None
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend_v2()
            ):

                if forward_batch.attn_attend_prefix_cache:
                    assert not get_schedule().disable_chunked_prefix_cache

                    assert forward_batch.prefix_chunk_idx is not None
                    assert forward_batch.prefix_chunk_cu_seq_lens is not None
                    assert forward_batch.prefix_chunk_max_seq_lens is not None

                    chunk_idx = forward_batch.prefix_chunk_idx
                    assert chunk_idx >= 0

                    assert forward_batch.mha_return_lse
                    output = flash_attn_varlen_func(
                        q=q.view(-1, layer.tp_q_head_num, layer.head_dim),
                        k=k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),
                        v=v.view(-1, layer.tp_k_head_num, layer.v_head_dim).to(q.dtype),
                        cu_seqlens_q=metadata.cu_seqlens_q,
                        cu_seqlens_k=forward_batch.prefix_chunk_cu_seq_lens[chunk_idx],
                        max_seqlen_q=metadata.max_seq_len_q,
                        max_seqlen_k=forward_batch.prefix_chunk_max_seq_lens[chunk_idx],
                        softmax_scale=layer.scaling,
                        causal=False,
                        return_softmax_lse=True,
                        out=_fa_out,
                        ver=self.fa_impl_ver,
                        **kwargs,
                    )
                else:

                    cu_seqlens_k = (
                        metadata.cu_seqlens_q
                        if not forward_batch.mha_one_shot
                        else metadata.cu_seqlens_k
                    )
                    max_seqlen_k = (
                        metadata.max_seq_len_q
                        if not forward_batch.mha_one_shot
                        else metadata.max_seq_len_k
                    )
                    output = flash_attn_varlen_func(
                        q=q.view(-1, layer.tp_q_head_num, layer.head_dim),
                        k=k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),
                        v=v.view(-1, layer.tp_k_head_num, layer.v_head_dim).to(q.dtype),
                        cu_seqlens_q=metadata.cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        max_seqlen_q=metadata.max_seq_len_q,
                        max_seqlen_k=max_seqlen_k,
                        softmax_scale=layer.scaling,
                        causal=True,
                        return_softmax_lse=forward_batch.mha_return_lse,
                        out=_fa_out,
                        ver=self.fa_impl_ver,
                        **kwargs,
                    )
                if forward_batch.mha_return_lse:
                    output, lse, *rest = output
                    lse = torch.transpose(lse, 0, 1).contiguous()
                    return output, lse
                return output
            else:


                assert self.fa_impl_ver in (3, 4), "Only FA3/FA4 support here"

                kv_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id).to(
                    q.dtype
                )
                k_rope = kv_cache[:, :, layer.v_head_dim :]
                c_kv = kv_cache[:, :, : layer.v_head_dim]
                k_rope_cache = k_rope.view(
                    -1,
                    self.page_size,
                    layer.tp_k_head_num,
                    layer.head_dim - layer.v_head_dim,
                )
                c_kv_cache = c_kv.view(
                    -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                )
                if q_rope is not None:
                    q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
                    q_rope = q_rope.view(
                        -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
                    )
                else:
                    q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
                    q_nope = q_all[:, :, : layer.v_head_dim]
                    q_rope = q_all[:, :, layer.v_head_dim :]

                if is_cp_mode:


                    assert (
                        not use_cascade_attn
                    ), "Cascade attention under MLA CP is not supported in v1."
                    q_fused = torch.cat([q_nope, q_rope], dim=-1)

                    def _mla_cp_attn(
                        q_chunk,
                        cu_seqlens_q_cp,
                        cache_seqlens_cp,
                        max_seqlen_q_cp,
                    ):
                        q_nope_chunk = q_chunk[..., : layer.v_head_dim]
                        q_rope_chunk = q_chunk[..., layer.v_head_dim :]
                        return flash_attn_with_kvcache(
                            q=q_rope_chunk,
                            qv=q_nope_chunk,
                            k_cache=k_rope_cache,
                            v_cache=c_kv_cache,
                            page_table=page_table,
                            cache_seqlens=cache_seqlens_cp,
                            cu_seqlens_q=cu_seqlens_q_cp,
                            cu_seqlens_k_new=(
                                cu_seqlens_k if not use_local_attn else None
                            ),
                            max_seqlen_q=max_seqlen_q_cp,
                            softmax_scale=layer.scaling,
                            causal=causal,
                            softcap=layer.logit_cap,
                            k_descale=fa_k_descale,
                            v_descale=fa_v_descale,
                            num_splits=self.num_splits,
                            ver=self.fa_impl_ver,
                        )

                    if is_cp_v2_active(forward_batch):
                        cp_strategy = get_cp_strategy()
                        assert cp_strategy is not None
                        o = cp_strategy.run_attention(
                            q_fused,
                            forward_batch,
                            self.device,
                            _mla_cp_attn,
                            attention_backend=CPAttentionBackendKind.FLASH_ATTENTION,
                        )
                    else:
                        o = cp_attn_forward_extend(
                            forward_batch, q_fused, self.device, _mla_cp_attn
                        )
                else:
                    result = flash_attn_with_kvcache(
                        q=q_rope,
                        k_cache=k_rope_cache,
                        v_cache=c_kv_cache,
                        qv=q_nope,
                        page_table=page_table,
                        cache_seqlens=cache_seqlens,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k_new=cu_seqlens_k if not use_local_attn else None,
                        max_seqlen_q=max_seqlen_q,
                        softmax_scale=layer.scaling,
                        causal=False if use_cascade_attn else causal,
                        softcap=layer.logit_cap,
                        k_descale=fa_k_descale,
                        v_descale=fa_v_descale,
                        return_softmax_lse=use_cascade_attn,
                        num_splits=self.num_splits,
                        ver=self.fa_impl_ver,
                    )
                    if use_cascade_attn:
                        o, softmax_lse, *rest = result
                        o_expand, softmax_lse_expand, *rest_expand = (
                            flash_attn_with_kvcache(
                                q=q_rope,
                                k_cache=k_rope_cache,
                                v_cache=c_kv_cache,
                                qv=q_nope,
                                page_table=self.forward_metadata_spec_decode_expand.page_table,
                                cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                                cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                                cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                                max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                                softmax_scale=layer.scaling,
                                causal=False,
                                window_size=window_size,
                                softcap=layer.logit_cap,
                                k_descale=fa_k_descale,
                                v_descale=fa_v_descale,
                                return_softmax_lse=True,
                                num_splits=self.num_splits,
                                ver=self.fa_impl_ver,
                            )
                        )
                        o, _ = merge_state_v2_wrapper(
                            o,
                            softmax_lse.T.contiguous(),
                            o_expand,
                            softmax_lse_expand.T.contiguous(),
                        )
                    else:
                        o = result

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,

        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
        q_descale: Optional[torch.Tensor] = None,
        k_descale: Optional[torch.Tensor] = None,
        v_descale: Optional[torch.Tensor] = None,
        score_mod=None,
        aux_tensors=None,
        rel_bias=None,
        rel_bias_event=None,
    ) -> torch.Tensor:
        if score_mod is not None and self.fa_impl_ver != 4:
            raise RuntimeError("score_mod is only supported by the FA4 backend.")
        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                if not self.use_mla:
                    k_scale = k_descale if self.kv_cache_is_mxfp8 else layer.k_scale
                    v_scale = v_descale if self.kv_cache_is_mxfp8 else layer.v_scale
                    self.token_to_kv_pool.set_kv_buffer(
                        layer,
                        KVWriteLoc(cache_loc, self.forward_metadata.swa_out_cache_loc),
                        k,
                        v,
                        k_scale,
                        v_scale,
                    )
                else:
                    self.token_to_kv_pool.set_mla_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        k_rope,
                    )


        metadata = self.forward_metadata
        local_attn_metadata = getattr(metadata, "local_attn_metadata", None)
        use_local_attn = (
            self.has_local_attention
            and self.attention_chunk_size is not None
            and local_attn_metadata is not None
            and (hasattr(layer, "use_irope") and layer.use_irope)
        )


        use_cascade_attn = forward_batch.spec_info is not None and self.topk > 1


        is_swa_layer = (
            layer.sliding_window_size is not None and layer.sliding_window_size > -1
        )
        window_size = (layer.sliding_window_size, 0) if is_swa_layer else (-1, -1)

        causal = True
        if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
            causal = False

        kwargs = {}
        if sinks is not None:
            kwargs["sinks"] = sinks
        if score_mod is not None:
            kwargs["score_mod"] = score_mod
            kwargs["aux_tensors"] = aux_tensors
        kwargs.update(self._mxfp8_sf_kwargs(layer, forward_batch, q_descale))
        if rel_bias is not None:
            if self.fa_impl_ver != 4:
                raise RuntimeError(
                    "rel_bias (sheared bias) is only supported by the FA4 backend."
                )
            if rel_bias_event is not None:


                rel_bias_event.wait()
            kwargs["rel_bias"] = rel_bias
            if metadata is self.full_cg_prefill_metadata:


                kwargs["rel_bias_prep_cache"] = None
            else:
                if metadata.rel_bias_prep_cache is None:
                    metadata.rel_bias_prep_cache = {}
                kwargs["rel_bias_prep_cache"] = metadata.rel_bias_prep_cache

        _fa_out = (
            forward_batch._attn_output.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            if getattr(forward_batch, "_attn_output", None) is not None
            else None
        )

        q, q_rope, k_rope, fa_k_descale, fa_v_descale = self.prepare_paged_mha_query(
            q,
            q_rope,
            k_rope,
            layer,
            logical_batch_size=forward_batch.batch_size,
            kv_head_num=layer.tp_k_head_num,
            is_prefill=False,
        )
        if fa_k_descale is not None:
            kwargs["k_descale"] = fa_k_descale
            kwargs["v_descale"] = fa_v_descale
        if not self.use_mla:


            key_cache, value_cache = self.get_paged_mha_kv_cache(
                layer,
            )

            if layer.is_cross_attention:

                if self._decode_uses_static_max_seqlen_k:
                    kwargs["max_seqlen_k"] = metadata.encoder_max_seq_len_k
                o = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=metadata.encoder_page_table,
                    cache_seqlens=metadata.encoder_lens_int32,
                    cu_seqlens_q=metadata.cu_seqlens_q,
                    cu_seqlens_k_new=metadata.encoder_cu_seqlens_k,
                    max_seqlen_q=1,
                    softmax_scale=layer.scaling,
                    causal=False,
                    window_size=(-1, -1),
                    softcap=layer.logit_cap,
                    num_splits=self.num_splits,
                    ver=self.fa_impl_ver,
                    **kwargs,
                )
            elif use_local_attn:

                if self._decode_uses_static_max_seqlen_k:
                    kwargs["max_seqlen_k"] = local_attn_metadata.local_max_seq_len
                o = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=local_attn_metadata.local_block_table,
                    cache_seqlens=local_attn_metadata.local_seqused_k,
                    cu_seqlens_q=local_attn_metadata.local_query_start_loc,
                    cu_seqlens_k_new=None,
                    max_seqlen_q=local_attn_metadata.local_max_query_len,
                    softmax_scale=layer.scaling,
                    causal=True,
                    window_size=(-1, -1),
                    softcap=layer.logit_cap,
                    num_splits=self.num_splits,
                    ver=self.fa_impl_ver,
                    **kwargs,
                )
            else:
                page_table = metadata.page_table
                if is_swa_layer and self.use_sliding_window_kv_pool:
                    if metadata.swa_page_table is not None:
                        page_table = metadata.swa_page_table
                    else:
                        page_table = (
                            self.token_to_kv_pool.translate_loc_from_full_to_swa(
                                metadata.page_table
                            ).to(torch.int32)
                        )
                cache_seqlens = metadata.cache_seqlens_int32
                max_seqlen_q = metadata.max_seq_len_q

                pa_swa_active = False
                if self.is_prefill_aware_swa and metadata.pa_swa_page_table is not None:
                    page_table = metadata.pa_swa_page_table
                    cache_seqlens = metadata.pa_swa_cache_seqlens
                    window_size = (-1, -1)
                    pa_swa_active = True

                q_reshaped = q.contiguous().view(
                    -1, layer.tp_q_head_num, layer.head_dim
                )


                sched_meta = None
                if (
                    metadata.scheduler_metadata is not None
                    and not is_swa_layer
                    and not use_cascade_attn
                    and not pa_swa_active
                ):
                    sched_meta = metadata.scheduler_metadata
                if self._decode_uses_static_max_seqlen_k:
                    kwargs["max_seqlen_k"] = metadata.max_seq_len_k
                result = flash_attn_with_kvcache(
                    q=q_reshaped,
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=metadata.cu_seqlens_q,
                    max_seqlen_q=max_seqlen_q,
                    softmax_scale=layer.scaling,
                    causal=False if use_cascade_attn else causal,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    return_softmax_lse=use_cascade_attn,
                    num_splits=(
                        self.decode_num_splits
                        if not is_swa_layer
                        and not use_cascade_attn
                        and not pa_swa_active
                        else self.num_splits
                    ),
                    out=_fa_out,
                    ver=self.fa_impl_ver,
                    scheduler_metadata=sched_meta,
                    **kwargs,
                )
                if use_cascade_attn:
                    o, softmax_lse, *rest = result
                    if self._decode_uses_static_max_seqlen_k:
                        kwargs["max_seqlen_k"] = (
                            self.forward_metadata_spec_decode_expand.max_seq_len_k
                        )
                    o_expand, softmax_lse_expand, *rest_expand = (
                        flash_attn_with_kvcache(
                            q=q_reshaped,
                            k_cache=key_cache,
                            v_cache=value_cache,
                            page_table=self.forward_metadata_spec_decode_expand.page_table,
                            cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                            cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                            cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                            max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                            softmax_scale=layer.scaling,
                            causal=False,
                            window_size=window_size,
                            softcap=layer.logit_cap,
                            return_softmax_lse=True,
                            num_splits=self.num_splits,
                            ver=self.fa_impl_ver,
                            **kwargs,
                        )
                    )
                    o, _ = merge_state_v2(
                        o,
                        softmax_lse.T.contiguous(),
                        o_expand,
                        softmax_lse_expand.T.contiguous(),
                    )
                else:
                    o = result
        else:

            kv_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id).to(q.dtype)
            k_rope = kv_cache[:, :, layer.v_head_dim :]
            c_kv = kv_cache[:, :, : layer.v_head_dim]
            k_rope_cache = k_rope.view(
                -1,
                self.page_size,
                layer.tp_k_head_num,
                layer.head_dim - layer.v_head_dim,
            )
            c_kv_cache = c_kv.view(
                -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
            )

            if q_rope is not None:
                q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
                q_rope = q_rope.view(
                    -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
                )
            else:
                q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
                q_nope = q_all[:, :, : layer.v_head_dim]
                q_rope = q_all[:, :, layer.v_head_dim :]
            max_seqlen_q = metadata.max_seq_len_q

            result = flash_attn_with_kvcache(
                q=q_rope,
                k_cache=k_rope_cache,
                v_cache=c_kv_cache,
                qv=q_nope,
                page_table=metadata.page_table,
                cache_seqlens=metadata.cache_seqlens_int32,
                cu_seqlens_q=metadata.cu_seqlens_q,
                cu_seqlens_k_new=metadata.cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                softmax_scale=layer.scaling,
                causal=False if use_cascade_attn else causal,
                softcap=layer.logit_cap,
                k_descale=fa_k_descale,
                v_descale=fa_v_descale,
                return_softmax_lse=use_cascade_attn,
                num_splits=self.num_splits,
                ver=self.fa_impl_ver,
            )
            if use_cascade_attn:
                o, softmax_lse, *rest = result
                o_expand, softmax_lse_expand, *rest_expand = flash_attn_with_kvcache(
                    q=q_rope,
                    k_cache=k_rope_cache,
                    v_cache=c_kv_cache,
                    qv=q_nope,
                    page_table=self.forward_metadata_spec_decode_expand.page_table,
                    cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                    cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                    cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                    max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                    softmax_scale=layer.scaling,
                    causal=False,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    k_descale=fa_k_descale,
                    v_descale=fa_v_descale,
                    return_softmax_lse=True,
                    num_splits=self.num_splits,
                    ver=self.fa_impl_ver,
                )
                o, _ = merge_state_v2(
                    o,
                    softmax_lse.T.contiguous(),
                    o_expand,
                    softmax_lse_expand.T.contiguous(),
                )
            else:
                o = result

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):

        self._request_forest_graph_max_tokens = max_num_tokens
        self._request_forest_graph_depth = 0
        self._request_forest_tree_workspace = None
        self._request_forest_tree_cu_seqlens_k = None
        self._request_forest_tree_metadata = {}
        self._request_forest_graph_ragged_layouts = {}
        self._request_forest_prefix_workspace = None
        max_num_pages = (self.max_context_len + self.page_size - 1) // self.page_size


        self.decode_cuda_graph_metadata = {
            "cache_seqlens": torch.zeros(max_bs, dtype=torch.int32, device=self.device),
            "cu_seqlens_q": torch.arange(
                0, max_bs + 1, dtype=torch.int32, device=self.device
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
            "strided_indices": torch.arange(
                0, self.max_context_len, self.page_size, device=self.device
            ),
        }


        if self._get_scheduler_metadata is not None and not self.use_mla:
            b_rounded = ((max_bs + 3) // 4) * 4
            self._sched_meta_buf = torch.zeros(
                1 + b_rounded * 4, dtype=torch.int32, device=self.device
            )
        else:
            self._sched_meta_buf = None


        if self.has_local_attention:

            max_seq_len = self.max_context_len
            page_size = self.page_size or 1
            attn_chunk_size = self.attention_chunk_size
            max_virtual_batches = max_bs * (
                (max_seq_len + attn_chunk_size - 1) // attn_chunk_size
            )
            max_pages_per_block = (attn_chunk_size + page_size - 1) // page_size

            self.decode_cuda_graph_local_attn_metadata = {
                "local_query_start_loc": torch.zeros(
                    max_virtual_batches + 1, dtype=torch.int32, device=self.device
                ),
                "local_seqused_k": torch.zeros(
                    max_virtual_batches, dtype=torch.int32, device=self.device
                ),
                "local_block_table": torch.zeros(
                    max_virtual_batches,
                    max_pages_per_block,
                    dtype=torch.int32,
                    device=self.device,
                ),
            }

        if self.use_sliding_window_kv_pool:
            self.decode_cuda_graph_metadata["swa_page_table"] = torch.zeros(
                max_bs,
                max_num_pages,
                dtype=torch.int32,
                device=self.device,
            )


            self.cuda_graph_swa_out_cache_loc = torch.zeros(
                max_num_tokens,
                dtype=torch.int64,
                device=self.device,
            )


        if self.topk > 1:
            self.draft_decode_metadata_topk_normal = {
                "cache_seqlens": torch.zeros(
                    max_bs, dtype=torch.int32, device=self.device
                ),
                "cu_seqlens_q": torch.arange(
                    0,
                    max_bs * self.topk + 1,
                    step=self.topk,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_k": torch.zeros(
                    max_bs + 1, dtype=torch.int32, device=self.device
                ),
                "page_table": torch.zeros(
                    max_bs,
                    self.max_context_len,
                    dtype=torch.int32,
                    device=self.device,
                ),
            }


            decode_length = self.speculative_step_id + 1
            self.draft_decode_metadata_topk_expand = {
                "cache_seqlens": torch.full(
                    (max_bs * self.topk,),
                    decode_length,
                    device=self.device,
                    dtype=torch.int32,
                ),
                "cu_seqlens_q": torch.arange(
                    0,
                    max_bs * self.topk + 1,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_k": torch.arange(
                    0,
                    max_bs * self.topk * decode_length + 1,
                    step=decode_length,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "page_table": torch.zeros(
                    max_bs * self.topk,
                    decode_length + 1,
                    dtype=torch.int32,
                    device=self.device,
                ),
            }

        if (
            self.speculative_num_draft_tokens is not None
            and self.speculative_num_draft_tokens > 0
        ):

            self.decode_cuda_graph_metadata["page_table_draft_decode"] = torch.zeros(
                max_bs,
                max_num_pages,
                dtype=torch.int32,
                device=self.device,
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
                "strided_indices": torch.arange(
                    0, self.max_context_len, self.page_size, device=self.device
                ),
            }


            self._verify_mask = maybe_create_verify_mask(
                is_draft_runner=self.is_draft_runner,
                skip_prefill=self.skip_prefill,
                max_bs=max_bs,
                max_context_len=self.max_context_len,
                num_draft_tokens=self.speculative_num_draft_tokens,
                device=self.device,
                is_read=self.topk > 1,
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
                "strided_indices": torch.arange(
                    0, self.max_context_len, self.page_size, device=self.device
                ),
            }

            if self.use_sliding_window_kv_pool:
                self.target_verify_metadata["swa_page_table"] = torch.zeros(
                    max_bs,
                    max_num_pages,
                    dtype=torch.int32,
                    device=self.device,
                )
                self.draft_extend_metadata["swa_page_table"] = torch.zeros(
                    max_bs,
                    max_num_pages,
                    dtype=torch.int32,
                    device=self.device,
                )

        if self.topk > 1:
            self.target_verify_metadata_topk_normal = {
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
                    self.max_context_len,
                    dtype=torch.int32,
                    device=self.device,
                ),
            }

            self.target_verify_metadata_topk_expand = {
                "cache_seqlens": torch.zeros(
                    max_bs * self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_k": torch.zeros(
                    max_bs * self.speculative_num_draft_tokens + 1,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_q": torch.arange(
                    0,
                    max_bs * self.speculative_num_draft_tokens + 1,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "page_table": torch.zeros(
                    max_bs * self.speculative_num_draft_tokens,
                    self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ),
            }

            if self.has_swa:
                self.target_verify_metadata_topk_swa = {
                    "cache_seqlens": torch.zeros(
                        max_bs * self.speculative_num_draft_tokens,
                        dtype=torch.int32,
                        device=self.device,
                    ),
                    "cu_seqlens_k": torch.zeros(
                        max_bs * self.speculative_num_draft_tokens + 1,
                        dtype=torch.int32,
                        device=self.device,
                    ),
                    "cu_seqlens_q": torch.arange(
                        0,
                        max_bs * self.speculative_num_draft_tokens + 1,
                        dtype=torch.int32,
                        device=self.device,
                    ),


                    "page_table": torch.zeros(
                        max_bs * self.speculative_num_draft_tokens,
                        self.max_context_len + self.speculative_num_draft_tokens,
                        dtype=torch.int32,
                        device=self.device,
                    ),
                }


        if self.is_encoder_decoder:
            self.encoder_metadata = {
                "encoder_page_table": torch.zeros(
                    max_bs,
                    self.max_context_len,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "encoder_lens_int32": torch.zeros(
                    max_bs, dtype=torch.int32, device=self.device
                ),
                "encoder_cu_seqlens_k": torch.zeros(
                    max_bs + 1, dtype=torch.int32, device=self.device
                ),
            }
        else:

            self.encoder_metadata = {}

    def _bind_metadata_buffers(
        self,
        bs: int,
        num_tokens: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        device: torch.device,
    ) -> tuple:

        metadata = FlashAttentionMetadata()
        metadata_expand = FlashAttentionMetadata()

        if forward_mode.is_decode_or_idle():
            if spec_info is not None:
                if self.topk <= 1:

                    metadata.cache_seqlens_int32 = self.decode_cuda_graph_metadata[
                        "cache_seqlens"
                    ][:bs]
                    metadata.cu_seqlens_q = self.decode_cuda_graph_metadata[
                        "cu_seqlens_q"
                    ][: bs + 1]
                    metadata.cu_seqlens_k = self.decode_cuda_graph_metadata[
                        "cu_seqlens_k"
                    ][: bs + 1]
                    metadata.page_table = self.decode_cuda_graph_metadata[
                        "page_table_draft_decode"
                    ][:bs, :]
                    if self.use_sliding_window_kv_pool:
                        metadata.swa_page_table = self.decode_cuda_graph_metadata[
                            "swa_page_table"
                        ][:bs, :]
                        metadata.swa_out_cache_loc = self.cuda_graph_swa_out_cache_loc[
                            :num_tokens
                        ]
                    self.decode_cuda_graph_metadata[bs] = metadata
                else:

                    metadata.cache_seqlens_int32 = (
                        self.draft_decode_metadata_topk_normal["cache_seqlens"][:bs]
                    )
                    metadata.max_seq_len_q = self.topk
                    metadata.cu_seqlens_q = self.draft_decode_metadata_topk_normal[
                        "cu_seqlens_q"
                    ][: bs + 1]
                    metadata.cu_seqlens_k = self.draft_decode_metadata_topk_normal[
                        "cu_seqlens_k"
                    ][: bs + 1]
                    metadata.page_table = self.draft_decode_metadata_topk_normal[
                        "page_table"
                    ][:bs, :]

                    metadata_expand.cache_seqlens_int32 = (
                        self.draft_decode_metadata_topk_expand["cache_seqlens"][
                            : bs * self.topk
                        ]
                    )
                    metadata_expand.max_seq_len_q = 1
                    metadata_expand.cu_seqlens_q = (
                        self.draft_decode_metadata_topk_expand["cu_seqlens_q"][
                            : bs * self.topk + 1
                        ]
                    )
                    metadata_expand.cu_seqlens_k = (
                        self.draft_decode_metadata_topk_expand["cu_seqlens_k"][
                            : bs * self.topk + 1
                        ]
                    )
                    metadata_expand.page_table = self.draft_decode_metadata_topk_expand[
                        "page_table"
                    ][: bs * self.topk]
                    self.draft_decode_metadata_topk_normal[bs] = metadata
                    self.draft_decode_metadata_topk_expand[bs] = metadata_expand
            else:

                metadata.cache_seqlens_int32 = self.decode_cuda_graph_metadata[
                    "cache_seqlens"
                ][:bs]
                metadata.cu_seqlens_q = self.decode_cuda_graph_metadata["cu_seqlens_q"][
                    : bs + 1
                ]
                metadata.cu_seqlens_k = self.decode_cuda_graph_metadata["cu_seqlens_k"][
                    : bs + 1
                ]
                metadata.page_table = self.decode_cuda_graph_metadata["page_table"][
                    :bs, :
                ]
                if self.is_prefill_aware_swa:
                    metadata.pa_swa_page_table = metadata.page_table
                    metadata.pa_swa_cache_seqlens = metadata.cache_seqlens_int32
                if self.use_sliding_window_kv_pool:
                    metadata.swa_page_table = self.decode_cuda_graph_metadata[
                        "swa_page_table"
                    ][:bs, :]
                    metadata.swa_out_cache_loc = self.cuda_graph_swa_out_cache_loc[
                        :num_tokens
                    ]
                self.decode_cuda_graph_metadata[bs] = metadata

        elif forward_mode.is_target_verify():
            if self.topk <= 1:
                metadata.cache_seqlens_int32 = self.target_verify_metadata[
                    "cache_seqlens"
                ][:bs]
                metadata.max_seq_len_q = self.speculative_num_draft_tokens
                metadata.cu_seqlens_q = self.target_verify_metadata["cu_seqlens_q"][
                    : bs + 1
                ]
                metadata.cu_seqlens_k = self.target_verify_metadata["cu_seqlens_k"][
                    : (bs + 1)
                ]
                metadata.page_table = self.target_verify_metadata["page_table"][:bs, :]
                if self.use_sliding_window_kv_pool:
                    metadata.swa_page_table = self.target_verify_metadata[
                        "swa_page_table"
                    ][:bs, :]
                    metadata.swa_out_cache_loc = self.cuda_graph_swa_out_cache_loc[
                        :num_tokens
                    ]
                self.target_verify_metadata[bs] = metadata
                if spec_info is not None and spec_info.is_request_forest_verify_input():
                    self._validate_request_forest_verify_support()
                    layout = spec_info.ragged_verify_layout
                    if (
                        layout is None
                        or layout.graph_num_tokens != num_tokens
                        or layout.bs != bs
                    ):
                        raise ValueError(
                            "Request Forest capture layout disagrees with its tier"
                        )
                    key = (num_tokens, bs)
                    self._request_forest_graph_ragged_layouts[key] = layout


                    if self._request_forest_prefix_workspace is None:
                        capacity, columns = self.target_verify_metadata[
                            "page_table"
                        ].shape
                        self._request_forest_prefix_workspace = (
                            torch.zeros(capacity + 1, dtype=torch.int32, device=device),
                            torch.zeros(capacity + 2, dtype=torch.int32, device=device),
                            torch.zeros(capacity + 2, dtype=torch.int32, device=device),
                            torch.zeros(
                                (capacity + 1, columns),
                                dtype=metadata.page_table.dtype,
                                device=device,
                            ),
                        )
                    lens, qo, ko, pages = self._request_forest_prefix_workspace
                    metadata.cache_seqlens_int32 = lens[: bs + 1]
                    metadata.cu_seqlens_q = qo[: bs + 2]
                    metadata.cu_seqlens_q[: bs + 1].copy_(spec_info.request_offsets)
                    metadata.cu_seqlens_q[-1] = num_tokens
                    metadata.cu_seqlens_k = ko[: bs + 2]
                    metadata.page_table = pages[: bs + 1]
                    self.target_verify_metadata[key] = metadata
                    metadata.max_seq_len_q = num_tokens
                    metadata_expand = self._bind_request_forest_graph_tree_metadata(
                        num_tokens
                    )
            else:

                metadata.cache_seqlens_int32 = self.target_verify_metadata_topk_normal[
                    "cache_seqlens"
                ][:bs]
                metadata.max_seq_len_q = self.speculative_num_draft_tokens
                metadata.cu_seqlens_q = self.target_verify_metadata_topk_normal[
                    "cu_seqlens_q"
                ][: bs + 1]
                metadata.cu_seqlens_k = self.target_verify_metadata_topk_normal[
                    "cu_seqlens_k"
                ][: bs + 1]
                metadata.page_table = self.target_verify_metadata_topk_normal[
                    "page_table"
                ][:bs, :]

                metadata_expand.cache_seqlens_int32 = (
                    self.target_verify_metadata_topk_expand["cache_seqlens"][
                        : bs * self.speculative_num_draft_tokens
                    ]
                )
                metadata_expand.max_seq_len_q = 1
                metadata_expand.cu_seqlens_q = self.target_verify_metadata_topk_expand[
                    "cu_seqlens_q"
                ][: bs * self.speculative_num_draft_tokens + 1]
                metadata_expand.cu_seqlens_k = self.target_verify_metadata_topk_expand[
                    "cu_seqlens_k"
                ][: bs * self.speculative_num_draft_tokens + 1]
                metadata_expand.page_table = self.target_verify_metadata_topk_expand[
                    "page_table"
                ][: bs * self.speculative_num_draft_tokens]

                self.target_verify_metadata_topk_normal[bs] = metadata
                self.target_verify_metadata_topk_expand[bs] = metadata_expand


                if self.use_sliding_window_kv_pool:
                    metadata.swa_out_cache_loc = self.cuda_graph_swa_out_cache_loc[
                        :num_tokens
                    ]

                if self.has_swa:
                    metadata_swa = FlashAttentionMetadata()
                    metadata_swa.cache_seqlens_int32 = (
                        self.target_verify_metadata_topk_swa["cache_seqlens"][
                            : bs * self.speculative_num_draft_tokens
                        ]
                    )
                    metadata_swa.max_seq_len_q = 1
                    metadata_swa.cu_seqlens_q = self.target_verify_metadata_topk_swa[
                        "cu_seqlens_q"
                    ][: bs * self.speculative_num_draft_tokens + 1]
                    metadata_swa.cu_seqlens_k = self.target_verify_metadata_topk_swa[
                        "cu_seqlens_k"
                    ][: bs * self.speculative_num_draft_tokens + 1]
                    metadata_swa.page_table = self.target_verify_metadata_topk_swa[
                        "page_table"
                    ][: bs * self.speculative_num_draft_tokens]
                    self.target_verify_metadata_topk_swa[bs] = metadata_swa
                    metadata.swa_spec_metadata = metadata_swa

        elif forward_mode.is_draft_extend_v2():
            num_tokens_per_req = spec_info.num_tokens_per_req
            metadata.cache_seqlens_int32 = self.draft_extend_metadata["cache_seqlens"][
                :bs
            ]
            metadata.max_seq_len_q = num_tokens_per_req
            metadata.cu_seqlens_q = self.draft_extend_metadata["cu_seqlens_q"][: bs + 1]
            metadata.cu_seqlens_k = self.draft_extend_metadata["cu_seqlens_k"][
                : (bs + 1)
            ]
            metadata.page_table = self.draft_extend_metadata["page_table"][:bs, :]
            if self.use_sliding_window_kv_pool:
                metadata.swa_page_table = self.draft_extend_metadata["swa_page_table"][
                    :bs, :
                ]
                metadata.swa_out_cache_loc = self.cuda_graph_swa_out_cache_loc[
                    :num_tokens
                ]
            self.draft_extend_metadata[bs] = metadata

        if encoder_lens is not None:
            encoder_bs = encoder_lens.numel()
            metadata.encoder_lens_int32 = self.encoder_metadata["encoder_lens_int32"][
                :encoder_bs
            ]
            metadata.encoder_cu_seqlens_k = self.encoder_metadata[
                "encoder_cu_seqlens_k"
            ][: (encoder_bs + 1)]
            metadata.encoder_page_table = self.encoder_metadata["encoder_page_table"][
                :bs, :
            ]

        return metadata, metadata_expand

    @property
    def verify_mask(self) -> Optional[VerifyMask]:
        return self._verify_mask

    @staticmethod
    def _host_max_seq_len(
        seq_lens_cpu: Optional[torch.Tensor], seq_lens: torch.Tensor
    ) -> int:

        src = seq_lens_cpu if seq_lens_cpu is not None else seq_lens.cpu()
        return src.max().item()

    def _apply_cuda_graph_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        out_cache_loc: Optional[torch.Tensor] = None,
    ):

        seq_lens = seq_lens[:bs]


        seq_lens_cpu = seq_lens_cpu[:bs] if seq_lens_cpu is not None else None
        req_pool_indices = req_pool_indices[:bs]
        device = seq_lens.device
        metadata = None
        metadata_expand = None


        if self.use_sliding_window_kv_pool and out_cache_loc is not None:
            n = out_cache_loc.shape[0]
            self.cuda_graph_swa_out_cache_loc[n:].zero_()
            self.cuda_graph_swa_out_cache_loc[:n].copy_(
                self.token_to_kv_pool.translate_loc_from_full_to_swa(out_cache_loc)
            )

        if forward_mode.is_decode_or_idle():
            if spec_info is not None:

                if self.topk <= 1:

                    metadata = self.decode_cuda_graph_metadata[bs]


                    normal_decode_set_metadata(
                        metadata.cache_seqlens_int32,
                        metadata.cu_seqlens_k,
                        metadata.page_table,
                        self.req_to_token,
                        req_pool_indices,
                        self.decode_cuda_graph_metadata["strided_indices"],
                        self.max_num_pages,
                        seq_lens,
                        self.speculative_step_id + 1,
                        self.page_size,
                        metadata.swa_page_table,
                        (
                            self.token_to_kv_pool
                            if self.use_sliding_window_kv_pool
                            else None
                        ),
                        v2p_page_table=(
                            self._unified_hooks.v2p_page_table
                            if self._unified_dense
                            else None
                        ),
                        kernel_page_multiplier=self._unified_hooks.kernel_page_multiplier,
                    )

                else:


                    metadata = self.draft_decode_metadata_topk_normal[bs]
                    if self.page_size > 1:

                        last_page_lens = seq_lens % self.page_size
                        seq_lens = seq_lens - last_page_lens
                    metadata.cache_seqlens_int32.copy_(seq_lens)


                    metadata.max_seq_len_k = (
                        seq_lens_cpu.max().item()
                        if seq_lens_cpu is not None
                        else self.max_context_len
                    )
                    max_seq_pages = (
                        metadata.max_seq_len_k + self.page_size - 1
                    ) // self.page_size
                    strided_indices = self.decode_cuda_graph_metadata["strided_indices"]
                    strided_indices = strided_indices[:max_seq_pages]
                    page_table = (
                        self.req_to_token[
                            req_pool_indices[:, None],
                            strided_indices[None, :],
                        ]
                        // self.page_size
                    )
                    metadata.page_table[:, :max_seq_pages].copy_(page_table)

                    metadata_expand = self.draft_decode_metadata_topk_expand[bs]
                    decode_length = self.speculative_step_id + 1

                    cache_loc = out_cache_loc.view(-1, self.speculative_num_steps)
                    if self.page_size > 1:


                        cache_loc = cache_loc[:, :decode_length]
                        assert_buffer_fits(
                            cache_loc.shape[1],
                            metadata_expand.page_table.shape[1],
                            "draft expand page_table (width decode_length + 1)",
                        )
                        draft_decode_set_expand_metadata(
                            cache_seqlens_int32=metadata_expand.cache_seqlens_int32,
                            page_table=metadata_expand.page_table,
                            last_page_lens=last_page_lens,
                            decode_length=decode_length,
                            cache_loc=cache_loc,
                            topk=self.topk,
                            page_size=self.page_size,
                        )
                    else:
                        num_seqs = cache_loc.shape[0]
                        metadata_expand.page_table[:num_seqs, :decode_length].copy_(
                            cache_loc[:, :decode_length]
                        )

            else:

                metadata = self.decode_cuda_graph_metadata[bs]
                if self.is_prefill_aware_swa:


                    max_len = self._host_max_seq_len(seq_lens_cpu, seq_lens)
                    metadata.max_seq_len_k = max_len
                    pa_max_len = min(
                        self._pa_swa_max_prefill_len + self.sliding_window_size,
                        max_len,
                    )
                    if pa_max_len > 0:
                        _build_pa_page_table(
                            self.req_to_token,
                            req_pool_indices,
                            seq_lens,
                            self._pa_swa_prefill_lens,
                            self.sliding_window_size,
                            bs,
                            pa_max_len,
                            device,
                            dst_page_table=metadata.page_table,
                            dst_kv_lens=metadata.cache_seqlens_int32,
                        )
                else:


                    metadata.max_seq_len_k = (
                        seq_lens_cpu.max().item()
                        if seq_lens_cpu is not None
                        else self.max_context_len
                    )
                    normal_decode_set_metadata(
                        metadata.cache_seqlens_int32,
                        metadata.cu_seqlens_k,
                        metadata.page_table,
                        self.req_to_token,
                        req_pool_indices,
                        self.decode_cuda_graph_metadata["strided_indices"],
                        self.max_num_pages,
                        seq_lens,
                        0,
                        self.page_size,
                        metadata.swa_page_table,
                        (
                            self.token_to_kv_pool
                            if self.use_sliding_window_kv_pool
                            else None
                        ),
                        v2p_page_table=(
                            self._unified_hooks.v2p_page_table
                            if self._unified_dense
                            else None
                        ),
                        kernel_page_multiplier=self._unified_hooks.kernel_page_multiplier,
                    )

                self._maybe_update_local_attn_metadata_for_replay(
                    metadata,
                    bs,
                )


                if (
                    self._sched_meta_buf is not None
                    and metadata.scheduler_metadata is not None
                ):
                    sched = self._compute_scheduler_metadata(
                        bs,
                        self._cuda_graph_scheduler_max_seq_len_k(metadata),
                        metadata.cache_seqlens_int32,
                        metadata.cu_seqlens_q,
                    )
                    if sched is not None:
                        n = sched.shape[0]
                        self._sched_meta_buf[:n] = sched
                        self._sched_meta_buf[n:] = 0

        elif forward_mode.is_target_verify():
            if self.topk <= 1:
                if spec_info is not None and spec_info.is_request_forest_verify_input():
                    metadata, metadata_expand = (
                        self._refill_request_forest_graph_metadata(
                            bs=bs,
                            num_tokens=spec_info.physical_rows,
                            req_pool_indices=req_pool_indices,
                            seq_lens=seq_lens,
                            spec_info=spec_info,
                            out_cache_loc=out_cache_loc,
                        )
                    )
                else:
                    metadata = self.target_verify_metadata[bs]
                    ragged_layout = getattr(spec_info, "ragged_verify_layout", None)
                    if ragged_layout is not None:
                        padded = ragged_layout.padded_to_bucket(padded_bs=bs)
                        geometry = build_ragged_target_verify_geometry(
                            seq_lens=seq_lens, layout=padded
                        )
                        metadata.cache_seqlens_int32.copy_(geometry.cache_seqlens_int32)
                        metadata.cu_seqlens_q.copy_(geometry.cu_seqlens_q)
                    else:
                        metadata.cache_seqlens_int32.copy_(
                            (seq_lens + self.speculative_num_draft_tokens)
                        )


                    metadata.cu_seqlens_k[1:].copy_(
                        torch.cumsum(
                            metadata.cache_seqlens_int32,
                            dim=0,
                            dtype=torch.int32,
                        )
                    )
                    has_swa = self.use_sliding_window_kv_pool
                    build_trtllm_mha_page_table(
                        req_to_token=self.req_to_token,
                        req_pool_indices=req_pool_indices,
                        cache_seqlens=metadata.cache_seqlens_int32,
                        page_table=metadata.page_table,
                        page_size=self.page_size,
                        swa_page_table=(metadata.swa_page_table if has_swa else None),
                        full_to_swa=(
                            self.token_to_kv_pool.full_to_swa_index_mapping
                            if has_swa
                            else None
                        ),
                    )
            else:


                metadata = self.target_verify_metadata_topk_normal[bs]
                metadata.cache_seqlens_int32.copy_(seq_lens)


                metadata.max_seq_len_k = self.max_context_len

                metadata.cu_seqlens_k[1:].copy_(
                    torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32)
                )
                build_trtllm_mha_page_table(
                    req_to_token=self.req_to_token,
                    req_pool_indices=req_pool_indices,
                    cache_seqlens=metadata.cache_seqlens_int32,
                    page_table=metadata.page_table,
                    page_size=self.page_size,
                )


                metadata_expand = self.target_verify_metadata_topk_expand[bs]


                offsets = torch.arange(
                    self.speculative_num_draft_tokens, device=device
                ).unsqueeze(
                    0
                )

                cols = offsets.expand(seq_lens.numel(), -1) + seq_lens.unsqueeze(1)
                cum_len = torch.nn.functional.pad(
                    torch.cumsum(
                        (
                            seq_lens + self.speculative_num_draft_tokens
                        ).repeat_interleave(self.speculative_num_draft_tokens),
                        dim=0,
                    ),
                    (1, 0),
                )[:-1]
                mask_extraction_indices = (
                    cols.repeat_interleave(self.speculative_num_draft_tokens, dim=0)
                    + cum_len[:, None]
                ).view(1, -1)

                mask_extraction_indices[
                    :,
                    spec_info.positions.numel() * self.speculative_num_draft_tokens :,
                ].fill_(0)
                mask = spec_info.custom_mask[mask_extraction_indices].view(
                    -1, self.speculative_num_draft_tokens
                )

                col_indices = offsets.expand(
                    mask.shape[0], self.speculative_num_draft_tokens
                )
                keys = torch.where(
                    mask,
                    col_indices,
                    col_indices + self.speculative_num_draft_tokens,
                )
                _, sort_order = torch.sort(keys, dim=1)

                non_masked_page_table = (
                    self.req_to_token[req_pool_indices, :]
                    .gather(1, cols)
                    .repeat_interleave(self.speculative_num_draft_tokens, dim=0)
                )

                metadata_expand.page_table.copy_(
                    non_masked_page_table.gather(1, sort_order)
                )
                metadata_expand.cache_seqlens_int32.copy_(mask.sum(dim=1))
                metadata_expand.cu_seqlens_k[1:].copy_(
                    torch.cumsum(
                        metadata_expand.cache_seqlens_int32,
                        dim=0,
                        dtype=torch.int32,
                    )
                )
                if self.has_swa:
                    metadata_swa = self.target_verify_metadata_topk_swa[bs]
                    self._init_sliding_window_attn_spec_metadata(
                        metadata, metadata_expand, metadata_swa
                    )

        elif forward_mode.is_draft_extend_v2():
            metadata = self.draft_extend_metadata[bs]
            metadata.cache_seqlens_int32.copy_(seq_lens)


            metadata.max_seq_len_k = (
                seq_lens_cpu.max().item()
                if seq_lens_cpu is not None
                else self.max_context_len
            )
            metadata.cu_seqlens_k[1:].copy_(
                torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32)
            )

            extend_seq_lens_tensor = getattr(spec_info, "extend_seq_lens_tensor", None)
            extend_seq_lens_cpu = getattr(spec_info, "extend_seq_lens_cpu", None)
            if extend_seq_lens_tensor is not None:
                extend_seq_lens = extend_seq_lens_tensor.to(torch.int32)
            elif extend_seq_lens_cpu is not None:
                extend_seq_lens = torch.as_tensor(
                    extend_seq_lens_cpu,
                    dtype=torch.int32,
                    device=device,
                )
            else:
                default_extend = spec_info.num_tokens_per_req
                extend_seq_lens = torch.full(
                    (bs,), default_extend, dtype=torch.int32, device=device
                )
                extend_seq_lens_cpu = [default_extend] * bs

            if extend_seq_lens_cpu:
                metadata.max_seq_len_q = int(max(extend_seq_lens_cpu))
            else:
                metadata.max_seq_len_q = spec_info.num_tokens_per_req

            metadata.cu_seqlens_q[1:].copy_(
                torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32)
            )

            max_seq_pages = (
                metadata.max_seq_len_k + self.page_size - 1
            ) // self.page_size
            page_indices = self.req_to_token[
                req_pool_indices[:, None],
                self.draft_extend_metadata["strided_indices"][:max_seq_pages],
            ]
            if self.use_sliding_window_kv_pool and metadata.swa_page_table is not None:
                swa_page_indices = self.token_to_kv_pool.translate_loc_from_full_to_swa(
                    page_indices
                )
                metadata.swa_page_table[:, :max_seq_pages].copy_(
                    swa_page_indices // self.page_size
                )
            metadata.page_table[:, :max_seq_pages].copy_(page_indices // self.page_size)

        else:
            raise ValueError(
                f"FA3 `_apply_cuda_graph_metadata` only supports the modes the "
                f"full cuda-graph runner captures (decode / idle / target_verify "
                f"/ draft_extend / draft_extend_v2). Got {forward_mode=}. "
                f"Piecewise / breakable capture must route through "
                f"`init_forward_metadata(fb)` (the eager entry) instead of "
                f"`init_forward_metadata_out_graph(fb, in_capture=True)`."
            )

        if encoder_lens is not None:

            metadata.encoder_max_seq_len_k = int(encoder_lens.max().item())
            metadata.encoder_lens_int32[:bs].copy_(encoder_lens[:bs].to(torch.int32))
            metadata.encoder_cu_seqlens_k[1 : bs + 1].copy_(
                torch.cumsum(metadata.encoder_lens_int32[:bs], dim=0, dtype=torch.int32)
            )

            metadata.encoder_page_table[:bs, : metadata.encoder_max_seq_len_k].copy_(
                self.req_to_token[req_pool_indices, : metadata.encoder_max_seq_len_k]
            )


            text_max = metadata.max_seq_len_k
            arange_text = torch.arange(text_max, device=req_pool_indices.device)
            text_col = encoder_lens[:bs].long().unsqueeze(1) + arange_text.unsqueeze(0)
            text_row = req_pool_indices.unsqueeze(1).expand(-1, text_max)
            metadata.page_table[:bs, :text_max].copy_(
                self.req_to_token[text_row, text_col]
            )

        self.forward_metadata = metadata
        self.forward_metadata_spec_decode_expand = metadata_expand

    def get_cuda_graph_seq_len_fill_value(self):

        return 1

    def _maybe_init_local_attn_metadata(
        self,
        forwardbatch: ForwardBatch,
        metadata: FlashAttentionMetadata,
        device,
    ):

        if not self.has_local_attention:
            metadata.local_attn_metadata = None
            return

        cu_seqlens_q = metadata.cu_seqlens_q
        cache_seqlens_int32 = metadata.cache_seqlens_int32
        if self.use_sliding_window_kv_pool:
            page_table = self.token_to_kv_pool.translate_loc_from_full_to_swa(
                metadata.page_table
            ).to(torch.int32)
        else:
            page_table = metadata.page_table
        if cu_seqlens_q is None or cache_seqlens_int32 is None or page_table is None:
            metadata.local_attn_metadata = None
            return

        cu_seqlens_q_np = cu_seqlens_q.cpu().numpy()
        seq_lens_np = cache_seqlens_int32.cpu().numpy()
        (
            seqlens_q_local_np,
            cu_seqlens_q_local_np,
            seqlens_k_local_np,
            block_table_local,
        ) = make_local_attention_virtual_batches(
            self.attention_chunk_size,
            cu_seqlens_q_np,
            seq_lens_np,
            page_table,
            self.page_size,
        )

        local_metadata = FlashAttentionMetadata.LocalAttentionMetadata(
            local_query_start_loc=torch.from_numpy(cu_seqlens_q_local_np).to(device),
            local_seqused_k=torch.from_numpy(seqlens_k_local_np).to(device),
            local_block_table=block_table_local.to(device),
            local_max_query_len=int(seqlens_q_local_np.max()),
            local_max_seq_len=int(seqlens_k_local_np.max()),
        )
        metadata.local_attn_metadata = local_metadata

    def _maybe_update_local_attn_metadata_for_capture(
        self, metadata: FlashAttentionMetadata, bs: int
    ):

        if not self.has_local_attention:
            return

        seq_lens_capture = metadata.cache_seqlens_int32
        max_seq_len = int(seq_lens_capture.max().item())
        page_table_capture = metadata.page_table

        cu_seqlens_q_np = metadata.cu_seqlens_q.cpu().numpy()
        seqlens_np = seq_lens_capture.cpu().numpy()
        (
            seqlens_q_local_np,
            cu_seqlens_q_local_np,
            seqlens_k_local_np,
            block_table_local_np,
        ) = make_local_attention_virtual_batches(
            self.attention_chunk_size,
            cu_seqlens_q_np,
            seqlens_np,
            page_table_capture,
            self.page_size,
        )


        q_len = len(cu_seqlens_q_local_np)
        k_len = len(seqlens_k_local_np)
        b0 = block_table_local_np.shape[0] if block_table_local_np.shape[0] > 0 else bs
        b1 = block_table_local_np.shape[1] if block_table_local_np.shape[1] > 0 else 1


        local_query_start_loc = self.decode_cuda_graph_local_attn_metadata[
            "local_query_start_loc"
        ][:q_len]

        local_seqused_k = self.decode_cuda_graph_local_attn_metadata["local_seqused_k"][
            :k_len
        ]

        local_block_table = self.decode_cuda_graph_local_attn_metadata[
            "local_block_table"
        ][:b0, :b1]

        metadata.local_attn_metadata = FlashAttentionMetadata.LocalAttentionMetadata(
            local_query_start_loc=local_query_start_loc,
            local_seqused_k=local_seqused_k,
            local_block_table=local_block_table,
            local_max_query_len=1,
            local_max_seq_len=max_seq_len,
        )

    def _maybe_update_local_attn_metadata_for_replay(
        self,
        metadata: FlashAttentionMetadata,
        bs: int,
    ):

        if not self.has_local_attention:
            return


        local_q_buf = self.decode_cuda_graph_local_attn_metadata[
            "local_query_start_loc"
        ]
        local_k_buf = self.decode_cuda_graph_local_attn_metadata["local_seqused_k"]
        local_block_buf = self.decode_cuda_graph_local_attn_metadata[
            "local_block_table"
        ]
        cu_seqlens_q = self.decode_cuda_graph_metadata["cu_seqlens_q"]


        cu_seqlens_q = torch.arange(
            bs + 1, device=cu_seqlens_q.device, dtype=cu_seqlens_q.dtype
        )
        seqlens = metadata.cache_seqlens_int32[:bs]


        max_seq_len = int(seqlens.max().item())
        if self.use_sliding_window_kv_pool:
            sliced_page_table = self.token_to_kv_pool.translate_loc_from_full_to_swa(
                metadata.page_table[:bs, :max_seq_len]
            ).to(torch.int32)
        else:
            sliced_page_table = metadata.page_table[:bs, :max_seq_len]

        cu_seqlens_q_np = cu_seqlens_q.cpu().numpy()
        seqlens_np = seqlens.cpu().numpy()
        (
            seqlens_q_local_np,
            cu_seqlens_q_local_np,
            seqlens_k_local_np,
            block_table_local,
        ) = make_local_attention_virtual_batches(
            self.attention_chunk_size,
            cu_seqlens_q_np,
            seqlens_np,
            sliced_page_table,
            self.page_size,
        )


        device = local_q_buf.device
        cu_seqlens_q_local = torch.from_numpy(cu_seqlens_q_local_np).to(device)
        seqlens_k_local = torch.from_numpy(seqlens_k_local_np).to(device)
        block_table_local = block_table_local.to(device)

        q_len = cu_seqlens_q_local.shape[0]
        k_len = seqlens_k_local.shape[0]
        b0, b1 = block_table_local.shape


        local_q_buf[:q_len].copy_(cu_seqlens_q_local)
        local_q_buf[q_len:].fill_(0)
        local_k_buf[:k_len].copy_(seqlens_k_local)
        local_k_buf[k_len:].fill_(0)
        local_block_buf[:b0, :b1].copy_(block_table_local)
        local_block_buf[b0:, :].fill_(0)
        local_block_buf[:b0, b1:].fill_(0)

        if metadata.local_attn_metadata is not None:
            lam = metadata.local_attn_metadata
            lam.local_max_query_len = int(seqlens_q_local_np.max())
            lam.local_max_seq_len = int(seqlens_k_local_np.max())

    def _init_sliding_window_attn_spec_metadata(
        self,
        metadata: FlashAttentionMetadata,
        metadata_expand: FlashAttentionMetadata,
        metadata_swa: Optional[FlashAttentionMetadata] = None,
    ):

        assert (
            self.page_size == 1
        ), "FlashAttention backend doesn't support topk > 1 speculative decoding with page size > 1 sliding window attention"

        cache_seqlens_int32 = (
            metadata.cache_seqlens_int32.repeat_interleave(
                self.speculative_num_draft_tokens
            )
            + metadata_expand.cache_seqlens_int32
        )
        cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(cache_seqlens_int32, dim=0, dtype=torch.int32), (1, 0)
        )
        bs = cache_seqlens_int32.shape[0]
        page_table = (
            metadata.page_table.new_zeros(
                (bs, metadata.max_seq_len_k + metadata_expand.page_table.shape[1])
            )
            if metadata_swa is None
            else metadata_swa.page_table
        )
        assert_buffer_fits(
            metadata.max_seq_len_k + metadata_expand.page_table.shape[1],
            page_table.shape[1],
            "FA3 swa-spec page_table",
        )

        page_table_a = metadata.page_table
        page_table_b = metadata_expand.page_table
        if self.use_sliding_window_kv_pool:
            page_table_a = self.token_to_kv_pool.translate_loc_from_full_to_swa(
                page_table_a
            ).to(torch.int32)
            page_table_b = self.token_to_kv_pool.translate_loc_from_full_to_swa(
                page_table_b
            ).to(torch.int32)

        prepare_swa_spec_page_table_triton(
            page_table,
            page_table_a,
            page_table_b,
            metadata.cache_seqlens_int32,
            metadata_expand.cache_seqlens_int32,
            self.speculative_num_draft_tokens,
        )

        if metadata_swa is None:
            metadata_swa = FlashAttentionMetadata()
            metadata_swa.max_seq_len_q = 1
            metadata_swa.cu_seqlens_q = metadata_expand.cu_seqlens_q
            metadata_swa.cache_seqlens_int32 = cache_seqlens_int32
            metadata_swa.cu_seqlens_k = cu_seqlens_k
            metadata_swa.page_table = page_table
        else:
            metadata_swa.cache_seqlens_int32.copy_(cache_seqlens_int32)
            metadata_swa.cu_seqlens_k.copy_(cu_seqlens_k)

        metadata.swa_spec_metadata = metadata_swa


class FlashAttentionMultiStepBackend:


    needs_cpu_seq_lens: bool = False

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
        fa_impl_ver: int = 3,
    ):
        self.model_runner = model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                FlashAttentionBackend(
                    model_runner,
                    speculative_step_id=i,
                    topk=self.topk,
                    speculative_num_steps=self.speculative_num_steps,
                    fa_impl_ver=fa_impl_ver,
                )
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
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata_in_graph(forward_batch)


@torch.compile(dynamic=True, backend=get_compiler_backend())
def draft_decode_set_expand_metadata(
    cache_seqlens_int32: torch.Tensor,
    page_table: torch.Tensor,
    last_page_lens: torch.Tensor,
    decode_length: int,
    cache_loc: torch.Tensor,
    topk: int,
    page_size: int,
):
    expanded_last_page_lens = last_page_lens.repeat_interleave(topk)
    cache_seqlens_int32.copy_(decode_length + expanded_last_page_lens)
    cache_loc = (cache_loc // page_size).to(torch.int32)
    if cache_loc.dim() == 1:
        cache_loc = cache_loc.unsqueeze(0)


    mask = torch.ones_like(cache_loc, dtype=torch.bool)
    mask[:, 1:] = cache_loc[:, 1:] != cache_loc[:, :-1]
    positions = mask.cumsum(dim=1) - 1
    num_seqs = cache_loc.shape[0]
    page_table[:num_seqs, :].scatter_(1, positions, cache_loc)


def make_local_attention_virtual_batches(
    attn_chunk_size: int,
    query_start_loc_np: np.ndarray,
    seq_lens_np: np.ndarray,
    block_table: torch.Tensor,
    page_size: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:


    max_seq_len = seq_lens_np.max()
    effective_chunk_size = min(attn_chunk_size, max_seq_len)

    effective_chunk_size = (effective_chunk_size // page_size) * page_size
    if effective_chunk_size < page_size:
        effective_chunk_size = page_size
    attn_chunk_size = effective_chunk_size

    q_seqlens = query_start_loc_np[1:] - query_start_loc_np[:-1]
    actual_batch_size = seq_lens_np.shape[0]


    q_tokens_in_first_block = np.minimum(
        attn_chunk_size - ((seq_lens_np - q_seqlens) % attn_chunk_size), q_seqlens
    ).astype(np.int32)
    tokens_in_last_block = attn_chunk_size + (seq_lens_np % -attn_chunk_size)
    local_blocks = 1 + cdiv(q_seqlens - q_tokens_in_first_block, attn_chunk_size)


    cu_num_blocks = np.cumsum(local_blocks)
    virtual_batches = cu_num_blocks[-1]

    block_offsets = np.repeat(cu_num_blocks - local_blocks, local_blocks)

    arange = np.arange(virtual_batches, dtype=np.int32) - block_offsets

    rarange = np.repeat(local_blocks, local_blocks) - arange - 1


    seqlens_q_local = np.repeat(q_seqlens - q_tokens_in_first_block, local_blocks)

    seqlens_q_local[arange == 0] = q_tokens_in_first_block

    seqlens_q_local[arange > 0] = np.minimum(
        seqlens_q_local - attn_chunk_size * (arange - 1), attn_chunk_size
    )[arange > 0]


    cu_seqlens_q_local = np.pad(np.cumsum(seqlens_q_local), (1, 0)).astype(np.int32)


    seqlens_k_local = np.full(cu_num_blocks[-1], attn_chunk_size, dtype=np.int32)
    seqlens_k_local[cu_num_blocks - 1] = tokens_in_last_block

    k_seqstarts_absolute = np.repeat(seq_lens_np, local_blocks) - (
        rarange * attn_chunk_size + np.repeat(tokens_in_last_block, local_blocks)
    )


    block_starts = k_seqstarts_absolute // page_size

    assert attn_chunk_size % page_size == 0, (
        f"attn_chunk_size {attn_chunk_size} is not "
        f"divisible by page_size {page_size}"
    )
    pages_per_local_batch = attn_chunk_size // page_size


    block_indices = np.broadcast_to(
        np.arange(pages_per_local_batch, dtype=np.int32),
        (virtual_batches, pages_per_local_batch),
    ) + np.expand_dims(block_starts, axis=1)


    block_indices = block_indices.flatten().clip(max=block_table.shape[1] - 1)
    batch_indices = np.repeat(
        np.arange(actual_batch_size, dtype=np.int32),
        local_blocks * pages_per_local_batch,
    )


    batch_indices_torch = torch.from_numpy(batch_indices)
    block_indices_torch = torch.from_numpy(block_indices)
    block_table_local = block_table[batch_indices_torch, block_indices_torch].view(
        virtual_batches, -1
    )

    return seqlens_q_local, cu_seqlens_q_local, seqlens_k_local, block_table_local


def cdiv(a: int, b: int) -> int:

    return -(a // -b)


@torch._dynamo.disable()
def merge_state_v2_wrapper(o, s_a, o_exp, s_b):
    return merge_state_v2(o, s_a, o_exp, s_b)
