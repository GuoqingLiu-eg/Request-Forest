from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import torch
import triton

from sglang.kernels.ops.attention.fixup_zero_kv import fixup_zero_kv_rows
from sglang.kernels.ops.attention.pad import (
    pad_draft_extend_query as pad_draft_extend_query_triton,
)
from sglang.kernels.ops.attention.pad import (
    unpad_draft_extend_output as unpad_draft_extend_output_triton,
)
from sglang.kernels.ops.attention.set_mla_kv_concat_q import (
    can_use_set_mla_kv_concat_q,
    can_use_set_mla_kv_concat_q_fp8,
)
from sglang.kernels.ops.attention.set_mla_kv_concat_q import (
    covered as set_mla_kv_concat_q_covered,
)
from sglang.kernels.ops.attention.set_mla_kv_concat_q import (
    covered_fp8 as set_mla_kv_concat_q_fp8_covered,
)
from sglang.kernels.ops.attention.set_mla_kv_concat_q import (
    set_mla_kv_concat_q,
    set_mla_kv_concat_q_fp8,
)
from sglang.kernels.ops.attention.utils import (
    concat_mla_absorb_q_general,
    mla_quantize_and_rope_for_fp8,
    mla_quantize_without_rope_for_fp8,
)
from sglang.kernels.ops.kvcache.kv_indices import (
    create_flashmla_kv_indices_triton,
    get_num_kv_index_blocks_flashmla,
    get_num_page_per_block_flashmla,
)
from sglang.kernels.ops.quantization.fp8_kernel import scaled_fp8_quant
from sglang.srt.environ import envs
from sglang.srt.layers.attention.flashinfer_mla_backend import (
    FlashInferMLAAttnBackend,
    FlashInferMLAMultiStepDraftBackend,
)
from sglang.srt.layers.attention.unified_mem_hooks import unified_mla_hooks
from sglang.srt.layers.attention.verify_mask import VerifyMask, maybe_create_verify_mask
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.runtime_context import (
    get_buffer,
    get_parallel,
    get_schedule,
    get_spec,
)
from sglang.srt.utils import is_flashinfer_available, is_float4_e2m1fn_x2

if is_flashinfer_available():
    import flashinfer

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


DEFAULT_WORKSPACE_SIZE_MB = 150


TRTLLM_BLOCK_CONSTRAINT = 128

TRTLLM_MLA_MAX_BATCH_SIZE = 8192


def _multi_ctas_kv_counter_bytes(
    device: torch.device, num_q_heads: int, batch_size: int
) -> int:
    sm_count = flashinfer.utils.get_device_sm_count(device)
    return flashinfer.utils.get_trtllm_gen_multi_ctas_kv_counter_bytes(
        batch_size, num_q_heads, sm_count
    )


def make_persistent_multi_ctas_kv_counter_buffer(
    device: torch.device, num_q_heads: int, max_batch_size: int
) -> torch.Tensor:
    num_bytes = _multi_ctas_kv_counter_bytes(
        device, num_q_heads, max(TRTLLM_MLA_MAX_BATCH_SIZE, max_batch_size)
    )
    return torch.zeros(num_bytes, dtype=torch.uint8, device=device)


def grow_multi_ctas_kv_counter_buffer_if_needed(
    buffer: torch.Tensor, device: torch.device, num_q_heads: int, batch_size: int
) -> torch.Tensor:
    required_bytes = _multi_ctas_kv_counter_bytes(device, num_q_heads, batch_size)
    if buffer.numel() >= required_bytes:
        return buffer
    return torch.zeros(required_bytes, dtype=torch.uint8, device=device)


def _quantize_fp8_qkv(q, k, v, layer):
    q = q.to(torch.float8_e4m3fn)

    k_scale = getattr(layer, "k_scale_float", None)
    if k_scale is None:
        k_scale = 1.0
    if k_scale != 1.0:
        assert hasattr(layer, "k_scale"), "k_scale is not set"
        k_2d, _ = scaled_fp8_quant(
            k.reshape(-1, k.shape[-1]).contiguous(), layer.k_scale
        )
        k = k_2d.reshape(k.shape)
    else:
        k = k.to(torch.float8_e4m3fn)

    v_scale = getattr(layer, "v_scale_float", None)
    if v_scale is None:
        v_scale = 1.0
    if v_scale != 1.0:
        assert hasattr(layer, "v_scale"), "v_scale is not set"
        v_2d, _ = scaled_fp8_quant(
            v.reshape(-1, v.shape[-1]).contiguous(), layer.v_scale
        )
        v = v_2d.reshape(v.shape)
    else:
        v = v.to(torch.float8_e4m3fn)

    return q, k, v, k_scale, v_scale


global_cute_dsl_workspace_buffer = None


@dataclass
class TRTLLMMLAPrefillMetadata:


    max_seq_len: int
    cum_seq_lens: torch.Tensor
    seq_lens: torch.Tensor
    fallback_to_flashinfer_impl: bool = False


from sglang.kernels.jit.utils import is_arch_support_pdl


_ENABLE_PDL = is_arch_support_pdl()


@dataclass
class TRTLLMMLADecodeMetadata:


    block_kv_indices: Optional[torch.Tensor] = None
    max_seq_len_k: Optional[int] = None
    max_seq_len_q: Optional[int] = None
    sum_seq_lens_q: Optional[int] = None
    cu_seqlens_q: Optional[torch.Tensor] = None
    seq_lens_q: Optional[torch.Tensor] = None
    seq_lens_k: Optional[torch.Tensor] = None
    global_seq_lens_k: Optional[torch.Tensor] = None


class TRTLLMMLABackend(FlashInferMLAAttnBackend):


    needs_cpu_seq_lens: bool = False


    supports_ragged_verify_graph: bool = True

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        q_indptr_decode_buf: Optional[torch.Tensor] = None,
        backend: str = "trtllm-gen",
    ):
        super().__init__(
            model_runner,
            skip_prefill,
            kv_indptr_buf,
            q_indptr_decode_buf,
        )

        config = model_runner.model_config


        self.num_q_heads = config.num_attention_heads // get_parallel().attn_tp_size
        self.num_kv_heads = config.get_num_kv_heads(get_parallel().attn_tp_size)
        self.num_local_heads = config.num_attention_heads // get_parallel().attn_tp_size


        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_cache_dim = self.kv_lora_rank + self.qk_rope_head_dim


        self.backend = backend
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.page_size = model_runner.page_size
        self.req_to_token = model_runner.req_to_token_pool.req_to_token


        self.workspace_size = DEFAULT_WORKSPACE_SIZE_MB * 1024 * 1024
        if self.backend == "cute-dsl":


            global global_cute_dsl_workspace_buffer
            if global_cute_dsl_workspace_buffer is None:
                global_cute_dsl_workspace_buffer = torch.zeros(
                    self.workspace_size,
                    dtype=torch.int8,
                    device=model_runner.device,
                )
            self.workspace_buffer = global_cute_dsl_workspace_buffer
        else:
            self.workspace_buffer = get_buffer(
                "trtllm_mla_zero_workspace",
                lambda: torch.zeros(
                    self.workspace_size,
                    dtype=torch.int8,
                    device=model_runner.device,
                ),
            )

        self._multi_ctas_kv_counter_buffer = (
            make_persistent_multi_ctas_kv_counter_buffer(
                torch.device(self.device),
                self.num_q_heads,
                max_batch_size=model_runner.max_running_requests,
            )
        )


        self.decode_cuda_graph_metadata = {}
        self.decode_cuda_graph_kv_indices = None
        self.padded_q_buffer = None
        self.unpad_output_buffer = None
        self.forward_prefill_metadata: Optional[TRTLLMMLAPrefillMetadata] = None
        self.forward_decode_metadata: Union[TRTLLMMLADecodeMetadata, None] = None

        self.disable_chunked_prefix_cache = get_schedule().disable_chunked_prefix_cache

        self.num_draft_tokens = get_spec().speculative_num_draft_tokens
        self.dense_q_indptr_verify = (
            self.q_indptr_decode * self.num_draft_tokens
            if self.num_draft_tokens
            else None
        )
        self._verify_mask = None

        self.is_draft_runner = model_runner.is_draft_worker


        _hooks = unified_mla_hooks(model_runner.token_to_kv_pool_allocator)
        self._v2p_page_table = _hooks.v2p_page_table
        self._kernel_page_multiplier = _hooks.kernel_page_multiplier
        self._unified_mla = _hooks.enabled

        self._translate_kv_loc_dense = _hooks.translate_kv_loc_dense


        self._decode_dense_loc: Optional[torch.Tensor] = None
        self.cuda_graph_out_cache_loc_dense: Optional[torch.Tensor] = None


        self._fused_set_kv_concat_q = (
            self.data_type == torch.bfloat16
            and not envs.SGLANG_ENABLE_ASYNC_ASSERT.get()
            and can_use_set_mla_kv_concat_q(
                self.kv_lora_rank * 2, self.qk_rope_head_dim * 2
            )
        )


        self._fused_set_kv_concat_q_fp8 = (
            self.data_type == torch.float8_e4m3fn
            and not envs.SGLANG_ENABLE_ASYNC_ASSERT.get()
            and self.kv_lora_rank == 512
            and self.qk_rope_head_dim == 64
            and can_use_set_mla_kv_concat_q_fp8()
        )

    def _calc_padded_blocks(self, max_seq_len: int) -> int:

        blocks = triton.cdiv(max_seq_len, self.page_size)


        trtllm_constraint = TRTLLM_BLOCK_CONSTRAINT // self.page_size
        triton_constraint = get_num_page_per_block_flashmla(self.page_size)
        constraint_lcm = math.lcm(trtllm_constraint, triton_constraint)

        if blocks % constraint_lcm != 0:
            blocks = triton.cdiv(blocks, constraint_lcm) * constraint_lcm
        return blocks

    def _create_block_kv_indices(
        self,
        batch_size: int,
        max_blocks: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:

        block_kv_indices = torch.full(
            (batch_size, max_blocks), -1, dtype=torch.int32, device=device
        )

        create_flashmla_kv_indices_triton[
            (
                batch_size,
                get_num_kv_index_blocks_flashmla(max_blocks, self.page_size),
            )
        ](
            self.req_to_token,
            req_pool_indices,
            seq_lens,
            None,
            block_kv_indices,
            self.req_to_token.stride(0),
            max_blocks,
            PAGED_SIZE=self.page_size,
            v2p_ptr=self._v2p_page_table,
            PAGE_MULT=self._kernel_page_multiplier,
        )

        return block_kv_indices

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):


        max_blocks_per_seq = self._calc_padded_blocks(self.max_context_len)

        self.decode_cuda_graph_kv_indices = torch.full(
            (max_bs, max_blocks_per_seq), -1, dtype=torch.int32, device=self.device
        )


        if self._unified_mla:
            self.cuda_graph_out_cache_loc_dense = torch.zeros(
                max_num_tokens, dtype=torch.int64, device=self.device
            )
        num_tokens_per_req = max_num_tokens // max_bs

        if is_float4_e2m1fn_x2(self.data_type):

            self.store_dtype = torch.uint8
            self.padded_q_buffer = torch.zeros(
                (max_bs, num_tokens_per_req // 2, self.num_q_heads, self.kv_cache_dim),
                dtype=self.store_dtype,
                device=self.device,
            )


            self.unpad_output_buffer = torch.zeros(
                (max_num_tokens // 2, self.num_q_heads, 512),
                dtype=self.store_dtype,
                device=self.device,
            )
        else:

            self.padded_q_buffer = torch.zeros(
                (max_bs, num_tokens_per_req, self.num_q_heads, self.kv_cache_dim),
                dtype=self.data_type,
                device=self.device,
            )


            self.unpad_output_buffer = torch.zeros(
                (max_num_tokens, self.num_q_heads, 512),
                dtype=self.data_type,
                device=self.device,
            )


        self._verify_mask = maybe_create_verify_mask(
            is_draft_runner=self.is_draft_runner,
            skip_prefill=self.skip_prefill,
            max_bs=max_bs,
            max_context_len=self.max_context_len,
            num_draft_tokens=self.num_draft_tokens,
            device=self.device,
            is_read=False,
        )

        super().init_cuda_graph_state(max_bs, max_num_tokens, kv_indices_buf)

    @property
    def verify_mask(self) -> Optional[VerifyMask]:
        return self._verify_mask

    def _init_cuda_graph_metadata(
        self,
        bs: int,
        num_tokens: int,
        forward_mode: ForwardMode,
        seq_lens: torch.Tensor,
        device: torch.device,
    ):

        if forward_mode.is_target_verify() and bs in self.decode_cuda_graph_metadata:


            self.forward_decode_metadata = self.decode_cuda_graph_metadata[bs]
            return
        metadata = TRTLLMMLADecodeMetadata()

        if forward_mode.is_target_verify():
            metadata.seq_lens_k = torch.zeros((bs,), dtype=torch.int32, device=device)
            metadata.global_seq_lens_k = torch.zeros(
                (bs,), dtype=torch.int32, device=device
            )
        elif forward_mode.is_draft_extend_v2():
            num_tokens_per_req = self.num_draft_tokens
            metadata.max_seq_len_q = num_tokens_per_req
            metadata.sum_seq_lens_q = num_tokens_per_req * bs
            metadata.cu_seqlens_q = torch.arange(
                0,
                bs * num_tokens_per_req + 1,
                num_tokens_per_req,
                dtype=torch.int32,
                device=device,
            )
            metadata.seq_lens_q = torch.full(
                (bs,), num_tokens_per_req, dtype=torch.int32, device=device
            )
            metadata.seq_lens_k = torch.zeros((bs,), dtype=torch.int32, device=device)

        if metadata.seq_lens_k is None:


            metadata.seq_lens_k = torch.zeros((bs,), dtype=torch.int32, device=device)


        max_blocks_per_seq = self._calc_padded_blocks(self.max_context_len)
        block_kv_indices = self.decode_cuda_graph_kv_indices[:bs, :max_blocks_per_seq]
        metadata.block_kv_indices = block_kv_indices
        metadata.max_seq_len_k = self.max_context_len

        self.decode_cuda_graph_metadata[bs] = metadata
        self.forward_decode_metadata = metadata

    def _apply_cuda_graph_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
    ):

        metadata = self.decode_cuda_graph_metadata[bs]

        if forward_mode.is_target_verify():

            torch.add(
                seq_lens[:bs],
                self.num_draft_tokens,
                out=metadata.global_seq_lens_k,
            )
            metadata.seq_lens_k.copy_(metadata.global_seq_lens_k)
            seq_lens = metadata.seq_lens_k
        elif forward_mode.is_draft_extend_v2():
            num_tokens_per_req = self.num_draft_tokens
            metadata.max_seq_len_q = num_tokens_per_req
            metadata.sum_seq_lens_q = num_tokens_per_req * bs
            seq_lens = seq_lens[:bs]
            metadata.seq_lens_k.copy_(seq_lens)
        elif metadata.seq_lens_k is not None:


            metadata.seq_lens_k.copy_(seq_lens[:bs])


        create_flashmla_kv_indices_triton[
            (
                bs,
                get_num_kv_index_blocks_flashmla(
                    metadata.block_kv_indices.shape[1], self.page_size
                ),
            )
        ](
            self.req_to_token,
            req_pool_indices[:bs],
            seq_lens,
            None,
            metadata.block_kv_indices,
            self.req_to_token.stride(0),
            metadata.block_kv_indices.shape[1],
            PAGED_SIZE=self.page_size,
            v2p_ptr=self._v2p_page_table,
            PAGE_MULT=self._kernel_page_multiplier,
        )

    def get_cuda_graph_seq_len_fill_value(self) -> int:

        return 1

    def init_mha_chunk_metadata(
        self, forward_batch: ForwardBatch, disable_flashinfer_ragged: bool = False
    ) -> None:
        has_prefix = any(forward_batch.extend_prefix_lens_cpu)
        fallback_to_flashinfer_impl = (
            (self.disable_chunked_prefix_cache and has_prefix)
            or is_in_tc_piecewise_cuda_graph()
            or is_in_breakable_cuda_graph()
        )
        if fallback_to_flashinfer_impl:
            super().init_mha_chunk_metadata(
                forward_batch, disable_flashinfer_ragged=True
            )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        forward_mode = forward_batch.forward_mode

        if (
            not forward_mode.is_decode_or_idle()
            and not forward_mode.is_target_verify()
            and not forward_mode.is_draft_extend_v2()
        ):
            return super().init_forward_metadata_out_graph(
                forward_batch, in_capture=in_capture
            )

        bs = forward_batch.batch_size
        if in_capture:
            num_tokens = forward_batch.positions.numel()
            self._init_cuda_graph_metadata(
                bs,
                num_tokens,
                forward_mode,
                forward_batch.seq_lens,
                forward_batch.seq_lens.device,
            )
            self._apply_cuda_graph_metadata(
                bs=bs,
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
                forward_mode=forward_mode,
            )
        else:
            self._apply_cuda_graph_metadata(
                bs=bs,
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
                forward_mode=forward_mode,
            )


        if self._unified_mla and (
            forward_mode.is_decode_or_idle() or forward_mode.is_target_verify()
        ):
            out_cache_loc = forward_batch.out_cache_loc
            n = out_cache_loc.shape[0]
            dst = self.cuda_graph_out_cache_loc_dense[:n]
            self._translate_kv_loc_dense(out_cache_loc, out=dst)


            self.cuda_graph_out_cache_loc_dense[n:].zero_()
            self._decode_dense_loc = dst
        else:
            self._decode_dense_loc = None

    def init_forward_metadata(self, forward_batch: ForwardBatch):


        self._decode_dense_loc = None

        if (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_target_verify()
            and not forward_batch.forward_mode.is_draft_extend_v2()
        ):


            has_prefix = any(forward_batch.extend_prefix_lens_cpu)
            fallback_to_flashinfer_impl = (
                (self.disable_chunked_prefix_cache and has_prefix)
                or is_in_tc_piecewise_cuda_graph()
                or is_in_breakable_cuda_graph()
            )
            if fallback_to_flashinfer_impl:
                super().init_forward_metadata(forward_batch)

            seq_lens = forward_batch.seq_lens - forward_batch.extend_prefix_lens
            cum_seq_lens_q = torch.cat(
                (
                    torch.zeros(
                        1, dtype=torch.int32, device=forward_batch.seq_lens.device
                    ),
                    torch.cumsum(seq_lens, dim=0),
                )
            ).int()
            max_seq_len = max(forward_batch.extend_seq_lens_cpu)
            self.forward_prefill_metadata = TRTLLMMLAPrefillMetadata(
                max_seq_len,
                cum_seq_lens_q,
                seq_lens,
                fallback_to_flashinfer_impl,
            )
        elif (
            forward_batch.forward_mode.is_decode_or_idle()
            or forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            bs = forward_batch.batch_size
            self.forward_decode_metadata = TRTLLMMLADecodeMetadata()


            if (
                forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend_v2()
            ):
                self.forward_prefill_metadata = None


            if getattr(forward_batch, "seq_lens_cpu", None) is not None:
                max_seq = forward_batch.seq_lens_cpu.max().item()
            else:
                max_seq = self.max_context_len

            seq_lens = forward_batch.seq_lens

            if forward_batch.forward_mode.is_target_verify():
                max_seq = max_seq + self.num_draft_tokens
                seq_lens = seq_lens + self.num_draft_tokens
                self.forward_decode_metadata.seq_lens_k = seq_lens.to(torch.int32)
                self.forward_decode_metadata.global_seq_lens_k = (
                    self.forward_decode_metadata.seq_lens_k
                )
            elif forward_batch.forward_mode.is_decode_or_idle():


                self.forward_decode_metadata.seq_lens_k = seq_lens.to(torch.int32)
            elif forward_batch.forward_mode.is_draft_extend_v2():
                sum_seq_lens_q = sum(forward_batch.extend_seq_lens_cpu)
                max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)
                cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(
                        forward_batch.extend_seq_lens, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )

                seq_lens = seq_lens - forward_batch.extend_seq_lens + max_seq_len_q

                self.forward_decode_metadata.max_seq_len_q = max_seq_len_q
                self.forward_decode_metadata.sum_seq_lens_q = sum_seq_lens_q
                self.forward_decode_metadata.cu_seqlens_q = cu_seqlens_q
                self.forward_decode_metadata.seq_lens_q = forward_batch.extend_seq_lens
                self.forward_decode_metadata.seq_lens_k = seq_lens.to(torch.int32)

            max_seqlen_pad = self._calc_padded_blocks(max_seq)
            block_kv_indices = self._create_block_kv_indices(
                bs,
                max_seqlen_pad,
                forward_batch.req_pool_indices,
                seq_lens,
                seq_lens.device,
            )

            self.forward_decode_metadata.block_kv_indices = block_kv_indices
            self.forward_decode_metadata.max_seq_len_k = int(max_seq)
            self.forward_decode_metadata.batch_size = bs

            forward_batch.decode_trtllm_mla_metadata = self.forward_decode_metadata
        else:
            return super().init_forward_metadata(forward_batch)

    def pad_draft_extend_query(
        self,
        q: torch.Tensor,
        padded_q: torch.Tensor,
        seq_lens_q: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
    ) -> torch.Tensor:

        return pad_draft_extend_query_triton(
            q,
            padded_q,
            seq_lens_q,
            cu_seqlens_q,
        )

    def unpad_draft_extend_output(
        self,
        raw_out: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        seq_lens_q: torch.Tensor,
        sum_seq_lens_q: int,
        zero_uncovered: bool = False,
    ) -> torch.Tensor:

        output_buffer = self.unpad_output_buffer
        if zero_uncovered:
            if output_buffer is not None:
                output_buffer[:sum_seq_lens_q].zero_()
            else:


                output_buffer = torch.zeros(
                    (sum_seq_lens_q, raw_out.shape[2], raw_out.shape[3]),
                    dtype=raw_out.dtype,
                    device=raw_out.device,
                )
        return unpad_draft_extend_output_triton(
            raw_out,
            cu_seqlens_q,
            seq_lens_q,
            sum_seq_lens_q,
            output_buffer,
        )

    def _compute_decode_bmm1_scale(self, layer: RadixAttention) -> float:

        q_scale = 1.0
        if self.data_type == torch.float8_e4m3fn:
            k_scale = (
                layer.k_scale_float
                if getattr(layer, "k_scale_float", None) is not None
                else 1.0
            )
        else:
            if getattr(layer, "k_scale_float", None) is not None:
                logger.warning_once(
                    "Checkpoint has k_scale but KV cache dtype is not FP8. "
                    "Ignoring k_scale for BMM1 (k_scale=%.4f, kv_dtype=%s).",
                    layer.k_scale_float,
                    self.data_type,
                )
            k_scale = 1.0
        return q_scale * k_scale * layer.scaling

    def _dense_q_indptr(self, bs: int, draft_token_num: int) -> torch.Tensor:

        if draft_token_num == self.num_draft_tokens:
            return self.dense_q_indptr_verify[: bs + 1]
        return self.q_indptr_decode[: bs + 1] * draft_token_num

    def _run_decode_kernel(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        layer: RadixAttention,
        *,
        causal_seqs: Optional[torch.Tensor] = None,
        cp_world: int = 1,
        cp_rank: int = 0,
        return_lse: bool = False,
    ) -> torch.Tensor:

        if cp_world > 1 or return_lse:
            raise NotImplementedError(
                "trtllm_mla does not forward the cyclic DCP metadata to its "
                "decode kernel and returns no rank-local LSE for the cross-rank "
                "merge; select cutedsl_mla or tokenspeed_mla for a DCP "
                "target-verify run"
            )


        bmm1_scale = self._compute_decode_bmm1_scale(layer)
        seq_lens_i32 = (
            seq_lens if seq_lens.dtype == torch.int32 else seq_lens.to(torch.int32)
        )
        extra_kwargs = {"backend": self.backend} if self.backend != "trtllm-gen" else {}
        if self.backend == "trtllm-gen":
            extra_kwargs["multi_ctas_kv_counter_buffer"] = (
                self._multi_ctas_kv_counter_buffer
            )
        return flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache,
            enable_pdl=_ENABLE_PDL,
            workspace_buffer=self.workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens_i32,
            max_seq_len=max_seq_len,
            bmm1_scale=bmm1_scale,
            skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),
            **extra_kwargs,
        )

    def _run_prefill_kernel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        batch_size: int,
        cum_seq_lens_q: torch.Tensor,
        max_q_len: int,
        seq_lens_kv: torch.Tensor,
        cum_seq_lens_kv: torch.Tensor,
        max_kv_len: int,
        is_causal: bool,
        return_lse: bool,
        out_buffer: torch.Tensor,
        o_sf_scale: float = 1.0,
    ):

        q_scale = k_scale = v_scale = 1.0
        if self.data_type == torch.float8_e4m3fn:
            q, k, v, k_scale, v_scale = _quantize_fp8_qkv(q, k, v, layer)
        return flashinfer.prefill.trtllm_ragged_attention_deepseek(
            query=q,
            key=k,
            value=v,
            workspace_buffer=self.workspace_buffer,
            batch_size=batch_size,
            window_left=-1,
            enable_pdl=False,
            max_q_len=max_q_len,
            bmm1_scale=q_scale * k_scale * layer.scaling,
            bmm2_scale=v_scale,
            cum_seq_lens_q=cum_seq_lens_q,
            cum_seq_lens_kv=cum_seq_lens_kv,
            seq_lens=seq_lens_kv,
            max_kv_len=max_kv_len,
            is_causal=is_causal,
            return_lse=return_lse,
            o_sf_scale=o_sf_scale,
            out=out_buffer,
            skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_PREFILL_THRESHOLD_SCALE_FACTOR.get(),
        )

    def _set_kv_and_concat_q_fused(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        k: torch.Tensor,
        k_rope: torch.Tensor,
        q: torch.Tensor,
        q_rope: torch.Tensor,
    ) -> Optional[torch.Tensor]:

        k_nope_2d = k.view(k.shape[0], -1)
        k_rope_2d = k_rope.view(k_rope.shape[0], -1)
        q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        q_rope_3d = q_rope.view(
            -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
        )


        kv_raw = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        kv_2d = kv_raw.view(kv_raw.shape[0], -1) if kv_raw.dim() != 2 else kv_raw
        if not set_mla_kv_concat_q_covered(
            kv_buffer=kv_2d,
            loc=loc,
            k_nope=k_nope_2d,
            k_rope=k_rope_2d,
            q_nope=q_nope,
            q_rope=q_rope_3d,
        ):
            return None
        return set_mla_kv_concat_q(
            kv_buffer=kv_2d,
            loc=loc,
            cache_k_nope=k_nope_2d,
            cache_k_rope=k_rope_2d,
            q_nope=q_nope,
            q_rope=q_rope_3d,
        )

    def _set_kv_and_concat_q_fp8_fused(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        q: torch.Tensor,
        q_rope: torch.Tensor,
        k: torch.Tensor,
        k_rope: torch.Tensor,
    ) -> Optional[torch.Tensor]:

        k_nope_2d = k.view(k.shape[0], -1)
        k_rope_2d = k_rope.view(k_rope.shape[0], -1)
        q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        q_rope_3d = q_rope.view(
            -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
        )


        kv_raw = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        kv_2d = kv_raw.view(kv_raw.shape[0], -1) if kv_raw.dim() != 2 else kv_raw
        if not set_mla_kv_concat_q_fp8_covered(
            kv_buffer=kv_2d,
            loc=loc,
            k_nope=k_nope_2d,
            k_rope=k_rope_2d,
            q_nope=q_nope,
            q_rope=q_rope_3d,
        ):
            return None
        parallel = get_parallel()
        return set_mla_kv_concat_q_fp8(
            kv_buffer=kv_2d,
            loc=loc,
            cache_k_nope=k_nope_2d,
            cache_k_rope=k_rope_2d,
            q_nope=q_nope,
            q_rope=q_rope_3d,


            dcp_world_size=parallel.attn_dcp_size,
            dcp_rank=parallel.attn_dcp_rank,
        )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        is_neox: Optional[bool] = False,
        llama_4_scaling: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        merge_query = q_rope is not None
        fused_fp8_query = None
        if self.data_type == torch.float8_e4m3fn:
            assert q_rope is not None and k_rope is not None
            if cos_sin_cache is None:
                if save_kv_cache and self._fused_set_kv_concat_q_fp8:
                    loc = (
                        self._decode_dense_loc
                        if self._decode_dense_loc is not None
                        else (
                            None if self._unified_mla else forward_batch.out_cache_loc
                        )
                    )
                    if loc is not None:


                        fused_fp8_query = self._set_kv_and_concat_q_fp8_fused(
                            layer=layer,
                            loc=loc,
                            q=q,
                            q_rope=q_rope,
                            k=k,
                            k_rope=k_rope,
                        )
                if fused_fp8_query is None:
                    q, k, k_rope = mla_quantize_without_rope_for_fp8(
                        q, q_rope, k.squeeze(1), k_rope.squeeze(1)
                    )
            else:
                q, k, k_rope = mla_quantize_and_rope_for_fp8(
                    q,
                    q_rope,
                    k.squeeze(1),
                    k_rope.squeeze(1),
                    forward_batch.positions,
                    cos_sin_cache,
                    is_neox,
                    self.kv_lora_rank,
                    self.qk_rope_head_dim,
                )
            merge_query = False


        query = fused_fp8_query
        if query is None and save_kv_cache:
            assert (
                k is not None and k_rope is not None
            ), "For populating trtllm_mla kv cache, both k_nope and k_rope should be not None."
            if self._decode_dense_loc is not None:


                if merge_query and self._fused_set_kv_concat_q:


                    query = self._set_kv_and_concat_q_fused(
                        layer=layer,
                        loc=self._decode_dense_loc,
                        k=k,
                        k_rope=k_rope,
                        q=q,
                        q_rope=q_rope,
                    )
                if query is None:
                    self.token_to_kv_pool.set_mla_kv_buffer(
                        layer, self._decode_dense_loc, k, k_rope, loc_is_dense=True
                    )
            else:

                if (
                    merge_query
                    and self._fused_set_kv_concat_q
                    and not self._unified_mla
                ):


                    query = self._set_kv_and_concat_q_fused(
                        layer=layer,
                        loc=forward_batch.out_cache_loc,
                        k=k,
                        k_rope=k_rope,
                        q=q,
                        q_rope=q_rope,
                    )
                if query is None:
                    self.token_to_kv_pool.set_mla_kv_buffer(
                        layer, forward_batch.out_cache_loc, k, k_rope
                    )


        if query is None:
            if merge_query:

                q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
                q_rope_reshaped = q_rope.view(
                    -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
                )
                query = concat_mla_absorb_q_general(q_nope, q_rope_reshaped)
            else:

                query = q.view(-1, layer.tp_q_head_num, layer.head_dim)


        if llama_4_scaling is not None:
            query = query.to(self.q_data_type) * llama_4_scaling
            query = query.to(self.data_type)


        if query.dim() == 3:
            query = query.unsqueeze(1)


        k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        kv_cache = k_cache.view(-1, self.page_size, self.kv_cache_dim).unsqueeze(1)


        metadata = (
            getattr(forward_batch, "decode_trtllm_mla_metadata", None)
            or self.forward_decode_metadata
        )


        batch_size = getattr(metadata, "batch_size", None)
        if batch_size is not None and batch_size < forward_batch.batch_size:
            self.init_forward_metadata(forward_batch)
            metadata = forward_batch.decode_trtllm_mla_metadata

        raw_out = self._run_decode_kernel(
            query=query,
            kv_cache=kv_cache,
            block_tables=metadata.block_kv_indices,
            seq_lens=(
                metadata.seq_lens_k
                if metadata.seq_lens_k is not None
                else forward_batch.seq_lens
            ),
            max_seq_len=metadata.max_seq_len_k,
            layer=layer,
        )


        output = raw_out.view(-1, layer.tp_q_head_num * layer.v_head_dim)
        return output

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        is_neox: Optional[bool] = False,
        llama_4_scaling: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:


        if (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_target_verify()
            and not forward_batch.forward_mode.is_draft_extend_v2()
            and self.forward_prefill_metadata is not None
            and self.forward_prefill_metadata.fallback_to_flashinfer_impl
        ):
            return super().forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache, q_rope, k_rope
            )


        merge_query = q_rope is not None
        if (
            self.data_type == torch.float8_e4m3fn
        ) and forward_batch.forward_mode.is_target_verify():
            assert q_rope is not None and k_rope is not None
            if cos_sin_cache is None:
                q, k, k_rope = mla_quantize_without_rope_for_fp8(
                    q, q_rope, k.squeeze(1), k_rope.squeeze(1)
                )
            else:
                q, k, k_rope = mla_quantize_and_rope_for_fp8(
                    q,
                    q_rope,
                    k.squeeze(1),
                    k_rope.squeeze(1),
                    forward_batch.positions,
                    cos_sin_cache,
                    is_neox,
                    self.kv_lora_rank,
                    self.qk_rope_head_dim,
                )
            merge_query = False


        if save_kv_cache:
            assert (
                k is not None and k_rope is not None
            ), "For populating trtllm_mla kv cache, both k_nope and k_rope should be not None."
            if self._decode_dense_loc is not None:
                self.token_to_kv_pool.set_mla_kv_buffer(
                    layer, self._decode_dense_loc, k, k_rope, loc_is_dense=True
                )
            else:
                self.token_to_kv_pool.set_mla_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, k_rope
                )


        if merge_query:

            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope_reshaped = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
            q = concat_mla_absorb_q_general(q_nope, q_rope_reshaped)

        q = q.view(-1, layer.tp_q_head_num, layer.head_dim)


        if llama_4_scaling is not None:
            q = q.to(self.q_data_type) * llama_4_scaling
            q = q.to(self.data_type)

        if (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            metadata = (
                getattr(forward_batch, "decode_trtllm_mla_metadata", None)
                or self.forward_decode_metadata
            )


            batch_size = getattr(metadata, "batch_size", None)
            if batch_size is not None and batch_size < forward_batch.batch_size:
                self.init_forward_metadata(forward_batch)
                metadata = forward_batch.decode_trtllm_mla_metadata


            bs = forward_batch.batch_size

            k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
            kv_cache = k_cache.view(-1, self.page_size, self.kv_cache_dim).unsqueeze(1)

            q = q.to(self.data_type)

            if forward_batch.forward_mode.is_target_verify():
                draft_token_num = forward_batch.spec_info.draft_token_num
                dcp_enabled = get_parallel().dcp_enabled
                max_seq_len = metadata.max_seq_len_k + (
                    0 if dcp_enabled else draft_token_num
                )
                ragged_layout = forward_batch.spec_info.ragged_verify_layout
                if ragged_layout is None or dcp_enabled:
                    q = q.view(bs, -1, layer.tp_q_head_num, layer.head_dim)
                    needs_unpad = False
                else:
                    if ragged_layout.bs != bs:


                        ragged_layout = ragged_layout.padded_to_bucket(
                            padded_bs=bs, cap=draft_token_num
                        )
                    total_tokens = q.shape[0]
                    seq_lens_q = torch.clamp(
                        ragged_layout.verify_lens, max=draft_token_num
                    )
                    cu_seqlens_q = ragged_layout.qo_indptr_device
                    padded_q = torch.zeros(
                        (bs, draft_token_num, layer.tp_q_head_num, layer.head_dim),
                        dtype=q.dtype,
                        device=q.device,
                    )
                    q = self.pad_draft_extend_query(
                        q, padded_q, seq_lens_q, cu_seqlens_q
                    )
                    needs_unpad = True
                    unpad_zero_uncovered = True
                    unpad_seq_lens_q = seq_lens_q
                    unpad_cu_seqlens_q = cu_seqlens_q
                    unpad_sum_seq_lens_q = total_tokens
            else:


                total_tokens = q.shape[0]
                tokens_per_seq = total_tokens // bs if bs > 0 else 0
                can_direct_view = bs > 0 and (total_tokens % bs == 0)

                if can_direct_view:
                    max_seq_len = metadata.max_seq_len_k + tokens_per_seq
                    q = q.view(bs, tokens_per_seq, layer.tp_q_head_num, layer.head_dim)
                    needs_unpad = False
                else:

                    actual_seq_lens_q = forward_batch.extend_seq_lens
                    actual_max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)
                    max_seq_len = metadata.max_seq_len_k + actual_max_seq_len_q

                    actual_cu_seqlens_q = torch.nn.functional.pad(
                        torch.cumsum(actual_seq_lens_q, dim=0, dtype=torch.int32),
                        (1, 0),
                    )

                    if self.padded_q_buffer is not None:
                        padded_q = self.padded_q_buffer[
                            :bs, :actual_max_seq_len_q, :, :
                        ].to(dtype=q.dtype)
                        padded_q.zero_()
                    else:
                        padded_q = torch.zeros(
                            (
                                bs,
                                actual_max_seq_len_q,
                                layer.tp_q_head_num,
                                layer.head_dim,
                            ),
                            dtype=q.dtype,
                            device=q.device,
                        )

                    q = self.pad_draft_extend_query(
                        q, padded_q, actual_seq_lens_q, actual_cu_seqlens_q
                    )
                    needs_unpad = True
                    unpad_zero_uncovered = False
                    unpad_seq_lens_q = actual_seq_lens_q
                    unpad_cu_seqlens_q = actual_cu_seqlens_q
                    unpad_sum_seq_lens_q = total_tokens

            assert kv_cache.dtype == self.data_type

            if (
                forward_batch.forward_mode.is_target_verify()
                and get_parallel().dcp_enabled
            ):
                raw_out, lse = self._run_decode_kernel(
                    query=q,
                    kv_cache=kv_cache,
                    block_tables=metadata.block_kv_indices,
                    seq_lens=metadata.seq_lens_k,
                    max_seq_len=max_seq_len,
                    layer=layer,
                    causal_seqs=metadata.global_seq_lens_k,
                    cp_world=get_parallel().dcp_size,
                    cp_rank=get_parallel().dcp_rank,
                    return_lse=True,
                )
                output = raw_out.view(
                    bs * draft_token_num,
                    layer.tp_q_head_num,
                    layer.v_head_dim,
                )
                lse = lse.view(bs * draft_token_num, layer.tp_q_head_num)
                fixup_zero_kv_rows(
                    output,
                    lse,
                    metadata.seq_lens_k,
                    self._dense_q_indptr(bs, draft_token_num),
                    draft_token_num,
                )
                return output.flatten(1), lse

            raw_out = self._run_decode_kernel(
                query=q,
                kv_cache=kv_cache,
                block_tables=metadata.block_kv_indices,
                seq_lens=metadata.seq_lens_k,
                max_seq_len=max_seq_len,
                layer=layer,
            )

            if needs_unpad:


                output = self.unpad_draft_extend_output(
                    raw_out,
                    unpad_cu_seqlens_q,
                    unpad_seq_lens_q,
                    unpad_sum_seq_lens_q,
                    zero_uncovered=unpad_zero_uncovered,
                )
                output = output.view(-1, layer.tp_q_head_num * layer.v_head_dim)
            else:
                output = raw_out.view(-1, layer.tp_q_head_num * layer.v_head_dim)
            return output

        if k_rope is not None:
            k = torch.cat([k, k_rope], dim=-1)
        k = k.view(-1, layer.tp_k_head_num, layer.head_dim)
        v = v.view(-1, layer.tp_k_head_num, layer.v_head_dim)


        if forward_batch.attn_attend_prefix_cache:

            assert forward_batch.prefix_chunk_idx is not None
            assert forward_batch.prefix_chunk_cu_seq_lens is not None
            assert q_rope is None
            assert k_rope is None
            chunk_idx = forward_batch.prefix_chunk_idx

            out = torch.empty(
                q.shape[0],
                layer.tp_q_head_num,
                layer.v_head_dim,
                dtype=self.q_data_type,
                device=q.device,
            )
            result = self._run_prefill_kernel(
                q=q,
                k=k,
                v=v,
                layer=layer,
                batch_size=forward_batch.batch_size,
                cum_seq_lens_q=self.forward_prefill_metadata.cum_seq_lens,
                max_q_len=self.forward_prefill_metadata.max_seq_len,
                seq_lens_kv=forward_batch.prefix_chunk_seq_lens[chunk_idx],
                cum_seq_lens_kv=forward_batch.prefix_chunk_cu_seq_lens[chunk_idx],
                max_kv_len=forward_batch.prefix_chunk_max_seq_lens[chunk_idx],
                is_causal=False,
                return_lse=True,
                out_buffer=out,
                o_sf_scale=-1.0,
            )


            if forward_batch.prefix_chunk_has_zero_kv[chunk_idx]:
                out_tensor, lse_tensor = result
                fixup_zero_kv_rows(
                    out_tensor,
                    lse_tensor,
                    forward_batch.prefix_chunk_seq_lens[chunk_idx],
                    self.forward_prefill_metadata.cum_seq_lens,
                    self.forward_prefill_metadata.max_seq_len,
                )

            return result
        else:
            out = torch.empty(
                q.shape[0],
                q.shape[1],
                v.shape[2],
                device=q.device,
                dtype=self.q_data_type,
            )
            return self._run_prefill_kernel(
                q=q,
                k=k,
                v=v,
                layer=layer,
                batch_size=forward_batch.batch_size,
                cum_seq_lens_q=self.forward_prefill_metadata.cum_seq_lens,
                max_q_len=self.forward_prefill_metadata.max_seq_len,
                seq_lens_kv=self.forward_prefill_metadata.seq_lens,
                cum_seq_lens_kv=self.forward_prefill_metadata.cum_seq_lens,
                max_kv_len=self.forward_prefill_metadata.max_seq_len,
                is_causal=True,
                return_lse=forward_batch.mha_return_lse,
                out_buffer=out,
                o_sf_scale=1.0,
            )


class TRTLLMMLAMultiStepDraftBackend(FlashInferMLAMultiStepDraftBackend):


    needs_cpu_seq_lens: bool = False

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
        backend: str = "trtllm-gen",
    ):
        super().__init__(model_runner, topk, speculative_num_steps)

        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i] = TRTLLMMLABackend(
                model_runner,
                skip_prefill=True,
                kv_indptr_buf=self.kv_indptr[i],
                q_indptr_decode_buf=self.q_indptr_decode,
                backend=backend,
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata(forward_batch)

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        from sglang.srt.model_executor.forward_batch_info import build_inner_fb_view

        if in_capture:
            return super().init_forward_metadata_out_graph(
                forward_batch, in_capture=in_capture
            )
        inner_fb = build_inner_fb_view(
            forward_batch,
            bs=forward_batch.batch_size,
            forward_mode=ForwardMode.DECODE,
        )
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata_out_graph(inner_fb)
