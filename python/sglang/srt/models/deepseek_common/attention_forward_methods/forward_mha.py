from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.ops.attention.dsa.dequant_k_cache import dequantize_k_cache_paged
from sglang.kernels.ops.attention.utils import concat_and_cast_mha_k_triton
from sglang.srt.environ import envs
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.layers.communicator import get_attn_tp_context
from sglang.srt.layers.dcp import (
    all_gather_kv_cache_for_mha_chunk_extend,
    all_gather_kv_cache_for_mha_extend,
    filter_dcp_local_kv_indices,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.models.deepseek_common.utils import (
    _is_cuda,
    _is_musa,
    _is_npu,
    _use_aiter_gfx95,
)
from sglang.srt.runtime_context import (
    get_exec,
    get_parallel,
    get_schedule,
)
from sglang.srt.utils import BumpAllocator, next_power_of_2

if TYPE_CHECKING:
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA

if _is_cuda:
    from sgl_kernel import merge_state_v2

    from sglang.kernels.ops.attention.concat_mla import concat_mla_k
elif _is_musa:
    from sgl_kernel import concat_mla_k


def resolve_attn_backend(forward_batch: ForwardBatch):
    backend = get_attn_backend()
    if isinstance(backend, TboAttnBackend):
        backend = backend.primary
    return backend


def forward_dsa_indexer_for_mha(
    indexer,
    *,
    hidden_states: torch.Tensor,
    q_lora: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    layer_id: int,
) -> None:

    spec_info = forward_batch.spec_info
    seed_buf = spec_info.dsa_seed_topk_capture if spec_info is not None else None
    topk_indices = indexer(
        x=hidden_states,
        q_lora=q_lora,
        positions=positions,
        forward_batch=forward_batch,
        layer_id=layer_id,
        return_indices=seed_buf is not None,
    )
    if seed_buf is None:
        return
    if topk_indices is None:
        raise RuntimeError("DSA MHA indexer did not produce the requested MTP seed")

    select = spec_info.dsa_seed_topk_select
    src = topk_indices if select is None else topk_indices[select]
    seed_buf[: src.shape[0]].copy_(src)


class DeepseekMHAForwardMixin:
    def init_mha_forward(self: DeepseekV2AttentionMLA):
        self.disable_chunked_prefix_cache = get_schedule().disable_chunked_prefix_cache


        self.chunked_prefix_cache_threshold = (
            envs.SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD.get()
        )

    def forward_normal_prepare(
        self: DeepseekV2AttentionMLA,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        if self.q_lora_rank is not None:
            q, latent_cache = (
                get_attn_tp_context()
                .fetch_qkv_latent()
                .split(
                    [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                    dim=-1,
                )
            )


            if self.use_dsa:
                q_lora = self.q_a_layernorm(q)
                q = self.q_b_proj(q_lora)[0].view(
                    -1, self.num_local_heads, self.qk_head_dim
                )
                if self.should_run_indexer():
                    forward_dsa_indexer_for_mha(
                        self.indexer,
                        hidden_states=hidden_states,
                        q_lora=q_lora,
                        positions=positions,
                        forward_batch=forward_batch,
                        layer_id=self.layer_id,
                    )
            else:
                q = self.q_a_layernorm(q)
                q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)

        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]

        _, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        kv_a, _ = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        latent_cache = latent_cache.unsqueeze(1)

        kv_a = self.kv_a_layernorm(kv_a)

        k_pe = latent_cache[:, :, self.kv_lora_rank :]


        backend = resolve_attn_backend(forward_batch)
        if hasattr(backend, "prepare_prefill_qkv"):
            q_out, k_out, v_out = backend.prepare_prefill_qkv(
                q=q,
                q_pe=q_pe,
                kv_a=kv_a,
                k_pe=k_pe,
                positions=positions,
                layer=self,
                forward_batch=forward_batch,
            )
            return q_out, k_out, v_out, forward_batch

        if self.rotary_emb is not None:
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        q[..., self.qk_nope_head_dim :] = q_pe

        self._set_mla_kv_buffer(latent_cache, kv_a, k_pe, forward_batch)
        if (
            forward_batch.mha_one_shot
            and sum(forward_batch.extend_prefix_lens_cpu) != 0
        ):
            if (
                self.use_dsa
                and self.kv_cache_dtype == "fp8_e4m3"
                and (
                    not get_exec().kernel.dsa_decode_backend == "trtllm"
                    or not get_exec().kernel.dsa_prefill_backend == "trtllm"
                )
            ):

                kv_a, k_pe = self._get_mla_kv_buffer_from_fp8_for_dsa(forward_batch)
            else:

                if get_parallel().dcp_enabled:
                    kv_a, k_pe = all_gather_kv_cache_for_mha_extend(
                        get_token_to_kv_pool(),
                        self.attn_mha,
                        forward_batch.attn_dcp_metadata.dcp_local_prefix_kv_indices,
                        forward_batch.seq_lens,
                        forward_batch.extend_prefix_lens,
                        forward_batch.extend_prefix_lens_cpu,
                        forward_batch.extend_seq_lens,
                        kv_a,
                        k_pe,
                    )
                else:
                    kv_a, k_pe = self._get_mla_kv_buffer(
                        forward_batch.fetch_mha_one_shot_kv_indices(),
                        q.dtype,
                        forward_batch,
                    )
        kv = self.kv_b_proj(kv_a)[0]
        kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope = kv[..., : self.qk_nope_head_dim]
        v = kv[..., self.qk_nope_head_dim :]

        k = self._concat_and_cast_mha_k(k_nope, k_pe, forward_batch)
        return q, k, v, forward_batch

    def forward_normal_core(
        self: DeepseekV2AttentionMLA,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        attn_output = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)
        attn_output = attn_output.reshape(-1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)
        return output

    def forward_normal_chunked_kv_prepare(
        self: DeepseekV2AttentionMLA,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):


        return self.forward_normal_prepare(
            positions, hidden_states, forward_batch, zero_allocator
        )

    def forward_normal_chunked_kv_core(
        self: DeepseekV2AttentionMLA,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        has_extend_prefix = forward_batch.extend_prefix_lens_cpu is not None and any(
            forward_batch.extend_prefix_lens_cpu
        )

        if has_extend_prefix and forward_batch.num_prefix_chunks is None:
            forward_batch.prepare_chunked_prefix_cache_info(q.device)
            if hasattr(get_attn_backend(), "init_mha_chunk_metadata"):
                get_attn_backend().init_mha_chunk_metadata(forward_batch)

        forward_batch.mha_return_lse = has_extend_prefix

        forward_batch.set_attn_attend_prefix_cache(False)
        attn_output = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)


        if has_extend_prefix:
            attn_output, lse = attn_output
            forward_batch.set_attn_attend_prefix_cache(True)
            attn_output = self._chunked_prefix_attn_mha(
                q=q,
                accum_output=attn_output,
                accum_lse=lse,
                forward_batch=forward_batch,
            )

        attn_output = attn_output.reshape(-1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)
        return output

    def forward_normal_one_shot_prepare(
        self: DeepseekV2AttentionMLA,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        forward_batch.mha_one_shot = True
        return self.forward_normal_prepare(
            positions, hidden_states, forward_batch, zero_allocator
        )

    def forward_normal_one_shot_core(
        self: DeepseekV2AttentionMLA,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        has_extend_prefix = any(forward_batch.extend_prefix_lens_cpu)

        if has_extend_prefix and forward_batch.num_prefix_chunks is None:
            forward_batch.num_prefix_chunks = 0
            if hasattr(get_attn_backend(), "init_mha_chunk_metadata"):
                get_attn_backend().init_mha_chunk_metadata(forward_batch)
        forward_batch.mha_return_lse = False

        forward_batch.set_attn_attend_prefix_cache(False)
        return self.forward_normal_core(q, k, v, forward_batch)

    def _chunked_prefix_attn_mha(
        self: DeepseekV2AttentionMLA,
        q: torch.Tensor,
        accum_output: torch.Tensor,
        accum_lse: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:

        backend = resolve_attn_backend(forward_batch)
        pack_fn = getattr(backend, "pack_prefix_chunk_kv", None)
        kv_a_dtype = torch.bfloat16 if pack_fn is not None else q.dtype

        assert forward_batch.num_prefix_chunks is not None
        for i in range(forward_batch.num_prefix_chunks):
            forward_batch.set_prefix_chunk_idx(i)

            kv_indices = forward_batch.prefix_chunk_kv_indices[i]

            kv_a_normed, k_pe = self._get_mla_kv_buffer(
                kv_indices, kv_a_dtype, forward_batch
            )
            kv_a_normed, k_pe = all_gather_kv_cache_for_mha_chunk_extend(
                kv_a_normed,
                k_pe,
                forward_batch.prefix_chunk_seq_lens_cpu[i],
                forward_batch.prefix_chunk_starts_cpu[i],
            )
            kv = self.kv_b_proj(kv_a_normed)[0]
            kv = kv.view(
                -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            v = kv[..., self.qk_nope_head_dim :]
            k_nope = kv[..., : self.qk_nope_head_dim]

            if pack_fn is not None:
                k, v = pack_fn(k_nope, k_pe, v)
            else:
                k = torch.empty(
                    (
                        k_nope.shape[0],
                        self.num_local_heads,
                        self.qk_nope_head_dim + self.qk_rope_head_dim,
                    ),
                    dtype=v.dtype,
                    device=v.device,
                )
                k[..., : self.qk_nope_head_dim] = k_nope
                k[..., self.qk_nope_head_dim :] = k_pe

            output, lse = self.attn_mha(
                q,
                k,
                v,
                forward_batch,
                save_kv_cache=False,


                key_value_num_tokens=k.shape[0],
            )
            tmp_output = torch.empty_like(accum_output)
            tmp_lse = torch.empty_like(accum_lse)
            merge_state_v2(output, lse, accum_output, accum_lse, tmp_output, tmp_lse)
            accum_output, accum_lse = tmp_output, tmp_lse
            del kv, k, v, output, lse, tmp_output, tmp_lse

        return accum_output

    def _set_mla_kv_buffer(
        self: DeepseekV2AttentionMLA,
        latent_cache: torch.Tensor,
        kv_a: torch.Tensor,
        k_pe: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        if _is_cuda:

            get_token_to_kv_pool().set_mla_kv_buffer(
                self.attn_mha, forward_batch.out_cache_loc, kv_a.unsqueeze(1), k_pe
            )
        elif _is_npu:

            get_token_to_kv_pool().set_kv_buffer(
                self.attn_mha, forward_batch.out_cache_loc, kv_a.unsqueeze(1), k_pe
            )
        else:
            latent_cache[:, :, : self.kv_lora_rank] = kv_a.unsqueeze(1)
            latent_cache[:, :, self.kv_lora_rank :] = k_pe.clone()


            get_token_to_kv_pool().set_kv_buffer(
                self.attn_mha, forward_batch.out_cache_loc, latent_cache, None
            )

    def _get_mla_kv_buffer(
        self: DeepseekV2AttentionMLA,
        kv_indices: torch.Tensor,
        dst_dtype: torch.dtype,
        forward_batch: ForwardBatch,
    ):
        if _is_cuda:
            kv_a, k_pe = get_token_to_kv_pool().get_mla_kv_buffer(
                self.attn_mha, kv_indices, dst_dtype
            )
            kv_a = kv_a.squeeze(1)
        else:
            latent_cache_buf = get_token_to_kv_pool().get_key_buffer(
                self.attn_mha.layer_id
            )
            latent_cache = latent_cache_buf[kv_indices].contiguous().to(dst_dtype)

            kv_a, k_pe = latent_cache.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            kv_a = kv_a.squeeze(1).contiguous()
        return kv_a, k_pe

    def _get_mla_kv_buffer_from_fp8_for_dsa(
        self: DeepseekV2AttentionMLA,
        forward_batch: ForwardBatch,
    ):

        backend = get_attn_backend()
        if isinstance(backend, TboAttnBackend):
            backend = backend.primary
        kv_indices = backend.forward_metadata.page_table_1_flattened
        assert (
            kv_indices is not None
        ), "page_table_1_flattened should have been generated for FP8 MHA path"

        if _use_aiter_gfx95:


            kv_indices = filter_dcp_local_kv_indices(kv_indices=kv_indices)
            kv_a, k_pe = get_token_to_kv_pool().get_mla_kv_buffer(
                self.attn_mha, kv_indices, torch.bfloat16
            )
            kv_a = kv_a.squeeze(1).contiguous()
            return kv_a, k_pe

        kv_cache_fp8 = get_token_to_kv_pool().get_key_buffer(self.attn_mha.layer_id)

        kv_latent_bf16 = dequantize_k_cache_paged(kv_cache_fp8, kv_indices)

        kv_a = kv_latent_bf16[:, :, : self.kv_lora_rank].squeeze(1).contiguous()
        k_pe = kv_latent_bf16[:, :, self.kv_lora_rank :]

        return kv_a, k_pe

    def _concat_and_cast_mha_k(
        self: DeepseekV2AttentionMLA,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        forward_batch: ForwardBatch,
    ):

        k_shape = (k_nope.shape[0], self.num_local_heads, self.qk_head_dim)
        if (
            (_is_cuda or _is_musa)
            and (self.num_local_heads == 128)
            and (self.qk_nope_head_dim == 128)
            and (self.qk_rope_head_dim == 64)
        ):
            k = k_nope.new_empty(*k_shape)
            concat_mla_k(k=k, k_nope=k_nope, k_rope=k_pe)
        elif (
            _is_cuda
            and next_power_of_2(self.num_local_heads) == self.num_local_heads
            and next_power_of_2(self.qk_nope_head_dim) == self.qk_nope_head_dim
            and next_power_of_2(self.qk_rope_head_dim) == self.qk_rope_head_dim
        ):

            if (
                self.current_attention_backend == "fa3"
                and self.kv_cache_dtype != "auto"
            ):
                attn_dtype = get_token_to_kv_pool().dtype
            else:
                attn_dtype = k_nope.dtype
            k = k_nope.new_empty(*k_shape, dtype=attn_dtype)
            concat_and_cast_mha_k_triton(k, k_nope, k_pe)
        else:
            k = k_nope.new_empty(*k_shape)
            k[..., : self.qk_nope_head_dim] = k_nope
            k[..., self.qk_nope_head_dim :] = k_pe
        return k
