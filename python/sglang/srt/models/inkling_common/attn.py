from __future__ import annotations

from collections.abc import Callable
from functools import cache

import torch
from torch import nn

from sglang.kernels.ops.attention.flash_attn.cute.batch_invariance import (
    is_batch_invariant,
)
from sglang.kernels.ops.attention.inkling_rel_proj import rel_proj_small_t
from sglang.kernels.ops.attention.inkling_row_scale import row_compact_bf16
from sglang.kernels.ops.attention.log_scaling_tau import (
    apply_log_scaling_tau as _apply_log_scaling_tau,
)
from sglang.kernels.ops.attention.score_mod import (
    relative_bias_score_mod as triton_relative_bias_score_mod,
)
from sglang.srt.environ import envs
from sglang.srt.layers.linear import MergedColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode
from sglang.srt.models.inkling_common.kernels.comm import (
    get_ar_buffer,
    reduce_scatter_hidden,
    symm_mem_all_reduce,
)
from sglang.srt.models.inkling_common.norm import RMSNorm
from sglang.srt.models.inkling_common.sconv import SconvType, ShortConvolution
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.runtime_context import (
    attention_backends,
    get_exec,
    get_model,
    get_parallel,
    get_spec,
)
from sglang.srt.utils import add_prefix, get_current_device_stream_fast

try:
    import cutlass.cute as cute
    from cutlass.cute import Float32

    from sglang.kernels.ops.attention.flash_attn.cute.seqlen_info import SeqlenInfoQK
except Exception as _import_error:
    cute = None
    Float32 = None
    SeqlenInfoQK = None
    _cute_import_error = _import_error
else:
    _cute_import_error = None


@cache
def get_inkling_relative_attention_score_mod(rel_extent: int) -> Callable:
    if cute is None or Float32 is None or SeqlenInfoQK is None:
        raise ImportError(
            "Inkling relative attention requires the vendored FA4 CUTE interface."
        ) from _cute_import_error

    @cute.jit
    def score_mod_rel_bias(
        scores: cute.TensorSSA,
        b_idx: cute.TensorSSA,
        h_idx: cute.TensorSSA,
        q_idx: cute.TensorSSA,
        kv_idx: cute.TensorSSA,
        seqlen_info: SeqlenInfoQK,
        aux_tensors: list[cute.Tensor],
    ) -> cute.TensorSSA:
        rel_logits = aux_tensors[0]

        seqlen_local_offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q
        rel_dist = (q_idx + seqlen_local_offset) - kv_idx
        global_q_idx = seqlen_info.offset_q + q_idx

        rel_dist_0 = rel_dist[0]
        rel_idx = rel_dist_0 if rel_dist_0 >= 0 else 0
        rel_idx = rel_idx if rel_idx < rel_extent else (rel_extent - 1)

        rel_bias = rel_logits[global_q_idx[0], h_idx[0], rel_idx]
        rel_bias = Float32(rel_bias) if rel_dist_0 == rel_idx else Float32(0.0)
        return scores + rel_bias

    return score_mod_rel_bias


def compute_log_scaling_tau(
    positions: torch.Tensor, n_floor: int, alpha: float
) -> torch.Tensor:
    effective_n = (positions + 1).to(torch.float32)
    return 1.0 + alpha * torch.log(torch.clamp(effective_n / float(n_floor), min=1.0))


_REL_PROJ_MATMUL_MAX_T = 48


_REL_PROJ_TAU_KERNEL_MAX_T = 32


def serving_attention_backend(forward_batch: ForwardBatch) -> str:

    from sglang.srt.model_executor.forward_context import get_attn_backend

    backend = get_attn_backend()
    configured_prefill, configured_decode = attention_backends()
    prefill = backend.prefill_attention_backend_str or configured_prefill
    decode = backend.decode_attention_backend_str or configured_decode
    mode = forward_batch.forward_mode
    if mode.is_decode_or_idle():
        return decode
    if mode.is_target_verify():
        return decode if get_spec().speculative_attention_mode == "decode" else prefill
    return prefill


def _rel_proj_kernel_eligible(r: torch.Tensor) -> bool:

    return (
        r.is_cuda
        and r.dtype == torch.bfloat16
        and r.stride(-1) == 1
        and r.stride(-2) == r.shape[-1]
        and r.shape[2] % 8 == 0
        and r.data_ptr() % 16 == 0
        and (r.stride(0) * 2) % 16 == 0
    )


class RelLogitsProj(nn.Module):
    def __init__(self, d_rel: int, rel_extent: int, *, deterministic: bool = False):
        super().__init__()
        self.d_rel = d_rel
        self.rel_extent = rel_extent
        self.proj = nn.Parameter(torch.empty(d_rel, rel_extent), requires_grad=False)


        self._prescale_tau = envs.SGLANG_OPT_USE_INKLING_FUSED_LOG_TAU.get()


        self._proj_dispatch = (
            envs.SGLANG_OPT_USE_INKLING_REL_PROJ_DISPATCH.get() and not deterministic
        )

    def _project(self, r: torch.Tensor) -> torch.Tensor:

        if not self._proj_dispatch or r.is_contiguous():
            return torch.einsum("thd,de->the", r, self.proj)
        if r.shape[0] <= _REL_PROJ_MATMUL_MAX_T:
            return r @ self.proj
        rows, inner = r.shape[0], r.shape[1] * r.shape[2]
        if (
            r.is_cuda
            and r.dtype == torch.bfloat16
            and r.stride(-1) == 1
            and r.stride(-2) == r.shape[-1]
            and inner % 8 == 0
            and r.data_ptr() % 16 == 0
            and (r.stride(0) * 2) % 16 == 0
        ):
            r2d = torch.as_strided(r, (rows, inner), (r.stride(0), 1))
            r = row_compact_bf16(r2d).view(r.shape)
        return torch.einsum("thd,de->the", r, self.proj)

    def forward(
        self, r_out: torch.Tensor, log_scaling_tau: torch.Tensor | None = None
    ) -> torch.Tensor:

        if log_scaling_tau is not None and self._prescale_tau:
            if (
                self._proj_dispatch
                and r_out.shape[0] <= _REL_PROJ_TAU_KERNEL_MAX_T
                and _rel_proj_kernel_eligible(r_out)
            ):


                tau_flat = log_scaling_tau.reshape(-1)
                if tau_flat.dtype != torch.float32:
                    tau_flat = tau_flat.float()
                return rel_proj_small_t(r_out, self.proj, tau_flat)


            r_out = _apply_log_scaling_tau(r_out, log_scaling_tau.view(-1, 1, 1))
            log_scaling_tau = None
        out = self._project(r_out)
        if log_scaling_tau is not None:
            out = _apply_log_scaling_tau(out, log_scaling_tau.view(-1, 1, 1))
        return out


class InklingQKVRLinear(MergedColumnParallelLinear):


    is_inkling_qkvr = True

    def __init__(
        self,
        *args,
        inkling_num_kv_heads: int,
        inkling_head_dim: int,
        inkling_num_heads: int,
        inkling_d_rel: int,
        inkling_tp_size: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.inkling_num_kv_heads = inkling_num_kv_heads
        self.inkling_head_dim = inkling_head_dim
        self.inkling_num_heads = inkling_num_heads
        self.inkling_d_rel = inkling_d_rel
        self.inkling_tp_size = inkling_tp_size


class InklingAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int | None,
        d_rel: int,
        rel_extent: int,
        local_extent: int,
        norm_eps: float,
        is_local: bool,
        layer_id: int,
        q_bias: bool = False,
        o_bias: bool = False,
        kv_conv: bool = False,
        sconv_kernel_size: int | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.alt_stream = alt_stream

        attn_tp_rank = get_parallel().attn_tp_rank
        attn_tp_size = get_parallel().attn_tp_size

        self.tp_size = attn_tp_size
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        self.d_rel = d_rel
        self.num_total_heads = num_heads
        self.num_total_kv_heads = num_kv_heads
        self.scaling = 1.0 / self.head_dim

        assert self.num_total_heads % self.tp_size == 0
        self.num_tp_heads = self.num_total_heads // self.tp_size

        if self.num_total_kv_heads >= self.tp_size:
            assert self.num_total_kv_heads % self.tp_size == 0
        else:
            assert self.tp_size % self.num_total_kv_heads == 0
        self.num_tp_kv_heads = max(1, self.num_total_kv_heads // self.tp_size)
        self.layer_id = layer_id
        self.is_local = is_local
        self._kv_total_for_sizing = max(self.num_total_kv_heads, self.tp_size)

        output_sizes = [
            self.head_dim * self.num_total_heads,
            self.head_dim * self._kv_total_for_sizing,
            self.head_dim * self._kv_total_for_sizing,
            self.d_rel * self.num_total_heads,
        ]

        self.qkvr = InklingQKVRLinear(
            input_size=self.hidden_size,
            output_sizes=output_sizes,
            bias=q_bias,
            prefix=add_prefix("qkvr", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            quant_config=quant_config,
            inkling_num_kv_heads=self.num_total_kv_heads,
            inkling_head_dim=self.head_dim,
            inkling_num_heads=self.num_total_heads,
            inkling_d_rel=self.d_rel,
            inkling_tp_size=attn_tp_size,
        )
        self.wo_ud = RowParallelLinear(
            input_size=self.head_dim * self.num_total_heads,
            output_size=self.hidden_size,
            bias=o_bias,
            prefix=add_prefix("wo_ud", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
            use_dp_attention_reduce=True,
            quant_config=quant_config,
        )


        self.scattered_sconv = get_exec().comm.enable_scattered_sconv

        if is_local:
            self.rel_extent = local_extent
            self.local_extent = local_extent
        else:
            self.rel_extent = rel_extent
            self.local_extent = None

        self.rel_logits_proj = RelLogitsProj(
            self.d_rel,
            self.rel_extent,
            deterministic=get_exec().deterministic.enable_deterministic_inference,
        )


        self._fused_log_tau = envs.SGLANG_OPT_USE_INKLING_FUSED_LOG_TAU.get()
        self.q_norm = RMSNorm(self.head_dim, eps=norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=norm_eps)

        self.kv_conv = kv_conv
        self.sconv_kernel_size = sconv_kernel_size
        self.k_sconv = (
            ShortConvolution(
                hidden_size=self.head_dim * self.num_tp_kv_heads,
                kernel_size=self.sconv_kernel_size,
                sconv_type=SconvType.K_LOCAL if is_local else SconvType.K_FULL,
                layer_id=layer_id,
            )
            if self.kv_conv
            else None
        )
        self.v_sconv = (
            ShortConvolution(
                hidden_size=self.head_dim * self.num_tp_kv_heads,
                kernel_size=self.sconv_kernel_size,
                sconv_type=SconvType.V_LOCAL if is_local else SconvType.V_FULL,
                layer_id=layer_id,
            )
            if self.kv_conv
            else None
        )

        self.attn = RadixAttention(
            self.num_tp_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_tp_kv_heads,
            layer_id=self.layer_id,
            sliding_window_size=self.local_extent - 1 if self.is_local else -1,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def _project_qkvr(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tokens = hidden_states.size(0)
        qkvr, _ = self.qkvr(hidden_states)
        qkvr = qkvr.view(num_tokens, -1)
        split_sizes = [
            self.head_dim * self.num_tp_heads,
            self.head_dim * self.num_tp_kv_heads,
            self.head_dim * self.num_tp_kv_heads,
            self.d_rel * self.num_tp_heads,
        ]
        q, k, v, r = qkvr.split(split_sizes, dim=-1)
        return q, k, v, r

    def _fused_attn_prologue_verify(self, q, k, v, forward_batch, log_scaling_tau=None):

        from sglang.kernels.ops.attention.inkling_attn_prologue import (
            inkling_attn_prologue_verify,
        )
        from sglang.srt.model_executor.forward_context import (
            get_attn_backend,
            get_token_to_kv_pool,
        )

        if not hasattr(self, "_qk_gamma_bf16"):
            self._qk_gamma_bf16 = (
                self.q_norm.weight.to(torch.bfloat16),
                self.k_norm.weight.to(torch.bfloat16),
            )
        k_cache, ci, cm, kw, k_inter = self.k_sconv.verify_fused_ar_inputs(
            forward_batch
        )
        v_cache, _, _, vw, v_inter = self.v_sconv.verify_fused_ar_inputs(forward_batch)
        pool = get_token_to_kv_pool()
        k_buf = pool.get_key_buffer(self.layer_id)
        v_buf = pool.get_value_buffer(self.layer_id)


        do_bf16_store = (
            k_buf.dtype == torch.bfloat16
            and k_buf.dim() == 3
            and k_buf.shape[-1] == self.head_dim
            and k_buf.is_contiguous()
        )
        sfk = sfv = None
        do_mxfp8_store = False
        if get_model().kv_cache_dtype == "mxfp8" and hasattr(
            pool, "get_kv_scale_buffer"
        ):
            sfk, sfv = pool.get_kv_scale_buffer(self.layer_id)
            do_mxfp8_store = (
                k_buf.dtype == torch.float8_e4m3fn
                and v_buf.dtype == torch.float8_e4m3fn
                and k_buf.dim() == 3
                and v_buf.dim() == 3
                and k_buf.shape[-1] == self.head_dim
                and v_buf.shape[-1] == self.head_dim
                and k_buf.is_contiguous()
                and v_buf.is_contiguous()
                and sfk.dim() == 5
                and sfv.dim() == 5
                and getattr(pool, "page_size", 0) == 128
            )
        do_store = do_bf16_store or do_mxfp8_store
        metadata = get_attn_backend().forward_metadata
        if do_store and self.is_local:

            loc = metadata.swa_out_cache_loc
        elif do_store:
            loc = getattr(metadata, "out_cache_loc_full_physical", None)
            if loc is None:
                loc = forward_batch.out_cache_loc
        else:
            loc = forward_batch.out_cache_loc
        es = q.element_size()
        q, k, v, q_descale = inkling_attn_prologue_verify(
            q,
            k_cache,
            v_cache,
            ci,
            cm,
            kw,
            vw,
            k_inter,
            v_inter,
            self._qk_gamma_bf16[0],
            self._qk_gamma_bf16[1],
            self.q_norm.variance_epsilon,
            loc,
            k_buf,
            v_buf,
            0,
            (k.data_ptr() - q.data_ptr()) // es,
            (v.data_ptr() - q.data_ptr()) // es,
            q.shape[1],
            k.shape[1],
            forward_batch.spec_info.draft_token_num,
            activation=self.k_sconv.activation,
            use_residual=self.k_sconv.use_residual,
            do_store=do_store,
            mxfp8_quant=do_mxfp8_store,
            sfk=sfk,
            sfv=sfv,
            page_size=getattr(pool, "page_size", 128),
            log_scaling_tau=log_scaling_tau,
        )
        return q, k, v, do_store, q_descale

    def _fused_attn_prologue_extend(self, q, k, v, forward_batch, log_scaling_tau=None):

        from sglang.kernels.ops.attention.inkling_attn_prologue import (
            inkling_attn_prologue_extend,
        )
        from sglang.srt.model_executor.forward_context import (
            get_attn_backend,
            get_token_to_kv_pool,
        )

        if not hasattr(self, "_qk_gamma_bf16"):
            self._qk_gamma_bf16 = (
                self.q_norm.weight.to(torch.bfloat16),
                self.k_norm.weight.to(torch.bfloat16),
            )
        (
            k_cache,
            _safe_idx,
            cm,
            cu,
            si,
            kw,
            _qsl,
            ci,
            has_init,
            trows,
            tmask,
            tdst,
        ) = self.k_sconv.extend_fused_ar_inputs(forward_batch)
        v_inputs = self.v_sconv.extend_fused_ar_inputs(forward_batch)
        v_cache, vw = v_inputs[0], v_inputs[5]
        pool = get_token_to_kv_pool()
        k_buf = pool.get_key_buffer(self.layer_id)
        v_buf = pool.get_value_buffer(self.layer_id)
        do_bf16_store = (
            k_buf.dtype == torch.bfloat16
            and v_buf.dtype == torch.bfloat16
            and k_buf.dim() == 3
            and v_buf.dim() == 3
            and k_buf.shape[-1] == self.head_dim
            and v_buf.shape[-1] == self.head_dim
            and k_buf.is_contiguous()
            and v_buf.is_contiguous()
        )
        sfk = sfv = None
        do_mxfp8_store = False
        if get_model().kv_cache_dtype == "mxfp8" and hasattr(
            pool, "get_kv_scale_buffer"
        ):
            sfk, sfv = pool.get_kv_scale_buffer(self.layer_id)
            do_mxfp8_store = (
                k_buf.dtype == torch.float8_e4m3fn
                and v_buf.dtype == torch.float8_e4m3fn
                and k_buf.dim() == 3
                and v_buf.dim() == 3
                and k_buf.shape[-1] == self.head_dim
                and v_buf.shape[-1] == self.head_dim
                and k_buf.is_contiguous()
                and v_buf.is_contiguous()
                and sfk.dim() == 5
                and sfv.dim() == 5
                and getattr(pool, "page_size", 0) == 128
            )
        do_store = do_bf16_store or do_mxfp8_store
        metadata = get_attn_backend().forward_metadata
        if do_store and self.is_local:
            loc = metadata.swa_out_cache_loc
        elif do_store:
            loc = getattr(metadata, "out_cache_loc_full_physical", None)
            if loc is None:
                loc = forward_batch.out_cache_loc
        else:
            loc = forward_batch.out_cache_loc
        is_v2 = forward_batch.forward_mode.is_draft_extend_v2()
        if is_v2:


            dev = trows.device
            trows = torch.empty((0, trows.shape[1]), dtype=torch.int64, device=dev)
            tmask = torch.empty((0,), dtype=torch.bool, device=dev)
            tdst = torch.empty((0,), dtype=torch.int64, device=dev)
        k_pre, v_pre = k, v
        es = q.element_size()
        q, k, v, q_descale = inkling_attn_prologue_extend(
            q,
            k_cache,
            v_cache,
            ci,
            cm,
            has_init,
            cu,
            si,
            kw,
            vw,
            trows,
            tmask,
            tdst,
            self._qk_gamma_bf16[0],
            self._qk_gamma_bf16[1],
            self.q_norm.variance_epsilon,
            loc,
            k_buf,
            v_buf,
            0,
            (k.data_ptr() - q.data_ptr()) // es,
            (v.data_ptr() - q.data_ptr()) // es,
            q.shape[1],
            k.shape[1],
            activation=self.k_sconv.activation,
            use_residual=self.k_sconv.use_residual,
            do_store=do_store,
            mxfp8_quant=do_mxfp8_store,
            sfk=sfk,
            sfv=sfv,
            page_size=getattr(pool, "page_size", 128),
            do_cache_update=not is_v2,
            log_scaling_tau=log_scaling_tau,
        )
        if is_v2:
            self.k_sconv._update_sconv_cache_for_draft_extend(
                forward_batch, k_cache, ci, k_pre
            )
            self.v_sconv._update_sconv_cache_for_draft_extend(
                forward_batch, v_cache, ci, v_pre
            )
        return q, k, v, do_store, q_descale

    def _fused_attn_prologue_decode(self, q, k, v, forward_batch, log_scaling_tau=None):

        from sglang.kernels.ops.attention.inkling_attn_prologue import (
            inkling_attn_prologue_decode,
        )
        from sglang.srt.model_executor.forward_context import (
            get_attn_backend,
            get_token_to_kv_pool,
        )

        if not hasattr(self, "_qk_gamma_bf16"):
            self._qk_gamma_bf16 = (
                self.q_norm.weight.to(torch.bfloat16),
                self.k_norm.weight.to(torch.bfloat16),
            )
        k_cache, ci, cm, kw = self.k_sconv.decode_fused_ar_inputs(forward_batch)
        v_cache, _, _, vw = self.v_sconv.decode_fused_ar_inputs(forward_batch)
        pool = get_token_to_kv_pool()
        k_buf = pool.get_key_buffer(self.layer_id)
        v_buf = pool.get_value_buffer(self.layer_id)
        do_bf16_store = (
            k_buf.dtype == torch.bfloat16
            and v_buf.dtype == torch.bfloat16
            and k_buf.dim() == 3
            and v_buf.dim() == 3
            and k_buf.shape[-1] == self.head_dim
            and v_buf.shape[-1] == self.head_dim
            and k_buf.is_contiguous()
            and v_buf.is_contiguous()
        )
        sfk = sfv = None
        do_mxfp8_store = False
        if get_model().kv_cache_dtype == "mxfp8" and hasattr(
            pool, "get_kv_scale_buffer"
        ):
            sfk, sfv = pool.get_kv_scale_buffer(self.layer_id)
            do_mxfp8_store = (
                k_buf.dtype == torch.float8_e4m3fn
                and v_buf.dtype == torch.float8_e4m3fn
                and k_buf.dim() == 3
                and v_buf.dim() == 3
                and k_buf.shape[-1] == self.head_dim
                and v_buf.shape[-1] == self.head_dim
                and k_buf.is_contiguous()
                and v_buf.is_contiguous()
                and sfk.dim() == 5
                and sfv.dim() == 5
                and getattr(pool, "page_size", 0) == 128
            )
        do_store = do_bf16_store or do_mxfp8_store
        metadata = get_attn_backend().forward_metadata
        if do_store and self.is_local:
            loc = metadata.swa_out_cache_loc
        elif do_store:
            loc = getattr(metadata, "out_cache_loc_full_physical", None)
            if loc is None:
                loc = forward_batch.out_cache_loc
        else:
            loc = forward_batch.out_cache_loc
        es = q.element_size()
        q, k, v, q_descale = inkling_attn_prologue_decode(
            q,
            k_cache,
            v_cache,
            ci,
            cm,
            kw,
            vw,
            self._qk_gamma_bf16[0],
            self._qk_gamma_bf16[1],
            self.q_norm.variance_epsilon,
            loc,
            k_buf,
            v_buf,
            0,
            (k.data_ptr() - q.data_ptr()) // es,
            (v.data_ptr() - q.data_ptr()) // es,
            q.shape[1],
            k.shape[1],
            activation=self.k_sconv.activation,
            use_residual=self.k_sconv.use_residual,
            track_mask=forward_batch.mamba_track_mask,
            track_indices=forward_batch.mamba_track_indices,
            do_store=do_store,
            mxfp8_quant=do_mxfp8_store,
            sfk=sfk,
            sfv=sfv,
            page_size=getattr(pool, "page_size", 128),
            log_scaling_tau=log_scaling_tau,
        )
        return q, k, v, do_store, q_descale

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        *,
        log_scaling_tau: torch.Tensor | None = None,
        reduce: bool = True,
    ) -> torch.Tensor:

        assert hidden_states.ndim == 2
        num_tokens = hidden_states.size(0)
        q, k, v, r = self._project_qkvr(hidden_states)
        r = r.view(num_tokens, -1, self.d_rel)

        apply_log_scaling = log_scaling_tau is not None and not self.is_local


        attention_backend = serving_attention_backend(forward_batch)
        assert attention_backend in ("fa4", "triton")


        fa4 = attention_backend == "fa4"

        rel_event = None
        prologue_did_store = False
        prologue_q_descale = None
        _fm = forward_batch.forward_mode
        _prologue_verify = _fm.is_target_verify()
        _prologue_decode = _fm.is_decode()


        _prologue_extend = (
            _fm.is_extend(include_draft_extend_v2=True) and not _prologue_verify
        )
        fused_prologue = (
            fa4
            and self.kv_conv
            and self.alt_stream is not None
            and num_tokens > 0
            and self.head_dim == 128
            and (_prologue_verify or _prologue_decode or _prologue_extend)
            and envs.SGLANG_OPT_USE_INKLING_FUSED_ATTN_PROLOGUE.get()
        )


        fold_tau_q = apply_log_scaling and self._fused_log_tau
        if fused_prologue:


            current_stream = get_current_device_stream_fast()
            self.alt_stream.wait_stream(current_stream)
            _tau_arg = log_scaling_tau if fold_tau_q else None
            if _prologue_verify:
                (
                    q,
                    k,
                    v,
                    prologue_did_store,
                    prologue_q_descale,
                ) = self._fused_attn_prologue_verify(
                    q, k, v, forward_batch, log_scaling_tau=_tau_arg
                )
            elif _prologue_decode:
                (
                    q,
                    k,
                    v,
                    prologue_did_store,
                    prologue_q_descale,
                ) = self._fused_attn_prologue_decode(
                    q, k, v, forward_batch, log_scaling_tau=_tau_arg
                )
            else:
                (
                    q,
                    k,
                    v,
                    prologue_did_store,
                    prologue_q_descale,
                ) = self._fused_attn_prologue_extend(
                    q, k, v, forward_batch, log_scaling_tau=_tau_arg
                )
            with torch.cuda.stream(self.alt_stream):
                rel_logits = self.rel_logits_proj(
                    r, log_scaling_tau if apply_log_scaling else None
                )
                rel_event = torch.cuda.Event()
                rel_event.record()
        use_alt = (
            not fused_prologue
            and fa4
            and self.alt_stream is not None
            and hidden_states.is_cuda
            and num_tokens > 0
            and get_is_capture_mode()
        )
        if use_alt:


            current_stream = get_current_device_stream_fast()
            self.alt_stream.wait_stream(current_stream)
            with torch.cuda.stream(self.alt_stream):
                if self.kv_conv:
                    assert self.v_sconv is not None
                    v = self.v_sconv(v, positions, forward_batch)
                v_event = torch.cuda.Event()
                v_event.record()
                rel_logits = self.rel_logits_proj(
                    r, log_scaling_tau if apply_log_scaling else None
                )
                rel_event = torch.cuda.Event()
                rel_event.record()
            if self.kv_conv:
                assert self.k_sconv is not None
                k = self.k_sconv(k, positions, forward_batch)
        elif not fused_prologue:
            if self.kv_conv:
                assert self.k_sconv is not None
                assert self.v_sconv is not None
                k = self.k_sconv(k, positions, forward_batch)
                v = self.v_sconv(v, positions, forward_batch)


        if fused_prologue:
            pass
        else:
            q, k = apply_qk_norm(
                q=q,
                k=k,
                q_norm=self.q_norm,
                k_norm=self.k_norm,
                head_dim=self.head_dim,
                alt_stream=None,
            )

        if apply_log_scaling and not (fused_prologue and fold_tau_q):


            q = _apply_log_scaling_tau(q, log_scaling_tau.view(-1, 1))

        if use_alt:

            current_stream.wait_event(v_event)
        elif not fused_prologue:
            rel_logits = self.rel_logits_proj(
                r, log_scaling_tau if apply_log_scaling else None
            )

        extra_attn_kwargs = {}
        if get_model().kv_cache_dtype == "mxfp8":


            if prologue_q_descale is not None:
                extra_attn_kwargs["q_descale"] = prologue_q_descale
            else:
                from sglang.kernels.ops.quantization.mxfp8_quant import to_mxfp8

                q_mxfp = to_mxfp8(q.view(num_tokens, self.num_tp_heads, self.head_dim))
                q = q_mxfp.data.view(num_tokens, -1)
                extra_attn_kwargs["q_descale"] = q_mxfp.scale.view(torch.float8_e8m0fnu)
            if (
                not prologue_did_store
                and not envs.SGLANG_OPT_INKLING_MXFP8_FUSED_QUANT_STORE.get()
            ):
                k_mxfp = to_mxfp8(
                    k.view(num_tokens, self.num_tp_kv_heads, self.head_dim)
                )
                v_mxfp = to_mxfp8(
                    v.view(num_tokens, self.num_tp_kv_heads, self.head_dim)
                )
                k = k_mxfp.data.view(num_tokens, -1)
                v = v_mxfp.data.view(num_tokens, -1)
                extra_attn_kwargs["k_descale"] = k_mxfp.scale.view(torch.float8_e8m0fnu)
                extra_attn_kwargs["v_descale"] = v_mxfp.scale.view(torch.float8_e8m0fnu)


        if (
            envs.SGLANG_OPT_USE_INKLING_SHEARED_BIAS.get()
            and fa4
            and not is_batch_invariant()
        ):


            attn_output = self.attn(
                q,
                k,
                v,
                forward_batch,
                save_kv_cache=not prologue_did_store,
                rel_bias=rel_logits,
                rel_bias_event=rel_event,
                **extra_attn_kwargs,
            )
        else:


            if rel_event is not None:
                get_current_device_stream_fast().wait_event(rel_event)
            if fa4:
                attn_output = self.attn(
                    q,
                    k,
                    v,
                    forward_batch,
                    save_kv_cache=not prologue_did_store,
                    score_mod=get_inkling_relative_attention_score_mod(self.rel_extent),
                    aux_tensors=[rel_logits],
                    **extra_attn_kwargs,
                )
            else:
                attn_output = self.attn(
                    q,
                    k,
                    v,
                    forward_batch,
                    save_kv_cache=not prologue_did_store,
                    score_mod=triton_relative_bias_score_mod,
                    aux_tensors=[rel_logits],
                    **extra_attn_kwargs,
                )
        attn_output = attn_output.view(num_tokens, -1)


        tp = get_parallel().attn_tp_group
        wo_ud = self.wo_ud


        is_lora_wrapped = not isinstance(wo_ud, RowParallelLinear)
        buf = (
            None
            if is_lora_wrapped
            else get_ar_buffer(tp, num_tokens, self.hidden_size, attn_output.dtype)
        )
        if buf is not None and type(wo_ud.quant_method) is UnquantizedLinearMethod:
            torch.matmul(attn_output, wo_ud.weight.t(), out=buf)
            bias_ = None if (wo_ud.tp_rank > 0 or wo_ud.skip_bias_add) else wo_ud.bias
            if bias_ is not None:
                buf.add_(bias_)
            if not reduce:
                return buf
            if self.scattered_sconv:


                return reduce_scatter_hidden(buf, tp, input_is_ar_buffer=True)
            return symm_mem_all_reduce(buf, tp, input_is_ar_buffer=True)

        result, _ = wo_ud(attn_output)
        if not reduce:
            return result
        if self.scattered_sconv:
            return reduce_scatter_hidden(result, tp)
        return symm_mem_all_reduce(result, tp)
