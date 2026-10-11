# SPDX-License-Identifier: Apache-2.0


import logging
from typing import Optional

import torch

from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)
from sglang.srt.runtime_context import mamba_cache_chunk_size

logger = logging.getLogger(__name__)

_CHUNK = 64
_PAD_BUCKETS = (1024, 2048, 4096, 8192, 16384, 32768)


_PAD_GATE = -1000.0


class PtxKDAKernel(LinearAttnKernelBase):
    def __init__(self):

        self.supports_prefill = torch.cuda.is_available() and (
            torch.cuda.get_device_capability() == (10, 3)
        )
        self._fwd = None
        self._triton = TritonKDAKernel()


        self._param_flat = {}
        self._unsupported_logged = False

        self._staging = {}

    def _ensure_loaded(self):
        if self._fwd is None:
            from sglang.kernels.ops.attention.linear.kda_ptx_prefill import (
                chunk_kda_fwd,
                load_ext,
            )

            logger.info("Building the PTX KDA prefill extension (first use, ~1-2 min)")
            load_ext()
            self._fwd = chunk_kda_fwd
            logger.info("Using PTX KDA chunked prefill (GB300 / sm_103a)")

    def decode(self, *args, **kwargs):
        raise NotImplementedError("PtxKDAKernel is prefill-only")

    def target_verify(self, *args, **kwargs):
        raise NotImplementedError("PtxKDAKernel does not support target_verify")

    def _flat_param(self, t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if t is None:
            return None
        key = (t.data_ptr(), t.dtype, tuple(t.shape))
        flat = self._param_flat.get(key)
        if flat is None:
            flat = t.detach().reshape(-1).float().contiguous()
            self._param_flat[key] = flat
        return flat

    def _get_staging(self, bucket, num_heads, head_k_dim, head_v_dim, dev):
        key = (bucket, num_heads, head_k_dim, head_v_dim, dev.index)
        st = self._staging.get(key)
        if st is None:
            st = {
                "q": torch.zeros(
                    1, bucket, num_heads, head_k_dim, dtype=torch.bfloat16, device=dev
                ),
                "k": torch.zeros(
                    1, bucket, num_heads, head_k_dim, dtype=torch.bfloat16, device=dev
                ),
                "v": torch.zeros(
                    1, bucket, num_heads, head_v_dim, dtype=torch.bfloat16, device=dev
                ),
                "g": torch.full(
                    (1, bucket, num_heads, head_k_dim),
                    _PAD_GATE,
                    dtype=torch.bfloat16,
                    device=dev,
                ),
                "beta": torch.zeros(
                    1, bucket, num_heads, dtype=torch.bfloat16, device=dev
                ),
            }
            self._staging[key] = st
        return st

    def _triton_extend(
        self,
        q,
        k,
        v,
        g,
        beta,
        ssm_states,
        cache_indices,
        query_start_loc,
        A_log,
        dt_bias,
        lower_bound,
        return_intermediate_states,
        kwargs,
    ):
        return self._triton.extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            return_intermediate_states=return_intermediate_states,
            **kwargs,
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        lower_bound: Optional[float] = None,
        return_intermediate_states: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        num_tokens = q.shape[1]
        seq_lens_cpu = kwargs.get("extend_seq_lens_cpu")
        track_h_src = kwargs.get("track_ssm_h_src")
        supported_shape = (
            q.ndim == 4
            and k.ndim == 4
            and v.ndim == 4
            and q.shape[-1] == 128
            and k.shape[-1] == 128
            and v.shape[-1] == 128
            and q.dtype == torch.bfloat16
            and k.dtype == torch.bfloat16
            and v.dtype == torch.bfloat16
            and g.dtype == torch.bfloat16
            and A_log is not None
        )
        needs_interior_snapshot = (
            return_intermediate_states
            and track_h_src is not None
            and track_h_src.numel() > 0
        )
        intermediate_stride_supported = True
        if needs_interior_snapshot:


            intermediate_stride_supported = mamba_cache_chunk_size() == _CHUNK
        seq_lens = (
            [int(length) for length in seq_lens_cpu] if seq_lens_cpu is not None else []
        )
        shape_known = (
            len(seq_lens) >= 1
            and len(seq_lens) == cache_indices.numel()
            and sum(seq_lens) == num_tokens
            and min(seq_lens, default=0) > 0
        )


        padded_tokens = num_tokens
        if len(seq_lens) == 1 and num_tokens % _CHUNK != 0:
            padded_tokens = next(
                (bucket for bucket in _PAD_BUCKETS if bucket >= num_tokens),
                num_tokens,
            )
        eligible = (
            not kwargs.get("is_spec_decode")
            and intermediate_stride_supported
            and shape_known
            and supported_shape
        )
        if not eligible:
            if shape_known and not supported_shape and not self._unsupported_logged:
                self._unsupported_logged = True
                logger.warning(
                    "PTX KDA prefill only supports BF16 q/k/v/g with K=V=128 and "
                    "A_log; got q/k/v/g=%s/%s/%s/%s, Kq/Kk/V=%d/%d/%d, A_log=%s. "
                    "Falling back to Triton.",
                    q.dtype,
                    k.dtype,
                    v.dtype,
                    g.dtype,
                    q.shape[-1],
                    k.shape[-1],
                    v.shape[-1],
                    A_log is not None,
                )
            return self._triton_extend(
                q,
                k,
                v,
                g,
                beta,
                ssm_states,
                cache_indices,
                query_start_loc,
                A_log,
                dt_bias,
                lower_bound,
                return_intermediate_states,
                kwargs,
            )

        self._ensure_loaded()

        num_heads = q.shape[2]
        head_k_dim = q.shape[-1]
        head_v_dim = v.shape[-1]

        slot = torch.where(
            cache_indices >= 0, cache_indices, ssm_states.shape[0] - 1
        ).to(torch.int64)
        state_backup = ssm_states.index_select(0, slot).clone()

        try:
            qn = q
            kn = k
            g4 = g.reshape(1, num_tokens, num_heads, head_k_dim)
            b3 = beta.reshape(1, num_tokens, num_heads)
            if padded_tokens != num_tokens:
                st = self._get_staging(
                    padded_tokens, num_heads, head_k_dim, head_v_dim, q.device
                )
                for name, src in (("q", qn), ("k", kn), ("v", v), ("g", g4)):
                    st[name][0, :num_tokens].copy_(src[0])
                st["beta"][0, :num_tokens].copy_(b3[0])
                st["q"][0, num_tokens:].zero_()
                st["k"][0, num_tokens:].zero_()
                st["v"][0, num_tokens:].zero_()
                st["g"][0, num_tokens:].fill_(_PAD_GATE)
                st["beta"][0, num_tokens:].zero_()
                qn, kn, v_in, g4, b3 = st["q"], st["k"], st["v"], st["g"], st["beta"]
            else:
                v_in = v


            cu_kwargs = {}
            if len(seq_lens) > 1 or padded_tokens % _CHUNK != 0:
                cu_cpu = torch.zeros(len(seq_lens) + 1, dtype=torch.int32)
                cu_cpu[1:] = torch.tensor(seq_lens, dtype=torch.int32).cumsum(0)
                cu_kwargs = dict(cu_seqlens=query_start_loc, cu_seqlens_cpu=cu_cpu)

            result = self._fwd(
                qn,
                kn,
                v_in,
                g4,
                b3,
                scale=head_k_dim**-0.5,
                initial_state=ssm_states.index_select(0, slot),
                output_final_state=True,
                **cu_kwargs,


                state_v_first=True,
                safe_gate=lower_bound is not None,
                lower_bound=lower_bound,
                use_gate_in_kernel=True,
                A_log=self._flat_param(A_log),
                dt_bias=self._flat_param(dt_bias),
                return_intermediate_states=return_intermediate_states,
                use_qk_l2norm_in_kernel=True,
            )
            out, final_state, h = result[0], result[1], result[10]
            ssm_states.index_copy_(0, slot, final_state.to(ssm_states.dtype))
        except Exception:
            ssm_states.index_copy_(0, slot, state_backup)
            logger.warning(
                "PTX KDA prefill failed (T=%d); restored state and fell back to "
                "Triton for this batch",
                num_tokens,
                exc_info=True,
            )
            return self._triton_extend(
                q,
                k,
                v,
                g,
                beta,
                ssm_states,
                cache_indices,
                query_start_loc,
                A_log,
                dt_bias,
                lower_bound,
                return_intermediate_states,
                kwargs,
            )

        packed_output = out[:, :num_tokens].contiguous()
        if return_intermediate_states:
            return packed_output, h
        return packed_output
