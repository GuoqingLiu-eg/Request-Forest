# Copyright 2023-2024 SGLang Team
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

from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from typing import TYPE_CHECKING, Optional

import torch
from torch import nn

from sglang.srt.compilation.compilation_config import register_split_op
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    get_tc_piecewise_forward_context,
)
from sglang.srt.utils.common import is_hip
from sglang.srt.utils.custom_op import register_custom_op

_is_hip = is_hip()


_force_eager_attn: ContextVar[bool] = ContextVar("_force_eager_attn", default=False)


@contextmanager
def force_eager_attention():
    token = _force_eager_attn.set(True)
    try:
        yield
    finally:
        _force_eager_attn.reset(token)


def _zero_padded_pcg_tail(buf: torch.Tensor, context) -> None:

    pcg_static_tokens = context.num_tokens
    actual_tokens = context.raw_num_tokens
    if (
        pcg_static_tokens is not None
        and actual_tokens is not None
        and pcg_static_tokens > actual_tokens
    ):
        first_dim = buf.shape[0]
        elems_per_token = buf.numel() // first_dim
        buf.view(first_dim, elems_per_token)[actual_tokens:].zero_()


if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class AttentionType(Enum):


    DECODER = "decoder"

    DECODER_BIDIRECTIONAL = "decoder_bidirectional"

    ENCODER_ONLY = "encoder_only"


class RadixAttention(nn.Module):


    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scaling: float,
        num_kv_heads: int,
        layer_id: int,
        logit_cap: float = 0.0,
        v_head_dim: int = -1,
        sliding_window_size: int = -1,
        is_cross_attention: bool = False,
        pos_encoding_mode: str = "NONE",
        logit_capping_method: str = "tanh",
        quant_config: Optional[QuantizationConfig] = None,
        attn_type: AttentionType = AttentionType.DECODER,
        use_irope: bool = False,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_q_head_num = num_heads
        self.tp_k_head_num = num_kv_heads
        self.tp_v_head_num = num_kv_heads
        self.head_dim = head_dim
        self.qk_head_dim = head_dim
        self.v_head_dim = v_head_dim if v_head_dim != -1 else head_dim
        self.scaling = scaling
        self.layer_id = layer_id
        self.logit_cap = logit_cap
        self.sliding_window_size = sliding_window_size or -1
        self.is_cross_attention = is_cross_attention
        self.use_irope = use_irope
        self.k_scale = None
        self.v_scale = None
        self.k_scale_float = None
        self.v_scale_float = None


        self.q_scale_float = None
        self.idx_q_scale_float = None
        self.idx_k_scale_float = None
        self.idx_v_scale_float = None
        self.quant_method = None

        if quant_config is not None:
            self.quant_method = quant_config.get_quant_method(self, prefix=prefix)
        if self.quant_method is not None:
            self.quant_method.create_weights(self)
        self.attn_type = attn_type

        self.pos_encoding_mode = pos_encoding_mode
        self.logit_capping_method = logit_capping_method
        self.xai_temperature_len = -1

    def forward(
        self,
        q,
        k,
        v,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        key_value_num_tokens: Optional[int] = None,
        **kwargs,
    ):
        if k is not None:

            assert v is not None
            if "k_rope" not in kwargs:
                k = k.view(-1, self.tp_k_head_num, self.qk_head_dim)
                v = v.view(-1, self.tp_v_head_num, self.v_head_dim)
            else:
                k = k.view(-1, self.tp_k_head_num, self.v_head_dim)

        context = get_tc_piecewise_forward_context()
        if (
            forward_batch.forward_mode.is_extend()
            and context is not None


            and (torch.compiler.is_compiling() or not _force_eager_attn.get())
        ):
            if kwargs.get("idx_q") is not None:
                if is_in_breakable_cuda_graph():
                    return get_attn_backend().forward(
                        q, k, v, self, forward_batch, save_kv_cache, **kwargs
                    )
                idx_q = kwargs["idx_q"]
                idx_k = kwargs["idx_k"]
                idx_v = kwargs.get("idx_v")
                attn_out = q.new_empty(
                    (q.shape[0], self.tp_q_head_num * self.v_head_dim)
                )
                idx_out = q.new_empty((q.shape[0], idx_q.shape[1] * idx_q.shape[2]))
                unified_sparse_attention_with_output(
                    q,
                    k,
                    v,
                    attn_out,
                    idx_out,
                    idx_q,
                    idx_k,
                    save_kv_cache,
                    self.layer_id,
                    idx_v=idx_v,
                )
                return idx_out, attn_out


            out_dtype = v.dtype if v is not None else q.dtype
            if out_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                out_dtype = torch.bfloat16
            if self.qk_head_dim != self.v_head_dim:
                output = q.new_empty(
                    (q.shape[0], self.tp_q_head_num * self.v_head_dim),
                    dtype=out_dtype,
                )
            else:
                output = torch.empty_like(q, dtype=out_dtype)
            if any(
                key in kwargs
                for key in (
                    "score_mod",
                    "aux_tensors",
                    "rel_bias",
                    "q_descale",
                    "k_descale",
                    "v_descale",
                )
            ):


                if is_in_breakable_cuda_graph():
                    breakable_attention_with_output_extra_kwargs(
                        q, k, v, output, save_kv_cache, self.layer_id, kwargs
                    )
                else:
                    attention_with_output_extra_kwargs(
                        q, k, v, output, save_kv_cache, self.layer_id, kwargs
                    )
                return output


            return_lse = bool(forward_batch.mha_return_lse)
            mha_companion_layers = context.mha_companion_layers
            use_mha_companion = (
                mha_companion_layers is not None
                and mha_companion_layers[self.layer_id] is self
            )
            if is_in_breakable_cuda_graph():
                op = (
                    breakable_unified_attention_with_output_and_lse
                    if return_lse
                    else breakable_unified_attention_with_output
                )
            else:
                op = (
                    unified_attention_with_output_and_lse
                    if return_lse
                    else unified_attention_with_output
                )
            lse = op(
                q,
                k,
                v,
                output,
                save_kv_cache,
                self.layer_id,
                use_mha_companion=use_mha_companion,
                key_value_num_tokens=key_value_num_tokens,
                **kwargs,
            )
            if return_lse:
                return output.view(-1, self.tp_q_head_num, self.v_head_dim), lse
            return output
        else:
            return get_attn_backend().forward(
                q,
                k,
                v,
                self,
                forward_batch,
                save_kv_cache,
                **kwargs,
            )


def _unified_attention_with_output_impl(
    query: torch.Tensor,
    key: Optional[torch.Tensor],
    value: Optional[torch.Tensor],
    output: torch.Tensor,
    save_kv_cache: bool,
    layer_id: int,
    use_mha_companion: bool,
    return_lse: bool,
    *,
    key_value_num_tokens: Optional[int] = None,
    q_rope: Optional[torch.Tensor] = None,
    k_rope: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,


    cos_sin_cache: Optional[torch.Tensor] = None,
    is_neox: Optional[bool] = None,
    llama_4_scaling: Optional[torch.Tensor] = None,
    topk_indices: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    context = get_tc_piecewise_forward_context()
    forward_batch = context.forward_batch
    attention_layers = context.attention_layers
    attention_layer = attention_layers[layer_id]
    real_query_num_tokens = forward_batch.num_token_non_padded_cpu


    if key_value_num_tokens is None:
        key_value_num_tokens = real_query_num_tokens

    query = query[:real_query_num_tokens]
    if key is not None:
        key = key[:key_value_num_tokens]
    if value is not None:
        value = value[:key_value_num_tokens]


    if use_mha_companion:
        assert context.mha_companion_layers is not None
        attention_layer = context.mha_companion_layers[layer_id]
        assert attention_layer is not None

    kwargs = {}
    if q_rope is not None:
        kwargs["q_rope"] = q_rope[:real_query_num_tokens]
    if k_rope is not None:
        kwargs["k_rope"] = k_rope[:key_value_num_tokens]
    if sinks is not None:
        kwargs["sinks"] = sinks
    if cos_sin_cache is not None:
        kwargs["cos_sin_cache"] = cos_sin_cache
    if is_neox is not None:
        kwargs["is_neox"] = is_neox
    if llama_4_scaling is not None:
        kwargs["llama_4_scaling"] = llama_4_scaling
    if topk_indices is not None:
        kwargs["topk_indices"] = topk_indices[:real_query_num_tokens]

    original_out_cache_loc = forward_batch.out_cache_loc
    original_positions = forward_batch.positions


    forward_batch.out_cache_loc = original_out_cache_loc[:real_query_num_tokens]
    if original_positions is not None:
        forward_batch.positions = original_positions[:real_query_num_tokens]


    forward_batch._attn_output = output[:real_query_num_tokens]

    ret = get_attn_backend().forward(
        query,
        key,
        value,
        attention_layer,
        forward_batch,
        save_kv_cache,
        **kwargs,
    )
    forward_batch.out_cache_loc = original_out_cache_loc
    forward_batch.positions = original_positions

    lse = None
    if return_lse:
        assert isinstance(ret, tuple)
        ret, lse, *_ = ret
    else:
        assert isinstance(ret, torch.Tensor)

    if ret.data_ptr() != output.data_ptr():
        output[:real_query_num_tokens].view(ret.shape).copy_(ret)


    _zero_padded_pcg_tail(output, context)
    if lse is not None and lse.shape[0] != output.shape[0]:
        padded_lse = lse.new_zeros((output.shape[0], *lse.shape[1:]))
        padded_lse[:real_query_num_tokens].copy_(lse)
        lse = padded_lse
    return lse


@register_custom_op(mutates_args=["output"])
@register_split_op()
def unified_attention_with_output(
    query: torch.Tensor,
    key: Optional[torch.Tensor],
    value: Optional[torch.Tensor],
    output: torch.Tensor,
    save_kv_cache: bool,
    layer_id: int,
    *,
    use_mha_companion: bool = False,
    key_value_num_tokens: Optional[int] = None,
    q_rope: Optional[torch.Tensor] = None,
    k_rope: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    cos_sin_cache: Optional[torch.Tensor] = None,
    is_neox: Optional[bool] = None,
    llama_4_scaling: Optional[torch.Tensor] = None,
    topk_indices: Optional[torch.Tensor] = None,
) -> None:
    _unified_attention_with_output_impl(
        query,
        key,
        value,
        output,
        save_kv_cache,
        layer_id,
        use_mha_companion,
        False,
        key_value_num_tokens=key_value_num_tokens,
        q_rope=q_rope,
        k_rope=k_rope,
        sinks=sinks,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
        llama_4_scaling=llama_4_scaling,
        topk_indices=topk_indices,
    )


def _unified_attention_with_output_and_lse_fake(
    query: torch.Tensor, *args, **kwargs
) -> torch.Tensor:
    return query.new_empty((query.shape[0], query.shape[1]), dtype=torch.float32)


@register_custom_op(
    mutates_args=["output"], fake_impl=_unified_attention_with_output_and_lse_fake
)
@register_split_op()
def unified_attention_with_output_and_lse(
    query: torch.Tensor,
    key: Optional[torch.Tensor],
    value: Optional[torch.Tensor],
    output: torch.Tensor,
    save_kv_cache: bool,
    layer_id: int,
    *,
    use_mha_companion: bool = False,
    key_value_num_tokens: Optional[int] = None,
    q_rope: Optional[torch.Tensor] = None,
    k_rope: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    cos_sin_cache: Optional[torch.Tensor] = None,
    is_neox: Optional[bool] = None,
    llama_4_scaling: Optional[torch.Tensor] = None,
    topk_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    lse = _unified_attention_with_output_impl(
        query,
        key,
        value,
        output,
        save_kv_cache,
        layer_id,
        use_mha_companion,
        True,
        key_value_num_tokens=key_value_num_tokens,
        q_rope=q_rope,
        k_rope=k_rope,
        sinks=sinks,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
        llama_4_scaling=llama_4_scaling,
        topk_indices=topk_indices,
    )
    assert lse is not None
    return lse


@register_custom_op(mutates_args=["attn_out", "idx_out"])
@register_split_op()
def unified_sparse_attention_with_output(
    query: torch.Tensor,
    key: Optional[torch.Tensor],
    value: Optional[torch.Tensor],
    attn_out: torch.Tensor,
    idx_out: torch.Tensor,
    idx_q: torch.Tensor,
    idx_k: torch.Tensor,
    save_kv_cache: bool,
    layer_id: int,
    *,
    idx_v: Optional[torch.Tensor] = None,
) -> None:
    context = get_tc_piecewise_forward_context()
    forward_batch = context.forward_batch
    attention_layer = context.attention_layers[layer_id]
    real_num_tokens = forward_batch.num_token_non_padded_cpu

    query = query[:real_num_tokens]
    if key is not None:
        key = key[:real_num_tokens]
    if value is not None:
        value = value[:real_num_tokens]
    idx_q = idx_q[:real_num_tokens]
    idx_k = idx_k[:real_num_tokens]
    if idx_v is not None:
        idx_v = idx_v[:real_num_tokens]

    original_out_cache_loc = forward_batch.out_cache_loc
    forward_batch.out_cache_loc = original_out_cache_loc[:real_num_tokens]

    ret_idx, ret_out = get_attn_backend().forward(
        query,
        key,
        value,
        attention_layer,
        forward_batch,
        save_kv_cache,
        idx_q=idx_q,
        idx_k=idx_k,
        idx_v=idx_v,
    )
    forward_batch.out_cache_loc = original_out_cache_loc

    attn_out[:real_num_tokens].view(ret_out.shape).copy_(ret_out)


    if ret_idx is not None:
        idx_out[:real_num_tokens].view(ret_idx.shape).copy_(ret_idx)

    for buf in (attn_out, idx_out):
        _zero_padded_pcg_tail(buf, context)
    return


breakable_unified_attention_with_output = eager_on_graph(True)(
    unified_attention_with_output
)
breakable_unified_attention_with_output_and_lse = eager_on_graph(True)(
    unified_attention_with_output_and_lse
)


def attention_with_output_extra_kwargs(
    query: torch.Tensor,
    key: Optional[torch.Tensor],
    value: Optional[torch.Tensor],
    output: torch.Tensor,
    save_kv_cache: bool,
    layer_id: int,
    extra_kwargs: dict,
) -> None:

    context = get_tc_piecewise_forward_context()
    forward_batch = context.forward_batch
    attention_layer = context.attention_layers[layer_id]
    real_num_tokens = forward_batch.num_token_non_padded_cpu

    query = query[:real_num_tokens]
    if key is not None:
        key = key[:real_num_tokens]
    if value is not None:
        value = value[:real_num_tokens]

    kwargs = dict(extra_kwargs)
    aux_tensors = kwargs.get("aux_tensors")
    if aux_tensors is not None:
        kwargs["aux_tensors"] = [t[:real_num_tokens] for t in aux_tensors]
    for per_token_key in ("rel_bias", "q_descale", "k_descale", "v_descale"):
        t = kwargs.get(per_token_key)
        if t is not None:
            kwargs[per_token_key] = t[:real_num_tokens]

    original_out_cache_loc = forward_batch.out_cache_loc
    forward_batch.out_cache_loc = original_out_cache_loc[:real_num_tokens]
    forward_batch._attn_output = output[:real_num_tokens]

    ret = get_attn_backend().forward(
        query, key, value, attention_layer, forward_batch, save_kv_cache, **kwargs
    )
    forward_batch.out_cache_loc = original_out_cache_loc

    if ret.data_ptr() != output.data_ptr():
        output[:real_num_tokens].view(ret.shape).copy_(ret)

    if _is_hip:
        _zero_padded_pcg_tail(output, context)
    return


breakable_attention_with_output_extra_kwargs = eager_on_graph(True)(
    attention_with_output_extra_kwargs
)
