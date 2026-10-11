# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/ops/causal_conv1d.py
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2024, Tri Dao.
# Adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py

from typing import Optional

import torch

from sglang.kernels.ops.mamba.causal_conv1d_triton import (
    PAD_SLOT_ID,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import (
    causal_conv1d_fn as _causal_conv1d_fn_triton,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import (
    causal_conv1d_update as _causal_conv1d_update_triton,
)
from sglang.srt.utils import is_cuda


_HAS_CONV1D_KERNEL = is_cuda()

if _HAS_CONV1D_KERNEL:
    from sglang.kernels.ops.mamba import (
        causal_conv1d_fwd,
    )
    from sglang.kernels.ops.mamba import (
        causal_conv1d_update as causal_conv1d_update_kernel,
    )


def _get_seq_lens_cpu(query_start_loc, x):
    if query_start_loc is not None:
        return (query_start_loc[1:] - query_start_loc[:-1]).cpu().tolist()
    return [x.shape[-1]]


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    conv_states: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    **kwargs,
):


    use_triton = not _HAS_CONV1D_KERNEL or (
        x.stride(-1) != 1 and "seq_lens_cpu" in kwargs
    )
    if use_triton:
        if "seq_lens_cpu" not in kwargs:
            kwargs["seq_lens_cpu"] = _get_seq_lens_cpu(query_start_loc, x)
        return _causal_conv1d_fn_triton(
            x,
            weight,
            bias,
            conv_states=conv_states,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            activation=activation,
            pad_slot_id=pad_slot_id,
            **kwargs,
        )
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    if x.stride(-1) != 1:
        x = x.contiguous()
    bias = bias.contiguous() if bias is not None else None

    if cache_indices is not None and cache_indices.dtype != torch.int32:
        cache_indices = cache_indices.to(torch.int32)
    causal_conv1d_fwd(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        activation in ["silu", "swish"],
        pad_slot_id,
    )
    return x


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
):

    use_triton = not _HAS_CONV1D_KERNEL
    if use_triton:
        return _causal_conv1d_update_triton(
            x,
            conv_state,
            weight,
            bias=bias,
            activation=activation,
            cache_seqlens=cache_seqlens,
            conv_state_indices=conv_state_indices,
            pad_slot_id=pad_slot_id,
        )
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError(
            f"activation must be None, silu, or swish, actual: {activation}"
        )
    activation_val = activation in ["silu", "swish"]
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    if conv_state_indices is not None and conv_state_indices.dtype != torch.int32:
        conv_state_indices = conv_state_indices.to(torch.int32)
    causal_conv1d_update_kernel(
        x,
        conv_state,
        weight,
        bias,
        activation_val,
        cache_seqlens,
        conv_state_indices,
        pad_slot_id,
    )
    if unsqueeze:
        x = x.squeeze(-1)
    return x
