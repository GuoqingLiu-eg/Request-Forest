# SPDX-License-Identifier: Apache-2.0

import math
import re

import torch

from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_cuda

_is_cuda = is_cuda()


def load_gptoss_weight_quark(
    model,
    weights,
    *,
    is_nextn: bool,
    weight_name_mapping,
) -> None:


    quark_expert_pat = re.compile(
        r"^(.*\.mlp\.experts)\.(\d+)\.(gate_up_proj|down_proj)\."
        r"(weight|weight_scale|input_scale|bias)$"
    )
    quark_experts_weights = []
    normal_weights = []

    for name, weight in weights:
        if quark_expert_pat.match(name) is not None:
            quark_experts_weights.append((name, weight))
        else:
            normal_weights.append((name, weight))

    quark_loaded = _load_gptoss_quark_expert_weights(
        model, quark_experts_weights, quark_expert_pat
    )
    model._load_normal_weights(
        normal_weights,
        is_nextn=is_nextn,
        weight_name_mapping=weight_name_mapping,
        other_loaded_param_names=quark_loaded,
    )


def _load_gptoss_quark_expert_weights(model, weights, quark_expert_pat):

    params_dict = dict(model.named_parameters())
    loaded_params: set[str] = set()
    mxfp4_block = 32

    moe_tp_rank = get_parallel().moe_tp_rank
    moe_tp_size = get_parallel().moe_tp_size
    moe_ep_rank = get_parallel().moe_ep_rank
    moe_ep_size = get_parallel().moe_ep_size

    intermediate_size = model.config.intermediate_size
    assert (
        intermediate_size % mxfp4_block == 0
    ), f"{intermediate_size=} must be divisible by {mxfp4_block=}"
    intermediate_size_block = intermediate_size // mxfp4_block

    per_rank_intermediate_size_block = math.ceil(intermediate_size_block / moe_tp_size)

    per_rank_intermediate_size = per_rank_intermediate_size_block * mxfp4_block


    assert model.config.num_local_experts % moe_ep_size == 0
    moe_num_local_experts = model.config.num_local_experts // moe_ep_size

    moe_tp_rank_start = moe_tp_rank * per_rank_intermediate_size
    moe_tp_rank_end = min(
        (moe_tp_rank + 1) * per_rank_intermediate_size, intermediate_size
    )

    moe_ep_rank_start = moe_ep_rank * moe_num_local_experts
    moe_ep_rank_end = (moe_ep_rank + 1) * moe_num_local_experts

    for name, weight in weights:


        m = quark_expert_pat.match(name)
        if m is None:
            continue
        prefix, expert_str, proj, suffix = m.groups()
        global_expert_id = int(expert_str)
        if global_expert_id < moe_ep_rank_start or global_expert_id >= moe_ep_rank_end:
            continue
        local_expert_id = global_expert_id - moe_ep_rank_start

        if _is_cuda:
            weight = weight.cuda()

        dispatch_key = f"{proj}.{suffix}"

        if dispatch_key == "gate_up_proj.weight":

            new_name = f"{prefix}.w13_weight"


            narrow_gate = weight[0::2][moe_tp_rank_start:moe_tp_rank_end].contiguous()
            narrow_up = weight[1::2][moe_tp_rank_start:moe_tp_rank_end].contiguous()

            param = params_dict[new_name]
            intermediate_pad = param.data.shape[1] // 2
            g0, g1 = narrow_gate.shape
            u0, u1 = narrow_up.shape
            param.data[local_expert_id, :g0, :g1].copy_(
                narrow_gate.to(param.data.dtype)
            )
            param.data[
                local_expert_id,
                intermediate_pad : intermediate_pad + u0,
                :u1,
            ].copy_(narrow_up.to(param.data.dtype))
            loaded_params.add(new_name)

        elif dispatch_key == "down_proj.weight":


            new_name = f"{prefix}.w2_weight"

            narrow_weight = weight[
                ...,
                moe_tp_rank_start // 2 : moe_tp_rank_end // 2,
            ]

            param = params_dict[new_name]
            d0, d1 = narrow_weight.shape
            param.data[local_expert_id, :d0, :d1].copy_(
                narrow_weight.to(param.data.dtype)
            )
            loaded_params.add(new_name)

        elif dispatch_key == "gate_up_proj.weight_scale":

            new_name = f"{prefix}.w13_weight_scale"

            narrow_gate = weight[0::2][moe_tp_rank_start:moe_tp_rank_end].contiguous()
            narrow_up = weight[1::2][moe_tp_rank_start:moe_tp_rank_end].contiguous()

            param = params_dict[new_name]
            intermediate_pad = param.data.shape[1] // 2
            g0, g1 = narrow_gate.shape
            u0, u1 = narrow_up.shape
            param.data[local_expert_id, :g0, :g1].copy_(
                narrow_gate.to(param.data.dtype)
            )
            param.data[
                local_expert_id,
                intermediate_pad : intermediate_pad + u0,
                :u1,
            ].copy_(narrow_up.to(param.data.dtype))
            loaded_params.add(new_name)

        elif dispatch_key == "down_proj.weight_scale":


            new_name = f"{prefix}.w2_weight_scale"

            narrow_weight = weight[
                ...,
                moe_tp_rank_start // mxfp4_block : moe_tp_rank_end // mxfp4_block,
            ]

            param = params_dict[new_name]
            d0, d1 = narrow_weight.shape
            param.data[local_expert_id, :d0, :d1].copy_(
                narrow_weight.to(param.data.dtype)
            )
            loaded_params.add(new_name)

        elif dispatch_key == "gate_up_proj.bias":

            new_name = f"{prefix}.w13_weight_bias"

            narrow_gate = weight[0::2][moe_tp_rank_start:moe_tp_rank_end].contiguous()
            narrow_up = weight[1::2][moe_tp_rank_start:moe_tp_rank_end].contiguous()

            param = params_dict[new_name]
            intermediate_pad = param.data.shape[1] // 2
            param.data[local_expert_id, : narrow_gate.shape[0]].copy_(
                narrow_gate.to(param.data.dtype)
            )
            param.data[
                local_expert_id,
                intermediate_pad : intermediate_pad + narrow_up.shape[0],
            ].copy_(narrow_up.to(param.data.dtype))
            loaded_params.add(new_name)

        elif dispatch_key == "down_proj.bias":


            narrow_weight = weight
            if moe_tp_rank != 0:
                narrow_weight = torch.zeros_like(narrow_weight)

            new_name = f"{prefix}.w2_weight_bias"
            param = params_dict[new_name]
            d0 = narrow_weight.shape[0]
            param.data[local_expert_id, :d0].copy_(narrow_weight.to(param.data.dtype))
            loaded_params.add(new_name)

        elif dispatch_key == "gate_up_proj.input_scale":

            new_name = f"{prefix}.w13_input_scale"
            if new_name not in params_dict:

                continue

            param = params_dict[new_name]
            param.data[local_expert_id].copy_(weight.to(param.data.dtype).reshape(()))
            loaded_params.add(new_name)

        elif dispatch_key == "down_proj.input_scale":

            new_name = f"{prefix}.w2_input_scale"
            if new_name not in params_dict:

                continue

            param = params_dict[new_name]
            param.data[local_expert_id].copy_(weight.to(param.data.dtype).reshape(()))
            loaded_params.add(new_name)

    return loaded_params
