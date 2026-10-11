# Copyright 2023-2025 SGLang Team
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

from collections.abc import Callable, Iterable
from typing import Union

import msgspec
import torch
from torch import nn
from torch.nn import Parameter

from sglang.srt.layers.utils.common import get_layer_id
from sglang.srt.models.utils import AutoWeightsLoader, WeightsMapper

__all__ = [
    "AutoWeightsLoader",
    "WeightsMapper",
    "StackedParamsDispatch",
    "STANDARD_QKV_MAPPING",
    "STANDARD_GATE_UP_MAPPING",
    "STANDARD_STACKED_MAPPING",
    "LLAMA_STACKED_MAPPING",
    "filter_pp_weights",
    "register_weight_remap",
    "get_weight_remap",
]


class StackedParamsDispatch(msgspec.Struct, frozen=True):


    mappings: tuple[tuple[str, str, Union[int, str]], ...] = ()

    def try_load(
        self,
        name: str,
        tensor: torch.Tensor,
        params_dict: dict[str, Parameter],
    ) -> str | None:

        for fused_name, source_name, shard_id in self.mappings:
            if source_name not in name:
                continue
            target = name.replace(source_name, fused_name)
            param = params_dict.get(target)
            if param is None:


                return target
            param.weight_loader(param, tensor, shard_id)
            return target
        return None


STANDARD_QKV_MAPPING = StackedParamsDispatch(
    mappings=(
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
    )
)

STANDARD_GATE_UP_MAPPING = StackedParamsDispatch(
    mappings=(
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    )
)

STANDARD_STACKED_MAPPING = StackedParamsDispatch(
    mappings=(
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    )
)


LLAMA_STACKED_MAPPING = StackedParamsDispatch(
    mappings=(
        (".qkv_proj", ".q_proj", "q"),
        (".qkv_proj", ".k_proj", "k"),
        (".qkv_proj", ".v_proj", "v"),
        (".gate_up_proj", ".gate_proj", 0),
        (".gate_up_proj", ".up_proj", 1),
    )
)


def filter_pp_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    start_layer: int,
    end_layer: int,
) -> Iterable[tuple[str, torch.Tensor]]:

    for name, tensor in weights:
        layer_id = get_layer_id(name)
        if layer_id is not None and (layer_id < start_layer or layer_id >= end_layer):
            continue
        yield name, tensor


_REMAP_REGISTRY: dict[str, Callable[[nn.Module], WeightsMapper]] = {}


def register_weight_remap(*class_names: str):


    def decorator(fn: Callable[[nn.Module], WeightsMapper]):
        for cn in class_names:
            _REMAP_REGISTRY[cn] = fn
        return fn

    return decorator


def get_weight_remap(model: nn.Module) -> WeightsMapper | None:

    fn = _REMAP_REGISTRY.get(type(model).__name__)
    if fn is None:
        return None
    return fn(model)


@register_weight_remap("LlamaForCausalLM")
def _llama_remap(model: nn.Module) -> WeightsMapper:

    return WeightsMapper(
        orig_to_new_suffix={
            ".activation_scale": ".input_scale",
            ".weight_scale_inv": ".weight_scale",
        }
    )
