# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://raw.githubusercontent.com/vllm-project/vllm/v0.5.5/vllm/model_executor/layers/quantization/base_config.py
from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

import torch
from torch import nn

from sglang.srt.layers.modelopt_utils import canonicalize_modelopt_quant_algo

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput
    from sglang.srt.models.utils import WeightsMapper


class QuantizeMethodBase(ABC):


    def create_weights(
        self, layer: torch.nn.Module, *weight_args, **extra_weight_attrs
    ):

        raise NotImplementedError()

    @abstractmethod
    def apply(self, layer: torch.nn.Module, *args, **kwargs) -> torch.Tensor:

        raise NotImplementedError()

    def process_weights_after_loading(self, layer: nn.Module) -> None:

        return


class LinearMethodBase(QuantizeMethodBase):


    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):

        raise NotImplementedError()

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        raise NotImplementedError()


class FusedMoEMethodBase(QuantizeMethodBase):

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        raise NotImplementedError

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:
        raise NotImplementedError

    def get_triton_quant_info(self, layer: torch.nn.Module) -> TritonMoeQuantInfo:

        raise NotImplementedError(
            f"{type(self).__name__} must implement get_triton_quant_info()"
        )


class QuantizationConfig(ABC):


    weight_block_size: Optional[List[int]] = None

    def __init__(self):
        super().__init__()

        self.packed_modules_mapping: Dict[str, List[str]] = dict()

    def update_packed_modules_mapping(self, mapping: Dict[str, List[str]]) -> None:
        self.packed_modules_mapping = mapping

    @abstractmethod
    def get_name(self) -> str:

        raise NotImplementedError()

    @abstractmethod
    def get_supported_act_dtypes(self) -> List[torch.dtype]:

        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def get_min_capability(cls) -> int:

        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_config_filenames() -> List[str]:

        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> QuantizationConfig:

        raise NotImplementedError()

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:

        return None

    @classmethod
    def _modelopt_override_quantization_method(
        cls, hf_quant_config, user_quant
    ) -> Optional[str]:

        if hf_quant_config is None:
            return None


        from sglang.srt.configs.model_config import REQUANTIZATION_METHODS

        if user_quant == "nvfp4_online" or user_quant in REQUANTIZATION_METHODS:
            return None


        quant_algo = hf_quant_config.get("quant_algo", "").upper()


        if user_quant == "modelopt":
            canonical_method = canonicalize_modelopt_quant_algo(quant_algo)
            if canonical_method is not None:
                return canonical_method


        if hf_quant_config.get("quant_method", "") == "modelopt_fp8":
            return "modelopt_fp8"
        elif hf_quant_config.get("quant_method", "") == "modelopt_fp4":
            return "modelopt_fp4"

        return None

    @staticmethod
    def get_from_keys(config: Dict[str, Any], keys: List[str]) -> Any:

        for key in keys:
            if key in config:
                return config[key]
        raise ValueError(
            f"Cannot find any of {keys} in the model's " "quantization config."
        )

    @staticmethod
    def get_from_keys_or(config: Dict[str, Any], keys: List[str], default: Any) -> Any:

        try:
            return QuantizationConfig.get_from_keys(config, keys)
        except ValueError:
            return default

    @abstractmethod
    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:

        raise NotImplementedError()

    @abstractmethod
    def get_scaled_act_names(self) -> List[str]:

        raise NotImplementedError()

    def apply_weight_name_mapper(
        self, hf_to_sglang_mapper: WeightsMapper
    ):

        pass


def method_has_implemented_embedding(method_class: Type[QuantizeMethodBase]) -> bool:

    base_embedding = inspect.getattr_static(QuantizeMethodBase, "embedding", None)
    class_embedding = inspect.getattr_static(method_class, "embedding", None)

    return class_embedding is not None and class_embedding is not base_embedding
