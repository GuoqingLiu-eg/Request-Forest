# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import abstractmethod
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.quantization.base_scheme import BaseLinearScheme, BaseMoEScheme

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput

__all__ = ["CompressedTensorsLinearScheme", "CompressedTensorsMoEScheme"]


class CompressedTensorsLinearScheme(BaseLinearScheme):


    @classmethod
    def get_min_capability(cls) -> int:

        raise NotImplementedError

    @abstractmethod
    def create_weights(self, *args, **kwargs):

        raise NotImplementedError

    @abstractmethod
    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor]
    ):

        raise NotImplementedError

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module):

        raise NotImplementedError


class CompressedTensorsMoEScheme(BaseMoEScheme):


    load_up_proj_weight_first = False

    @classmethod
    def get_min_capability(cls) -> int:

        raise NotImplementedError

    @abstractmethod
    def create_weights(self, *args, **kwargs):

        raise NotImplementedError

    @abstractmethod
    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        raise NotImplementedError

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module):

        raise NotImplementedError

    @abstractmethod
    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ):

        raise NotImplementedError
