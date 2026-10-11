# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import abstractmethod
from typing import Optional

import torch

from sglang.srt.layers.quantization.base_scheme import BaseLinearScheme, BaseMoEScheme

__all__ = ["ModelSlimLinearScheme", "ModelSlimMoEScheme"]


class ModelSlimLinearScheme(BaseLinearScheme):


    @abstractmethod
    def create_weights(self, *args, **kwargs):

        raise NotImplementedError

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module):

        raise NotImplementedError

    @abstractmethod
    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor]
    ):

        raise NotImplementedError


class ModelSlimMoEScheme(BaseMoEScheme):


    @abstractmethod
    def create_weights(self, *args, **kwargs):

        raise NotImplementedError

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module):

        raise NotImplementedError
