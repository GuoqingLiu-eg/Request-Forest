# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Optional

import torch

__all__ = ["BaseLinearScheme", "BaseMoEScheme"]


class BaseLinearScheme(ABC):


    requires_weight_loader_v2: bool = False

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


class BaseMoEScheme(ABC):


    @abstractmethod
    def create_weights(self, *args, **kwargs):

        raise NotImplementedError

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module):

        raise NotImplementedError
