# SPDX-License-Identifier: Apache-2.0

import abc
from abc import ABC, abstractmethod

import torch


class Serializer(ABC):

    @abstractmethod
    def to_bytes(self, t: torch.Tensor) -> bytes:

        raise NotImplementedError


class Deserializer(metaclass=abc.ABCMeta):

    def __init__(self, dtype):
        self.dtype = dtype

    @abstractmethod
    def from_bytes(self, bs: bytes) -> torch.Tensor:

        raise NotImplementedError
