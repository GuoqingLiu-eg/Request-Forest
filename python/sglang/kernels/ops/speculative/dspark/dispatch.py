from __future__ import annotations

import torch


def inputs_on_cuda(*args, **kwargs) -> bool:

    for value in (*args, *kwargs.values()):
        if isinstance(value, torch.Tensor):
            return value.is_cuda
    raise AssertionError("kernel dispatch requires at least one tensor argument")
