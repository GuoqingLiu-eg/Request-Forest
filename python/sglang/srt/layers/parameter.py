# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import logging
from fractions import Fraction
from typing import Callable, Optional, Union

import torch
from torch.nn import Parameter

from sglang.srt.environ import envs
from sglang.srt.layers.utils import pad_or_narrow_weight
from sglang.srt.utils import is_cpu

__all__ = [
    "BasevLLMParameter",
    "PackedvLLMParameter",
    "PerTensorScaleParameter",
    "ModelWeightParameter",
    "ChannelQuantScaleParameter",
    "GroupQuantScaleParameter",
    "BlockQuantScaleParameter",
    "PackedColumnParameter",
    "RowvLLMParameter",
]

logger = logging.getLogger(__name__)

_is_cpu = is_cpu()


def _dtype_rank(dtype: torch.dtype) -> Optional[int]:
    if dtype in (
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
        torch.float8_e8m0fnu,
    ):
        return 0
    if dtype in (torch.float16, torch.bfloat16):
        return 1
    if dtype == torch.float32:
        return 2
    if dtype == torch.float64:
        return 3
    return None


def copy_with_check(target: torch.Tensor, loaded_weight: torch.Tensor):


    assert (
        target.shape == loaded_weight.shape
    ), f"{target.shape=}, {loaded_weight.shape=}"

    if target.dtype == loaded_weight.dtype:
        target.copy_(loaded_weight)
        return

    target_rank = _dtype_rank(target.dtype)
    loaded_rank = _dtype_rank(loaded_weight.dtype)

    if target_rank is None or loaded_rank is None:
        raise ValueError(
            f"Unsupported copy between dtypes: {target.dtype=}, {loaded_weight.dtype=}"
        )
    if target_rank < loaded_rank and not envs.SGLANG_QUANT_ALLOW_DOWNCASTING.get():
        raise ValueError(
            f"Downcasting not allowed: {target.dtype=}, {loaded_weight.dtype=}"
        )
    if loaded_rank == torch.float8_e8m0fnu:
        assert target_rank in {torch.float8_e8m0fnu, torch.float32}

    target.copy_(loaded_weight)


class BasevLLMParameter(Parameter):


    def __new__(cls, data: torch.Tensor, **kwargs):

        return super().__new__(cls, data=data, requires_grad=False)

    def __init__(self, data: torch.Tensor, weight_loader: Callable):


        self._weight_loader = weight_loader

    @property
    def weight_loader(self):
        return self._weight_loader

    def _assert_and_load(self, loaded_weight: torch.Tensor):
        assert self.data.shape == loaded_weight.shape
        self.data.copy_(loaded_weight)

    def load_column_parallel_weight(self, loaded_weight: torch.Tensor):
        self._assert_and_load(loaded_weight)

    def load_row_parallel_weight(self, loaded_weight: torch.Tensor):
        self._assert_and_load(loaded_weight)

    def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs):
        self._assert_and_load(loaded_weight)

    def load_qkv_weight(self, loaded_weight: torch.Tensor, **kwargs):
        self._assert_and_load(loaded_weight)


class _ColumnvLLMParameter(BasevLLMParameter):


    def __init__(self, output_dim: int, **kwargs):
        self._output_dim = output_dim
        super().__init__(**kwargs)

    @property
    def output_dim(self):
        return self._output_dim

    def load_column_parallel_weight(
        self,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        use_presharded_weights: bool = False,
    ):
        if not use_presharded_weights:
            shard_size = self.data.shape[self.output_dim]

            from sglang.srt.model_loader.weight_utils import (
                narrow_padded_param_and_loaded_weight,
            )

            if _is_cpu:
                param_data, loaded_weight = narrow_padded_param_and_loaded_weight(
                    self.data,
                    loaded_weight,
                    0,
                    tp_rank * shard_size,
                    self.output_dim,
                    shard_size,
                )
                assert param_data.shape == loaded_weight.shape
                param_data.copy_(loaded_weight)
                return
            else:
                loaded_weight = loaded_weight.narrow(
                    self.output_dim, tp_rank * shard_size, shard_size
                )

        copy_with_check(self.data, loaded_weight)

    def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs):

        shard_offset = kwargs.get("shard_offset")
        shard_size = kwargs.get("shard_size")
        tp_rank = kwargs.get("tp_rank")
        use_presharded_weights = kwargs.get("use_presharded_weights")
        if (
            isinstance(self, (PackedColumnParameter, PackedvLLMParameter))
            and self.packed_dim == self.output_dim
        ):
            shard_size, shard_offset = self.adjust_shard_indexes_for_packing(
                shard_offset=shard_offset, shard_size=shard_size
            )

        param_data = self.data

        param_data = param_data.narrow(self.output_dim, shard_offset, shard_size)

        from sglang.srt.model_loader.weight_utils import (
            narrow_padded_param_and_loaded_weight,
        )

        if _is_cpu:
            param_data, loaded_weight = narrow_padded_param_and_loaded_weight(
                param_data,
                loaded_weight,
                0,
                tp_rank * shard_size,
                self.output_dim,
                shard_size,
                not use_presharded_weights,
            )
        else:
            if not use_presharded_weights:

                start_idx = tp_rank * shard_size
                end_idx = start_idx + shard_size
                if end_idx > loaded_weight.shape[self.output_dim]:
                    loaded_weight = pad_or_narrow_weight(
                        loaded_weight, self.output_dim, start_idx, shard_size
                    )
                else:
                    loaded_weight = loaded_weight.narrow(
                        self.output_dim, start_idx, shard_size
                    )

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def load_qkv_weight(
        self,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        use_presharded_weights: bool = False,
        **kwargs,
    ):

        shard_offset = kwargs.get("shard_offset")
        shard_size = kwargs.get("shard_size")
        shard_id = kwargs.get("shard_id")
        num_heads = kwargs.get("num_heads")

        if (
            isinstance(self, (PackedColumnParameter, PackedvLLMParameter))
            and self.output_dim == self.packed_dim
        ):
            shard_size, shard_offset = self.adjust_shard_indexes_for_packing(
                shard_offset=shard_offset, shard_size=shard_size
            )

        param_data = self.data
        shard_id = tp_rank if shard_id == "q" else tp_rank // num_heads
        param_data = param_data.narrow(self.output_dim, shard_offset, shard_size)

        if _is_cpu:
            from sglang.srt.model_loader.weight_utils import (
                narrow_padded_param_and_loaded_weight,
            )

            param_data, loaded_weight = narrow_padded_param_and_loaded_weight(
                param_data,
                loaded_weight,
                0,
                shard_id * shard_size,
                self.output_dim,
                shard_size,
                not use_presharded_weights,
            )
        else:
            if not use_presharded_weights:
                loaded_weight = loaded_weight.narrow(
                    self.output_dim, shard_id * shard_size, shard_size
                )

        assert (
            param_data.shape == loaded_weight.shape
        ), f"{param_data.shape=}, {loaded_weight.shape=}"
        param_data.copy_(loaded_weight)


class RowvLLMParameter(BasevLLMParameter):


    def __init__(self, input_dim: int, **kwargs):
        self._input_dim = input_dim
        super().__init__(**kwargs)

    @property
    def input_dim(self):
        return self._input_dim

    def load_row_parallel_weight(
        self,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        use_presharded_weights: bool = False,
    ):
        if not use_presharded_weights:
            shard_size = self.data.shape[self.input_dim]

            from sglang.srt.model_loader.weight_utils import (
                narrow_padded_param_and_loaded_weight,
            )

            if _is_cpu:
                param_data, loaded_weight = narrow_padded_param_and_loaded_weight(
                    self.data,
                    loaded_weight,
                    0,
                    tp_rank * shard_size,
                    self.input_dim,
                    shard_size,
                )

                assert param_data.shape == loaded_weight.shape
                param_data.copy_(loaded_weight)

                return
            else:

                start_idx = tp_rank * shard_size
                end_idx = start_idx + shard_size
                if end_idx > loaded_weight.shape[self.input_dim]:
                    loaded_weight = pad_or_narrow_weight(
                        loaded_weight, self.input_dim, start_idx, shard_size
                    )
                else:
                    loaded_weight = loaded_weight.narrow(
                        self.input_dim, start_idx, shard_size
                    )

        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert self.data.shape == loaded_weight.shape
        self.data.copy_(loaded_weight)


class ModelWeightParameter(_ColumnvLLMParameter, RowvLLMParameter):


    pass


class GroupQuantScaleParameter(_ColumnvLLMParameter, RowvLLMParameter):


    pass


class ChannelQuantScaleParameter(_ColumnvLLMParameter):


    pass


class BlockQuantScaleParameter(_ColumnvLLMParameter, RowvLLMParameter):


    pass


class PerTensorScaleParameter(BasevLLMParameter):


    def __init__(self, **kwargs):
        self.qkv_idxs = {"q": 0, "k": 1, "v": 2}
        super().__init__(**kwargs)

    def _shard_id_as_int(self, shard_id: Union[str, int]) -> int:
        if isinstance(shard_id, int):
            return shard_id


        assert isinstance(shard_id, str)
        assert shard_id in self.qkv_idxs
        return self.qkv_idxs[shard_id]


    def load_row_parallel_weight(self, *args, **kwargs):
        kwargs.pop("tp_rank", None)
        kwargs.pop("use_presharded_weights", None)
        super().load_row_parallel_weight(*args, **kwargs)

    def load_merged_column_weight(self, *args, **kwargs):
        self._load_into_shard_id(*args, **kwargs)

    def load_qkv_weight(self, *args, **kwargs):
        self._load_into_shard_id(*args, **kwargs)

    def load_column_parallel_weight(self, *args, **kwargs):
        kwargs.pop("tp_rank", None)
        kwargs.pop("use_presharded_weights", None)
        super().load_row_parallel_weight(*args, **kwargs)

    def _load_into_shard_id(
        self, loaded_weight: torch.Tensor, shard_id: Union[str, int], **kwargs
    ):


        param_data = self.data
        shard_id = self._shard_id_as_int(shard_id)


        if len(loaded_weight.shape) != 0:
            assert loaded_weight.shape[0] == 1
            loaded_weight = loaded_weight[0]

        param_data = param_data[shard_id]
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


class PackedColumnParameter(_ColumnvLLMParameter):


    def __init__(
        self,
        packed_factor: Union[int, Fraction],
        packed_dim: int,
        marlin_tile_size: Optional[int] = None,
        **kwargs,
    ):
        self._packed_factor = packed_factor
        self._packed_dim = packed_dim
        self._marlin_tile_size = marlin_tile_size
        super().__init__(**kwargs)

    @property
    def packed_dim(self):
        return self._packed_dim

    @property
    def packed_factor(self):
        return self._packed_factor

    @property
    def marlin_tile_size(self):
        return self._marlin_tile_size

    def adjust_shard_indexes_for_packing(self, shard_size, shard_offset):
        return _adjust_shard_indexes_for_packing(
            shard_size=shard_size,
            shard_offset=shard_offset,
            packed_factor=self.packed_factor,
            marlin_tile_size=self.marlin_tile_size,
        )


class PackedvLLMParameter(ModelWeightParameter):


    def __init__(
        self,
        packed_factor: Union[int, Fraction],
        packed_dim: int,
        marlin_tile_size: Optional[int] = None,
        **kwargs,
    ):
        self._packed_factor = packed_factor
        self._packed_dim = packed_dim
        self._marlin_tile_size = marlin_tile_size
        super().__init__(**kwargs)

    @property
    def packed_dim(self):
        return self._packed_dim

    @property
    def packed_factor(self):
        return self._packed_factor

    @property
    def marlin_tile_size(self):
        return self._marlin_tile_size

    def adjust_shard_indexes_for_packing(self, shard_size, shard_offset):
        return _adjust_shard_indexes_for_packing(
            shard_size=shard_size,
            shard_offset=shard_offset,
            packed_factor=self.packed_factor,
            marlin_tile_size=self.marlin_tile_size,
        )


def permute_param_layout_(
    param: BasevLLMParameter, input_dim: int, output_dim: int, **kwargs
) -> BasevLLMParameter:


    curr_input_dim = getattr(param, "input_dim", None)
    curr_output_dim = getattr(param, "output_dim", None)

    if curr_input_dim is None or curr_output_dim is None:
        assert param.data.dim() == 2, (
            "permute_param_layout_ only supports 2D parameters when either "
            "input_dim or output_dim is not set"
        )


    if curr_input_dim is None:
        assert curr_output_dim is not None, "either input or output dim must be set"
        curr_input_dim = (curr_output_dim + 1) % 2
    if curr_output_dim is None:
        assert curr_input_dim is not None, "either input or output dim must be set"
        curr_output_dim = (curr_input_dim + 1) % 2


    perm = [
        i for i in range(param.data.dim()) if i not in [curr_input_dim, curr_output_dim]
    ]
    perm.insert(input_dim, curr_input_dim)
    perm.insert(output_dim, curr_output_dim)

    if "packed_dim" in kwargs:
        assert (
            hasattr(param, "packed_dim")
            and param.packed_dim == perm[kwargs["packed_dim"]]
        ), "permute_param_layout_ currently doesn't support repacking"

    param.data = param.data.permute(*perm)
    if hasattr(param, "_input_dim"):
        param._input_dim = input_dim
    if hasattr(param, "_output_dim"):
        param._output_dim = output_dim
    if "packed_dim" in kwargs and hasattr(param, "_packed_dim"):
        param._packed_dim = kwargs["packed_dim"]

    return param


def _adjust_shard_indexes_for_marlin(shard_size, shard_offset, marlin_tile_size):
    return shard_size * marlin_tile_size, shard_offset * marlin_tile_size


def _adjust_shard_indexes_for_packing(
    shard_size, shard_offset, packed_factor, marlin_tile_size
):
    shard_size = round(shard_size // packed_factor)
    shard_offset = round(shard_offset // packed_factor)
    if marlin_tile_size is not None:
        return _adjust_shard_indexes_for_marlin(
            shard_size=shard_size,
            shard_offset=shard_offset,
            marlin_tile_size=marlin_tile_size,
        )
    return shard_size, shard_offset
