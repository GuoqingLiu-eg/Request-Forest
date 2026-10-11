from abc import ABC, abstractmethod
from typing import Tuple

import torch


class BaseMatmul(ABC):
    @abstractmethod
    def forward(
        self,
        layer: torch.nn.Module,
        weight_prefix: str,
        hidden_states: torch.Tensor,
        expert_tokens: torch.Tensor,
        output_dtype: torch.dtype,
        group_list_type: int,
        transposed: bool,
        **scale_args,
    ) -> torch.Tensor:
        pass


class GroupedMatmul(BaseMatmul):
    def forward(
        self,
        layer: torch.nn.Module,
        weight_prefix: str,
        hidden_states: torch.Tensor,
        expert_tokens: torch.Tensor,
        output_dtype: torch.dtype,
        group_list_type: int,
        transposed: bool,
        **scale_args,
    ) -> torch.Tensor:

        weight = getattr(layer, f"{weight_prefix}_weight", None)
        if weight is None:
            raise AttributeError(
                f"Weight attribute '{weight_prefix}_weight' not found in layer"
            )
        return torch.ops.npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[weight] if transposed else [weight.transpose(1, 2)],
            **scale_args,
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=expert_tokens,
            output_dtype=output_dtype,
        )[0]


class GroupedMatmulSwigluQuant(BaseMatmul):


    def forward(
        self,
        layer: torch.nn.Module,
        weight_prefix: str,
        hidden_states: torch.Tensor,
        expert_tokens: torch.Tensor,
        output_dtype: torch.dtype = None,
        group_list_type: int = 1,
        transposed: bool = True,
        **scale_args,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        weight = getattr(layer, f"{weight_prefix}_weight", None)
        if weight is None:
            raise AttributeError(
                f"Weight attribute '{weight_prefix}_weight' not found in layer"
            )


        group_list = expert_tokens.cumsum(0) if group_list_type == 1 else expert_tokens
        return torch.ops.npu.npu_grouped_matmul_swiglu_quant_v2(
            x=hidden_states,
            weight=[weight] if transposed else [weight.transpose(1, 2)],
            group_list=group_list,
            **scale_args,
        )
