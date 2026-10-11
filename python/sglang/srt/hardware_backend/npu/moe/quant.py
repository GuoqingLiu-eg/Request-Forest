from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch


class BaseHiddenStatesQuant(ABC):


    def __init__(self, quant_dtype: torch.dtype) -> None:
        self.quant_dtype = quant_dtype

    @abstractmethod
    def __call__(
        self, hidden_states: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]: ...


class HiddenStatesDynamicQuant(BaseHiddenStatesQuant):


    def __init__(self, quant_dtype: torch.dtype, use_mx_quant: bool = False) -> None:
        super().__init__(quant_dtype)
        if use_mx_quant or quant_dtype == torch.float8_e4m3fn:
            self._op = torch.ops.npu.npu_dynamic_mx_quant
        elif quant_dtype in (torch.int8, torch.quint4x2):
            self._op = torch.ops.npu.npu_dynamic_quant
        else:
            raise ValueError(f"Unsupported dynamic quant dtype: {quant_dtype}")

    def __call__(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        quantized, scale = self._op(hidden_states, dst_type=self.quant_dtype)
        return quantized, scale


class HiddenStatesStaticQuant(BaseHiddenStatesQuant):


    def __call__(
        self,
        hidden_states: torch.Tensor,
        layer: torch.nn.Module,
    ) -> Tuple[torch.Tensor, None]:

        if not hasattr(layer, "aclnn_input_scale_reciprocal") or not hasattr(
            layer, "aclnn_input_offset"
        ):
            raise AttributeError(
                "Static quantisation requires layer attributes "
                "'aclnn_input_scale_reciprocal' and 'aclnn_input_offset'."
            )

        quantized = torch.ops.npu.npu_quantize(
            hidden_states,
            layer.aclnn_input_scale_reciprocal,
            layer.aclnn_input_offset,
            self.quant_dtype,
            -1,
            False,
        )
        return quantized, None
