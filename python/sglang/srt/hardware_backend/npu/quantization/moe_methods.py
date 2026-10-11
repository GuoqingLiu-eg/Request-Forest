from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.nn.parameter import Parameter

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.utils import npu_format_cast
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig
    from sglang.srt.layers.moe.moe_runner.ascend import AscendQuantInfo

import logging

from sglang.srt.hardware_backend.npu.moe.matmul import (
    GroupedMatmul,
    GroupedMatmulSwigluQuant,
)
from sglang.srt.hardware_backend.npu.moe.quant import HiddenStatesDynamicQuant
from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    _get_float4_e2m1fn_x2_dtype,
    _get_float8_e8m0fnu_dtype,
)

logger = logging.getLogger(__name__)


_E8M0_DTYPE = None


def _require_e8m0_dtype():

    global _E8M0_DTYPE
    if _E8M0_DTYPE is None:
        from sglang.srt.utils import is_npu

        if is_npu():
            import torch_npu

            _E8M0_DTYPE = getattr(torch_npu, "float8_e8m0fnu", None)
        if _E8M0_DTYPE is None:
            _E8M0_DTYPE = _get_float8_e8m0fnu_dtype()
        if _E8M0_DTYPE is None:
            raise RuntimeError(
                "float8_e8m0fnu dtype not found — MXFP8 MoE requires Ascend A5 "
                "with a torch_npu build exposing float8_e8m0fnu (torch_npu >= 2.9)."
            )
    return _E8M0_DTYPE


def fused_moe_npu(
    x,
    w1,
    w2,
    topk_output,
    moe_runner_config,
):
    logger.warning_once(
        f"The fused_moe_npu method deprecated and will be removed in future releases"
    )
    topk_weights, topk_ids, _ = topk_output
    original_dtype = x.dtype
    num_tokens = x.shape[0]
    topk_weights = topk_weights.to(x.dtype)
    topk_ids = topk_ids.to(torch.int32)
    num_experts = w1.shape[0]
    top_k = topk_weights.shape[-1]
    row_idx_len = num_tokens * top_k
    row_idx = (
        torch.arange(0, row_idx_len, dtype=torch.int32, device=topk_weights.device)
        .view(top_k, -1)
        .permute(1, 0)
        .contiguous()
    )

    hidden_states, expanded_row_idx, expanded_expert_idx = (
        torch.ops.npu.npu_moe_init_routing(
            x, row_idx=row_idx, expert_idx=topk_ids, active_num=num_tokens
        )
    )

    expert_tokens = torch.ops.npu.npu_moe_compute_expert_tokens(
        expanded_expert_idx, num_experts
    )

    expert_tokens = expert_tokens.to(torch.int64)


    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w1.permute(0, 2, 1)],
        bias=None,
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]


    if moe_runner_config.activation == "silu":
        hidden_states = torch.ops.npu.npu_swiglu(hidden_states)
    else:
        from sglang.srt.layers.activation import GeluAndMul

        hidden_states = GeluAndMul()(hidden_states)


    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2.permute(0, 2, 1)],
        bias=None,
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]

    final_hidden_states = torch.ops.npu.npu_moe_finalize_routing(
        hidden_states,
        skip1=None,
        skip2=None,
        bias=None,
        scales=topk_weights,
        expanded_src_to_dst_row=expanded_row_idx,
        export_for_source_row=topk_ids,
    )
    return final_hidden_states


class _NPUMoEMethodBase(FusedMoEMethodBase):


    def __init__(
        self,
        quant_config: Optional["QuantizationConfig"] = None,
    ):
        super().__init__()
        self.quant_config = quant_config

    @staticmethod
    def _set_dispatcher_output_dtype(layer: torch.nn.Module, dtype) -> None:

        if hasattr(layer, "dispatcher"):
            layer.dispatcher.set_quant_config({"dispatcher_output_dtype": dtype})

    @staticmethod
    def _validate_weight_prefix(layer: torch.nn.Module, weight_prefix: str) -> None:

        required = [f"{weight_prefix}_weight"]
        for attr in required:
            if not hasattr(layer, attr):
                raise AttributeError(
                    f"Layer {layer} is missing required attribute '{attr}' for "
                    f"weight_prefix='{weight_prefix}'"
                )

    @staticmethod
    def _get_bias_args(
        quant_info: "AscendQuantInfo", weight_prefix: str
    ) -> Dict[str, Any]:
        bias = getattr(quant_info, f"{weight_prefix}_scale_bias", None)
        if bias is None:
            bias = getattr(quant_info, f"{weight_prefix}_weight_bias", None)
        return {"bias": [bias]} if bias is not None else {}


class NPUW4A8MXFP4MoEMethod(_NPUMoEMethodBase):


    def __init__(self):
        super().__init__(quant_config=None)
        self.matmul = GroupedMatmul()
        self.hidden_states_quantizer = HiddenStatesDynamicQuant(
            quant_dtype=torch.float8_e4m3fn
        )

    def process_weights_after_loading(
        self, layer: torch.nn.Module, weight_prefix: str
    ) -> None:
        self._validate_weight_prefix(layer, weight_prefix)

        fp4_dtype = _get_float4_e2m1fn_x2_dtype()
        if fp4_dtype is None:
            raise RuntimeError("NPU W4A8 MXFP MoE requires float4 support.")

        weight = getattr(layer, f"{weight_prefix}_weight")
        weight.data = npu_format_cast(
            weight.data,
            customize_dtype=torch.float8_e4m3fn,
            input_dtype=fp4_dtype,
        ).transpose(-1, -2)

        weight_scale = getattr(layer, f"{weight_prefix}_weight_scale")
        scale = weight_scale.data.reshape(
            weight_scale.shape[0],
            weight_scale.shape[1],
            weight_scale.shape[2] // 2,
            2,
        ).transpose(1, 2)
        weight_scale.data = scale


        if weight_prefix == "w13":
            self._set_dispatcher_output_dtype(layer, "bf16")

    def apply(
        self,
        quant_info: "AscendQuantInfo",
        hidden_states: torch.Tensor,
        expert_tokens: torch.Tensor,
        pertoken_scale: Optional[torch.Tensor],
        output_dtype: torch.dtype,
        weight_prefix: str,
        group_list_type: int,
    ) -> torch.Tensor:
        fp4_dtype = _get_float4_e2m1fn_x2_dtype()
        if fp4_dtype is None:
            raise RuntimeError("NPU W4A8 MXFP MoE requires float4 support.")
        e8m0_dtype = _require_e8m0_dtype()

        if pertoken_scale is None:
            hidden_states, pertoken_scale = self.hidden_states_quantizer(hidden_states)
        elif pertoken_scale is not None:
            pertoken_scale = pertoken_scale.reshape(
                hidden_states.shape[0], hidden_states.shape[1] // 64, 2
            )

        return self.matmul.forward(
            quant_info,
            weight_prefix,
            hidden_states,
            expert_tokens.to(torch.int64),
            output_dtype,
            group_list_type=group_list_type,
            transposed=True,
            scale=None,
            scale_dtype=None,
            per_token_scale=[pertoken_scale],
            antiquant_scale=[
                getattr(quant_info, f"{weight_prefix}_weight_scale", None)
            ],
            x_dtype=torch.float8_e4m3fn,
            weight_dtype=fp4_dtype,
            per_token_scale_dtype=e8m0_dtype,
        )


class NPUW4A4MXFP4MoEMethod(_NPUMoEMethodBase):


    def __init__(self):
        super().__init__(quant_config=None)
        self.matmul = GroupedMatmul()
        fp4_dtype = _get_float4_e2m1fn_x2_dtype()
        if fp4_dtype is None:
            raise RuntimeError("NPU W4A4 MXFP4 MoE requires float4 support.")
        self.hidden_states_quantizer = HiddenStatesDynamicQuant(
            quant_dtype=fp4_dtype,
            use_mx_quant=True,
        )

    def process_weights_after_loading(
        self, layer: torch.nn.Module, weight_prefix: str
    ) -> None:
        self._validate_weight_prefix(layer, weight_prefix)

        weight = getattr(layer, f"{weight_prefix}_weight")
        weight.data = npu_format_cast(weight.data).transpose(-1, -2)

        weight_scale = getattr(layer, f"{weight_prefix}_weight_scale")
        scale = weight_scale.data.reshape(
            weight_scale.shape[0],
            weight_scale.shape[1],
            weight_scale.shape[2] // 2,
            2,
        ).transpose(1, 2)
        weight_scale.data = scale


        if weight_prefix == "w13":
            self._set_dispatcher_output_dtype(layer, "bf16")

    def apply(
        self,
        quant_info: "AscendQuantInfo",
        hidden_states: torch.Tensor,
        expert_tokens: torch.Tensor,
        pertoken_scale: Optional[torch.Tensor],
        output_dtype: torch.dtype,
        weight_prefix: str,
        group_list_type: int,
    ) -> torch.Tensor:
        fp4_dtype = self.hidden_states_quantizer.quant_dtype
        e8m0_dtype = _require_e8m0_dtype()

        if pertoken_scale is None:
            hidden_states, pertoken_scale = self.hidden_states_quantizer(hidden_states)
        elif pertoken_scale is not None:
            pertoken_scale = pertoken_scale.reshape(
                hidden_states.shape[0], hidden_states.shape[1] // 32, 2
            )

        scale_args: Dict[str, Any] = {
            "scale": [getattr(quant_info, f"{weight_prefix}_weight_scale", None)],
            "scale_dtype": e8m0_dtype,
            "per_token_scale": [pertoken_scale],
            "per_token_scale_dtype": e8m0_dtype,
            "x_dtype": fp4_dtype,
            "weight_dtype": fp4_dtype,
        }
        scale_args.update(self._get_bias_args(quant_info, weight_prefix))
        return self.matmul.forward(
            quant_info,
            weight_prefix,
            hidden_states,
            expert_tokens.to(torch.int64),
            output_dtype,
            group_list_type=group_list_type,
            transposed=True,
            **scale_args,
        )


class NPUW4A4Int4MoEMethod(_NPUMoEMethodBase):


    def __init__(self):
        super().__init__(quant_config=None)
        self.matmul = GroupedMatmul()
        self.hidden_states_quantizer = HiddenStatesDynamicQuant(
            quant_dtype=torch.quint4x2
        )

    def process_weights_after_loading(
        self, layer: torch.nn.Module, weight_prefix: str
    ) -> None:
        self._validate_weight_prefix(layer, weight_prefix)


        scale: torch.Tensor = getattr(layer, f"{weight_prefix}_weight_scale")
        scale_np = scale.data.cpu().contiguous().numpy()
        scale_np.dtype = np.uint32
        scale_uint64_tensor = torch.from_numpy(scale_np.astype(np.int64)).npu()
        processed_scale = torch.nn.Parameter(
            scale_uint64_tensor.squeeze(-1), requires_grad=False
        )
        setattr(layer, f"{weight_prefix}_weight_scale", processed_scale)


        offset: Optional[torch.Tensor] = getattr(
            layer, f"{weight_prefix}_weight_offset", None
        )
        if offset is not None:
            processed_offset = torch.nn.Parameter(
                offset.data.squeeze(-1), requires_grad=False
            )
            setattr(layer, f"{weight_prefix}_weight_offset", processed_offset)


        weight: torch.Tensor = getattr(layer, f"{weight_prefix}_weight")
        if not envs.SGLANG_NPU_W4A4_NEW_PACKING.get():
            weight.data = self._w4a4_pack_int4(weight.data)
        weight.data = weight.data.transpose(-2, -1).contiguous()
        weight.data = npu_format_cast(weight.data)
        weight.data = self._pack_to_int32(weight.data)


        if weight_prefix == "w13":
            self._set_dispatcher_output_dtype(layer, "bf16")

    def _pack_int4(self, weight) -> torch.Tensor:

        weight = weight.to(torch.int8)
        e = 0
        if len(weight.shape) == 2:
            k, n = weight.shape
        elif len(weight.shape) == 3:
            e, k, n = weight.shape
        n_new = n // 2 + n % 2

        if n_new != n // 2:
            raise AssertionError("n dimension should be even")

        weight = weight.reshape(-1, 2)
        weight0 = weight[:, :1]
        weight1 = weight[:, 1:]

        weight1_4 = torch.bitwise_left_shift(weight1, 4)
        weight2_4 = weight0 & 0b00001111

        weight_add = torch.bitwise_or(weight1_4, weight2_4)
        if e == 0:
            weight_res = weight_add.reshape(k, n_new)
        else:
            weight_res = weight_add.reshape(e, k, n_new)
        return weight_res

    def _w4a4_pack_int4(self, save_quant_weight):

        weight = save_quant_weight.transpose(-1, -2).contiguous()
        packed_weight_tensor = self._pack_int4(weight)
        packed_weight_tensor = packed_weight_tensor.transpose(-1, -2).contiguous()
        return packed_weight_tensor

    def _pack_to_int32(self, weight: torch.Tensor):

        return weight.contiguous().view(torch.int32)

    def apply(
        self,
        quant_info: "AscendQuantInfo",
        hidden_states: torch.Tensor,
        expert_tokens: torch.Tensor,
        pertoken_scale: torch.Tensor,
        output_dtype: torch.dtype,
        weight_prefix: str,
        group_list_type,
    ) -> torch.Tensor:
        scale = getattr(quant_info, f"{weight_prefix}_weight_scale", None)
        if pertoken_scale is None:
            hidden_states, pertoken_scale = self.hidden_states_quantizer.__call__(
                hidden_states
            )
        scale_args: Dict[str, Any] = {
            "scale": [scale],
            "per_token_scale": [pertoken_scale],
        }
        scale_args.update(self._get_bias_args(quant_info, weight_prefix))
        return self.matmul.forward(
            quant_info,
            weight_prefix,
            hidden_states,
            expert_tokens,
            output_dtype,
            group_list_type=group_list_type,
            transposed=True,
            **scale_args,
        )


class NPUW8A8Int8MoEMethod(_NPUMoEMethodBase):


    def __init__(self):
        super().__init__(quant_config=None)
        self.matmul = GroupedMatmul()
        self.hidden_states_quantizer = HiddenStatesDynamicQuant(quant_dtype=torch.int8)

    @staticmethod
    def maybe_process_fuseep_weights(layer: torch.nn.Module) -> bool:

        from sglang.srt.layers.moe import get_moe_a2a_backend

        if not get_moe_a2a_backend().is_ascend_fuseep():
            return False


        if getattr(layer, "_fuseep_weights_processed", False):
            return True

        from sglang.srt.hardware_backend.npu.moe.fuseep import process_fuseep_weights

        for prefix in ("w13", "w2"):
            process_fuseep_weights(layer, prefix)
        layer._fuseep_weights_processed = True
        return True

    def process_weights_after_loading(
        self, layer: torch.nn.Module, weight_prefix: str
    ) -> None:


        if self.maybe_process_fuseep_weights(layer):
            return

        self._validate_weight_prefix(layer, weight_prefix)


        scale: torch.Tensor = getattr(layer, f"{weight_prefix}_weight_scale")
        processed_scale = torch.nn.Parameter(
            scale.data.squeeze(-1).to(dtype=torch.bfloat16), requires_grad=False
        )
        setattr(layer, f"{weight_prefix}_weight_scale", processed_scale)


        offset: Optional[torch.Tensor] = getattr(
            layer, f"{weight_prefix}_weight_offset", None
        )
        if offset is not None:
            processed_offset = torch.nn.Parameter(
                offset.data.squeeze(-1), requires_grad=False
            )
            setattr(layer, f"{weight_prefix}_weight_offset", processed_offset)


        weight: torch.Tensor = getattr(layer, f"{weight_prefix}_weight")
        weight.data = npu_format_cast(weight.data.transpose(1, 2).contiguous())


        if weight_prefix == "w13":
            self._set_dispatcher_output_dtype(layer, "int8")

    def apply(
        self,
        quant_info: "AscendQuantInfo",
        hidden_states: torch.Tensor,
        expert_tokens: torch.Tensor,
        pertoken_scale: torch.Tensor,
        output_dtype: torch.dtype,
        weight_prefix: str,
        group_list_type,
    ) -> torch.Tensor:
        scale = getattr(quant_info, f"{weight_prefix}_weight_scale", None)
        if pertoken_scale is None:
            hidden_states, pertoken_scale = self.hidden_states_quantizer.__call__(
                hidden_states
            )
        scale_args: Dict[str, Any] = {
            "scale": [scale],
            "per_token_scale": [pertoken_scale],
        }
        scale_args.update(self._get_bias_args(quant_info, weight_prefix))
        return self.matmul.forward(
            quant_info,
            weight_prefix,
            hidden_states,
            expert_tokens,
            output_dtype,
            group_list_type=group_list_type,
            transposed=True,
            **scale_args,
        )


class NPUW4A8Int8MoEMethod(_NPUMoEMethodBase):


    def __init__(
        self,
        quant_config: Optional["QuantizationConfig"] = None,
        is_per_channel_weight: bool = False,
        activation_use_clip: bool = False,
    ):
        super().__init__(quant_config)
        self.is_per_channel_weight = is_per_channel_weight
        self.activation_use_clip = activation_use_clip
        self.matmul = GroupedMatmul()
        self.hidden_states_quantizer = HiddenStatesDynamicQuant(quant_dtype=torch.int8)

    def process_weights_after_loading(
        self, layer: torch.nn.Module, weight_prefix: str
    ) -> None:
        self._validate_weight_prefix(layer, weight_prefix)


        scale = getattr(layer, f"{weight_prefix}_weight_scale")
        scale_second = getattr(layer, f"{weight_prefix}_weight_scale_second", None)
        bias = getattr(layer, f"{weight_prefix}_bias", None)

        if not self.activation_use_clip:

            processed_scale = self._process_scale(
                getattr(layer, f"{weight_prefix}_weight"),
                scale,
                scale_second,
                self.is_per_channel_weight,
            )
            setattr(
                layer,
                f"{weight_prefix}_weight_scale",
                torch.nn.Parameter(processed_scale.squeeze(-1), requires_grad=False),
            )
            if scale_second is not None:
                delattr(layer, f"{weight_prefix}_weight_scale_second")
                delattr(layer, f"{weight_prefix}_weight_offset_second")
            self._update_bias(layer, weight_prefix)
        else:

            processed_scale = scale.data.squeeze(-1).unsqueeze(1).contiguous()
            setattr(
                layer,
                f"{weight_prefix}_weight_scale",
                torch.nn.Parameter(processed_scale, requires_grad=False),
            )
            if bias is not None:
                setattr(
                    layer,
                    f"{weight_prefix}_scale_bias",
                    torch.nn.Parameter(
                        bias.data.contiguous(),
                        requires_grad=False,
                    ),
                )


        weight = getattr(layer, f"{weight_prefix}_weight")
        weight.data = weight.data.transpose(1, 2).contiguous()
        weight.data = npu_format_cast(weight.data)
        weight.data = self._pack_to_int32(weight.data)


        if weight_prefix == "w13":
            self._set_dispatcher_output_dtype(layer, "int8")

    @staticmethod
    def _update_bias(
        layer: torch.nn.Module,
        weight_prefix: str,
    ) -> None:
        scale_bias_name = f"{weight_prefix}_scale_bias"
        if hasattr(layer, scale_bias_name):
            scale_bias = getattr(layer, scale_bias_name)
            scale_bias.data = scale_bias.data.transpose(1, 2).contiguous().sum(dim=1)

    def _process_scale(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        per_group_scale: Optional[torch.Tensor],
        is_per_channel: bool,
    ) -> torch.Tensor:
        scale = scale.transpose(1, 2).contiguous()
        if is_per_channel:
            scale_np = scale.cpu().contiguous().numpy()
            scale_np.dtype = np.uint32
            scale_uint64_tensor = torch.from_numpy(scale_np.astype(np.int64)).npu()
            return scale_uint64_tensor


        per_group_scale = per_group_scale.transpose(1, 2).contiguous()
        group_num, k, n = weight.shape
        n = n * 2
        per_group_scale = per_group_scale.reshape(group_num, -1, n)
        group_num, quantgroup_num, n = per_group_scale.shape

        scale_fp32 = (scale * per_group_scale).to(torch.float16).to(torch.float32)
        scale_fp32_np = scale_fp32.cpu().numpy()
        scale_fp32_np.dtype = np.uint32
        sscale_uint64 = np.zeros((group_num, quantgroup_num, n * 2), dtype=np.uint32)
        sscale_uint64[..., ::2] = scale_fp32_np
        sscale_uint64_tensor = (
            torch.from_numpy(sscale_uint64.view(np.int64).copy())
            .reshape(group_num, quantgroup_num, n)
            .npu()
        )
        return sscale_uint64_tensor

    def _pack_to_int32(self, weight: torch.Tensor) -> torch.Tensor:

        assert weight.shape[-1] % 4 == 0, (
            f"Last dimension of weight must be divisible by 4 for int8→int32 packing, "
            f"got shape {weight.shape}"
        )
        return weight.view(torch.int32).contiguous()

    def apply(
        self,
        quant_info: "AscendQuantInfo",
        hidden_states: torch.Tensor,
        expert_tokens: torch.Tensor,
        pertoken_scale: torch.Tensor,
        output_dtype: torch.dtype,
        weight_prefix: str,
        group_list_type,
    ) -> torch.Tensor:
        scale = getattr(quant_info, f"{weight_prefix}_weight_scale", None)
        if pertoken_scale is None:
            hidden_states, pertoken_scale = self.hidden_states_quantizer.__call__(
                hidden_states
            )
        scale_args: Dict[str, Any] = {
            "scale": [scale],
            "per_token_scale": [pertoken_scale],
        }
        scale_args.update(self._get_bias_args(quant_info, weight_prefix))

        return self.matmul.forward(
            quant_info,
            weight_prefix,
            hidden_states,
            expert_tokens,
            output_dtype,
            group_list_type=group_list_type,
            transposed=True,
            **scale_args,
        )


class NPUWNA16Int4MoEMethod(_NPUMoEMethodBase):


    def __init__(self):
        super().__init__(quant_config=None)
        self.matmul = GroupedMatmul()

    def process_weights_after_loading(
        self, layer: torch.nn.Module, weight_prefix: str
    ) -> None:
        self._validate_weight_prefix(layer, weight_prefix)


        scale = getattr(layer, f"{weight_prefix}_weight_scale")
        scale = scale.data.transpose(-1, -2).contiguous()
        setattr(
            layer,
            f"{weight_prefix}_weight_scale",
            torch.nn.Parameter(scale, requires_grad=False),
        )


        offset = getattr(layer, f"{weight_prefix}_weight_offset", None)
        if offset is not None:
            offset = offset.data.transpose(-1, -2).contiguous()
            setattr(
                layer,
                f"{weight_prefix}_weight_offset",
                torch.nn.Parameter(offset, requires_grad=False),
            )


        weight: torch.Tensor = getattr(layer, f"{weight_prefix}_weight")
        unpacked_weight = (
            self._unpack_from_int32(weight.data.flatten(0, 1), 4)
            .view(weight.shape[0], weight.shape[1], -1)
            .transpose(1, 2)
            .int()
        )
        weight.data = self._pack_to_int32(unpacked_weight)


        if weight_prefix == "w13":
            self._set_dispatcher_output_dtype(layer, "bf16")

    def _pack_to_int32(self, weight: torch.Tensor) -> torch.Tensor:
        assert weight.dim() == 3
        if weight.dtype == torch.int32:
            assert weight.shape[-1] % 8 == 0, (
                f"Last dimension of int32 weight must be divisible by 8 for int4 packing, "
                f"got {weight.shape}"
            )
            new_weight = torch.ops.npu.npu_convert_weight_to_int4pack(
                weight.flatten(0, 1)
            )
            new_weight = new_weight.view(weight.shape[0], weight.shape[1], -1)
        elif weight.dtype == torch.int8:
            assert weight.shape[-1] % 4 == 0, (
                f"Last dimension of int8 weight must be divisible by 4 for int32 packing, "
                f"got {weight.shape}"
            )
            new_weight = weight.contiguous().view(torch.int32)
        else:
            raise ValueError(f"Unsupported weight dtype for packing: {weight.dtype}")
        return new_weight.contiguous()

    def _unpack_from_int32(
        self,
        value: torch.Tensor,
        num_bits: int,
        shape: Optional[torch.Size] = None,
        packed_dim: int = 1,
    ) -> torch.Tensor:

        if value.dtype is not torch.int32:
            raise ValueError(
                f"Expected {torch.int32} but got {value.dtype}, Aborting unpack."
            )
        if num_bits > 8:
            raise ValueError("Unpacking is only supported for less than 8 bits")

        pack_factor = 32 // num_bits
        mask = (1 << num_bits) - 1

        if packed_dim == 1:
            unpacked = torch.zeros(
                (value.shape[0], value.shape[1] * pack_factor),
                device=value.device,
                dtype=torch.int32,
            )
            for i in range(pack_factor):
                unpacked[:, i::pack_factor] = (value >> (num_bits * i)) & mask
            if shape is not None:
                original_row_size = int(shape[1])
                unpacked = unpacked[:, :original_row_size]
        else:
            unpacked = torch.zeros(
                (value.shape[0] * pack_factor, value.shape[1]),
                device=value.device,
                dtype=torch.int32,
            )
            for i in range(pack_factor):
                unpacked[i::pack_factor, :] = (value >> (num_bits * i)) & mask
            if shape is not None:
                original_row_size = int(shape[0])
                unpacked = unpacked[:original_row_size, :]

        offset = pow(2, num_bits) // 2
        unpacked = (unpacked - offset).to(torch.int8)
        return unpacked

    def apply(
        self,
        quant_info: "AscendQuantInfo",
        hidden_states: torch.Tensor,
        expert_tokens: torch.Tensor,
        pertoken_scale: torch.Tensor,
        output_dtype: torch.dtype,
        weight_prefix: str,
        group_list_type,
    ) -> torch.Tensor:
        scale = getattr(quant_info, f"{weight_prefix}_weight_scale", None)
        offset = getattr(quant_info, f"{weight_prefix}_weight_offset", None)
        scale_args: Dict[str, Any] = {
            "antiquant_scale": [scale],
            "antiquant_offset": [offset] if offset is not None else [],
        }
        scale_args.update(self._get_bias_args(quant_info, weight_prefix))
        return self.matmul.forward(
            quant_info,
            weight_prefix,
            hidden_states,
            expert_tokens,
            output_dtype,
            group_list_type=group_list_type,
            transposed=True,
            **scale_args,
        )


class NPUUnquantMoEMethod(_NPUMoEMethodBase):


    def __init__(self):
        super().__init__(quant_config=None)
        self.matmul = GroupedMatmul()

    def process_weights_after_loading(
        self, layer: torch.nn.Module, weight_prefix: str
    ) -> None:
        self._validate_weight_prefix(layer, weight_prefix)

        weight: torch.Tensor = getattr(layer, f"{weight_prefix}_weight")
        weight.data = npu_format_cast(weight)

        if weight_prefix == "w13":
            self._set_dispatcher_output_dtype(layer, "bf16")

    def apply(
        self,
        quant_info: "AscendQuantInfo",
        hidden_states: torch.Tensor,
        expert_tokens: torch.Tensor,
        pertoken_scale: torch.Tensor,
        output_dtype: torch.dtype,
        weight_prefix: str,
        group_list_type,
    ) -> torch.Tensor:
        return self.matmul.forward(
            quant_info,
            weight_prefix,
            hidden_states,
            expert_tokens,
            output_dtype,
            group_list_type=group_list_type,
            transposed=False,
            **self._get_bias_args(quant_info, weight_prefix),
        )


class NPUMXFP8MoEMethod(_NPUMoEMethodBase):


    def __init__(self, weight_prefix: str):
        super().__init__(quant_config=None)
        if weight_prefix == "w13":
            self.matmul = GroupedMatmulSwigluQuant()
            self.hidden_states_quantizer = HiddenStatesDynamicQuant(
                quant_dtype=torch.float8_e4m3fn
            )
        else:
            self.matmul = GroupedMatmul()
            self.hidden_states_quantizer = None

    @staticmethod
    def _quantize_weight_online(
        weight: torch.Tensor, weight_prefix: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if weight.dtype not in (torch.float16, torch.bfloat16):
            logger.warning(
                "NPUMXFP8MoEMethod: %s_weight dtype %s is not float16/bfloat16; "
                "casting to bfloat16 before MXFP8 quantisation.",
                weight_prefix,
                weight.dtype,
            )
            weight = weight.to(torch.bfloat16)

        if not weight.is_npu:
            weight = weight.to(f"npu:{torch.npu.current_device()}")
        return torch.ops.npu.npu_dynamic_mx_quant(weight, dst_type=torch.float8_e4m3fn)

    def process_weights_after_loading(
        self, layer: torch.nn.Module, weight_prefix: str
    ) -> None:
        self._validate_weight_prefix(layer, weight_prefix)

        weight: torch.Tensor = getattr(layer, f"{weight_prefix}_weight").data
        if weight.dtype == torch.float8_e4m3fn:


            scale: torch.Tensor = getattr(layer, f"{weight_prefix}_weight_scale").data
            scale = scale.reshape(scale.shape[0], -1, scale.shape[-1] // 2, 2)
        else:
            weight, scale = self._quantize_weight_online(weight, weight_prefix)


        weight = npu_format_cast(weight)


        setattr(
            layer,
            f"{weight_prefix}_weight",
            Parameter(weight.transpose(1, 2), requires_grad=False),
        )
        setattr(
            layer,
            f"{weight_prefix}_weight_scale",
            Parameter(scale.transpose(1, 2), requires_grad=False),
        )

        if weight_prefix == "w13":
            from sglang.srt.layers.moe import get_moe_a2a_backend


            dispatcher_dtype = "bf16" if get_moe_a2a_backend().is_deepep() else "mxfp8"
            self._set_dispatcher_output_dtype(layer, dispatcher_dtype)

    def apply_fused_gmm1_swiglu(
        self,
        quant_info: "AscendQuantInfo",
        hidden_states: torch.Tensor,
        expert_tokens: torch.Tensor,
        pertoken_scale: Optional[torch.Tensor],
        group_list_type,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if pertoken_scale is None:
            hidden_states, pertoken_scale = self.hidden_states_quantizer(hidden_states)

        e8m0_dtype = _require_e8m0_dtype()
        return self.matmul.forward(
            quant_info,
            "w13",
            hidden_states,
            expert_tokens,
            group_list_type=group_list_type,
            transposed=True,
            weight_scale=[quant_info.w13_weight_scale],
            x_scale=pertoken_scale,
            dequant_mode=2,
            quant_mode=2,
            dequant_dtype=torch.float32,
            quant_dtype=torch.float8_e4m3fn,

            x_dtype=None,
            weight_dtype=None,
            weight_scale_dtype=e8m0_dtype,
            x_scale_dtype=e8m0_dtype,
        )

    def apply(
        self,
        quant_info: "AscendQuantInfo",
        hidden_states: torch.Tensor,
        expert_tokens: torch.Tensor,
        pertoken_scale: torch.Tensor,
        output_dtype: torch.dtype,
        weight_prefix: str,
        group_list_type,
    ) -> torch.Tensor:
        if weight_prefix != "w2":
            raise ValueError(
                f"NPUMXFP8MoEMethod.apply only serves the w2 gmm, got "
                f"'{weight_prefix}'. gmm1 fuses swiglu into a single op and must "
                f"go through apply_fused_gmm1_swiglu, which returns a scale too."
            )

        e8m0_dtype = _require_e8m0_dtype()
        scale_args: Dict[str, Any] = {
            "scale": [getattr(quant_info, f"{weight_prefix}_weight_scale", None)],
            "per_token_scale": [pertoken_scale],
            "scale_dtype": e8m0_dtype,
            "per_token_scale_dtype": e8m0_dtype,
            "x_dtype": None,
            "weight_dtype": None,
        }
        return self.matmul.forward(
            quant_info,
            weight_prefix,
            hidden_states,
            expert_tokens,
            output_dtype,
            group_list_type=group_list_type,
            transposed=True,
            **scale_args,
        )
