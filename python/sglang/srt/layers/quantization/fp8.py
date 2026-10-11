# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/layers/quantization/fp8.py

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.parameter import Parameter

from sglang.kernels.ops.quantization.fp8_kernel import (
    fp8_dtype,
    is_fp8_fnuz,
    per_token_group_quant_fp8,
    scaled_fp8_quant,
)
from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.layers.amx_utils import (
    CPUQuantMethod,
    _amx_process_weight_after_loading,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.deep_gemm import DeepGemmMoeQuantInfo
from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
    FlashInferTrtllmFp8MoeQuantInfo,
)
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.moe.utils import (
    RoutingMethodType,
    get_moe_a2a_backend,
    get_moe_padding_size,
    get_moe_runner_backend,
    get_moe_weight_sizes,
)
from sglang.srt.layers.parameter import (
    BlockQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.fp8_utils import (
    _use_aiter_bpreshuffle_gfx95,
    apply_fp8_linear,
    can_auto_enable_marlin_fp8,
    cutlass_fp8_supported,
    deepgemm_w8a8_block_fp8_linear_with_fallback,
    dispatch_w8a8_block_fp8_linear,
    dispatch_w8a8_mxfp8_linear,
    input_to_float8,
    mxfp8_group_quantize,
    normalize_e4m3fn_to_e4m3fnuz,
    requant_block_scale_ue8m0_for_deepgemm,
    resolve_mxfp8_dense_gemm_backend,
)
from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod
from sglang.srt.layers.quantization.marlin_utils_fp8 import prepare_fp8_layer_for_marlin
from sglang.srt.layers.quantization.unquant import (
    UnquantizedFusedMoEMethod,
    UnquantizedLinearMethod,
)
from sglang.srt.layers.quantization.utils import (
    all_close_1d,
    convert_to_channelwise,
    is_layer_skipped,
    per_tensor_dequantize,
    requantize_with_max_scale,
)
from sglang.srt.layers.utils import copy_or_rebind_param
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_blackwell_supported,
    is_cpu,
    is_cuda,
    is_flashinfer_available,
    is_gfx95_supported,
    is_hip,
    is_musa,
    is_npu,
    is_sm90_supported,
    is_sm100_supported,
    is_sm120_supported,
    log_info_on_rank0,
    mxfp8_block_convert_required,
    print_warning_once,
    set_weight_attrs,
    use_intel_amx_backend,
    use_intel_xpu_backend,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner.aiter import AiterMoeQuantInfo
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput
    from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config
    from sglang.srt.models.utils import WeightsMapper

_is_hip = is_hip()
_is_cuda = is_cuda()
_is_musa = is_musa()
_is_npu = is_npu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_is_fp8_fnuz = is_fp8_fnuz()
_is_gfx95_supported = is_gfx95_supported()


_mxfp8_to_block_fp8_required = mxfp8_block_convert_required() or get_bool_env_var(
    "SGLANG_FORCE_MXFP8_BLOCK_CONVERT"
)
_use_hip_int4 = get_bool_env_var("SGLANG_INT4_WEIGHT") and _is_hip
_use_aiter = envs.SGLANG_USE_AITER.get() and _is_hip
_is_shuffle_moe_mxfp4 = is_gfx95_supported()


def _require_fp4_dtype():
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4_dtype is None:
        raise RuntimeError(
            "DeepSeek-V4 FP4 experts require torch.float4_e2m1fn_x2 support."
        )
    return fp4_dtype


if _use_aiter or _use_hip_int4:
    from aiter.ops.shuffle import shuffle_scale, shuffle_weight

if _use_aiter:
    from sglang.srt.layers.quantization.fp8_utils import (
        aiter_w8a8_block_fp8_linear,
        use_aiter_triton_gemm_w8a8_tuned_gfx950,
    )


ACTIVATION_SCHEMES = ["static", "dynamic"]

logger = logging.getLogger(__name__)

DSV4_DEQUANT_FP4_TABLE = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def cast_e2m1fn_to_e4m3fn(
    x: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:

    assert x.dtype == torch.int8
    assert x.ndim == 2
    out_dim, in_dim = x.size()
    in_dim *= 2
    fp8_block_size = 128
    fp4_block_size = 32
    assert in_dim % fp8_block_size == 0 and out_dim % fp8_block_size == 0
    assert scale.size(0) == out_dim and scale.size(1) == in_dim // fp4_block_size

    x = x.view(torch.uint8)
    low = x & 0x0F
    high = (x >> 4) & 0x0F
    table = DSV4_DEQUANT_FP4_TABLE.to(x.device)
    x = torch.stack([table[low.long()], table[high.long()]], dim=-1).flatten(2)


    MAX_OFFSET_BITS = 6

    bOut = out_dim // fp8_block_size
    bIn = in_dim // fp8_block_size

    x = x.view(bOut, fp8_block_size, bIn, fp8_block_size).transpose(1, 2)

    scale = scale.float().view(bOut, fp8_block_size, bIn, -1).transpose(1, 2).flatten(2)

    scale_max_offset_bits = scale.amax(dim=-1, keepdim=True) / (2**MAX_OFFSET_BITS)

    offset = scale / scale_max_offset_bits

    offset = offset.unflatten(-1, (fp8_block_size, -1)).repeat_interleave(
        fp4_block_size, dim=-1
    )
    x = (x * offset).transpose(1, 2).reshape(out_dim, in_dim)
    return x.to(torch.float8_e4m3fn), scale_max_offset_bits.squeeze(-1).to(
        torch.float8_e8m0fnu
    )


class Fp8Config(QuantizationConfig):


    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: Optional[List[str]] = None,
        weight_block_size: List[int] = None,
        packed_modules_mapping: Optional[Dict[str, List[str]]] = None,
        use_mxfp8: bool = False,
        is_fp4_experts: bool = False,
        kv_cache_quant_algo: Optional[str] = None,
    ) -> None:
        super().__init__()


        self.is_fp4_experts = is_fp4_experts
        self.dequant_fp4_to_fp8 = False
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        if is_checkpoint_fp8_serialized:
            log_info_on_rank0(logger, "Detected fp8 checkpoint.")
        if activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []
        if ignored_layers_str := envs.SGLANG_FP8_IGNORED_LAYERS.get():
            self.ignored_layers.extend(
                [
                    layer.strip()
                    for layer in ignored_layers_str.split(",")
                    if layer.strip()
                ]
            )
        self.packed_modules_mapping = packed_modules_mapping or {}
        self.use_mxfp8 = use_mxfp8
        self.kv_cache_quant_algo = kv_cache_quant_algo
        if weight_block_size is not None:
            if not is_checkpoint_fp8_serialized:
                raise ValueError(
                    "The block-wise quantization only supports fp8-serialized checkpoint for now."
                )
            if len(weight_block_size) != 2:
                raise ValueError(
                    f"The quantization block size of weight must have 2 dimensions, but got {len(weight_block_size)} dimensions."
                )
            if activation_scheme != "dynamic":
                raise ValueError(
                    f"The block-wise quantization only supports dynamic activation scheme for now, but got {activation_scheme} activation scheme."
                )
        if self.use_mxfp8:
            if weight_block_size is None:
                weight_block_size = [1, 32]
            elif weight_block_size != [1, 32]:
                raise ValueError("MXFP8 requires weight_block_size=[1, 32].")
        self.weight_block_size = weight_block_size

    def get_name(self) -> str:
        return "mxfp8" if self.use_mxfp8 else "fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    def get_min_capability(self) -> int:
        if is_npu():
            return 0
        if _is_musa:
            return 31
        if self.use_mxfp8 and _is_hip and _is_gfx95_supported:
            return 95
        if self.use_mxfp8 and _mxfp8_to_block_fp8_required:
            return 94

        return 100 if self.use_mxfp8 else 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Fp8Config:
        quant_method = cls.get_from_keys(config, ["quant_method"])
        use_mxfp8 = "mxfp8" in quant_method
        is_checkpoint_fp8_serialized = ("fp8" in quant_method) or use_mxfp8
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        packed_modules_mapping = (
            cls.get_from_keys_or(config, ["packed_modules_mapping"], {}) or {}
        )
        ignored_layers = cls.get_from_keys_or(
            config, ["ignored_layers", "modules_to_not_convert"], None
        )
        if ignored_layers:

            normalized = []
            for layer in ignored_layers:
                base = layer.removeprefix("model.")
                normalized.append(base)
                normalized.append(f"model.{base}")
            ignored_layers = normalized
        weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"], None)
        kv_cache_quant_algo = cls.get_from_keys_or(
            config, ["kv_cache_quant_algo"], None
        )
        if use_mxfp8:

            if weight_block_size is not None and weight_block_size != [1, 32]:
                logger.warning(
                    "MXFP8 overriding weight_block_size=%s from config.json -> [1, 32].",
                    weight_block_size,
                )
            weight_block_size = [1, 32]
        return cls(
            is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
            weight_block_size=weight_block_size,
            packed_modules_mapping=packed_modules_mapping,
            use_mxfp8=use_mxfp8,
            kv_cache_quant_algo=kv_cache_quant_algo,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.radix_attention import RadixAttention

        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix, self.ignored_layers, fused_mapping=self.packed_modules_mapping
            ):
                return UnquantizedLinearMethod()
            if is_npu() and self.use_mxfp8:
                from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
                    NPUMXFP8LinearMethod,
                )

                return NPUMXFP8LinearMethod(self)
            return Fp8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            if is_layer_skipped(
                prefix, self.ignored_layers, fused_mapping=self.packed_modules_mapping
            ):
                return UnquantizedFusedMoEMethod(
                    layer.use_triton_kernels, layer.use_flashinfer_trtllm_moe
                )

            if is_npu() and self.use_mxfp8:
                from sglang.srt.hardware_backend.npu.quantization.online_moe_methods import (
                    NPUMXFP8OnlineMoEMethod,
                )

                return NPUMXFP8OnlineMoEMethod(self)

            fp8_method = Fp8MoEMethod(self)

            if self.is_fp4_experts and self.dequant_fp4_to_fp8:
                assert (
                    get_moe_runner_backend().is_auto()
                ), f"{get_moe_runner_backend()} is not compatible with SGLANG_DSV4_FP4_DEQUANT=1"
                return fp8_method

            if self.is_fp4_experts and get_moe_runner_backend().is_marlin():
                from sglang.srt.layers.quantization.mxfp4_marlin_moe import (
                    Mxfp4MarlinMoEMethod,
                )

                return Mxfp4MarlinMoEMethod(fp8_method, prefix=prefix)

            if self.is_fp4_experts and get_moe_runner_backend().is_humming():
                from sglang.srt.layers.quantization.mxfp4_humming_moe import (
                    Mxfp4HummingMoEMethod,
                )

                return Mxfp4HummingMoEMethod(fp8_method, prefix=prefix)

            if self.is_fp4_experts and get_moe_runner_backend().is_flashinfer_mxfp4():

                if is_sm90_supported() or is_sm120_supported():
                    from sglang.srt.layers.quantization.mxfp4_flashinfer_cutlass_moe import (
                        Mxfp4FlashinferCutlassMoEMethod,
                    )

                    return Mxfp4FlashinferCutlassMoEMethod(fp8_method, prefix=prefix)

                from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
                    Mxfp4FlashinferTrtllmMoEMethod,
                )

                return Mxfp4FlashinferTrtllmMoEMethod(fp8_method, prefix=prefix)
            return fp8_method
        elif isinstance(layer, RadixAttention):
            return Fp8KVCacheMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []

    def apply_weight_name_mapper(self, hf_to_sglang_mapper: WeightsMapper):
        if self.ignored_layers:
            self.ignored_layers = list(
                dict.fromkeys(hf_to_sglang_mapper.apply_list(self.ignored_layers))
            )


class Fp8LinearMethod(LinearMethodBase):


    def __init__(self, quant_config: Union[Fp8Config, W4AFp8Config]):
        self.quant_config = quant_config
        self.cutlass_fp8_supported = cutlass_fp8_supported()


        self.use_marlin = False
        if _is_cuda:
            force_marlin = get_bool_env_var("SGLANG_FORCE_FP8_MARLIN")
            auto_enable = can_auto_enable_marlin_fp8()
            self.use_marlin = force_marlin or auto_enable

        self.use_mxfp8 = getattr(self.quant_config, "use_mxfp8", False)
        self.block_quant = (
            self.use_mxfp8 or self.quant_config.weight_block_size is not None
        )
        self.convert_mxfp8_to_block = self.use_mxfp8 and _mxfp8_to_block_fp8_required
        self.weight_block_size = self.quant_config.weight_block_size
        self.w8a8_block_fp8_linear = None
        self.w8a8_mxfp8_linear = None
        self.mxfp8_dense_backend = None
        if self.use_mxfp8 and not self.convert_mxfp8_to_block:
            self.mxfp8_dense_backend = resolve_mxfp8_dense_gemm_backend()
            self.w8a8_mxfp8_linear = dispatch_w8a8_mxfp8_linear()
        else:
            self.w8a8_block_fp8_linear = dispatch_w8a8_block_fp8_linear()
        self.is_checkpoint_fp8_serialized = (
            self.quant_config.is_checkpoint_fp8_serialized
        )
        self.use_aiter_fp8_per_token = envs.SGLANG_USE_AITER_FP8_PER_TOKEN.get()
        self.use_per_token_if_dynamic = False

    @staticmethod
    def validate_block_quant_shapes(
        quant_config,
        input_size: int,
        input_size_per_partition: int,
        output_size: int,
        output_size_per_partition: int,
        output_partition_sizes: List[int],
        skip_block_quant_check: bool = False,
    ):
        block_n, block_k = (
            quant_config.weight_block_size[0],
            quant_config.weight_block_size[1],
        )

        if skip_block_quant_check:
            print_warning_once(
                "Skipping block quantization checks for weight partition."
            )
        else:
            tp_size = get_parallel().tp_size

            if tp_size > 1 and input_size // input_size_per_partition == tp_size:
                if input_size_per_partition % block_k != 0:
                    raise ValueError(
                        f"Weight input_size_per_partition = "
                        f"{input_size_per_partition} is not divisible by "
                        f"weight quantization block_k = {block_k}."
                    )

            if (
                tp_size > 1 and output_size // output_size_per_partition == tp_size
            ) or len(output_partition_sizes) > 1:
                for output_partition_size in output_partition_sizes:
                    if output_partition_size % block_n != 0:
                        raise ValueError(
                            f"Weight output_partition_size = "
                            f"{output_partition_size} is not divisible by "
                            f"weight quantization block_n = {block_n}."
                        )

    @staticmethod
    def create_fp8_weight_(
        layer: torch.nn.Module,
        block_quant: bool,
        quant_config,
        use_mxfp8: bool,
        output_size_per_partition: int,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        weight_loader,
        is_checkpoint_fp8_serialized: bool,
        skip_block_quant_check: bool = False,
        **extra_weight_attrs,
    ):


        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        if block_quant:
            block_n, block_k = quant_config.weight_block_size
            Fp8LinearMethod.validate_block_quant_shapes(
                quant_config,
                input_size,
                input_size_per_partition,
                output_size,
                output_size_per_partition,
                output_partition_sizes,
                skip_block_quant_check,
            )
        else:
            block_n, block_k = None, None


        weight_dtype = (
            torch.float8_e4m3fn if is_checkpoint_fp8_serialized else params_dtype
        )
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition, input_size_per_partition, dtype=weight_dtype
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        if is_checkpoint_fp8_serialized:
            if block_quant:
                if hasattr(quant_config, "activation_scheme"):
                    assert quant_config.activation_scheme == "dynamic"
                elif hasattr(quant_config, "linear_activation_scheme"):
                    assert quant_config.linear_activation_scheme == "dynamic"

                scale_dtype = torch.uint8 if use_mxfp8 else torch.float32
                scale_init = torch.zeros if scale_dtype == torch.uint8 else torch.empty
                scale = BlockQuantScaleParameter(
                    data=scale_init(
                        (output_size_per_partition + block_n - 1) // block_n,
                        (input_size_per_partition + block_k - 1) // block_k,
                        dtype=scale_dtype,
                    ),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=weight_loader,
                )
                scale.format_ue8m0 = use_mxfp8
                if scale_dtype != torch.uint8:
                    scale[:] = torch.finfo(torch.float32).min
                layer.register_parameter("weight_scale_inv", scale)
            else:
                scale = PerTensorScaleParameter(
                    data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                    weight_loader=weight_loader,
                )
                scale[:] = torch.finfo(torch.float32).min
                layer.register_parameter("weight_scale", scale)


            if (
                hasattr(quant_config, "activation_scheme")
                and quant_config.activation_scheme == "static"
            ) or (
                hasattr(quant_config, "linear_activation_scheme")
                and quant_config.linear_activation_scheme == "static"
            ):
                scale = PerTensorScaleParameter(
                    data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                    weight_loader=weight_loader,
                )

                scale[:] = torch.finfo(torch.float32).min
                layer.register_parameter("input_scale", scale)
            else:
                layer.register_parameter("input_scale", None)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        skip_block_quant_check: bool = False,
        **extra_weight_attrs,
    ):
        Fp8LinearMethod.create_fp8_weight_(
            layer,
            block_quant=self.block_quant,
            quant_config=self.quant_config,
            use_mxfp8=self.use_mxfp8,
            output_size_per_partition=sum(output_partition_sizes),
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            weight_loader=extra_weight_attrs.get("weight_loader"),
            skip_block_quant_check=skip_block_quant_check,
            input_size=input_size,
            output_size=output_size,
            is_checkpoint_fp8_serialized=self.is_checkpoint_fp8_serialized,
            params_dtype=params_dtype,
        )

    def process_weights_after_loading_block_quant(self, layer: Module) -> None:
        if self.convert_mxfp8_to_block:
            from sglang.srt.layers.quantization.mxfp8_block_convert import (
                convert_mxfp8_weight_to_block_fp8,
            )

            qweight, scale = convert_mxfp8_weight_to_block_fp8(
                layer.weight.data, layer.weight_scale_inv.data, block=128
            )
            layer.weight = Parameter(qweight, requires_grad=False)
            layer.weight_scale_inv = Parameter(scale, requires_grad=False)
            self.use_mxfp8 = False
            self.convert_mxfp8_to_block = False
            self.weight_block_size = [128, 128]
        elif self.use_mxfp8:


            if not self.is_checkpoint_fp8_serialized:
                self._quantize_mxfp8_weights(layer)
                return


            layer.weight_scale_inv.requires_grad_(False)
            layer.weight_scale_inv.format_ue8m0 = True
            self._process_mxfp8_linear_weight_scale(layer)
            return

        if _is_fp8_fnuz:

            weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                weight=layer.weight,
                weight_scale=layer.weight_scale_inv,
                input_scale=None,
            )
            layer.input_scale = None
        elif _is_cpu:
            assert (
                _is_cpu_amx_available
            ), "Fp8LinearMethod on CPU requires that CPU has AMX support"
            _amx_process_weight_after_loading(layer, ["weight"])
            layer.weight_scale_inv = torch.nn.Parameter(
                layer.weight_scale_inv.data, requires_grad=False
            )
            return
        else:

            use_deepgemm_runner = (
                self.w8a8_block_fp8_linear
                is deepgemm_w8a8_block_fp8_linear_with_fallback
            )
            requant_block_scale_ue8m0_for_deepgemm(
                layer.weight,
                layer.weight_scale_inv,
                getattr(self.quant_config, "weight_block_size", None),
                use_deepgemm_runner=use_deepgemm_runner,
                output_dtype=getattr(layer, "orig_dtype", None),
                weight_shape=layer.weight.shape,
            )
            weight, weight_scale = layer.weight.data, layer.weight_scale_inv.data

        layer.weight.data = weight.data
        layer.weight_scale_inv.data = weight_scale.data

        if (
            _use_aiter_bpreshuffle_gfx95
            and self.w8a8_block_fp8_linear is aiter_w8a8_block_fp8_linear
        ):
            n, k = layer.weight.shape
            if not use_aiter_triton_gemm_w8a8_tuned_gfx950(n, k):


                t = shuffle_weight(layer.weight, (16, 16))
                layer.weight.copy_(t)
                del t

    def _process_mxfp8_linear_weight_scale(self, layer: Module) -> None:
        if not self.use_mxfp8:
            return

        backend = self.mxfp8_dense_backend
        if backend.is_flashinfer_trtllm():
            from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a

            weight = layer.weight.data
            scale_u8 = layer.weight_scale_inv.data
            n, k = weight.shape
            epilogue_tile_m = 128
            sf_cols = k // 32

            scale_u8 = scale_u8.contiguous().view(torch.uint8).reshape(n, sf_cols)
            padded_n = ((n + epilogue_tile_m - 1) // epilogue_tile_m) * (
                epilogue_tile_m
            )
            pad_rows = padded_n - n

            if pad_rows:
                scale_u8 = F.pad(
                    scale_u8,
                    (0, 0, 0, pad_rows),
                    mode="constant",
                    value=0,
                )

            copy_or_rebind_param(
                layer,
                "weight",
                shuffle_matrix_a(
                    weight.contiguous().view(torch.uint8), epilogue_tile_m
                ).view(torch.float8_e4m3fn),
            )
            copy_or_rebind_param(
                layer,
                "weight_scale_inv_shuffled",
                shuffle_matrix_sf_a(
                    scale_u8,
                    epilogue_tile_m,
                    num_elts_per_sf=32,
                )
                .reshape_as(scale_u8)
                .contiguous(),
            )
        elif backend.is_flashinfer_cutlass() or backend.is_flashinfer_cutedsl():
            from flashinfer import block_scale_interleave

            scale_u8 = layer.weight_scale_inv.data


            copy_or_rebind_param(
                layer,
                "weight_scale_inv_swizzled",
                block_scale_interleave(scale_u8.contiguous()).contiguous(),
            )
        elif backend.is_deep_gemm():
            from sglang.srt.layers.deep_gemm_wrapper.configurer import (
                DEEPGEMM_SCALE_UE8M0,
            )

            n, k = layer.weight.shape
            scale_u8 = layer.weight_scale_inv.data
            layer.weight_scale_inv_swizzled = None
            if n % 64 != 0 or k % 128 != 0:
                if not (is_blackwell_supported() and is_flashinfer_available()):
                    raise RuntimeError(
                        f"--fp8-gemm-backend=deep_gemm cannot serve MXFP8 weight shape "
                        f"({n}, {k}) (needs N % 64 == 0 and K % 128 == 0), and this "
                        "device has no FlashInfer MXFP8 fallback kernel."
                    )
                from flashinfer import block_scale_interleave

                copy_or_rebind_param(
                    layer,
                    "weight_scale_inv_swizzled",
                    block_scale_interleave(scale_u8.contiguous()).contiguous(),
                )
            scale_fp32 = (
                (scale_u8.contiguous().view(-1).to(torch.int32) << 23)
                .view(torch.float32)
                .view(n, k // 32)
            )
            if DEEPGEMM_SCALE_UE8M0:

                import deep_gemm.utils.layout

                scale_packed = (
                    deep_gemm.utils.layout.get_mn_major_tma_aligned_packed_ue8m0_tensor(
                        scale_fp32
                    )
                )
            else:
                scale_packed = scale_fp32
            copy_or_rebind_param(layer, "weight_scale_inv_deepgemm", scale_packed)

    def _quantize_mxfp8_weights(self, layer: Module) -> None:
        weight = layer.weight.data
        qweight, weight_scale = mxfp8_group_quantize(weight)

        layer.weight.data = qweight
        layer.weight.requires_grad_(False)
        if hasattr(layer, "weight_scale_inv") and layer.weight_scale_inv is not None:
            layer.weight_scale_inv.data = weight_scale
            layer.weight_scale_inv.requires_grad_(False)
        else:

            layer.register_parameter(
                "weight_scale_inv", Parameter(weight_scale, requires_grad=False)
            )
        layer.weight_scale_inv.format_ue8m0 = True
        self._process_mxfp8_linear_weight_scale(layer)
        layer.input_scale = None

    def process_weights_after_loading(self, layer: Module) -> None:
        if self.block_quant:
            self.process_weights_after_loading_block_quant(layer)
        else:
            layer.weight = Parameter(layer.weight.data, requires_grad=False)


            if not self.is_checkpoint_fp8_serialized:
                if (
                    self.cutlass_fp8_supported
                    or self.use_marlin
                    or (_use_aiter and self.use_aiter_fp8_per_token)
                ):


                    qweight, weight_scale = per_token_group_quant_fp8(
                        layer.weight, layer.weight.shape[-1]
                    )
                    weight_scale = weight_scale.t().contiguous()
                    if _use_aiter and self.use_aiter_fp8_per_token:
                        self.use_per_token_if_dynamic = True
                        qweight = shuffle_weight(qweight.contiguous(), (16, 16))
                else:

                    qweight, weight_scale = input_to_float8(layer.weight)


                layer.weight = Parameter(qweight.t(), requires_grad=False)
                layer.weight_scale = Parameter(weight_scale, requires_grad=False)
                layer.input_scale = None


            else:
                layer.weight_scale = Parameter(
                    layer.weight_scale.data, requires_grad=False
                )
                if (
                    hasattr(self.quant_config, "activation_scheme")
                    and self.quant_config.activation_scheme == "static"
                ) or (
                    hasattr(self.quant_config, "linear_activation_scheme")
                    and self.quant_config.linear_activation_scheme == "static"
                ):
                    layer.input_scale = Parameter(
                        layer.input_scale.data, requires_grad=False
                    )


                if (
                    self.cutlass_fp8_supported
                    or self.use_marlin
                    or (_use_aiter and self.use_aiter_fp8_per_token)
                ):
                    weight = layer.weight
                    weight_scale = convert_to_channelwise(
                        layer.weight_scale, layer.logical_widths
                    )
                    if _use_aiter and self.use_aiter_fp8_per_token:

                        self.use_per_token_if_dynamic = True
                        if _is_fp8_fnuz:
                            weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                                weight=weight,
                                weight_scale=weight_scale,
                            )
                        weight = shuffle_weight(weight.contiguous(), (16, 16))
                else:

                    weight = layer.weight
                    weight_scale = layer.weight_scale

                    if _is_fp8_fnuz:
                        weight, weight_scale, input_scale = (
                            normalize_e4m3fn_to_e4m3fnuz(
                                weight=weight,
                                weight_scale=weight_scale,
                                input_scale=layer.input_scale,
                            )
                        )
                        if input_scale is not None:
                            layer.input_scale = Parameter(
                                input_scale, requires_grad=False
                            )

                    weight_scale, weight = requantize_with_max_scale(
                        weight=weight,
                        weight_scale=weight_scale,
                        logical_widths=layer.logical_widths,
                    )


                layer.weight = Parameter(weight.t(), requires_grad=False)
                layer.weight_scale = Parameter(weight_scale, requires_grad=False)
                if (
                    hasattr(self.quant_config, "activation_scheme")
                    and self.quant_config.activation_scheme == "static"
                ) or (
                    hasattr(self.quant_config, "linear_activation_scheme")
                    and self.quant_config.linear_activation_scheme == "static"
                ):
                    layer.input_scale = Parameter(
                        layer.input_scale.max(), requires_grad=False
                    )

        if self.use_marlin:
            if self.block_quant:
                layer.weight_block_size = self.quant_config.weight_block_size
            prepare_fp8_layer_for_marlin(layer, not self.block_quant)

            del layer.input_scale

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_marlin:
            return torch.ops.sglang.apply_fp8_marlin_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                workspace=layer.workspace,
                size_n=layer.output_size_per_partition,
                size_k=layer.input_size_per_partition,
                bias=bias,
            )

        if self.use_mxfp8:
            backend = self.mxfp8_dense_backend
            extra_kwargs = {}
            if backend.is_flashinfer_cutlass() or backend.is_flashinfer_cutedsl():
                weight_scale = layer.weight_scale_inv_swizzled
            elif backend.is_flashinfer_trtllm():
                weight_scale = layer.weight_scale_inv_shuffled
            elif backend.is_deep_gemm():
                weight_scale = layer.weight_scale_inv_deepgemm
                extra_kwargs["weight_scale_swizzled"] = layer.weight_scale_inv_swizzled
            else:
                weight_scale = layer.weight_scale_inv
            if isinstance(x, tuple):
                return self.w8a8_mxfp8_linear(
                    input=x[0],
                    weight=layer.weight,
                    weight_scale=weight_scale,
                    input_scale=x[1],
                    bias=bias,
                    **extra_kwargs,
                )
            return self.w8a8_mxfp8_linear(
                input=x,
                weight=layer.weight,
                weight_scale=weight_scale,
                input_scale=None,
                bias=bias,
                **extra_kwargs,
            )

        if self.block_quant:
            if use_intel_amx_backend(layer):
                return torch.ops.sgl_kernel.fp8_scaled_mm_cpu(
                    x,
                    layer.weight,
                    layer.weight_scale_inv,
                    self.weight_block_size,
                    bias,
                    x.dtype,
                    True,
                )

            if isinstance(x, tuple):
                return self.w8a8_block_fp8_linear(
                    input=x[0],
                    weight=layer.weight,
                    block_size=self.weight_block_size,
                    weight_scale=layer.weight_scale_inv,
                    input_scale=x[1],
                    bias=bias,
                )

            return self.w8a8_block_fp8_linear(
                input=x,
                weight=layer.weight,
                block_size=self.weight_block_size,
                weight_scale=layer.weight_scale_inv,
                input_scale=None,
                bias=bias,
            )

        if isinstance(x, tuple):


            qx, x_scale = x[0], x[1]
            out_dtype = x[2] if len(x) > 2 else None
            return apply_fp8_linear(
                input=qx,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                input_scale=x_scale,
                bias=bias,
                cutlass_fp8_supported=self.cutlass_fp8_supported,
                use_per_token_if_dynamic=self.use_per_token_if_dynamic,
                pre_quant_output_dtype=out_dtype,
            )

        return apply_fp8_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            input_scale=layer.input_scale,
            bias=bias,
            cutlass_fp8_supported=self.cutlass_fp8_supported,
            use_per_token_if_dynamic=self.use_per_token_if_dynamic,
        )


class Fp8MoEMethod(FusedMoEMethodBase):


    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config
        self.use_mxfp8 = getattr(self.quant_config, "use_mxfp8", False)
        self.block_quant = (
            self.use_mxfp8 or self.quant_config.weight_block_size is not None
        )
        self.convert_mxfp8_to_block = self.use_mxfp8 and _mxfp8_to_block_fp8_required
        self.weight_block_size = self.quant_config.weight_block_size
        self.is_fp4_expert = self.quant_config.is_fp4_experts
        self.dequant_fp4_to_fp8 = self.quant_config.dequant_fp4_to_fp8
        self.with_bias = False


        self._owns_moe_runner = False
        if get_moe_runner_backend().is_cutlass():
            assert (
                cutlass_fp8_supported()
            ), "cutlass_fp8 MoE requires CUDA 12.0+ with SM90 or CUDA 12.4+ with SM89"
            assert self.block_quant, "cutlass_fp8 MoE requires block quantization"
            assert (
                is_sm100_supported() or is_sm90_supported() or is_sm120_supported()
            ), "cutlass_fp8 MoE requires SM90, SM100, or SM120 GPUs"

    @staticmethod
    def is_deepgemm_moe_runner_backend_enabled(
        moe_runner_backend=None, moe_a2a_backend=None
    ) -> bool:

        from sglang.srt.layers.moe.utils import get_moe_a2a_backend

        if moe_runner_backend is None:
            moe_runner_backend = get_moe_runner_backend()
        if moe_runner_backend.is_deep_gemm():
            return True
        if moe_runner_backend.is_auto():
            if moe_a2a_backend is None:
                moe_a2a_backend = get_moe_a2a_backend()
            if not (
                moe_a2a_backend.is_deepep()
                or moe_a2a_backend.is_mooncake()
                or moe_a2a_backend.is_nixl()
            ):
                return False
            from sglang.srt.layers import deep_gemm_wrapper

            return deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
        return False

    @staticmethod
    def create_fp8_moe_weight_(
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        block_quant: bool,
        quant_config,
        use_mxfp8: bool,
        is_checkpoint_fp8_serialized: bool,
        is_fp4_expert: bool,
        params_dtype: torch.dtype,
        with_bias: bool = False,
        fp4_scale_dtype: Optional[torch.dtype] = None,
        **extra_weight_attrs,
    ):

        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        if is_checkpoint_fp8_serialized:
            params_dtype = torch.uint32 if _use_hip_int4 else torch.float8_e4m3fn

        tp_size = get_parallel().tp_size
        w13_num_shards = 2 if layer.moe_runner_config.is_gated else 1

        w13_up_dim, w2_up_dim, weight_padded = get_moe_weight_sizes(
            intermediate_size_per_partition,
            is_aiter_moe=_use_aiter,
            is_concat=layer.moe_runner_config.is_gated,
            is_packed=False,
        )

        if block_quant:
            block_n, block_k = (
                quant_config.weight_block_size[0],
                quant_config.weight_block_size[1],
            )

            padding_size = get_moe_padding_size(_use_aiter)
            if not (_use_aiter and padding_size == block_n == block_k):


                if intermediate_size_per_partition % block_n != 0:
                    raise ValueError(
                        f"The output_size of gate's and up's weight = "
                        f"{intermediate_size_per_partition} is not divisible by "
                        f"weight quantization block_n = {block_n}."
                    )
                if tp_size > 1:

                    if intermediate_size_per_partition % block_k != 0:
                        raise ValueError(
                            f"The input_size of down's weight = "
                            f"{intermediate_size_per_partition} is not divisible by "
                            f"weight quantization block_k = {block_k}."
                        )


        if is_fp4_expert:
            w13_weight = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    w13_num_shards * intermediate_size_per_partition,
                    hidden_size // 2,
                    dtype=torch.int8,
                ),
                requires_grad=False,
            )
            w2_weight = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition // 2,
                    dtype=torch.int8,
                ),
                requires_grad=False,
            )
        elif _is_hip and _use_hip_int4:

            w13_weight = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    w13_num_shards * intermediate_size_per_partition,
                    hidden_size // 8,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
            w2_weight = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition // 8,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
        else:
            w13_weight = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    w13_up_dim,
                    hidden_size,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
            w2_weight = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    hidden_size,
                    w2_up_dim,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )

        extra_weight_attrs.update(
            {"weight_padded": weight_padded},
        )

        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)


        if with_bias:
            w13_weight_bias = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    w13_num_shards * intermediate_size_per_partition,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_bias", w13_weight_bias)
            set_weight_attrs(w13_weight_bias, extra_weight_attrs)

            w2_weight_bias = torch.nn.Parameter(
                torch.empty(num_experts, hidden_size, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_bias", w2_weight_bias)
            set_weight_attrs(w2_weight_bias, extra_weight_attrs)


        if is_fp4_expert:
            fp4_block_k = 32
            if fp4_scale_dtype is None:
                fp4_scale_dtype = torch.float8_e8m0fnu if _use_aiter else torch.float32
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    w13_num_shards * intermediate_size_per_partition,
                    hidden_size // fp4_block_k,
                    dtype=fp4_scale_dtype,
                ),
                requires_grad=False,
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition // fp4_block_k,
                    dtype=fp4_scale_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
            layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
        elif block_quant:
            scale_dtype = torch.uint8 if use_mxfp8 else torch.float32
            scale_init = torch.zeros if scale_dtype == torch.uint8 else torch.ones
            w13_weight_scale = torch.nn.Parameter(
                scale_init(
                    num_experts,
                    w13_num_shards
                    * ((intermediate_size_per_partition + block_n - 1) // block_n),
                    (hidden_size + block_k - 1) // block_k,
                    dtype=scale_dtype,
                ),
                requires_grad=False,
            )
            w2_weight_scale = torch.nn.Parameter(
                scale_init(
                    num_experts,
                    (hidden_size + block_n - 1) // block_n,
                    (intermediate_size_per_partition + block_k - 1) // block_k,
                    dtype=scale_dtype,
                ),
                requires_grad=False,
            )

            w13_weight_scale.format_ue8m0 = use_mxfp8
            w2_weight_scale.format_ue8m0 = use_mxfp8
            layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
            layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)

            assert quant_config.activation_scheme == "dynamic"

        else:


            w13_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, w13_num_shards, dtype=torch.float32),
                requires_grad=False,
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            layer.register_parameter("w2_weight_scale", w2_weight_scale)

            if _is_hip:

                w13_weight_scale1 = torch.nn.Parameter(
                    torch.ones(
                        num_experts,
                        w13_num_shards * intermediate_size_per_partition,
                        dtype=torch.float32,
                    ),
                    requires_grad=False,
                )
                w2_weight_scale1 = torch.nn.Parameter(
                    torch.ones(num_experts, hidden_size, dtype=torch.float32),
                    requires_grad=False,
                )
                layer.register_parameter("w13_weight_scale1", w13_weight_scale1)
                layer.register_parameter("w2_weight_scale1", w2_weight_scale1)


        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
            if block_quant
            else {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )


        if quant_config.is_checkpoint_fp8_serialized:
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

            if _is_hip and _use_hip_int4:
                extra_weight_attrs.update(
                    {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
                )
                set_weight_attrs(w13_weight_scale1, extra_weight_attrs)
                set_weight_attrs(w2_weight_scale1, extra_weight_attrs)


        if quant_config.activation_scheme == "static":
            if not quant_config.is_checkpoint_fp8_serialized:
                raise ValueError(
                    "Found static activation scheme for checkpoint that "
                    "was not serialized fp8."
                )

            w13_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)

        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        with_bias: bool = False,
        **extra_weight_attrs,
    ):
        Fp8MoEMethod.create_fp8_moe_weight_(
            layer=layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            block_quant=self.block_quant,
            quant_config=self.quant_config,
            use_mxfp8=self.use_mxfp8,
            is_checkpoint_fp8_serialized=self.quant_config.is_checkpoint_fp8_serialized,
            is_fp4_expert=self.is_fp4_expert,
            params_dtype=params_dtype,
            with_bias=with_bias,
            **extra_weight_attrs,
        )

        if (
            not self.is_fp4_expert
            and self.block_quant
            and get_moe_runner_backend().is_cutlass()
        ):
            self._ensure_cutlass_buffers_initialized(layer)

    def process_weights_after_loading_block_quant(self, layer: Module) -> None:

        if _use_aiter and self.is_fp4_expert:
            gu_intv = envs.SGLANG_USE_AITER_MOE_GU_ITLV.get()
            fp4_weight_dtype = _require_fp4_dtype()


            fp4_k_align = 128
            E, w13_N, w13_K_packed = layer.w13_weight.shape
            _, w2_N, w2_K_packed = layer.w2_weight.shape
            inter_per_part = w13_N // 2
            padded_inter = (
                (inter_per_part + fp4_k_align - 1) // fp4_k_align * fp4_k_align
            )


            layer.intermediate_pad = padded_inter - inter_per_part
            layer.hidden_pad = 0
            if padded_inter != inter_per_part:
                pad_amount = padded_inter - inter_per_part
                fp4_block_k = 32


                old_w13 = layer.w13_weight.data
                new_w13 = torch.zeros(
                    E,
                    2 * padded_inter,
                    w13_K_packed,
                    dtype=old_w13.dtype,
                    device=old_w13.device,
                )
                new_w13[:, :inter_per_part, :] = old_w13[:, :inter_per_part, :]
                new_w13[:, padded_inter : padded_inter + inter_per_part, :] = old_w13[
                    :, inter_per_part:, :
                ]
                layer.w13_weight = torch.nn.Parameter(new_w13, requires_grad=False)


                old_w2 = layer.w2_weight.data
                new_w2 = torch.zeros(
                    E,
                    w2_N,
                    padded_inter // 2,
                    dtype=old_w2.dtype,
                    device=old_w2.device,
                )
                new_w2[:, :, :w2_K_packed] = old_w2
                layer.w2_weight = torch.nn.Parameter(new_w2, requires_grad=False)


                old_s13 = layer.w13_weight_scale_inv.data
                _, _, s13_K = old_s13.shape
                new_s13 = torch.zeros(
                    E,
                    2 * padded_inter,
                    s13_K,
                    dtype=old_s13.dtype,
                    device=old_s13.device,
                )
                new_s13[:, :inter_per_part, :] = old_s13[:, :inter_per_part, :]
                new_s13[:, padded_inter : padded_inter + inter_per_part, :] = old_s13[
                    :, inter_per_part:, :
                ]
                layer.w13_weight_scale_inv = torch.nn.Parameter(
                    new_s13, requires_grad=False
                )


                old_s2 = layer.w2_weight_scale_inv.data
                new_s2 = torch.zeros(
                    E,
                    w2_N,
                    padded_inter // fp4_block_k,
                    dtype=old_s2.dtype,
                    device=old_s2.device,
                )
                new_s2[:, :, : old_s2.shape[2]] = old_s2
                layer.w2_weight_scale_inv = torch.nn.Parameter(
                    new_s2, requires_grad=False
                )

            for scale_name in ("w13_weight_scale_inv", "w2_weight_scale_inv"):
                scale = getattr(layer, scale_name)
                num_experts, num_rows, _ = scale.shape
                is_w13_scale = scale_name == "w13_weight_scale_inv"
                scale_2d = scale.reshape(-1, scale.shape[-1])
                scale.data = shuffle_scale(scale_2d, num_experts, gu_intv, is_w13_scale)

            layer.w13_weight.data = layer.w13_weight.data.view(fp4_weight_dtype)
            layer.w2_weight.data = layer.w2_weight.data.view(fp4_weight_dtype)

            is_shuffled = _is_shuffle_moe_mxfp4
            if is_shuffled:
                layer.w13_weight.data = shuffle_weight(
                    layer.w13_weight,
                    is_guinterleave=gu_intv,
                    gate_up=True,
                )
                layer.w2_weight.data = shuffle_weight(
                    layer.w2_weight,
                    is_guinterleave=gu_intv,
                    gate_up=False,
                )
            layer.w13_weight.is_shuffled = is_shuffled
            layer.w2_weight.is_shuffled = is_shuffled
            return

        if self.convert_mxfp8_to_block:


            self._convert_mxfp8_moe_to_block_fp8(layer)
            self.use_mxfp8 = False
            self.convert_mxfp8_to_block = False
            self.weight_block_size = [128, 128]
            if _is_fp8_fnuz:
                w13_weight, w13_weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                    weight=layer.w13_weight,
                    weight_scale=layer.w13_weight_scale_inv,
                    input_scale=None,
                )
                w2_weight, w2_weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                    weight=layer.w2_weight,
                    weight_scale=layer.w2_weight_scale_inv,
                    input_scale=None,
                )
                layer.w13_weight = Parameter(w13_weight, requires_grad=False)
                layer.w13_weight_scale_inv = Parameter(
                    w13_weight_scale, requires_grad=False
                )
                layer.w2_weight = Parameter(w2_weight, requires_grad=False)
                layer.w2_weight_scale_inv = Parameter(
                    w2_weight_scale, requires_grad=False
                )
                layer.w13_input_scale = None
                layer.w2_input_scale = None
            runner_is_aiter = (
                getattr(self, "runner", None) is not None
                and self.runner.runner_backend.is_aiter()
            )
            if _use_aiter and runner_is_aiter:
                layer.w13_weight.data = shuffle_weight(
                    layer.w13_weight.contiguous(), (16, 16)
                )
                layer.w2_weight.data = shuffle_weight(
                    layer.w2_weight.contiguous(), (16, 16)
                )
            return
        elif self.use_mxfp8:
            self._process_mxfp8_moe_weights(
                layer, quantize=not self.quant_config.is_checkpoint_fp8_serialized
            )
            return


        if _is_fp8_fnuz:

            w13_weight, w13_weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                weight=layer.w13_weight,
                weight_scale=layer.w13_weight_scale_inv,
                input_scale=None,
            )
            w2_weight, w2_weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                weight=layer.w2_weight,
                weight_scale=layer.w2_weight_scale_inv,
                input_scale=None,
            )

            layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
            layer.w13_weight_scale_inv = torch.nn.Parameter(
                w13_weight_scale, requires_grad=False
            )
            layer.w13_input_scale = None
            layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
            layer.w2_weight_scale_inv = torch.nn.Parameter(
                w2_weight_scale, requires_grad=False
            )
            layer.w2_input_scale = None
            if _use_aiter:
                layer.w13_weight.data = shuffle_weight(
                    layer.w13_weight.contiguous(), (16, 16)
                )
                layer.w2_weight.data = shuffle_weight(
                    layer.w2_weight.contiguous(), (16, 16)
                )
        elif _use_aiter:

            t = shuffle_weight(layer.w13_weight, (16, 16))
            layer.w13_weight.copy_(t)
            del t
            t = shuffle_weight(layer.w2_weight, (16, 16))
            layer.w2_weight.copy_(t)
            del t
        elif _is_cpu:
            assert (
                _is_cpu_amx_available
            ), "Fp8MoEMethod on CPU requires that CPU has AMX support"
            _amx_process_weight_after_loading(layer, ["w13_weight", "w2_weight"])
        else:

            from sglang.srt.layers import deep_gemm_wrapper


            will_use_deepgemm = self.is_deepgemm_moe_runner_backend_enabled()

            if self.is_fp4_expert and self.dequant_fp4_to_fp8:
                for weight_param, scale_param in [
                    (layer.w13_weight, layer.w13_weight_scale_inv),
                    (layer.w2_weight, layer.w2_weight_scale_inv),
                ]:
                    num_experts = weight_param.shape[0]
                    new_weights = []
                    new_scales = []
                    for e in range(num_experts):
                        w, s = cast_e2m1fn_to_e4m3fn(
                            weight_param.data[e], scale_param.data[e]
                        )
                        new_weights.append(w)
                        new_scales.append(s)
                    weight_param.data = torch.stack(new_weights)
                    scale_param.data = torch.stack(new_scales).float()
                    scale_param.format_ue8m0 = False
                self.is_fp4_expert = False
                logger.warning_once("Dequantized FP4 expert weights to FP8.")

            if self.is_fp4_expert:
                if get_moe_runner_backend().is_marlin():
                    layer.w13_weight.data = layer.w13_weight.data.view(torch.int8)
                    layer.w2_weight.data = layer.w2_weight.data.view(torch.int8)
                    return

                fp4_weight_dtype = _require_fp4_dtype() if _use_aiter else torch.int8
                layer.w13_weight.data = layer.w13_weight.data.view(fp4_weight_dtype)
                layer.w2_weight.data = layer.w2_weight.data.view(fp4_weight_dtype)

                if get_moe_a2a_backend().is_megamoe():
                    from sglang.srt.layers.moe.mega_moe import (
                        build_mega_moe_experts_weights,
                    )

                    build_mega_moe_experts_weights(layer)
                    return

                if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0 and will_use_deepgemm:
                    from deep_gemm import transform_sf_into_required_layout

                    for scale_param, weight_param in [
                        (layer.w13_weight_scale_inv, layer.w13_weight),
                        (layer.w2_weight_scale_inv, layer.w2_weight),
                    ]:
                        num_experts, n, _ = scale_param.data.shape
                        k = weight_param.shape[2] * 2
                        scale_param.data = transform_sf_into_required_layout(
                            scale_param.data,
                            mn=n,
                            k=k,
                            recipe=(1, 32),
                            num_groups=num_experts,
                            disable_ue8m0_cast=False,
                        )
                    layer.w13_weight_scale_inv.format_ue8m0 = True
                    layer.w2_weight_scale_inv.format_ue8m0 = True

            if get_moe_a2a_backend().is_megamoe() and is_sm90_supported():
                from sglang.srt.layers.moe.mega_moe_sm90 import (
                    build_sm90_mega_moe_experts_weights,
                )

                assert not self.is_fp4_expert
                build_sm90_mega_moe_experts_weights(layer)
                return

            if not self.is_fp4_expert:
                weight_block_size = self.quant_config.weight_block_size
                for weight, weight_scale in (
                    (layer.w13_weight, layer.w13_weight_scale_inv),
                    (layer.w2_weight, layer.w2_weight_scale_inv),
                ):
                    requant_block_scale_ue8m0_for_deepgemm(
                        weight,
                        weight_scale,
                        weight_block_size,
                        use_deepgemm_runner=will_use_deepgemm,
                        output_dtype=torch.bfloat16,
                        weight_shape=weight.shape[-2:],
                    )

    def _convert_mxfp8_moe_to_block_fp8(self, layer: Module) -> None:
        from sglang.srt.layers.quantization.mxfp8_block_convert import (
            convert_mxfp8_weight_to_block_fp8,
        )

        def convert(w, s):
            E, N, K = w.shape
            qw = torch.empty_like(w)
            sn = (N + 127) // 128
            sk = (K + 127) // 128
            scale = torch.empty((E, sn, sk), dtype=torch.float32, device=w.device)
            for e in range(E):
                qe, se = convert_mxfp8_weight_to_block_fp8(w[e], s[e], block=128)
                qw[e] = qe
                scale[e] = se
            return qw, scale

        w13_q, w13_s = convert(layer.w13_weight.data, layer.w13_weight_scale_inv.data)
        w2_q, w2_s = convert(layer.w2_weight.data, layer.w2_weight_scale_inv.data)
        layer.w13_weight = Parameter(w13_q, requires_grad=False)
        layer.w2_weight = Parameter(w2_q, requires_grad=False)
        layer.w13_weight_scale_inv = Parameter(w13_s, requires_grad=False)
        layer.w2_weight_scale_inv = Parameter(w2_s, requires_grad=False)
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def _process_mxfp8_moe_weights(self, layer: Module, quantize: bool = True) -> None:

        if not (
            (_is_cuda and is_sm100_supported()) or (_is_hip and _is_gfx95_supported)
        ):
            raise RuntimeError(
                "MXFP8 MoE quantization requires SM100 or ROCm gfx95 "
                "(gfx942 converts MXFP8 to block-fp8 at load instead)."
            )

        def _quantize_and_swizzle_with_cutlass_es_kernel(weight: torch.Tensor):
            from sgl_kernel import es_sm100_mxfp8_blockscaled_grouped_quant

            weight = weight.contiguous()
            num_experts, m, k = weight.shape
            assert k % 32 == 0, f"{k=} must be divisible by 32 for MXFP8"

            weight_flat = weight.view(-1, k).contiguous()
            problem_sizes = torch.empty(
                (num_experts, 3), dtype=torch.int32, device=weight.device
            )
            problem_sizes[:, 0] = m
            problem_sizes[:, 1] = 0
            problem_sizes[:, 2] = k
            expert_offsets = torch.arange(
                0, num_experts * m, m, dtype=torch.int32, device=weight.device
            )
            aligned_m = ((m + 127) // 128) * 128
            blockscale_offsets = torch.arange(
                0,
                num_experts * aligned_m,
                aligned_m,
                dtype=torch.int32,
                device=weight.device,
            )
            qweight = torch.empty_like(weight_flat, dtype=torch.float8_e4m3fn)
            scale = torch.empty(
                (num_experts * aligned_m, k // 32),
                dtype=torch.uint8,
                device=weight.device,
            )
            es_sm100_mxfp8_blockscaled_grouped_quant(
                weight_flat,
                problem_sizes,
                expert_offsets,
                blockscale_offsets,
                qweight,
                scale,
            )
            qweight = qweight.view_as(weight)
            scale = scale.view(num_experts, aligned_m, k // 32)
            if aligned_m != m:
                scale = scale[:, :m, :]
            return qweight, scale

        def _swizzle_mxfp8_sf(scale, num_warps):
            from triton_kernels.tensor import convert_layout, wrap_torch_tensor
            from triton_kernels.tensor_details import layout

            scale_layout = layout.make_default_matmul_mxfp4_w_scale_layout(
                mx_axis=-2, num_warps=num_warps
            )
            scale_layout_opts = {}
            scale = scale.transpose(-2, -1)
            scale = convert_layout(
                wrap_torch_tensor(scale), scale_layout, **scale_layout_opts
            )
            return scale

        def _swizzle_with_triton_kernel(
            weight_shape: tuple[int, int, int], scale: torch.Tensor
        ):
            num_experts, m, k = weight_shape
            aligned_m = ((m + 127) // 128) * 128
            scale = scale.view(num_experts, aligned_m, k // 32)
            num_warps = 8
            scale = _swizzle_mxfp8_sf(scale, num_warps)


            return scale.data

        def _quantize_and_swizzle_with_triton_kernel(weight: torch.Tensor):

            weight = weight.contiguous()
            _, _, k = weight.shape
            assert k % 32 == 0, f"{k=} must be divisible by 32 for MXFP8"

            weight_flat = weight.view(-1, k).contiguous()
            qweight, scale = mxfp8_group_quantize(weight_flat)
            qweight = qweight.view_as(weight)
            scale = _swizzle_with_triton_kernel(weight.shape, scale)
            return qweight, scale

        def _quantize_with_flashinfer_trtllm(weight: torch.Tensor):
            weight = weight.contiguous()
            num_experts, m, k = weight.shape
            assert k % 32 == 0, f"{k=} must be divisible by 32 for MXFP8"
            from sglang.srt.layers.quantization.fp8_utils import (
                flashinfer_mxfp8_quantize,
            )

            weight_flat = weight.view(-1, k).contiguous()
            qweight, scale = flashinfer_mxfp8_quantize(weight_flat, False)
            scale_u8 = (
                scale.view(torch.uint8).contiguous().view(num_experts, m, k // 32)
            )
            return qweight.view_as(weight), scale_u8

        from sglang.srt.layers.quantization.mxfp8_block_convert import _ue8m0_to_fp32

        def _quantize_for_deepgemm(weight: torch.Tensor):
            weight = weight.contiguous()
            num_experts, m, k = weight.shape
            assert k % 32 == 0, f"{k=} must be divisible by 32 for MXFP8"

            weight_flat = weight.view(-1, k).contiguous()
            qweight, scale_u8 = mxfp8_group_quantize(weight_flat)
            qweight = qweight.view_as(weight)
            scale_fp32 = _ue8m0_to_fp32(scale_u8).view(num_experts, m, k // 32)
            scale_packed = _pack_moe_scale_for_deepgemm(scale_fp32)
            return qweight, scale_packed

        def _pack_moe_scale_for_deepgemm(scale_fp32: torch.Tensor) -> torch.Tensor:

            from sglang.srt.layers.deep_gemm_wrapper.configurer import (
                DEEPGEMM_SCALE_UE8M0,
            )

            if DEEPGEMM_SCALE_UE8M0:
                import deep_gemm.utils.layout

                return (
                    deep_gemm.utils.layout.get_mn_major_tma_aligned_packed_ue8m0_tensor(
                        scale_fp32
                    )
                )
            return scale_fp32

        def _convert_ue8m0_scales_for_deepgemm(
            scale_u8: torch.Tensor, shape: tuple
        ) -> torch.Tensor:
            num_experts, m, k_groups = shape[0], shape[1], scale_u8.shape[-1]
            scale_fp32 = _ue8m0_to_fp32(scale_u8.contiguous().view(-1)).view(
                num_experts, m, k_groups
            )
            return _pack_moe_scale_for_deepgemm(scale_fp32)

        if quantize:
            if _is_hip:
                w13_q, w13_s_u8 = mxfp8_group_quantize(
                    layer.w13_weight.data.contiguous().view(
                        -1, layer.w13_weight.data.shape[-1]
                    )
                )
                w2_q, w2_s_u8 = mxfp8_group_quantize(
                    layer.w2_weight.data.contiguous().view(
                        -1, layer.w2_weight.data.shape[-1]
                    )
                )
                w13_q = w13_q.view_as(layer.w13_weight.data)
                w2_q = w2_q.view_as(layer.w2_weight.data)
                w13_s = w13_s_u8.view(
                    layer.w13_weight.data.shape[0],
                    layer.w13_weight.data.shape[1],
                    layer.w13_weight.data.shape[2] // 32,
                )
                w2_s = w2_s_u8.view(
                    layer.w2_weight.data.shape[0],
                    layer.w2_weight.data.shape[1],
                    layer.w2_weight.data.shape[2] // 32,
                )
            elif get_moe_runner_backend().is_cutlass():
                w13_q, w13_s = _quantize_and_swizzle_with_cutlass_es_kernel(
                    layer.w13_weight.data
                )
                w2_q, w2_s = _quantize_and_swizzle_with_cutlass_es_kernel(
                    layer.w2_weight.data
                )
            elif get_moe_runner_backend().is_deep_gemm():
                w13_q, w13_s = _quantize_for_deepgemm(layer.w13_weight.data)
                w2_q, w2_s = _quantize_for_deepgemm(layer.w2_weight.data)
            elif (
                get_moe_runner_backend().is_flashinfer_trtllm()
                or get_moe_runner_backend().is_flashinfer_trtllm_routed()
            ):


                w13_q, w13_s = _quantize_with_flashinfer_trtllm(layer.w13_weight.data)
                w2_q, w2_s = _quantize_with_flashinfer_trtllm(layer.w2_weight.data)
            else:
                w13_q, w13_s = _quantize_and_swizzle_with_triton_kernel(
                    layer.w13_weight.data
                )
                w2_q, w2_s = _quantize_and_swizzle_with_triton_kernel(
                    layer.w2_weight.data
                )
        else:
            if _is_hip:
                w13_q = layer.w13_weight.data
                w2_q = layer.w2_weight.data
                w13_s = layer.w13_weight_scale_inv.data
                w2_s = layer.w2_weight_scale_inv.data
            elif (
                get_moe_runner_backend().is_flashinfer_trtllm()
                or get_moe_runner_backend().is_flashinfer_trtllm_routed()
            ):
                w13_q = layer.w13_weight.data
                w2_q = layer.w2_weight.data
                w13_s = layer.w13_weight_scale_inv.data
                w2_s = layer.w2_weight_scale_inv.data
            elif get_moe_runner_backend().is_deep_gemm():
                w13_q = layer.w13_weight.data
                w2_q = layer.w2_weight.data
                w13_s = _convert_ue8m0_scales_for_deepgemm(
                    layer.w13_weight_scale_inv.data, layer.w13_weight.data.shape
                )
                w2_s = _convert_ue8m0_scales_for_deepgemm(
                    layer.w2_weight_scale_inv.data, layer.w2_weight.data.shape
                )
            else:
                w13_q = layer.w13_weight.data
                w2_q = layer.w2_weight.data
                w13_s = _swizzle_with_triton_kernel(
                    layer.w13_weight.data.shape, layer.w13_weight_scale_inv.data
                )
                w2_s = _swizzle_with_triton_kernel(
                    layer.w2_weight.data.shape, layer.w2_weight_scale_inv.data
                )


        def _copy_or_rebind(param: Parameter, new_value: torch.Tensor) -> None:
            if (
                param.data.shape == new_value.shape
                and param.data.dtype == new_value.dtype
            ):
                param.data.copy_(new_value)
            else:
                param.data = new_value

        _copy_or_rebind(layer.w13_weight, w13_q)
        _copy_or_rebind(layer.w2_weight, w2_q)
        _copy_or_rebind(layer.w13_weight_scale_inv, w13_s)
        _copy_or_rebind(layer.w2_weight_scale_inv, w2_s)
        layer.w13_weight.requires_grad_(False)
        layer.w2_weight.requires_grad_(False)
        layer.w13_weight_scale_inv.requires_grad_(False)
        layer.w2_weight_scale_inv.requires_grad_(False)
        layer.w13_weight_scale_inv.format_ue8m0 = True
        layer.w2_weight_scale_inv.format_ue8m0 = True
        layer.w13_input_scale = None
        layer.w2_input_scale = None

        if (
            get_moe_runner_backend().is_flashinfer_trtllm()
            or get_moe_runner_backend().is_flashinfer_trtllm_routed()
        ):
            from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
                align_mxfp8_moe_weights_for_flashinfer_trtllm,
            )

            align_mxfp8_moe_weights_for_flashinfer_trtllm(layer)

    def process_weights_after_loading(self, layer: Module) -> None:
        if _is_hip and _use_hip_int4:
            self.process_weights_hip_int4(layer)

        elif self.block_quant:

            self.process_weights_after_loading_block_quant(layer)


        elif not self.quant_config.is_checkpoint_fp8_serialized:

            w13_weight = torch.empty_like(layer.w13_weight.data, dtype=fp8_dtype)
            w2_weight = torch.empty_like(layer.w2_weight.data, dtype=fp8_dtype)


            layer.w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    layer.num_local_experts,
                    dtype=torch.float32,
                    device=w13_weight.device,
                ),
                requires_grad=False,
            )
            for expert in range(layer.num_local_experts):
                w13_weight[expert, :, :], layer.w13_weight_scale[expert] = (
                    scaled_fp8_quant(layer.w13_weight.data[expert, :, :])
                )
                w2_weight[expert, :, :], layer.w2_weight_scale[expert] = (
                    scaled_fp8_quant(layer.w2_weight.data[expert, :, :])
                )
            layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)

            if _is_hip:
                self.process_weights_hip_scale_padding(layer)


        else:


            if self.quant_config.activation_scheme == "static":
                if layer.w13_input_scale is None or layer.w2_input_scale is None:
                    raise ValueError(
                        "QuantConfig has static quantization, but found "
                        "activation scales are None."
                    )
                if not all_close_1d(layer.w13_input_scale) or not all_close_1d(
                    layer.w2_input_scale
                ):
                    print_warning_once(
                        "Found input_scales that are not equal for "
                        "fp8 MoE layer. Using the maximum across experts "
                        "for each layer. "
                    )
                layer.w13_input_scale = torch.nn.Parameter(
                    layer.w13_input_scale.max(), requires_grad=False
                )
                layer.w2_input_scale = torch.nn.Parameter(
                    layer.w2_input_scale.max(), requires_grad=False
                )


            if _is_fp8_fnuz:

                w13_weight, w13_weight_scale, w13_input_scale = (
                    normalize_e4m3fn_to_e4m3fnuz(
                        layer.w13_weight, layer.w13_weight_scale, layer.w13_input_scale
                    )
                )
                w2_weight, w2_weight_scale, w2_input_scale = (
                    normalize_e4m3fn_to_e4m3fnuz(
                        layer.w2_weight, layer.w2_weight_scale, layer.w2_input_scale
                    )
                )

                layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
                layer.w13_weight_scale = torch.nn.Parameter(
                    w13_weight_scale, requires_grad=False
                )
                if w13_input_scale is not None:
                    layer.w13_input_scale = torch.nn.Parameter(
                        w13_input_scale, requires_grad=False
                    )
                layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
                layer.w2_weight_scale = torch.nn.Parameter(
                    w2_weight_scale, requires_grad=False
                )
                if w2_input_scale is not None:
                    layer.w2_input_scale = torch.nn.Parameter(
                        w2_input_scale, requires_grad=False
                    )


            assert layer.w13_weight_scale is not None
            w13_num_shards = 2 if layer.moe_runner_config.is_gated else 1
            shard_size = layer.intermediate_size_per_partition
            max_w13_scales = layer.w13_weight_scale.max(dim=1).values

            if w13_num_shards > 1:
                for expert_id in range(layer.num_local_experts):
                    start = 0
                    for shard_id in range(w13_num_shards):
                        dq_weight = per_tensor_dequantize(
                            layer.w13_weight[expert_id][start : start + shard_size, :],
                            layer.w13_weight_scale[expert_id][shard_id],
                        )
                        (
                            layer.w13_weight[expert_id][start : start + shard_size, :],
                            _,
                        ) = scaled_fp8_quant(dq_weight, max_w13_scales[expert_id])
                        start += shard_size

            layer.w13_weight_scale = torch.nn.Parameter(
                max_w13_scales, requires_grad=False
            )

            if _is_hip:
                self.process_weights_hip_scale_padding(layer)


            if (
                get_moe_runner_backend().is_flashinfer_trtllm()
                or get_moe_runner_backend().is_flashinfer_trtllm_routed()
            ):
                from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
                    align_fp8_moe_weights_for_flashinfer_trtllm,
                )

                align_fp8_moe_weights_for_flashinfer_trtllm(layer)


        if (
            get_moe_runner_backend().is_flashinfer_trtllm()
            or get_moe_runner_backend().is_flashinfer_trtllm_routed()
        ) and self._owns_moe_runner:
            self._prepare_flashinfer_trtllm_activation_params(layer)

        if get_moe_runner_backend().is_hpc_ops():
            self._prepare_hpc_ops_weights(layer)

        if hasattr(layer, "dispatcher"):
            layer.dispatcher.set_quant_config({"weight_dtype": layer.w13_weight.dtype})

    def _prepare_flashinfer_trtllm_activation_params(self, layer: Module) -> None:

        num_experts = int(layer.num_local_experts)
        device = layer.w13_weight.device
        for name, value in (
            ("gemm1_alpha", self.moe_runner_config.gemm1_alpha),
            ("gemm1_beta", self.moe_runner_config.gemm1_beta),
            ("gemm1_clamp_limit", self.moe_runner_config.gemm1_clamp_limit),
        ):
            tensor = (
                None
                if value is None
                else torch.full(
                    (num_experts,),
                    float(value),
                    dtype=torch.float32,
                    device=device,
                )
            )
            setattr(layer, f"_flashinfer_trtllm_{name}", tensor)

    def _prepare_hpc_ops_weights(self, layer: Module) -> None:

        from sglang.srt.layers.moe.moe_runner.hpc_ops import pad_hpc_ops_block_scale

        if self.block_quant:
            layer.hpc_ops_w13_weight_scale = pad_hpc_ops_block_scale(
                layer.w13_weight_scale_inv.data.float()
            )
            layer.hpc_ops_w2_weight_scale = pad_hpc_ops_block_scale(
                layer.w2_weight_scale_inv.data.float()
            )
        else:
            if layer.w13_input_scale is None or layer.w2_input_scale is None:
                raise ValueError(
                    "The hpc_ops MoE runner backend requires static activation "
                    "scales for per-tensor FP8 models (activation_scheme="
                    "'static' in the checkpoint quantization config)."
                )
            layer.hpc_ops_gate_up_alphas = (
                layer.w13_weight_scale.data.float() * layer.w13_input_scale.data.float()
            )
            layer.hpc_ops_down_alphas = (
                layer.w2_weight_scale.data.float() * layer.w2_input_scale.data.float()
            )

    def _get_hpc_ops_quant_info(self, layer: torch.nn.Module):
        from sglang.srt.layers.moe.moe_runner.hpc_ops import HpcOpsMoeQuantInfo


        if (
            getattr(layer, "w13_weight_bias", None) is not None
            or getattr(layer, "w2_weight_bias", None) is not None
        ):
            raise ValueError(
                "The hpc_ops MoE runner backend does not support MoE GEMM "
                "biases (w13_weight_bias / w2_weight_bias); use another "
                "--moe-runner-backend for this model."
            )

        if self.block_quant:
            return HpcOpsMoeQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                block_quant=True,
                global_num_experts=int(layer.num_experts),
                moe_ep_rank=int(layer.moe_ep_rank),
                w13_weight_scale_inv=layer.hpc_ops_w13_weight_scale,
                w2_weight_scale_inv=layer.hpc_ops_w2_weight_scale,
                block_shape=self.quant_config.weight_block_size,
            )
        else:
            return HpcOpsMoeQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                block_quant=False,
                global_num_experts=int(layer.num_experts),
                moe_ep_rank=int(layer.moe_ep_rank),
                gate_up_alphas=layer.hpc_ops_gate_up_alphas,
                down_alphas=layer.hpc_ops_down_alphas,
                w13_input_scale=layer.w13_input_scale,
                w2_input_scale=layer.w2_input_scale,
            )

    def process_weights_hip_int4(self, layer: Module):


        layer.w13_weight = torch.nn.Parameter(
            shuffle_weight(layer.w13_weight.data, (16, 16)),
            requires_grad=False,
        )
        torch.cuda.empty_cache()
        layer.w2_weight = torch.nn.Parameter(
            shuffle_weight(layer.w2_weight.data, (16, 16)),
            requires_grad=False,
        )
        torch.cuda.empty_cache()


        assert layer.w13_weight_scale is not None
        w13_num_shards = 2 if layer.moe_runner_config.is_gated else 1
        shard_size = layer.intermediate_size_per_partition
        max_w13_scales = layer.w13_weight_scale.max(dim=1).values
        for expert_id in range(layer.num_local_experts):
            start = 0
            max_w13_scale_fp8 = max_w13_scales[expert_id]
            for shard_id in range(w13_num_shards):
                if layer.w13_weight_scale[expert_id][shard_id] != max_w13_scale_fp8:
                    int4_rescale = (
                        layer.w13_weight_scale[expert_id][shard_id] / max_w13_scale_fp8
                    )
                    layer.w13_weight_scale1[expert_id][
                        start : start + shard_size
                    ] *= int4_rescale
                start += shard_size

        layer.w13_weight_scale = torch.nn.Parameter(max_w13_scales, requires_grad=False)


        for expert_id in range(layer.num_local_experts):
            layer.w13_weight_scale1[expert_id] *= max_w13_scales[expert_id]
            layer.w2_weight_scale1[expert_id] *= layer.w2_weight_scale[expert_id]

    def process_weights_hip_scale_padding(self, layer: Module):
        padding_size = get_moe_padding_size(_use_aiter)
        if _use_aiter:
            layer.w13_weight = torch.nn.Parameter(
                shuffle_weight(layer.w13_weight.data, (16, 16)),
                requires_grad=False,
            )
            torch.cuda.empty_cache()
            layer.w2_weight = torch.nn.Parameter(
                shuffle_weight(layer.w2_weight.data, (16, 16)),
                requires_grad=False,
            )
            torch.cuda.empty_cache()


            layer.w13_weight_scale1 *= layer.w13_weight_scale.unsqueeze(-1)
            layer.w2_weight_scale1 *= layer.w2_weight_scale.unsqueeze(-1)
        elif get_bool_env_var("SGLANG_MOE_PADDING"):

            layer.w13_weight = torch.nn.Parameter(
                F.pad(layer.w13_weight.data, (0, padding_size), "constant", 0),
                requires_grad=False,
            )
            torch.cuda.empty_cache()
            layer.w2_weight = torch.nn.Parameter(
                F.pad(layer.w2_weight.data, (0, padding_size), "constant", 0),
                requires_grad=False,
            )
            torch.cuda.empty_cache()

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self._owns_moe_runner = False
        self.moe_runner_config = moe_runner_config
        moe_runner_backend = get_moe_runner_backend()

        if moe_runner_backend.is_auto():
            if self.is_deepgemm_moe_runner_backend_enabled():
                moe_runner_backend = MoeRunnerBackend.DEEP_GEMM
            elif (
                _is_hip
                and (_use_aiter or _use_hip_int4)
                and get_moe_a2a_backend().supports_aiter()
            ):
                moe_runner_backend = MoeRunnerBackend.AITER
            else:
                moe_runner_backend = MoeRunnerBackend.TRITON

        if (
            moe_runner_backend.is_deep_gemm()
            or moe_runner_backend.is_triton()
            or moe_runner_backend.is_aiter()
            or moe_runner_backend.is_flashinfer_trtllm()
            or moe_runner_backend.is_flashinfer_trtllm_routed()
            or moe_runner_backend.is_hpc_ops()
        ):
            self.runner = MoeRunner(moe_runner_backend, moe_runner_config)
            self._owns_moe_runner = True
        else:

            pass

    def get_triton_quant_info(self, layer: torch.nn.Module) -> TritonMoeQuantInfo:
        use_rocm_mxfp8 = self.use_mxfp8 and _is_hip and _is_gfx95_supported
        return TritonMoeQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            b13=getattr(layer, "w13_weight_bias", None),
            b2=getattr(layer, "w2_weight_bias", None),
            use_mxfp8=use_rocm_mxfp8,
            use_fp8_w8a8=not use_rocm_mxfp8,
            w13_scale=(
                layer.w13_weight_scale_inv
                if self.block_quant
                else layer.w13_weight_scale
            ),
            w2_scale=(
                layer.w2_weight_scale_inv if self.block_quant else layer.w2_weight_scale
            ),
            a13_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            block_shape=self.weight_block_size,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        moe_runner_config = self.moe_runner_config

        if use_intel_amx_backend(layer):
            from sglang.srt.layers.moe.topk import apply_topk_weights_cpu

            topk_weights, topk_ids, _ = dispatch_output.topk_output
            x, topk_weights = apply_topk_weights_cpu(
                moe_runner_config.apply_router_weight_on_input, topk_weights, x
            )

            output = torch.ops.sgl_kernel.fused_experts_cpu(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                False,
                CPUQuantMethod.FP8_W8A16,
                layer.w13_weight_scale_inv,
                layer.w2_weight_scale_inv,
                None,
                None,
                self.quant_config.weight_block_size,
                None,
                None,
                None,
                None,
                True,
                moe_runner_config.activation,
            )
            return StandardCombineInput(hidden_states=output)

        if (
            _is_hip
            and getattr(self, "runner", None) is not None
            and self.runner.runner_backend.is_aiter()
        ):
            quant_info = self.maybe_get_hip_aiter_quant_info(
                layer,
                moe_runner_config.no_combine,
            )
            if quant_info is not None:
                return self.runner.run(dispatch_output, quant_info)

        if use_intel_xpu_backend() and not (
            getattr(self, "runner", None) is not None
            and self.runner.runner_backend.is_triton()
        ):

            from sgl_kernel import fused_experts

            topk_weights, topk_ids, _ = dispatch_output.topk_output
            assert layer.w13_weight.dtype == layer.w2_weight.dtype
            use_fp8_w8a8 = layer.w13_weight.dtype == torch.float8_e4m3fn
            use_mxfp4_w4a16 = layer.w13_weight.dtype == torch.int8
            assert self.is_fp4_expert == use_mxfp4_w4a16
            output = fused_experts(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                b1=getattr(layer, "w13_weight_bias", None),
                b2=getattr(layer, "w2_weight_bias", None),
                use_mxfp4_w4a16=use_mxfp4_w4a16,
                use_fp8_w8a8=use_fp8_w8a8,
                w1_scale=(
                    layer.w13_weight_scale_inv
                    if self.block_quant
                    else layer.w13_weight_scale
                ),
                w2_scale=(
                    layer.w2_weight_scale_inv
                    if self.block_quant
                    else layer.w2_weight_scale
                ),
                activation=moe_runner_config.activation,
                routed_scaling_factor=moe_runner_config.routed_scaling_factor,
                gemm1_alpha=moe_runner_config.gemm1_alpha,
                gemm1_limit=moe_runner_config.gemm1_clamp_limit,
                swiglu_limit=moe_runner_config.swiglu_limit,
            )
            return StandardCombineInput(hidden_states=output)

        if get_moe_runner_backend().is_cutlass():
            from sglang.srt.layers.moe.cutlass_moe import cutlass_fused_experts_fp8

            with use_symmetric_memory(
                get_tp_group(), disabled=not is_allocation_symmetric()
            ):
                symm_output = torch.empty_like(x)

            topk_weights, topk_ids, _ = dispatch_output.topk_output
            use_mxfp8 = getattr(self.quant_config, "use_mxfp8", False)
            output = cutlass_fused_experts_fp8(
                x,
                layer.w13_weight.transpose(1, 2),
                layer.w2_weight.transpose(1, 2),
                layer.w13_weight_scale_inv.transpose(1, 2),
                layer.w2_weight_scale_inv.transpose(1, 2),
                topk_weights,
                topk_ids,
                self.ab_strides1,
                self.c_strides1,
                self.ab_strides2,
                self.c_strides2,
                self.workspace,
                self.a_ptr,
                self.b_ptr,
                self.out_ptr,
                self.a_scales_ptr,
                self.b_scales_ptr,
                self.expert_offsets,
                self.problem_sizes1,
                self.problem_sizes2,
                use_fp8_blockscale=True,
                use_mxfp8=use_mxfp8,
                output=symm_output,
                enable_es=(use_mxfp8, use_mxfp8),
            )
            return StandardCombineInput(hidden_states=output)

        if self.runner.runner_backend.is_deep_gemm():

            w13_weight = layer.w13_weight
            w2_weight = layer.w2_weight

            if self.block_quant:
                block_shape = self.quant_config.weight_block_size
                w13_scale = layer.w13_weight_scale_inv
                w2_scale = layer.w2_weight_scale_inv
            else:

                scale_block_size = 128
                block_shape = [scale_block_size, scale_block_size]
                w13_scale_n = (w13_weight.shape[1] - 1) // scale_block_size + 1
                w13_scale_k = (w13_weight.shape[2] - 1) // scale_block_size + 1
                w13_scale = (
                    layer.w13_weight_scale.unsqueeze(1)
                    .repeat_interleave(w13_scale_n, dim=1)
                    .unsqueeze(2)
                    .repeat_interleave(w13_scale_k, dim=2)
                )
                w2_scale_n = (w2_weight.shape[1] - 1) // scale_block_size + 1
                w2_scale_k = (w2_weight.shape[2] - 1) // scale_block_size + 1
                w2_scale = (
                    layer.w2_weight_scale.unsqueeze(1)
                    .repeat_interleave(w2_scale_n, dim=1)
                    .unsqueeze(2)
                    .repeat_interleave(w2_scale_k, dim=2)
                )
            quant_info = DeepGemmMoeQuantInfo(
                w13_weight=w13_weight,
                w2_weight=w2_weight,
                use_fp8=True,
                w13_scale=w13_scale,
                w2_scale=w2_scale,
                block_shape=block_shape,
                is_fp4_experts=self.is_fp4_expert,
                use_mxfp8=self.use_mxfp8,
            )
        elif (
            self.runner.runner_backend.is_flashinfer_trtllm()
            or self.runner.runner_backend.is_flashinfer_trtllm_routed()
        ):


            global_num_experts = int(getattr(layer, "num_experts"))
            num_local_experts = int(getattr(layer, "num_local_experts"))
            moe_ep_rank = int(getattr(layer, "moe_ep_rank"))

            from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
                get_activation_type,
            )

            activation_type = get_activation_type(
                self.moe_runner_config.activation,
                is_gated=self.moe_runner_config.is_gated,
            )

            quant_info = FlashInferTrtllmFp8MoeQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                global_num_experts=global_num_experts,
                local_expert_offset=moe_ep_rank * num_local_experts,
                local_num_experts=num_local_experts,
                intermediate_size=layer.w2_weight.shape[2],
                routing_method_type=int(
                    getattr(layer, "routing_method_type", None)
                    or RoutingMethodType.DeepSeekV3
                ),
                block_quant=self.block_quant,
                use_mxfp8=getattr(self.quant_config, "use_mxfp8", False),
                weight_block_k=(
                    None
                    if self.quant_config.weight_block_size is None
                    else self.quant_config.weight_block_size[1]
                ),
                w13_weight_scale_inv=(
                    layer.w13_weight_scale_inv if self.block_quant else None
                ),
                w2_weight_scale_inv=(
                    layer.w2_weight_scale_inv if self.block_quant else None
                ),
                gemm1_alpha=layer._flashinfer_trtllm_gemm1_alpha,
                gemm1_beta=layer._flashinfer_trtllm_gemm1_beta,
                gemm1_clamp_limit=layer._flashinfer_trtllm_gemm1_clamp_limit,
                w13_input_scale=layer.w13_input_scale if not self.block_quant else None,
                output1_scales_scalar=(
                    getattr(layer, "output1_scales_scalar", None)
                    if not self.block_quant
                    else None
                ),
                output1_scales_gate_scalar=(
                    getattr(layer, "output1_scales_gate_scalar", None)
                    if not self.block_quant
                    else None
                ),
                output2_scales_scalar=(
                    getattr(layer, "output2_scales_scalar", None)
                    if not self.block_quant
                    else None
                ),
                activation_type=activation_type,
            )
        elif self.runner.runner_backend.is_hpc_ops():
            quant_info = self._get_hpc_ops_quant_info(layer)
        elif self.runner.runner_backend.is_triton():
            quant_info = self.get_triton_quant_info(layer)
        else:
            raise NotImplementedError(
                "Unsupported runner backend: %s" % self.runner.runner_backend
            )

        return self.runner.run(dispatch_output, quant_info)

    def _ensure_cutlass_buffers_initialized(self, layer: Module) -> None:
        if getattr(self, "_cutlass_buffers_ready", False):
            return

        device = layer.w13_weight.device
        num_experts = layer.w13_weight.shape[0]
        hidden_size = layer.w2_weight.shape[1]
        intermediate_size_per_partition = layer.intermediate_size_per_partition
        w13_num_shards = 2 if layer.moe_runner_config.is_gated else 1

        self.ab_strides1 = torch.full(
            (num_experts,), hidden_size, device=device, dtype=torch.int64
        )
        self.c_strides1 = torch.full(
            (num_experts,),
            w13_num_shards * intermediate_size_per_partition,
            device=device,
            dtype=torch.int64,
        )
        self.ab_strides2 = torch.full(
            (num_experts,),
            intermediate_size_per_partition,
            device=device,
            dtype=torch.int64,
        )
        self.c_strides2 = torch.full(
            (num_experts,), hidden_size, device=device, dtype=torch.int64
        )
        self.workspace = torch.empty(90000, device=device, dtype=torch.uint8)
        self.a_ptr = torch.empty(num_experts, device=device, dtype=torch.int64)
        self.b_ptr = torch.empty(num_experts, device=device, dtype=torch.int64)
        self.out_ptr = torch.empty(num_experts, device=device, dtype=torch.int64)
        self.a_scales_ptr = torch.empty(num_experts, device=device, dtype=torch.int64)
        self.b_scales_ptr = torch.empty(num_experts, device=device, dtype=torch.int64)
        self.expert_offsets = torch.empty(
            num_experts + 1, device=device, dtype=torch.int32
        )
        self.problem_sizes1 = torch.empty(
            num_experts, 3, device=device, dtype=torch.int32
        )
        self.problem_sizes2 = torch.empty(
            num_experts, 3, device=device, dtype=torch.int32
        )

        self._cutlass_buffers_ready = True

    def maybe_get_hip_aiter_quant_info(
        self,
        layer: torch.nn.Module,
        no_combine: bool = False,
    ) -> Optional[AiterMoeQuantInfo]:
        if not (_use_aiter or _use_hip_int4):
            return None
        assert not no_combine, f"{no_combine=} is not supported."

        from sglang.srt.layers.moe.moe_runner.aiter import (
            AiterMoeQuantInfo,
            AiterQuantType,
        )

        w13_weight = layer.w13_weight
        w2_weight = layer.w2_weight

        if self.block_quant:
            quant_type = (
                AiterQuantType.PER_1X32
                if self.is_fp4_expert
                else AiterQuantType.PER_128X128
            )

            if self.is_fp4_expert:
                fp4_weight_dtype = _require_fp4_dtype()
                w13_weight = w13_weight.view(fp4_weight_dtype)
                w2_weight = w2_weight.view(fp4_weight_dtype)
                if getattr(layer.w13_weight, "is_shuffled", False):
                    w13_weight.is_shuffled = True
                    w2_weight.is_shuffled = True
            w13_scale = layer.w13_weight_scale_inv
            w2_scale = layer.w2_weight_scale_inv
        else:
            quant_type = AiterQuantType.PER_TOKEN
            w13_scale = layer.w13_weight_scale1
            w2_scale = layer.w2_weight_scale1
        return AiterMoeQuantInfo(
            w13_weight=w13_weight,
            w2_weight=w2_weight,
            quant_type=quant_type,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
            expert_mask=layer.dispatcher.expert_mask_gpu if _use_aiter else None,
            swiglu_limit=self.moe_runner_config.swiglu_limit or 0.0,
            hidden_pad=getattr(layer, "hidden_pad", 0),
            intermediate_pad=getattr(layer, "intermediate_pad", 0),
        )


class Fp8KVCacheMethod(BaseKVCacheMethod):


    def __init__(self, quant_config: Fp8Config):
        super().__init__(quant_config)
