# SPDX-License-Identifier: Apache-2.0

import fnmatch
import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

import torch

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod
from sglang.srt.layers.quantization.quark.schemes import (
    QuarkLinearScheme,
    QuarkMoEScheme,
    QuarkW4A4MXFP4,
    QuarkW4A4MXFp4MoE,
    QuarkW4A8MXFp4MoE,
    QuarkW8A8Fp8,
    QuarkW8A8FP8MoE,
)
from sglang.srt.layers.quantization.quark.utils import (
    Nvfp4SourceConfig,
    deep_compare,
    should_ignore_layer,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.utils import get_device_capability

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput

__all__ = ["QuarkLinearMethod", "QuarkFusedMoEMethod"]


def _parse_nvfp4_excludes(hf_quant_config: Dict[str, Any]) -> List[str]:

    pats = (
        hf_quant_config.get("ignore")
        or hf_quant_config.get("exclude_modules")
        or hf_quant_config.get("exclude")
        or []
    )
    return [p if p.startswith("re:") else "re:" + fnmatch.translate(p) for p in pats]


def _detect_nvfp4_source(config: Dict[str, Any]) -> Optional["Nvfp4SourceConfig"]:

    from sglang.srt.layers.quantization.quark.utils import Nvfp4SourceConfig

    quant_method = config.get("quant_method", "")
    quant_algo = (config.get("quant_algo") or "").upper()

    if quant_method in ("modelopt", "modelopt_fp4", "nvfp4") and quant_algo in (
        "",
        "NVFP4",
        "FP4",
    ):
        return Nvfp4SourceConfig()
    if quant_method == "quark":
        gqc = config.get("global_quant_config", {})
        weight = gqc.get("weight")
        if not (isinstance(weight, list) and len(weight) == 2):
            return None
        w0, w1 = weight
        is_nvfp4_weight = (
            isinstance(w0, dict)
            and w0.get("dtype") == "fp4"
            and w0.get("qscheme") == "per_group"
            and w0.get("group_size") == 16
            and not w0.get("is_dynamic")
        )
        is_nvfp4_scale_2 = (
            isinstance(w1, dict)
            and w1.get("dtype") == "fp8_e4m3"
            and w1.get("qscheme") == "per_tensor"
            and not w1.get("is_dynamic")
        )
        if is_nvfp4_weight and is_nvfp4_scale_2:
            return Nvfp4SourceConfig()
        return None
    if quant_method in ("compressed-tensors", "compressed_tensors"):
        raise NotImplementedError(
            "Online MXFP4 requantization from compressed-tensors NVFP4 "
            "checkpoints is not supported at this time."
        )
    return None


_MXFP4_TARGET_SPEC: Dict[str, Any] = {
    "weight": {
        "dtype": "fp4",
        "qscheme": "per_group",
        "group_size": 32,
        "is_dynamic": False,
        "scale_format": "e8m0",
    },
    "input_tensors": {
        "dtype": "fp4",
        "qscheme": "per_group",
        "group_size": 32,
        "is_dynamic": True,
        "scale_format": "e8m0",
    },
    "output_tensors": None,
    "bias": None,
}


def _fp8_per_tensor_spec(is_dynamic_input: bool) -> Dict[str, Any]:
    return {
        "weight": {
            "dtype": "fp8_e4m3",
            "qscheme": "per_tensor",
            "is_dynamic": False,
        },
        "input_tensors": {
            "dtype": "fp8_e4m3",
            "qscheme": "per_tensor",
            "is_dynamic": is_dynamic_input,
        },
        "output_tensors": None,
        "bias": None,
    }


def _fp8_is_dynamic_from_config_groups(
    config_groups: Any,
) -> bool:

    if not isinstance(config_groups, dict):
        return True
    for group in config_groups.values():
        if not isinstance(group, dict):
            continue
        input_act = group.get("input_activations") or {}
        if input_act.get("num_bits") == 8:
            return bool(input_act.get("dynamic", True))
    return True


def _mixed_precision_layer_map(config: Dict[str, Any]) -> Optional[Dict[str, str]]:

    if (config.get("quant_algo") or "").upper() != "MIXED_PRECISION":
        return None
    quantized_layers = config.get("quantized_layers")
    if not isinstance(quantized_layers, dict) or not quantized_layers:
        return None
    layer_map: Dict[str, str] = {}
    for name, info in quantized_layers.items():
        if isinstance(info, dict):
            layer_map[name] = str(info.get("quant_algo", "")).upper()
    return layer_map


def _build_mixed_precision_layer_quant_config(
    layer_map: Dict[str, str],
    config_groups: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], bool]:


    tail_algos: Dict[str, set] = {}
    for name, algo in layer_map.items():


        tail = re.split(r"\.layers\.\d+\.", name, maxsplit=1)[-1]
        tail_algos.setdefault(tail, set()).add(algo)

    fp8_is_dynamic = _fp8_is_dynamic_from_config_groups(config_groups or {})
    fp8_spec = _fp8_per_tensor_spec(is_dynamic_input=fp8_is_dynamic)

    layer_quant_config: Dict[str, Any] = {}
    has_nvfp4 = False
    for tail, algos in tail_algos.items():
        if len(algos) != 1:
            raise NotImplementedError(
                f"MIXED_PRECISION layer group {tail!r} has inconsistent "
                f"quant algos across layers: {sorted(algos)}. SGLang requires "
                "all layers in a group to share one algo."
            )
        algo = next(iter(algos))
        pattern = "*" + tail
        if algo in ("NVFP4", "W4A16_NVFP4"):
            layer_quant_config[pattern] = _MXFP4_TARGET_SPEC
            has_nvfp4 = True
        elif algo == "FP8":
            layer_quant_config[pattern] = fp8_spec
        else:
            raise NotImplementedError(
                f"MIXED_PRECISION layer group {tail!r} uses unsupported "
                f"quant algo {algo!r}; online requantization supports NVFP4 "
                "(-> MXFP4) and FP8 (kept as-is) only."
            )
    return layer_quant_config, has_nvfp4


def _build_excluded_fp8_config(config: Dict[str, Any]) -> Optional["Fp8Config"]:

    if config.get("quant_method") != "fp8":
        return None


    return Fp8Config.from_config(config)


logger = logging.getLogger(__name__)

_MOE_SHARED_EXPERT_QUANT_LAYER0_BASES: tuple[str, ...] = (
    "model.layers.0",
    "model.language_model.layers.0",
)

_SHARED_EXPERT_BODY_PROJ_SUFFIXES: tuple[str, ...] = (
    "gate_proj",
    "up_proj",
    "gate_up_proj",
    "down_proj",
)


class QuarkConfig(QuantizationConfig):

    def __init__(
        self,
        quant_config: dict[str, Any] | None = None,
        hf_config: "PretrainedConfig | None" = None,
        kv_cache_group: Optional[list[str]] = None,
        kv_cache_config: Optional[dict[str, Any]] = None,
        pack_method: str = "reorder",
        is_prequantized: bool = False,
        online_scheme: Optional[str] = None,
        dequantization_config: Optional[QuantizationConfig] = None,
        excluded_fp8_config: Optional[Fp8Config] = None,
    ):
        super().__init__()
        if kv_cache_group is None:
            kv_cache_group = []

        if online_scheme is not None:
            assert not is_prequantized
            if online_scheme == "quark_mxfp4":
                quant_config = self._create_online_mxfp4_config(
                    model_type=hf_config.model_type
                )
            else:
                raise ValueError(f"Unsupported online_scheme: {online_scheme}")

        if quant_config is None:
            raise ValueError("Either quant_config or online_scheme must be provided")

        self.online_scheme = online_scheme
        self.quant_config = quant_config
        self.kv_cache_group = kv_cache_group
        self.kv_cache_config = kv_cache_config
        self.pack_method = pack_method
        self.exclude_layers = cast(list[str], self.quant_config.get("exclude", []))


        self.num_hidden_layers = getattr(hf_config, "num_hidden_layers", None)
        self.num_nextn_predict_layers = int(
            getattr(hf_config, "num_nextn_predict_layers", 0) or 0
        )
        self.is_prequantized = is_prequantized
        self.dequantization_config = dequantization_config


        self.excluded_fp8_config = excluded_fp8_config
        self.packed_modules_mapping = self.quant_config["packed_modules_mapping"]
        self._online_quantized_layers = set()

        if isinstance(self.dequantization_config, Fp8Config):
            self.weight_block_size = self.dequantization_config.weight_block_size

    @property
    def quantized_layers(self) -> tuple[list[str], int]:


        layer_types = sorted(
            set(name.split(".")[-1] for name in self._online_quantized_layers)
        )
        return layer_types, len(self._online_quantized_layers)

    def get_linear_method(self) -> "QuarkLinearMethod":
        return QuarkLinearMethod(self)

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def get_name(self) -> str:
        return "quark"

    def apply_weight_name_mapper(self, hf_to_sglang_mapper):
        mapped = hf_to_sglang_mapper.apply_list(self.exclude_layers)
        expanded = []
        for name in mapped:
            expanded.append(name)
            if name.startswith("language_model."):
                expanded.append(name.removeprefix("language_model."))
        self.exclude_layers = list(dict.fromkeys(expanded))

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:

        if should_ignore_layer(
            prefix,
            ignore=self.exclude_layers,
            fused_mapping=self.packed_modules_mapping,
        ):
            if isinstance(layer, LinearBase):


                if self.excluded_fp8_config is not None:
                    return Fp8LinearMethod(quant_config=self.excluded_fp8_config)
                return UnquantizedLinearMethod()
            elif isinstance(layer, RadixAttention):
                return QuarkKVCacheMethod(self)
            return None

        if isinstance(layer, LinearBase):
            scheme = self.get_linear_scheme(layer=layer, layer_name=prefix)
            layer.scheme = scheme
            self._online_quantized_layers.add(prefix)
            return QuarkLinearMethod(self)

        if isinstance(layer, RadixAttention):
            self._online_quantized_layers.add(prefix)
            return QuarkKVCacheMethod(self)

        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        if isinstance(layer, FusedMoE):
            self._online_quantized_layers.add(prefix)
            layer.scheme = self.get_moe_scheme(layer, prefix)
            return QuarkFusedMoEMethod(self)

        return None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "QuarkConfig":


        if config.get("requantization_method") == "quark_mxfp4":
            hf_config = config["hf_config"]


            layer_map = _mixed_precision_layer_map(config)
            if layer_map is not None:
                config_groups = config.get("config_groups")
                layer_quant_config, has_nvfp4 = (
                    _build_mixed_precision_layer_quant_config(layer_map, config_groups)
                )
                if not has_nvfp4:
                    raise NotImplementedError(
                        "MIXED_PRECISION checkpoint has no NVFP4 layers to "
                        "requantize; load it with its native quantization "
                        "method instead of --quantization quark_mxfp4."
                    )
                source_excludes = _parse_nvfp4_excludes(config)
                quant_config = QuarkConfig._create_online_mxfp4_config(
                    model_type=hf_config.model_type,
                    source_excludes=source_excludes,
                    layer_quant_config=layer_quant_config,
                    packed_modules_mapping=config.get("packed_modules_mapping"),
                )


                excluded_fp8_config = _build_excluded_fp8_config(config)
                return cls(
                    quant_config=quant_config,
                    hf_config=hf_config,
                    is_prequantized=False,
                    dequantization_config=Nvfp4SourceConfig(),
                    excluded_fp8_config=excluded_fp8_config,
                )

            nvfp4_src = _detect_nvfp4_source(config)
            if nvfp4_src is not None:
                source_excludes = _parse_nvfp4_excludes(config)
                quant_config = QuarkConfig._create_online_mxfp4_config(
                    model_type=hf_config.model_type,
                    source_excludes=source_excludes,
                )
                return cls(
                    quant_config=quant_config,
                    hf_config=hf_config,
                    is_prequantized=False,
                    dequantization_config=nvfp4_src,
                )


            if (
                config.get("quant_method") == "fp8"
                and config.get("activation_scheme") == "dynamic"
            ):
                quant_config = QuarkConfig._create_online_mxfp4_config(
                    model_type=hf_config.model_type
                )
                dequantization_config = Fp8Config.from_config(config)
                return cls(
                    quant_config=quant_config,
                    hf_config=hf_config,
                    is_prequantized=False,
                    dequantization_config=dequantization_config,
                    online_scheme=config["requantization_method"],
                )

            raise NotImplementedError(
                f"Requantization into {config['requantization_method']} is not supported, "
                f"from the original quant_method={config['quant_method']} "
                f"and activation_scheme={config.get('activation_scheme')}."
            )

        if config["quant_method"] != "quark":
            raise ValueError(
                f"QuarkConfig.from_config invoked with non-quark quant_method "
                f"{config['quant_method']!r} but no requantization_method set."
            )

        export_config = config.get("export")
        if export_config is None:
            raise ValueError(
                "The export key should be included in "
                "the configurations of Quark quantized model"
            )

        kv_cache_group = cast(list[str], export_config.get("kv_cache_group"))
        pack_method = cast(str, export_config.get("pack_method"))


        if len(kv_cache_group) == 0:
            kv_cache_config = None
        else:
            kv_cache_set = set(kv_cache_group)
            layer_quant_config = cast(dict[str, Any], config.get("layer_quant_config"))
            layer_quant_names = list(layer_quant_config.keys())
            layer_quant_set = set(layer_quant_names)

            if not kv_cache_set.issubset(layer_quant_set):
                raise ValueError(
                    "The Quark quantized model has the "
                    "kv_cache_group parameter setting, "
                    "but no kv_cache quantization settings "
                    "were found in the quantization "
                    "configuration."
                )

            q_configs = [
                cast(dict[str, Any], layer_quant_config.get(name))
                for name in kv_cache_group
            ]
            if not all(deep_compare(q_config, q_configs[0]) for q_config in q_configs):
                raise ValueError(
                    "The quantization method used for kv_cache should "
                    "be the same, but the quantization method for the "
                    "kv_cache layer in the config is different."
                )
            kv_cache_config = q_configs[0].get("output_tensors")
            if kv_cache_config is None:
                raise ValueError("The kv_cache quantization configuration is empty.")


            for q_config in q_configs:
                q_config["output_tensors"] = None


            q_proj_q_config = cast(dict[str, Any], layer_quant_config.get("*q_proj"))
            if q_proj_q_config is not None:
                q_proj_q_config["output_tensors"] = None

        return cls(
            quant_config=config,


            hf_config=config.get("hf_config"),
            kv_cache_group=kv_cache_group,
            kv_cache_config=kv_cache_config,
            pack_method=pack_method,
            is_prequantized=True,
        )

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @staticmethod
    def _create_online_mxfp4_config(
        model_type: str,
        source_excludes: Optional[list[str]] = None,
        layer_quant_config: Optional[dict[str, Any]] = None,
        packed_modules_mapping: Optional[dict[str, list[str]]] = None,
    ) -> dict[str, Any]:


        exclude = [
            "re:.*gate$",
            "re:.*router",
            "re:.*lm_head",
            "re:.*embed_tokens",
        ]

        if source_excludes:
            exclude.extend(source_excludes)
        elif layer_quant_config is None:


            if model_type in ("deepseek_v3", "deepseek_v32", "deepseek_v4"):
                exclude.extend(
                    [
                        "re:.*model.layers.61.*",
                        "re:.*self_attn.*",
                        "re:.*mlp.gate$",
                    ]
                )
            elif model_type == "qwen3_5_moe":


                exclude.extend(
                    [
                        "re:.*n_proj_a",
                        "re:.*in_proj_b",
                        "re:.*in_proj_qkv",
                        "re:.*in_proj_z",
                        "re:.*o_proj",
                        "re:.*out_proj",
                        "re:.*qkv_proj",
                        "re:.*shared_expert",
                    ]
                )

        return {
            "packed_modules_mapping": packed_modules_mapping or {},
            "exclude": exclude,
            "global_quant_config": {
                "weight": {
                    "dtype": "fp4",
                    "qscheme": "per_group",
                    "group_size": 32,
                    "is_dynamic": False,
                    "scale_format": "e8m0",
                },
                "input_tensors": {
                    "dtype": "fp4",
                    "qscheme": "per_group",
                    "group_size": 32,
                    "is_dynamic": True,
                    "scale_format": "e8m0",
                },
                "output_tensors": None,
                "bias": None,
            },
            "layer_quant_config": layer_quant_config or {},
            "layer_type_quant_config": {},
            "export": {
                "kv_cache_group": [],
                "pack_method": "reorder",
            },
        }

    def _check_scheme_supported(self, min_capability: int, error: bool = True) -> bool:
        capability_tuple = get_device_capability()

        if capability_tuple is not None:
            assert 0 <= capability_tuple[1] < 10
            capability = capability_tuple[0] * 10 + capability_tuple[1]

            supported = capability >= min_capability
            if error and not supported:


                raise RuntimeError(
                    "Quantization scheme is not supported for "
                    f"the current GPU. Min capability: {min_capability}. "
                    f"Current capability: {capability}."
                )
            return supported
        else:
            return False

    def _is_fp8_w8a8(
        self,
        weight_quant: Optional[dict[str, Any]],
        input_quant: Optional[dict[str, Any]],
    ) -> bool:

        if weight_quant is None or input_quant is None:
            return False


        is_fp8_dtype = (
            weight_quant.get("dtype") == "fp8_e4m3"
            and input_quant.get("dtype") == "fp8_e4m3"
        )
        is_static_weight = not weight_quant.get("is_dynamic")
        is_per_tensor_or_channel_weight = weight_quant.get("qscheme") in [
            "per_tensor",
            "per_channel",
        ]

        if not (is_fp8_dtype and is_static_weight and is_per_tensor_or_channel_weight):
            return False


        if input_quant.get("is_dynamic"):
            return True


        is_per_tensor_activation = input_quant.get("qscheme") == "per_tensor"
        return is_per_tensor_activation

    def _is_mx_fp4(
        self,
        weight_quant: Optional[dict[str, Any]],
        input_quant: Optional[dict[str, Any]],
    ) -> bool:

        if weight_quant is None or input_quant is None:
            logger.debug(
                "Quark model is not in MX-FP4 format: "
                "weight_quant or input_quant not set"
            )
            return False


        if weight_quant.get("dtype") != "fp4" or input_quant.get("dtype") != "fp4":
            logger.debug("Quark model is not in MX-FP4 format: dtype not fp4")
            return False


        if (
            weight_quant.get("qscheme") != "per_group"
            or input_quant.get("qscheme") != "per_group"
        ):
            logger.debug("Quark model is not in MX-FP4 format: not per_group")
            return False


        if weight_quant.get("group_size") != 32 or input_quant.get("group_size") != 32:
            logger.debug("Quark model is not in MX-FP4 format: not group_size=32")
            return False


        if weight_quant.get("is_dynamic") is True:
            logger.debug("Quark model is not in MX-FP4 format: not weight static")
            return False


        if input_quant.get("is_dynamic") is False:
            logger.debug("Quark model is not in MX-FP4 format: not activation dynamic")
            return False


        if (
            weight_quant.get("scale_format") != "e8m0"
            or input_quant.get("scale_format") != "e8m0"
        ):
            logger.debug("Quark model is not in MX-FP4 format: not scale_format e8m0")
            return False

        return True

    def _is_mx_w4a8(
        self,
        weight_quant: Optional[dict[str, Any]],
        input_quant: Optional[dict[str, Any]],
    ) -> bool:
        if weight_quant is None or input_quant is None:
            return False

        is_mx_fp4_weight = (
            weight_quant.get("dtype") == "fp4"
            and weight_quant.get("qscheme") == "per_group"
            and weight_quant.get("group_size") == 32
            and not weight_quant.get("is_dynamic")
            and weight_quant.get("scale_format") == "e8m0"
        )
        is_static_fp8_activation = (
            input_quant.get("dtype") in ("fp8_e4m3", "fp8_e4m3fn")
            and input_quant.get("qscheme") == "per_tensor"
            and not input_quant.get("is_dynamic")
        )
        return is_mx_fp4_weight and is_static_fp8_activation

    def _find_matched_config(
        self, layer_name: str, module: torch.nn.Module
    ) -> dict[str, Any]:

        proj_name = layer_name.split(".")[-1]
        if proj_name in self.packed_modules_mapping:
            shard_proj_names = self.packed_modules_mapping[proj_name]


            shard_names = [
                layer_name.replace(proj_name, shard_proj_name)
                for shard_proj_name in shard_proj_names
            ]
            shard_configs = [
                self._find_matched_config(shard_name, module)
                for shard_name in shard_names
            ]
            if not all(
                deep_compare(q_config, shard_configs[0]) for q_config in shard_configs
            ):
                raise ValueError(
                    f"Found a different quantization configuration for "
                    f"{shard_proj_names} in {layer_name}. SGLang "
                    "requires all to use the same scheme."
                )
            return shard_configs[0]
        else:
            layer_quant_config = cast(
                dict[str, Any], self.quant_config.get("layer_quant_config")
            )
            for name_pattern in layer_quant_config:
                if fnmatch.fnmatch(layer_name, name_pattern):
                    return layer_quant_config[name_pattern]

            layer_type = type(module).__name__
            layer_type_quant_config = cast(
                dict[str, Any], self.quant_config.get("layer_type_quant_config")
            )
            if layer_type in layer_type_quant_config:
                return layer_type_quant_config[layer_type]

            global_quant_config = cast(
                dict[str, Any], self.quant_config.get("global_quant_config")
            )
            return global_quant_config

    def _get_scheme_from_config(self, config: dict[str, Any]) -> "QuarkLinearScheme":
        if config.get("output_tensors") or config.get("bias"):
            raise NotImplementedError(
                "Currently, Quark models with output_tensors "
                "and bias quantized are not supported"
            )
        weight_config = cast(dict[str, Any], config.get("weight"))
        input_config = cast(dict[str, Any], config.get("input_tensors"))

        if self._is_mx_fp4(weight_config, input_config):
            return QuarkW4A4MXFP4(
                weight_config,
                input_config,
                is_checkpoint_mxfp4_serialized=self.is_prequantized,
                dequantization_config=self.dequantization_config,
            )
        if self._is_fp8_w8a8(weight_config, input_config):
            is_fp8_w8a8_supported = self._check_scheme_supported(
                QuarkW8A8Fp8.get_min_capability(), error=False
            )
            if is_fp8_w8a8_supported:
                return QuarkW8A8Fp8(weight_config, input_config)

        raise NotImplementedError(
            "No quark compatible scheme was found. "
            f"Weight config: {weight_config}, "
            f"Input config: {input_config}"
        )

    def get_linear_scheme(
        self, layer: torch.nn.Module, layer_name: str
    ) -> "QuarkLinearScheme":

        layer_quant_config = self._find_matched_config(layer_name, layer)


        scheme = self._get_scheme_from_config(layer_quant_config)


        self._check_scheme_supported(scheme.get_min_capability())

        return scheme

    def get_moe_scheme(
        self,
        module: torch.nn.Module,
        layer_name: str,
    ) -> "QuarkMoEScheme":
        layer_quant_config = self._find_matched_config(layer_name, module)

        if layer_quant_config.get("output_tensors") or layer_quant_config.get("bias"):
            raise NotImplementedError(
                "Currently, Quark models with "
                "output_tensors and bias "
                "quantized are not supported"
            )
        weight_config = layer_quant_config.get("weight")
        input_config = layer_quant_config.get("input_tensors")

        if self._is_mx_fp4(weight_config, input_config):
            return QuarkW4A4MXFp4MoE(
                weight_config,
                input_config,
                is_checkpoint_mxfp4_serialized=self.is_prequantized,
                dequantization_config=self.dequantization_config,
            )
        elif self._is_mx_w4a8(weight_config, input_config):
            logger.info_once("Using Quark MXFP4-W/FP8-A MoE scheme")
            return QuarkW4A8MXFp4MoE(weight_config, input_config)
        elif self._is_fp8_w8a8(weight_config, input_config):
            return QuarkW8A8FP8MoE(weight_config, input_config)
        else:
            raise RuntimeError("Unsupported FusedMoe scheme")

    def get_scaled_act_names(self) -> List[str]:
        return []

    def _is_draft_layer(self, layer: str) -> bool:

        if layer.startswith("mtp."):
            return True
        if self.num_hidden_layers is None:
            return False
        base = self.num_hidden_layers
        return any(
            layer.startswith(f"model.layers.{base + i}.")
            for i in range(self.num_nextn_predict_layers)
        )

    def can_fuse_shared_expert(self) -> bool:

        if any(
            "shared_expert" in layer
            and "shared_expert_gate" not in layer
            and not self._is_draft_layer(layer)
            for layer in self.exclude_layers
        ):
            return False


        layer_quant_config = self.quant_config.get("layer_quant_config") or {}
        if not layer_quant_config:
            return True


        lookup_stub = torch.nn.Module()
        try:
            for base in _MOE_SHARED_EXPERT_QUANT_LAYER0_BASES:
                moe_name = f"{base}.mlp.experts"
                moe_cfg = self._find_matched_config(moe_name, lookup_stub)
                for suffix in _SHARED_EXPERT_BODY_PROJ_SUFFIXES:
                    shared_name = f"{base}.mlp.shared_expert.{suffix}"
                    shared_cfg = self._find_matched_config(shared_name, lookup_stub)
                    if not deep_compare(moe_cfg, shared_cfg):
                        return False
        except ValueError:
            return False

        return True


class QuarkLinearMethod(LinearMethodBase):

    def __init__(self, quantization_config: QuarkConfig):
        self.quantization_config = quantization_config
        self.quant_config = quantization_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.scheme.process_weights_after_loading(layer)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):

        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.scheme.create_weights(
            layer=layer,
            input_size=input_size,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            output_size=output_size,
            params_dtype=params_dtype,
            weight_loader=weight_loader,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):

        scheme = layer.scheme
        if scheme is None:
            raise ValueError("A scheme must be defined for each layer")
        return scheme.apply_weights(layer, x, bias=bias)


class QuarkFusedMoEMethod(FusedMoEMethodBase):

    def __init__(self, quantization_config: QuarkConfig):
        self.quantization_config = quantization_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.scheme.process_weights_after_loading(layer)

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):

        layer.scheme.create_weights(
            layer=layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        layer.scheme.create_moe_runner(layer, moe_runner_config)

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ):

        scheme = layer.scheme
        if scheme is None:
            raise ValueError("A scheme must be defined for each layer")
        return scheme.apply_weights(layer, dispatch_output)


class QuarkKVCacheMethod(BaseKVCacheMethod):


    def __init__(self, quant_config: QuarkConfig):
        self.validate_kv_cache_config(quant_config.kv_cache_config)
        super().__init__(quant_config)

    @staticmethod
    def validate_kv_cache_config(kv_cache_config: Optional[dict[str, Any]]):

        if kv_cache_config is None:
            return

        dtype = kv_cache_config.get("dtype")
        if dtype != "fp8_e4m3":
            raise NotImplementedError(
                "Currently supported kv cache quantization is "
                f"dtype=fp8_e4m3, however received {dtype}"
            )

        qscheme = kv_cache_config.get("qscheme")
        if qscheme != "per_tensor":
            raise NotImplementedError(
                "Only support per-tensor scaling factor "
                "for quark KV cache. "
                f"Expected qscheme: per_tensor, found qscheme: {qscheme}"
            )
