# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from __future__ import annotations

import copy
import dataclasses
import inspect
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from sglang.srt.arg_groups.arg_utils import field_names, resolvable_fields
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.mlx.runtime import use_mlx
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.utils.common import (
    cpu_has_amx_support,
    get_device_capability,
    get_device_name,
    get_device_sm,
    get_nvidia_driver_version,
    get_quantization_config,
    is_blackwell_supported,
    is_cpu,
    is_cuda,
    is_flashinfer_available,
    is_gfx95_supported,
    is_hip,
    is_mnnvl_fabric_device,
    is_mps,
    is_musa,
    is_npu,
    is_sm90_supported,
    is_sm100_supported,
    is_sm120_supported,
    is_triton_kernels_available,
    is_xpu,
    xpu_has_xmx_support,
)

logger = logging.getLogger(__name__)


MODEL_OVERRIDES: Dict[str, Dict[str, Any]] = {


    "MistralLarge3ForCausalLM": {"dtype": "bfloat16"},
    "PixtralForConditionalGeneration": {"dtype": "bfloat16"},
}


_MODEL_OVERRIDE_FNS: Dict[str, List[Callable[..., dict]]] = {}


_PREDICATE_OVERRIDE_FNS: List[Tuple[Callable[[str], bool], Callable[..., dict]]] = []


def register_model_override(architecture: str):


    def decorator(fn: Callable[..., dict]) -> Callable[..., dict]:
        _MODEL_OVERRIDE_FNS.setdefault(architecture, []).append(fn)
        return fn

    return decorator


def register_model_override_predicate(predicate: Callable[[str], bool]):


    def decorator(fn: Callable[..., dict]) -> Callable[..., dict]:
        _PREDICATE_OVERRIDE_FNS.append((predicate, fn))
        return fn

    return decorator


def _invoke_provider(
    fn: Callable[..., dict], server_args: Any, hf_config: Any
) -> Dict[str, Any]:
    declared = fn(server_args, hf_config)
    if not isinstance(declared, dict):
        raise TypeError(
            f"model override provider {fn.__qualname__} must return a dict, "
            f"got {type(declared).__name__}"
        )
    return declared


class ResolvedView:


    __slots__ = ("_server_args", "_overlay")

    def __init__(self, server_args: Any, overlay: Optional[Dict[str, Any]] = None):
        object.__setattr__(self, "_server_args", server_args)
        object.__setattr__(self, "_overlay", overlay or {})

    def __getattr__(self, name: str) -> Any:
        overlay = object.__getattribute__(self, "_overlay")
        if name in overlay:
            return overlay[name]
        return getattr(object.__getattribute__(self, "_server_args"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            "ResolvedView is read-only; post-process passes return declarations"
        )


class ResolvingConfig:


    __slots__ = ("_server_args",)

    def __init__(self, server_args: Any):
        object.__setattr__(self, "_server_args", server_args)

    def __getattr__(self, name: str) -> Any:
        server_args = object.__getattribute__(self, "_server_args")
        for _source, declared in reversed(
            getattr(server_args, "_resolved_overrides", None) or ()
        ):
            if name in declared:
                return declared[name]
        return getattr(server_args, name)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            "ResolvingConfig is read-only; resolution writes through declarations"
        )


def resolving_view(server_args: Any) -> ResolvingConfig:

    return ResolvingConfig(server_args)


POST_PROCESS_PASSES: List[Callable[..., dict]] = []


def register_post_process(fn: Callable[..., dict]) -> Callable[..., dict]:

    POST_PROCESS_PASSES.append(fn)
    return fn


def _declaration_overlay(server_args: Any) -> Dict[str, Any]:

    overlay: Dict[str, Any] = {}
    for _source, declared in getattr(server_args, "_resolved_overrides", None) or ():
        overlay.update(declared)
    return overlay


def run_post_process_pass(server_args: Any, fn: Callable[..., dict]) -> None:

    declared = fn(ResolvedView(server_args, overlay=_declaration_overlay(server_args)))
    if not isinstance(declared, dict):
        raise TypeError(
            f"post-process pass {fn.__qualname__} must return a dict, "
            f"got {type(declared).__name__}"
        )
    if declared:
        entry = (fn.__qualname__, dict(declared))
        stash = getattr(server_args, "_resolved_overrides", None)
        if stash is None:


            stash = server_args._resolved_overrides = []
        stash.append(entry)
        validate_declarations(server_args, [entry])
        if getattr(server_args, "_resolution_finished", False):
            _apply_fields(server_args, declared)


def _apply_fields(server_args: Any, fields: Dict[str, Any]) -> None:

    object.__setattr__(server_args, "_internal_write", True)
    try:
        for field, value in fields.items():
            setattr(server_args, field, value)
    finally:
        object.__setattr__(server_args, "_internal_write", False)


def declare_resolution(server_args: Any, source: str, **fields: Any) -> None:

    if dataclasses.is_dataclass(type(server_args)):
        unknown = sorted(set(fields) - field_names(type(server_args)))
        if unknown:
            raise AttributeError(f"{source}: {unknown} are not ServerArgs fields")
    stash = getattr(server_args, "_resolved_overrides", None)
    if stash is None:
        stash = []
        object.__setattr__(server_args, "_resolved_overrides", stash)
    stash.append((source, dict(fields)))


def declare_late_resolution(server_args: Any, source: str, **fields: Any) -> None:

    from sglang.srt.runtime_context import get_context

    try:
        published = get_context().server_args
    except ValueError:
        published = None
    if published is server_args:
        raise ValueError(
            f"declare_late_resolution({source!r}) called on the published config; "
            "post-publish changes go to the bags via get_context().override(...)"
        )
    log = getattr(server_args, "_runtime_mutations", None)
    if log is None:
        log = []
        object.__setattr__(server_args, "_runtime_mutations", log)
    log.append((source, dict(fields)))
    stash = getattr(server_args, "_resolved_overrides", None)
    if stash is None:
        stash = []
        object.__setattr__(server_args, "_resolved_overrides", stash)
    stash.append((source, dict(fields)))


def declare_direct_writes(
    server_args: Any, source: str, resolve: Callable[[Any], Any]
) -> Any:

    if not dataclasses.is_dataclass(server_args):
        return resolve(server_args)
    before = {
        field.name: getattr(server_args, field.name)
        for field in dataclasses.fields(server_args)
    }
    already = len(getattr(server_args, "_resolved_overrides", None) or ())
    result = resolve(server_args)
    stash = getattr(server_args, "_resolved_overrides", None)
    if stash is None:
        stash = []
        object.__setattr__(server_args, "_resolved_overrides", stash)


    declared = {name for _source, fields in stash[already:] for name in fields}
    changed = {
        name: getattr(server_args, name)
        for name, previous in before.items()
        if name not in declared and getattr(server_args, name) is not previous
    }
    if changed:
        stash.append((source, changed))
    return result


def resolution_result(server_args: Any, field: str, default: Any = None) -> Any:

    for _source, declared in reversed(
        getattr(server_args, "_resolved_overrides", None) or ()
    ):
        if field in declared:
            return declared[field]
    raw = getattr(server_args, "_raw_input", None)
    if raw is not None and field in raw:
        return raw[field]
    return getattr(server_args, field, default)


def resolution_projection(server_args: Any) -> Dict[str, Any]:

    return {
        field.name: _plain(resolution_result(server_args, field.name))
        for field in dataclasses.fields(server_args)
    }


def _plain(value: Any) -> Any:

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _plain(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(_plain(item) for item in value))
    if isinstance(value, (list, tuple)):
        return type(value)(_plain(item) for item in value)
    if isinstance(value, dict):
        return type(value)((_plain(k), _plain(v)) for k, v in value.items())
    return copy.deepcopy(value)


def resolved_view(server_args: Any) -> ResolvedView:

    return ResolvedView(server_args, overlay=_declaration_overlay(server_args))


def attention_backends_of(cfg: Any) -> tuple:

    prefill = (
        cfg.prefill_attention_backend
        if cfg.prefill_attention_backend
        else cfg.attention_backend
    )
    decode = (
        cfg.decode_attention_backend
        if cfg.decode_attention_backend
        else cfg.attention_backend
    )
    return prefill, decode


def modelexpress_transport_of(cfg: Any) -> str:

    raw = cfg.modelexpress_config
    if raw is None:
        parsed = {}
    elif isinstance(raw, str):
        parsed = json.loads(raw)
    else:
        parsed = raw
    return parsed.get("transport", "nixl")


def remote_instance_transfer_engine_of(cfg: Any, load_format: Any = None) -> bool:

    if cfg.remote_instance_weight_loader_start_seed_via_transfer_engine:
        return True
    if (load_format or cfg.load_format) != "remote_instance":
        return False
    backend = cfg.remote_instance_weight_loader_backend
    return backend == "transfer_engine" or (
        backend == "modelexpress"
        and modelexpress_transport_of(cfg) == "transfer_engine"
    )


def mamba_extra_buffer_of(cfg: Any) -> bool:

    return cfg.disable_radix_cache is False and cfg.mamba_radix_cache_strategy in (
        "extra_buffer",
        "extra_buffer_lazy",
    )


def mamba_extra_buffer_lazy_of(cfg: Any) -> bool:

    return (
        cfg.disable_radix_cache is False
        and cfg.mamba_radix_cache_strategy == "extra_buffer_lazy"
    )


def collect_model_override_declarations(
    architecture: str, server_args: Any, hf_config: Any
) -> List[Tuple[str, Dict[str, Any]]]:

    declarations: List[Tuple[str, Dict[str, Any]]] = []
    const = MODEL_OVERRIDES.get(architecture)
    if const:
        declarations.append((f"MODEL_OVERRIDES[{architecture!r}]", dict(const)))
    for fn in _MODEL_OVERRIDE_FNS.get(architecture, ()):
        declared = _invoke_provider(fn, server_args, hf_config)
        if declared:
            declarations.append((fn.__qualname__, dict(declared)))
    for predicate, fn in _PREDICATE_OVERRIDE_FNS:
        if predicate(architecture):
            declared = _invoke_provider(fn, server_args, hf_config)
            if declared:
                declarations.append((fn.__qualname__, dict(declared)))
    return declarations


def _register_for(*architectures: str):


    def decorator(fn: Callable[..., dict]) -> Callable[..., dict]:
        for architecture in architectures:
            register_model_override(architecture)(fn)
        return fn

    return decorator


def _dspark_verify_on_decode_backend(
    backend: Optional[str], q_len: int, kv_cache_dtype: Optional[str]
) -> bool:

    if backend == "trtllm_mla":
        return True
    if backend == "tokenspeed_mla":
        return kv_cache_dtype == "fp8_e4m3" and q_len <= 8
    if backend == "cutedsl_mla":


        return True
    return False


def _require_kimi_k3_cutedsl_dcp_support() -> None:
    try:
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

        parameters = inspect.signature(trtllm_batch_decode_with_kv_cache_mla).parameters
    except (ImportError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Kimi-K3 DCP with decode_attention_backend='cutedsl_mla' requires "
            "FlashInfer 0.6.17 or newer with "
            "trtllm_batch_decode_with_kv_cache_mla exposing enable_dcp."
        ) from exc

    if "enable_dcp" not in parameters:
        raise RuntimeError(
            "Kimi-K3 DCP with decode_attention_backend='cutedsl_mla' requires "
            "enable_dcp in the signature of "
            "flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla; upgrade "
            "to FlashInfer 0.6.17 or newer."
        )


@_register_for("KimiK3ForConditionalGeneration")
def _kimi_k3_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    if cfg.dcp_size > 1:
        overrides = {}
        if cfg.enable_symm_mem:
            logger.warning(
                "Kimi-K3 DCP disables --enable-symm-mem due to decode CUDA "
                "graph correctness issues."
            )
            overrides["enable_symm_mem"] = False

        if cfg.speculative_algorithm == "DSPARK":
            from sglang.srt.speculative.ragged_verify import (
                RaggedVerifyMode,
                read_ragged_verify_mode,
            )

            ragged_mode = read_ragged_verify_mode()
            if ragged_mode is not RaggedVerifyMode.STATIC:
                raise ValueError(
                    "Kimi-K3 DCP + DSPARK currently requires "
                    "SGLANG_RAGGED_VERIFY_MODE=static; compact/cap-accept are "
                    f"not validated under DCP (got {ragged_mode.value!r})."
                )


            overrides["speculative_attention_mode"] = "decode"

        prefill_backend, decode_backend = attention_backends_of(cfg)
        if decode_backend == "cutedsl_mla" or decode_backend is None:
            _require_kimi_k3_cutedsl_dcp_support()
            logger.info(
                "Kimi-K3 DCP keeps decode attention backend 'cutedsl_mla' "
                f"(prefill={prefill_backend!r} -> 'trtllm_mla')."
            )
            overrides.update(
                prefill_attention_backend="trtllm_mla",
                decode_attention_backend="cutedsl_mla",
            )
        elif decode_backend == "tokenspeed_mla":
            logger.info(
                "Kimi-K3 DCP overrides attention backends: "
                f"prefill={prefill_backend!r}, decode={decode_backend!r} -> "
                "'tokenspeed_mla'."
            )
            logger.info(
                "Kimi-K3 DCP with tokenspeed mla backend overrides KV cache dtype: "
                f"{cfg.kv_cache_dtype!r} -> 'fp8_e4m3'."
            )
            overrides.update(
                prefill_attention_backend="tokenspeed_mla",
                decode_attention_backend="tokenspeed_mla",
                kv_cache_dtype="fp8_e4m3",
            )
        else:
            raise AssertionError(
                f"Decode attention backend for Kimi-K3 DCP must be 'cutedsl_mla' or 'tokenspeed_mla', got {decode_backend!r}."
            )

        if cfg.dcp_replicate_q_proj is None:
            logger.info("Kimi-K3 DCP enables replicated Q projection by default.")
            overrides["dcp_replicate_q_proj"] = True

        device_name = get_device_name()
        dcp_comm_backend = "fi_a2a" if is_mnnvl_fabric_device() else "a2a"
        logger.info(
            "Kimi-K3 DCP selects communication backend on "
            f"{device_name!r}: {cfg.dcp_comm_backend!r} -> "
            f"{dcp_comm_backend!r}."
        )
        overrides["dcp_comm_backend"] = dcp_comm_backend
        return overrides

    if not (is_sm100_supported() and get_device_sm() in (100, 103)):
        return {}
    backends_unset = server_args.is_attention_backend_not_set()
    if cfg.speculative_algorithm != "DSPARK":
        if not backends_unset:
            return {}
        logger.info(
            "Use trtllm_mla as the default prefill and decode attention "
            "backend for Kimi-K3 on SM100/SM103."
        )
        return {
            "decode_attention_backend": "trtllm_mla",
            "prefill_attention_backend": "trtllm_mla",
        }


    q_len = cfg.speculative_num_draft_tokens or (
        cfg.speculative_dspark_block_size + 1
        if cfg.speculative_dspark_block_size is not None

        else 8
    )
    overrides = {}
    if backends_unset:
        backend = "trtllm_mla"
        overrides["decode_attention_backend"] = backend
        overrides["prefill_attention_backend"] = "trtllm_mla"
    else:


        _, backend = attention_backends_of(cfg)
    if _dspark_verify_on_decode_backend(backend, q_len, cfg.kv_cache_dtype):
        overrides["speculative_attention_mode"] = "decode"
        logger.info(
            "Kimi-K3 DSPARK on SM100/SM103: decode/verify attention backend "
            f"{backend} (speculative_attention_mode=decode)."
        )
    else:
        logger.warning(
            f"Kimi-K3 DSPARK: decode attention backend {backend!r} cannot serve "
            f"target verify at q_len={q_len}, so verify runs on the prefill "
            "backend (speculative_attention_mode=prefill). A host-plan prefill "
            "backend costs a per-step seq_lens D2H sync; leave the attention "
            "backend knobs unset for the sync-free default."
        )
    return overrides


def _is_mxfp4_pack_quantized(hf_config: Any) -> bool:
    qc = getattr(
        getattr(hf_config, "text_config", hf_config), "quantization_config", None
    )
    if not isinstance(qc, dict):
        return False
    groups = qc.get("config_groups") or {}
    return any(
        "mxfp4" in str(g.get("format", ""))
        for g in groups.values()
        if isinstance(g, dict)
    )


@_register_for("KimiK3ForConditionalGeneration")
def _kimi_k3_moe_runner_overrides(server_args: Any, hf_config: Any) -> dict:


    cfg = resolving_view(server_args)
    if cfg.moe_runner_backend != "auto":
        return {}
    if not (is_sm100_supported() and get_device_sm() in (100, 103, 107)):
        return {}
    if not _is_mxfp4_pack_quantized(hf_config):
        return {}
    logger.info(
        "Kimi-K3 on SM100/SM103/SM107: moe_runner_backend=flashinfer_mxfp4 "
        "(FlashInfer SiTU kernels)."
    )
    return {"moe_runner_backend": "flashinfer_mxfp4"}


@_register_for(
    "DeepseekV3ForCausalLM",
    "DeepseekV32ForCausalLM",
    "KimiK25ForConditionalGeneration",
    "MistralLarge3ForCausalLM",
    "PixtralForConditionalGeneration",
    "GlmMoeDsaForCausalLM",
    "LongcatFlashForCausalLM",
    "LongcatFlashForCausalLMNextN",
    "Dots3NoteForCausalLM",
)
def _deepseek_family_overrides(server_args: Any, hf_config: Any) -> dict:

    cfg = resolving_view(server_args)
    from sglang.srt.configs.model_config import is_deepseek_dsa

    overrides: Dict[str, Any] = {}

    if is_deepseek_dsa(hf_config):

        if server_args.is_attention_backend_not_set():
            overrides["attention_backend"] = "dsa"
            logger.info("Use dsa attention backend for DeepSeek with DSA.")
        if not is_npu() and not is_xpu():
            if cfg.enable_prefill_cp:
                logger.warning(
                    "Context parallel feature is still under experiment. It has only been verified on Hopper platform."
                )
                overrides["enable_dp_attention"] = True
                overrides["moe_dense_tp_size"] = 1
                if cfg.cp_strategy == "zigzag":
                    overrides["moe_a2a_backend"] = "deepep"
                    overrides["ep_size"] = cfg.tp_size
                    logger.warning(
                        "zigzag DSA CP requires moe_dense_tp_size=1, "
                        "moe_a2a_backend=deepep, ep_size=tp_size, batch_size=1."
                    )
                else:
                    assert (
                        cfg.dp_size == 1
                    ), "interleave DSA CP does not support DP attention."
                assert (
                    cfg.tp_size <= 8
                ), "Context parallel only supports single machine (tp_size <= 8). Cross-machine CP has precision issues."


                attn_cp_size = cfg.tp_size // cfg.dp_size
                overrides["attn_cp_size"] = attn_cp_size
                logger.warning(
                    "Enabled DSA context parallel: "
                    f"strategy={cfg.cp_strategy}, dp_size={cfg.dp_size}, "
                    f"moe_dense_tp_size={overrides['moe_dense_tp_size']}, "
                    f"ep_size={overrides.get('ep_size', cfg.ep_size)}, tp_size={cfg.tp_size}, "
                    f"attn_cp_size={attn_cp_size}, "
                    f"kv_cache_dtype={cfg.kv_cache_dtype}, "
                    f"moe_a2a_backend={overrides.get('moe_a2a_backend', cfg.moe_a2a_backend)}, "
                    f"cuda_graph_config[prefill].backend=disabled"
                )


            from sglang.srt.layers.attention.dsa.utils import (
                aiter_can_use_preshuffle_paged_mqa,
            )

            if is_hip() and not aiter_can_use_preshuffle_paged_mqa():


                overrides["page_size"] = 1
                logger.warning(
                    "Setting page size to 1 for DeepSeek DSA on ROCm "
                    "(aiter preshuffle paged-MQA path unavailable: "
                    "needs Triton>=3.5.0 or AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1)."
                )
            else:
                overrides["page_size"] = 64
                logger.warning("Setting page size to 64 for DeepSeek DSA.")
    else:

        if is_sm100_supported():
            if (
                cfg.attention_backend is None
                and cfg.prefill_attention_backend is None
                and cfg.decode_attention_backend is None
            ):
                overrides["attention_backend"] = "trtllm_mla"
                logger.info(
                    "Use trtllm_mla as attention backend on sm100 for DeepseekV3ForCausalLM"
                )


        if cfg.enable_prefill_cp and server_args.use_mla_backend():
            logger.warning(
                "MLA prefill context parallel is still experimental. "
                "Verified on Hopper with the fa3 backend."
            )
            overrides["enable_dp_attention"] = True

            overrides["moe_dense_tp_size"] = 1
            overrides["moe_a2a_backend"] = "deepep"
            overrides["ep_size"] = cfg.tp_size
            logger.warning(
                "For MLA CP, we have the following restrictions: moe_dense_tp_size == 1, moe_a2a_backend == deepep, ep_size == tp_size, batch_size == 1"
            )


            attn_cp_size = cfg.tp_size // cfg.dp_size
            overrides["attn_cp_size"] = attn_cp_size
            logger.warning(
                f"Enable Context Parallel opt for MLA, "
                f"Setting dp_size == {cfg.dp_size} and "
                f"attn_cp_size == {attn_cp_size}, "
                f"moe_dense_tp_size == {overrides['moe_dense_tp_size']}, "
                f"ep_size == {overrides['ep_size']}, "
                f"tp_size == {cfg.tp_size}, "
                f"moe_a2a_backend {overrides['moe_a2a_backend']}, "
                f"cuda_graph_config[prefill].backend=disabled"
            )
    return overrides


@_register_for("MiMoV2ForCausalLM", "MiMoV2FlashForCausalLM")
def _mimo_v2_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    overrides: Dict[str, Any] = {}
    if cfg.speculative_algorithm == "EAGLE":
        logger.info("Enable multi-layer EAGLE speculative decoding for MiMoV2 model.")
        overrides["enable_multi_layer_eagle"] = True


    if (
        is_sm100_supported()
        and cfg.moe_runner_backend == "auto"
        and get_quantization_config(hf_config) == "fp8"
    ):
        overrides["moe_runner_backend"] = "flashinfer_trtllm"
        logger.info("MiMoV2 FP8 on SM100: moe_runner_backend=flashinfer_trtllm.")
    return overrides


@_register_for("MiniMaxM2ForCausalLM")
def _minimax_m2_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    overrides = {"enable_tf32_matmul": True}
    logger.info(
        "Enable TF32 matmul for MiniMaxM2ForCausalLM model to improve gate gemm performance."
    )
    if (
        is_sm100_supported()
        and cfg.moe_runner_backend == "auto"
        and server_args.get_model_config().quantization == "modelopt_fp4"
    ):
        overrides["moe_runner_backend"] = "flashinfer_trtllm_routed"
        logger.info(
            "Use flashinfer_trtllm_routed as MoE runner backend on SM10X "
            "for MiniMaxM2ForCausalLM with modelopt_fp4."
        )
    return overrides


@_register_for("MiniMaxM3SparseForCausalLM", "MiniMaxM3SparseForConditionalGeneration")
def _minimax_m3_overrides(server_args: Any, hf_config: Any) -> dict:

    cfg = resolving_view(server_args)
    overrides: Dict[str, Any] = {}

    quant_method = get_quantization_config(hf_config)
    quant_resolved = cfg.quantization
    if (
        quant_resolved is None
        and not server_args._quantization_explicitly_unset
        and quant_method is not None
    ):
        overrides["quantization"] = quant_method
        quant_resolved = quant_method

    if is_hip():
        if server_args.is_attention_backend_not_set():
            overrides["attention_backend"] = "triton"
        if cfg.moe_runner_backend == "auto" and quant_resolved == "mxfp8":
            overrides["moe_runner_backend"] = "triton"
        if not envs.USE_ROCM_AITER_ROPE_BACKEND.is_set():
            envs.USE_ROCM_AITER_ROPE_BACKEND.set("0")
        aiter_fusion_resolved = cfg.enable_aiter_allreduce_fusion
        if cfg.ep_size > 1 and cfg.moe_a2a_backend == "none" and aiter_fusion_resolved:
            logger.warning(
                "Disable --enable-aiter-allreduce-fusion for MiniMax-M3 "
                "standard EP on ROCm because the deferred fused all-reduce "
                "corrupts sparse MoE partial outputs."
            )
            overrides["enable_aiter_allreduce_fusion"] = False
            aiter_fusion_resolved = False


        if not aiter_fusion_resolved and not envs.SGLANG_M3_ALLOW_CUSTOM_AR.get():
            overrides["disable_custom_all_reduce"] = True
    elif is_sm100_supported():
        if server_args.is_attention_backend_not_set():
            if (
                cfg.kv_cache_dtype == "fp8_e4m3"
                and not envs.SGLANG_DISABLE_M3_FP8_ATTN_GEMM.get()
            ):


                overrides["attention_backend"] = "trtllm_mha"
            else:
                overrides["attention_backend"] = "fa4"
        backend_resolved = overrides.get("attention_backend", cfg.attention_backend)
        page_resolved = cfg.page_size


        if page_resolved is None and backend_resolved in ("fa4", "trtllm_mha"):
            overrides["page_size"] = 128
            page_resolved = 128
        if cfg.moe_runner_backend == "auto" and quant_resolved == "mxfp8":
            overrides["moe_runner_backend"] = "deep_gemm"
        elif cfg.moe_runner_backend == "auto" and quant_resolved == "modelopt_mixed":
            overrides["moe_runner_backend"] = "flashinfer_trtllm_routed"
        logger.info(
            "MiniMax-M3 on SM100: attention_backend="
            f"{overrides.get('attention_backend', cfg.attention_backend)}, page_size={page_resolved}, "
            f"moe_runner_backend={overrides.get('moe_runner_backend', cfg.moe_runner_backend)}."
        )
    elif is_sm90_supported():
        if server_args.is_attention_backend_not_set():
            overrides["attention_backend"] = "fa3"
        page_resolved = cfg.page_size
        if (
            page_resolved is None
            and overrides.get("attention_backend", cfg.attention_backend) == "fa3"
        ):
            overrides["page_size"] = 128
            page_resolved = 128
        logger.info(
            "MiniMax-M3 on Hopper: attention_backend="
            f"{overrides.get('attention_backend', cfg.attention_backend)}, page_size={page_resolved} "
            "(MSA is SM100-only; sparse attention runs on the Triton path)."
        )


    if cfg.kv_cache_dtype == "fp8_e5m2":
        logger.warning(
            "MiniMax-M3 with kv_cache_dtype fp8_e5m2: fp8 attention GEMMs stay "
            "DISABLED (fmha_sm100's variant lookup would silently dispatch the "
            "e4m3 kernel for e5m2); sparse attention runs on the widening "
            "Triton path. Use --kv-cache-dtype fp8_e4m3 for fp8 attention GEMMs."
        )
    elif (
        cfg.kv_cache_dtype == "fp8_e4m3"
        and overrides.get("attention_backend", cfg.attention_backend) == "trtllm_mha"
        and is_sm100_supported()
    ):
        if envs.SGLANG_DISABLE_M3_FP8_ATTN_GEMM.get():
            logger.info(
                "MiniMax-M3 fp8 attention GEMMs DISABLED by "
                "SGLANG_DISABLE_M3_FP8_ATTN_GEMM: bf16 indexer + widening "
                "Triton sparse path, bf16 q; dense layers keep trtllm_mha's "
                "fp8 KV cache."
            )
        else:
            logger.info(
                "MiniMax-M3 fp8 attention GEMMs active (kv_cache_dtype fp8_e4m3 + "
                "trtllm_mha on SM100): fp8 main/index KV, fp8-cast q, fp8 "
                "sparse/MSA kernels. Set SGLANG_DISABLE_M3_FP8_ATTN_GEMM=1 to "
                "force the pre-fp8 numerics."
            )

    moe_runner_resolved = overrides.get("moe_runner_backend", cfg.moe_runner_backend)
    if quant_resolved is None and moe_runner_resolved in ("auto", "deep_gemm"):
        if moe_runner_resolved == "deep_gemm":
            logger.warning(
                "MiniMax-M3: the deep_gemm MoE runner produces corrupted output "
                "on bf16 full weights; overriding --moe-runner-backend to 'triton'."
            )
        overrides["moe_runner_backend"] = "triton"

    return overrides


@_register_for(
    "Gemma2ForCausalLM",
    "Gemma3ForCausalLM",
    "Gemma3ForConditionalGeneration",
    "Gemma3nForCausalLM",
    "Gemma3nForConditionalGeneration",
)
def _gemma2_gemma3_overrides(server_args: Any, hf_config: Any) -> dict:


    logger.warning(
        f"Disable hybrid SWA memory for {hf_config.architectures[0]} as it is not yet supported."
    )
    return {"disable_hybrid_swa_memory": True}


@_register_for("Exaone4ForCausalLM", "ExaoneMoEForCausalLM")
def _exaone_overrides(server_args: Any, hf_config: Any) -> dict:
    if hf_config.sliding_window_pattern is not None:
        logger.warning(
            f"Disabling hybrid SWA memory for {hf_config.architectures[0]} as it is not yet supported."
        )
        return {"disable_hybrid_swa_memory": True}
    return {}


@_register_for("GptOssForCausalLM")
def _gpt_oss_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    overrides: Dict[str, Any] = {}

    if server_args.is_attention_backend_not_set():
        if is_sm100_supported():
            overrides["attention_backend"] = "trtllm_mha"
        elif is_sm90_supported():
            overrides["attention_backend"] = "fa3"
        elif is_cpu() and cpu_has_amx_support():
            overrides["attention_backend"] = "intel_amx"
        elif is_xpu():
            overrides["attention_backend"] = "intel_xpu"
        elif is_hip():
            overrides["attention_backend"] = "aiter"
        elif not (is_mps() and use_mlx()):


            overrides["attention_backend"] = "triton"
    if is_xpu():


        if cfg.dtype == "auto":
            logger.warning(
                "GptOssForCausalLM on Intel XPU currently supports bfloat16 dtype only"
            )
        elif cfg.dtype not in ["bfloat16"]:
            raise NotImplementedError(
                f"GptOssForCausalLM on Intel XPU only supports bfloat16 dtype, "
                f"but got '{cfg.dtype}'. Please use --dtype bfloat16 or remove --dtype to use auto."
            )
    quantization_config = getattr(hf_config, "quantization_config", None)
    is_mxfp4_quant_format = (
        quantization_config is not None
        and quantization_config.get("quant_method") == "mxfp4"
    )
    if is_mxfp4_quant_format:

        overrides["dtype"] = "bfloat16"
    if cfg.moe_runner_backend == "auto":

        if is_sm100_supported() and is_mxfp4_quant_format:
            overrides["moe_runner_backend"] = "flashinfer_mxfp4"
            logger.warning(
                "Detected SM100 and MXFP4 quantization format for GPT-OSS model, enabling FlashInfer MXFP4 MOE kernel."
            )
        elif is_sm120_supported() and is_mxfp4_quant_format:
            overrides["moe_runner_backend"] = "flashinfer_mxfp4"
            logger.warning(
                "Detected SM120 and MXFP4 quantization format for GPT-OSS model, "
                "enabling FlashInfer CUTLASS MXFP4 MOE kernel."
            )
        elif (is_hip() and envs.SGLANG_USE_AITER.get()) and is_mxfp4_quant_format:
            overrides["moe_runner_backend"] = "auto"
            logger.warning(
                "Detected ROCm and MXFP4 quantization format for GPT-OSS model, enabling aiter MXFP4 MOE kernel."
            )


        elif is_hip() and envs.SGLANG_USE_AITER.get():


            overrides["moe_runner_backend"] = "triton"
            logger.warning(
                "Detected ROCm with SGLANG_USE_AITER for GPT-OSS bf16 model, using triton MOE kernel."
            )
        elif is_musa() and envs.SGLANG_DEEPEP_BF16_DISPATCH.get():
            overrides["moe_runner_backend"] = "deep_gemm"
            logger.warning(
                "Detected MUSA with SGLANG_DEEPEP_BF16_DISPATCH for bf16 model, using deep_gemm kernel."
            )
        elif (
            cfg.ep_size == 1
            and is_triton_kernels_available()
            and cfg.quantization is None
            and not (is_cpu() and cpu_has_amx_support())
        ):


            if is_blackwell_supported() and get_nvidia_driver_version() >= (595,):
                overrides["moe_runner_backend"] = "triton"
                logger.warning(
                    "Detected GPT-OSS model on Blackwell with driver >= 595, "
                    "using triton MOE kernel to avoid triton_kernels SIGSEGV."
                )
            else:
                overrides["moe_runner_backend"] = "triton_kernel"
                logger.warning(
                    "Detected GPT-OSS model, enabling triton_kernels MOE kernel."
                )
    return overrides


@_register_for("Llama4ForConditionalGeneration", "Llama4ForCausalLM")
def _llama4_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    if cfg.device == "cpu":
        return {}
    overrides: Dict[str, Any] = {}

    if cfg.attention_backend is None:
        if is_sm100_supported():
            backend, platform = "trtllm_mha", "sm100"
        elif is_sm90_supported():
            backend, platform = "fa3", "sm90"
        elif is_hip():
            backend, platform = "aiter", "hip"
        elif cfg.device == "xpu":
            backend, platform = "intel_xpu", "xpu"
        else:
            backend, platform = "triton", "other platforms"
        logger.warning(
            f"Use {backend} as attention backend on {platform} for Llama4 model"
        )
        overrides["attention_backend"] = backend
    if is_sm100_supported() and cfg.moe_runner_backend == "auto":
        if cfg.quantization in {"fp8", "modelopt_fp8"}:
            overrides["moe_runner_backend"] = "flashinfer_trtllm"
            logger.info(
                "Use flashinfer_trtllm as MoE runner backend on SM100 for Llama4"
            )
    return overrides


@_register_for(
    "Gemma4ForConditionalGeneration",
    "Gemma4ForCausalLM",
    "Gemma4UnifiedForConditionalGeneration",
)
def _gemma4_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    overrides: Dict[str, Any] = {}
    default_attention_backend = "trtllm_mha" if is_sm100_supported() else "triton"
    if server_args.is_attention_backend_not_set():
        logger.info(
            f"Use {default_attention_backend} as default attention backend for Gemma4"
        )
        overrides["attention_backend"] = default_attention_backend


    elif cfg.attention_backend is None:
        overrides["attention_backend"] = default_attention_backend
    if is_sm100_supported() and cfg.moe_runner_backend == "auto":
        if server_args.get_model_config().quantization == "modelopt_fp4":
            overrides["quantization"] = "modelopt_fp4"
            overrides["moe_runner_backend"] = "flashinfer_trtllm"
            logger.info(
                "Use flashinfer_trtllm as MoE runner backend on "
                "SM100 for Gemma-4 (modelopt_fp4)"
            )
    return overrides


@_register_for("MossVLForConditionalGeneration")
def _moss_vl_overrides(server_args: Any, hf_config: Any) -> dict:
    overrides: Dict[str, Any] = {}
    if server_args.is_attention_backend_not_set():
        overrides["prefill_attention_backend"] = "flashinfer"
        logger.info("Use flashinfer as default prefill attention backend for Moss-VL")
    prefill_backend = (
        overrides.get("prefill_attention_backend")
        or server_args.get_attention_backends()[0]
    )
    assert prefill_backend == "flashinfer", (
        "MossVLForConditionalGeneration requires flashinfer prefill "
        "attention backend for cross-attention custom mask support."
    )
    return overrides


@_register_for("MiniCPMForCausalLM", "MiniCPMSALAForCausalLM")
def _minicpm_sala_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    if cfg.enable_dp_attention:
        raise ValueError("MiniCPM does not support DP attention")
    has_sparse_attention = getattr(hf_config, "has_minicpm_sparse_attention", False)
    has_hybrid_attention = has_sparse_attention or getattr(
        hf_config, "has_lightning_layers", False
    )
    overrides: Dict[str, Any] = {}
    if has_hybrid_attention:
        if cfg.enable_hierarchical_cache:
            raise ValueError("MiniCPM SALA does not support hierarchical cache")
        overrides["disable_radix_cache"] = True
    if envs.SGLANG_MINICPM_FORCE_DENSE.get():
        dense_backends = {
            "minicpm_flashattn": ("fa4" if is_blackwell_supported() else "fa3"),
            "minicpm_flashinfer": "flashinfer",
        }


        dense_attention = dense_backends.get(cfg.attention_backend)
        if dense_attention is not None:
            overrides["attention_backend"] = dense_attention
        dense_prefill = dense_backends.get(cfg.prefill_attention_backend)
        if dense_prefill is not None:
            overrides["prefill_attention_backend"] = dense_prefill
        dense_decode = dense_backends.get(cfg.decode_attention_backend)
        if dense_decode is not None:
            overrides["decode_attention_backend"] = dense_decode
    elif has_sparse_attention:
        uses_sparse_backend = cfg.is_attention_backend_not_set() or any(
            backend in ("minicpm_flashattn", "minicpm_flashinfer")
            for backend in (
                cfg.attention_backend,
                cfg.prefill_attention_backend,
                cfg.decode_attention_backend,
            )
        )
        if uses_sparse_backend and cfg.disaggregation_mode != "null":
            raise ValueError(
                "MiniCPM sparse attention does not support PD disaggregation"
            )
        if cfg.is_attention_backend_not_set():
            overrides["attention_backend"] = (
                "minicpm_flashinfer"
                if is_blackwell_supported()
                else "minicpm_flashattn"
            )
    return overrides


@_register_for("MiniCPMV4_6ForConditionalGeneration")
def _minicpm_v4_6_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    if is_sm100_supported() and cfg.attention_backend is None:
        return {"attention_backend": "triton"}
    return {}


@_register_for(
    "FalconH1ForCausalLM", "JetNemotronForCausalLM", "JetVLMForConditionalGeneration"
)
def _falcon_h1_jet_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    if is_sm100_supported() and cfg.attention_backend is None:
        return {"attention_backend": "triton"}
    return {}


@_register_for("GraniteMoeHybridForCausalLM")
def _granite_moe_hybrid_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    has_mamba = any(
        layer_type == "mamba" for layer_type in getattr(hf_config, "layer_types", [])
    )
    if has_mamba and is_sm100_supported() and cfg.attention_backend is None:
        return {"attention_backend": "flashinfer"}
    return {}


@_register_for("Lfm2ForCausalLM", "Lfm2MoeForCausalLM")
def _lfm2_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    if is_sm100_supported() and cfg.attention_backend is None:
        return {"attention_backend": "flashinfer"}
    return {}


@_register_for("DeepseekV4ForCausalLM")
def _deepseek_v4_overrides(server_args: Any, hf_config: Any) -> dict:

    cfg = resolving_view(server_args)
    from sglang.srt.server_args import ServerArgs

    model_arch = hf_config.architectures[0]
    overrides: Dict[str, Any] = {"attention_backend": "dsv4"}

    page_size = 256
    if cfg.device == "npu":


        page_size = 128
        overrides["prefill_attention_backend"] = "dsv4"
        overrides["decode_attention_backend"] = "dsv4"
    overrides["page_size"] = page_size
    logger.info(
        f"Use dsv4 attention backend for {model_arch}, setting page_size to {page_size}."
    )

    if cfg.swa_full_tokens_ratio == ServerArgs.swa_full_tokens_ratio:
        overrides["swa_full_tokens_ratio"] = 0.1
        logger.info(f"Setting swa_full_tokens_ratio to 0.1 for {model_arch}.")

    if cfg.moe_runner_backend == "auto":
        model_config = server_args.get_model_config()

        if model_config.nvfp4_moe_meta is not None:
            overrides["moe_runner_backend"] = "flashinfer_trtllm_routed"
            logger.info(
                "Use flashinfer_trtllm_routed as MoE runner backend for "
                f"{model_arch} hybrid FP8+NVFP4 checkpoint."
            )
        elif (
            cfg.device == "cuda"
            and not is_hip()
            and cfg.moe_a2a_backend == "none"
            and not envs.SGLANG_DSV4_FP4_DEQUANT.get()
            and model_config.is_fp4_experts
            and (is_sm90_supported() or is_sm100_supported() or is_sm120_supported())
        ):
            overrides["moe_runner_backend"] = "flashinfer_mxfp4"
            logger.info(
                "Use flashinfer_mxfp4 as MoE runner backend for " f"{model_arch}."
            )
    return overrides


@_register_for(
    "InklingForConditionalGeneration",
    "InklingForConditionalGenerationMTP",
)
def _inkling_overrides(server_args: Any, hf_config: Any) -> dict:

    cfg = resolving_view(server_args)
    from sglang.srt.server_args import ServerArgs

    overrides: Dict[str, Any] = {}


    if cfg.swa_full_tokens_ratio == ServerArgs.swa_full_tokens_ratio:
        overrides["swa_full_tokens_ratio"] = 0.1
    if cfg.mamba_full_memory_ratio == ServerArgs.mamba_full_memory_ratio:
        overrides["mamba_full_memory_ratio"] = 0.1


    if cfg.mamba_radix_cache_strategy == ServerArgs.mamba_radix_cache_strategy:
        overrides["mamba_radix_cache_strategy"] = "extra_buffer"


    if server_args.is_attention_backend_not_set():
        inkling_attn_backend = "fa4" if is_sm100_supported() else "triton"
        overrides["attention_backend"] = inkling_attn_backend
        logger.info(
            f"Use {inkling_attn_backend} as the attention backend for Inkling "
            "(requires fa4 or triton)."
        )
    envs.SGLANG_ENABLE_UNIFIED_RADIX_TREE.set(True)
    return overrides


@_register_for("NemotronHForCausalLM", "NemotronHPuzzleForCausalLM")
def _nemotron_h_overrides(server_args: Any, hf_config: Any) -> dict:

    cfg = resolving_view(server_args)
    model_arch = hf_config.architectures[0]
    model_config = server_args.get_model_config()
    overrides: Dict[str, Any] = {}

    is_modelopt = model_config.quantization in [
        "modelopt",
        "modelopt_fp8",
        "modelopt_fp4",
        "modelopt_mixed",
    ]
    quantization = cfg.quantization
    if is_modelopt:
        assert model_config.hf_config.mlp_hidden_act == "relu2"
        if model_config.quantization == "modelopt":
            quant_algo = model_config.hf_config.quantization_config["quant_algo"]
            if quant_algo == "MIXED_PRECISION":
                quantization = "modelopt_mixed"
            else:
                quantization = (
                    "modelopt_fp4" if quant_algo == "NVFP4" else "modelopt_fp8"
                )
        else:
            quantization = model_config.quantization
        overrides["quantization"] = quantization

    has_w4a16_moe_layers = False
    if is_modelopt and quantization == "modelopt_mixed":
        has_w4a16_moe_layers = any(
            info.get("quant_algo") == "W4A16_NVFP4" and ".experts." in name
            for name, info in hf_config.quantization_config.get(
                "quantized_layers", {}
            ).items()
        )

    if has_w4a16_moe_layers:
        if cfg.moe_a2a_backend != "none":
            raise ValueError("W4A16_NVFP4 MoE layers require --moe-a2a-backend=none.")
        if cfg.moe_runner_backend not in ("auto", "marlin"):
            raise ValueError(
                "W4A16_NVFP4 MoE layers require --moe-runner-backend=marlin."
            )
        if cfg.moe_runner_backend == "auto":
            overrides["moe_runner_backend"] = "marlin"
            logger.info(
                "Use marlin as MoE runner backend for "
                f"{model_arch} with W4A16_NVFP4 MoE layers"
            )
    elif (is_modelopt or model_config.quantization is None) and (
        cfg.moe_runner_backend == "auto"
    ):
        if is_sm100_supported() and cfg.moe_a2a_backend == "none":
            overrides["moe_runner_backend"] = "flashinfer_trtllm"
            logger.info(
                f"Use flashinfer_trtllm as MoE runner backend on sm100 for {model_arch}"
            )
        elif (
            (
                model_config.quantization in ("modelopt_fp4", "modelopt_mixed")
                or quantization == "modelopt_fp4"
            )
            and is_cuda()
            and (8, 0) <= get_device_capability() < (10, 0)
        ):
            overrides["moe_runner_backend"] = "marlin"
            logger.info(
                "Use marlin as MoE runner backend on SM80-SM90 for "
                f"{model_arch} {model_config.quantization}"
            )
        else:
            overrides["moe_runner_backend"] = "flashinfer_cutlass"

    if is_blackwell_supported() and cfg.is_attention_backend_not_set():
        if cfg.speculative_algorithm is not None:
            speculative_algorithm = cfg.speculative_algorithm.upper()
            if is_sm100_supported() and cfg.speculative_eagle_topk in (
                None,
                1,
            ):
                overrides["attention_backend"] = "trtllm_mha"
                if cfg.page_size is None:
                    overrides["page_size"] = 64
                if cfg.mamba_radix_cache_strategy == "auto":
                    overrides["mamba_radix_cache_strategy"] = "extra_buffer"
                if (
                    cfg.speculative_draft_attention_backend is None
                    and speculative_algorithm in ("EAGLE", "NEXTN", "DSPARK")
                ):
                    overrides["speculative_draft_attention_backend"] = "trtllm_mha"
            else:
                overrides["attention_backend"] = "triton"
                if (
                    cfg.speculative_draft_attention_backend is None
                    and speculative_algorithm in ("EAGLE", "NEXTN", "DFLASH", "DSPARK")
                ):
                    overrides["speculative_draft_attention_backend"] = "flashinfer"
        elif is_sm100_supported():
            overrides["attention_backend"] = "trtllm_mha"
    return overrides


@_register_for(
    "Qwen3NextForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
    "InternS2PreviewForConditionalGeneration",
    "InternS2MobiusForConditionalGeneration",
    "Qwen3_5ForConditionalGeneration",
)
def _qwen3_5_hybrid_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    if not is_sm100_supported() or cfg.attention_backend is not None:
        return {}
    sm100_default_attn_backend = "triton"


    default_attn_backend = server_args._get_default_attn_backend(
        use_mla_backend=server_args.use_mla_backend(),
        model_config=server_args.get_model_config(),
    )


    if default_attn_backend == "trtllm_mha" and not (
        not mamba_extra_buffer_of(resolved_view(server_args))
        and not cfg.disable_radix_cache
        and cfg.speculative_algorithm is None
    ):
        sm100_default_attn_backend = "trtllm_mha"
    return {
        "attention_backend": sm100_default_attn_backend,
        "page_size": 64 if sm100_default_attn_backend == "trtllm_mha" else 1,
    }


@_register_for("InternS2MobiusForConditionalGeneration")
def _interns2_mobius_baseline_overrides(server_args: Any, hf_config: Any) -> dict:

    cfg = resolving_view(server_args)
    if cfg.moe_runner_backend == "auto":
        return {"moe_runner_backend": "triton_kernel"}
    return {}


@_register_for("Qwen3VLForConditionalGeneration")
def _qwen3vl_overrides(server_args: Any, hf_config: Any) -> dict:

    cfg = resolving_view(server_args)
    if is_hip() and envs.SGLANG_USE_AITER_UNIFIED_ATTN.get() and cfg.page_size is None:
        logger.info(
            "Setting page_size=16 for aiter unified attention on Qwen3VLForConditionalGeneration."
        )
        return {"page_size": 16}
    return {}


@_register_for(
    "Qwen3MoeForCausalLM",
    "Qwen3VLMoeForConditionalGeneration",
    "Qwen3NextForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
    "InternS2PreviewForConditionalGeneration",
    "Qwen3_5ForConditionalGeneration",
)
def _qwen3_moe_family_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    overrides: Dict[str, Any] = {}
    if is_sm100_supported():
        quant_method = get_quantization_config(hf_config)
        quantization = cfg.quantization
        if (
            quantization is None
            and not server_args._quantization_explicitly_unset
            and quant_method is not None
        ):
            overrides["quantization"] = quant_method
            quantization = quant_method
        if (
            (quantization in ("fp8", "modelopt_fp4") or quantization is None)
            and cfg.moe_a2a_backend == "none"
            and cfg.moe_runner_backend == "auto"
        ):
            overrides["moe_runner_backend"] = "flashinfer_trtllm"
            logger.info(
                "Use flashinfer_trtllm as MoE runner backend on sm100 for "
                f"{hf_config.architectures[0]}"
            )
    return overrides


@_register_for("Glm4MoeForCausalLM")
def _glm4_moe_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    overrides: Dict[str, Any] = {}
    if is_sm100_supported():
        quantization_config = getattr(hf_config, "quantization_config", None)
        quant_method = (
            quantization_config.get("quant_method")
            if quantization_config is not None
            else None
        )
        quantization = cfg.quantization
        if (
            quantization is None
            and not server_args._quantization_explicitly_unset
            and quant_method is not None
        ):
            overrides["quantization"] = quant_method
            quantization = quant_method
        if (
            quantization in {"modelopt_fp4", None}
            and cfg.moe_a2a_backend == "none"
            and cfg.moe_runner_backend == "auto"
        ):
            overrides["moe_runner_backend"] = "flashinfer_trtllm"
            logger.info(
                "Use flashinfer_trtllm as MoE runner backend on sm100 for Glm4MoeForCausalLM"
            )
    logger.info(
        "Enable TF32 matmul for Glm4MoeForCausalLM model to improve gate gemm performance."
    )
    overrides["enable_tf32_matmul"] = True
    return overrides


@_register_for("Olmo2ForCausalLM")
def _olmo2_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    overrides: Dict[str, Any] = {}

    logger.warning(
        f"Disabling hybrid SWA memory for {hf_config.architectures[0]} as it is not yet supported."
    )
    overrides["disable_hybrid_swa_memory"] = True
    if cfg.attention_backend is None:
        if is_cuda() and is_sm100_supported():
            overrides["attention_backend"] = "trtllm_mha"
        elif is_cuda() and get_device_sm() >= 80:
            overrides["attention_backend"] = "fa3"
        else:
            overrides["attention_backend"] = "triton"
    return overrides


@register_model_override_predicate(
    lambda arch: "Step3p5ForCausalLM" in arch
    or "Step3p7ForConditionalGeneration" in arch
)
def _step3p_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    overrides: Dict[str, Any] = {}
    if server_args.is_attention_backend_not_set():
        if is_blackwell_supported():
            logger.info("Auto-select fa4 attention backend for Step3p7 on Blackwell.")
            overrides["attention_backend"] = "fa4"
        elif is_sm90_supported():
            logger.info("Auto-select fa3 attention backend for Step3p7 on Hopper.")
            overrides["attention_backend"] = "fa3"
    if cfg.speculative_algorithm == "EAGLE":
        logger.info(
            "Enable multi-layer EAGLE speculative decoding for Step3p5ForCausalLM model."
        )
        overrides["enable_multi_layer_eagle"] = True
    if cfg.enable_hierarchical_cache:
        logger.warning(
            "Reset swa_full_tokens_ratio to 1.0 for Step3p5ForCausalLM model with hierarchical cache"
        )
        overrides["swa_full_tokens_ratio"] = 1.0
        logger.warning(
            "Disable hybrid SWA memory for Step3p5ForCausalLM model with hierarchical cache"
        )
        overrides["disable_hybrid_swa_memory"] = True
    return overrides


_MAMBA_RADIX_CACHE_ARCHS = frozenset(
    {
        "KimiLinearForCausalLM",
        "KimiK3ForConditionalGeneration",
        "BailingMoeV2_5ForCausalLM",
        "Qwen3NextForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
        "InternS2PreviewForConditionalGeneration",
        "InternS2MobiusForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration",


        "Qwen3_5MoeForCausalLM",
        "Qwen3_5ForCausalLM",
        "MiniCPMV4_6ForConditionalGeneration",
        "NemotronHForCausalLM",
        "NemotronHPuzzleForCausalLM",
        "FalconH1ForCausalLM",
        "JetNemotronForCausalLM",
        "JetVLMForConditionalGeneration",
        "Lfm2ForCausalLM",
        "Lfm2MoeForCausalLM",
        "ZayaForCausalLM",
    }
)


_MAMBA_EXTRA_BUFFER_ARCHS = frozenset(
    {
        "KimiLinearForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",


        "Qwen3_5MoeForCausalLM",
        "Qwen3_5ForCausalLM",
        "Qwen3NextForCausalLM",
        "InternS2PreviewForConditionalGeneration",
        "MiniCPMV4_6ForConditionalGeneration",
        "BailingMoeV2_5ForCausalLM",
        "FalconH1ForCausalLM",
        "GraniteMoeHybridForCausalLM",
        "NemotronHForCausalLM",
        "NemotronHPuzzleForCausalLM",


        "KimiK3ForConditionalGeneration",
    }
)


def supports_mamba_cache_extra_buffer(view: Any, model_arch: str) -> bool:

    if model_arch in _MAMBA_EXTRA_BUFFER_ARCHS:
        return view.linear_attn_backend == "triton"
    return False


@register_post_process
def _mamba_radix_cache_resolution(view: Any) -> dict:

    from sglang.srt.configs.linear_attn_model_registry import (
        get_linear_attn_spec_by_arch,
    )

    hf_config = view.get_model_config().hf_config
    model_arch = hf_config.architectures[0]

    in_branch = model_arch in _MAMBA_RADIX_CACHE_ARCHS
    if model_arch == "GraniteMoeHybridForCausalLM":
        in_branch = any(
            layer_type == "mamba"
            for layer_type in getattr(hf_config, "layer_types", [])
        )
    spec = get_linear_attn_spec_by_arch(model_arch)
    if not ((spec is not None and spec.uses_mamba_radix_cache) or in_branch):
        return {}

    if view.disable_radix_cache:
        return {}

    declared: Dict[str, Any] = {"uses_mamba_radix_cache": True}
    if view.mamba_radix_cache_strategy == "auto":
        wants_overlap = not view.disable_overlap_schedule
        wants_paging = view.page_size is not None and view.page_size > 1
        if (wants_overlap or wants_paging) and supports_mamba_cache_extra_buffer(
            view, model_arch
        ):
            declared["mamba_radix_cache_strategy"] = "extra_buffer"
        else:
            declared["mamba_radix_cache_strategy"] = "no_buffer"
            declared["disable_overlap_schedule"] = True
    return declared


@register_post_process
def _dsa_kv_cache_dtype_default(view: Any) -> dict:

    from sglang.srt.configs.model_config import is_deepseek_dsa

    hf_config = view.get_model_config().hf_config
    if hf_config.architectures[0] not in _DEEPSEEK_FAMILY_ARCHS:
        return {}
    if not is_deepseek_dsa(hf_config):
        return {}
    if is_npu() or is_xpu():
        return {}

    import torch

    major, _ = torch.cuda.get_device_capability()


    if (
        view.dsa_prefill_backend is not None or view.dsa_decode_backend is not None
    ) and view.kv_cache_dtype == "auto":
        logger.warning(
            "When specifying --dsa-prefill-backend or --dsa-decode-backend, "
            "you should also explicitly set --kv-cache-dtype (e.g., 'fp8_e4m3' or 'bfloat16'). "
            "DeepSeek V3.2 defaults to FP8 KV cache which may not be compatible with all backends."
        )

    kv_cache_dtype = view.kv_cache_dtype
    if kv_cache_dtype == "auto":
        kv_cache_dtype = "fp8_e4m3" if major >= 10 else "bfloat16"
        logger.warning(
            f"Setting KV cache dtype to {kv_cache_dtype} for DeepSeek DSA on SM{major} device."
        )
    if kv_cache_dtype == "bf16":
        kv_cache_dtype = "bfloat16"
    assert kv_cache_dtype in [
        "bfloat16",
        "fp8_e4m3",
    ], "DeepSeek DSA only supports bf16/bfloat16 or fp8_e4m3 kv_cache_dtype"
    if kv_cache_dtype != view.kv_cache_dtype:
        return {"kv_cache_dtype": kv_cache_dtype}
    return {}


def _check_tilelang_dsa_fp8_kv(
    kv_cache_dtype: str,
    prefill_backend: Optional[str],
    decode_backend: Optional[str],
    *,
    hip: bool,
) -> None:

    if (
        not hip
        and kv_cache_dtype == "fp8_e4m3"
        and "tilelang" in {prefill_backend, decode_backend}
    ):
        raise ValueError(
            "The tilelang DSA prefill/decode kernels only support an fp8_e4m3 KV "
            "cache on ROCm/HIP; on CUDA they require a bfloat16 KV cache. Use "
            "--kv-cache-dtype bfloat16 with the tilelang backend, or keep "
            "--kv-cache-dtype fp8_e4m3 and pick an fp8-capable DSA backend "
            "(flashmla_kv on Hopper, trtllm on Blackwell)."
        )


@register_post_process
def _dsa_split_backend_resolution(view: Any) -> dict:

    from sglang.srt.configs.model_config import is_deepseek_dsa

    hf_config = view.get_model_config().hf_config
    if hf_config.architectures[0] not in _DEEPSEEK_FAMILY_ARCHS:
        return {}
    if not is_deepseek_dsa(hf_config):
        return {}
    if is_npu() or is_xpu():
        return {}

    import torch

    major, _ = torch.cuda.get_device_capability()
    kv_cache_dtype = view.kv_cache_dtype
    user_set_prefill = view.dsa_prefill_backend is not None
    user_set_decode = view.dsa_decode_backend is not None
    declared: Dict[str, Any] = {}
    model_arch = hf_config.architectures[0]
    is_glm_sm12_fp8 = (
        model_arch == "GlmMoeDsaForCausalLM"
        and major == 12
        and kv_cache_dtype == "fp8_e4m3"
        and not is_hip()
    )

    if is_glm_sm12_fp8:
        backend = "flashinfer_sparse_mla"
        if not user_set_prefill:
            declared["dsa_prefill_backend"] = backend
        if not user_set_decode:
            declared["dsa_decode_backend"] = backend
        logger.warning(
            "Set DSA backends for GLM FP8 KV Cache on SM120/SM121: "
            f"prefill={backend}, decode={backend}."
        )
        return declared

    if view.enable_hisparse:
        from sglang.srt.arg_groups.hisparse_hook import _hisparse_default_backend

        backend = _hisparse_default_backend(kv_cache_dtype)
        if not user_set_prefill:
            declared["dsa_prefill_backend"] = backend
        if not user_set_decode:
            declared["dsa_decode_backend"] = backend
        prefill = declared.get("dsa_prefill_backend", view.dsa_prefill_backend)
        decode = declared.get("dsa_decode_backend", view.dsa_decode_backend)
        logger.warning(
            f"HiSparse enabled ({kv_cache_dtype}): using DSA backends "
            f"prefill={prefill}, decode={decode}."
        )
        return declared

    if not user_set_prefill and not user_set_decode and is_hip():
        declared["dsa_prefill_backend"] = "tilelang"
        declared["dsa_decode_backend"] = "tilelang"
    elif kv_cache_dtype == "fp8_e4m3":

        default = "trtllm" if major >= 10 else "flashmla_kv"
        if not user_set_prefill:
            declared["dsa_prefill_backend"] = default
        if not user_set_decode:
            declared["dsa_decode_backend"] = default
    else:

        if not user_set_prefill:
            declared["dsa_prefill_backend"] = "flashmla_sparse"
        if not user_set_decode:
            declared["dsa_decode_backend"] = "trtllm" if major >= 10 else "fa3"

    prefill = declared.get("dsa_prefill_backend", view.dsa_prefill_backend)
    decode = declared.get("dsa_decode_backend", view.dsa_decode_backend)
    _check_tilelang_dsa_fp8_kv(kv_cache_dtype, prefill, decode, hip=is_hip())
    logger.warning(
        f"Set DSA backends for {kv_cache_dtype} KV Cache: "
        f"prefill={prefill}, decode={decode}."
    )
    return declared


_DEEPSEEK_FAMILY_ARCHS = frozenset(
    {
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
        "KimiK25ForConditionalGeneration",
        "MistralLarge3ForCausalLM",
        "PixtralForConditionalGeneration",
        "GlmMoeDsaForCausalLM",
        "LongcatFlashForCausalLM",
        "LongcatFlashForCausalLMNextN",
        "Dots3NoteForCausalLM",
    }
)


@register_post_process
def _deepseek_moe_quant_resolution(view: Any) -> dict:

    hf_config = view.get_model_config().hf_config
    model_arch = hf_config.architectures[0]
    if model_arch not in _DEEPSEEK_FAMILY_ARCHS:
        return {}
    overrides: Dict[str, Any] = {}
    if is_sm100_supported():
        quant_method = get_quantization_config(hf_config)
        quant_cfg = getattr(hf_config, "quantization_config", None) or {}
        config_groups = quant_cfg.get("config_groups", {})
        group0 = config_groups.get("group_0", {})
        weights_cfg = group0.get("weights", {})


        is_kimi_k2_k25_thinking_int4 = (
            quant_method == "compressed-tensors"
            and weights_cfg.get("num_bits") == 4
            and weights_cfg.get("group_size") == 32
            and weights_cfg.get("strategy") == "group"
            and weights_cfg.get("type") == "int"
        )
        quantization = view.quantization
        if quantization is None and not view._quantization_explicitly_unset:


            if quant_method is None and model_arch in ["DeepseekV3ForCausalLM"]:
                from sglang.srt.utils.common import has_fp8_weights_in_checkpoint

                if has_fp8_weights_in_checkpoint(view.model_path):
                    overrides["quantization"] = quantization = "fp8"
                    logger.info(
                        "Detected FP8 expert weights in checkpoint, "
                        "default to fp8 for DeepSeek on sm100"
                    )
                else:
                    logger.info(
                        "No FP8 expert weights found in checkpoint, "
                        "keeping bf16 for DeepSeek-arch model on sm100"
                    )
            else:
                overrides["quantization"] = quantization = quant_method
        if (
            view.moe_a2a_backend == "none"
            and view.moe_runner_backend == "auto"


            and not model_arch.startswith("LongcatFlash")
            and (
                quantization
                in ["fp8", "modelopt_fp8", "modelopt_fp4", "modelopt_mixed"]
                or is_kimi_k2_k25_thinking_int4
                or quantization is None
            )
        ):
            overrides["moe_runner_backend"] = "flashinfer_trtllm"
            if is_kimi_k2_k25_thinking_int4:
                logger.info(
                    "Use flashinfer_trtllm as MoE runner backend on Blackwell for Kimi K2 / K2.5 thinking int4"
                )
            else:
                logger.info(
                    "Use flashinfer_trtllm as MoE runner backend on sm100 for DeepseekV3ForCausalLM"
                )
        if (
            model_arch in ["LongcatFlashForCausalLM", "LongcatFlashForCausalLMNextN"]
            and view.fp8_gemm_runner_backend == "auto"
            and quantization in ["fp8", "modelopt_fp8"]
            and quant_cfg.get("scale_fmt", None) != "ue8m0"
        ):
            overrides["fp8_gemm_runner_backend"] = "flashinfer_trtllm"
            logger.info(
                "Use flashinfer_trtllm as FP8 GEMM backend on Blackwell for LongCat FP8 "
                "checkpoint with non-ue8m0 scales"
            )
    return overrides


@register_post_process
def _deepseek_spec_moe_resolution(view: Any) -> dict:


    hf_config = view.get_model_config().hf_config
    model_arch = hf_config.architectures[0]
    if model_arch not in _DEEPSEEK_FAMILY_ARCHS:
        return {}
    if not is_hip():
        return {}
    if not (
        view.quantization == "modelopt_fp4"
        and view.speculative_algorithm == "EAGLE"
        and (
            view.speculative_moe_runner_backend is None
            or view.speculative_moe_a2a_backend is None
        )
    ):
        return {}
    if envs.SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE.get():
        logger.info(
            "Use deep_gemm moe runner and deepep a2a backend for bf16 nextn layer in deepseek fp4 checkpoint."
        )

        if view.ep_size == 1:
            raise ValueError(
                "Invalid configuration: 'deep_gemm' speculative MoE runner backend with "
                "'deepep' a2a backend requires expert parallelism (ep_size > 1). "
                f"Current ep_size is {view.ep_size}. "
                "Please set --ep-size > 1 (e.g., --ep-size 8) to use this configuration, "
                "or change --speculative-moe-a2a-backend to 'none' if expert parallelism is not available."
            )
        return {
            "speculative_moe_runner_backend": "deep_gemm",
            "speculative_moe_a2a_backend": "deepep",
        }
    logger.info(
        "Use triton fused moe by default for bf16 nextn layer in deepseek fp4 checkpoint."
    )
    return {
        "speculative_moe_runner_backend": "triton",
        "speculative_moe_a2a_backend": "none",
    }


@register_post_process
def _deepseek_v4_kv_cache_dtype(view: Any) -> dict:

    hf_config = view.get_model_config().hf_config
    model_arch = hf_config.architectures[0]
    if model_arch != "DeepseekV4ForCausalLM":
        return {}

    kv_cache_dtype = view.kv_cache_dtype
    if kv_cache_dtype == "auto":
        kv_cache_dtype = "fp8_e4m3"
        logger.warning(f"Setting KV cache dtype to {kv_cache_dtype} for {model_arch}.")
    if view.device == "npu":
        kv_cache_dtype = "bfloat16"
    assert kv_cache_dtype in [
        "fp8_e4m3",
        "bfloat16",
    ], f"{kv_cache_dtype} is not supported for {model_arch}"
    if kv_cache_dtype != view.kv_cache_dtype:
        return {"kv_cache_dtype": kv_cache_dtype}
    return {}


@_register_for("MuseGlimmerForConditionalGeneration", "MuseGlimmerForCausalLM")
def _muse_glimmer_fp4_gemm_runner_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    if is_sm120_supported() and cfg.fp4_gemm_runner_backend == "auto":
        logger.info("Use marlin as FP4 GEMM runner backend on SM120 for Muse Glimmer")
        return {"fp4_gemm_runner_backend": "marlin"}
    return {}


@register_post_process
def _sparse_head_overlap_disable(view: Any) -> dict:

    if envs.SGLANG_EMBEDDINGS_SPARSE_HEAD.is_set():
        logger.warning(
            "Overlap scheduler is disabled when using sparse head for embedding model."
        )
        return {"disable_overlap_schedule": True}
    return {}


_FLASHINFER_ALLREDUCE_FUSION_ARCHS = frozenset(
    {
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
        "DeepseekV4ForCausalLM",
        "GptOssForCausalLM",
        "GlmMoeDsaForCausalLM",
        "Glm4MoeForCausalLM",
        "Glm4MoeLiteForCausalLM",
        "MistralLarge3ForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3VLMoeForConditionalGeneration",
        "Qwen3NextForCausalLM",
        "KimiK25ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        "InternS2PreviewForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration",
        "NemotronHForCausalLM",
        "NemotronHPuzzleForCausalLM",
    }
)


@register_post_process
def _flashinfer_allreduce_fusion_auto_enable(view: Any) -> dict:

    model_arch = view.get_model_config().hf_config.architectures[0]
    if (
        view.flashinfer_allreduce_fusion_backend is None
        and model_arch in _FLASHINFER_ALLREDUCE_FUSION_ARCHS
        and (is_sm90_supported() or is_sm100_supported())
        and view.tp_size > 1
        and not view.enable_dp_attention
        and (view.nnodes == 1 or is_sm100_supported())
        and view.moe_a2a_backend == "none"
    ):
        logger.info(
            f"Auto-enabling FlashInfer AllReduce Fusion on SM90/SM10X for {model_arch}"
        )
        return {"flashinfer_allreduce_fusion_backend": "auto"}
    return {}


@register_post_process
def _enforce_disable_allreduce_fusion(view: Any) -> dict:

    if view.enforce_disable_flashinfer_allreduce_fusion:
        logger.info(
            "FlashInfer allreduce fusion is forcibly disabled "
            "via --enforce-disable-flashinfer-allreduce-fusion."
        )
        return {"flashinfer_allreduce_fusion_backend": None}
    return {}


@register_post_process
def _sampling_backend_default(view: Any) -> dict:
    if view.sampling_backend is None:
        return {
            "sampling_backend": (
                "flashinfer" if is_flashinfer_available() else "pytorch"
            )
        }
    return {}


@register_post_process
def _deterministic_sampling_backend(view: Any) -> dict:
    if view.enable_deterministic_inference and view.sampling_backend != "ascend":
        logger.warning(
            "Sampling backend is set to pytorch for deterministic inference."
        )
        return {"sampling_backend": "pytorch"}
    return {}


def _deterministic_is_deepseek_model(view: Any) -> bool:

    from sglang.srt.connector import ConnectorType
    from sglang.srt.utils.common import parse_connector_type

    if parse_connector_type(view.model_path) == ConnectorType.INSTANCE:
        return False
    try:
        hf_config = view.get_model_config().hf_config
        return hf_config.architectures[0] in [
            "DeepseekV2ForCausalLM",
            "DeepseekV3ForCausalLM",
            "DeepseekV32ForCausalLM",
            "MistralLarge3ForCausalLM",
            "PixtralForConditionalGeneration",
            "GlmMoeDsaForCausalLM",
            "Glm4MoeLiteForCausalLM",
        ]
    except Exception:
        return False


@register_post_process
def _deterministic_allreduce_fusion_disable(view: Any) -> dict:
    if (
        view.enable_deterministic_inference
        and view.flashinfer_allreduce_fusion_backend is not None
    ):
        logger.warning(
            "Disable --flashinfer-allreduce-fusion-backend because deterministic inference is enabled."
        )
        return {"flashinfer_allreduce_fusion_backend": None}
    return {}


@register_post_process
def _deterministic_attention_backend(view: Any) -> dict:
    if not view.enable_deterministic_inference:
        return {}
    from sglang.srt.server_args import DETERMINISTIC_ATTENTION_BACKEND_CHOICES

    if view.attention_backend is None:

        if is_sm100_supported() or is_sm120_supported():

            if _deterministic_is_deepseek_model(view):


                backend = "triton"
            else:

                backend = "flashinfer"
        else:

            backend = "fa3"
        logger.warning(
            f"Attention backend not specified. Falling back to '{backend}' for deterministic inference. "
            f"You can explicitly set --attention-backend to one of {DETERMINISTIC_ATTENTION_BACKEND_CHOICES}."
        )
        return {"attention_backend": backend}
    elif view.attention_backend not in DETERMINISTIC_ATTENTION_BACKEND_CHOICES:

        raise ValueError(
            f"Currently only {DETERMINISTIC_ATTENTION_BACKEND_CHOICES} attention backends are supported for deterministic inference, "
            f"but you explicitly specified '{view.attention_backend}'."
        )
    return {}


@register_post_process
def _attention_backend_default(view: Any) -> dict:
    if view.prefill_attention_backend is not None and (
        view.prefill_attention_backend == view.decode_attention_backend
    ):
        return {"attention_backend": view.prefill_attention_backend}
    if view.attention_backend is None:
        backend = view._get_default_attn_backend(
            view.use_mla_backend(), view.get_model_config()
        )
        logger.info(
            f"Attention backend not specified. Use {backend} backend by default."
        )
        return {"attention_backend": backend}
    return {}


@register_post_process
def _mla_backend_page_constraints(view: Any) -> dict:

    page_size = view.page_size
    if (
        view.attention_backend == "flashmla"
        or view.decode_attention_backend == "flashmla"
    ):
        logger.warning(
            "FlashMLA only supports a page_size of 64, change page_size to 64."
        )
        page_size = 64
    if (
        view.attention_backend == "cutlass_mla"
        or view.decode_attention_backend == "cutlass_mla"
    ):
        logger.warning(
            "Cutlass MLA only supports a page_size of 128, change page_size to 128."
        )
        page_size = 128
    if (
        view.attention_backend == "trtllm_mla"
        or view.decode_attention_backend == "trtllm_mla"
    ):
        if page_size not in [32, 64]:
            logger.warning(
                f"TensorRT-LLM MLA only supports page_size of 32 or 64, changing page_size from {page_size} to 64."
            )
            page_size = 64
    if (
        view.attention_backend == "tokenspeed_mla"
        or view.decode_attention_backend == "tokenspeed_mla"
    ):
        if page_size not in [32, 64]:
            logger.warning(
                f"tokenspeed_mla only supports page_size of 32 or 64, changing page_size from {page_size} to 64."
            )
            page_size = 64
    if (
        view.attention_backend == "cutedsl_mla"
        or view.decode_attention_backend == "cutedsl_mla"
        or view.prefill_attention_backend == "cutedsl_mla"
    ):
        if page_size not in [32, 64]:
            logger.warning(
                f"CuteDSL MLA only supports page_size of 32 or 64, changing page_size from {page_size} to 64."
            )
            page_size = 64
    if (
        view.attention_backend == "trtllm_mha"
        or view.decode_attention_backend == "trtllm_mha"
        or view.prefill_attention_backend == "trtllm_mha"
        or view.speculative_draft_attention_backend == "trtllm_mha"
    ):


        if page_size not in [16, 32, 64, 128]:
            logger.warning(
                f"TensorRT-LLM MHA only supports page_size of 16, 32, 64 or 128, changing page_size from {page_size} to 64."
            )
            page_size = 64
    if (
        view.attention_backend == "hpc_ops"
        or view.decode_attention_backend == "hpc_ops"
        or view.prefill_attention_backend == "hpc_ops"
    ):
        if page_size != 64:
            logger.warning(
                f"HPC-Ops attention only supports a page_size of 64, changing page_size from {page_size} to 64."
            )
            page_size = 64
    if page_size != view.page_size:
        return {"page_size": page_size}
    return {}


@register_post_process
def _mla_kv_cache_dtype_checks(view: Any) -> dict:

    if (
        view.attention_backend == "trtllm_mla"
        or view.decode_attention_backend == "trtllm_mla"
    ):
        if not is_blackwell_supported():
            raise ValueError(
                "TRTLLM MLA backend is only supported on Blackwell GPUs (SM100/SM12x). Please use a different backend."
            )
        if view.kv_cache_dtype not in ["fp8_e4m3", "fp4_e2m1", "bf16", "auto"]:
            raise ValueError(
                "TensorRT-LLM MLA backend only supports kv-cache-dtype of fp8_e4m3, fp4_e2m1, bf16, or auto."
            )
    if (
        view.attention_backend == "tokenspeed_mla"
        or view.decode_attention_backend == "tokenspeed_mla"
    ):
        if not is_blackwell_supported():
            raise ValueError(
                "tokenspeed_mla backend is only supported on Blackwell GPUs (SM100/SM12x)."
            )
        if view.kv_cache_dtype not in ["fp8_e4m3"]:
            raise ValueError(
                "tokenspeed_mla backend requires kv-cache-dtype=fp8_e4m3, "
                f"got {view.kv_cache_dtype}."
            )
    return {}


@register_post_process
def _hisparse_validation(view: Any) -> dict:

    from sglang.srt.arg_groups.hisparse_hook import validate_hisparse

    validate_hisparse(view)
    return {}


@register_post_process
def _cutedsl_prefill_backend_fill(view: Any) -> dict:

    if not (
        view.attention_backend == "cutedsl_mla"
        or view.decode_attention_backend == "cutedsl_mla"
        or view.prefill_attention_backend == "cutedsl_mla"
    ):
        return {}
    assert (
        view.prefill_attention_backend != "cutedsl_mla"
    ), "CuteDSL MLA only supports decoding for now"
    if not is_sm100_supported():
        raise ValueError(
            "CuteDSL MLA backend is only supported on Blackwell GPUs (SM100). Please use a different backend."
        )
    if view.kv_cache_dtype not in [
        "fp8_e4m3",
        "bf16",
        "bfloat16",
        "auto",
    ]:
        raise ValueError(
            "CuteDSL MLA backend only supports kv-cache-dtype of fp8_e4m3, bf16, or auto."
        )
    if view.prefill_attention_backend is None:
        return {"prefill_attention_backend": "trtllm_mla"}
    return {}


@register_post_process
def _attention_backend_fa3_fp8_fallback(view: Any) -> dict:
    if view.attention_backend == "fa3" and view.kv_cache_dtype == "fp8_e5m2":
        logger.warning(
            "FlashAttention3 only supports fp8_e4m3 if using FP8; "
            "Setting attention backend to triton."
        )
        return {"attention_backend": "triton"}
    return {}


@register_post_process
def _fa4_page_constraint(view: Any) -> dict:
    if (
        (
            view.attention_backend == "fa4"
            or view.decode_attention_backend == "fa4"
            or view.prefill_attention_backend == "fa4"
        )
        and not view.use_mla_backend()
        and is_sm100_supported()


        and (view.speculative_eagle_topk or 0) <= 1
    ):
        logger.warning(
            f"FA4 backend only supports page size 128 for non-MLA model architectures, changing page_size from {view.page_size} to 128."
        )
        return {"page_size": 128}
    return {}


@register_post_process
def _attention_backend_platform_fallbacks(view: Any) -> dict:
    if (
        view.attention_backend == "intel_amx"
        and view.device == "cpu"
        and not cpu_has_amx_support()
    ):
        logger.warning(
            "The current platform does not support Intel AMX, will fallback to torch_native backend."
        )
        return {"attention_backend": "torch_native"}
    if (
        view.attention_backend == "intel_xpu"
        and view.device == "xpu"
        and not xpu_has_xmx_support()
    ):
        logger.warning(
            "The current platform does not support Intel XMX, will fallback to triton backend."
        )
        return {"attention_backend": "triton"}
    return {}


@register_post_process
def _intel_xpu_page_constraint(view: Any) -> dict:
    _, decode_backend = attention_backends_of(view)
    if decode_backend == "intel_xpu":
        if view.use_mla_backend():
            supported_page_sizes = [16, 32, 64, 128]
            msg = "Intel XPU attention backend for MLA Decode"
        else:
            supported_page_sizes = [64, 128]
            msg = "Intel XPU attention backend"
        if view.page_size not in supported_page_sizes:
            logger.warning(
                f"{msg} only supports page_sizes of {supported_page_sizes}, changing page_size from {view.page_size} to 128."
            )
            return {"page_size": 128}
    return {}


@register_post_process
def _attention_backend_dual_chunk(view: Any) -> dict:
    if (
        getattr(view.get_model_config().hf_config, "dual_chunk_attention_config", None)
        is not None
    ):
        if view.attention_backend is None:
            logger.info("Dual chunk attention is turned on by default.")
            return {"attention_backend": "dual_chunk_flash_attn"}
        elif view.attention_backend != "dual_chunk_flash_attn":
            raise ValueError(
                "Dual chunk attention is enabled, but attention backend is set to "
                f"{view.attention_backend}. Please set it to 'dual_chunk_flash_attn'."
            )
    return {}


@register_post_process
def _page_size_default(view: Any) -> dict:
    if view.page_size is not None:
        return {}


    if is_hip() and envs.SGLANG_AITER_KV_CACHE_LAYOUT.get().lower() == "vectorized_5d":
        logger.info(
            "Setting page_size=64 as default for "
            "SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d."
        )
        return {"page_size": 64}
    if not is_musa():
        return {"page_size": 1}
    return {"page_size": 64}


@register_post_process
def _data_parallelism_defaults(view: Any) -> dict:
    if view.dp_size == 1 and view.ep_join_mode != "scale":
        return {"enable_dp_attention": False, "enable_dp_lm_head": False}
    return {}


@register_post_process
def _tp_lm_head_all_to_all_default(view: Any) -> dict:

    if view.enable_tp_lm_head_all_to_all is not None:
        return {}

    enable = (
        view.disaggregation_mode == "decode"
        and view.enable_dp_attention
        and view.dp_size > 1
        and view.tp_size == view.dp_size
        and view.attn_cp_size == 1
        and not view.enable_dp_lm_head
    )
    return {"enable_tp_lm_head_all_to_all": enable}


@register_post_process
def _dp_lm_head_validation(view: Any) -> dict:

    if view.enable_dp_lm_head:
        assert (
            view.enable_dp_attention
        ), "Please enable dp attention when setting enable_dp_lm_head. "
    if view.enable_tp_lm_head_all_to_all:
        assert view.enable_dp_attention, (
            "Please enable dp attention when setting " "enable_tp_lm_head_all_to_all."
        )
        assert not view.enable_dp_lm_head, (
            "--enable-tp-lm-head-all-to-all uses a TP-sharded LM head and is "
            "incompatible with --enable-dp-lm-head."
        )
        assert view.tp_size == view.dp_size, (
            "--enable-tp-lm-head-all-to-all currently requires tp_size == "
            f"dp_size, got tp_size={view.tp_size}, dp_size={view.dp_size}."
        )
        assert view.attn_cp_size == 1, (
            "--enable-tp-lm-head-all-to-all currently requires "
            f"attn_cp_size == 1, got {view.attn_cp_size}."
        )
    return {}


@register_post_process
def _moe_runner_backend_quant_constraints(view: Any) -> dict:

    moe_runner_backend = view.moe_runner_backend
    if view.quantization == "nvfp4_online":
        if not is_sm100_supported():
            raise ValueError(
                "--quantization nvfp4_online is supported only on "
                "NVIDIA Blackwell SM100/SM103 GPUs."
            )
        if moe_runner_backend == "auto":
            moe_runner_backend = "flashinfer_trtllm"
        elif moe_runner_backend not in [
            "flashinfer_trtllm",
            "flashinfer_trtllm_routed",
            "flashinfer_cutedsl",
        ]:
            raise ValueError(
                "--quantization nvfp4_online supports only "
                "--moe-runner-backend flashinfer_trtllm or "
                "flashinfer_trtllm_routed, or flashinfer_cutedsl."
            )


    if view.quantization == "mxfp8" and not is_npu():
        from sglang.srt.server_args import MXFP8_MOE_RUNNER_BACKEND_CHOICES

        is_gfx95_mxfp8 = is_hip() and is_gfx95_supported()
        allowed = list(MXFP8_MOE_RUNNER_BACKEND_CHOICES)
        if is_gfx95_mxfp8:
            allowed.append("triton")
        mxfp8_default = "triton" if is_gfx95_mxfp8 else "flashinfer_trtllm"
        if moe_runner_backend == "auto":
            moe_runner_backend = mxfp8_default
        elif moe_runner_backend not in allowed:
            logger.warning(
                "mxfp8 quantization supports only %s backends. " "Overriding %r.",
                ", ".join(allowed),
                moe_runner_backend,
            )
            moe_runner_backend = mxfp8_default
    if (
        moe_runner_backend == "auto"
        and view.quantization == "modelopt_fp4"
        and is_sm120_supported()
    ):
        moe_runner_backend = "flashinfer_cutlass"
        logger.info(
            "Use flashinfer_cutlass as MoE runner backend on SM120 for "
            "modelopt_fp4 (trtllm-gen MoE kernels are SM100-only)"
        )
    if moe_runner_backend != view.moe_runner_backend:
        return {"moe_runner_backend": moe_runner_backend}
    return {}


@register_post_process
def _moe_runner_fusion_disable(view: Any) -> dict:

    runner = view.moe_runner_backend
    if runner == "flashinfer_cutedsl":
        logger.warning(
            "FlashInfer CuteDSL MoE is enabled. --disable-shared-experts-fusion is automatically set."
        )
        return {"disable_shared_experts_fusion": True}
    if runner in ("flashinfer_trtllm", "experimental_sgl_trtllm"):
        logger.warning(
            "FlashInfer TRTLLM MoE is enabled. --disable-shared-experts-fusion is automatically set."
        )
        return {"disable_shared_experts_fusion": True}
    if runner == "flashinfer_trtllm_routed":
        logger.warning(
            "FlashInfer TRTLLM routed MoE is enabled. --disable-shared-experts-fusion is automatically set."
        )
        return {"disable_shared_experts_fusion": True}
    return {}


def _a2a_fusion_adjustments(view: Any) -> dict:

    if view.moe_a2a_backend in ("deepep", "megamoe") and view.enable_waterfill:
        if view.disable_shared_experts_fusion:
            logger.warning(
                "disable_shared_experts_fusion is overridden to False because Waterfill requires shared expert fusion."
            )
            return {"disable_shared_experts_fusion": False}
        return {}
    if view.moe_a2a_backend == "flashinfer":
        logger.warning(
            "Flashinfer MoE A2A is enabled. --disable-shared-experts-fusion is automatically set."
        )
        return {"disable_shared_experts_fusion": True}
    return {}


_A2A_EP_SPANNING_BACKENDS = frozenset(
    {
        "megamoe",
        "deepep",
        "mooncake",
        "nixl",
        "ascend_fuseep",
        "flashinfer",
        "mori",
        "pplx",
    }
)


@register_post_process
def _a2a_backend_overrides(view: Any) -> dict:

    moe_a2a_backend = view.moe_a2a_backend
    if view.enable_waterfill and moe_a2a_backend not in ("deepep", "megamoe"):
        logger.warning(
            "moe_a2a_backend is overridden to 'deepep' because Waterfill "
            "requires the DeepEP or MegaMOE backend."
        )
        moe_a2a_backend = "deepep"
    if moe_a2a_backend != view.moe_a2a_backend:
        return {"moe_a2a_backend": moe_a2a_backend}
    return {}


@register_post_process
def _a2a_ep_size(view: Any) -> dict:
    if view.moe_a2a_backend in _A2A_EP_SPANNING_BACKENDS:
        if view.ep_size != view.tp_size:
            logger.info(
                f"{view.moe_a2a_backend} MoE is enabled. The expert parallel size "
                f"is adjusted from {view.ep_size} to the tensor parallel size "
                f"[{view.tp_size}]."
            )
        return {"ep_size": view.tp_size}
    return {}


@register_post_process
def _pipeline_parallel_overlap_disable(view: Any) -> dict:
    if view.pp_size > 1:
        logger.warning("Pipeline parallelism is incompatible with overlap schedule.")
        return {"disable_overlap_schedule": True}
    return {}


@register_post_process
def _speculative_moe_runner_default(view: Any) -> dict:

    if view.speculative_moe_runner_backend is None:
        return {"speculative_moe_runner_backend": view.moe_runner_backend}
    return {}


@register_post_process
def _gguf_quantization(view: Any) -> dict:
    from sglang.srt.utils.hf_transformers_utils import check_gguf_file

    if (view.load_format == "auto" or view.load_format == "gguf") and check_gguf_file(
        view.model_path
    ):
        return {"quantization": "gguf"}
    return {}


@register_post_process
def _dllm_attention_backend(view: Any) -> dict:
    if view.dllm_algorithm is None:
        return {}
    if is_hip():
        if view.attention_backend not in ["triton", "aiter"]:
            logger.warning(
                "Attention backend is set to triton for diffusion LLM inference on AMD GPUs"
            )
            return {"attention_backend": "triton"}
    elif is_npu():
        if view.attention_backend != "ascend":
            logger.warning(
                "Attention backend is overridden to 'ascend' when running on NPU for diffusion LLM inference."
            )
            return {"attention_backend": "ascend"}
    elif view.cuda_graph_config.decode.backend != Backend.DISABLED:
        if view.attention_backend != "flashinfer":
            logger.warning(
                "Attention backend is set to flashinfer because of enabling cuda graph in diffusion LLM inference"
            )
            return {"attention_backend": "flashinfer"}
    return {}


@register_post_process
def _dllm_overlap_disable(view: Any) -> dict:
    if view.dllm_algorithm is None:
        return {}
    if view.disable_overlap_schedule:
        return {}
    logger.warning(
        "Overlap schedule is disabled because of using diffusion LLM inference"
    )
    return {"disable_overlap_schedule": True}


@register_post_process
def _dllm_page_size(view: Any) -> dict:
    if view.dllm_algorithm is None:
        return {}
    from sglang.srt.dllm.config import DllmConfig

    config = DllmConfig.from_server_args(view)
    if not view.disable_radix_cache and view.page_size % config.block_size != 0:
        logger.warning(
            f"Setting page size to {config.block_size} for diffusion LLM inference"
        )
        return {"page_size": config.block_size}
    if view.page_size > config.block_size:


        logger.warning(
            "WARNING: "
            f"The page size {view.page_size} should not be larger than dllm block size {config.block_size}."
            f"Page size now falls back to {config.block_size}"
        )
        return {"page_size": config.block_size}
    return {}


def validate_declarations(
    server_args: Any,
    declarations: Sequence[Tuple[str, Dict[str, Any]]],
) -> None:


    if not dataclasses.is_dataclass(type(server_args)):
        return
    whitelist = resolvable_fields(type(server_args))
    for source, decl in declarations:
        unknown = set(decl) - whitelist
        if unknown:
            raise ValueError(
                f"{source}: {sorted(unknown)} not model-overridable; "
                "declarations are limited to the fields the publish gate "
                "accepts."
            )


def _hrm_text_attention_force(view: Any) -> dict:

    if view.attention_backend not in (None, "triton"):
        logger.warning(
            f"Overriding --attention-backend "
            f"{view.attention_backend!r} -> 'triton': only the "
            "Triton backend supports HRM-Text's bidirectional prefix "
            "attention."
        )
    return {"attention_backend": "triton"}
