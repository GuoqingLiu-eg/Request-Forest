# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/model_loader/weight_utils.py


import collections
import concurrent.futures
import fnmatch
import glob
import hashlib
import itertools
import json
import logging
import os
import re
import struct
import tempfile
import threading
from collections import defaultdict
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import filelock
import huggingface_hub.constants
import numpy as np
import safetensors.torch
import torch
from huggingface_hub import HfFileSystem, hf_hub_download, snapshot_download
from pydantic import BaseModel, ConfigDict, ValidationInfo, model_validator
from tqdm.auto import tqdm

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import REQUANTIZATION_METHODS, ModelConfig
from sglang.srt.distributed import get_world_group
from sglang.srt.layers.quantization import QuantizationConfig, get_quantization_config
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp8Config,
)
from sglang.srt.model_loader.checkpoint_quantization import (
    resolve_checkpoint_quant_spec,
)
from sglang.srt.model_loader.ci_weight_validation import (
    ci_download_with_validation_and_retry,
    ci_validate_and_cleanup_local_snapshot,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import (
    BAR_FORMAT,
    find_local_repo_dir,
    is_cpu,
    log_info_on_rank0,
    print_warning_once,
)
from sglang.srt.utils.common import is_cuda_alike
from sglang.utils import is_in_ci

try:
    from fastsafetensors import SafeTensorsFileLoader, SingleGroup
except ImportError:
    SafeTensorsFileLoader = SingleGroup = None

logger = logging.getLogger(__name__)

RUNAI_STREAMER_TENSOR_ATTR = "_sglang_runai_streamer_tensor"


_ROUTED_EXPERT_KEY_RE = re.compile(
    r"\.experts\.\d+\.(?:w[123]|down_proj|up_proj|gate_proj)\.weight$"
)


def probe_routed_expert_weight_dtype(model_path: str) -> Optional[str]:

    if not os.path.isdir(model_path):
        return None

    index_file = os.path.join(model_path, "model.safetensors.index.json")
    target_key = None
    target_shard_path = None

    if os.path.exists(index_file):
        with open(index_file) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {}) or {}
        for k, shard in weight_map.items():
            if _ROUTED_EXPERT_KEY_RE.search(k):
                target_key = k
                target_shard_path = os.path.join(model_path, shard)
                break
        if target_key is None:
            return None
    else:
        shards = sorted(Path(model_path).glob("*.safetensors"))
        if not shards:
            return None
        target_shard_path = str(shards[0])

    with open(target_shard_path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))

    if target_key is not None:
        meta = header.get(target_key)
        return meta.get("dtype") if meta else None

    for k, meta in header.items():
        if k == "__metadata__" or not isinstance(meta, dict):
            continue
        if _ROUTED_EXPERT_KEY_RE.search(k):
            return meta.get("dtype")
    return None


_PREFETCH_BLOCK_SIZE = None
_PREFETCH_STOP_TIMEOUT_SECONDS = 60.0
CAPTURE_SAFE_WEIGHT_SENTINEL = 1e-3


def _get_prefetch_block_size() -> int:
    global _PREFETCH_BLOCK_SIZE
    if _PREFETCH_BLOCK_SIZE is None:
        from sglang.srt.environ import envs

        _PREFETCH_BLOCK_SIZE = envs.SGLANG_PREFETCH_BLOCK_SIZE_MB.get() * 1024 * 1024
    return _PREFETCH_BLOCK_SIZE


temp_dir = tempfile.gettempdir()


def get_lock(
    model_name_or_path: str, cache_dir: Optional[str] = None, suffix: str = ""
):
    lock_dir = cache_dir or temp_dir
    os.makedirs(os.path.dirname(lock_dir), exist_ok=True)
    model_name = model_name_or_path.replace("/", "-")
    hash_name = hashlib.sha256(model_name.encode()).hexdigest()

    lock_file_name = hash_name + model_name + suffix + ".lock"

    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name), mode=0o666)
    return lock


def _shared_pointers(tensors):
    ptrs = defaultdict(list)
    for k, v in tensors.items():
        ptrs[v.data_ptr()].append(k)
    failing = []
    for _, names in ptrs.items():
        if len(names) > 1:
            failing.append(names)
    return failing


def convert_bin_to_safetensor_file(
    pt_filename: str,
    sf_filename: str,
) -> None:
    loaded = torch.load(pt_filename, map_location="cpu", weights_only=True)
    if "state_dict" in loaded:
        loaded = loaded["state_dict"]
    shared = _shared_pointers(loaded)
    for shared_weights in shared:
        for name in shared_weights[1:]:
            loaded.pop(name)


    loaded = {k: v.contiguous() for k, v in loaded.items()}

    dirname = os.path.dirname(sf_filename)
    os.makedirs(dirname, exist_ok=True)

    from safetensors.torch import save_file

    save_file(loaded, sf_filename, metadata={"format": "pt"})


    sf_size = os.stat(sf_filename).st_size
    pt_size = os.stat(pt_filename).st_size
    if (sf_size - pt_size) / pt_size > 0.01:
        raise RuntimeError(f"""The file size different is more than 1%:
         - {sf_filename}: {sf_size}
         - {pt_filename}: {pt_size}
         """)


    reloaded = safetensors.torch.load_file(sf_filename)
    for k in loaded:
        pt_tensor = loaded[k]
        sf_tensor = reloaded[k]
        if not torch.equal(pt_tensor, sf_tensor):
            raise RuntimeError(f"The output tensors do not match for key {k}")


def replace_prefix(key: str, prefix_mapping: dict[str, str]) -> str:
    for prefix, new_prefix in prefix_mapping.items():
        if key.startswith(prefix):
            key = key.replace(prefix, new_prefix, 1)
    return key


def replace_substrings(key: str, substring_mapping: dict[str, str]) -> str:
    for substr, new_substr in substring_mapping.items():
        if substr in key:
            key = key.replace(substr, new_substr)
    return key


class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


def _resolve_explicit_draft_quant_config(
    model_config: ModelConfig,
    quant_config: QuantizationConfig,
) -> QuantizationConfig:
    if not (
        model_config.is_draft_model and model_config.is_draft_quantization_explicit
    ):
        return quant_config

    if model_config.quantization == "modelopt_fp4" and (
        isinstance(quant_config, ModelOptFp4Config)
        and quant_config.is_checkpoint_nvfp4_serialized
        and quant_config.is_layer_excluded("mtp.layers.0.mlp.experts")
    ):
        return ModelOptFp4Config.for_online_weight_quantization(
            quant_config.packed_modules_mapping
        )

    return quant_config


def get_quant_config(
    model_config: ModelConfig,
    load_config: LoadConfig,
    packed_modules_mapping: Dict[str, List[str]],
    remap_prefix: Dict[str, str] | None = None,
) -> QuantizationConfig:
    quant_cls = get_quantization_config(model_config.quantization)


    if model_config.quantization == "gguf":
        return quant_cls.from_config({})

    checkpoint_quant_spec = resolve_checkpoint_quant_spec(model_config.hf_config)
    if checkpoint_quant_spec is not None:
        hf_quant_config = checkpoint_quant_spec.config


        modelopt_mixed_config_incomplete = (
            model_config.quantization == "modelopt_mixed"
            and (
                "quantized_layers" not in hf_quant_config
                or (
                    "kv_cache_quant_algo" not in hf_quant_config
                    and "kv_cache_scheme" not in hf_quant_config
                )
            )
        )
        if not modelopt_mixed_config_incomplete:
            hf_quant_config["packed_modules_mapping"] = packed_modules_mapping
            hf_quant_config["hf_config"] = model_config.hf_config


            if model_config.quantization in REQUANTIZATION_METHODS:
                hf_quant_config["requantization_method"] = model_config.quantization

            return _resolve_explicit_draft_quant_config(
                model_config, quant_cls.from_config(hf_quant_config)
            )


    if model_config.quantization == "bitsandbytes":
        if (
            not load_config.model_loader_extra_config
            or "qlora_adapter_name_or_path" not in load_config.model_loader_extra_config
        ):
            return quant_cls.from_config({"adapter_name_or_path": ""})
        model_name_or_path = load_config.model_loader_extra_config[
            "qlora_adapter_name_or_path"
        ]
    else:
        model_name_or_path = model_config.model_path

    is_local = os.path.isdir(model_name_or_path)
    if not is_local:

        with get_lock(model_name_or_path, load_config.download_dir):
            hf_folder = snapshot_download(
                model_name_or_path,
                revision=model_config.revision,
                allow_patterns="*.json",
                cache_dir=load_config.download_dir,
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                tqdm_class=DisabledTqdm,
            )
    else:
        hf_folder = model_name_or_path

    possible_config_filenames = quant_cls.get_config_filenames()


    if not possible_config_filenames:
        if model_config.quantization == "mxfp8":
            if not issubclass(quant_cls, Fp8Config):
                quant_cls = Fp8Config
            return quant_cls(use_mxfp8=True, is_checkpoint_fp8_serialized=False)
        if model_config.quantization == "quark_mxfp4":


            modelopt_quant_path = os.path.join(hf_folder, "hf_quant_config.json")
            if os.path.isfile(modelopt_quant_path):
                with open(modelopt_quant_path) as f:
                    raw_quant_config = json.load(f)
                source_quant = raw_quant_config.get("quantization", raw_quant_config)
                if "FP4" in (source_quant.get("quant_algo") or "").upper():
                    flat_quant_config = dict(source_quant)
                    flat_quant_config["quant_method"] = (
                        raw_quant_config.get("producer", {}).get("name") or "modelopt"
                    )
                    flat_quant_config["requantization_method"] = (
                        model_config.quantization
                    )
                    flat_quant_config["packed_modules_mapping"] = packed_modules_mapping
                    flat_quant_config["hf_config"] = model_config.hf_config
                    return quant_cls.from_config(flat_quant_config)
            return quant_cls(
                online_scheme=model_config.quantization,
                hf_config=model_config.hf_config,
            )
        return quant_cls()

    config_files = glob.glob(os.path.join(hf_folder, "*.json"))

    quant_config_files = [
        f for f in config_files if any(f.endswith(x) for x in possible_config_filenames)
    ]
    if len(quant_config_files) == 0:
        if model_config.quantization == "modelopt_fp4":


            if not issubclass(quant_cls, ModelOptFp4Config):
                quant_cls = ModelOptFp4Config
            return quant_cls.for_online_weight_quantization(packed_modules_mapping)
        raise ValueError(f"Cannot find the config file for {model_config.quantization}")
    if len(quant_config_files) > 1:
        raise ValueError(
            f"Found multiple config files for {model_config.quantization}: "
            f"{quant_config_files}"
        )

    quant_config_file = quant_config_files[0]
    with open(quant_config_file) as f:
        config = json.load(f)
        if remap_prefix is not None:
            exclude_modules = [
                replace_prefix(key, remap_prefix)
                for key in config["quantization"]["exclude_modules"]
            ]
            config["quantization"]["exclude_modules"] = exclude_modules
        config["packed_modules_mapping"] = packed_modules_mapping

        if model_config.quantization == "bitsandbytes":
            config["adapter_name_or_path"] = model_name_or_path
        elif model_config.quantization.startswith("modelopt") and (
            config.get("producer", {}).get("name", "").startswith("modelopt")
        ):
            quant_algo = config["quantization"]["quant_algo"]
            if quant_algo is None:

                if model_config.hf_config.architectures[0] != "LlamaForCausalLMEagle3":
                    raise ValueError(
                        f"Invalid quant_config, quantization method: {model_config.quantization},"
                        f"hf architectures: {model_config.hf_config.architectures[0]}. "
                    )
                return None
            elif quant_algo == "FP8" or model_config.quantization == "modelopt_fp8":
                if not issubclass(quant_cls, ModelOptFp8Config):
                    quant_cls = ModelOptFp8Config
                return quant_cls.from_config(config)
            elif "FP4" in quant_algo:
                if not issubclass(quant_cls, ModelOptFp4Config):
                    quant_cls = ModelOptFp4Config
                return _resolve_explicit_draft_quant_config(
                    model_config, quant_cls.from_config(config)
                )
        return _resolve_explicit_draft_quant_config(
            model_config, quant_cls.from_config(config)
        )


def _check_index_files_exist(snapshot_dir: str) -> Tuple[bool, Optional[str]]:

    index_files = [
        f for f in os.listdir(snapshot_dir) if f.endswith(".safetensors.index.json")
    ]

    if not index_files:
        return True, None

    for index_file in index_files:
        index_path = os.path.join(snapshot_dir, index_file)
        if not os.path.exists(index_path):
            continue
        try:
            with open(index_path) as f:
                weight_map = json.load(f).get("weight_map", {})
            if not weight_map:
                continue
            required_files = set(weight_map.values())
            missing_files = [
                fn
                for fn in required_files
                if not os.path.exists(os.path.join(snapshot_dir, fn))
            ]
            if missing_files:
                return (
                    False,
                    f"Missing {len(missing_files)} file(s) from index {index_file}: "
                    f"{missing_files[:3]}{'...' if len(missing_files) > 3 else ''}",
                )
        except Exception as e:
            logger.warning("Failed to read index file %s: %s", index_file, e)
            continue

    return True, None


def _find_local_hf_snapshot_dir_unlocked(
    model_name_or_path: str,
    cache_dir: Optional[str],
    allow_patterns: List[str],
    revision: Optional[str] = None,
) -> Optional[str]:

    if os.path.isdir(model_name_or_path):
        return None

    found_local_snapshot_dir = None


    if cache_dir:
        try:
            repo_folder = os.path.join(
                cache_dir,
                huggingface_hub.constants.REPO_ID_SEPARATOR.join(
                    ["models", *model_name_or_path.split("/")]
                ),
            )
            rev_to_use = revision
            if not rev_to_use:
                ref_main = os.path.join(repo_folder, "refs", "main")
                if os.path.isfile(ref_main):
                    with open(ref_main) as f:
                        rev_to_use = f.read().strip()
            if rev_to_use:
                rev_dir = os.path.join(repo_folder, "snapshots", rev_to_use)
                if os.path.isdir(rev_dir):
                    found_local_snapshot_dir = rev_dir
        except Exception as e:
            logger.warning(
                "Failed to find local snapshot in custom cache_dir %s: %s",
                cache_dir,
                e,
            )


    if not found_local_snapshot_dir:
        try:
            rev_dir = find_local_repo_dir(model_name_or_path, revision)
            if rev_dir and os.path.isdir(rev_dir):
                found_local_snapshot_dir = rev_dir
        except Exception as e:
            logger.warning("Failed to find local snapshot in default HF cache: %s", e)


    if found_local_snapshot_dir is None:
        return None


    if not os.path.isdir(found_local_snapshot_dir):
        return None

    local_weight_files: List[str] = []
    try:
        for pattern in allow_patterns:
            matched_files = glob.glob(os.path.join(found_local_snapshot_dir, pattern))
            for f in matched_files:

                if not os.path.exists(f):
                    continue
                local_weight_files.append(f)
    except Exception as e:
        logger.warning(
            "Failed to scan local snapshot %s with patterns %s: %s",
            found_local_snapshot_dir,
            allow_patterns,
            e,
        )
        local_weight_files = []


    if local_weight_files:
        is_complete, error_msg = _check_index_files_exist(found_local_snapshot_dir)
        if not is_complete:
            log_info_on_rank0(
                logger,
                f"Local snapshot incomplete for {model_name_or_path}: {error_msg}. "
                f"Will download missing files.",
            )
            return None


    if is_in_ci() and local_weight_files:
        is_valid = ci_validate_and_cleanup_local_snapshot(
            model_name_or_path, found_local_snapshot_dir, local_weight_files
        )
        if not is_valid:
            return None

    if len(local_weight_files) > 0:
        log_info_on_rank0(
            logger,
            f"Found local HF snapshot for {model_name_or_path} at "
            f"{found_local_snapshot_dir}; skipping download.",
        )
        return found_local_snapshot_dir
    else:
        log_info_on_rank0(
            logger,
            f"Local HF snapshot at {found_local_snapshot_dir} has no files matching "
            f"{allow_patterns}; will attempt download.",
        )
        return None


def download_weights_from_hf(
    model_name_or_path: str,
    cache_dir: Optional[str],
    allow_patterns: List[str],
    revision: Optional[str] = None,
    ignore_patterns: Optional[Union[str, List[str]]] = None,
    max_retries: int = 3,
) -> str:


    if os.path.isdir(model_name_or_path):
        return model_name_or_path


    with get_lock(model_name_or_path, cache_dir):

        path = _find_local_hf_snapshot_dir_unlocked(
            model_name_or_path, cache_dir, allow_patterns, revision
        )
        if path is not None:

            return path


        if not huggingface_hub.constants.HF_HUB_OFFLINE:

            fs = HfFileSystem()
            file_list = fs.ls(model_name_or_path, detail=False, revision=revision)


            for pattern in allow_patterns:
                matching = fnmatch.filter(file_list, pattern)
                if len(matching) > 0:
                    allow_patterns = [pattern]
                    break

        log_info_on_rank0(logger, f"Using model weights format {allow_patterns}")

        if not is_in_ci():

            hf_folder = snapshot_download(
                model_name_or_path,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                cache_dir=cache_dir,
                tqdm_class=DisabledTqdm,
                revision=revision,
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
            )
            return hf_folder
        else:

            return ci_download_with_validation_and_retry(
                model_name_or_path=model_name_or_path,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                cache_dir=cache_dir,
                revision=revision,
                max_retries=max_retries,
            )


def download_safetensors_index_file_from_hf(
    model_name_or_path: str,
    index_file: str,
    cache_dir: Optional[str],
    revision: Optional[str] = None,
) -> None:


    with get_lock(model_name_or_path, cache_dir):
        try:

            hf_hub_download(
                repo_id=model_name_or_path,
                filename=index_file,
                cache_dir=cache_dir,
                revision=revision,
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
            )


        except huggingface_hub.utils.EntryNotFoundError:
            logger.debug("No %s found in remote.", index_file)
        except huggingface_hub.utils.LocalEntryNotFoundError:
            logger.debug("No %s found in local cache.", index_file)


def filter_duplicate_safetensors_files(
    hf_weights_files: List[str], hf_folder: str, index_file: str
) -> List[str]:


    index_file_name = os.path.join(hf_folder, index_file)
    if not os.path.isfile(index_file_name):


        if len(hf_weights_files) == 2:
            hf_weights_files.sort()
            if hf_weights_files[0].endswith(
                "consolidated.safetensors"
            ) and hf_weights_files[1].endswith("model.safetensors"):
                return [hf_weights_files[1]]
        return hf_weights_files


    with open(index_file_name) as f:
        weight_map = json.load(f)["weight_map"]
    weight_files_in_index = set()
    for weight_name in weight_map:
        weight_files_in_index.add(os.path.join(hf_folder, weight_map[weight_name]))


    missing_files = sorted(f for f in weight_files_in_index if not os.path.isfile(f))
    if missing_files:
        raise RuntimeError(
            f"{index_file} references {len(missing_files)} shard file(s) missing "
            f"from {hf_folder} (incomplete download?): "
            f"{[os.path.basename(f) for f in missing_files]}"
        )

    hf_weights_files = [f for f in hf_weights_files if f in weight_files_in_index]
    return hf_weights_files


def maybe_add_mtp_safetensors(
    hf_weights_files: List[str], hf_folder: str, index_file: str, hf_config
) -> List[str]:


    arch = getattr(hf_config, "architectures", [None])[0]
    num_nextn_layers = getattr(
        getattr(hf_config, "text_config", hf_config),
        "num_nextn_predict_layers",
        getattr(hf_config, "num_nextn_predict_layers", 0),
    )
    if not (
        arch
        in [
            "Glm4MoeForCausalLM",
            "Glm4MoeForCausalLMNextN",
            "Glm4MoeLiteForCausalLM",
            "Glm4MoeLiteForCausalLMNextN",
        ]
        and num_nextn_layers > 0
    ):
        return hf_weights_files


    mtp_path = os.path.join(hf_folder, "mtp.safetensors")
    if not os.path.isfile(mtp_path) or mtp_path in hf_weights_files:
        return hf_weights_files


    logger.warning(
        f"Found mtp.safetensors but it's not referenced in {index_file}. "
        f"This is a checkpoint packaging bug. Auto-adding it for loading. "
        f"Please report this to the checkpoint provider."
    )


    return hf_weights_files + [mtp_path]


def filter_files_not_needed_for_inference(hf_weights_files: List[str]) -> List[str]:

    blacklist = [
        "training_args.bin",
        "optimizer.bin",
        "optimizer.pt",
        "scheduler.pt",
        "scaler.pt",
    ]
    hf_weights_files = [
        f for f in hf_weights_files if not any(f.endswith(x) for x in blacklist)
    ]
    return hf_weights_files


def np_cache_weights_iterator(
    model_name_or_path: str,
    cache_dir: Optional[str],
    hf_folder: str,
    hf_weights_files: List[str],
) -> Generator[Tuple[str, torch.Tensor], None, None]:

    enable_tqdm = (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )


    np_folder = os.path.join(hf_folder, "np")
    os.makedirs(np_folder, exist_ok=True)
    weight_names_file = os.path.join(np_folder, "weight_names.json")


    with get_lock(model_name_or_path, cache_dir):
        if not os.path.exists(weight_names_file):
            weight_names: List[str] = []
            for bin_file in tqdm(
                hf_weights_files,
                desc="Loading np_cache checkpoint shards",
                disable=not enable_tqdm,
                bar_format=BAR_FORMAT,
                position=tqdm._get_free_pos(),
            ):
                state = torch.load(bin_file, map_location="cpu", weights_only=True)
                for name, param in state.items():
                    param_path = os.path.join(np_folder, name)
                    with open(param_path, "wb") as f:
                        np.save(f, param.cpu().detach().numpy())
                    weight_names.append(name)
            with open(weight_names_file, "w") as f:
                json.dump(weight_names, f)

    with open(weight_names_file) as f:
        weight_names = json.load(f)

    for name in weight_names:
        param_path = os.path.join(np_folder, name)
        with open(param_path, "rb") as f:
            param = np.load(f)
        yield name, torch.from_numpy(param)


def _prefetch_checkpoint_file(
    file_path: str,
    cancel_event: Optional[threading.Event] = None,
) -> None:

    with open(file_path, "rb") as f:
        while cancel_event is None or not cancel_event.is_set():
            if not f.read(_get_prefetch_block_size()):
                break


class CheckpointFilePrefetchHandle:


    def __init__(
        self,
        *,
        thread: threading.Thread,
        cancel_event: threading.Event,
        succeeded_event: threading.Event,
        errors: List[Tuple[str, Exception]],
    ) -> None:
        self._thread = thread
        self._cancel_event = cancel_event
        self._succeeded_event = succeeded_event
        self._errors = errors

    def wait(self, timeout: Optional[float] = None) -> None:
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError("Timed out waiting for checkpoint prefetching")

    def cancel(self) -> None:

        self._cancel_event.set()

    def stop(self, timeout: Optional[float] = _PREFETCH_STOP_TIMEOUT_SECONDS) -> None:

        self.cancel()
        self.wait(timeout)

    @property
    def done(self) -> bool:
        return not self._thread.is_alive()

    @property
    def failed(self) -> bool:
        return bool(self._errors) or (self.done and not self._succeeded_event.is_set())

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set()

    @property
    def errors(self) -> Tuple[Tuple[str, Exception], ...]:
        return tuple(self._errors)


def _prefetch_all_checkpoints(
    sorted_files: List[str],
    num_threads: int = 4,
) -> CheckpointFilePrefetchHandle:

    import time

    if num_threads < 1:
        raise ValueError("weight loader prefetch num_threads must be >= 1")


    if torch.distributed.is_initialized():
        world_group = get_world_group()
        local_rank = world_group.local_rank
        local_world_size = world_group.local_size or world_group.world_size
    else:
        local_rank = 0
        local_world_size = 1

    my_files = sorted_files[local_rank::local_world_size]
    total_for_rank = len(my_files)
    cancel_event = threading.Event()
    succeeded_event = threading.Event()
    errors: List[Tuple[str, Exception]] = []

    logger.info(
        "Rank %d: prefetching %d/%d checkpoint shards into page cache "
        "(background, %d local ranks sharing the work, %d threads per rank)...",
        local_rank,
        total_for_rank,
        len(sorted_files),
        local_world_size,
        num_threads,
    )

    def _prefetch_all() -> None:
        completed = 0
        next_log_pct = 10

        def record_complete() -> None:
            nonlocal completed, next_log_pct

            completed += 1
            if total_for_rank > 0 and next_log_pct <= 100:
                pct = 100 * completed / total_for_rank
                while pct >= next_log_pct and next_log_pct <= 100:
                    logger.info(
                        "Rank %d: prefetching checkpoint files: %d%% (%d/%d)",
                        local_rank,
                        next_log_pct,
                        completed,
                        total_for_rank,
                    )
                    next_log_pct += 10

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            file_iter = iter(my_files)
            pending: Dict[concurrent.futures.Future, str] = {}

            for path in itertools.islice(file_iter, num_threads):
                if cancel_event.is_set():
                    break
                pending[
                    executor.submit(_prefetch_checkpoint_file, path, cancel_event)
                ] = path

            while pending:
                done, _ = concurrent.futures.wait(
                    pending,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    path = pending.pop(future)
                    exc = future.exception()
                    if exc is not None:
                        errors.append((path, exc))
                        logger.warning(
                            "Failed to prefetch checkpoint file %r: %s",
                            path,
                            exc,
                        )
                    record_complete()

                    next_path = None if cancel_event.is_set() else next(file_iter, None)
                    if next_path is not None:
                        pending[
                            executor.submit(
                                _prefetch_checkpoint_file,
                                next_path,
                                cancel_event,
                            )
                        ] = next_path

    def _run_prefetch() -> None:
        start = time.perf_counter()
        _prefetch_all()
        succeeded_event.set()
        logger.info(
            "Rank %d: prefetching checkpoint files into page cache "
            "finished in %.2fs",
            local_rank,
            time.perf_counter() - start,
        )

    thread = threading.Thread(target=_run_prefetch, daemon=True)
    handle = CheckpointFilePrefetchHandle(
        thread=thread,
        cancel_event=cancel_event,
        succeeded_event=succeeded_event,
        errors=errors,
    )
    thread.start()
    return handle


def _drop_file_cache_after_load(path: str) -> None:

    posix_fadvise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if posix_fadvise is None or dontneed is None:
        return

    fd = None
    try:
        fd = os.open(path, os.O_RDONLY)
        posix_fadvise(fd, 0, 0, dontneed)
    except OSError as e:
        logger.debug("Failed to drop file cache for %s: %s", path, e)
    finally:
        if fd is not None:
            os.close(fd)


def safetensors_weights_iterator(
    hf_weights_files: List[str],
    disable_mmap: bool = False,
    prefetch: bool = False,
    prefetch_num_threads: int = 4,
    drop_cache_after_load: bool = False,
) -> Generator[Tuple[str, torch.Tensor], None, None]:

    enable_tqdm = (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )

    if prefetch and not disable_mmap:
        _prefetch_all_checkpoints(
            sorted(hf_weights_files), num_threads=prefetch_num_threads
        )

    for st_file in tqdm(
        hf_weights_files,
        desc="Loading safetensors checkpoint shards",
        disable=not enable_tqdm,
        bar_format=BAR_FORMAT,
        position=tqdm._get_free_pos(),
    ):
        if disable_mmap:
            with open(st_file, "rb") as f:
                result = safetensors.torch.load(f.read())
                for name in sorted(result.keys()):
                    yield name, result[name]
        else:
            with safetensors.safe_open(st_file, framework="pt", device="cpu") as f:
                for name in f.keys():
                    yield name, f.get_tensor(name)
        if drop_cache_after_load:
            _drop_file_cache_after_load(st_file)


def fastsafetensors_weights_iterator(
    hf_weights_files: List[str],
    enable_gds: bool = True,
    drop_cache_after_load: bool = False,
) -> Generator[Tuple[str, torch.Tensor], None, None]:

    if SafeTensorsFileLoader is None:
        raise ImportError(
            "Please install fastsafetensors via `pip install fastsafetensors`"
        )

    if torch.distributed.is_initialized():
        pg = torch.distributed.group.WORLD
    else:
        pg = SingleGroup()

    try:
        rank = pg.rank()
    except Exception:
        rank = 0

    device = torch.device(f"cuda:{rank}")

    weight_files_sub_lists = [
        hf_weights_files[i : i + pg.size()]
        for i in range(0, len(hf_weights_files), pg.size())
    ]

    _BAR_FORMAT = (
        "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    for f_list in tqdm(
        weight_files_sub_lists,
        desc="Loading safetensors using Fastsafetensor loader",
        disable=False,
        bar_format=_BAR_FORMAT,
    ):
        loader = SafeTensorsFileLoader(pg, device, nogds=not enable_gds)
        rank_file_map = {i: [f] for i, f in enumerate(f_list)}
        loader.add_filenames(rank_file_map)
        try:
            fb = loader.copy_files_to_device()
            try:
                keys = list(fb.key_to_rank_lidx.keys())
                for k in keys:
                    t = fb.get_tensor(k)
                    yield k, t
            finally:
                pass
        finally:
            loader.close()
        if drop_cache_after_load:
            for loaded_file in rank_file_map.get(rank, []):
                _drop_file_cache_after_load(loaded_file)


def multi_thread_safetensors_weights_iterator(
    hf_weights_files: List[str],
    max_workers: int,
    disable_mmap: bool = False,
    drop_cache_after_load: bool = False,
) -> Generator[Tuple[str, torch.Tensor], None, None]:

    enable_tqdm = (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )

    def _load_file(st_file: str):
        if disable_mmap:
            with open(st_file, "rb") as f:
                result = safetensors.torch.load(f.read())
        else:
            with safetensors.safe_open(st_file, framework="pt", device="cpu") as f:
                result = {k: f.get_tensor(k) for k in f.keys()}

        return st_file, result

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_load_file, st_file) for st_file in hf_weights_files]

        if enable_tqdm:
            futures_iter = tqdm(
                concurrent.futures.as_completed(futures),
                total=len(hf_weights_files),
                desc="Multi-thread loading shards",
                disable=not enable_tqdm,
                bar_format=BAR_FORMAT,
            )
        else:
            futures_iter = concurrent.futures.as_completed(futures)

        for future in futures_iter:
            st_file, state_dict = future.result()
            for name, param in state_dict.items():
                yield name, param
            del state_dict
            if drop_cache_after_load:
                _drop_file_cache_after_load(st_file)


def buffered_multi_thread_safetensors_weights_iterator(
    hf_weights_files: List[str],
    max_workers: int,
    disable_mmap: bool = False,
    prefetch: bool = False,
    prefetch_num_threads: int = 4,
    drop_cache_after_load: bool = False,
) -> Generator[Tuple[str, torch.Tensor], None, None]:

    if prefetch and not disable_mmap:
        _prefetch_all_checkpoints(
            sorted(hf_weights_files), num_threads=prefetch_num_threads
        )
    enable_tqdm = (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )

    def _load_file(st_file: str):
        if disable_mmap:
            with open(st_file, "rb") as f:
                result = safetensors.torch.load(f.read())
        else:
            with safetensors.safe_open(st_file, framework="pt", device="cpu") as f:
                result = {k: f.get_tensor(k) for k in f.keys()}
        return result


    buffer_size = max_workers + 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        file_iter = iter(hf_weights_files)
        pending: collections.deque = collections.deque()


        for st_file in itertools.islice(file_iter, buffer_size):
            pending.append((st_file, executor.submit(_load_file, st_file)))

        with tqdm(
            total=len(hf_weights_files),
            desc="Multi-thread loading shards",
            disable=not enable_tqdm,
            bar_format=BAR_FORMAT,
            position=tqdm._get_free_pos(),
        ) as pbar:
            while pending:
                st_file, future = pending.popleft()
                state_dict = future.result()
                del future


                next_file = next(file_iter, None)
                if next_file is not None:
                    pending.append((next_file, executor.submit(_load_file, next_file)))

                for name in sorted(state_dict.keys()):
                    yield name, state_dict[name]
                del state_dict
                if drop_cache_after_load:


                    _drop_file_cache_after_load(st_file)
                pbar.update(1)


def _load_pt_file(bin_file: str) -> dict:

    try:
        return torch.load(bin_file, map_location="cpu", weights_only=True)
    except RuntimeError as e:
        if "legacy .tar format" in str(e):
            logger.warning(
                "Loading %s with weights_only=False (legacy tar format)",
                os.path.basename(bin_file),
            )
            return torch.load(bin_file, map_location="cpu", weights_only=False)
        raise


def pt_weights_iterator(
    hf_weights_files: List[str],
) -> Generator[Tuple[str, torch.Tensor], None, None]:

    enable_tqdm = (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )
    for bin_file in tqdm(
        hf_weights_files,
        desc="Loading pt checkpoint shards",
        disable=not enable_tqdm,
        bar_format=BAR_FORMAT,
        position=tqdm._get_free_pos(),
    ):
        state = _load_pt_file(bin_file)
        yield from state.items()
        del state


def multi_thread_pt_weights_iterator(
    hf_weights_files: List[str],
    max_workers: int,
) -> Generator[Tuple[str, torch.Tensor], None, None]:

    enable_tqdm = (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_load_pt_file, bin_file) for bin_file in hf_weights_files
        ]

        if enable_tqdm:
            futures_iter = tqdm(
                concurrent.futures.as_completed(futures),
                total=len(hf_weights_files),
                desc="Multi-thread loading pt checkpoint shards",
                disable=not enable_tqdm,
                bar_format=BAR_FORMAT,
            )
        else:
            futures_iter = concurrent.futures.as_completed(futures)

        for future in futures_iter:
            state = future.result()
            yield from state.items()


def get_gguf_extra_tensor_names(
    gguf_file: str, gguf_to_hf_name_map: Dict[str, str]
) -> List[str]:
    import gguf

    reader = gguf.GGUFReader(gguf_file)
    expected_gguf_keys = set(gguf_to_hf_name_map.keys())
    exact_gguf_keys = set([tensor.name for tensor in reader.tensors])
    extra_keys = expected_gguf_keys - exact_gguf_keys
    return [gguf_to_hf_name_map[key] for key in extra_keys]


def gguf_quant_weights_iterator(
    gguf_file: str, gguf_to_hf_name_map: Dict[str, str]
) -> Generator[Tuple[str, torch.Tensor], None, None]:


    import gguf

    reader = gguf.GGUFReader(gguf_file)


    MOE_WEIGHT_PATTERNS = {
        "ffn_gate_exps": "gate_proj",
        "ffn_up_exps": "up_proj",
        "ffn_down_exps": "down_proj",
    }


    for tensor in reader.tensors:
        weight_type = tensor.tensor_type
        tensor_name = tensor.name


        is_moe_weight = any(
            pattern in tensor_name for pattern in MOE_WEIGHT_PATTERNS.keys()
        )

        if is_moe_weight:


            import re

            match = re.match(r"blk\.(\d+)\.(ffn_\w+_exps)\.weight", tensor_name)
            if match:
                layer_id = int(match.group(1))
                weight_pattern = match.group(2)
                hf_weight_name = MOE_WEIGHT_PATTERNS.get(weight_pattern)

                if hf_weight_name and weight_type.name != "F32":

                    weight = tensor.data
                    num_experts = weight.shape[0]
                    for expert_id in range(num_experts):
                        hf_name = f"model.layers.{layer_id}.mlp.experts.{expert_id}.{hf_weight_name}.qweight_type"
                        yield hf_name, torch.tensor(weight_type)
        elif tensor_name in gguf_to_hf_name_map:

            name = gguf_to_hf_name_map[tensor_name]

            if weight_type.name != "F32":
                weight_type_name = name.replace("weight", "qweight_type")
                yield weight_type_name, torch.tensor(weight_type)


    for tensor in reader.tensors:
        weight = tensor.data
        weight_type = tensor.tensor_type
        tensor_name = tensor.name


        is_moe_weight = any(
            pattern in tensor_name for pattern in MOE_WEIGHT_PATTERNS.keys()
        )

        if is_moe_weight:

            import re

            match = re.match(r"blk\.(\d+)\.(ffn_\w+_exps)\.weight", tensor_name)
            if match:
                layer_id = int(match.group(1))
                weight_pattern = match.group(2)
                hf_weight_name = MOE_WEIGHT_PATTERNS.get(weight_pattern)

                if hf_weight_name:

                    num_experts = weight.shape[0]
                    for expert_id in range(num_experts):
                        expert_weight = weight[expert_id]

                        if weight_type.name != "F32":
                            hf_name = f"model.layers.{layer_id}.mlp.experts.{expert_id}.{hf_weight_name}.qweight"
                        else:
                            hf_name = f"model.layers.{layer_id}.mlp.experts.{expert_id}.{hf_weight_name}.weight"

                        yield hf_name, torch.tensor(expert_weight)
        elif tensor_name in gguf_to_hf_name_map:

            name = gguf_to_hf_name_map[tensor_name]

            if weight_type.name != "F32":
                name = name.replace("weight", "qweight")
            param = torch.tensor(weight)
            yield name, param


def convert_pyslice_to_tensor(x: Any) -> torch.Tensor:

    if not isinstance(x, torch.Tensor):
        x = x[:]
    return x


def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:

    try:
        if param.numel() == 1 and loaded_weight.numel() == 1:


            param.data.fill_(loaded_weight.item())
        else:
            assert param.size() == loaded_weight.size(), (
                f"Attempted to load weight ({loaded_weight.size()}) "
                f"into parameter ({param.size()})"
            )

            param.data.copy_(loaded_weight)
    except Exception:


        raise


def row_parallel_weight_loader(
    param: torch.Tensor, loaded_weight: torch.Tensor
) -> None:

    tp_rank = get_parallel().tp_rank
    shard_dim = 0 if param.dim() != 1 else None

    if shard_dim is not None:
        shard_size = param.data.shape[shard_dim]
        start_idx = tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(shard_dim, start_idx, shard_size)

    return default_weight_loader(param, loaded_weight)


LoaderFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def sharded_weight_loader(shard_axis: int) -> LoaderFunction:


    def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        tp_rank = get_parallel().attn_tp_rank

        shard_size = param.data.shape[shard_axis]
        start_idx = tp_rank * shard_size

        if (
            is_cpu()
            and (
                loaded_weight.size(0) % get_parallel().tp_size != 0
                or loaded_weight.size(0) < get_parallel().tp_size * shard_size
            )
            and loaded_weight.dim() == 1
        ):
            param_data = param.data
            param_data, loaded_weight = narrow_padded_param_and_loaded_weight(
                param_data,
                loaded_weight,
                0,
                start_idx,
                shard_axis,
                shard_size,
            )
            return default_weight_loader(param_data, loaded_weight)
        else:
            loaded_weight = loaded_weight.narrow(shard_axis, start_idx, shard_size)
            return default_weight_loader(param, loaded_weight)

    return loader


def composed_weight_loader(
    loader: LoaderFunction, fn: Callable[[torch.Tensor], torch.Tensor]
) -> LoaderFunction:


    def composed_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        loader(param, loaded_weight)
        param.data.copy_(fn(param))
        return

    return composed_loader


def runai_safetensors_weights_iterator(
    hf_weights_files: List[str], is_distributed: bool = False, device: str = "cpu"
) -> Generator[Tuple[str, torch.Tensor], None, None]:

    from runai_model_streamer import SafetensorsStreamer

    enable_tqdm = (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )
    device = device if is_distributed and is_cuda_alike() else "cpu"

    with SafetensorsStreamer() as streamer:

        streamer.stream_files(
            hf_weights_files,
            device=device,
            is_distributed=is_distributed,
        )
        total_tensors = sum(
            len(tensors_meta)
            for tensors_meta in streamer.files_to_tensors_metadata.values()
        )

        tensor_iter = tqdm(
            streamer.get_tensors(),
            total=total_tensors,
            desc="Loading safetensors using Runai Model Streamer",
            bar_format=BAR_FORMAT,
            disable=not enable_tqdm,
            mininterval=2,
        )

        for name, tensor in tensor_iter:
            setattr(tensor, RUNAI_STREAMER_TENSOR_ATTR, True)
            yield name, tensor


def set_runai_streamer_env(load_config: LoadConfig):
    if load_config.model_loader_extra_config:
        extra_config = load_config.model_loader_extra_config

        if "concurrency" in extra_config and isinstance(
            extra_config.get("concurrency"), int
        ):
            os.environ["RUNAI_STREAMER_CONCURRENCY"] = str(
                extra_config.get("concurrency")
            )

        if "memory_limit" in extra_config and isinstance(
            extra_config.get("memory_limit"), int
        ):
            os.environ["RUNAI_STREAMER_MEMORY_LIMIT"] = str(
                extra_config.get("memory_limit")
            )

    runai_streamer_s3_endpoint = os.getenv("RUNAI_STREAMER_S3_ENDPOINT")
    aws_endpoint_url = os.getenv("AWS_ENDPOINT_URL")
    if runai_streamer_s3_endpoint is None and aws_endpoint_url is not None:
        os.environ["RUNAI_STREAMER_S3_ENDPOINT"] = aws_endpoint_url


@torch.no_grad()
def initialize_capture_safe_weights(
    model: torch.nn.Module,
    value: float = CAPTURE_SAFE_WEIGHT_SENTINEL,
) -> None:

    for param in model.parameters():
        if torch.is_floating_point(param):
            param.fill_(value)


def initialize_dummy_weights(
    model: torch.nn.Module,
    low: float = -1e-3,
    high: float = 1e-3,
    seed: int = 1234,
) -> None:

    for param in model.state_dict().values():
        if torch.is_floating_point(param):
            generator = torch.Generator(device=param.data.device)
            generator.manual_seed(seed)


            if torch.finfo(param.dtype).bits < 16:

                dtype = param.data.dtype
                tmp_param = param.data.to(torch.float16)
                tmp_param = tmp_param.uniform_(low, high, generator=generator).to(dtype)
                param.data.copy_(tmp_param)
            else:
                param.uniform_(low, high, generator=generator)


def maybe_remap_kv_scale_name(name: str, params_dict: dict) -> Optional[str]:

    if name.endswith(".kv_scale"):
        print_warning_once(
            "DEPRECATED. Found kv_scale in the checkpoint. "
            "This format is deprecated in favor of separate k_scale and "
            "v_scale tensors and will be removed in a future release. "
            "Functionally, we will remap kv_scale to k_scale and duplicate "
            "k_scale to v_scale"
        )

        remapped_name = name.replace(".kv_scale", ".attn.k_scale")
        if remapped_name not in params_dict:
            print_warning_once(
                f"Found kv_scale in the checkpoint (e.g. {name}), "
                "but not found the expected name in the model "
                f"(e.g. {remapped_name}). kv_scale is "
                "not loaded."
            )
            return None
        return remapped_name

    possible_scale_names = [".k_scale", ".v_scale"]


    modelopt_attn_prefixes = [".self_attn.", ".mixer."]
    for scale_name in possible_scale_names:
        if name.endswith(scale_name):

            matched_prefix = None
            for attn_prefix in modelopt_attn_prefixes:
                if f"{attn_prefix}{scale_name[1]}_proj{scale_name}" in name:
                    matched_prefix = attn_prefix
                    break

            if matched_prefix is not None:
                remapped_name = name.replace(
                    f"{matched_prefix}{scale_name[1]}_proj{scale_name}",
                    f"{matched_prefix}attn{scale_name}",
                )
            else:
                remapped_name = name.replace(scale_name, f".attn{scale_name}")
            if remapped_name not in params_dict:
                print_warning_once(
                    f"Found {scale_name} in the checkpoint (e.g. {name}), "
                    "but not found the expected name in the model "
                    f"(e.g. {remapped_name}). {scale_name} is "
                    "not loaded."
                )
                return None
            return remapped_name

    quark_scale_names = {
        ".q_proj.output_scale": ".attn.q_scale",
        ".k_proj.output_scale": ".attn.k_scale",
        ".v_proj.output_scale": ".attn.v_scale",
        "self_attn.prob_output_scale": ".attn.prob_scale",
    }
    for quark_scale_name, sglang_scale_name in quark_scale_names.items():
        if name.endswith(quark_scale_name):
            return name.replace(quark_scale_name, sglang_scale_name)


    return name


class KVCacheQuantSchema(BaseModel):
    dtype: str


    scaling_factor: Dict[int, Dict[int, float]]

    @model_validator(mode="after")
    def check_is_fp8(self) -> "KVCacheQuantSchema":
        assert self.dtype == "float8_e4m3fn", (
            "Loaded scaling factors intended for KV cache dtype = "
            f"{self.dtype} rather than float8_e4m3fn!"
        )
        return self

    @model_validator(mode="after")
    def check_tp_ranks(self, info: ValidationInfo) -> "KVCacheQuantSchema":
        context = info.context
        if context:
            tp_size = context["tp_size"]
            num_hidden_layers = context["num_hidden_layers"]
            assert len(self.scaling_factor) == tp_size, (
                f"Loaded dictionary has TP size {len(self.scaling_factor)} "
                f"but LLM engine is currently running with TP size {tp_size}."
            )
            for tp_rank, layer_maps in self.scaling_factor.items():
                assert len(layer_maps) == num_hidden_layers, (
                    f"KV cache scales map for TP rank {tp_rank} is malformed. "
                    f"Expected {num_hidden_layers} layers, got "
                    f"{len(layer_maps)}."
                )
            for i in range(tp_size):
                assert (
                    i in self.scaling_factor
                ), f"KV cache scales map for TP rank {i} not found."
        return self

    @model_validator(mode="after")
    def check_current_rank(self, info: ValidationInfo) -> "KVCacheQuantSchema":
        context = info.context
        if context:
            tp_rank = context["tp_rank"]
            num_hidden_layers = context["num_hidden_layers"]
            layer_scales_map = self.scaling_factor[tp_rank]
            for i in range(num_hidden_layers):
                assert i in layer_scales_map, (
                    f"Could not find KV cache scales for layer {i} in "
                    f"TP rank {tp_rank}."
                )
        return self


class QuantParamSchema(BaseModel):


    model_config = ConfigDict(protected_namespaces=())
    model_type: Optional[str]
    kv_cache: KVCacheQuantSchema

    @model_validator(mode="after")
    def check_model_type(self, info: ValidationInfo) -> "QuantParamSchema":
        context = info.context
        if context:
            model_type = context.get("model_type", None)
            if model_type is not None:
                assert model_type == self.model_type, (
                    f"Model type is {model_type} but loaded "
                    f"scaling factors belonging to different "
                    f"model type {self.model_type}!"
                )
        return self


def kv_cache_scales_loader(
    filename: str,
    tp_rank: int,
    tp_size: int,
    num_hidden_layers: int,
    model_type: Optional[str],
) -> Iterable[Tuple[int, float]]:

    try:
        with open(filename) as f:
            context = {
                "model_type": model_type,
                "num_hidden_layers": num_hidden_layers,
                "tp_rank": tp_rank,
                "tp_size": tp_size,
            }
            schema_dct = json.load(f)
            schema = QuantParamSchema.model_validate(schema_dct, context=context)
            layer_scales_map = schema.kv_cache.scaling_factor[tp_rank]
            return layer_scales_map.items()
    except FileNotFoundError:
        logger.error("File or directory '%s' not found.", filename)
    except json.JSONDecodeError:
        logger.error("Error decoding JSON in file '%s'.", filename)
    except Exception:
        logger.error("An error occurred while reading '%s'.", filename)


    logger.warning(
        "Defaulting to KV cache scaling factors = 1.0 for all "
        "layers in TP rank %d as an error occurred during loading.",
        tp_rank,
    )
    return []


def get_actual_shard_size(shard_size, weight_start, weight_end):
    if weight_end < weight_start:
        return 0

    return min(shard_size, weight_end - weight_start)


def reset_param_data_if_needed(param_data, dim, start, length):
    if length == 0:
        return

    assert length > 0, f"Length should be positive, but got {length}"

    param_data.narrow(dim, start, length).zero_()
    return


def narrow_padded_param_and_loaded_weight(
    param_data,
    loaded_weight,
    param_data_start,
    weight_start,
    dim,
    shard_size,
    narrow_weight=True,
):
    actual_shard_size = get_actual_shard_size(
        shard_size, weight_start, loaded_weight.size(dim)
    )

    if narrow_weight:
        if actual_shard_size > 0:
            loaded_weight = loaded_weight.narrow(dim, weight_start, actual_shard_size)
        else:

            loaded_weight = torch.zeros_like(
                param_data.narrow(dim, param_data_start, actual_shard_size)
            )


    reset_param_data_if_needed(
        param_data,
        dim,
        param_data_start + actual_shard_size,
        shard_size - actual_shard_size,
    )

    param_data = param_data.narrow(dim, param_data_start, actual_shard_size)

    return param_data, loaded_weight


def pad_loaded_weight(loaded_weight, output_dim, output_sizes):


    total_output_size = sum(output_sizes)
    raw_output_size = loaded_weight.size(output_dim)
    if total_output_size > raw_output_size:
        loaded_weight_pad = []
        weight_split_size = [
            int(output_size / total_output_size * raw_output_size)
            for output_size in output_sizes
        ]
        assert (
            sum(weight_split_size) == raw_output_size
        ), f"Padding the loaded weight failed due to sizes are not divisible cleanly from {output_sizes} to {raw_output_size}"

        split_weight = loaded_weight.split_with_sizes(weight_split_size, dim=output_dim)
        for i, output_size in enumerate(output_sizes):
            pad_size = output_size - weight_split_size[i]
            target_pad_shape = list(loaded_weight.size())
            target_pad_shape[output_dim] = pad_size
            pad_tensor = torch.zeros(target_pad_shape).to(loaded_weight.dtype)
            loaded_weight_pad.append(
                torch.cat([split_weight[i], pad_tensor], dim=output_dim)
            )
        return torch.cat(loaded_weight_pad, dim=output_dim)
    else:
        return loaded_weight
