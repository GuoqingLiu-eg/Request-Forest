# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/config.py
import enum
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

import orjson

from sglang.srt.configs.modelopt_config import ModelOptConfig
from sglang.srt.utils import is_hip

logger = logging.getLogger(__name__)


class LoadFormat(str, enum.Enum):
    AUTO = "auto"
    PT = "pt"
    SAFETENSORS = "safetensors"
    NPCACHE = "npcache"
    DUMMY = "dummy"
    SHARDED_STATE = "sharded_state"
    PRESHARDED = "presharded"
    GGUF = "gguf"
    EXPERT_PACK = "expert_pack"
    BITSANDBYTES = "bitsandbytes"
    MISTRAL = "mistral"
    LAYERED = "layered"
    FLASH_RL = "flash_rl"
    JAX = "jax"
    REMOTE = "remote"
    REMOTE_INSTANCE = "remote_instance"
    RDMA = "rdma"
    LOCAL_CACHED = "local_cached"
    FASTSAFETENSORS = "fastsafetensors"
    PRIVATE = "private"
    RUNAI_STREAMER = "runai_streamer"
    IPC_CACHE = "ipc_cache"


@dataclass
class LoadConfig:


    load_format: Union[str, LoadFormat] = LoadFormat.AUTO
    download_dir: Optional[str] = None
    model_loader_extra_config: Optional[Union[str, dict]] = field(default_factory=dict)
    ignore_patterns: Optional[Union[List[str], str]] = None
    decryption_key_file: Optional[str] = None
    decrypt_max_concurrency: int = -1
    tp_rank: Optional[int] = None
    remote_instance_weight_loader_seed_instance_ip: Optional[str] = None
    remote_instance_weight_loader_seed_instance_service_port: Optional[int] = None
    remote_instance_weight_loader_send_weights_group_ports: Optional[List[int]] = None
    remote_instance_weight_loader_backend: Optional[str] = None
    remote_instance_weight_loader_transfer_engine: Optional[Any] = None
    remote_instance_weight_loader_transfer_engine_session_id: Optional[str] = None
    modelexpress_url: Optional[str] = None
    modelexpress_transport: str = "nixl"


    modelopt_checkpoint_restore_path: Optional[str] = None
    modelopt_checkpoint_save_path: Optional[str] = None
    modelopt_export_path: Optional[str] = None


    modelopt_config: Optional[ModelOptConfig] = None


    inc_save_path: Optional[str] = None
    inc_tuning_iters: Optional[int] = 0
    inc_disable_opt_rtn: Optional[bool] = None


    rl_quant_profile: Optional[str] = (
        None
    )


    draft_model_idx: Optional[int] = None


    weight_cache_mode: str = "off"
    weight_cache_socket: Optional[str] = None
    fallback_load_format: Union[str, "LoadFormat"] = LoadFormat.AUTO

    def __post_init__(self):
        model_loader_extra_config = self.model_loader_extra_config or {}
        if isinstance(model_loader_extra_config, str):
            self.model_loader_extra_config = orjson.loads(model_loader_extra_config)
        self._verify_load_format()

        if self.ignore_patterns is not None and len(self.ignore_patterns) > 0:
            logger.info(
                "Ignoring the following patterns when downloading weights: %s",
                self.ignore_patterns,
            )
        else:
            self.ignore_patterns = ["original/**/*"]


        if self.modelopt_config is None:
            self.modelopt_config = ModelOptConfig(
                checkpoint_restore_path=self.modelopt_checkpoint_restore_path,
                checkpoint_save_path=self.modelopt_checkpoint_save_path,
                export_path=self.modelopt_export_path,
            )

    def _verify_load_format(self) -> None:
        if not isinstance(self.load_format, str):
            return

        load_format = self.load_format.lower()
        self.load_format = LoadFormat(load_format)

        rocm_not_supported_load_format: List[str] = []
        if is_hip() and load_format in rocm_not_supported_load_format:
            rocm_supported_load_format = [
                f
                for f in LoadFormat.__members__
                if (f not in rocm_not_supported_load_format)
            ]
            raise ValueError(
                f"load format '{load_format}' is not supported in ROCm. "
                f"Supported load formats are "
                f"{rocm_supported_load_format}"
            )
