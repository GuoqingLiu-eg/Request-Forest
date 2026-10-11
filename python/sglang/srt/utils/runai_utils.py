# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/model_loader/runai_utils.py

import hashlib
import logging
import os
from pathlib import Path

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

SUPPORTED_SCHEMES = ["s3://", "gs://", "az://"]


def list_safetensors(path: str = "") -> list[str]:

    from runai_model_streamer import list_safetensors as runai_list_safetensors

    return runai_list_safetensors(path)


def is_runai_obj_uri(model_or_path: str | Path) -> bool:

    return str(model_or_path).lower().startswith(tuple(SUPPORTED_SCHEMES))


class ObjectStorageModel:


    def __init__(self, url: str) -> None:
        self.dir = ObjectStorageModel.get_path(url)

        from runai_model_streamer import ObjectStorageModel as RunaiObjectStorageModel

        self._runai_obj = RunaiObjectStorageModel(model_path=url, dst=self.dir)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._runai_obj.__exit__(exc_type, exc_val, exc_tb)

    def pull_files(
        self,
        allow_pattern: list[str] | None = None,
        ignore_pattern: list[str] | None = None,
    ) -> None:

        self._runai_obj.pull_files(allow_pattern, ignore_pattern)

    @classmethod
    def download_and_get_path(cls, model_path: str) -> str:

        with cls(url=model_path) as downloader:
            downloader.pull_files(
                ignore_pattern=[
                    "*.pt",
                    "*.safetensors",
                    "*.bin",
                    "*.tensors",
                    "*.pth",
                ],
            )
            cache_dir = downloader.dir
            logger.info(f"Runai Model : {cache_dir}, metadata ready.")
        return cache_dir

    @classmethod
    def get_path(cls, model_path: str) -> str:

        model_hash = hashlib.sha256(str(model_path).encode()).hexdigest()[:16]
        base_dir = envs.SGLANG_CACHE_DIR.get()


        os.makedirs(os.path.join(base_dir, "model_streamer"), exist_ok=True)

        return os.path.join(
            base_dir,
            "model_streamer",
            model_hash,
        )
