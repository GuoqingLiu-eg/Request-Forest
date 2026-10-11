# SPDX-License-Identifier: Apache-2.0

import fnmatch
import os
from pathlib import Path
from typing import Generator, Optional, Tuple

import torch

from sglang.srt.connector import BaseFileConnector


def _filter_allow(paths: list[str], patterns: list[str]) -> list[str]:
    return [
        path
        for path in paths
        if any(fnmatch.fnmatch(path, pattern) for pattern in patterns)
    ]


def _filter_ignore(paths: list[str], patterns: list[str]) -> list[str]:
    return [
        path
        for path in paths
        if not any(fnmatch.fnmatch(path, pattern) for pattern in patterns)
    ]


def _normalize_url(url: str) -> str:

    return url.rstrip("/")


def list_files(
    bf,
    path: str,
    allow_pattern: Optional[list[str]] = None,
    ignore_pattern: Optional[list[str]] = None,
) -> Tuple[str, list[str]]:

    base_dir = _normalize_url(path)
    files = [p for p in bf.glob(base_dir + "/**") if not bf.isdir(p)]

    files = _filter_ignore(files, ["*/"])
    if allow_pattern is not None:
        files = _filter_allow(files, allow_pattern)
    if ignore_pattern is not None:
        files = _filter_ignore(files, ignore_pattern)

    return base_dir, files


class AzureBlobConnector(BaseFileConnector):


    def __init__(self, url: str) -> None:
        try:
            import blobfile as bf
        except ImportError as e:
            raise ImportError(
                "AzureBlobConnector requires the 'blobfile' package. "
                "Install it with `pip install blobfile`."
            ) from e

        super().__init__(url)
        self.bf = bf

    def glob(self, allow_pattern: Optional[list[str]] = None) -> list[str]:
        _, files = list_files(self.bf, self.url, allow_pattern=allow_pattern)
        return files

    def pull_files(
        self,
        allow_pattern: Optional[list[str]] = None,
        ignore_pattern: Optional[list[str]] = None,
    ) -> None:

        base_dir, files = list_files(self.bf, self.url, allow_pattern, ignore_pattern)
        if not files:
            return

        for file in files:
            relative = file[len(base_dir) :].lstrip("/")
            destination_file = os.path.join(self.local_dir, relative)
            os.makedirs(Path(destination_file).parent, exist_ok=True)
            self.bf.copy(file, destination_file, overwrite=True)

    def weight_iterator(
        self, rank: int = 0
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        from sglang.srt.model_loader.weight_utils import (
            runai_safetensors_weights_iterator,
        )


        self.pull_files(allow_pattern=["*.safetensors"])
        local_files = [
            os.path.join(root, f)
            for root, _, fs in os.walk(self.local_dir)
            for f in fs
            if f.endswith(".safetensors")
        ]
        return runai_safetensors_weights_iterator(local_files)

    def close(self):
        super().close()
