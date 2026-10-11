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


def list_files(
    s3,
    path: str,
    allow_pattern: Optional[list[str]] = None,
    ignore_pattern: Optional[list[str]] = None,
) -> tuple[str, str, list[str]]:

    parts = path.removeprefix("s3://").split("/")
    prefix = "/".join(parts[1:])
    bucket_name = parts[0]

    objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    paths = [obj["Key"] for obj in objects.get("Contents", [])]

    paths = _filter_ignore(paths, ["*/"])
    if allow_pattern is not None:
        paths = _filter_allow(paths, allow_pattern)

    if ignore_pattern is not None:
        paths = _filter_ignore(paths, ignore_pattern)

    return bucket_name, prefix, paths


class S3Connector(BaseFileConnector):

    def __init__(self, url: str) -> None:
        import boto3

        super().__init__(url)
        self.client = boto3.client("s3")

    def glob(self, allow_pattern: Optional[list[str]] = None) -> list[str]:
        bucket_name, _, paths = list_files(
            self.client, path=self.url, allow_pattern=allow_pattern
        )
        return [f"s3://{bucket_name}/{path}" for path in paths]

    def pull_files(
        self,
        allow_pattern: Optional[list[str]] = None,
        ignore_pattern: Optional[list[str]] = None,
    ) -> None:

        bucket_name, base_dir, files = list_files(
            self.client, self.url, allow_pattern, ignore_pattern
        )
        if len(files) == 0:
            return

        for file in files:
            destination_file = os.path.join(self.local_dir, file.removeprefix(base_dir))
            local_dir = Path(destination_file).parent
            os.makedirs(local_dir, exist_ok=True)
            self.client.download_file(bucket_name, file, destination_file)

    def weight_iterator(
        self, rank: int = 0
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        from sglang.srt.model_loader.weight_utils import (
            runai_safetensors_weights_iterator,
        )


        hf_weights_files = self.glob(allow_pattern=["*.safetensors"])
        return runai_safetensors_weights_iterator(hf_weights_files)

    def close(self):
        self.client.close()
        super().close()
