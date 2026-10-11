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

from contextlib import contextmanager
from typing import Optional

import torch

from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)


is_capture_mode = False


_capture_lora_variant: Optional[str] = None


_capture_dsa_variant: Optional[str] = None


def get_is_capture_mode() -> bool:
    return is_capture_mode or is_in_breakable_cuda_graph()


def compile_in_capture_mode(func):

    if is_capture_mode:
        return torch.compile(func)
    return func


def get_capture_lora_variant() -> Optional[str]:

    return _capture_lora_variant


def _set_capture_lora_variant(variant: Optional[str]) -> None:
    global _capture_lora_variant
    _capture_lora_variant = variant


def get_capture_dsa_variant() -> Optional[str]:

    return _capture_dsa_variant


def _set_capture_dsa_variant(variant: Optional[str]) -> None:
    global _capture_dsa_variant
    _capture_dsa_variant = variant


@contextmanager
def model_capture_mode():
    global is_capture_mode
    from sglang.srt.runtime_context import get_flags


    is_capture_mode = True
    get_flags().capture.disable_dispose_tensor = True
    try:
        yield
    finally:
        is_capture_mode = False
        get_flags().capture.disable_dispose_tensor = False
