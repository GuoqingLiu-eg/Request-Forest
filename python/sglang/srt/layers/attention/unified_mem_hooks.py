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

from typing import Callable, Optional

import msgspec
import torch


class UnifiedMLAHooks(msgspec.Struct, frozen=True):


    v2p_page_table: Optional[torch.Tensor]

    translate_kv_loc_dense: Optional[Callable[..., torch.Tensor]]

    kernel_page_multiplier: int
    enabled: bool


_STATIC_POOL = UnifiedMLAHooks(
    v2p_page_table=None,
    translate_kv_loc_dense=None,
    kernel_page_multiplier=1,
    enabled=False,
)


def unified_mla_hooks(allocator) -> UnifiedMLAHooks:

    v2p = getattr(allocator, "full_v2p_page_table", None)
    if v2p is None:
        return _STATIC_POOL
    return UnifiedMLAHooks(
        v2p_page_table=v2p,
        translate_kv_loc_dense=getattr(allocator, "translate_kv_loc_dense", None),
        kernel_page_multiplier=getattr(allocator, "kernel_page_multiplier", 1),
        enabled=True,
    )
