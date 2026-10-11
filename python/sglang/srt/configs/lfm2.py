# coding=utf-8
# Copyright 2024 Liquid AI and the HuggingFace Inc. team. All rights reserved.
#
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


from typing import List, Optional

from transformers import CONFIG_MAPPING
from transformers import Lfm2Config as HFLfm2Config
from transformers.utils import logging

from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateShape,
    mamba2_state_dtype,
)
from sglang.srt.runtime_context import get_parallel

logger = logging.get_logger(__name__)


class Lfm2Config(HFLfm2Config):


    @property
    def full_attention_layer_ids(self) -> List[int]:

        return [i for i, lt in enumerate(self.layer_types) if lt == "full_attention"]

    @property
    def linear_layer_ids(self) -> List[int]:

        return [
            i for i, lt in enumerate(self.layer_types) if lt in ("conv", "short_conv")
        ]

    @property
    def mamba_chunk_size(self) -> int:

        return 1

    @property
    def mamba2_cache_params(self) -> Optional[Mamba2CacheParams]:


        conv_layer_ids = self.linear_layer_ids
        if not conv_layer_ids:
            return None

        hidden_size = self.hidden_size
        conv_kernel = int(self.conv_L_cache)


        try:
            tp_size = get_parallel().attn_tp_size
        except (AssertionError, RuntimeError):
            tp_size = 1


        shape = Mamba2StateShape.create(
            tp_world_size=tp_size,
            intermediate_size=hidden_size,
            n_groups=1,
            num_heads=tp_size,
            head_dim=hidden_size,
            state_size=0,
            conv_kernel=conv_kernel,
        )

        return Mamba2CacheParams(
            shape=shape,
            layers=conv_layer_ids,
            dtype=mamba2_state_dtype(self),
        )


CONFIG_MAPPING._extra_content["lfm2"] = Lfm2Config
