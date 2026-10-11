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


from sglang.kernels.ops.attention.dcp_kernels import (
    create_triton_kv_indices_for_dcp_triton,
)
from sglang.srt.layers.dcp.comm import (
    all_gather_kv_cache_for_dcp,
    all_gather_kv_cache_for_mha_chunk_extend,
    all_gather_kv_cache_for_mha_extend,
    all_gather_kv_cache_for_mla_extend,
    all_gather_q_for_mla_decode,
    cp_lse_ag_out_rs_mha,
    cp_lse_ag_out_rs_mla,
    dcp_a2a_lse_reduce,
    dcp_enabled,
    get_attention_dcp_rank,
    get_attention_dcp_world_size,
    init_fi_a2a_workspace,
)
from sglang.srt.layers.dcp.layout import (
    filter_dcp_local_kv_indices,
    get_dcp_lens,
    update_local_kv_lens_for_dcp,
)
from sglang.srt.layers.dcp.metadata import DecodeContextParallelMetadata


__all__ = [
    "DecodeContextParallelMetadata",
    "dcp_a2a_lse_reduce",
    "init_fi_a2a_workspace",
    "all_gather_kv_cache_for_dcp",
    "all_gather_kv_cache_for_mha_chunk_extend",
    "all_gather_kv_cache_for_mha_extend",
    "all_gather_kv_cache_for_mla_extend",
    "all_gather_q_for_mla_decode",
    "cp_lse_ag_out_rs_mha",
    "cp_lse_ag_out_rs_mla",
    "create_triton_kv_indices_for_dcp_triton",
    "dcp_enabled",
    "filter_dcp_local_kv_indices",
    "get_attention_dcp_rank",
    "get_attention_dcp_world_size",
    "get_dcp_lens",
    "update_local_kv_lens_for_dcp",
]
