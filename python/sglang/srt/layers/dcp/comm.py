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


import warnings
from typing import Optional

import torch

from sglang.kernels.ops.attention.dcp_kernels import (
    CPTritonContext,
    _lse_pack_dim,
    correct_attn_out,
    dcp_lse_combine_triton,
    dcp_pack_a2a_send,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.runtime_context import get_parallel


def _warn_deprecated_dcp_accessor(name: str, replacement: str) -> None:
    warnings.warn(
        f"{name} is deprecated; use {replacement} instead.",
        DeprecationWarning,
        stacklevel=2,
    )


def dcp_enabled() -> bool:

    _warn_deprecated_dcp_accessor("dcp_enabled()", "get_parallel().dcp_enabled")
    return get_parallel().dcp_enabled


def get_attention_dcp_world_size() -> int:

    _warn_deprecated_dcp_accessor(
        "get_attention_dcp_world_size()", "get_parallel().attn_dcp_size"
    )
    return get_parallel().attn_dcp_size


def get_attention_dcp_rank() -> int:

    _warn_deprecated_dcp_accessor(
        "get_attention_dcp_rank()", "get_parallel().attn_dcp_rank"
    )
    return get_parallel().attn_dcp_rank


def _ag_lse(cp_attn_lse: torch.Tensor, cp_group: GroupCoordinator) -> torch.Tensor:

    return cp_group.all_gather(cp_attn_lse, dim=0).view(
        (cp_group.world_size,) + cp_attn_lse.shape
    )


def cp_lse_ag_out_rs_mha(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    return_lse: bool = False,
):
    if cp_group.world_size == 1:
        return (cp_attn_out, cp_attn_lse) if return_lse else cp_attn_out

    cp_attn_lse = cp_attn_lse.contiguous()
    lses = _ag_lse(cp_attn_lse, cp_group)
    global_lse = torch.logsumexp(lses, dim=0)
    scale = torch.exp(cp_attn_lse - global_lse).unsqueeze(-1)
    scale = torch.nan_to_num(scale, nan=0.0, posinf=0.0, neginf=0.0)

    out = cp_attn_out.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    out.mul_(scale)
    out = cp_group.all_reduce(out)

    cp_num_heads = global_lse.shape[1] // cp_group.world_size
    cp_rank = cp_group.rank_in_group
    head_start = cp_num_heads * cp_rank
    head_end = cp_num_heads * (cp_rank + 1)
    out = out[:, head_start:head_end, :].contiguous()
    if return_lse:
        return out, global_lse[:, head_start:head_end].contiguous()
    return out


def cp_lse_ag_out_rs_mla(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    ctx: Optional[CPTritonContext] = None,
    is_lse_base_on_e: bool = False,
):

    if cp_group.world_size == 1:
        return cp_attn_out

    if ctx is None:
        ctx = CPTritonContext()

    with use_symmetric_memory(cp_group):

        new_output = cp_attn_out.new_empty(
            cp_attn_out.transpose(0, 1).shape, dtype=torch.float32
        )
        cp_attn_lse = cp_attn_lse.to(torch.float32)
    lses = _ag_lse(cp_attn_lse, cp_group)
    out, _ = correct_attn_out(
        cp_attn_out,
        lses,
        cp_group.rank_in_group,
        ctx,
        new_output,
        is_lse_base_on_e=is_lse_base_on_e,
    )
    out = cp_group.reduce_scatter_along_dim(out, dim=0)
    return out.to(cp_attn_out.dtype)


def _all_gather_dcp_kv_cache(kv_a: torch.Tensor):
    parallel = get_parallel()
    dcp_world_size = parallel.dcp_size

    gathered_kv_a = kv_a.new_empty(
        (kv_a.shape[0] * dcp_world_size, *kv_a.shape[1:]),
    )


    if kv_a.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        parallel.dcp_group.all_gather_into_tensor(
            gathered_kv_a.view(torch.uint8), kv_a.contiguous().view(torch.uint8)
        )
    else:
        parallel.dcp_group.all_gather_into_tensor(gathered_kv_a, kv_a)
    gathered_kv_a = (
        gathered_kv_a.reshape((dcp_world_size,) + kv_a.shape)
        .transpose(0, 1)
        .reshape(-1, *kv_a.shape[1:])
    )
    return gathered_kv_a


def all_gather_kv_cache_for_mha_chunk_extend(
    kv_a: torch.Tensor,
    k_pe: torch.Tensor,
    prefix_kv_lens_cpu: torch.Tensor,
    prefix_starts_cpu: torch.Tensor = None,
):
    if get_parallel().dcp_enabled:
        kv_a = kv_a.unsqueeze(1)
        gathered_kv = all_gather_kv_cache_for_dcp(
            kv_a,
            k_pe,
            prefix_kv_lens_cpu,
            prefix_starts_cpu,
        )
        kv_a, k_pe = gathered_kv.split([kv_a.shape[-1], k_pe.shape[-1]], dim=-1)
        kv_a = kv_a.squeeze(1)
    return kv_a.contiguous(), k_pe.contiguous()


def all_gather_kv_cache_for_mha_extend(
    token_to_kv_pool,
    attn_mqa,
    dcp_local_prefix_kv_indices,
    seq_lens,
    extend_prefix_lens,
    extend_prefix_lens_cpu: list[int],
    extend_seq_lens,
    kv_a: torch.Tensor,
    k_pe: torch.Tensor,
):
    prefix_kv_a, prefix_k_pe = token_to_kv_pool.get_mla_kv_buffer(
        attn_mqa, dcp_local_prefix_kv_indices, dst_dtype=kv_a.dtype
    )
    extend_prefix_lens_cpu = torch.tensor(extend_prefix_lens_cpu)
    gathered_kv_cache = all_gather_kv_cache_for_dcp(
        prefix_kv_a,
        prefix_k_pe,
        extend_prefix_lens_cpu,
    )
    prefix_kv_a, prefix_k_pe = gathered_kv_cache.split(
        [kv_a.shape[-1], k_pe.shape[-1]], dim=-1
    )
    prefix_kv_a = prefix_kv_a.squeeze(1)


    if prefix_kv_a.dtype != kv_a.dtype:
        prefix_kv_a = prefix_kv_a.to(kv_a.dtype)
    if prefix_k_pe.dtype != k_pe.dtype:
        prefix_k_pe = prefix_k_pe.to(k_pe.dtype)

    prefix_lens_cu = torch.zeros(
        len(seq_lens) + 1,
        dtype=torch.int32,
        device=kv_a.device,
    )
    extend_lens_cu = torch.zeros_like(prefix_lens_cu)
    prefix_lens_cu[1:] = torch.cumsum(extend_prefix_lens, dim=0)
    extend_lens_cu[1:] = torch.cumsum(extend_seq_lens, dim=0)
    kv_a_tuple = ()
    k_pe_tuple = ()
    for i in range(len(seq_lens)):
        kv_a_tuple += (
            prefix_kv_a[prefix_lens_cu[i] : prefix_lens_cu[i + 1]],
            kv_a[extend_lens_cu[i] : extend_lens_cu[i + 1]],
        )
        k_pe_tuple += (
            prefix_k_pe[prefix_lens_cu[i] : prefix_lens_cu[i + 1]],
            k_pe[extend_lens_cu[i] : extend_lens_cu[i + 1]],
        )
    kv_a = torch.cat(kv_a_tuple, dim=0)
    k_pe = torch.cat(k_pe_tuple, dim=0)
    return kv_a.contiguous(), k_pe.contiguous()


def all_gather_q_for_mla_decode(
    q_nope_out: torch.Tensor,
    q_pe: torch.Tensor,
):
    group = get_parallel().dcp_group
    with use_symmetric_memory(group):

        combined = torch.cat([q_pe.transpose(0, 1), q_nope_out.transpose(0, 1)], dim=-1)
    gathered = group.all_gather(combined, dim=0)
    d_pe = q_pe.size(-1)
    d_nope = q_nope_out.size(-1)
    q_pe, q_nope_out = gathered.split([d_pe, d_nope], dim=-1)
    q_pe = q_pe.transpose(0, 1)
    q_nope_out = q_nope_out.transpose(0, 1)
    return q_nope_out, q_pe


def all_gather_kv_cache_for_mla_extend(
    token_to_kv_pool,
    attn_mqa,
    extend_prefix_lens_cpu: list[int],
    dcp_local_prefix_kv_indices,
    dcp_extend_prefix_lens_sum,
    dcp_kv_buffer,
    kv_lora_rank,
    k_nope,
    k_pe,
):
    cache_k_nope, cache_k_rope = token_to_kv_pool.get_mla_kv_buffer(
        attn_mqa,
        dcp_local_prefix_kv_indices,
    )
    extend_prefix_lens_cpu = torch.tensor(extend_prefix_lens_cpu)

    gathered_kv = all_gather_kv_cache_for_dcp(
        cache_k_nope,
        cache_k_rope,
        extend_prefix_lens_cpu,
        prefix_starts_cpu=torch.zeros_like(extend_prefix_lens_cpu),
    )
    dcp_kv_buffer[:dcp_extend_prefix_lens_sum] = gathered_kv


    dcp_kv_buffer[
        dcp_extend_prefix_lens_sum:,
        ...,
        :kv_lora_rank,
    ] = k_nope
    dcp_kv_buffer[
        dcp_extend_prefix_lens_sum:,
        ...,
        kv_lora_rank:,
    ] = k_pe


def all_gather_kv_cache_for_dcp(
    prefix_kv_a: torch.Tensor,
    prefix_k_pe: torch.Tensor,
    prefix_kv_lens_cpu: torch.Tensor,
    prefix_starts_cpu: torch.Tensor = None,
):

    parallel = get_parallel()
    if not parallel.dcp_enabled:
        return torch.cat([prefix_kv_a, prefix_k_pe], dim=-1)

    dcp_world_size = parallel.dcp_size
    dcp_rank = parallel.dcp_rank

    if prefix_starts_cpu is None:
        prefix_starts_cpu = torch.zeros_like(prefix_kv_lens_cpu)

    left_pads = prefix_starts_cpu % dcp_world_size > dcp_rank
    left_pads = left_pads.to(torch.int32)
    right_pads = (
        prefix_starts_cpu + prefix_kv_lens_cpu - 1
    ) % dcp_world_size < dcp_rank
    right_pads = right_pads.to(torch.int32)
    padded_lens = (
        prefix_kv_lens_cpu + (prefix_starts_cpu % dcp_world_size) + dcp_world_size - 1
    ) // dcp_world_size

    local_kv_lens = padded_lens - left_pads - right_pads
    local_kv_lens_cu = torch.zeros(
        len(prefix_kv_lens_cpu) + 1,
        dtype=torch.int32,
    )
    local_kv_lens_cu[1:] = torch.cumsum(local_kv_lens, dim=0)

    padded_kv_cache_arr = []
    prefix_kv_cache = torch.cat([prefix_kv_a, prefix_k_pe], dim=-1)
    for req_idx in range(len(prefix_kv_lens_cpu)):
        padded_tensor = prefix_kv_cache.new_empty(
            (padded_lens[req_idx].item(),) + prefix_kv_cache.size()[1:]
        )
        padded_tensor[
            left_pads[req_idx] : left_pads[req_idx] + local_kv_lens[req_idx]
        ] = prefix_kv_cache[local_kv_lens_cu[req_idx] : local_kv_lens_cu[req_idx + 1]]
        padded_kv_cache_arr.append(padded_tensor)

    padded_kv_cache = torch.cat(padded_kv_cache_arr, dim=0)

    gatherd_kv_cache = _all_gather_dcp_kv_cache(padded_kv_cache)


    padded_lens_cu = torch.zeros(
        len(prefix_kv_lens_cpu) + 1,
        dtype=torch.int32,
    )
    padded_lens_cu[1:] = torch.cumsum(padded_lens, dim=0)
    kv_cache_tuple = ()
    for req_idx in range(len(prefix_kv_lens_cpu)):
        kv_cache_tuple += (
            gatherd_kv_cache[
                padded_lens_cu[req_idx] * dcp_world_size
                + (prefix_starts_cpu[req_idx] % dcp_world_size) :
            ][: prefix_kv_lens_cpu[req_idx]],
        )
    gatherd_kv_cache = torch.cat(kv_cache_tuple, dim=0)

    return gatherd_kv_cache


_FI_A2A_STATE: Optional[dict] = None


def init_fi_a2a_workspace(cp_group: "GroupCoordinator") -> None:


    global _FI_A2A_STATE
    if _FI_A2A_STATE is not None:
        return
    if cp_group.world_size == 1:
        return

    import torch.distributed as dist

    try:
        from flashinfer.comm.dcp_alltoall import (
            decode_cp_a2a_allocate_mnnvl_workspace,
            decode_cp_a2a_init_workspace,
        )
        from flashinfer.comm.mapping import Mapping
        from flashinfer.comm.mnnvl import MnnvlConfig, is_mnnvl_fabric_supported
    except ImportError as e:
        raise ImportError(
            "--dcp-comm-backend fi_a2a requires FlashInfer with the DCP "
            "all-to-all kernel (flashinfer #2951); could not import "
            "flashinfer.comm.dcp_alltoall."
        ) from e


    from sglang.srt.layers.moe.token_dispatcher.flashinfer_utils import (
        TorchDistributedCommBackend,
    )

    if not is_mnnvl_fabric_supported(torch.cuda.current_device()):
        raise RuntimeError(
            "--dcp-comm-backend fi_a2a requires MNNVL fabric memory (e.g. "
            "GB200 NVL72); is_mnnvl_fabric_supported() returned False. Use "
            "--dcp-comm-backend a2a or ag_rs on clusters without MNNVL."
        )

    cp_size = cp_group.world_size
    cp_rank = cp_group.rank_in_group
    mapping = Mapping(
        world_size=cp_size,
        rank=cp_rank,
        gpus_per_node=torch.cuda.device_count(),
        cp_size=cp_size,
        tp_size=1,
        pp_size=1,
    )
    workspace = decode_cp_a2a_allocate_mnnvl_workspace(
        mapping,
        mnnvl_config=MnnvlConfig(
            comm_backend=TorchDistributedCommBackend(cp_group.device_group)
        ),
    )
    decode_cp_a2a_init_workspace(workspace, cp_rank, cp_size)


    dist.barrier(group=cp_group.device_group)
    _FI_A2A_STATE = {
        "workspace": workspace,
        "cp_rank": cp_rank,
    }


def dcp_a2a_lse_reduce(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: "GroupCoordinator",
    is_lse_base_on_e: bool = True,
    cuda_graph_buffers: Optional[dict] = None,
    comm_backend: str = "a2a",
) -> torch.Tensor:

    if cp_group.world_size == 1:
        return cp_attn_out

    if comm_backend == "fi_a2a":
        return _dcp_fi_a2a_lse_reduce(
            cp_attn_out, cp_attn_lse, cp_group, is_lse_base_on_e
        )

    N = cp_group.world_size
    B, H, D = cp_attn_out.shape
    assert H % N == 0, f"num_heads ({H}) must be divisible by dcp_size ({N})"
    H_per_rank = H // N
    out_dtype = cp_attn_out.dtype
    lpd = _lse_pack_dim(out_dtype)

    if cuda_graph_buffers is not None:
        send_combined = cuda_graph_buffers["send_combined"]
        recv_combined = cuda_graph_buffers["recv_combined"]
    else:
        send_combined = torch.empty(
            N,
            B,
            H_per_rank,
            D + lpd,
            dtype=out_dtype,
            device=cp_attn_out.device,
        )
        recv_combined = torch.empty_like(send_combined)

    send_words = send_combined.view(torch.float32)
    dcp_pack_a2a_send(
        cp_attn_out,
        cp_attn_lse,
        send_combined[:, :, :, :D],
        send_words[:, :, :, D // lpd],
    )


    cp_group.all_to_all_single(
        recv_combined.reshape(-1).view(torch.uint8),
        send_combined.reshape(-1).view(torch.uint8),
    )

    recv_output = recv_combined[:, :B, :, :D]
    recv_lse = recv_combined.view(torch.float32)[:, :B, :, D // lpd]

    combined, _ = dcp_lse_combine_triton(
        recv_output, recv_lse, is_lse_base_on_e=is_lse_base_on_e
    )
    return combined


def _dcp_fi_a2a_lse_reduce(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: "GroupCoordinator",
    is_lse_base_on_e: bool = True,
) -> torch.Tensor:

    from flashinfer.comm.dcp_alltoall import decode_cp_a2a_alltoall

    state = _FI_A2A_STATE
    assert state is not None, (
        "fi_a2a workspace not initialized — call init_fi_a2a_workspace(dcp_group) "
        "at model-runner init (before CUDA graph capture)."
    )

    N = cp_group.world_size
    B, H, D = cp_attn_out.shape
    assert H % N == 0, f"num_heads ({H}) must be divisible by dcp_size ({N})"
    H_per_rank = H // N


    partial_o = torch.empty(
        B, H_per_rank, N, D, dtype=cp_attn_out.dtype, device=cp_attn_out.device
    )
    softmax_stats = torch.empty(
        B, H_per_rank, N, 2, dtype=torch.float32, device=cp_attn_out.device
    )
    dcp_pack_a2a_send(
        cp_attn_out,
        cp_attn_lse,
        partial_o.permute(2, 0, 1, 3),
        softmax_stats[..., 0].permute(2, 0, 1),
    )

    o_out, stats_out = decode_cp_a2a_alltoall(
        partial_o,
        softmax_stats,
        state["workspace"],
        state["cp_rank"],
        N,
    )

    recv_output = o_out.permute(2, 0, 1, 3)
    recv_lse = stats_out[..., 0].permute(2, 0, 1)

    combined, _ = dcp_lse_combine_triton(
        recv_output, recv_lse, is_lse_base_on_e=is_lse_base_on_e
    )
    return combined
