from __future__ import annotations

import functools
import logging
from contextlib import contextmanager
from enum import IntEnum, auto
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.distributed import (
    GroupCoordinator,
    get_attn_cp_group,
    get_attn_cp_overlap_group,
    get_attn_tensor_model_parallel_rank,
    get_attn_tensor_model_parallel_world_size,
    get_attn_tp_group,
)
from sglang.srt.distributed import get_moe_dp_group as _get_moe_dp_group
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.runtime_context import (
    get_device,
    get_exec,
    get_flags,
    get_parallel,
)
from sglang.srt.utils import get_bool_env_var, is_cpu, is_hip

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

_ATTN_DP_RANK: Optional[int] = None
_ATTN_DP_SIZE: Optional[int] = None


def world_dp_gather_enabled() -> bool:

    dp = get_flags().dp
    return dp.use_world_group_for_gather and not dp.joiner_skip_all_gather


def enable_joiner_all_gather():
    get_flags().dp.joiner_skip_all_gather = False


def update_dp_attention_post_scale(new_dp_size: int, new_dp_rank: int):
    global _ATTN_DP_SIZE, _ATTN_DP_RANK
    _ATTN_DP_SIZE = new_dp_size
    _ATTN_DP_RANK = new_dp_rank
    get_flags().dp.use_world_group_for_gather = True
    logger.debug(
        "[Elastic EP] dp_attention switched to WORLD: dp_size=%d dp_rank=%d",
        new_dp_size,
        new_dp_rank,
    )


_is_hip = is_hip()
_USE_ROCM700A_WA = _is_hip and get_bool_env_var("SGLANG_USE_ROCM700A")
_is_cpu = is_cpu()


class DpPaddingMode(IntEnum):


    MAX_LEN = auto()

    SUM_LEN = auto()

    def is_max_len(self):
        return self == DpPaddingMode.MAX_LEN

    def is_sum_len(self):
        return self == DpPaddingMode.SUM_LEN

    @classmethod
    def get_dp_padding_mode(
        cls, is_extend_in_batch, global_num_tokens: List[int]
    ) -> DpPaddingMode:
        dp_size = get_attention_dp_size()


        from sglang.srt.layers.moe.utils import get_moe_a2a_backend

        if get_moe_a2a_backend().is_pplx():
            return DpPaddingMode.MAX_LEN


        if is_extend_in_batch and dp_size > 1:


            if get_flags().dp.max_len_with_idle and min(global_num_tokens) == 0:
                return DpPaddingMode.MAX_LEN
            return DpPaddingMode.SUM_LEN


        max_len = max(global_num_tokens)
        sum_len = sum(global_num_tokens)
        if sum_len * 2 >= max_len * dp_size:
            return cls.MAX_LEN
        else:
            return cls.SUM_LEN

    @classmethod
    def get_default_mode_in_cuda_graph(cls) -> DpPaddingMode:


        if _USE_ROCM700A_WA:
            return cls.SUM_LEN
        else:
            return cls.MAX_LEN


class _DpGatheredBufferWrapper:


    _global_dp_buffer_len: int = 0
    _local_dp_buffer_len: int = 0
    _dp_max_padding: bool = False
    _global_num_tokens: Optional[List[int]] = None
    _global_num_tokens_gpu: Optional[torch.Tensor] = None

    @classmethod
    def set_metadata(cls, hidden_size: int, dtype: torch.dtype, device: torch.device):
        from sglang.srt.runtime_context import get_flags

        dp = get_flags().dp
        dp.buffer_hidden_size = hidden_size
        dp.buffer_dtype = dtype
        dp.buffer_device = device

    @classmethod
    def set_dp_buffer_len(
        cls,
        global_dp_buffer_len: int,
        local_dp_buffer_len: int,
        dp_max_padding: bool,
        global_num_tokens: Optional[List[int]] = None,
        global_num_tokens_gpu: Optional[torch.Tensor] = None,
    ):
        cls._global_dp_buffer_len = global_dp_buffer_len
        cls._local_dp_buffer_len = local_dp_buffer_len
        cls._dp_max_padding = dp_max_padding
        cls._global_num_tokens = global_num_tokens
        cls._global_num_tokens_gpu = global_num_tokens_gpu

    @classmethod
    def get_global_dp_buffer(cls, group: GroupCoordinator) -> torch.Tensor:
        from sglang.srt.runtime_context import get_flags

        dp = get_flags().dp
        with use_symmetric_memory(group, disabled=not cls._dp_max_padding):
            buffer = torch.empty(
                (cls._global_dp_buffer_len, dp.buffer_hidden_size),
                dtype=dp.buffer_dtype,
                device=dp.buffer_device,
            )
        return buffer

    @classmethod
    def get_local_dp_buffer(cls, group: GroupCoordinator) -> torch.Tensor:
        from sglang.srt.runtime_context import get_flags

        dp = get_flags().dp
        with use_symmetric_memory(group, disabled=not cls._dp_max_padding):
            buffer = torch.empty(
                (cls._local_dp_buffer_len, dp.buffer_hidden_size),
                dtype=dp.buffer_dtype,
                device=dp.buffer_device,
            )
        return buffer

    @classmethod
    def get_global_dp_buffer_len(cls) -> int:
        return cls._global_dp_buffer_len

    @classmethod
    def get_local_dp_buffer_len(cls) -> int:
        return cls._local_dp_buffer_len

    @classmethod
    def set_local_dp_buffer_len(cls, local_dp_buffer_len: int) -> None:
        cls._local_dp_buffer_len = local_dp_buffer_len

    @classmethod
    def get_dp_global_num_tokens(cls) -> List[int]:
        return cls._global_num_tokens

    @classmethod
    def get_dp_global_num_tokens_gpu(cls) -> Optional[torch.Tensor]:
        return cls._global_num_tokens_gpu

    @classmethod
    def get_dp_hidden_size(cls) -> int:
        from sglang.srt.runtime_context import get_flags

        return get_flags().dp.buffer_hidden_size

    @classmethod
    def get_dp_dtype(cls) -> torch.dtype:
        from sglang.srt.runtime_context import get_flags

        return get_flags().dp.buffer_dtype

    @classmethod
    def get_dp_device(cls) -> torch.device:
        from sglang.srt.runtime_context import get_flags

        return get_flags().dp.buffer_device

    @classmethod
    def is_dp_max_padding(cls) -> bool:
        return cls._dp_max_padding


def set_dp_buffer_len(
    global_dp_buffer_len: int,
    local_dp_buffer_len: int,
    dp_max_padding: bool,
    global_num_tokens: Optional[List[int]] = None,
    global_num_tokens_gpu: Optional[torch.Tensor] = None,
):
    _DpGatheredBufferWrapper.set_dp_buffer_len(
        global_dp_buffer_len,
        local_dp_buffer_len,
        dp_max_padding,
        global_num_tokens,
        global_num_tokens_gpu,
    )


def get_global_dp_buffer(group: GroupCoordinator) -> torch.Tensor:
    return _DpGatheredBufferWrapper.get_global_dp_buffer(group=group)


def get_local_dp_buffer(group: GroupCoordinator) -> torch.Tensor:
    return _DpGatheredBufferWrapper.get_local_dp_buffer(group=group)


def get_global_dp_buffer_len() -> int:
    return _DpGatheredBufferWrapper.get_global_dp_buffer_len()


def get_local_dp_buffer_len() -> int:
    return _DpGatheredBufferWrapper.get_local_dp_buffer_len()


def set_local_dp_buffer_len(local_dp_buffer_len: int) -> None:
    _DpGatheredBufferWrapper.set_local_dp_buffer_len(local_dp_buffer_len)


def get_dp_global_num_tokens() -> List[int]:
    return _DpGatheredBufferWrapper.get_dp_global_num_tokens()


def get_dp_hidden_size() -> int:
    return _DpGatheredBufferWrapper.get_dp_hidden_size()


def get_dp_dtype() -> torch.dtype:
    return _DpGatheredBufferWrapper.get_dp_dtype()


def get_dp_device() -> torch.device:
    return _DpGatheredBufferWrapper.get_dp_device()


def set_is_extend_in_batch(is_extend_in_batch: bool):


    from sglang.srt.runtime_context import get_forward

    get_forward().set("is_extend_in_batch", is_extend_in_batch)


def get_is_extend_in_batch() -> bool:
    from sglang.srt.runtime_context import get_forward

    return get_forward().is_extend_in_batch


def is_dp_max_padding() -> bool:
    return _DpGatheredBufferWrapper.is_dp_max_padding()


def compute_dp_attention_world_info(
    enable_dp_attention, tp_rank, tp_size, dp_size, attn_cp_size: int = 1
):
    attn_dp_size = dp_size if enable_dp_attention else 1
    attn_tp_size = tp_size // attn_dp_size // attn_cp_size
    attn_tp_rank = tp_rank % attn_tp_size

    if not enable_dp_attention:
        attn_dp_rank = 0
    else:


        attn_dp_rank = tp_rank // (attn_tp_size * attn_cp_size)

    return attn_tp_rank, attn_tp_size, attn_dp_rank, attn_dp_size


def initialize_dp_attention(
    server_args: ServerArgs,
    model_config: ModelConfig,
):
    global _ATTN_DP_RANK, _ATTN_DP_SIZE
    dp = get_flags().dp
    dp.max_len_with_idle = (
        getattr(model_config.hf_config, "hybrid_override_pattern", None) is not None
    )
    enable_dp_attention = get_parallel().config.enable_dp_attention
    dp_size = get_parallel().config.dp_size
    attn_cp_size = get_parallel().config.attn_cp_size

    dp.enabled = enable_dp_attention

    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()

    _, _, _ATTN_DP_RANK, _ = compute_dp_attention_world_info(
        enable_dp_attention, tp_rank, tp_size, dp_size, attn_cp_size
    )
    _ATTN_DP_SIZE = dp_size if enable_dp_attention else 1

    if (
        get_exec().moe.elastic_ep_backend is not None
        and get_parallel().config.max_ep_size
    ):
        _ATTN_DP_RANK = tp_rank + get_parallel().config.ep_join_rank_offset
        if server_args.is_ep_scale_joiner:
            dp.joiner_skip_all_gather = True

    _DpGatheredBufferWrapper.set_metadata(
        hidden_size=model_config.hidden_size,
        dtype=model_config.dtype,
        device=torch.device(get_device().device),
    )


def is_dp_attention_enabled() -> bool:
    return get_flags().dp.enabled


def is_allocation_symmetric() -> bool:
    return not is_dp_attention_enabled() or is_dp_max_padding()


def get_attention_dp_rank() -> int:
    assert _ATTN_DP_RANK is not None, "dp attention not initialized!"
    return _ATTN_DP_RANK


def get_attention_dp_size() -> int:
    assert _ATTN_DP_SIZE is not None, "dp attention not initialized!"
    return _ATTN_DP_SIZE


@contextmanager
def disable_dp_size():

    global _ATTN_DP_SIZE
    assert _ATTN_DP_SIZE is not None, "dp attention not initialized!"

    old_dp_size = _ATTN_DP_SIZE
    _ATTN_DP_SIZE = 1
    try:
        yield
    finally:
        _ATTN_DP_SIZE = old_dp_size


def get_dp_local_info(forward_batch: ForwardBatch) -> Tuple[torch.Tensor, torch.Tensor]:

    dp_rank = get_attention_dp_rank()

    if forward_batch.dp_local_start_pos is None:
        cumtokens = torch.cumsum(forward_batch.global_num_tokens_gpu, dim=0)
        if dp_rank == 0:
            local_start_pos = torch.zeros_like(cumtokens[0])
        else:
            local_start_pos = cumtokens[dp_rank - 1]
        local_num_tokens = forward_batch.global_num_tokens_gpu[dp_rank]

        forward_batch.dp_local_start_pos = local_start_pos
        forward_batch.dp_local_num_tokens = local_num_tokens

    return forward_batch.dp_local_start_pos, forward_batch.dp_local_num_tokens


def get_dp_local_slice_cpu(
    forward_batch: ForwardBatch,
    can_run_graph: bool,
    cuda_graph_batch: Optional[int],
) -> Tuple[int, int]:


    global_num_tokens = forward_batch.global_num_tokens_cpu
    dp_rank = get_attention_dp_rank()
    local_num_tokens = global_num_tokens[dp_rank]
    if can_run_graph:
        local_start_pos = dp_rank * cuda_graph_batch
    else:
        local_start_pos = sum(global_num_tokens[:dp_rank])
    return local_start_pos, local_num_tokens


from sglang.kernels.ops.memory.memcpy_triton import memcpy_triton


def memcpy_cpu(dst, src, dim, offset, sz, offset_src):
    assert dim == 0, "Only dim=0 supported"
    assert src.shape[1:] == dst.shape[1:], "src and dst must have same trailing shape"

    total_rows_dst, total_rows_src = dst.shape[0], src.shape[0]
    dst_start, src_start = 0, 0

    if offset_src:

        src_start = offset
        dst_start = 0
    else:

        src_start = 0
        dst_start = offset

    dst_end = min(dst_start + sz, total_rows_dst)
    src_end = min(src_start + sz, total_rows_src)
    actual_sz = min(dst_end - dst_start, src_end - src_start)

    if actual_sz <= 0:
        return

    dst[dst_start : dst_start + actual_sz].copy_(src[src_start : src_start + actual_sz])


memcpy_func = memcpy_cpu if _is_cpu else memcpy_triton


def memcpy(dst, src, dim, offset, sz, offset_src):
    memcpy_func(dst, src, dim, offset, sz, offset_src)


def _dp_gather_via_all_reduce(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    is_partial: bool,
):
    local_start_pos, local_num_tokens = get_dp_local_info(forward_batch)

    global_tokens.fill_(0)
    assert local_tokens.is_contiguous()
    assert global_tokens.is_contiguous()

    if local_tokens.shape[0] > 0 and (
        is_partial or get_attn_tensor_model_parallel_rank() == 0
    ):
        assert (
            local_tokens.untyped_storage() is not global_tokens.untyped_storage()
        ), "aliasing between global_tokens and local_tokens not allowed"

        memcpy(global_tokens, local_tokens, 0, local_start_pos, local_num_tokens, False)


    if world_dp_gather_enabled():
        torch.distributed.all_reduce(
            global_tokens,
            op=torch.distributed.ReduceOp.SUM,
            group=torch.distributed.group.WORLD,
        )
    else:
        NUM_GPUS_PER_NODE = 8
        if (
            not local_tokens.dtype.is_floating_point
            and get_tensor_model_parallel_world_size() <= NUM_GPUS_PER_NODE
        ):
            from sglang.srt.distributed.parallel_state import inplace_all_reduce

            inplace_all_reduce(global_tokens, group_name=get_tp_group().unique_name)

        else:
            global_tokens[:] = tensor_model_parallel_all_reduce(global_tokens)


def _dp_gather_via_all_gather(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    is_partial: bool,
):
    use_world = world_dp_gather_enabled()

    if get_attn_tensor_model_parallel_world_size() == 1:
        if use_world:
            torch.distributed.all_gather_into_tensor(
                global_tokens,
                local_tokens,
                group=torch.distributed.group.WORLD,
            )
        else:
            get_tp_group().all_gather_into_tensor(global_tokens, local_tokens)
        return

    if not is_partial:
        if get_attn_tensor_model_parallel_rank() != 0:
            local_tokens.fill_(0)
    scattered_local_tokens = local_tokens.tensor_split(
        get_attn_tensor_model_parallel_world_size()
    )[get_attn_tensor_model_parallel_rank()]
    get_attn_tp_group().reduce_scatter_tensor(scattered_local_tokens, local_tokens)
    if use_world:
        torch.distributed.all_gather_into_tensor(
            global_tokens,
            scattered_local_tokens,
            group=torch.distributed.group.WORLD,
        )
    else:
        get_tp_group().all_gather_into_tensor(global_tokens, scattered_local_tokens)


_USE_DP_GATHERV = get_bool_env_var("SGLANG_DP_USE_GATHERV")

_DP_GATHER_FP8_GROUP = 128

_dp_gather_fp8_bufs: dict = {}


@functools.lru_cache(maxsize=1)
def _use_dp_gather_fp8() -> bool:
    from sglang.srt.environ import envs

    return envs.SGLANG_ENABLE_DP_GATHER_FP8.get()


def _get_dp_gather_fp8_bufs(rows: int, hidden: int, device: torch.device):
    key = str(device)
    bufs = _dp_gather_fp8_bufs.get(key)
    if bufs is None or bufs[0].shape[0] < rows:
        bufs = (
            torch.empty((rows, hidden), dtype=torch.uint8, device=device),
            torch.empty(
                (rows, hidden // _DP_GATHER_FP8_GROUP),
                dtype=torch.float32,
                device=device,
            ),
        )
        _dp_gather_fp8_bufs[key] = bufs
    return bufs[0][:rows], bufs[1][:rows]


@triton.jit
def _dequant_per_token_group_fp8_kernel(
    q_ptr,
    s_ptr,
    out_ptr,
    HIDDEN: tl.constexpr,
    NGROUPS: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)


    for start in tl.static_range(0, HIDDEN, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < HIDDEN
        qv = tl.load(q_ptr + row * HIDDEN + offs, mask=mask, other=0.0).to(tl.float32)
        sv = tl.load(s_ptr + row * NGROUPS + offs // GROUP, mask=mask, other=0.0)
        tl.store(out_ptr + row * HIDDEN + offs, (qv * sv).to(tl.bfloat16), mask=mask)


@triton.jit
def _mask_dp_pad_topk_ids_kernel(
    topk_ids_ptr,
    counts_ptr,
    max_len,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    rank = row // max_len
    pos = row % max_len
    valid = pos < tl.load(counts_ptr + rank)
    if valid == 0:
        offs = tl.arange(0, BLOCK)
        tl.store(topk_ids_ptr + row * TOPK + offs, -1, mask=offs < TOPK)


def mask_dp_pad_moe_topk_ids(topk_ids: torch.Tensor) -> None:

    counts = _DpGatheredBufferWrapper.get_dp_global_num_tokens_gpu()
    if counts is None:
        return
    max_len = _DpGatheredBufferWrapper.get_local_dp_buffer_len()
    rows, topk = topk_ids.shape
    if max_len <= 0 or rows != counts.shape[0] * max_len:

        return
    _mask_dp_pad_topk_ids_kernel[(rows,)](
        topk_ids,
        counts,
        max_len,
        TOPK=topk,
        BLOCK=triton.next_power_of_2(topk),
    )


def _dp_gather_via_all_gatherv_fp8(
    global_tokens: torch.Tensor,
    local_real: torch.Tensor,
    sizes: List[int],
):

    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    rows = global_tokens.shape[0]
    hidden = global_tokens.shape[-1]
    q, s = sglang_per_token_group_quant_fp8(
        local_real.contiguous(), _DP_GATHER_FP8_GROUP
    )
    gq, gs = _get_dp_gather_fp8_bufs(rows, hidden, global_tokens.device)
    tp_group = get_tp_group()
    tp_group.all_gatherv(q.view(torch.uint8), sizes=sizes, output=gq)
    tp_group.all_gatherv(s, sizes=sizes, output=gs)
    _dequant_per_token_group_fp8_kernel[(rows,)](
        gq.view(torch.float8_e4m3fn),
        gs,
        global_tokens,
        HIDDEN=hidden,
        NGROUPS=hidden // _DP_GATHER_FP8_GROUP,
        GROUP=_DP_GATHER_FP8_GROUP,
        BLOCK=2048,
    )


def is_dp_gatherv_active() -> bool:

    return (
        _USE_DP_GATHERV
        and not world_dp_gather_enabled()
        and get_attn_tensor_model_parallel_world_size() == 1
        and get_tensor_model_parallel_world_size() == get_attention_dp_size()
        and not _DpGatheredBufferWrapper.is_dp_max_padding()
    )


def _dp_gatherv_sizes(forward_batch) -> Optional[List[int]]:

    sizes = getattr(forward_batch, "global_num_tokens_for_logprob_cpu", None)
    if sizes is None:
        sizes = getattr(forward_batch, "global_num_tokens_cpu", None)
    if sizes is None:
        return None
    try:
        return [int(x) for x in sizes]
    except (TypeError, ValueError):
        return None


def _dp_gather_via_all_gatherv(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    is_partial: bool,
    sizes: List[int],
):


    rank = get_attention_dp_rank()
    local_rows = sizes[rank]
    if local_tokens.shape[0] == local_rows:
        local_real = local_tokens
    elif local_tokens.shape[0] > local_rows:
        local_real = local_tokens[:local_rows]
    else:
        local_real = local_tokens.new_zeros((local_rows, *local_tokens.shape[1:]))
        local_real[: local_tokens.shape[0]].copy_(local_tokens)


    if (
        _use_dp_gather_fp8()
        and global_tokens.dtype == torch.bfloat16
        and global_tokens.shape[-1] % _DP_GATHER_FP8_GROUP == 0
    ):
        _dp_gather_via_all_gatherv_fp8(global_tokens, local_real, sizes)
        return
    get_tp_group().all_gatherv(local_real, sizes=sizes, output=global_tokens)


def _dp_gather(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    is_partial: bool,
):
    if (
        is_dp_gatherv_active()
        and forward_batch.dp_padding_mode is not None
        and not forward_batch.dp_padding_mode.is_max_len()
    ):


        _gatherv_sizes = get_dp_global_num_tokens()
        if _gatherv_sizes is None or sum(_gatherv_sizes) != global_tokens.shape[0]:
            _gatherv_sizes = _dp_gatherv_sizes(forward_batch)
        if _gatherv_sizes is not None and sum(_gatherv_sizes) == global_tokens.shape[0]:
            _dp_gather_via_all_gatherv(
                global_tokens, local_tokens, forward_batch, is_partial, _gatherv_sizes
            )
            return
    if (
        forward_batch.dp_padding_mode is not None
        and forward_batch.dp_padding_mode.is_max_len()
    ):
        _dp_gather_via_all_gather(
            global_tokens, local_tokens, forward_batch, is_partial
        )
    else:
        _dp_gather_via_all_reduce(
            global_tokens, local_tokens, forward_batch, is_partial
        )


def dp_gather_partial(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
):
    _dp_gather(global_tokens, local_tokens, forward_batch, is_partial=True)


def dp_gather_replicate(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
):
    _dp_gather(global_tokens, local_tokens, forward_batch, is_partial=False)


def dp_scatter(
    local_tokens: torch.Tensor,
    global_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
):


    local_start_pos, local_num_tokens = get_dp_local_info(forward_batch)

    local_tokens.fill_(0)
    assert local_tokens.is_contiguous()
    assert global_tokens.is_contiguous()
    if local_tokens.shape[0] > 0:
        assert (
            local_tokens.untyped_storage() is not global_tokens.untyped_storage()
        ), "aliasing between local_tokens and global_tokens not allowed"

        memcpy(local_tokens, global_tokens, 0, local_start_pos, local_num_tokens, True)


def dp_reduce_scatter_tensor(output: torch.Tensor, input: torch.Tensor):
    if is_dp_gatherv_active():


        sizes = get_dp_global_num_tokens()
        if sizes is not None:
            get_tp_group().reduce_scatterv(input, output=output, sizes=sizes)
            return
    if get_tensor_model_parallel_world_size() == get_attention_dp_size():
        get_tp_group().reduce_scatter_tensor(output, input)
    else:
        scattered_local_tokens = input.tensor_split(
            get_tensor_model_parallel_world_size()
        )[get_tensor_model_parallel_rank()]
        get_tp_group().reduce_scatter_tensor(scattered_local_tokens, input)
        get_attn_tp_group().all_gather_into_tensor(output, scattered_local_tokens)


def get_dp_tbo_comm_stream() -> torch.cuda.Stream:
    from sglang.srt.runtime_context import get_stream

    return get_stream("dp_tbo_comm")


def _tbo_event(key) -> torch.cuda.Event:
    from sglang.srt.runtime_context import get_resources

    pool = get_resources().tbo_event_pool
    ev = pool.get(key)
    if ev is None:
        ev = torch.cuda.Event()
        pool[key] = ev
    return ev


def dp_gather_partial_async(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    event_key=("gather", 0),
) -> torch.cuda.Event:

    comm = get_dp_tbo_comm_stream()
    compute = torch.cuda.current_stream()

    local_tokens.record_stream(comm)
    global_tokens.record_stream(comm)
    ev = _tbo_event(event_key)
    with torch.cuda.stream(comm):
        comm.wait_stream(compute)
        dp_gather_partial(global_tokens, local_tokens, forward_batch)
        ev.record(comm)
    return ev


_TBO_PERSIST_BUF: dict = {}


def get_tbo_persistent_buffer(
    key, rows: int, hidden: int, dtype: torch.dtype, device
) -> torch.Tensor:

    buf = _TBO_PERSIST_BUF.get(key)
    cap = 0 if buf is None else buf.shape[0]
    if buf is None or rows > cap or buf.shape[1] != hidden or buf.dtype != dtype:
        new_rows = max(rows, cap)
        buf = torch.empty((new_rows, hidden), dtype=dtype, device=device)
        _TBO_PERSIST_BUF[key] = buf
    return buf[:rows]


def dp_reduce_scatterv_async(
    output_local: torch.Tensor,
    global_tokens: torch.Tensor,
    sizes: List[int],
    event_key=("combine", 0),
) -> torch.cuda.Event:

    comm = get_dp_tbo_comm_stream()
    compute = torch.cuda.current_stream()
    ev = _tbo_event(event_key)
    with torch.cuda.stream(comm):
        comm.wait_stream(compute)
        get_tp_group().reduce_scatterv(global_tokens, output=output_local, sizes=sizes)
        ev.record(comm)
    return ev


def attn_tp_reduce_scatter_tensor(output: torch.Tensor, input: torch.Tensor):
    return get_attn_tp_group().reduce_scatter_tensor(output, input)


def attn_cp_reduce_scatter_tensor(output: torch.Tensor, input: torch.Tensor):
    return get_attn_cp_group().reduce_scatter_tensor(output, input)


def attn_tp_all_reduce(input: torch.Tensor):
    return get_attn_tp_group().all_reduce(input)


def attn_tp_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor):
    return get_attn_tp_group().all_gather_into_tensor(output, input)


def attn_cp_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor):
    return get_attn_cp_group().all_gather_into_tensor(output, input)


def attn_cp_overlap_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor):
    return get_attn_cp_overlap_group().all_gather_into_tensor(output, input)


def attn_cp_overlap_reduce_scatter_tensor(output: torch.Tensor, input: torch.Tensor):
    return get_attn_cp_overlap_group().reduce_scatter_tensor(output, input)


def get_moe_cp_group() -> GroupCoordinator:

    return _get_moe_dp_group()


def get_moe_cp_rank() -> int:
    return _get_moe_dp_group().rank_in_group


def get_moe_cp_size() -> int:
    return _get_moe_dp_group().world_size


def is_enable_moe_cp_allgather() -> bool:

    return get_parallel().config.attn_cp_size > get_parallel().config.moe_dp_size


def moe_cp_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor):
    return _get_moe_dp_group().all_gather_into_tensor(output, input)


def attn_tp_all_gather(output_list: List[torch.Tensor], input: torch.Tensor):
    return get_attn_tp_group().all_gather(input, output_tensor_list=output_list)
