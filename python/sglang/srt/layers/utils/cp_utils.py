from dataclasses import dataclass
from itertools import accumulate
from typing import Callable, List

import torch
import torch.nn.functional as F

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import (
    _tbo_event,
    attn_cp_all_gather_into_tensor,
    attn_cp_overlap_all_gather_into_tensor,
    is_allocation_symmetric,
)
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from sglang.srt.runtime_context import (
    get_parallel,
    uses_mla_backend,
)


@dataclass
class ContextParallelMetadata:

    split_list: List[int] = None
    zigzag_index: List[int] = None
    cp_reverse_index: List[int] = None
    reverse_split_len: List[int] = None


    per_rank_actual_token: List[int] = None
    max_rank_len: List[int] = None


    kv_len_prev_tensor: torch.Tensor = None
    kv_len_next_tensor: torch.Tensor = None
    actual_seq_q_prev_tensor: torch.Tensor = None
    actual_seq_q_next_tensor: torch.Tensor = None
    cu_seqlens_q_prev_tensor: torch.Tensor = None
    cu_seqlens_q_next_tensor: torch.Tensor = None


    total_q_prev_tokens: int = 0
    total_q_next_tokens: int = 0
    max_seqlen_q_prev: int = 0
    max_seqlen_q_next: int = 0


    kv_len_prev_list: List[int] = None
    kv_len_next_list: List[int] = None
    actual_seq_q_prev_list: List[int] = None
    actual_seq_q_next_list: List[int] = None


    total_seq_lens: int = 0
    bs: int = 1


def is_prefill_context_parallel_enabled():
    return get_parallel().config.enable_prefill_context_parallel


def is_prefill_cp_in_seq_split():
    return (
        is_prefill_context_parallel_enabled()
        and get_parallel().config.prefill_cp_mode == "in-seq-split"
    )


def is_mla_prefill_cp_enabled() -> bool:
    return get_parallel().config.enable_prefill_context_parallel and uses_mla_backend()


def mla_use_prefill_cp(forward_batch, mla_enable_prefill_cp=None):
    if mla_enable_prefill_cp is None:
        mla_enable_prefill_cp = is_mla_prefill_cp_enabled()
    return (
        forward_batch.attn_cp_metadata is not None
        and mla_enable_prefill_cp
        and forward_batch.forward_mode.is_context_parallel_extend()
    )


def can_cp_split(seq_len: int, cp_size: int, forward_batch):


    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    cur_cp_seq_len = seq_len // (cp_size * 2)
    if not (
        cur_cp_seq_len != 0
        and cp_size > 1


        and forward_batch.forward_mode.is_context_parallel_extend()


        and forward_batch.forward_mode != ForwardMode.MIXED
        and is_prefill_context_parallel_enabled()
    ):
        return False


    extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
    if extend_lens is None:
        return True

    cp_min = cp_size * 2
    for L in extend_lens:
        if L < cp_min:


            return False

    return True


def cp_split_and_rebuild_data(forward_batch, input_: torch.Tensor):
    from sglang.srt.layers.attention.dsa.utils import (
        dsa_cp_round_robin_split_data,
        is_dsa_prefill_cp_round_robin_split,
    )

    if is_dsa_prefill_cp_round_robin_split():
        cp_size = get_parallel().attn_cp_size
        assert (
            input_.shape[0] % cp_size == 0
        ), f"Expect input shape 0 can divided by cp size, but got input shape {input_.shape}, cp size {cp_size}"
        return dsa_cp_round_robin_split_data(input_)

    input_list = list(
        torch.split(input_, forward_batch.attn_cp_metadata.split_list, dim=0)
    )
    result = torch.cat(
        [input_list[i] for i in forward_batch.attn_cp_metadata.zigzag_index], dim=0
    ).view(-1, input_.shape[-1])
    return result


def cp_split_and_rebuild_position(forward_batch, positions: torch.Tensor):
    from sglang.srt.layers.attention.dsa.utils import (
        dsa_cp_round_robin_split_data,
        is_dsa_prefill_cp_round_robin_split,
    )

    if is_dsa_prefill_cp_round_robin_split():
        cp_size = get_parallel().attn_cp_size
        assert positions.shape[0] % cp_size == 0, (
            f"Expect positions shape 0 can divided by cp size, but got positions shape {positions.shape}, "
            f"cp size {cp_size}"
        )
        return dsa_cp_round_robin_split_data(positions)

    position_id_list = list(
        torch.split(positions, forward_batch.attn_cp_metadata.split_list, dim=-1)
    )
    positions = torch.cat(
        [position_id_list[i] for i in forward_batch.attn_cp_metadata.zigzag_index],
        dim=-1,
    )
    return positions


def cp_round_robin_input_ids(input_ids):

    cp_size = get_parallel().attn_cp_size
    cp_rank = get_parallel().attn_cp_rank
    if get_moe_a2a_backend().is_none():
        input_ids = input_ids.reshape(-1, cp_size).T.flatten()
    else:
        input_ids = input_ids[cp_rank::cp_size].contiguous()
    return input_ids


def cp_all_gather_reorganized_into_tensor(input_tensor, cp_size, forward_batch, stream):

    max_len = forward_batch.attn_cp_metadata.max_rank_len[0]
    pad_size = max_len - input_tensor.shape[0]
    if pad_size > 0:
        input_tensor = F.pad(
            input_tensor, (0, 0, 0, pad_size), mode="constant", value=0
        )
    group = get_parallel().attn_cp_group
    with use_symmetric_memory(group, disabled=not is_allocation_symmetric()):
        input_tensor_full = torch.empty(
            max_len * cp_size,
            input_tensor.shape[1],
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        )

    group.all_gather_into_tensor(input_tensor_full, input_tensor)

    outputs_list_max = list(
        torch.split(
            input_tensor_full, forward_batch.attn_cp_metadata.max_rank_len, dim=0
        )
    )
    outputs = torch.cat(
        [
            outputs_list_max[index][:per_rank_len]
            for index, per_rank_len in enumerate(
                forward_batch.attn_cp_metadata.per_rank_actual_token
            )
        ],
        dim=0,
    )

    return outputs


def cp_all_gather_reorganized_into_tensor_kv_cache(
    input_tensor, cp_size, forward_batch, stream
):

    max_len = forward_batch.attn_cp_metadata.max_rank_len[0]
    pad_size = max_len - input_tensor.shape[0]
    if pad_size > 0:


        padding = [0, 0] * (input_tensor.ndim - 1) + [0, pad_size]
        input_tensor = F.pad(input_tensor, padding, mode="constant", value=0)


    group = get_parallel().attn_cp_group
    with use_symmetric_memory(group, disabled=not is_allocation_symmetric()):
        input_tensor_full = torch.empty(
            max_len * cp_size,
            *input_tensor.shape[1:],
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        )

    group.all_gather_into_tensor(input_tensor_full, input_tensor)

    outputs_list_max = list(
        torch.split(
            input_tensor_full, forward_batch.attn_cp_metadata.max_rank_len, dim=0
        )
    )
    outputs = torch.cat(
        [
            outputs_list_max[index][:per_rank_len]
            for index, per_rank_len in enumerate(
                forward_batch.attn_cp_metadata.per_rank_actual_token
            )
        ],
        dim=0,
    )

    return outputs


def cp_all_gather_rerange_launch(input_tensor, cp_size, comm_stream, event_key):

    from sglang.srt.distributed.parallel_state import (
        get_attn_cp_group,
        get_attn_cp_overlap_group,
    )

    group = get_attn_cp_overlap_group()
    assert group is not get_attn_cp_group(), (
        "the comm-stream path needs the duplicate attn_cp_overlap communicator; "
        "driving one communicator from two streams deadlocks RCCL"
    )

    input_tensor = input_tensor.contiguous()
    with use_symmetric_memory(group, disabled=not is_allocation_symmetric()):
        output_tensor = input_tensor.new_empty(
            (input_tensor.shape[0] * cp_size, *input_tensor.shape[1:]),
        )
    comm_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(comm_stream):
        attn_cp_overlap_all_gather_into_tensor(output_tensor, input_tensor)
        event = _tbo_event(event_key)
        event.record(comm_stream)
    return (output_tensor, input_tensor, event, cp_size)


def cp_all_gather_rerange_finish(handle):

    output_tensor, _keepalive, event, cp_size = handle
    torch.cuda.current_stream().wait_event(event)
    out_shape = output_tensor.shape
    return (
        output_tensor.view(cp_size, -1, *out_shape[1:])
        .transpose(0, 1)
        .reshape(out_shape)
    )


def cp_all_gather_rerange_output(input_tensor, cp_size, forward_batch, stream):

    from sglang.srt.layers.attention.dsa.utils import (
        is_dsa_prefill_cp_round_robin_split,
    )

    if is_dsa_prefill_cp_round_robin_split():
        with use_symmetric_memory(
            get_parallel().attn_cp_group, disabled=not is_allocation_symmetric()
        ):
            output_tensor = input_tensor.new_empty(
                (input_tensor.shape[0] * cp_size, *input_tensor.shape[1:]),
            )
        attn_cp_all_gather_into_tensor(
            output_tensor,
            input_tensor,
        )
        out_shape = output_tensor.shape
        output_tensor = (
            output_tensor.view(cp_size, -1, *out_shape[1:])
            .transpose(0, 1)
            .reshape(out_shape)
        )
        return output_tensor


    bs_seq_len, hidden_size = input_tensor.shape
    output_tensor = cp_all_gather_reorganized_into_tensor(
        input_tensor,
        cp_size,
        forward_batch,
        stream,
    )
    outputs_list = list(
        torch.split(
            output_tensor, forward_batch.attn_cp_metadata.reverse_split_len, dim=0
        )
    )
    output_tensor = torch.cat(
        [outputs_list[i] for i in forward_batch.attn_cp_metadata.cp_reverse_index],
        dim=0,
    )
    output_tensor = output_tensor.view(-1, hidden_size)
    return output_tensor


def cp_all_gather_rerange_kv_cache(input_tensor, cp_size, forward_batch, stream):

    output_tensor = cp_all_gather_reorganized_into_tensor_kv_cache(
        input_tensor,
        cp_size,
        forward_batch,
        stream,
    )
    outputs_list = list(
        torch.split(
            output_tensor, forward_batch.attn_cp_metadata.reverse_split_len, dim=0
        )
    )
    output_tensor = torch.cat(
        [outputs_list[i] for i in forward_batch.attn_cp_metadata.cp_reverse_index],
        dim=0,
    )

    return output_tensor


def cp_allgather_and_save_kv_cache(forward_batch, layer, k, v, cp_size, swa_loc=None):

    cache_loc = (
        forward_batch.out_cache_loc
        if not layer.is_cross_attention
        else forward_batch.encoder_out_cache_loc
    )

    k = k.contiguous()
    v = v.contiguous()

    key_cache_full = cp_all_gather_rerange_kv_cache(
        k, cp_size, forward_batch, torch.cuda.current_stream()
    )
    value_cache_full = cp_all_gather_rerange_kv_cache(
        v, cp_size, forward_batch, torch.cuda.current_stream()
    )

    get_token_to_kv_pool().set_kv_buffer(
        layer,
        KVWriteLoc(cache_loc, swa_loc),
        key_cache_full,
        value_cache_full,
        layer.k_scale,
        layer.v_scale,
    )


def cp_attn_forward_extend(
    forward_batch,
    q: torch.Tensor,
    device: torch.device,
    attn_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor],
) -> torch.Tensor:

    cp_meta = forward_batch.attn_cp_metadata

    q_prev = q[: cp_meta.total_q_prev_tokens]
    q_next = q[cp_meta.total_q_prev_tokens :]

    result_prev = attn_fn(
        q_prev,
        cp_meta.cu_seqlens_q_prev_tensor,
        cp_meta.kv_len_prev_tensor,
        cp_meta.max_seqlen_q_prev,
    )
    result_next = attn_fn(
        q_next,
        cp_meta.cu_seqlens_q_next_tensor,
        cp_meta.kv_len_next_tensor,
        cp_meta.max_seqlen_q_next,
    )

    return torch.concat([result_prev, result_next], dim=0)


def prepare_context_parallel_metadata(
    kv_len,
    cp_rank,
    cp_size,
    seqs_len,
    extend_seqs_len=None,
    device="cuda",
):
    from sglang.srt.layers.attention.dsa.utils import (
        is_dsa_prefill_cp_round_robin_split,
    )

    if is_dsa_prefill_cp_round_robin_split():
        return ContextParallelMetadata()

    """prepare_input_dp_with_cp_dsa-zigzag index
    Example (DP_ATTENT_TP == CP_SIZE == 4, single sequence):
        block0 | block1 | block2 | block3 | block4 | block5 | block6 | block7
        rank 0: block0, block7
        rank 1: block1, block6
        rank 2: block2, block5
        rank 3: block3, block4
    For bs > 1, each sequence is split into cp_segment_num = 2 * cp_size
    blocks independently; per-rank layout becomes:
        [s0.block_r, s1.block_r, ..., s_{bs-1}.block_r,
         s0.block_{2*cp_size-1-r}, ..., s_{bs-1}.block_{2*cp_size-1-r}]
    i.e. all prev blocks first, then all next blocks -- so torch.split at
    total_q_prev_tokens cleanly separates them.
    """
    assert extend_seqs_len is not None
    extend_seqs_len = [int(x) for x in extend_seqs_len]


    pad_len = int(kv_len) - sum(extend_seqs_len)
    if pad_len > 0:
        extend_seqs_len[-1] += pad_len
        if seqs_len is not None and len(seqs_len) == len(extend_seqs_len):
            seqs_len = list(seqs_len)
            seqs_len[-1] += pad_len

    bs = len(extend_seqs_len)
    cp_segment_num = cp_size * 2


    if seqs_len is not None and len(seqs_len) == bs:
        prefix_offsets = [
            max(int(seqs_len[s]) - extend_seqs_len[s], 0) for s in range(bs)
        ]
    else:
        prefix_offsets = [0] * bs


    per_seq_block_sizes: List[List[int]] = []
    split_list: List[int] = []
    for s in range(bs):
        L = extend_seqs_len[s]
        base = L // cp_segment_num
        rem = L % cp_segment_num
        blk = [base + 1 if i < rem else base for i in range(cp_segment_num)]
        per_seq_block_sizes.append(blk)
        split_list.extend(blk)


    per_rank_actual_token = [0] * cp_size
    for r in range(cp_size):
        total = 0
        for s in range(bs):
            total += (
                per_seq_block_sizes[s][r]
                + per_seq_block_sizes[s][cp_segment_num - 1 - r]
            )
        per_rank_actual_token[r] = total
    max_single_rank = max(per_rank_actual_token) if per_rank_actual_token else 0


    max_rank_len = [max_single_rank] * cp_size


    zigzag_index = list(
        range(cp_rank, cp_rank + bs * cp_segment_num, cp_segment_num)
    ) + list(
        range(
            cp_segment_num - cp_rank - 1,
            bs * cp_segment_num,
            cp_segment_num,
        )
    )


    cp_reverse_index: List[int] = []
    for batch_id in range(bs):
        cp_reverse_index.extend(
            list(range(batch_id, cp_segment_num * bs, 2 * bs))
            + list(
                range(
                    (cp_segment_num - 1) * bs + batch_id,
                    0,
                    -2 * bs,
                )
            )
        )


    reverse_split_len: List[int] = []
    for r in range(cp_size):
        for s in range(bs):
            reverse_split_len.append(per_seq_block_sizes[s][r])
        for s in range(bs):
            reverse_split_len.append(per_seq_block_sizes[s][cp_segment_num - 1 - r])


    from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp

    nsa_mode = is_dsa_enable_prefill_cp()
    kv_len_prev_list: List[int] = []
    kv_len_next_list: List[int] = []
    actual_seq_q_prev_list: List[int] = []
    actual_seq_q_next_list: List[int] = []
    for s in range(bs):
        blk = per_seq_block_sizes[s]
        cum_prev = sum(blk[: cp_rank + 1])
        cum_next = sum(blk[: cp_segment_num - cp_rank])


        if nsa_mode:
            kv_len_prev_list.append(cum_prev)
            kv_len_next_list.append(cum_next)
        else:
            kv_len_prev_list.append(prefix_offsets[s] + cum_prev)
            kv_len_next_list.append(prefix_offsets[s] + cum_next)
        actual_seq_q_prev_list.append(blk[cp_rank])
        actual_seq_q_next_list.append(blk[cp_segment_num - cp_rank - 1])


    kv_len_prev_tensor = torch.tensor(
        kv_len_prev_list, device=device, dtype=torch.int32
    )
    kv_len_next_tensor = torch.tensor(
        kv_len_next_list, device=device, dtype=torch.int32
    )
    actual_seq_q_prev_tensor = torch.tensor(
        actual_seq_q_prev_list, device=device, dtype=torch.int32
    )
    actual_seq_q_next_tensor = torch.tensor(
        actual_seq_q_next_list, device=device, dtype=torch.int32
    )
    cu_prev = [0] + list(accumulate(actual_seq_q_prev_list))
    cu_next = [0] + list(accumulate(actual_seq_q_next_list))
    cu_seqlens_q_prev_tensor = torch.tensor(cu_prev, device=device, dtype=torch.int32)
    cu_seqlens_q_next_tensor = torch.tensor(cu_next, device=device, dtype=torch.int32)

    total_q_prev_tokens = cu_prev[-1]
    total_q_next_tokens = cu_next[-1]
    max_seqlen_q_prev = max(actual_seq_q_prev_list) if actual_seq_q_prev_list else 0
    max_seqlen_q_next = max(actual_seq_q_next_list) if actual_seq_q_next_list else 0
    total_seq_lens = sum(extend_seqs_len)


    assert len(split_list) == bs * cp_segment_num
    assert sum(split_list) == total_seq_lens
    assert len(zigzag_index) == 2 * bs
    assert len(cp_reverse_index) == bs * cp_segment_num
    assert sorted(cp_reverse_index) == list(range(bs * cp_segment_num))
    assert sum(per_rank_actual_token) == total_seq_lens

    return ContextParallelMetadata(
        split_list=split_list,
        zigzag_index=zigzag_index,
        cp_reverse_index=cp_reverse_index,
        reverse_split_len=reverse_split_len,
        per_rank_actual_token=per_rank_actual_token,
        max_rank_len=max_rank_len,
        kv_len_prev_tensor=kv_len_prev_tensor,
        kv_len_next_tensor=kv_len_next_tensor,
        actual_seq_q_prev_tensor=actual_seq_q_prev_tensor,
        actual_seq_q_next_tensor=actual_seq_q_next_tensor,
        cu_seqlens_q_prev_tensor=cu_seqlens_q_prev_tensor,
        cu_seqlens_q_next_tensor=cu_seqlens_q_next_tensor,
        total_q_prev_tokens=total_q_prev_tokens,
        total_q_next_tokens=total_q_next_tokens,
        max_seqlen_q_prev=max_seqlen_q_prev,
        max_seqlen_q_next=max_seqlen_q_next,
        kv_len_prev_list=kv_len_prev_list,
        kv_len_next_list=kv_len_next_list,
        actual_seq_q_prev_list=actual_seq_q_prev_list,
        actual_seq_q_next_list=actual_seq_q_next_list,
        total_seq_lens=total_seq_lens,
        bs=bs,
    )
