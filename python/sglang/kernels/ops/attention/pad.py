import torch
import triton
import triton.language as tl


@triton.jit
def pad_sequence_with_mask_kernel(
    input_ptr,
    offsets_ptr,
    lengths_ptr,
    output_ptr,
    mask_ptr,
    max_len,
    hidden_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    m = tl.program_id(1)

    offset = tl.load(offsets_ptr + b)
    length = tl.load(lengths_ptr + b)

    seq_ids = m * BLOCK_M + tl.arange(0, BLOCK_M)
    hid_ids = tl.arange(0, BLOCK_D)

    seq_mask = seq_ids < max_len
    valid_token = seq_ids < length


    in_token = offset + seq_ids
    in_ptr = input_ptr + in_token[:, None] * hidden_dim + hid_ids[None, :]


    out_ptr = (
        output_ptr
        + b * max_len * hidden_dim
        + seq_ids[:, None] * hidden_dim
        + hid_ids[None, :]
    )

    values = tl.load(
        in_ptr,
        mask=valid_token[:, None] & (hid_ids[None, :] < hidden_dim),
        other=0.0,
    )

    tl.store(
        out_ptr,
        values,
        mask=seq_mask[:, None] & (hid_ids[None, :] < hidden_dim),
    )


    if tl.program_id(2) == 0:
        mask_out_ptr = mask_ptr + b * max_len + seq_ids
        tl.store(mask_out_ptr, valid_token, mask=seq_mask)


def pad_sequence_with_mask(
    input_emb,
    offsets,
    lengths,
    max_len,
):
    B = offsets.shape[0]
    hidden_dim = input_emb.shape[1]

    output = torch.zeros(
        (B, max_len, hidden_dim),
        device=input_emb.device,
        dtype=input_emb.dtype,
    )
    attn_mask = torch.empty(
        (B * max_len),
        device=input_emb.device,
        dtype=torch.bool,
    )

    BLOCK_D = triton.next_power_of_2(hidden_dim)
    BLOCK_M = triton.next_power_of_2(max_len)

    grid = (
        B,
        triton.cdiv(max_len, BLOCK_M),
        1,
    )

    pad_sequence_with_mask_kernel[grid](
        input_emb,
        offsets,
        lengths,
        output,
        attn_mask,
        max_len,
        hidden_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
    )

    return B, output, attn_mask


@triton.jit
def pad_draft_extend_query_kernel(
    q_ptr,
    padded_q_ptr,
    seq_lens_q_ptr,
    cumsum_ptr,
    batch_size,
    max_seq_len,
    num_heads,
    head_dim,
    BLOCK_SIZE: tl.constexpr,
):


    batch_seq_pid = tl.program_id(0)
    head_pid = tl.program_id(1)
    dim_pid = tl.program_id(2)

    batch_id = batch_seq_pid // max_seq_len
    seq_pos = batch_seq_pid % max_seq_len

    if batch_id >= batch_size:
        return


    seq_len = tl.load(seq_lens_q_ptr + batch_id)

    if seq_pos >= seq_len:
        return


    input_start = tl.load(cumsum_ptr + batch_id)
    input_pos = input_start + seq_pos


    head_start = head_pid * BLOCK_SIZE
    head_end = tl.minimum(head_start + BLOCK_SIZE, num_heads)
    head_mask = tl.arange(0, BLOCK_SIZE) < (head_end - head_start)

    dim_start = dim_pid * BLOCK_SIZE
    dim_end = tl.minimum(dim_start + BLOCK_SIZE, head_dim)
    dim_mask = tl.arange(0, BLOCK_SIZE) < (dim_end - dim_start)


    input_offset = (
        input_pos * num_heads * head_dim
        + (head_start + tl.arange(0, BLOCK_SIZE))[:, None] * head_dim
        + (dim_start + tl.arange(0, BLOCK_SIZE))[None, :]
    )


    data = tl.load(
        q_ptr + input_offset,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )


    output_offset = (
        batch_id * max_seq_len * num_heads * head_dim
        + seq_pos * num_heads * head_dim
        + (head_start + tl.arange(0, BLOCK_SIZE))[:, None] * head_dim
        + (dim_start + tl.arange(0, BLOCK_SIZE))[None, :]
    )


    tl.store(
        padded_q_ptr + output_offset,
        data,
        mask=head_mask[:, None] & dim_mask[None, :],
    )


def pad_draft_extend_query(
    q: torch.Tensor,
    padded_q: torch.Tensor,
    seq_lens_q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
) -> torch.Tensor:

    batch_size = cu_seqlens_q.shape[0] - 1
    max_seq_len_q = padded_q.shape[1]
    num_heads = padded_q.shape[2]
    head_dim = padded_q.shape[3]


    BLOCK_SIZE = 64
    num_head_blocks = triton.cdiv(num_heads, BLOCK_SIZE)
    num_dim_blocks = triton.cdiv(head_dim, BLOCK_SIZE)
    grid = (batch_size * max_seq_len_q, num_head_blocks, num_dim_blocks)

    pad_draft_extend_query_kernel[grid](
        q_ptr=q,
        padded_q_ptr=padded_q,
        seq_lens_q_ptr=seq_lens_q,
        cumsum_ptr=cu_seqlens_q,
        batch_size=batch_size,
        max_seq_len=max_seq_len_q,
        num_heads=num_heads,
        head_dim=head_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return padded_q


@triton.jit
def unpad_draft_extend_output_kernel(
    raw_out_ptr,
    output_ptr,
    num_accept_tokens_ptr,
    cumsum_ptr,
    batch_size,
    token_per_batch,
    tp_q_head_num,
    v_head_dim,
    BLOCK_SIZE: tl.constexpr,
):

    batch_seq_pid = tl.program_id(0)
    head_pid = tl.program_id(1)
    dim_pid = tl.program_id(2)

    batch_id = batch_seq_pid // token_per_batch
    seq_pos = batch_seq_pid % token_per_batch

    if batch_id >= batch_size:
        return


    accept_len = tl.load(num_accept_tokens_ptr + batch_id)

    if seq_pos >= accept_len:
        return


    output_start = tl.load(cumsum_ptr + batch_id)
    output_pos = output_start + seq_pos


    head_start = head_pid * BLOCK_SIZE
    head_end = tl.minimum(head_start + BLOCK_SIZE, tp_q_head_num)
    head_mask = tl.arange(0, BLOCK_SIZE) < (head_end - head_start)

    dim_start = dim_pid * BLOCK_SIZE
    dim_end = tl.minimum(dim_start + BLOCK_SIZE, v_head_dim)
    dim_mask = tl.arange(0, BLOCK_SIZE) < (dim_end - dim_start)


    input_offset = (
        batch_id * token_per_batch * tp_q_head_num * v_head_dim
        + seq_pos * tp_q_head_num * v_head_dim
        + (head_start + tl.arange(0, BLOCK_SIZE))[:, None] * v_head_dim
        + (dim_start + tl.arange(0, BLOCK_SIZE))[None, :]
    )


    data = tl.load(
        raw_out_ptr + input_offset,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    output_offset = (
        output_pos * tp_q_head_num * v_head_dim
        + (head_start + tl.arange(0, BLOCK_SIZE))[:, None] * v_head_dim
        + (dim_start + tl.arange(0, BLOCK_SIZE))[None, :]
    )


    tl.store(
        output_ptr + output_offset,
        data,
        mask=head_mask[:, None] & dim_mask[None, :],
    )


def unpad_draft_extend_output(
    raw_out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens_q: torch.Tensor,
    sum_seq_lens_q: int,
    unpad_output_buffer: torch.Tensor | None = None,
) -> torch.Tensor:


    batch_size = seq_lens_q.shape[0]
    token_per_batch = raw_out.shape[1]
    tp_q_head_num = raw_out.shape[2]
    v_head_dim = raw_out.shape[3]
    total_tokens = sum_seq_lens_q


    if unpad_output_buffer is not None:

        output = unpad_output_buffer[:total_tokens, :, :].to(dtype=raw_out.dtype)
    else:

        output = torch.empty(
            (total_tokens, tp_q_head_num, v_head_dim),
            dtype=raw_out.dtype,
            device=raw_out.device,
        )


    BLOCK_SIZE = 64
    num_head_blocks = triton.cdiv(tp_q_head_num, BLOCK_SIZE)
    num_dim_blocks = triton.cdiv(v_head_dim, BLOCK_SIZE)
    grid = (batch_size * token_per_batch, num_head_blocks, num_dim_blocks)

    unpad_draft_extend_output_kernel[grid](
        raw_out_ptr=raw_out,
        output_ptr=output,
        num_accept_tokens_ptr=seq_lens_q,
        cumsum_ptr=cu_seqlens_q,
        batch_size=batch_size,
        token_per_batch=token_per_batch,
        tp_q_head_num=tp_q_head_num,
        v_head_dim=v_head_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output[:total_tokens, :, :]


@triton.jit
def seqlens_expand_kernel(
    extend_seq_lens_ptr,
    seq_lens_ptr,
    offsets_ptr,
    output_ptr,
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)

    if pid >= N:
        return

    qo_len = tl.load(extend_seq_lens_ptr + pid)
    kv_len = tl.load(seq_lens_ptr + pid)

    start = kv_len - qo_len + 1
    out_offset = tl.load(offsets_ptr + pid)

    offs = tl.arange(0, BLOCK)
    mask = offs < qo_len


    values = tl.maximum(start + offs, 0)
    tl.store(output_ptr + out_offset + offs, values, mask=mask)


def seqlens_expand_triton(
    extend_seq_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    total_len: int,
    max_q_len: int,
):

    assert extend_seq_lens.is_cuda
    assert seq_lens.is_cuda

    N = extend_seq_lens.numel()

    offsets = torch.zeros(N + 1, device=extend_seq_lens.device, dtype=torch.int32)
    offsets[1:] = torch.cumsum(extend_seq_lens, dim=0)
    output = torch.empty(total_len, device=extend_seq_lens.device, dtype=torch.int32)

    BLOCK = triton.next_power_of_2(max_q_len)
    grid = (N,)

    seqlens_expand_kernel[grid](
        extend_seq_lens,
        seq_lens,
        offsets,
        output,
        N,
        BLOCK=BLOCK,
    )

    return output
