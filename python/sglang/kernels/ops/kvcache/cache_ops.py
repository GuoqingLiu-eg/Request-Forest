import torch
import triton
import triton.language as tl


@triton.jit
def concat_and_cast_mha_k_kernel(
    k_ptr,
    k_nope_ptr,
    k_rope_ptr,
    head_cnt: tl.constexpr,
    k_stride0: tl.constexpr,
    k_stride1: tl.constexpr,
    nope_stride0: tl.constexpr,
    nope_stride1: tl.constexpr,
    rope_stride0: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
):
    pid_loc = tl.program_id(0)
    head_range = tl.arange(0, head_cnt)

    k_head_ptr = k_ptr + pid_loc * k_stride0 + head_range[:, None] * k_stride1

    nope_offs = tl.arange(0, nope_dim)

    src_nope_ptr = (
        k_nope_ptr
        + pid_loc * nope_stride0
        + head_range[:, None] * nope_stride1
        + nope_offs[None, :]
    )
    dst_nope_ptr = k_head_ptr + nope_offs[None, :]

    src_nope = tl.load(src_nope_ptr)
    tl.store(dst_nope_ptr, src_nope)

    rope_offs = tl.arange(0, rope_dim)
    src_rope_ptr = k_rope_ptr + pid_loc * rope_stride0 + rope_offs[None, :]
    dst_rope_ptr = k_head_ptr + nope_dim + rope_offs[None, :]
    src_rope = tl.load(src_rope_ptr)
    tl.store(dst_rope_ptr, src_rope)


@triton.jit
def concat_and_cast_mha_k_pad_kernel(
    k_ptr,
    k_nope_ptr,
    k_rope_ptr,
    head_cnt: tl.constexpr,
    k_stride0: tl.constexpr,
    k_stride1: tl.constexpr,
    nope_stride0: tl.constexpr,
    nope_stride1: tl.constexpr,
    rope_stride0: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
):

    pid_loc = tl.program_id(0)
    head_range = tl.arange(0, HEAD_BLOCK)
    head_mask = head_range < head_cnt

    k_head_ptr = k_ptr + pid_loc * k_stride0 + head_range[:, None] * k_stride1

    nope_offs = tl.arange(0, nope_dim)

    src_nope_ptr = (
        k_nope_ptr
        + pid_loc * nope_stride0
        + head_range[:, None] * nope_stride1
        + nope_offs[None, :]
    )
    dst_nope_ptr = k_head_ptr + nope_offs[None, :]

    nope_mask = tl.broadcast_to(head_mask[:, None], (HEAD_BLOCK, nope_dim))
    src_nope = tl.load(src_nope_ptr, mask=nope_mask, other=0)
    tl.store(dst_nope_ptr, src_nope, mask=nope_mask)

    rope_offs = tl.arange(0, rope_dim)
    src_rope_ptr = k_rope_ptr + pid_loc * rope_stride0 + rope_offs[None, :]
    dst_rope_ptr = k_head_ptr + nope_dim + rope_offs[None, :]
    rope_mask = tl.broadcast_to(head_mask[:, None], (HEAD_BLOCK, rope_dim))
    src_rope = tl.load(src_rope_ptr)
    tl.store(
        dst_rope_ptr,
        tl.broadcast_to(src_rope, (HEAD_BLOCK, rope_dim)),
        mask=rope_mask,
    )


def concat_and_cast_mha_k_triton(
    k: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
):

    assert (
        len(k.shape) == 3 and len(k_nope.shape) == 3 and len(k_rope.shape) == 3
    ), f"shape should be 3d, but got {k.shape=}, {k_nope.shape=}, {k_rope.shape=}"
    assert (
        k.shape[0] == k_nope.shape[0] and k.shape[0] == k_rope.shape[0]
    ), f"invalid shape, got {k.shape=}, {k_nope.shape=}, {k_rope.shape=}"
    assert (
        k.shape[1] == k_nope.shape[1] and 1 == k_rope.shape[1]
    ), f"invalid shape, got {k.shape=}, {k_nope.shape=}, {k_rope.shape=}"
    assert (
        k.shape[-1] == k_nope.shape[-1] + k_rope.shape[-1]
    ), f"invalid shape, got {k.shape=}, {k_nope.shape=}, {k_rope.shape=}"

    nope_dim = k_nope.shape[-1]
    rope_dim = k_rope.shape[-1]
    grid = (k.shape[0],)

    head_cnt = k.shape[1]
    if head_cnt != triton.next_power_of_2(head_cnt):


        concat_and_cast_mha_k_pad_kernel[grid](
            k,
            k_nope,
            k_rope,
            head_cnt,
            k.stride(0),
            k.stride(1),
            k_nope.stride(0),
            k_nope.stride(1),
            k_rope.stride(0),
            nope_dim,
            rope_dim,
            HEAD_BLOCK=triton.next_power_of_2(head_cnt),
        )
        return

    concat_and_cast_mha_k_kernel[grid](
        k,
        k_nope,
        k_rope,
        k.shape[1],
        k.stride(0),
        k.stride(1),
        k_nope.stride(0),
        k_nope.stride(1),
        k_rope.stride(0),
        nope_dim,
        rope_dim,
    )


@triton.jit
def reshape_and_cache_flash(
    key_ptr,
    value_ptr,
    key_cache_ptr,
    value_cache_ptr,
    slot_mapping_ptr,
    swa_slot_mapping_ptr,
    k_scale_ptr,
    v_scale_ptr,
    block_stride,
    key_stride,
    value_stride,
    num_heads,
    head_size,
    block_size,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_SWA: tl.constexpr,
    USE_SCALE: tl.constexpr,
):


    token_idx = tl.program_id(0)
    head_block_idx = tl.program_id(1)


    slot_idx = tl.load(slot_mapping_ptr + token_idx)

    if HAS_SWA:
        slot_idx = tl.load(swa_slot_mapping_ptr + slot_idx)

    if slot_idx < 0:
        return

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size


    head_idx = head_block_idx * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)

    head_mask = head_idx < num_heads

    dim_idx = tl.arange(0, BLOCK_D)


    offs = head_idx[:, None] * head_size + dim_idx[None, :]

    mask = head_mask[:, None] & (dim_idx[None, :] < head_size)


    src_key = token_idx * key_stride + offs
    src_value = token_idx * value_stride + offs

    k = tl.load(key_ptr + src_key, mask=mask)
    v = tl.load(value_ptr + src_value, mask=mask)


    if USE_SCALE:
        k_scale = tl.load(k_scale_ptr)
        v_scale = tl.load(v_scale_ptr)

        k = k / k_scale
        v = v / v_scale


    tgt = block_idx * block_stride + block_offset * num_heads * head_size + offs

    tl.store(key_cache_ptr + tgt, k, mask=mask)
    tl.store(value_cache_ptr + tgt, v, mask=mask)


def launch_reshape_and_cache_flash(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    swa_slot_mapping=None,
    k_scale=None,
    v_scale=None,
):


    num_tokens = key.shape[0]
    num_heads = key.shape[1]
    head_size = key.shape[2]

    HEAD_BLOCK = 4

    BLOCK_D = triton.next_power_of_2(head_size)

    grid = (
        num_tokens,
        triton.cdiv(num_heads, HEAD_BLOCK),
    )

    reshape_and_cache_flash[grid](
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        swa_slot_mapping,
        k_scale if k_scale is not None else key,
        v_scale if v_scale is not None else key,
        key_cache.stride(0),
        key.stride(0),
        value.stride(0),
        num_heads,
        head_size,
        key_cache.shape[1],
        HEAD_BLOCK=HEAD_BLOCK,
        BLOCK_D=BLOCK_D,
        HAS_SWA=(swa_slot_mapping is not None),
        USE_SCALE=(k_scale is not None),
    )


@triton.jit
def concat_and_cast_q_fp8_pad_kernel(
    qpad_ptr,
    q_nope_ptr,
    q_rope_ptr,
    qpad_s0,
    qpad_s1,
    nope_s0,
    nope_s1,
    rope_s0,
    rope_s1,
    H: tl.constexpr,
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
):


    pid = tl.program_id(0)
    hr = tl.arange(0, H)
    qpad_head = qpad_ptr + pid * qpad_s0 + hr[:, None] * qpad_s1
    no = tl.arange(0, NOPE)
    src_n = tl.load(q_nope_ptr + pid * nope_s0 + hr[:, None] * nope_s1 + no[None, :])
    tl.store(qpad_head + no[None, :], src_n)
    ro = tl.arange(0, ROPE)
    src_r = tl.load(q_rope_ptr + pid * rope_s0 + hr[:, None] * rope_s1 + ro[None, :])
    tl.store(qpad_head + NOPE + ro[None, :], src_r)


def concat_and_cast_q_fp8_pad(q_fp8_pad, q_nope, q_rope, num_heads):

    num_tokens = q_nope.shape[0]
    nope_dim = q_nope.shape[-1]
    rope_dim = q_rope.shape[-1]
    concat_and_cast_q_fp8_pad_kernel[(num_tokens,)](
        q_fp8_pad,
        q_nope,
        q_rope,
        q_fp8_pad.stride(0),
        q_fp8_pad.stride(1),
        q_nope.stride(0),
        q_nope.stride(1),
        q_rope.stride(0),
        q_rope.stride(1),
        H=num_heads,
        NOPE=nope_dim,
        ROPE=rope_dim,
    )


@triton.jit
def absorbed_bmm_concat_cast_q_fp8_kernel(
    qout_ptr,
    a_ptr,
    b_ptr,
    rope_ptr,
    T,
    qout_s0,
    qout_s1,
    a_s0,
    a_s1,
    b_s0,
    b_s1,
    b_s2,
    rope_s0,
    rope_s1,
    K: tl.constexpr,
    N: tl.constexpr,
    ROPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_MODE: tl.constexpr,
):


    pid_m = tl.program_id(0)
    h = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < T


    offs_m64 = offs_m.to(tl.int64)
    qout_head = qout_ptr + offs_m64[:, None] * qout_s0 + h * qout_s1
    a_row = a_ptr + offs_m64[:, None] * a_s0 + h * a_s1
    b_head = b_ptr + h * b_s0
    if K_MODE == 0:

        offs_k = tl.arange(0, BLOCK_K)
        a = tl.load(a_row + offs_k[None, :], mask=m_mask[:, None], other=0.0)
        for nb in tl.static_range(N // BLOCK_N):
            offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
            b = tl.load(b_head + offs_k[:, None] * b_s1 + offs_n[None, :] * b_s2)
            acc = tl.dot(a, b)
            val = acc.to(tl.bfloat16)

            tl.store(qout_head + offs_n[None, :], val, mask=m_mask[:, None])
    elif K_MODE == 2:

        offs_k0 = tl.arange(0, BLOCK_K)
        offs_k1 = BLOCK_K + tl.arange(0, K - BLOCK_K)
        a0 = tl.load(a_row + offs_k0[None, :], mask=m_mask[:, None], other=0.0)
        a1 = tl.load(a_row + offs_k1[None, :], mask=m_mask[:, None], other=0.0)
        for nb in tl.static_range(N // BLOCK_N):
            offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
            b0 = tl.load(b_head + offs_k0[:, None] * b_s1 + offs_n[None, :] * b_s2)
            b1 = tl.load(b_head + offs_k1[:, None] * b_s1 + offs_n[None, :] * b_s2)
            acc = tl.dot(a0, b0)
            acc = tl.dot(a1, b1, acc)
            val = acc.to(tl.bfloat16)

            tl.store(qout_head + offs_n[None, :], val, mask=m_mask[:, None])
    elif K_MODE == 3:

        offs_k0 = tl.arange(0, BLOCK_K)
        offs_k1 = BLOCK_K + offs_k0
        offs_k2 = 2 * BLOCK_K + offs_k0
        a0 = tl.load(a_row + offs_k0[None, :], mask=m_mask[:, None], other=0.0)
        a1 = tl.load(a_row + offs_k1[None, :], mask=m_mask[:, None], other=0.0)
        a2 = tl.load(a_row + offs_k2[None, :], mask=m_mask[:, None], other=0.0)
        for nb in tl.static_range(N // BLOCK_N):
            offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
            b0 = tl.load(b_head + offs_k0[:, None] * b_s1 + offs_n[None, :] * b_s2)
            b1 = tl.load(b_head + offs_k1[:, None] * b_s1 + offs_n[None, :] * b_s2)
            b2 = tl.load(b_head + offs_k2[:, None] * b_s1 + offs_n[None, :] * b_s2)
            acc = tl.dot(a0, b0)
            acc = tl.dot(a1, b1, acc)
            acc = tl.dot(a2, b2, acc)
            val = acc.to(tl.bfloat16)

            tl.store(qout_head + offs_n[None, :], val, mask=m_mask[:, None])
    elif K_MODE == 4:

        offs_k = tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        a = tl.load(
            a_row + offs_k[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        for nb in tl.static_range(N // BLOCK_N):
            offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
            b = tl.load(
                b_head + offs_k[:, None] * b_s1 + offs_n[None, :] * b_s2,
                mask=k_mask[:, None],
                other=0.0,
            )
            acc = tl.dot(a, b)
            val = acc.to(tl.bfloat16)

            tl.store(qout_head + offs_n[None, :], val, mask=m_mask[:, None])
    else:

        for nb in tl.static_range(N // BLOCK_N):
            offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for kb in tl.static_range(K // BLOCK_K):
                offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
                a = tl.load(a_row + offs_k[None, :], mask=m_mask[:, None], other=0.0)
                b = tl.load(b_head + offs_k[:, None] * b_s1 + offs_n[None, :] * b_s2)
                acc = tl.dot(a, b, acc)
            val = acc.to(tl.bfloat16)

            tl.store(qout_head + offs_n[None, :], val, mask=m_mask[:, None])
    offs_r = tl.arange(0, ROPE)
    r = tl.load(
        rope_ptr + offs_m64[:, None] * rope_s0 + h * rope_s1 + offs_r[None, :],
        mask=m_mask[:, None],
        other=0.0,
    )
    tl.store(qout_head + N + offs_r[None, :], r, mask=m_mask[:, None])


_AUTO_NONPOW2_VARIANT = "two_dot"


def _qprep_env_variant():
    from sglang.srt.environ import envs

    return envs.SGLANG_OPT_Q8KV8_QPREP_VARIANT.get()


_ENV_QPREP_VARIANT = None


def absorbed_bmm_concat_cast_q_fp8(
    q_fp8_pad: "torch.Tensor",
    q_nope: "torch.Tensor",
    w_kc: "torch.Tensor",
    q_rope: "torch.Tensor",
    num_heads: int,
    block_m: int = 128,
    block_n: int = 64,
    variant: str = "auto",
    block_k: int = 0,
    num_warps: int = 8,
    num_stages: int = 0,
):

    num_tokens, _, k_dim = q_nope.shape
    n_dim = w_kc.shape[-1]
    rope_dim = q_rope.shape[-1]
    assert q_fp8_pad.dtype == torch.float8_e4m3fn
    assert q_nope.dtype == torch.bfloat16 and w_kc.dtype == torch.bfloat16
    assert q_rope.dtype == torch.bfloat16
    assert q_nope.shape[1] == num_heads and q_rope.shape[1] == num_heads
    assert w_kc.shape[0] == num_heads and w_kc.shape[1] == k_dim
    assert q_fp8_pad.shape[0] >= num_tokens and q_fp8_pad.shape[1] >= num_heads
    assert q_fp8_pad.shape[2] == n_dim + rope_dim

    assert (
        k_dim % 16 == 0 and 16 <= k_dim <= 256
    ), "K must be a multiple of 16 in [16, 256]"
    assert (rope_dim & (rope_dim - 1)) == 0, "ROPE must be a power of two"
    assert n_dim % block_n == 0, "N must be a multiple of block_n"
    assert q_nope.stride(2) == 1 and q_rope.stride(2) == 1
    assert q_fp8_pad.stride(2) == 1


    global _ENV_QPREP_VARIANT
    if _ENV_QPREP_VARIANT is None:
        _ENV_QPREP_VARIANT = _qprep_env_variant()
    if variant == "auto" and _ENV_QPREP_VARIANT != "auto":
        variant = _ENV_QPREP_VARIANT


    _valid = ("auto", "cuda", "loop", "two_dot", "three_dot", "pad", "single_k")
    if variant not in _valid:
        raise ValueError(
            f"unknown q-prep variant {variant!r} "
            f"(SGLANG_OPT_Q8KV8_QPREP_VARIANT); valid: {_valid}"
        )
    if variant == "cuda":
        from sglang.kernels.ops.attention.qprep_bf16_fp8_sm90 import q8kv8_qprep_fwd

        q8kv8_qprep_fwd(q_fp8_pad, q_nope, w_kc, q_rope, num_heads)
        return


    if k_dim & (k_dim - 1) == 0:

        k_mode, blk_k = 0, k_dim
    else:
        v = _AUTO_NONPOW2_VARIANT if variant == "auto" else variant
        if v == "loop":


            blk_k = block_k or min(k_dim & -k_dim, 128)
            assert (
                k_dim % blk_k == 0 and blk_k & (blk_k - 1) == 0 and blk_k >= 16
            ), "loop needs BLOCK_K a power-of-2 divisor of K >= 16"
            k_mode = 1
        elif v == "two_dot":
            blk_k = 1 << (k_dim.bit_length() - 1)
            k1 = k_dim - blk_k
            assert (
                k1 & (k1 - 1) == 0 and k1 >= 16
            ), "two_dot needs K = pow2 + pow2 with both halves >= 16"
            k_mode = 2
        elif v == "three_dot":
            blk_k = k_dim // 3
            assert (
                k_dim % 3 == 0 and blk_k & (blk_k - 1) == 0 and blk_k >= 16
            ), "three_dot needs K = 3 * pow2 with pow2 >= 16"
            k_mode = 3
        elif v == "pad":
            blk_k = 1 << k_dim.bit_length()
            k_mode = 4
        elif v == "single_k":


            blk_k = k_dim
            k_mode = 0
        else:
            raise ValueError(f"unknown absorbed-bmm K variant: {variant!r}")
    extra = {"num_stages": num_stages} if num_stages else {}
    grid = (triton.cdiv(num_tokens, block_m), num_heads)
    absorbed_bmm_concat_cast_q_fp8_kernel[grid](
        q_fp8_pad,
        q_nope,
        w_kc,
        q_rope,
        num_tokens,
        q_fp8_pad.stride(0),
        q_fp8_pad.stride(1),
        q_nope.stride(0),
        q_nope.stride(1),
        w_kc.stride(0),
        w_kc.stride(1),
        w_kc.stride(2),
        q_rope.stride(0),
        q_rope.stride(1),
        K=k_dim,
        N=n_dim,
        ROPE=rope_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=blk_k,
        K_MODE=k_mode,
        num_warps=num_warps,
        **extra,
    )


@triton.jit
def q8kv8_topk_length_backscan_kernel(
    indices_ptr,
    out_ptr,
    stride_row,
    topk,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    base = indices_ptr + row * stride_row
    off = topk
    length = 1
    found = 0
    while (found == 0) & (off > 0):
        off -= BLOCK
        idx = off + tl.arange(0, BLOCK)
        vals = tl.load(base + idx)
        pos = tl.max(tl.where(vals >= 0, idx, -1), axis=0)
        found = tl.where(pos >= 0, 1, found)
        length = tl.where(pos >= 0, pos + 1, length)
    tl.store(out_ptr + row, length)


def q8kv8_topk_length_from_indices(indices: torch.Tensor) -> torch.Tensor:

    s_q, topk = indices.shape
    assert indices.dtype == torch.int32 and indices.stride(1) == 1
    out = torch.empty(s_q, dtype=torch.int32, device=indices.device)
    block = 512 if topk % 512 == 0 else (256 if topk % 256 == 0 else 128)
    q8kv8_topk_length_backscan_kernel[(s_q,)](
        indices,
        out,
        indices.stride(0),
        topk,
        BLOCK=block,
    )
    return out
