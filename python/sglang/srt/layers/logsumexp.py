import torch
import triton
import triton.language as tl


FUSED_TOPK_MAX_K = 8


@triton.jit
def _accumulate_block_lse(x, m_i, l_i):

    m_blk = tl.max(x)
    l_blk = tl.sum(tl.exp(x - m_blk))
    l_blk = tl.where((m_blk == float("inf")) | (m_blk == float("-inf")), 1.0, l_blk)
    m_i, l_i = _combine_lse(m_i, l_i, m_blk, l_blk)
    return m_i, l_i, m_blk


@triton.jit
def _combine_lse(m_a, l_a, m_b, l_b):

    m_new = tl.maximum(m_a, m_b)
    l_new = l_a * tl.exp(tl.where(m_a == m_new, 0.0, m_a - m_new)) + l_b * tl.exp(
        tl.where(m_b == m_new, 0.0, m_b - m_new)
    )
    return m_new, l_new


@triton.jit
def _row_logsumexp_kernel(
    x_ptr,
    out_max_ptr,
    out_log_sum_ptr,
    row_stride,
    col_stride,

    num_cols,
    BLOCK_N: tl.constexpr,
):

    row = tl.program_id(0).to(tl.int64)
    row_ptr = x_ptr + row * row_stride

    m_i = float("-inf")
    l_i = 0.0
    for start in range(0, num_cols, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        x = tl.load(
            row_ptr + offs * col_stride, mask=offs < num_cols, other=float("-inf")
        ).to(tl.float32)
        m_i, l_i, _ = _accumulate_block_lse(x, m_i, l_i)

    tl.store(out_max_ptr + row, m_i)
    tl.store(out_log_sum_ptr + row, tl.log(l_i))


def row_logsumexp(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

    assert x.ndim == 2
    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32)

    num_rows, num_cols = x.shape
    out_max = torch.empty(num_rows, device=x.device, dtype=torch.float32)
    out_log_sum = torch.empty(num_rows, device=x.device, dtype=torch.float32)
    if num_rows == 0:
        return out_max, out_log_sum
    if num_cols == 0:
        return out_max.fill_(float("-inf")), out_log_sum.zero_()

    BLOCK_N = triton.next_power_of_2(min(num_cols, 16384))
    _row_logsumexp_kernel[(num_rows,)](
        x,
        out_max,
        out_log_sum,
        x.stride(0),
        x.stride(1),
        num_cols,
        BLOCK_N=BLOCK_N,
        num_warps=8,
    )
    return out_max, out_log_sum


@triton.jit
def _fpval_to_key(x):

    return x ^ tl.where((x & 0x80000000) != 0, 0xFFFFFFFF, 0x80000000)


@triton.jit
def _key_to_fpval(x):
    return x ^ tl.where((x & 0x80000000) == 0, 0xFFFFFFFF, 0x80000000)


@triton.jit
def _pack_key(vals, idxs):

    sortable = _fpval_to_key(vals.to(tl.uint32, bitcast=True))
    return (sortable.to(tl.int64) << 31) | (2147483647 - idxs).to(tl.int64)


@triton.jit
def _unpack_val(keys):
    sortable = ((keys >> 31) & 0xFFFFFFFF).to(tl.uint32)
    return _key_to_fpval(sortable).to(tl.float32, bitcast=True)


@triton.jit
def _unpack_idx(keys):
    return 2147483647 - (keys & 0x7FFFFFFF).to(tl.int32)


@triton.jit
def _row_logsumexp_topk_kernel(
    x_ptr,
    out_max_ptr,
    out_log_sum_ptr,
    out_top_vals_ptr,
    out_top_idx_ptr,
    row_stride,
    col_stride,

    num_cols,
    K: tl.constexpr,


    K_PAD: tl.constexpr,
    BLOCK_N: tl.constexpr,
):

    row = tl.program_id(0).to(tl.int64)
    row_ptr = x_ptr + row * row_stride

    m_i = float("-inf")
    l_i = 0.0
    slots = tl.arange(0, K_PAD)


    run_keys = _pack_key(
        tl.full((K_PAD,), float("-inf"), tl.float32),
        tl.full((K_PAD,), 2147483647, tl.int32),
    )
    kth_key = tl.min(run_keys)

    for start in range(0, num_cols, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        x = tl.load(
            row_ptr + offs * col_stride, mask=offs < num_cols, other=float("-inf")
        ).to(tl.float32)

        m_i, l_i, m_blk = _accumulate_block_lse(x, m_i, l_i)


        SEG: tl.constexpr = BLOCK_N // K_PAD
        x2 = tl.reshape(x, (K_PAD, SEG))
        i2 = start + tl.arange(0, K_PAD)[:, None] * SEG + tl.arange(0, SEG)[None, :]
        avail2 = i2 < num_cols
        go = (start == 0) | (m_blk > _unpack_val(kth_key))
        while go:
            xa2 = tl.where(avail2, x2, float("-inf"))
            cand_vals = tl.max(xa2, axis=1)
            attain = avail2 & (xa2 == cand_vals[:, None])
            cand_idxs = tl.min(tl.where(attain, i2, 2147483647), axis=1)


            avail2 = avail2 & (i2 != cand_idxs[:, None])

            cand_keys = _pack_key(cand_vals, cand_idxs)
            go = tl.max(cand_keys) > kth_key
            if go:


                cand_sorted = tl.sort(cand_keys, descending=True)
                merged = tl.maximum(run_keys, tl.flip(cand_sorted, 0))
                run_keys = tl.sort(merged, descending=True)
                kth_key = tl.max(tl.where(slots == K - 1, run_keys, -1))

    tl.store(out_max_ptr + row, m_i)
    tl.store(out_log_sum_ptr + row, tl.log(l_i))
    out_mask = slots < K
    tl.store(out_top_vals_ptr + row * K + slots, _unpack_val(run_keys), mask=out_mask)
    tl.store(
        out_top_idx_ptr + row * K + slots,
        _unpack_idx(run_keys).to(tl.int64),
        mask=out_mask,
    )


def row_logsumexp_topk(
    x: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    assert x.ndim == 2
    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    num_rows, num_cols = x.shape

    assert 1 <= k <= min(FUSED_TOPK_MAX_K, num_cols), (k, num_cols)

    out_max = torch.empty(num_rows, device=x.device, dtype=torch.float32)
    out_log_sum = torch.empty(num_rows, device=x.device, dtype=torch.float32)
    top_vals = torch.empty((num_rows, k), device=x.device, dtype=torch.float32)
    top_idx = torch.empty((num_rows, k), device=x.device, dtype=torch.int64)
    if num_rows == 0:
        return out_max, out_log_sum, top_vals, top_idx


    k_pad = max(2, triton.next_power_of_2(k))


    BLOCK_N = max(triton.next_power_of_2(min(num_cols, 16384)), k_pad)
    _row_logsumexp_topk_kernel[(num_rows,)](
        x,
        out_max,
        out_log_sum,
        top_vals,
        top_idx,
        x.stride(0),
        x.stride(1),
        num_cols,
        K=k,
        K_PAD=k_pad,
        BLOCK_N=BLOCK_N,
        num_warps=8,
    )
    return out_max, out_log_sum, top_vals, top_idx
