import torch
import triton
import triton.language as tl


@triton.jit
def get_topmask_and_fullmask(x):
    tl.static_assert(
        x.dtype.is_int_unsigned(), "floating-point value must be passed as bits"
    )
    tm: tl.constexpr = 1 << (-1 + x.dtype.primitive_bitwidth)
    fm: tl.constexpr = (1 << x.dtype.primitive_bitwidth) - 1
    tm_arr = tl.full(x.shape, tm, dtype=x.dtype)
    fm_arr = tl.full(x.shape, fm, dtype=x.dtype)
    return tm_arr, fm_arr


@triton.jit
def fpval_to_key(x):
    tm, fm = get_topmask_and_fullmask(x)
    return x ^ tl.where((x & tm) != 0, fm, tm)


@triton.jit
def key_to_fpval(x):
    tm, fm = get_topmask_and_fullmask(x)
    return x ^ tl.where((x & tm) == 0, fm, tm)


@triton.jit
def indx_to_key(idx, N_PAD: tl.constexpr):
    return N_PAD - idx


@triton.jit
def key_to_indx(idx, N_PAD: tl.constexpr):
    return N_PAD - idx


@triton.jit
def _streaming_topk_kernel(
    x_ptr,
    stride_xm,
    values_ptr,
    indices_ptr,
    M,
    N,
    N_PAD: tl.constexpr,
    K: tl.constexpr,
    K_POW2: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    RETURN_VALUES: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_m < M


    x_nbits: tl.constexpr = x_ptr.dtype.element_ty.primitive_bitwidth
    x_utype: tl.constexpr = tl.dtype(f"uint{x_nbits}")
    if x_nbits < 16:


        y_nbits: tl.constexpr = 32
    else:
        y_nbits: tl.constexpr = x_nbits * 2
    x_ultype: tl.constexpr = tl.dtype(f"uint{y_nbits}")
    x_dtype: tl.constexpr = x_ptr.dtype.element_ty


    num_iters: tl.constexpr = N_PAD // BLOCK_SIZE_N - 1
    offs_x_n = num_iters * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_x_n < N


    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_x_n[None, :]
    x = tl.load(x_ptrs, mask=(mask_m[:, None] & mask_n[None, :]), other=float("-inf"))
    x = fpval_to_key(x.to(x_utype, bitcast=True))
    x = (x.to(x_ultype) << 16) | indx_to_key(offs_x_n, N_PAD)[None, :]
    acc = tl.topk(x, K_POW2, dim=1)


    for _i in (tl.static_range if num_iters <= 4 else range)(num_iters):
        acc = tl.bitonic_merge(acc)
        x_ptrs -= BLOCK_SIZE_N
        offs_x_n -= BLOCK_SIZE_N
        x = tl.load(x_ptrs, mask=mask_m[:, None], other=float("-inf"))
        x = fpval_to_key(x.to(x_utype, bitcast=True))
        x = (x.to(x_ultype) << 16) | indx_to_key(offs_x_n, N_PAD)[None, :]
        acc = tl.maximum(acc, tl.topk(x, K_POW2, dim=1))

    offs_k = tl.arange(0, K_POW2)
    mask_k = offs_k < K

    acc = tl.sort(acc, dim=1, descending=True)

    acc = tl.where(mask_k[None, :], acc, 0)

    acc = (acc << (y_nbits - 16)) | (acc >> 16)

    y_indices_raw = (acc >> (y_nbits - 16)).to(tl.uint32)
    y_indices = key_to_indx(y_indices_raw, N_PAD)

    y_values_raw = acc.to(x_utype)
    y_values = key_to_fpval(y_values_raw).to(x_dtype, bitcast=True)

    offs_mk = offs_m[:, None] * K + offs_k[None, :]
    mask_mk = mask_m[:, None] & mask_k[None, :]
    if RETURN_VALUES:
        tl.store(values_ptr + offs_mk, y_values, mask=mask_mk)
    tl.store(indices_ptr + offs_mk, y_indices, mask=mask_mk)


def gate_topk(
    x: torch.Tensor,
    k: int,
    *,
    return_values: bool = True,
    _impl: str = "streaming",
):

    assert x.is_contiguous(), f"{x.shape=} {x.stride()=}"
    assert x.ndim == 2, f"{x.shape=}"
    assert x.numel() <= 2**31, f"assumes int32 indexing: {x.shape=}"
    n_rows, n_cols = x.shape
    if return_values:
        values = torch.empty((n_rows, k), dtype=x.dtype, device=x.device)
    else:
        values = None


    indices = torch.empty((n_rows, k), dtype=torch.int32, device=x.device)
    if k > 32:

        raise NotImplementedError(f"topk kernels only support k <= 32: {k=}")

    if _impl == "streaming":
        BLOCK_SIZE_N = 32
        BLOCK_SIZE_M = 32
        grid = (triton.cdiv(n_rows, BLOCK_SIZE_M),)
        _streaming_topk_kernel[grid](
            x_ptr=x,
            stride_xm=x.stride(0),
            values_ptr=values,
            indices_ptr=indices,
            M=n_rows,
            N=n_cols,
            N_PAD=triton.cdiv(n_cols, BLOCK_SIZE_N) * BLOCK_SIZE_N,
            K=k,
            K_POW2=triton.next_power_of_2(k),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            RETURN_VALUES=return_values,
        )
    else:
        raise NotImplementedError(
            f"topk kernels only support streaming implementation: {_impl=}"
        )

    if return_values:
        return values, indices
    return indices
