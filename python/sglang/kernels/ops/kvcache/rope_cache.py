import torch
import triton
import triton.language as tl


@triton.jit
def _get_gptj_rotated_x(
    x,
    x_rotated_mask,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
):


    x_rotated = tl.where(x_rotated_mask, x, -x)

    x_rotated = tl.reshape(x_rotated, (BLOCK_D_HALF, 2))

    x_rotated = tl.flip(x_rotated, 1)

    x_rotated = tl.reshape(x_rotated, (BLOCK_D,))
    return x_rotated


@triton.jit
def _get_neox_rotated_x(
    x,
    x_rotated_mask,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
):


    x_rotated = tl.where(x_rotated_mask, x, -x)

    x_rotated = tl.reshape(x_rotated, (2, BLOCK_D_HALF))

    x_rotated = tl.flip(x_rotated, 1)

    x_rotated = tl.reshape(x_rotated, (BLOCK_D,))
    x_rotated = tl.flip(x_rotated, 0)
    return x_rotated


@triton.jit
def _unit_rope(
    x_ptrs,
    cos,
    sin,
    d_pe_offs,
    IS_NEOX: tl.constexpr,
    BLOCK_D_pe: tl.constexpr,
    BLOCK_D_HALF_pe: tl.constexpr,
):

    x_pe = tl.load(x_ptrs)


    if IS_NEOX:
        x_rotated_mask = d_pe_offs < BLOCK_D_HALF_pe
        x_pe_rotated = _get_neox_rotated_x(
            x_pe, x_rotated_mask, BLOCK_D_pe, BLOCK_D_HALF_pe
        )
    else:
        x_rotated_mask = d_pe_offs % 2 == 0
        x_pe_rotated = _get_gptj_rotated_x(
            x_pe, x_rotated_mask, BLOCK_D_pe, BLOCK_D_HALF_pe
        )


    x_pe = x_pe * cos + x_pe_rotated * sin

    return x_pe


@triton.jit
def _load_cos_sin(
    cos_sin_ptr,
    pos,
    d_cos_offs,
    stride_t,
    stride_d,
    freq_dim,
):
    base = pos * stride_t
    cos = tl.load(cos_sin_ptr + base + d_cos_offs * stride_d)
    sin = tl.load(cos_sin_ptr + base + (d_cos_offs + freq_dim) * stride_d)
    return cos, sin


@triton.jit
def _fused_qk_rope_reshape_and_cache_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    pos_ptr,
    cos_sin_ptr,
    offs_ptr,
    key_cache_ptr,
    value_cache_ptr,
    slot_mapping_ptr,
    swa_slot_mapping_ptr,
    q_out_ptr,
    k_out_ptr,
    zeros_out_ptr,
    T,
    T_slot,
    q_stride_t,
    q_stride_h,
    q_stride_d,
    k_stride_t,
    k_stride_h,
    k_stride_d,
    v_stride_t,
    v_stride_h,
    v_stride_d,
    cos_sin_stride_t,
    cos_sin_stride_d,
    q_out_stride_t,
    q_out_stride_h,
    q_out_stride_d,
    k_out_stride_t,
    k_out_stride_h,
    k_out_stride_d,
    key_cache_stride_t,
    key_cache_stride_h,
    key_cache_stride_d,
    key_cache_stride_b,
    key_cache_stride_x,
    value_cache_stride_t,
    value_cache_stride_h,
    value_cache_stride_d,
    value_cache_stride_b,
    value_cache_stride_slot_chunk,
    value_cache_stride_x,
    zeros_out_stride_t,
    zeros_out_stride_h,
    zeros_out_stride_d,
    k_scale_ptr,
    v_scale_ptr,
    QH_PER_KH: tl.constexpr,
    QH: tl.constexpr,
    KH: tl.constexpr,
    REUSE_FREQS_FRONT_PART: tl.constexpr,
    IS_NEOX: tl.constexpr,
    BLOCK_D_pe: tl.constexpr,
    BLOCK_D_HALF_pe: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    X_SIZE: tl.constexpr,
    FLASH_LAYOUT: tl.constexpr,
    VALUE_SHUFFLE_LAYOUT: tl.constexpr = False,
    HAVE_POS: tl.constexpr = False,
    HAVE_K_SCALE: tl.constexpr = False,
    HAVE_V_SCALE: tl.constexpr = False,
    HAVE_ZEROS: tl.constexpr = False,
    HAS_SWA: tl.constexpr = False,
):


    tl.assume(q_stride_t >= 0)
    tl.assume(q_stride_h >= 0)
    tl.assume(q_stride_d >= 0)
    tl.assume(k_stride_t >= 0)
    tl.assume(k_stride_h >= 0)
    tl.assume(k_stride_d >= 0)
    tl.assume(v_stride_t >= 0)
    tl.assume(v_stride_h >= 0)
    tl.assume(v_stride_d >= 0)
    tl.assume(cos_sin_stride_t >= 0)
    tl.assume(cos_sin_stride_d >= 0)
    tl.assume(q_out_stride_t >= 0)
    tl.assume(q_out_stride_h >= 0)
    tl.assume(q_out_stride_d >= 0)
    tl.assume(k_out_stride_t >= 0)
    tl.assume(k_out_stride_h >= 0)
    tl.assume(k_out_stride_d >= 0)
    tl.assume(key_cache_stride_t >= 0)
    tl.assume(key_cache_stride_h >= 0)
    tl.assume(key_cache_stride_d >= 0)
    tl.assume(key_cache_stride_b >= 0)
    tl.assume(key_cache_stride_x >= 0)
    tl.assume(value_cache_stride_t >= 0)
    tl.assume(value_cache_stride_h >= 0)
    tl.assume(value_cache_stride_d >= 0)
    tl.assume(value_cache_stride_b >= 0)
    tl.assume(value_cache_stride_slot_chunk >= 0)
    tl.assume(value_cache_stride_x >= 0)
    tl.assume(zeros_out_stride_t >= 0)
    tl.assume(zeros_out_stride_h >= 0)
    tl.assume(zeros_out_stride_d >= 0)


    pid = tl.program_id(0)
    tl.assume(pid >= 0)

    d_pe_offs = tl.arange(0, BLOCK_D_pe).to(tl.int64)


    if pid < T * QH:
        pid_t = pid // QH
        pid_hq = pid % QH


        if REUSE_FREQS_FRONT_PART:
            if IS_NEOX:
                d_cos_offs = d_pe_offs
                d_cos_offs = tl.where(
                    (d_cos_offs >= BLOCK_D_HALF_pe) & (d_cos_offs < BLOCK_D_pe),
                    d_cos_offs - BLOCK_D_HALF_pe,
                    d_cos_offs,
                ).to(d_cos_offs.dtype)

            else:
                d_cos_offs = d_pe_offs // 2

        else:
            d_cos_offs = d_pe_offs


        pos = tl.load(pos_ptr + pid_t)
        if HAVE_POS:
            offset = tl.load(offs_ptr + pid_t)
            pos = pos + offset


        freq_dim = BLOCK_D_HALF_pe if REUSE_FREQS_FRONT_PART else BLOCK_D_pe

        cos, sin = _load_cos_sin(
            cos_sin_ptr,
            pos,
            d_cos_offs,
            cos_sin_stride_t,
            cos_sin_stride_d,
            freq_dim,
        )


        q_ptrs = (
            q_ptr + pid_t * q_stride_t + pid_hq * q_stride_h + d_pe_offs * q_stride_d
        )
        q_pe = _unit_rope(
            q_ptrs,
            cos,
            sin,
            d_pe_offs,
            IS_NEOX,
            BLOCK_D_pe,
            BLOCK_D_HALF_pe,
        )


        q_out_ptrs = (
            q_out_ptr
            + pid_t * q_out_stride_t
            + pid_hq * q_out_stride_h
            + d_pe_offs * q_out_stride_d
        )
        tl.store(q_out_ptrs, q_pe.to(q_out_ptr.dtype.element_ty))

        if HAVE_ZEROS:
            z = tl.zeros((BLOCK_D_pe,), dtype=zeros_out_ptr.dtype.element_ty)
            zeros_out_ptrs = (
                zeros_out_ptr
                + pid_t * zeros_out_stride_t
                + pid_hq * zeros_out_stride_h
                + d_pe_offs * zeros_out_stride_d
            )
            tl.store(zeros_out_ptrs, z)


        if pid_hq % QH_PER_KH == 0:


            pid_slot = tl.load(slot_mapping_ptr + pid_t).to(tl.int64)
            if HAS_SWA:
                pid_slot = tl.load(swa_slot_mapping_ptr + pid_slot)


            if pid_slot >= 0:
                pid_t_slot = pid_slot // BLOCK_SIZE
                pid_b = pid_slot % BLOCK_SIZE
                pid_hk = pid_hq // QH_PER_KH
                if HAVE_K_SCALE:
                    k_scale = tl.load(k_scale_ptr)
                else:
                    k_scale = 1
                k_ptrs = (
                    k_ptr
                    + pid_t * k_stride_t
                    + pid_hk * k_stride_h
                    + d_pe_offs * k_stride_d
                )
                k_pe = _unit_rope(
                    k_ptrs,
                    cos,
                    sin,
                    d_pe_offs,
                    IS_NEOX,
                    BLOCK_D_pe,
                    BLOCK_D_HALF_pe,
                )

                k_out_ptrs = (
                    k_out_ptr
                    + pid_t * k_out_stride_t
                    + pid_hk * k_out_stride_h
                    + d_pe_offs * k_out_stride_d
                )
                tl.store(k_out_ptrs, k_pe.to(k_out_ptr.dtype.element_ty))


                k_scale_rcprl = 1 / k_scale
                k_pe = k_pe * k_scale_rcprl


                if FLASH_LAYOUT:
                    k_out_ptrs = (
                        key_cache_ptr
                        + pid_t_slot * key_cache_stride_t
                        + pid_b * key_cache_stride_b
                        + pid_hk * key_cache_stride_h
                        + d_pe_offs * key_cache_stride_d
                    )
                else:
                    k_pe = tl.reshape(k_pe, (BLOCK_D_pe // X_SIZE, X_SIZE))
                    dx_offs = tl.arange(0, BLOCK_D_pe // X_SIZE).to(tl.int64)
                    x_offs = tl.arange(0, X_SIZE).to(tl.int64)
                    k_out_ptrs = (
                        key_cache_ptr
                        + pid_t_slot * key_cache_stride_t
                        + pid_hk * key_cache_stride_h
                        + dx_offs[:, None] * key_cache_stride_d
                        + pid_b * key_cache_stride_b
                        + x_offs[None, :] * key_cache_stride_x
                    )

                tl.store(k_out_ptrs, k_pe.to(key_cache_ptr.dtype.element_ty))


                v_ptrs = (
                    v_ptr
                    + pid_t * v_stride_t
                    + pid_hk * v_stride_h
                    + d_pe_offs * v_stride_d
                )
                if HAVE_V_SCALE:
                    v_scale = tl.load(v_scale_ptr)
                else:
                    v_scale = 1
                v_scale_rcprl = 1 / v_scale
                v = tl.load(v_ptrs) * v_scale_rcprl
                if VALUE_SHUFFLE_LAYOUT:
                    slot_chunk = pid_b // X_SIZE
                    x_off = pid_b % X_SIZE
                    v_out_ptrs = (
                        value_cache_ptr
                        + pid_t_slot * value_cache_stride_t
                        + pid_hk * value_cache_stride_h
                        + slot_chunk * value_cache_stride_slot_chunk
                        + d_pe_offs.to(tl.int64) * value_cache_stride_d
                        + x_off * value_cache_stride_x
                    )
                else:
                    v_out_ptrs = (
                        value_cache_ptr
                        + pid_t_slot * value_cache_stride_t
                        + pid_hk * value_cache_stride_h
                        + d_pe_offs.to(tl.int64) * value_cache_stride_d
                        + pid_b * value_cache_stride_b
                    )
                tl.store(v_out_ptrs, v.to(value_cache_ptr.dtype.element_ty))


    else:
        pid = pid - T * QH + T * KH
        if pid < T_slot * KH:
            pid_t = pid // KH
            pid_hk = pid % KH
            pid_slot = tl.load(slot_mapping_ptr + pid_t).to(tl.int64)
            if HAS_SWA:
                pid_slot = tl.load(swa_slot_mapping_ptr + pid_slot)

            if pid_slot >= 0:
                pid_t_slot = pid_slot // BLOCK_SIZE
                pid_b = pid_slot % BLOCK_SIZE
                if HAVE_K_SCALE:
                    k_scale = tl.load(k_scale_ptr)
                else:
                    k_scale = 1
                k_ptrs = (
                    k_ptr
                    + pid_t * k_stride_t
                    + pid_hk * k_stride_h
                    + d_pe_offs * k_stride_d
                )

                k_pe = tl.load(k_ptrs)

                k_out_ptrs = (
                    k_out_ptr
                    + pid_t * k_out_stride_t
                    + pid_hk * k_out_stride_h
                    + d_pe_offs * k_out_stride_d
                )
                tl.store(k_out_ptrs, k_pe.to(k_out_ptr.dtype.element_ty))

                k_scale_rcprl = 1 / k_scale
                k_pe = k_pe * k_scale_rcprl

                if FLASH_LAYOUT:
                    k_out_ptrs = (
                        key_cache_ptr
                        + pid_t_slot * key_cache_stride_t
                        + d_pe_offs * key_cache_stride_d
                        + pid_b * key_cache_stride_b
                        + pid_hk * key_cache_stride_h
                    )
                else:
                    k_pe = tl.reshape(k_pe, (BLOCK_D_pe // X_SIZE, X_SIZE))
                    dx_offs = tl.arange(0, BLOCK_D_pe // X_SIZE).to(tl.int64)
                    x_offs = tl.arange(0, X_SIZE).to(tl.int64)
                    k_out_ptrs = (
                        key_cache_ptr
                        + pid_t_slot * key_cache_stride_t
                        + pid_hk * key_cache_stride_h
                        + dx_offs[:, None] * key_cache_stride_d
                        + pid_b * key_cache_stride_b
                        + x_offs[None, :] * key_cache_stride_x
                    )
                tl.store(k_out_ptrs, k_pe.to(key_cache_ptr.dtype.element_ty))

                v_ptrs = (
                    v_ptr
                    + pid_t * v_stride_t
                    + pid_hk * v_stride_h
                    + d_pe_offs * v_stride_d
                )
                if HAVE_V_SCALE:
                    v_scale = tl.load(v_scale_ptr)
                else:
                    v_scale = 1
                v_scale_rcprl = 1 / v_scale
                v = tl.load(v_ptrs) * v_scale_rcprl
                if VALUE_SHUFFLE_LAYOUT:
                    slot_chunk = pid_b // X_SIZE
                    x_off = pid_b % X_SIZE
                    v_out_ptrs = (
                        value_cache_ptr
                        + pid_t_slot * value_cache_stride_t
                        + pid_hk * value_cache_stride_h
                        + slot_chunk * value_cache_stride_slot_chunk
                        + d_pe_offs * value_cache_stride_d
                        + x_off * value_cache_stride_x
                    )
                else:
                    v_out_ptrs = (
                        value_cache_ptr
                        + pid_t_slot * value_cache_stride_t
                        + pid_hk * value_cache_stride_h
                        + d_pe_offs * value_cache_stride_d
                        + pid_b * value_cache_stride_b
                    )
                tl.store(v_out_ptrs, v.to(value_cache_ptr.dtype.element_ty))


def fused_qk_rope_reshape_and_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    pos: torch.Tensor,
    cos_sin: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    is_neox: bool,
    flash_layout: bool,
    apply_scale: bool = True,
    offs: torch.Tensor = None,
    q_out: torch.Tensor = None,
    k_out: torch.Tensor = None,
    output_zeros: bool = True,
    zeros_out: torch.Tensor = None,
    swa_slot_mapping=None,
):


    t, qh, d = q.shape
    tk, kh, dk = k.shape
    tv, vh, dv = v.shape
    if flash_layout:
        t_cache, block_size, kh_cache, dk_cache = key_cache.shape
        t_cache_v, block_size_v, vh_cache, dv_cache = value_cache.shape
        value_shuffle_layout = False
    else:
        t_cache, kh_cache, dkx_cache, block_size, x_cache = key_cache.shape
        if value_cache.ndim == 5:

            t_cache_v, vh_cache, slot_chunk_v, dv_cache, x_v = value_cache.shape
            value_shuffle_layout = True
            block_size_v = slot_chunk_v * x_v
            assert block_size_v == block_size and x_v == x_cache, (
                f"value_cache shuffle (T,KH,block_size//x,D,x) must match key: "
                f"{block_size_v=} {block_size=} {x_v=} {x_cache=}"
            )
        else:
            t_cache_v, vh_cache, dv_cache, block_size_v = value_cache.shape
            value_shuffle_layout = False
    (t_slot,) = slot_mapping.shape

    assert (
        t == tk == tv and t_slot <= tk
    ), f"Number of tokens should be identical for q, kand v. The number of tokens of slot_mapping should no more than that of q, k and v, {t=} {tk=} {tv=} {t_slot=}"
    assert (
        block_size == block_size_v
    ), f"block size should be identical for key_cache, and value_cache {block_size} {block_size_v}"
    assert (
        kh == vh == kh_cache == vh_cache
    ), "KV head should be identical for k, v, key_cache, and value_cache"
    assert (
        t_cache == t_cache_v
    ), "Number of tokens should be identical for key_cache, and value_cache"
    if flash_layout:
        assert (
            d == dk == dv == dk_cache == dv_cache
        ), "D dimension should be identical for q, k, and v"
    else:
        assert (
            d == dk == dv == dkx_cache * x_cache == dv_cache
        ), "D dimension should be identical for q, k, and v"
        assert x_cache == triton.next_power_of_2(x_cache), "x_size should be power of 2"

    assert d == triton.next_power_of_2(d), "D dimension should be power of 2"
    assert block_size == triton.next_power_of_2(
        block_size
    ), "block_size should be power of 2"
    assert qh % kh == 0, "Q heads must be multiple of H heads"
    d_freq = cos_sin.shape[-1] // 2
    assert (d_freq == d // 2) or (
        d_freq == d
    ), "cos/sin last dim should be the same or half of the qk last dim"
    reuse_freqs_front_part = d_freq == d // 2

    if q_out is None:
        q_out = torch.empty((t, qh, d), dtype=q.dtype, device=q.device)

    if k_out is None:
        k_out = torch.empty((tk, kh, dk), dtype=k.dtype, device=q.device)

    if zeros_out is not None:
        tz, qhz, dz = zeros_out.shape
        assert (
            t == tz and qh == qhz and d == dz
        ), f"q and zeros shape mismatch {q.shape=} {zeros_out.shape=}"
        output_zeros = True
    elif output_zeros:
        zeros_out = torch.empty((t, qh, d), dtype=q.dtype, device=q.device)
    else:
        zeros_out = None

    n_pid = t * qh + (t_slot - t) * kh if t_slot >= t else t * qh
    grid = (n_pid, 1, 1)
    _fused_qk_rope_reshape_and_cache_kernel[grid](
        q,
        k,
        v,
        pos,
        cos_sin,
        offs,
        key_cache,
        value_cache,
        slot_mapping,
        swa_slot_mapping,
        q_out,
        k_out,
        zeros_out,
        t,
        t_slot,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        cos_sin.stride(0),
        cos_sin.stride(-1),
        *q_out.stride(),
        *k_out.stride(),
        key_cache.stride(0) if not flash_layout else key_cache.stride(0),
        key_cache.stride(1) if not flash_layout else key_cache.stride(2),
        key_cache.stride(2) if not flash_layout else key_cache.stride(3),
        key_cache.stride(3) if not flash_layout else key_cache.stride(1),
        key_cache.stride(4) if not flash_layout else 0,
        value_cache.stride(0) if not flash_layout else value_cache.stride(0),
        value_cache.stride(1) if not flash_layout else value_cache.stride(2),
        (
            value_cache.stride(3)
            if (not flash_layout and value_shuffle_layout)
            else (value_cache.stride(2) if not flash_layout else value_cache.stride(3))
        ),
        (
            0
            if (not flash_layout and value_shuffle_layout)
            else (value_cache.stride(3) if not flash_layout else value_cache.stride(1))
        ),
        value_cache.stride(2) if (not flash_layout and value_shuffle_layout) else 0,
        value_cache.stride(4) if (not flash_layout and value_shuffle_layout) else 0,
        zeros_out.stride(0) if zeros_out is not None else 0,
        zeros_out.stride(1) if zeros_out is not None else 0,
        zeros_out.stride(2) if zeros_out is not None else 0,
        k_scale_ptr=k_scale,
        v_scale_ptr=v_scale,
        QH_PER_KH=qh // kh,
        QH=qh,
        KH=kh,
        REUSE_FREQS_FRONT_PART=reuse_freqs_front_part,
        IS_NEOX=is_neox,
        BLOCK_D_pe=d,
        BLOCK_D_HALF_pe=d // 2,
        BLOCK_SIZE=block_size,
        X_SIZE=x_cache if not flash_layout else 0,
        FLASH_LAYOUT=flash_layout,
        VALUE_SHUFFLE_LAYOUT=value_shuffle_layout,
        HAVE_POS=(offs is not None),
        HAVE_K_SCALE=(k_scale is not None and apply_scale),
        HAVE_V_SCALE=(v_scale is not None and apply_scale),
        HAVE_ZEROS=output_zeros,
        HAS_SWA=(swa_slot_mapping is not None),
        num_warps=1,
    )

    if zeros_out is not None:
        return q_out.view(-1, qh * d), k_out, key_cache, value_cache, zeros_out
    return q_out.view(-1, qh * d), k_out, key_cache, value_cache
