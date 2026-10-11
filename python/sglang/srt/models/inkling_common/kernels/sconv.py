from typing import TypedDict

import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_cuda

PAD_SLOT_ID = -1


class SconvDecodeMetadata(TypedDict):
    cache_mask: torch.Tensor
    safe_idx: torch.Tensor
    cu: torch.Tensor
    si: torch.Tensor


class SconvExtendMetadata(TypedDict):
    cache_mask: torch.Tensor
    safe_idx: torch.Tensor
    cu: torch.Tensor
    si: torch.Tensor


class SconvMetadataOut(TypedDict):


    query_start_loc: torch.Tensor
    has_initial_state: torch.Tensor
    cache_mask: torch.Tensor
    safe_idx: torch.Tensor
    cu: torch.Tensor
    si: torch.Tensor


def _metadata_out(
    out: "SconvMetadataOut | None", *, B: int, T: int, device: torch.device
) -> SconvMetadataOut:

    spec = (
        ("query_start_loc", (B + 1,), torch.int32),
        ("has_initial_state", (B,), torch.bool),
        ("cache_mask", (B, 1, 1), torch.bool),
        ("safe_idx", (B,), torch.int64),
        ("cu", (B + 1,), torch.int64),
        ("si", (T,), torch.int32),
    )
    if out is None:
        return SconvMetadataOut(
            **{
                name: torch.empty(shape, dtype=dtype, device=device)
                for name, shape, dtype in spec
            }
        )
    for name, shape, dtype in spec:
        t = out[name]
        assert (
            tuple(t.shape) == shape and t.dtype == dtype and t.is_contiguous()
        ), f"{name}: got {tuple(t.shape)}/{t.dtype}, want {shape}/{dtype} contiguous"
    return out


CHUNK_SIZE = 64


def _conv_prefix_autotune_configs() -> list[triton.Config]:

    configs = []
    for block_t in (1, 2, 4, 8, 16, 32, 64, 128):
        for block_d in (128, 256):


            tile = block_t * block_d
            if tile <= 256:
                warps_opts = (2, 4)
            elif tile <= 4096:
                warps_opts = (4, 8)
            else:
                warps_opts = (8,)
            for num_warps in warps_opts:
                for num_stages in (2, 3, 4):
                    configs.append(
                        triton.Config(
                            {"BLOCK_T": block_t, "BLOCK_D": block_d},
                            num_warps=num_warps,
                            num_stages=num_stages,
                        )
                    )
    return configs


@triton.autotune(configs=_conv_prefix_autotune_configs(), key=["D", "W", "t_bucket"])
@triton.jit
def _causal_conv1d_fwd_with_prefix_kernel(
    x,
    sconv_cache,
    safe_idx,
    cache_mask,
    weight,
    cu_seqlens,
    seq_idx,
    y,
    t_bucket,
    stride_x_t,
    stride_x_d,
    stride_cache_slot,
    stride_cache_w,
    stride_cache_d,
    stride_cm,
    stride_weight_d,
    stride_weight_w,
    stride_y_t,
    stride_y_d,
    T,
    D,
    USE_SILU: tl.constexpr,
    USE_RESIDUAL: tl.constexpr,
    IS_DECODE: tl.constexpr,
    W: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):

    t_off = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    d_off = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    t_mask = t_off < T
    d_mask = d_off < D
    td_mask = t_mask[:, None] & d_mask[None, :]


    si = tl.load(seq_idx + t_off, mask=t_mask, other=0).to(tl.int64)
    bos = tl.load(cu_seqlens + si, mask=t_mask, other=0).to(tl.int64)
    slot = tl.load(safe_idx + si, mask=t_mask, other=0).to(tl.int64)


    if not IS_DECODE:
        m_val = tl.load(cache_mask + si * stride_cm, mask=t_mask, other=0).to(
            x.dtype.element_ty
        )

    x_row_base = x + d_off[None, :] * stride_x_d
    weight_base = weight + d_off * stride_weight_d
    acc = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)


    x_cur = tl.load(
        x_row_base + t_off[:, None] * stride_x_t,
        mask=td_mask,
        other=0,
    )

    for iw in tl.static_range(W):
        shifted = t_off - (W - 1) + iw

        if iw == W - 1:

            x_val = x_cur
        else:


            in_x = (shifted >= bos) & (shifted < T)
            x_val = tl.load(
                x_row_base + shifted[:, None] * stride_x_t,
                mask=in_x[:, None] & d_mask[None, :],
                other=0,
                eviction_policy="evict_last",
            )


        if iw == W - 1:
            tap = x_val.to(tl.float32)
        else:
            prefix_pos = shifted - bos + (W - 1)
            in_prefix = (shifted < bos) & (prefix_pos >= 0)
            p_val = tl.load(
                sconv_cache
                + slot[:, None] * stride_cache_slot
                + prefix_pos[:, None] * stride_cache_w
                + d_off[None, :] * stride_cache_d,
                mask=in_prefix[:, None] & d_mask[None, :],
                other=0,
            )
            if not IS_DECODE:

                p_val = p_val * m_val[:, None]


            tap = (x_val + p_val).to(tl.float32)

        w_val = tl.load(weight_base + iw * stride_weight_w, mask=d_mask, other=0).to(
            tl.float32
        )
        acc += tap * w_val[None, :]

    if USE_SILU:
        acc = acc * tl.sigmoid(acc)

    if USE_RESIDUAL:
        acc += x_cur.to(tl.float32)

    tl.store(
        y + t_off[:, None] * stride_y_t + d_off[None, :] * stride_y_d,
        acc.to(y.dtype.element_ty),
        mask=td_mask,
    )


def _seq_idx_from_cu_seqlens(cu_seqlens: torch.Tensor, T: int) -> torch.Tensor:

    t = torch.arange(T, dtype=torch.int64, device=cu_seqlens.device)


    num_seqs = cu_seqlens.shape[0] - 1
    return (
        (torch.searchsorted(cu_seqlens, t, side="right") - 1)
        .clamp(max=num_seqs - 1)
        .to(torch.int32)
    )


@triton.jit
def _fused_decode_metadata_kernel(
    cache_indices_ptr,
    query_start_loc_ptr,
    has_initial_state_ptr,
    cache_mask_ptr,
    safe_idx_ptr,
    cu_ptr,
    si_ptr,
    B,
    BLOCK: tl.constexpr,
):

    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask_b1 = offs < B + 1
    mask_b = offs < B
    tl.store(query_start_loc_ptr + offs, offs.to(tl.int32), mask=mask_b1)
    tl.store(cu_ptr + offs, offs.to(tl.int64), mask=mask_b1)
    tl.store(si_ptr + offs, offs.to(tl.int32), mask=mask_b)
    ones = tl.full([BLOCK], 1, tl.int1)
    tl.store(has_initial_state_ptr + offs, ones, mask=mask_b)
    ci = tl.load(cache_indices_ptr + offs, mask=mask_b, other=-1)
    tl.store(cache_mask_ptr + offs, ci != -1, mask=mask_b)
    tl.store(safe_idx_ptr + offs, tl.maximum(ci, 0).to(tl.int64), mask=mask_b)


def fused_decode_sconv_metadata(
    B: int, cache_indices: torch.Tensor, out: SconvMetadataOut | None = None
) -> tuple[torch.Tensor, torch.Tensor, SconvDecodeMetadata]:

    assert cache_indices.shape[0] == B and cache_indices.stride(0) == 1
    device = cache_indices.device
    dst = _metadata_out(out, B=B, T=B, device=device)
    query_start_loc = dst["query_start_loc"]
    has_initial_state = dst["has_initial_state"]
    cache_mask = dst["cache_mask"]
    safe_idx = dst["safe_idx"]
    cu = dst["cu"]
    si = dst["si"]
    BLOCK = 1024
    _fused_decode_metadata_kernel[(triton.cdiv(B + 1, BLOCK),)](
        cache_indices,
        query_start_loc,
        has_initial_state,
        cache_mask,
        safe_idx,
        cu,
        si,
        B,
        BLOCK=BLOCK,
    )
    return (
        query_start_loc,
        has_initial_state,
        SconvDecodeMetadata(cache_mask=cache_mask, safe_idx=safe_idx, cu=cu, si=si),
    )


def precompute_helion_decode_metadata(
    B: int,
    W: int,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> SconvDecodeMetadata:

    device = cache_indices.device
    valid = cache_indices != PAD_SLOT_ID
    cache_mask = (has_initial_state & valid)[:, None, None]
    safe_idx = cache_indices.clamp(min=0).long()

    cu = torch.arange(B + 1, dtype=torch.int64, device=device)
    si = torch.arange(B, dtype=torch.int32, device=device)
    return SconvDecodeMetadata(
        cache_mask=cache_mask,
        safe_idx=safe_idx,
        cu=cu,
        si=si,
    )


HIS_ZEROS = 0
HIS_PREFIX = 1
HIS_SEQ_MINUS_EXT = 2
HIS_ONES = 3


_FUSED_EXTEND_MAX_B = 1023


@triton.jit
def _fused_extend_metadata_kernel(
    cache_indices_ptr,
    extend_seq_lens_ptr,
    his_src_ptr,
    query_start_loc_ptr,
    has_initial_state_ptr,
    cache_mask_ptr,
    safe_idx_ptr,
    cu_ptr,
    si_ptr,
    B,
    T,
    draft_token_num,
    IS_VERIFY: tl.constexpr,
    HIS_MODE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_T: tl.constexpr,
):

    pid = tl.program_id(0)
    offs_b = tl.arange(0, BLOCK_B)
    mask_b = offs_b < B
    mask_b1 = offs_b < B + 1

    if IS_VERIFY:

        cu_local = offs_b.to(tl.int64) * draft_token_num
    else:


        lens = tl.load(
            extend_seq_lens_ptr + offs_b - 1,
            mask=mask_b1 & (offs_b > 0),
            other=0,
        ).to(tl.int64)
        cu_local = tl.cumsum(lens, axis=0)

    if pid == 0:
        tl.store(query_start_loc_ptr + offs_b, cu_local.to(tl.int32), mask=mask_b1)
        tl.store(cu_ptr + offs_b, cu_local, mask=mask_b1)
        if HIS_MODE == 0:
            his = offs_b < 0
        elif HIS_MODE == 1:
            his = tl.load(his_src_ptr + offs_b, mask=mask_b, other=0).to(tl.int64) > 0
        elif HIS_MODE == 2:
            seq = tl.load(his_src_ptr + offs_b, mask=mask_b, other=0).to(tl.int64)
            ext = tl.load(extend_seq_lens_ptr + offs_b, mask=mask_b, other=0).to(
                tl.int64
            )
            his = (seq - ext) > 0
        else:
            his = offs_b >= 0
        tl.store(has_initial_state_ptr + offs_b, his, mask=mask_b)
        ci = tl.load(cache_indices_ptr + offs_b, mask=mask_b, other=-1)
        tl.store(cache_mask_ptr + offs_b, his & (ci != -1), mask=mask_b)
        tl.store(safe_idx_ptr + offs_b, tl.maximum(ci, 0).to(tl.int64), mask=mask_b)
    else:
        offs_t = (pid - 1) * BLOCK_T + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T
        if IS_VERIFY:
            si = tl.minimum(offs_t // draft_token_num, B - 1)
        else:


            bounds = tl.where(mask_b1 & (offs_b > 0), cu_local, 9223372036854775807)
            cnt = tl.sum(
                (offs_t[:, None].to(tl.int64) >= bounds[None, :]).to(tl.int32), axis=1
            )
            si = tl.minimum(cnt, B - 1)
        tl.store(si_ptr + offs_t, si.to(tl.int32), mask=mask_t)


def fused_extend_sconv_metadata(
    *,
    B: int,
    T: int,
    cache_indices: torch.Tensor,
    his_mode: int,
    extend_seq_lens: torch.Tensor | None = None,
    his_src: torch.Tensor | None = None,
    draft_token_num: int | None = None,
    out: SconvMetadataOut | None = None,
) -> tuple[torch.Tensor, torch.Tensor, SconvExtendMetadata] | None:

    if B > _FUSED_EXTEND_MAX_B or not cache_indices.is_cuda:
        return None
    assert cache_indices.shape[0] >= B and cache_indices.stride(0) == 1
    is_verify = his_mode == HIS_ONES
    if is_verify:
        assert draft_token_num is not None
    else:
        assert extend_seq_lens is not None and extend_seq_lens.stride(0) == 1
    device = cache_indices.device
    dst = _metadata_out(out, B=B, T=T, device=device)
    query_start_loc = dst["query_start_loc"]
    has_initial_state = dst["has_initial_state"]
    cache_mask = dst["cache_mask"]
    safe_idx = dst["safe_idx"]
    cu = dst["cu"]
    si = dst["si"]
    BLOCK_T = 256
    dummy = cache_indices
    _fused_extend_metadata_kernel[(1 + triton.cdiv(T, BLOCK_T),)](
        cache_indices,
        extend_seq_lens if extend_seq_lens is not None else dummy,
        his_src if his_src is not None else dummy,
        query_start_loc,
        has_initial_state,
        cache_mask,
        safe_idx,
        cu,
        si,
        B,
        T,
        draft_token_num if draft_token_num is not None else 1,
        IS_VERIFY=is_verify,
        HIS_MODE=his_mode,
        BLOCK_B=triton.next_power_of_2(B + 1),
        BLOCK_T=BLOCK_T,
    )
    return (
        query_start_loc,
        has_initial_state,
        SconvExtendMetadata(cache_mask=cache_mask, safe_idx=safe_idx, cu=cu, si=si),
    )


def precompute_helion_extend_metadata(
    B: int,
    T: int,
    W: int,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> SconvExtendMetadata:

    device = cache_indices.device

    valid = cache_indices != PAD_SLOT_ID
    cache_mask = (has_initial_state & valid)[:, None, None]
    safe_idx = cache_indices.clamp(min=0).long()

    cu = query_start_loc.to(torch.int64)
    si = _seq_idx_from_cu_seqlens(cu, T)

    return SconvExtendMetadata(
        cache_mask=cache_mask,
        safe_idx=safe_idx,
        cu=cu,
        si=si,
    )


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    sconv_cache: torch.Tensor,
    cache_mask: torch.Tensor,
    safe_idx: torch.Tensor,
    cu: torch.Tensor,
    si: torch.Tensor,
    activation: str | None = None,
    use_residual: bool = True,
    is_decode: bool = False,
) -> torch.Tensor:

    if activation == "swish":
        activation = "silu"

    T = x.shape[0]

    if T == 0:
        return torch.empty_like(x)

    D = x.shape[1]
    W = weight.shape[1]
    use_silu = activation in ("silu", "swish")

    if (
        is_cuda()
        and not is_decode
        and x.dtype == torch.bfloat16
        and D % 2 == 0
        and x.stride(1) == 1
    ):
        from sglang.kernels.ops.mamba.inkling_sconv import (
            causal_conv1d as _cuda_causal_conv1d,
        )

        return _cuda_causal_conv1d(
            x,
            weight,
            sconv_cache,
            cache_mask,
            safe_idx,
            cu,
            si,
            activation=activation,
            use_residual=use_residual,
            is_decode=is_decode,
        )


    y = torch.empty(T, D, dtype=x.dtype, device=x.device)


    t_bucket = 0 if T <= 8192 else (1 if T <= 65536 else 2)

    grid = lambda meta: (
        triton.cdiv(T, meta["BLOCK_T"]),
        triton.cdiv(D, meta["BLOCK_D"]),
    )
    _causal_conv1d_fwd_with_prefix_kernel[grid](
        x,
        sconv_cache,
        safe_idx,
        cache_mask,
        weight,
        cu,
        si,
        y,
        t_bucket,
        x.stride(0),
        x.stride(1),
        sconv_cache.stride(0),
        sconv_cache.stride(1),
        sconv_cache.stride(2),
        cache_mask.stride(0),
        weight.stride(0),
        weight.stride(1),
        y.stride(0),
        y.stride(1),
        T,
        D,
        USE_SILU=use_silu,
        USE_RESIDUAL=use_residual,
        IS_DECODE=is_decode,
        W=W,
    )
    return y


@triton.jit
def _update_sconv_cache_kernel(
    x,
    sconv_cache,
    cache_indices,
    has_initial_state,
    query_start_loc,
    stride_x_t,
    stride_x_d,
    stride_cache_slot,
    stride_cache_w,
    stride_cache_d,
    B,
    D,
    W_MINUS_1: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
):

    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    b_off = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    b_mask = b_off < B
    d_mask = d_off < D
    bd_mask = b_mask[:, None] & d_mask[None, :]

    ci_raw = tl.load(cache_indices + b_off, mask=b_mask, other=-1)
    ci = tl.maximum(ci_raw, 0).to(tl.int64)
    end = tl.load(query_start_loc + b_off + 1, mask=b_mask, other=0).to(tl.int64)
    start = tl.load(query_start_loc + b_off, mask=b_mask, other=0).to(tl.int64)
    query_len = end - start
    has_state = tl.load(has_initial_state + b_off, mask=b_mask, other=0) != 0

    valid = (ci_raw != -1) & (query_len > 0)

    cache_base = (
        sconv_cache + ci[:, None] * stride_cache_slot + d_off[None, :] * stride_cache_d
    )


    write_mask = bd_mask & valid[:, None]

    for w in tl.static_range(W_MINUS_1):


        gets_x = query_len >= (W_MINUS_1 - w)
        x_idx = tl.maximum(end - W_MINUS_1 + w, 0)
        x_val = tl.load(
            x + x_idx[:, None] * stride_x_t + d_off[None, :] * stride_x_d,
            mask=bd_mask,
            other=0,
        )


        shift_val = tl.zeros([BLOCK_B, BLOCK_D], dtype=sconv_cache.dtype.element_ty)
        for src_w in tl.static_range(W_MINUS_1):
            match = query_len == (src_w - w)
            src_val = tl.load(
                cache_base + src_w * stride_cache_w, mask=bd_mask, other=0
            )
            shift_val = tl.where(match[:, None], src_val, shift_val)
        shift_val = tl.where(has_state[:, None], shift_val, 0)

        new_val = tl.where(gets_x[:, None], x_val, shift_val)
        tl.store(cache_base + w * stride_cache_w, new_val, mask=write_mask)


def update_sconv_cache(
    x: torch.Tensor,
    sconv_cache: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> None:
    B = cache_indices.shape[0]
    D = x.shape[-1]
    W_minus_1 = sconv_cache.shape[1]

    if (
        is_cuda()
        and x.dtype == torch.bfloat16
        and D % 2 == 0
        and x.stride(-1) == 1
        and sconv_cache.stride(2) == 1
    ):
        from sglang.kernels.ops.mamba.inkling_sconv import (
            update_sconv_cache as _cuda_update_sconv_cache,
        )

        _cuda_update_sconv_cache(
            x,
            sconv_cache,
            cache_indices.to(torch.int32),
            has_initial_state,
            query_start_loc.to(torch.int32),
        )
        return

    BLOCK_D = min(triton.next_power_of_2(D), 1024)
    BLOCK_B = 1
    num_warps = 4 if BLOCK_D >= 512 else (2 if BLOCK_D >= 128 else 1)
    grid = (triton.cdiv(B, BLOCK_B), triton.cdiv(D, BLOCK_D))
    _update_sconv_cache_kernel[grid](
        x,
        sconv_cache,
        cache_indices,
        has_initial_state,
        query_start_loc,
        x.stride(0),
        x.stride(1),
        sconv_cache.stride(0),
        sconv_cache.stride(1),
        sconv_cache.stride(2),
        B,
        D,
        W_MINUS_1=W_minus_1,
        BLOCK_B=BLOCK_B,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
    )


@triton.jit
def _fused_causal_conv1d_update_decode_kernel(
    x,
    sconv_cache,
    cache_indices,
    cache_mask,
    weight,
    y,
    track_mask,
    track_indices,
    stride_x_t,
    stride_x_d,
    stride_y_t,
    stride_y_d,
    stride_cache_slot,
    stride_cache_d,
    stride_cache_w,
    stride_weight_d,
    stride_weight_w,
    stride_track_idx,
    T,
    D,
    USE_SILU: tl.constexpr,
    USE_RESIDUAL: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    W: tl.constexpr,
    DO_TRACK: tl.constexpr,
):

    t_off = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    d_off = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    t_mask = t_off < T
    d_mask = d_off < D
    td_mask = t_mask[:, None] & d_mask[None, :]

    ci = tl.load(cache_indices + t_off, mask=t_mask, other=-1)
    safe_idx = tl.maximum(ci, 0).to(tl.int64)
    valid = ci != -1
    cm = tl.load(cache_mask + t_off, mask=t_mask, other=0).to(
        sconv_cache.dtype.element_ty
    )

    cache_base = (
        sconv_cache
        + safe_idx[:, None] * stride_cache_slot
        + d_off[None, :] * stride_cache_d
    )
    weight_base = weight + d_off * stride_weight_d

    if DO_TRACK:


        real = t_mask & valid
        do_track = tl.load(track_mask + t_off, mask=real, other=0) != 0
        track_slot = tl.load(
            track_indices + t_off * stride_track_idx, mask=real, other=0
        ).to(tl.int64)
        track_base = (
            sconv_cache
            + track_slot[:, None] * stride_cache_slot
            + d_off[None, :] * stride_cache_d
        )
        track_write_mask = td_mask & (valid & do_track)[:, None]


    acc = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)


    for iw in tl.static_range(W - 1):
        pv = tl.load(
            cache_base + iw * stride_cache_w,
            mask=td_mask,
            other=0,
            eviction_policy="evict_last",
        )
        w = tl.load(weight_base + iw * stride_weight_w, mask=d_mask, other=0).to(
            tl.float32
        )
        acc += (pv * cm[:, None]).to(tl.float32) * w[None, :]


    xv = tl.load(
        x + t_off[:, None] * stride_x_t + d_off[None, :] * stride_x_d,
        mask=td_mask,
        other=0,
    )
    xv_f32 = xv.to(tl.float32)
    w_last = tl.load(weight_base + (W - 1) * stride_weight_w, mask=d_mask, other=0).to(
        tl.float32
    )
    acc += xv_f32 * w_last[None, :]

    if USE_SILU:
        acc = acc * tl.sigmoid(acc)

    if USE_RESIDUAL:
        acc += xv_f32

    tl.store(
        y + t_off[:, None] * stride_y_t + d_off[None, :] * stride_y_d,
        acc.to(xv.dtype),
        mask=td_mask,
    )


    write_mask = td_mask & valid[:, None]
    for iw in tl.static_range(W - 2):

        shifted = tl.load(cache_base + (iw + 1) * stride_cache_w, mask=td_mask, other=0)
        new_val = shifted * cm[:, None]
        tl.store(cache_base + iw * stride_cache_w, new_val, mask=write_mask)
        if DO_TRACK:
            tl.store(track_base + iw * stride_cache_w, new_val, mask=track_write_mask)

    tl.store(cache_base + (W - 2) * stride_cache_w, xv, mask=write_mask)
    if DO_TRACK:
        tl.store(track_base + (W - 2) * stride_cache_w, xv, mask=track_write_mask)


def _select_fused_decode_config(T: int, D: int) -> tuple[int, int, int, int]:

    if T <= 2048:
        block_t = 2
    else:

        raw = min(T // 1024, 8)
        block_t = 1 << (raw.bit_length() - 1)

    target_blocks = 1024
    t_blocks = max(T // block_t, 1)
    needed_d_blocks = max(target_blocks // t_blocks, 1)
    block_d = max(D // needed_d_blocks, 64)
    block_d = 1 << max(min((block_d).bit_length() - 1, 9), 6)

    tile_elems = block_t * block_d
    if tile_elems <= 128:
        num_warps = 1
    elif tile_elems <= 512:
        num_warps = 2
    else:
        num_warps = 4

    return block_t, block_d, num_warps, 3


def fused_causal_conv1d_update_decode(
    x: torch.Tensor,
    weight: torch.Tensor,
    sconv_cache: torch.Tensor,
    cache_indices: torch.Tensor,
    cache_mask: torch.Tensor,
    activation: str | None = None,
    use_residual: bool = True,
    track_mask: torch.Tensor | None = None,
    track_indices: torch.Tensor | None = None,
) -> torch.Tensor:

    T, D = x.shape
    W = weight.shape[1]

    if (
        is_cuda()
        and x.dtype == torch.bfloat16
        and D % 2 == 0
        and x.stride(1) == 1
        and sconv_cache.stride(2) == 1
    ):
        from sglang.kernels.ops.mamba.inkling_sconv import (
            fused_causal_conv1d_update_decode as _cuda_fused_decode,
        )

        _ti = track_indices.to(torch.int64) if track_indices is not None else None
        return _cuda_fused_decode(
            x,
            weight,
            sconv_cache,
            cache_indices.to(torch.int32),
            cache_mask,
            activation=activation,
            use_residual=use_residual,
            track_mask=track_mask,
            track_indices=_ti,
        )


    y = torch.empty(T, D, dtype=x.dtype, device=x.device)
    cm = cache_mask.view(-1)
    use_silu = activation in ("silu", "swish")

    do_track = track_mask is not None
    if do_track:
        track_mask = track_mask.view(-1)

        stride_track_idx = track_indices.stride(0)
    else:

        track_mask = torch.empty(0, dtype=torch.bool, device=x.device)
        track_indices = torch.empty(0, dtype=torch.int64, device=x.device)
        stride_track_idx = 1

    bt, bd, nw, ns = _select_fused_decode_config(T, D)

    grid = (triton.cdiv(T, bt), triton.cdiv(D, bd))
    _fused_causal_conv1d_update_decode_kernel[grid](
        x,
        sconv_cache,
        cache_indices,
        cm,
        weight,
        y,
        track_mask,
        track_indices,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        sconv_cache.stride(0),
        sconv_cache.stride(2),
        sconv_cache.stride(1),
        weight.stride(0),
        weight.stride(1),
        stride_track_idx,
        T,
        D,
        USE_SILU=use_silu,
        USE_RESIDUAL=use_residual,
        BLOCK_T=bt,
        BLOCK_D=bd,
        W=W,
        DO_TRACK=do_track,
        num_warps=nw,
        num_stages=ns,
    )
    return y


@triton.jit
def _save_intermediate_conv_windows_kernel(
    sconv_cache_ptr,
    hidden_states_ptr,
    cache_indices_ptr,
    out_ptr,
    cache_slot_stride,
    cache_pos_stride,
    hidden_b_stride,
    hidden_t_stride,
    out_b_stride,
    out_t_stride,
    out_w_stride,
    D,
    W_MINUS_1: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PAD_SLOT_ID: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_d = tl.program_id(2)

    cache_idx = tl.load(cache_indices_ptr + pid_b).to(tl.int64)


    if cache_idx == PAD_SLOT_ID:
        return

    d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_off < D

    for w in tl.static_range(W_MINUS_1):
        position = pid_t + 1 + w
        if position < W_MINUS_1:
            src_offset = (
                cache_idx * cache_slot_stride + position * cache_pos_stride + d_off
            )
            val = tl.load(sconv_cache_ptr + src_offset, mask=d_mask, other=0.0)
        else:
            t_in_hidden = position - W_MINUS_1
            src_offset = (
                pid_b.to(tl.int64) * hidden_b_stride
                + t_in_hidden.to(tl.int64) * hidden_t_stride
                + d_off
            )
            val = tl.load(hidden_states_ptr + src_offset, mask=d_mask, other=0.0)

        dst_offset = (
            pid_b.to(tl.int64) * out_b_stride
            + pid_t.to(tl.int64) * out_t_stride
            + w * out_w_stride
            + d_off
        )
        tl.store(out_ptr + dst_offset, val, mask=d_mask)


def save_intermediate_conv_windows(
    sconv_cache: torch.Tensor,
    hidden_states: torch.Tensor,
    cache_indices: torch.Tensor,
    intermediate_out: torch.Tensor,
    batch_size: int,
    draft_token_num: int,
) -> None:

    if batch_size == 0 or draft_token_num == 0:
        return

    W_minus_1, D = sconv_cache.shape[1], sconv_cache.shape[2]
    if W_minus_1 == 0:
        return

    if hidden_states.dim() == 2:
        hidden_states = hidden_states.view(batch_size, -1, hidden_states.shape[-1])
    assert (
        hidden_states.dim() == 3
    ), f"unexpected hidden_states shape {hidden_states.shape}"
    assert hidden_states.shape[0] == batch_size
    assert hidden_states.shape[2] == D
    assert intermediate_out.shape[1] == draft_token_num
    assert intermediate_out.shape[2] == W_minus_1
    assert intermediate_out.shape[3] == D


    assert sconv_cache.stride(-1) == 1, "sconv_cache must be D-contiguous"
    assert hidden_states.stride(-1) == 1, "hidden_states must be D-contiguous"
    assert intermediate_out.stride(-1) == 1, "intermediate_out must be D-contiguous"

    cache_indices = cache_indices[:batch_size].to(torch.int32).contiguous()

    BLOCK_D = min(triton.next_power_of_2(D), 1024)
    grid = (batch_size, draft_token_num, triton.cdiv(D, BLOCK_D))

    _save_intermediate_conv_windows_kernel[grid](
        sconv_cache,
        hidden_states,
        cache_indices,
        intermediate_out,
        sconv_cache.stride(0),
        sconv_cache.stride(1),
        hidden_states.stride(0),
        hidden_states.stride(1),
        intermediate_out.stride(0),
        intermediate_out.stride(1),
        intermediate_out.stride(2),
        D,
        W_MINUS_1=W_minus_1,
        BLOCK_D=BLOCK_D,
        PAD_SLOT_ID=PAD_SLOT_ID,
    )
