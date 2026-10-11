import logging
import math
from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.utils import is_hip

logger = logging.getLogger(__name__)
_is_hip = is_hip()

_GLM_DSA_MODEL_ARCHS = (
    "GlmMoeDsaForCausalLM",
    "GlmMoeDsaForCausalLMNextN",
)


_NOPE_DIM = 448
_ROPE_DIM = 64
_NOPE_ROPE_STRIDE = _NOPE_DIM + _ROPE_DIM * 2
_TILE_SIZE = 64
_NUM_TILES = _NOPE_DIM // _TILE_SIZE
_SCALE_STRIDE = _NUM_TILES + 1
_D = _NOPE_DIM + _ROPE_DIM


def _gather_and_dequant(k_cache, indices, page_size):

    idx_shape = indices.shape
    flat_idx = indices.reshape(-1)
    N = flat_idx.shape[0]
    device = k_cache.device


    page_bytes = k_cache.stride(0)
    pages = flat_idx // page_size
    offsets = flat_idx % page_size


    safe_pages = pages.clamp(min=0)
    safe_offsets = offsets.clamp(min=0)


    num_pages = k_cache.shape[0]
    raw_pages = k_cache.as_strided(
        (num_pages, page_bytes),
        (page_bytes, 1),
    ).view(
        torch.uint8
    )


    nope_base = safe_offsets * _NOPE_ROPE_STRIDE
    nope_offsets = nope_base.unsqueeze(-1) + torch.arange(
        _NOPE_DIM, device=device, dtype=torch.long
    )

    rope_base = nope_base + _NOPE_DIM
    rope_offsets = rope_base.unsqueeze(-1) + torch.arange(
        _ROPE_DIM * 2, device=device, dtype=torch.long
    )

    scale_section_offset = page_size * _NOPE_ROPE_STRIDE
    scale_base = scale_section_offset + safe_offsets * _SCALE_STRIDE
    scale_offsets = scale_base.unsqueeze(-1) + torch.arange(
        _NUM_TILES, device=device, dtype=torch.long
    )


    page_idx_nope = safe_pages.unsqueeze(-1).expand_as(nope_offsets)
    nope_bytes = raw_pages[page_idx_nope, nope_offsets]

    page_idx_rope = safe_pages.unsqueeze(-1).expand_as(rope_offsets)
    rope_bytes = raw_pages[page_idx_rope, rope_offsets]

    page_idx_scale = safe_pages.unsqueeze(-1).expand_as(scale_offsets)
    scale_bytes = raw_pages[page_idx_scale, scale_offsets]


    nope_fp8 = nope_bytes.view(torch.float8_e4m3fn)
    rope_bf16 = rope_bytes.contiguous().view(torch.bfloat16)
    scale_e8m0 = scale_bytes.view(torch.float8_e8m0fnu)


    result = torch.empty(N, _D, dtype=torch.bfloat16, device=device)
    result[:, :_NOPE_DIM] = (
        (
            nope_fp8.view(N, _NUM_TILES, _TILE_SIZE).float()
            * scale_e8m0.view(N, _NUM_TILES, 1).float()
        )
        .view(N, _NOPE_DIM)
        .to(torch.bfloat16)
    )
    result[:, _NOPE_DIM:] = rope_bf16

    return result.reshape(*idx_shape, _D)


def _sm120_sparse_decode_fwd(
    q,
    k_cache,
    indices,
    topk_length,
    attn_sink,
    head_dim_v,
    softmax_scale,
    extra_k_cache=None,
    extra_indices=None,
    extra_topk_length=None,
):
    B, s_q, H_q, D_qk = q.shape
    num_pages, page_size, H_k, bpt = k_cache.shape
    topk = indices.shape[-1]

    invalid_mask = indices < 0
    safe_indices = indices.clamp(min=0)

    if topk_length is not None:
        topk_range = torch.arange(topk, device=topk_length.device).view(1, 1, topk)
        invalid_mask = invalid_mask | (topk_range >= topk_length.view(B, 1, 1))


    gathered_kv = _gather_and_dequant(k_cache, safe_indices, page_size)

    if extra_k_cache is not None and extra_indices is not None:
        extra_topk = extra_indices.shape[-1]
        extra_page_size = extra_k_cache.shape[1]
        extra_invalid = extra_indices < 0
        extra_safe = extra_indices.clamp(min=0)
        if extra_topk_length is not None:
            extra_range = torch.arange(
                extra_topk, device=extra_topk_length.device
            ).view(1, 1, extra_topk)
            extra_invalid = extra_invalid | (
                extra_range >= extra_topk_length.view(B, 1, 1)
            )
        extra_kv = _gather_and_dequant(extra_k_cache, extra_safe, extra_page_size)
        gathered_kv = torch.cat([gathered_kv, extra_kv], dim=2)
        invalid_mask = torch.cat([invalid_mask, extra_invalid], dim=2)

    gathered_kv[invalid_mask] = 0.0

    q_f = q.float()
    kv_f = gathered_kv.float()
    kv_d = kv_f.shape[-1]
    if D_qk != kv_d:
        q_f = q_f[..., :kv_d]

    scores = torch.einsum("bshd,bstd->bsht", q_f, kv_f) * softmax_scale
    scores.masked_fill_(invalid_mask.unsqueeze(2).expand_as(scores), float("-inf"))

    lse = torch.logsumexp(scores, dim=-1)

    if attn_sink is not None:
        lse_for_out = torch.logsumexp(
            torch.stack([lse, attn_sink.view(1, 1, H_q).expand_as(lse)], dim=0), dim=0
        )
    else:
        lse_for_out = lse.clone()

    lonely = lse == float("-inf")
    lse_for_out[lonely] = float("inf")
    weights = torch.exp(scores - lse_for_out.unsqueeze(-1))
    out = torch.einsum("bsht,bstv->bshv", weights, kv_f[..., :head_dim_v])
    out[lonely.unsqueeze(-1).expand_as(out)] = 0.0

    return out.to(torch.bfloat16), lse.permute(0, 2, 1)


_sm120_default_backend = envs.SGLANG_SM120_FLASHMLA_BACKEND.get()


def flash_mla_with_kvcache_sm120(**kwargs):

    q = kwargs["q"]
    k_cache = kwargs["k_cache"]
    indices = kwargs["indices"]
    topk_length = kwargs.get("topk_length")
    attn_sink = kwargs.get("attn_sink")
    head_dim_v = kwargs["head_dim_v"]
    softmax_scale = kwargs.get("softmax_scale")
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    extra_k_cache = kwargs.get("extra_k_cache")
    extra_indices = kwargs.get("extra_indices_in_kvcache")
    extra_topk_length = kwargs.get("extra_topk_length")

    if _sm120_default_backend == "flashinfer":
        return _flash_mla_flashinfer(
            q,
            k_cache,
            indices,
            topk_length,
            attn_sink,
            head_dim_v,
            softmax_scale,
            extra_k_cache,
            extra_indices,
            extra_topk_length,
        )

    if _sm120_default_backend == "triton":
        from sglang.kernels.ops.attention.flash_mla_sm120_triton import (
            flash_mla_sparse_decode_triton,
        )

        out, lse = flash_mla_sparse_decode_triton(
            q,
            k_cache,
            indices,
            topk_length,
            attn_sink,
            head_dim_v,
            softmax_scale,
            extra_k_cache,
            extra_indices,
            extra_topk_length,
        )
        return (out, lse)

    out, lse = _sm120_sparse_decode_fwd(
        q,
        k_cache,
        indices,
        topk_length,
        attn_sink,
        head_dim_v,
        softmax_scale,
        extra_k_cache,
        extra_indices,
        extra_topk_length,
    )
    return (out, lse)


_PBS_SRC = 256
_PBS_DST = 64
_NOPE_ROPE_STRIDE = 576
_SCALE_STRIDE = 8
_BYTES_PER_DST_PAGE = (
    _PBS_DST * _NOPE_ROPE_STRIDE + _PBS_DST * _SCALE_STRIDE
)


_BYTES_PER_DST_PAGE_PADDED = math.ceil(_BYTES_PER_DST_PAGE / 576) * 576


@triton.jit
def _page_split_kernel(
    src_ptr,
    dst_ptr,
    N_pages,
    src_stride0: tl.constexpr,
    dst_stride0: tl.constexpr,
    DATA_PER_SUB: tl.constexpr,
    SCALE_PER_SUB: tl.constexpr,
    SRC_SCALE_OFF: tl.constexpr,
    DST_SCALE_OFF: tl.constexpr,
    RATIO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    mask_ptr,
    HAS_MASK: tl.constexpr,
):

    pid = tl.program_id(0)
    page_idx = pid // RATIO
    sub = pid % RATIO

    if page_idx >= N_pages:
        return

    if HAS_MASK:
        if tl.load(mask_ptr + page_idx) == 0:
            return

    src_base = src_ptr + page_idx * src_stride0
    dst_base = dst_ptr + (page_idx * RATIO + sub) * dst_stride0


    data_src_off = sub * DATA_PER_SUB
    for start in tl.range(0, DATA_PER_SUB, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < DATA_PER_SUB
        vals = tl.load(src_base + data_src_off + offs, mask=mask)
        tl.store(dst_base + offs, vals, mask=mask)


    scale_src_off = SRC_SCALE_OFF + sub * SCALE_PER_SUB
    for start in tl.range(0, SCALE_PER_SUB, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < SCALE_PER_SUB
        vals = tl.load(src_base + scale_src_off + offs, mask=mask)
        tl.store(dst_base + DST_SCALE_OFF + offs, vals, mask=mask)


@triton.jit
def _page_mark_kernel(
    indices_ptr,
    mask_ptr,
    N_idx,
    SRC_PBS: tl.constexpr,
    BLOCK: tl.constexpr,
):

    pid = tl.program_id(0)
    if pid >= N_idx:
        return
    idx = tl.load(indices_ptr + pid)
    if idx < 0:
        return
    page = idx // SRC_PBS
    tl.store(mask_ptr + page, 1)


def _split_kv_pages_to_64(
    kv_u8: torch.Tensor,
    src_pbs: int,
    touched_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    assert src_pbs % _PBS_DST == 0 and src_pbs >= _PBS_DST
    if src_pbs == _PBS_DST:
        return kv_u8

    N = kv_u8.shape[0]
    ratio = src_pbs // _PBS_DST
    num_dst_pages = N * ratio

    from sglang.srt.runtime_context import get_resources


    dev = kv_u8.device
    buffers = get_resources().buffers
    key = f"flash_mla_sm120_split:{dev}"
    buf = buffers.get(key)
    if buf is None or buf.shape[0] < num_dst_pages:


        with torch.inference_mode(False):
            buf = torch.empty(
                num_dst_pages,
                _BYTES_PER_DST_PAGE_PADDED,
                dtype=torch.uint8,
                device=dev,
            )
        buffers[key] = buf
    out = buf[:num_dst_pages]


    src_2d = kv_u8
    if src_2d.ndim == 4:
        src_stride0 = src_2d.stride(0)
        src_2d = torch.as_strided(src_2d, (N, src_stride0), (src_stride0, 1))
    else:
        src_stride0 = src_2d.stride(0)

    use_mask = touched_indices is not None and touched_indices.numel() > 0
    mask_ptr = src_2d
    if use_mask:


        mkey = f"flash_mla_sm120_mask:{dev}"
        mbuf = buffers.get(mkey)
        if mbuf is None or mbuf.shape[0] < N:


            with torch.inference_mode(False):
                mbuf = torch.empty(N, dtype=torch.int8, device=dev)
            buffers[mkey] = mbuf
        mask = mbuf[:N]
        mask.zero_()
        idx_flat = touched_indices.reshape(-1).contiguous()
        if idx_flat.dtype != torch.int32:
            idx_flat = idx_flat.to(torch.int32)
        _page_mark_kernel[(idx_flat.numel(),)](
            idx_flat,
            mask,
            idx_flat.numel(),
            src_pbs,
            1024,
        )
        mask_ptr = mask

    grid = (N * ratio,)
    _page_split_kernel[grid](
        src_2d,
        out,
        N,
        src_stride0,
        _BYTES_PER_DST_PAGE_PADDED,
        _PBS_DST * _NOPE_ROPE_STRIDE,
        _PBS_DST * _SCALE_STRIDE,
        src_pbs * _NOPE_ROPE_STRIDE,
        _PBS_DST * _NOPE_ROPE_STRIDE,
        ratio,
        1024,
        mask_ptr,
        use_mask,
    )

    bpt = _NOPE_ROPE_STRIDE + _SCALE_STRIDE
    return out.as_strided(
        (num_dst_pages, _PBS_DST, 1, bpt),
        (_BYTES_PER_DST_PAGE_PADDED, bpt, bpt, 1),
    )


def _flash_mla_flashinfer(
    q,
    k_cache,
    indices,
    topk_length,
    attn_sink,
    head_dim_v,
    softmax_scale,
    extra_k_cache,
    extra_indices,
    extra_topk_length,
):

    from flashinfer.mla._sparse_mla_sm120 import (
        _DECODE_MAX_TOKENS as _FI_DECODE_MAX_TOKENS,
    )
    from flashinfer.mla._sparse_mla_sm120 import (
        _sparse_mla_sm120_paged_attention,
    )

    B, _, H, D = q.shape
    dev = q.device


    idx = indices.squeeze(1) if indices.dim() == 3 else indices


    kv_u8 = k_cache.view(torch.uint8) if k_cache.dtype != torch.uint8 else k_cache
    src_pbs = k_cache.shape[1] if k_cache.ndim >= 3 else _PBS_SRC
    kv_64 = (
        _split_kv_pages_to_64(kv_u8, src_pbs, touched_indices=idx)
        if src_pbs != _PBS_DST
        else kv_u8
    )

    extra_kv_u8 = (
        extra_k_cache.view(torch.uint8)
        if extra_k_cache is not None and extra_k_cache.dtype != torch.uint8
        else extra_k_cache
    )
    extra_kv_64 = extra_kv_u8

    extra_idx = (
        extra_indices.squeeze(1)
        if extra_indices is not None and extra_indices.dim() == 3
        else extra_indices
    )

    output = torch.empty(B, H, head_dim_v, dtype=torch.bfloat16, device=dev)
    out_lse = torch.empty(B, H, dtype=torch.float32, device=dev)


    if B <= _FI_DECODE_MAX_TOKENS:
        topk = idx.shape[-1]
        extra_topk = extra_idx.shape[-1] if extra_idx is not None else 0
        _BI = 64
        num_splits = (topk + _BI - 1) // _BI + (
            (extra_topk + _BI - 1) // _BI if extra_topk > 0 else 0
        )
        mid_out = torch.empty(
            B, H, num_splits, head_dim_v, dtype=torch.bfloat16, device=dev
        )
        mid_lse = torch.empty(B, H, num_splits, dtype=torch.float32, device=dev)
    else:
        mid_out = None
        mid_lse = None

    _sparse_mla_sm120_paged_attention(
        q.squeeze(1) if q.ndim == 4 else q,
        kv_64,
        idx,
        output,
        out_lse,
        softmax_scale,
        d_v=head_dim_v,
        topk_length=topk_length,
        attn_sink=attn_sink,
        extra_kv_cache=extra_kv_64,
        extra_indices=extra_idx,
        extra_topk_length=extra_topk_length,
        mid_out=mid_out,
        mid_lse=mid_lse,
    )

    return (output.unsqueeze(1), None)


def _validate_flashinfer_sparse_mla_backend(
    *,
    model_arch: str,
    device_sm_major: int,
    kv_cache_dtype: torch.dtype,
    prefill_impl: str,
    decode_impl: str,
) -> bool:
    selected = {prefill_impl, decode_impl}
    uses_flashinfer_sparse_mla = "flashinfer_sparse_mla" in selected
    is_glm_sm12_fp8 = (
        model_arch in _GLM_DSA_MODEL_ARCHS
        and device_sm_major == 12
        and kv_cache_dtype == torch.float8_e4m3fn
        and not _is_hip
    )
    if uses_flashinfer_sparse_mla and not is_glm_sm12_fp8:
        raise ValueError(
            "flashinfer_sparse_mla supports only GLM DSA with FP8 KV cache "
            "on NVIDIA SM120/SM121; "
            f"got model_arch={model_arch!r}, sm_major={device_sm_major}, "
            f"kv_cache_dtype={kv_cache_dtype}, prefill_impl={prefill_impl!r}, "
            f"decode_impl={decode_impl!r}."
        )
    if is_glm_sm12_fp8:
        unsupported = selected - {"flashinfer_sparse_mla"}
        if unsupported:
            raise ValueError(
                "GLM DSA with FP8 KV cache on NVIDIA SM120/SM121 supports "
                "only flashinfer_sparse_mla, "
                f"but got {sorted(unsupported)}."
            )
    return uses_flashinfer_sparse_mla


def flashinfer_sparse_mla_forward(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    seq_lens: torch.Tensor,
    workspace_buffer: torch.Tensor,
    *,
    page_size: int,
    kv_cache_dim: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    sm_scale: float,
    skip_softmax_threshold_scale_factor: float | None,
) -> torch.Tensor:

    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

    topk = indices.shape[1]
    result = trtllm_batch_decode_with_kv_cache_mla(
        query=q.unsqueeze(1),
        kv_cache=kv_cache.view(torch.uint8)
        .view(-1, page_size, kv_cache_dim)
        .unsqueeze(1),
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=indices.unsqueeze(1),
        seq_lens=seq_lens,
        max_seq_len=topk,
        sparse_mla_top_k=topk,
        bmm1_scale=float(sm_scale),
        bmm2_scale=1.0,
        kv_scale_format="arbitrary_fp32",
        skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
    )
    return result.squeeze(1)
