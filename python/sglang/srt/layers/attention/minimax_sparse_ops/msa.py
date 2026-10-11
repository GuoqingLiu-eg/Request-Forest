from __future__ import annotations

import functools
from typing import Optional

import torch

from sglang.kernels.ops.attention.minimax_sparse.common.utils import unit_scale


class MSAUnavailableError(RuntimeError):
    pass


def _check_msa_dtypes(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor):


    if q.dtype == torch.bfloat16:
        assert (
            k_cache.dtype == torch.bfloat16
        ), f"MSA bf16 requires a bf16 K cache, got {k_cache.dtype}"
    elif q.dtype == torch.float8_e4m3fn:


        assert (
            k_cache.dtype == torch.float8_e4m3fn
        ), f"MSA fp8 requires an fp8_e4m3fn K cache, got {k_cache.dtype}"
    else:
        raise AssertionError(f"MSA supports bf16 or fp8_e4m3fn Q, got {q.dtype}")
    assert v_cache.dtype == k_cache.dtype


@functools.lru_cache(maxsize=1)
def _load_fmha_sm100():
    try:
        from fmha_sm100 import fmha_sm100, fmha_sm100_plan
    except Exception as err:
        raise MSAUnavailableError(
            "fmha_sm100 or fmha_sm100_plan is not importable"
        ) from err
    if not callable(fmha_sm100) or not callable(fmha_sm100_plan):
        raise MSAUnavailableError("fmha_sm100 exports must be callable")
    return fmha_sm100, fmha_sm100_plan


def _run_fmha_sm100_plan(*args, **kwargs):
    _, fmha_sm100_plan = _load_fmha_sm100()
    try:
        return fmha_sm100_plan(*args, **kwargs)
    except (AttributeError, RuntimeError, TypeError) as err:
        raise MSAUnavailableError("fmha_sm100_plan failed") from err


@functools.lru_cache(maxsize=1)
def msa_available() -> bool:

    try:
        cap = torch.cuda.get_device_capability()
    except Exception:
        return False


    if cap[0] != 10 or cap[1] not in (0, 3):
        return False
    try:
        _load_fmha_sm100()
        return True
    except MSAUnavailableError:
        return False


def _build_page_table(
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    page_size: int,
) -> torch.Tensor:

    P = page_size
    n_pages = (seq_lens.to(torch.int64) + (P - 1)) // P
    offsets = (
        torch.cumsum(n_pages, 0) - n_pages
    )
    total = int(n_pages.sum().item())
    idx = torch.arange(
        total, device=req_to_token.device
    )
    req = torch.searchsorted(
        offsets + n_pages, idx, right=True
    )
    logical_first = (idx - offsets[req]) * P
    rows = slot_ids[req].to(torch.int64)
    return (req_to_token[rows, logical_first] // P).to(torch.int32)


def msa_sparse_prefill_main(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    topk_idx: torch.Tensor,
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    block_size_k: int,
    sm_scale: Optional[float] = None,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
) -> torch.Tensor:

    fmha_sm100, _ = _load_fmha_sm100()
    _check_msa_dtypes(q, k_cache, v_cache)
    is_fp8 = q.dtype == torch.float8_e4m3fn
    v_scale = unit_scale(v_scale)

    max_slots, num_kv_heads, head_dim = k_cache.shape
    num_q_heads = q.shape[1]
    P = block_size_k
    topk = topk_idx.shape[-1]
    if max_slots % P != 0:
        raise ValueError(f"max_slots={max_slots} not divisible by page_size={P}")
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    sm_scale = sm_scale * unit_scale(q_scale) * unit_scale(k_scale)


    n_phys_pages = max_slots // P
    k_paged = k_cache.view(n_phys_pages, P, num_kv_heads, head_dim).permute(0, 2, 1, 3)
    v_paged = v_cache.view(n_phys_pages, P, num_kv_heads, head_dim).permute(0, 2, 1, 3)


    qo_segment_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int32)
    kv_indices = _build_page_table(req_to_token, slot_ids, seq_lens, P)


    kv_block_indexes = topk_idx.permute(1, 0, 2).contiguous().to(torch.int32)

    plan = _run_fmha_sm100_plan(
        qo_segment_lens,
        seq_lens.to(torch.int32),
        num_q_heads,
        num_kv_heads=num_kv_heads,
        page_size=P,
        kv_block_num=topk,
        causal=True,
        qo_offset=prefix_lens.to(torch.int32),
        use_fp8_kvcache=is_fp8,
    )
    o, _ = fmha_sm100(
        q,
        k_paged,
        v_paged,
        plan,
        sm_scale=sm_scale,
        kv_indices=kv_indices,
        kv_block_indexes=kv_block_indexes,
    )


    if v_scale != 1.0:
        o = o * v_scale
    return o


def build_msa_decode_meta(
    k_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    num_q_heads: int,
    block_size_k: int,
    topk: int,
    is_fp8: bool = False,
):

    max_slots, num_kv_heads, _ = k_cache.shape
    P = block_size_k
    if max_slots % P != 0:
        raise ValueError(f"max_slots={max_slots} not divisible by page_size={P}")
    B = slot_ids.shape[0]
    kv_indices = _build_page_table(req_to_token, slot_ids, seq_lens, P)
    seq_lens_i32 = seq_lens.to(torch.int32)
    plan = _run_fmha_sm100_plan(
        torch.ones(B, dtype=torch.int32),
        seq_lens_i32,
        num_q_heads,
        num_kv_heads=num_kv_heads,
        page_size=P,
        kv_block_num=topk,
        causal=False,
        qo_offset=seq_lens_i32 - 1,
        use_fp8_kvcache=is_fp8,
    )
    return kv_indices, plan


_MSA_CG_LEN_KEYS = (
    "kv_segment_lens",
    "kv_segment_offsets",
    "kv_page_indptr",
    "qo_offset",
)


def _check_cg_plan_layout(plan) -> None:

    if not (isinstance(plan, tuple) and len(plan) > 3 and isinstance(plan[3], dict)):
        raise MSAUnavailableError(
            "fmha_sm100_plan no longer returns a tuple with a metadata dict at index 3; "
            "the MSA CUDA-graph decode path must be revalidated against this fmha_sm100 "
            "version. Set SGLANG_DISABLE_MSA=1 to use the Triton path meanwhile."
        )
    missing = [k for k in _MSA_CG_LEN_KEYS if not torch.is_tensor(plan[3].get(k))]
    if missing:
        raise MSAUnavailableError(
            f"fmha_sm100 plan is missing length tensors {missing}; the MSA CUDA-graph "
            "decode path must be revalidated against this fmha_sm100 version. "
            "Set SGLANG_DISABLE_MSA=1 to use the Triton path meanwhile."
        )


def build_msa_decode_cg_plan(
    num_q_heads: int,
    num_kv_heads: int,
    block_size_k: int,
    topk: int,
    batch_size: int,
    device: Optional[torch.device] = None,
    is_fp8: bool = False,
):

    P = block_size_k
    ref_len = topk * P
    qo = torch.ones(batch_size, dtype=torch.int32)
    kv = torch.full((batch_size,), ref_len, dtype=torch.int32)
    plan = _run_fmha_sm100_plan(
        qo,
        kv,
        num_q_heads,
        num_kv_heads=num_kv_heads,
        page_size=P,
        kv_block_num=topk,
        causal=False,
        qo_offset=kv - 1,
        device=device,
        use_fp8_kvcache=is_fp8,
    )
    _check_cg_plan_layout(plan)
    return plan


def update_msa_decode_cg_meta(
    plan,
    kv_indices_buf: torch.Tensor,
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size_k: int,
    topk: int,
    num_q_heads: int,
    num_kv_heads: int,
):

    P = block_size_k
    B = seq_lens.shape[0]
    if B == 0:
        return
    pd = plan[3]
    seq_lens_i32 = seq_lens.to(torch.int32)
    pd["kv_segment_lens"].copy_(seq_lens_i32)
    kv_off = pd["kv_segment_offsets"]
    kv_off[0].zero_()
    torch.cumsum(seq_lens_i32, 0, out=kv_off[1:])
    n_pages = torch.div(seq_lens_i32 + (P - 1), P, rounding_mode="floor")
    indptr = pd["kv_page_indptr"]
    indptr[0].zero_()
    torch.cumsum(n_pages, 0, out=indptr[1:])
    pd["qo_offset"].copy_(seq_lens_i32.max().expand(B))


    n = kv_indices_buf.numel()
    idx = torch.arange(n, device=kv_indices_buf.device)
    ends = torch.cumsum(n_pages.to(torch.int64), 0)
    req = torch.searchsorted(ends, idx, right=True).clamp_max_(B - 1)
    starts = ends - n_pages
    logical_first = ((idx - starts[req]) * P).clamp_(0, req_to_token.shape[1] - 1)
    rows = slot_ids[req].to(torch.int64)
    kv_indices_buf.copy_((req_to_token[rows, logical_first] // P).to(torch.int32))


def msa_sparse_decode_main(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    topk_idx: torch.Tensor,
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size_k: int,
    sm_scale: Optional[float] = None,
    kv_indices: Optional[
        torch.Tensor
    ] = None,
    plan=None,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
) -> torch.Tensor:

    fmha_sm100, _ = _load_fmha_sm100()
    _check_msa_dtypes(q, k_cache, v_cache)

    max_slots, num_kv_heads, head_dim = k_cache.shape
    H = q.shape[1]
    P = block_size_k
    topk = topk_idx.shape[-1]
    if max_slots % P != 0:
        raise ValueError(f"max_slots={max_slots} not divisible by page_size={P}")
    if sm_scale is None:
        sm_scale = head_dim**-0.5

    n_phys_pages = max_slots // P
    k_paged = k_cache.view(n_phys_pages, P, num_kv_heads, head_dim).permute(0, 2, 1, 3)
    v_paged = v_cache.view(n_phys_pages, P, num_kv_heads, head_dim).permute(0, 2, 1, 3)

    if kv_indices is None or plan is None:
        kv_indices, plan = build_msa_decode_meta(
            k_cache,
            req_to_token,
            slot_ids,
            seq_lens,
            H,
            P,
            topk,
            is_fp8=q.dtype == torch.float8_e4m3fn,
        )
    kv_block_indexes = topk_idx.permute(1, 0, 2).contiguous().to(torch.int32)

    o, _ = fmha_sm100(
        q,
        k_paged,
        v_paged,
        plan,
        sm_scale=sm_scale,
        q_scale=unit_scale(q_scale),
        k_scale=unit_scale(k_scale),
        v_scale=unit_scale(v_scale),
        kv_indices=kv_indices,
        kv_block_indexes=kv_block_indexes,
    )
    return o
