from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch


def maybe_write_dsv4_extend(
    batch: ScheduleBatch,
    req_pool_indices_cpu: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
) -> None:


    bundle = batch.out_cache_loc_dsv4
    if bundle is None:
        return

    req_to_token_pool = batch.req_to_token_pool
    if not hasattr(req_to_token_pool, "write_c128"):
        return

    _write_dsv4_tables(
        req_to_token_pool,
        req_pool_indices_cpu,
        prefix_lens_cpu,
        seq_lens_cpu,
        bundle,
    )


def dsv4_state_payloads(
    req_to_token_pool,
    req_pool_idx: int,
    seq_len: int,
    page_size: int,
    *,
    prefix_len: int = 0,
):


    import numpy as np

    from sglang.srt.disaggregation.ascend.conn import AscendStateType

    seq_len = max(0, int(seq_len))
    prefix_len = max(0, min(int(prefix_len), seq_len))

    def c128_kv_pages():
        c128_page_size = req_to_token_pool.c128_page_size
        lo = prefix_len // (128 * c128_page_size)
        hi = (seq_len // 128 + c128_page_size - 1) // c128_page_size
        if hi <= lo:
            return np.empty((0,), dtype=np.int32)
        pages = (
            req_to_token_pool.req_to_c128_sidecar[req_pool_idx, lo:hi]
            .cpu()
            .numpy()
            .astype(np.int32)
        )
        return pages[pages > 0]

    return {AscendStateType.DSV4_C128: c128_kv_pages}


def dsv4_prealloc_kwargs(allocator, req, fill_len, req_to_token_pool, *, device):

    if not hasattr(allocator, "c128_attn_allocator"):
        return {}
    return dict(
        req_pool_indices=torch.tensor(
            [req.req_pool_idx], dtype=torch.int64, device=device
        ),
        req_to_token_pool=req_to_token_pool,
    )


def dsv4_unwrap_prealloc(kv_loc, req_to_token_pool, req, prefix_len, fill_len):

    if kv_loc is None or not hasattr(kv_loc, "out_full_loc"):
        return kv_loc
    write_dsv4_prealloc_tables(req_to_token_pool, req, prefix_len, fill_len, kv_loc)
    return kv_loc.out_full_loc


def write_dsv4_prealloc_tables(
    req_to_token_pool,
    req: Req,
    prefix_len: int,
    fill_len: int,
    bundle,
) -> None:

    if bundle is None or not hasattr(req_to_token_pool, "write_c128"):
        return
    rp = torch.tensor([req.req_pool_idx])
    pl = torch.tensor([prefix_len])
    sl = torch.tensor([fill_len])

    _write_dsv4_tables(
        req_to_token_pool,
        rp,
        pl,
        sl,
        bundle,
    )


def _write_dsv4_tables(
    req_to_token_pool,
    req_pool_indices_cpu: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    bundle,
) -> None:

    _write_per_req_slice(
        req_to_token_pool.write_c128,
        req_pool_indices_cpu,
        prefix_lens_cpu,
        seq_lens_cpu,
        bundle.out_c128_loc,
        ratio=128,
    )


def maybe_write_dsv4_decode(
    batch: ScheduleBatch,
    seq_lens_cpu: torch.Tensor,
    token_per_req: int,
) -> None:


    bundle = batch.out_cache_loc_dsv4
    if bundle is None:
        return

    req_to_token_pool = batch.req_to_token_pool
    if not hasattr(req_to_token_pool, "write_c128"):
        return

    prefix_lens_cpu = (seq_lens_cpu - token_per_req).clamp(min=0)
    req_pool_indices_cpu = batch.req_pool_indices.cpu()

    _write_per_req_slice(
        req_to_token_pool.write_c128,
        req_pool_indices_cpu,
        prefix_lens_cpu,
        seq_lens_cpu,
        bundle.out_c128_loc,
        ratio=128,
    )


def maybe_build_dsv4_verify_bundle(
    batch: ScheduleBatch,
    draft_token_num: int,
    *,
    live_seq_lens_cpu: torch.Tensor | None = None,
):

    pool = batch.req_to_token_pool
    if not hasattr(pool, "req_to_c128_sidecar"):
        return None
    reserve_bundle = batch.out_cache_loc_dsv4
    if reserve_bundle is None:
        return None

    req_indices = batch.req_pool_indices_cpu.tolist()

    if live_seq_lens_cpu is None:
        live_seq_lens_cpu = batch.seq_lens_cpu
    if live_seq_lens_cpu is None:
        live_seq_lens_cpu = batch.seq_lens[: len(req_indices)].cpu()
    live_seq_lens = live_seq_lens_cpu[: len(req_indices)].tolist()

    verify_lens = [int(draft_token_num)] * len(req_indices)

    def flatten_interval(table: torch.Tensor, ratio: int) -> torch.Tensor:
        page_size = pool.c128_page_size
        chunks = []
        for req_idx, live_seq_len, verify_len in zip(
            req_indices, live_seq_lens, verify_lens
        ):
            start = int(live_seq_len) // ratio
            end = (int(live_seq_len) + int(verify_len)) // ratio
            if end > start:
                positions = torch.arange(start, end, device=table.device)
                pages = table[int(req_idx), positions // page_size]
                chunks.append(pages * page_size + positions % page_size)
        return torch.cat(chunks) if chunks else table.new_empty((0,))

    out_full_loc = batch.out_cache_loc
    out_c4_loc = out_full_loc[(out_full_loc >= 0) & ((out_full_loc % 4) == 3)] // 4
    return type(reserve_bundle)(
        out_full_loc=out_full_loc,
        out_swa_loc=batch.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
            out_full_loc
        ),
        out_c4_loc=out_c4_loc,
        out_c128_loc=flatten_interval(pool.req_to_c128_sidecar, 128),
    )


def _write_per_req(
    write_fn,
    req_pool_indices_cpu: torch.Tensor,
    flat_loc: torch.Tensor,
    bounds_fn,
) -> None:

    if flat_loc is None or flat_loc.numel() == 0:
        return
    pt = 0
    for i in range(req_pool_indices_cpu.shape[0]):
        lo, hi = bounds_fn(i)
        alloc_len = max(0, hi - lo)
        if alloc_len == 0:
            continue
        req_idx = int(req_pool_indices_cpu[i].item())
        chunk = flat_loc[pt : pt + alloc_len].to(torch.int32)
        write_fn((req_idx, slice(lo, hi)), chunk)
        pt += alloc_len


def _write_per_req_slice(
    write_fn,
    req_pool_indices_cpu: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    flat_loc: torch.Tensor,
    ratio: int,
) -> None:

    _write_per_req(
        write_fn,
        req_pool_indices_cpu,
        flat_loc,
        lambda i: (
            int(prefix_lens_cpu[i].item()) // ratio,
            int(seq_lens_cpu[i].item()) // ratio,
        ),
    )
