from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .draft_info import RequestForestDraftInput


@dataclass
class AdaptiveRound:
    decision: Any
    requests: tuple
    request_rows: tuple[int, ...]
    committed_lens: tuple[int, ...]
    decode_indices: tuple[int, ...]
    native_input: Any = None
    settlement_started: bool = False


@dataclass
class AdaptiveRequestForestDraftInput(RequestForestDraftInput):


    owner: Any = field(default=None, repr=False, compare=False)
    pending_round: AdaptiveRound | None = field(default=None, init=False, repr=False)

    def prepare_for_decode(self, batch) -> None:
        if self.owner is None:
            raise RuntimeError("Adaptive RF state is missing its worker")
        self.owner._prepare_adaptive_round(batch, self)

    def _require_unprepared(self) -> None:
        if self.pending_round is not None:
            raise RuntimeError("Cannot filter/merge an already prepared Adaptive RF round")

    def filter_batch(self, new_indices, new_indices_cpu=None) -> None:
        self._require_unprepared()
        super().filter_batch(new_indices, new_indices_cpu)

    def merge_batch(self, spec_info) -> None:
        self._require_unprepared()
        if not isinstance(spec_info, AdaptiveRequestForestDraftInput):
            raise TypeError("Adaptive RF requires matching adaptive relay states")
        spec_info._require_unprepared()
        if self.owner is not spec_info.owner:
            raise ValueError("Cannot merge Adaptive RF states from different workers")
        super().merge_batch(spec_info)


def release_native_reserve(batch, round_state: AdaptiveRound, committed_ends) -> None:

    if round_state.settlement_started:
        raise RuntimeError("Native reserve settlement cannot be repeated")
    allocator = batch.token_to_kv_pool_allocator
    if allocator.page_size != 1:
        raise RuntimeError("Adaptive RF reserve settlement requires page_size=1")
    ends = tuple(int(x) for x in committed_ends)
    if len(ends) != len(round_state.requests):
        raise ValueError("Native commit lengths do not match the prepared batch")
    table = batch.req_to_token_pool.req_to_token
    ranges = []
    slots = []
    for req, row, prefix, end in zip(
        round_state.requests, round_state.request_rows,
        round_state.committed_lens, ends,
    ):
        allocated = int(req.kv.kv_allocated_len)
        if int(req.kv_committed_len) != prefix:
            raise RuntimeError("Native forward unexpectedly changed committed KV length")
        if not prefix <= end <= allocated <= table.shape[1]:
            raise RuntimeError("Invalid native reserve/commit bounds")
        ranges.append((req, row, end, allocated))
        if end < allocated:
            slots.append(table[row, end:allocated])


    round_state.settlement_started = True
    if slots:
        allocator.free(torch.cat(slots).to(torch.int64))
    for req, row, end, allocated in ranges:
        table[row, end:allocated] = 0
        req.kv.kv_allocated_len = end


def release_native_reserve_at_boundary(batches) -> None:

    requests = []
    rows = []
    seen = set()
    owner_batch = None
    for batch in batches:
        if batch is None:
            continue
        for req in batch.reqs:
            if id(req) in seen:
                continue
            seen.add(id(req))
            if req.finished() or req.is_retracted:
                continue
            if req.req_pool_idx is None:
                raise RuntimeError("Live Adaptive RF request has no request-table row")
            if owner_batch is None:
                owner_batch = batch
            elif (
                batch.req_to_token_pool is not owner_batch.req_to_token_pool
                or batch.token_to_kv_pool_allocator
                is not owner_batch.token_to_kv_pool_allocator
            ):
                raise RuntimeError("Adaptive RF transition spans different KV pools")
            requests.append(req)
            rows.append(int(req.req_pool_idx))
    if not requests:
        return
    if len(set(rows)) != len(rows):
        raise RuntimeError("Live Adaptive RF requests share a request-table row")
    prefixes = tuple(int(req.kv_committed_len) for req in requests)
    round_state = AdaptiveRound(
        None, tuple(requests), tuple(rows), prefixes,
        tuple(req.decode_batch_idx for req in requests),
    )
    release_native_reserve(owner_batch, round_state, prefixes)
