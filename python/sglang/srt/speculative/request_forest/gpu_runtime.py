from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch

    from .forest_verify_input import ForestVerifyInput


_INTEGER_DTYPES = (torch.int32, torch.int64)
_NATIVE_COMMIT_KV = None
_COMMIT_LENS_WORKSPACES = threading.local()


def _native_commit_kv():
    global _NATIVE_COMMIT_KV
    if _NATIVE_COMMIT_KV is not None:
        return _NATIVE_COMMIT_KV
    try:
        operation = torch.ops.sgl_kernel.request_forest_commit_kv
    except AttributeError:
        return None
    _NATIVE_COMMIT_KV = operation
    return operation


@dataclass(slots=True)
class _StagedCommitLens:


    values: torch.Tensor
    completion: torch.cuda.Event
    source: torch.Tensor
    accepted_width: int
    physical_rows: int
    workspace: _CommitLensD2HWorkspace

    def committed_count(self) -> int:
        try:
            self.completion.synchronize()
            lengths = self.values.tolist()
        finally:
            self.workspace.busy = False
        committed_count = 0
        for length in lengths:
            if length < 0 or length > self.accepted_width:
                raise ValueError(
                    "commit_lens entries must be in [0, accepted_rows.shape[1]]"
                )
            committed_count += length
        if committed_count > self.physical_rows:
            raise ValueError("committed row count exceeds scratch slot count")
        return committed_count

    def wait(self) -> None:
        try:
            self.completion.synchronize()
        finally:
            self.workspace.busy = False


class _CommitLensD2HWorkspace:


    __slots__ = ("buffer", "busy", "completion", "stream")

    def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
        self.stream = torch.cuda.Stream(device=device)
        self.completion = torch.cuda.Event(blocking=False)
        self.busy = False
        self.buffer = torch.empty(
            64,
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )

    def _reserve(self, required: int, dtype: torch.dtype) -> torch.Tensor:
        if self.buffer.dtype != dtype or self.buffer.numel() < required:
            capacity = 1 << max(6, (required - 1).bit_length())
            self.buffer = torch.empty(
                capacity,
                dtype=dtype,
                device="cpu",
                pin_memory=True,
            )
        return self.buffer[:required]

    def stage(
        self,
        commit_lens: torch.Tensor,
        *,
        accepted_width: int,
        physical_rows: int,
    ) -> _StagedCommitLens:
        self.busy = True
        try:
            host_values = self._reserve(commit_lens.numel(), commit_lens.dtype)
            producer_stream = torch.cuda.current_stream(commit_lens.device)


            self.stream.wait_stream(producer_stream)
            with torch.cuda.stream(self.stream):
                host_values.copy_(commit_lens, non_blocking=True)
                self.completion.record(self.stream)
        except Exception:


            try:
                self.stream.synchronize()
            finally:
                self.busy = False
            raise
        return _StagedCommitLens(
            values=host_values,
            completion=self.completion,
            source=commit_lens,
            accepted_width=accepted_width,
            physical_rows=physical_rows,
            workspace=self,
        )


def _commit_lens_workspace(
    device: torch.device,
    dtype: torch.dtype,
) -> _CommitLensD2HWorkspace:
    workspaces = getattr(_COMMIT_LENS_WORKSPACES, "by_device_dtype", None)
    if workspaces is None:
        workspaces = {}
        _COMMIT_LENS_WORKSPACES.by_device_dtype = workspaces
    key = (device.index, dtype)
    candidates = workspaces.get(key)
    if candidates is None:
        candidates = []
        workspaces[key] = candidates
    for workspace in candidates:
        if not workspace.busy:
            return workspace
    workspace = _CommitLensD2HWorkspace(device, dtype)
    candidates.append(workspace)
    return workspace


def _reserve_token_slots(tree_cache: Any, physical_rows: int) -> torch.Tensor:


    from sglang.srt.mem_cache.allocation import alloc_token_slots

    return alloc_token_slots(tree_cache, physical_rows)


def ensure_tp_publish_ready(
    callback_error: Exception | None,
    *,
    tp_group: Any | None,
    scratch_device: torch.device,
) -> None:


    try:
        world_size = 1 if tp_group is None else int(tp_group.world_size)
        all_ranks_ready = callback_error is None
        if world_size > 1:
            ready = torch.tensor(
                [all_ranks_ready], dtype=torch.int32, device="cpu"
            )
            torch.distributed.all_reduce(
                ready,
                op=torch.distributed.ReduceOp.MIN,
                group=tp_group.cpu_group,
            )
            all_ranks_ready = bool(ready.item())
    except Exception:


        if scratch_device.type == "cuda":
            torch.cuda.current_stream(scratch_device).synchronize()
        raise

    if all_ranks_ready:
        return
    if scratch_device.type == "cuda":
        torch.cuda.current_stream(scratch_device).synchronize()
    if callback_error is not None:
        raise callback_error
    raise RuntimeError(
        "Request Forest cancelled KV publish because another TP rank failed "
        "while preparing accepted Draft state"
    )


@dataclass(slots=True)
class ForestKVCommit:


    committed_rows: torch.Tensor
    committed_slots: torch.Tensor
    commit_offsets: torch.Tensor
    commit_lens: torch.Tensor
    new_seq_lens: torch.Tensor
    released_slots: torch.Tensor


class ForestKVScratchTransaction:


    __slots__ = (
        "_allocator",
        "_prefix_lens",
        "_req_pool_indices",
        "_req_to_token",
        "_scratch_slots",
        "_staged_commit_lens",
        "_status",
    )

    def __init__(
        self,
        *,
        allocator: Any,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        prefix_lens: torch.Tensor,
        scratch_slots: torch.Tensor,
    ) -> None:
        page_size = getattr(allocator, "page_size", None)
        if page_size != 1:
            raise NotImplementedError(
                "Request Forest eager KV scratch requires page_size=1; "
                "page-granular accepted/rejected ownership is not implemented"
            )
        self._require_index_vector("req_pool_indices", req_pool_indices)
        self._require_index_vector("prefix_lens", prefix_lens)
        self._require_index_vector("scratch_slots", scratch_slots)
        if req_pool_indices.numel() != prefix_lens.numel():
            raise ValueError(
                "req_pool_indices and prefix_lens must have the same length"
            )
        if not isinstance(req_to_token, torch.Tensor) or req_to_token.dim() != 2:
            raise ValueError("req_to_token must be a two-dimensional Tensor")
        if req_to_token.device != scratch_slots.device:
            raise ValueError("KV transaction tensors must share one device")
        if (
            req_pool_indices.device != scratch_slots.device
            or prefix_lens.device != scratch_slots.device
        ):
            raise ValueError("KV transaction tensors must share one device")

        self._allocator = allocator
        self._req_to_token = req_to_token
        self._req_pool_indices = req_pool_indices
        self._prefix_lens = prefix_lens
        self._scratch_slots = scratch_slots
        self._staged_commit_lens: _StagedCommitLens | None = None
        self._status = "active"

    @staticmethod
    def _require_index_vector(name: str, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor) or value.dim() != 1:
            raise ValueError(f"{name} must be a one-dimensional Tensor")
        if value.dtype not in _INTEGER_DTYPES:
            raise TypeError(f"{name} must have dtype torch.int32 or torch.int64")
        if not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    @classmethod
    def reserve_for_batch(
        cls,
        batch: ScheduleBatch,
        verify_input: ForestVerifyInput,
    ) -> ForestKVScratchTransaction:


        allocator = batch.tree_cache.token_to_kv_pool_allocator
        if getattr(allocator, "page_size", None) != 1:
            raise NotImplementedError(
                "Request Forest eager KV scratch requires page_size=1; "
                "page-granular accepted/rejected ownership is not implemented"
            )


        scratch_slots = _reserve_token_slots(
            batch.tree_cache,
            verify_input.physical_rows,
        )
        try:
            transaction = cls(
                allocator=allocator,
                req_to_token=batch.req_to_token_pool.req_to_token,
                req_pool_indices=batch.req_pool_indices,
                prefix_lens=batch.seq_lens,
                scratch_slots=scratch_slots,
            )
            if transaction.request_count != verify_input.request_count:
                raise ValueError(
                    "Forest request_offsets must match the ScheduleBatch"
                )
            return transaction
        except Exception:
            allocator.free(scratch_slots)
            raise

    @property
    def scratch_slots(self) -> torch.Tensor:
        return self._scratch_slots

    @property
    def request_count(self) -> int:
        return self._req_pool_indices.numel()

    @property
    def physical_rows(self) -> int:
        return self._scratch_slots.numel()

    @property
    def is_active(self) -> bool:
        return self._status == "active"

    @property
    def is_committed(self) -> bool:
        return self._status == "committed"

    @property
    def is_rolled_back(self) -> bool:
        return self._status == "rolled_back"

    def validate_for(
        self,
        batch: ScheduleBatch,
        verify_input: ForestVerifyInput,
    ) -> None:


        self._require_active()
        if self.physical_rows != verify_input.physical_rows:
            raise ValueError("scratch transaction has the wrong physical row count")
        if self.request_count != verify_input.request_count:
            raise ValueError("scratch transaction has the wrong request count")
        if self._req_to_token is not batch.req_to_token_pool.req_to_token:
            raise ValueError("scratch transaction belongs to a different request pool")
        if self._allocator is not batch.tree_cache.token_to_kv_pool_allocator:
            raise ValueError("scratch transaction belongs to a different KV allocator")
        if self._req_pool_indices is not batch.req_pool_indices:
            raise ValueError("scratch transaction belongs to different request slots")
        if self._prefix_lens is not batch.seq_lens:
            raise ValueError("scratch transaction prefix lengths are stale")

    def stage_commit_lens(
        self,
        commit_lens: torch.Tensor,
        *,
        accepted_width: int,
    ) -> bool:


        self._require_active()
        self._require_index_vector("commit_lens", commit_lens)
        if commit_lens.device != self._scratch_slots.device:
            raise ValueError("commit_lens must share the scratch device")
        if commit_lens.numel() != self.request_count:
            raise ValueError("commit_lens must contain one entry per request")
        if accepted_width < 1:
            raise ValueError("accepted_width must be positive")
        if self._scratch_slots.device.type != "cuda" or _native_commit_kv() is None:
            return False
        if self._staged_commit_lens is not None:
            self._staged_commit_lens.wait()
        workspace = _commit_lens_workspace(
            self._scratch_slots.device,
            commit_lens.dtype,
        )
        self._staged_commit_lens = workspace.stage(
            commit_lens,
            accepted_width=accepted_width,
            physical_rows=self.physical_rows,
        )
        return True

    def commit(
        self,
        accepted_rows: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> ForestKVCommit:


        self._require_active()
        if not isinstance(accepted_rows, torch.Tensor) or accepted_rows.dim() != 2:
            raise ValueError("accepted_rows must be a two-dimensional Tensor")
        self._require_index_vector("commit_lens", commit_lens)
        if accepted_rows.dtype not in _INTEGER_DTYPES:
            raise TypeError(
                "accepted_rows must have dtype torch.int32 or torch.int64"
            )
        if accepted_rows.device != self._scratch_slots.device:
            raise ValueError("accepted_rows must share the scratch device")
        if commit_lens.device != self._scratch_slots.device:
            raise ValueError("commit_lens must share the scratch device")
        if accepted_rows.shape[0] != self.request_count:
            raise ValueError("accepted_rows must contain one row per request")
        if commit_lens.numel() != self.request_count:
            raise ValueError("commit_lens must contain one entry per request")
        if accepted_rows.shape[1] < 1:
            raise ValueError("accepted_rows needs room for each request anchor")

        native_commit = (
            _native_commit_kv()
            if self._scratch_slots.device.type == "cuda"
            else None
        )
        staged = self._staged_commit_lens
        can_use_native = (
            native_commit is not None
            and staged is not None
            and staged.source is commit_lens
            and staged.accepted_width == accepted_rows.shape[1]
        )
        if can_use_native:
            committed_count = staged.committed_count()
            (
                committed_rows,
                committed_slots,
                commit_offsets,
                new_seq_lens,
                released_slots,
            ) = native_commit(
                accepted_rows,
                commit_lens,
                self._scratch_slots,
                self._req_to_token,
                self._req_pool_indices,
                self._prefix_lens,
                committed_count,
            )
            self._staged_commit_lens = None
        else:
            if staged is not None:
                staged.wait()
                self._staged_commit_lens = None
            width = accepted_rows.shape[1]
            path_columns = torch.arange(
                width,
                dtype=commit_lens.dtype,
                device=commit_lens.device,
            ).view(1, width)
            valid = path_columns < commit_lens.view(-1, 1)


            committed_rows = accepted_rows[valid].to(torch.int64)
            committed_slots = self._scratch_slots.index_select(0, committed_rows)

            request_rows = self._req_pool_indices.view(-1, 1).expand(-1, width)
            token_columns = self._prefix_lens.view(-1, 1) + path_columns
            self._req_to_token[
                request_rows[valid].to(torch.int64),
                token_columns[valid].to(torch.int64),
            ] = committed_slots.to(self._req_to_token.dtype)

            keep_mask = torch.zeros(
                self.physical_rows,
                dtype=torch.bool,
                device=self._scratch_slots.device,
            )
            keep_mask[committed_rows] = True
            released_slots = self._scratch_slots[~keep_mask]
            commit_offsets = torch.nn.functional.pad(
                torch.cumsum(commit_lens, dim=0, dtype=torch.int32),
                (1, 0),
            )
            new_seq_lens = self._prefix_lens + commit_lens


        self._status = "committed"
        self._allocator.free(released_slots)

        return ForestKVCommit(
            committed_rows=committed_rows,
            committed_slots=committed_slots,
            commit_offsets=commit_offsets,
            commit_lens=commit_lens,
            new_seq_lens=new_seq_lens,
            released_slots=released_slots,
        )

    def rollback(self) -> bool:


        if self._status == "rolled_back":
            return False
        if self._status == "committed":
            raise RuntimeError("a committed Forest KV transaction cannot roll back")
        if self._staged_commit_lens is not None:
            self._staged_commit_lens.wait()
            self._staged_commit_lens = None
        self._allocator.free(self._scratch_slots)
        self._status = "rolled_back"
        return True

    def _require_active(self) -> None:
        if self._status != "active":
            raise RuntimeError(
                f"Forest KV transaction is already {self._status}"
            )

    def __enter__(self) -> ForestKVScratchTransaction:
        self._require_active()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        del exc_type, exc, traceback
        if self.is_active:
            self.rollback()
        return False


__all__ = [
    "ForestKVCommit",
    "ForestKVScratchTransaction",
    "ensure_tp_publish_ready",
]
