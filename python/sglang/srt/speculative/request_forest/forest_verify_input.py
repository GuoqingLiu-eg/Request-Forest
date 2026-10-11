from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

    from .gpu_runtime import ForestKVScratchTransaction
    from .native import NativeVerifyPlan


_NATIVE_BUILD_TREE_ATTENTION = None


def _native_build_tree_attention():
    global _NATIVE_BUILD_TREE_ATTENTION
    if _NATIVE_BUILD_TREE_ATTENTION is not None:
        return _NATIVE_BUILD_TREE_ATTENTION
    try:
        operation = torch.ops.sgl_kernel.request_forest_build_tree_attention
    except AttributeError:
        return None
    _NATIVE_BUILD_TREE_ATTENTION = operation
    return operation


@dataclass(slots=True)
class ForestTreeAttentionLayout:


    page_table: torch.Tensor
    cache_seqlens_int32: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seq_len_q: int
    max_seq_len_k: int


@dataclass(slots=True)
class ForestTreeAttentionWorkspace:


    page_table: torch.Tensor | None = None
    cache_seqlens: torch.Tensor | None = None
    cu_seqlens_q: torch.Tensor | None = None

    @staticmethod
    def _capacity(required: int) -> int:
        return 1 << max(0, required - 1).bit_length()

    @classmethod
    def _reserve(
        cls,
        value: torch.Tensor | None,
        *,
        required: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if (
            value is None
            or value.device != device
            or value.dtype != dtype
            or value.numel() < required
        ):
            value = torch.empty(
                cls._capacity(required),
                dtype=dtype,
                device=device,
            )
        return value

    def views(
        self,
        *,
        logical_rows: int,
        max_tree_depth: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        page_table_items = logical_rows * max_tree_depth
        self.page_table = self._reserve(
            self.page_table,
            required=page_table_items,
            dtype=torch.int32,
            device=device,
        )
        self.cache_seqlens = self._reserve(
            self.cache_seqlens,
            required=logical_rows,
            dtype=torch.int32,
            device=device,
        )
        self.cu_seqlens_q = self._reserve(
            self.cu_seqlens_q,
            required=logical_rows + 1,
            dtype=torch.int32,
            device=device,
        )
        return (
            self.page_table[:page_table_items].view(logical_rows, max_tree_depth),
            self.cache_seqlens[:logical_rows],
            self.cu_seqlens_q[: logical_rows + 1],
        )


def _build_gdn_tree_metadata(
    *,
    request_offsets: torch.Tensor,
    first_child: torch.Tensor,
    next_sibling: torch.Tensor,
    logical_rows: int,
    max_request_rows: int,
    parent_rows: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:


    request_count = request_offsets.numel() - 1
    device = request_offsets.device
    retrieve_next_token = torch.full(
        (request_count, max_request_rows),
        -1,
        dtype=torch.int64,
        device=device,
    )
    retrieve_next_sibling = torch.full_like(retrieve_next_token, -1)
    rows = torch.arange(logical_rows, dtype=torch.int64, device=device)
    request_slots = torch.bucketize(rows, request_offsets[1:], right=True)
    request_starts = request_offsets[request_slots].to(torch.int64)
    packed_to_dense = request_slots * max_request_rows + rows - request_starts
    for source, destination in (
        (first_child, retrieve_next_token),
        (next_sibling, retrieve_next_sibling),
    ):
        links = source[:logical_rows].to(torch.int64)
        destination.view(-1).index_copy_(
            0,
            packed_to_dense,
            torch.where(links >= 0, links - request_starts, -1),
        )
    retrieve_parent_token = None
    if parent_rows is not None:


        retrieve_parent_token = torch.zeros_like(retrieve_next_token)
        parents = parent_rows[:logical_rows].to(torch.int64)
        retrieve_parent_token.view(-1).index_copy_(
            0,
            packed_to_dense,
            torch.where(parents >= 0, parents - request_starts, 0),
        )
    return (
        retrieve_next_token,
        retrieve_next_sibling,
        packed_to_dense,
        retrieve_parent_token,
    )


@dataclass(slots=True)
class ForestVerifyInput(SpecInput):


    draft_token: torch.Tensor
    positions: torch.Tensor
    first_child: torch.Tensor
    next_sibling: torch.Tensor
    has_unverified_children: torch.Tensor
    request_offsets: torch.Tensor
    anchor_rows: torch.Tensor
    ancestor_row_ptr: torch.Tensor
    ancestor_columns: torch.Tensor
    logical_rows: int
    max_request_rows: int
    max_tree_depth: int
    request_slots: torch.Tensor | None = None
    retrieve_next_token: torch.Tensor | None = None
    retrieve_next_sibling: torch.Tensor | None = None
    gdn_packed_to_dense: torch.Tensor | None = None
    gdn_retrieve_parent_token: torch.Tensor | None = None


    gdn_graph_capacity: int | None = None


    capture_hidden_mode: object | None = None
    tree_attention_workspace: ForestTreeAttentionWorkspace | None = None
    ragged_verify_layout: RaggedVerifyLayout | None = None


    num_tokens_per_req: int = 1
    num_tokens_for_logprob_per_req: int = 1

    def __post_init__(self) -> None:
        SpecInput.__init__(self, SpecInputType.REQUEST_FOREST_VERIFY)
        if type(self.logical_rows) is not int or self.logical_rows < 1:
            raise ValueError("logical_rows must be a positive integer")
        if type(self.max_request_rows) is not int or self.max_request_rows < 1:
            raise ValueError("max_request_rows must be a positive integer")
        if type(self.max_tree_depth) is not int or self.max_tree_depth < 1:
            raise ValueError("max_tree_depth must be a positive integer")

        row_fields = (
            ("draft_token", self.draft_token, torch.int64),
            ("positions", self.positions, torch.int64),
            ("first_child", self.first_child, torch.int32),
            ("next_sibling", self.next_sibling, torch.int32),
            (
                "has_unverified_children",
                self.has_unverified_children,
                torch.bool,
            ),
        )
        device = self.draft_token.device
        physical_rows = self.draft_token.numel()
        if self.logical_rows > physical_rows:
            raise ValueError("logical_rows cannot exceed physical_rows")
        for name, value, dtype in row_fields:
            self._require_vector(name, value, dtype=dtype, device=device)
            if value.numel() != physical_rows:
                raise ValueError(f"{name} must contain physical_rows entries")

        for name, value, expected_size, dtype in (
            (
                "request_offsets",
                self.request_offsets,
                self.request_count + 1,
                torch.int32,
            ),
            ("anchor_rows", self.anchor_rows, self.request_count, torch.int32),
            (
                "ancestor_row_ptr",
                self.ancestor_row_ptr,
                physical_rows + 1,
                torch.int32,
            ),
        ):
            self._require_vector(name, value, dtype=dtype, device=device)
            if value.numel() != expected_size:
                raise ValueError(f"{name} has the wrong number of entries")
        self._require_vector(
            "ancestor_columns",
            self.ancestor_columns,
            dtype=torch.int64,
            device=device,
        )
        if self.request_slots is not None:
            self._require_vector(
                "request_slots",
                self.request_slots,
                dtype=torch.int64,
                device=device,
            )
            if self.request_slots.numel() != physical_rows:
                raise ValueError("request_slots must contain physical_rows entries")

        gdn_values = (
            self.retrieve_next_token,
            self.retrieve_next_sibling,
            self.gdn_packed_to_dense,
        )
        if any(value is not None for value in gdn_values):
            if not all(value is not None for value in gdn_values):
                raise ValueError("GDN tree metadata must be supplied together")
            assert self.retrieve_next_token is not None
            assert self.retrieve_next_sibling is not None
            assert self.gdn_packed_to_dense is not None
            expected_shape = (self.request_count, self.max_request_rows)
            for name, value in (
                ("retrieve_next_token", self.retrieve_next_token),
                ("retrieve_next_sibling", self.retrieve_next_sibling),
                ("gdn_retrieve_parent_token", self.gdn_retrieve_parent_token),
            ):
                if value is None:
                    continue
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"{name} must be a torch.Tensor")
                if value.shape != expected_shape:
                    raise ValueError(f"{name} must have shape {expected_shape}")
                if value.dtype != torch.int64:
                    raise TypeError(f"{name} must have dtype torch.int64")
                if value.device != device:
                    raise ValueError(
                        "all ForestVerifyInput tensors must share one device"
                    )
                if not value.is_contiguous():
                    raise ValueError(f"{name} must be contiguous")
            self._require_vector(
                "gdn_packed_to_dense",
                self.gdn_packed_to_dense,
                dtype=torch.int64,
                device=device,
            )
            if self.gdn_packed_to_dense.numel() != self.logical_rows:
                raise ValueError(
                    "gdn_packed_to_dense must contain logical_rows entries"
                )

    @staticmethod
    def _require_vector(
        name: str,
        value: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.dim() != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if value.dtype != dtype:
            raise TypeError(f"{name} must have dtype {dtype}")
        if value.device != device:
            raise ValueError("all ForestVerifyInput tensors must share one device")
        if not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    @property
    def request_count(self) -> int:
        return self.request_offsets.numel() - 1

    @property
    def token_ids(self) -> torch.Tensor:
        return self.draft_token

    @property
    def draft_token_num(self) -> int:
        return self.max_request_rows

    @property
    def physical_rows(self) -> int:
        return self.draft_token.numel()

    @property
    def tree_topk(self) -> int:
        return -1

    def ensure_gdn_tree_metadata(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:


        if self.request_slots is None:
            self.request_slots = torch.full(
                (self.physical_rows,),
                -1,
                dtype=torch.int64,
                device=self.draft_token.device,
            )
            offsets = self.request_offsets.tolist()
            for request_slot, (start, end) in enumerate(zip(offsets, offsets[1:])):
                self.request_slots[start:end] = request_slot

        if self.retrieve_next_token is None:
            (
                self.retrieve_next_token,
                self.retrieve_next_sibling,
                self.gdn_packed_to_dense,
                self.gdn_retrieve_parent_token,
            ) = _build_gdn_tree_metadata(
                request_offsets=self.request_offsets,
                first_child=self.first_child,
                next_sibling=self.next_sibling,
                logical_rows=self.logical_rows,
                max_request_rows=self.max_request_rows,
            )
        assert self.retrieve_next_sibling is not None
        assert self.gdn_packed_to_dense is not None
        return (
            self.retrieve_next_token,
            self.retrieve_next_sibling,
            self.gdn_packed_to_dense,
        )

    @classmethod
    def from_native_plan(
        cls,
        plan: NativeVerifyPlan,
        *,
        device: torch.device | str,
        non_blocking: bool = True,
    ) -> ForestVerifyInput:


        from .native import NativeVerifyPlan

        if not isinstance(plan, NativeVerifyPlan):
            raise TypeError("plan must be a NativeVerifyPlan")

        logical_rows = plan.logical_rows
        request_lens = plan.request_offsets[1:] - plan.request_offsets[:-1]
        ancestor_lens = (
            plan.ancestor_row_ptr[1 : logical_rows + 1]
            - plan.ancestor_row_ptr[:logical_rows]
        )
        max_request_rows = int(request_lens.max().item())
        max_tree_depth = int(ancestor_lens.max().item())
        target_device = torch.device(device)
        (
            retrieve_next_token,
            retrieve_next_sibling,
            gdn_packed_to_dense,
            gdn_retrieve_parent_token,
        ) = _build_gdn_tree_metadata(
            request_offsets=plan.request_offsets,
            first_child=plan.first_child,
            next_sibling=plan.next_sibling,
            logical_rows=logical_rows,
            max_request_rows=max_request_rows,
            parent_rows=plan.parent_rows,
        )

        def move(value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
            return value.to(
                device=target_device,
                dtype=dtype,
                non_blocking=non_blocking,
            )

        from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

        request_offsets = move(plan.request_offsets, torch.int32)
        ragged_verify_layout = RaggedVerifyLayout(
            verify_lens=request_offsets[1:] - request_offsets[:-1],
            graph_num_tokens=plan.physical_rows,
            extend_start_loc=request_offsets[:-1],
            qo_indptr_device=request_offsets,
        )
        return cls(
            draft_token=move(plan.token_ids, torch.int64),
            positions=move(plan.positions, torch.int64),
            first_child=move(plan.first_child, torch.int32),
            next_sibling=move(plan.next_sibling, torch.int32),
            has_unverified_children=move(plan.has_unverified_children, torch.bool),
            request_offsets=request_offsets,
            anchor_rows=move(plan.anchor_rows, torch.int32),
            ancestor_row_ptr=move(plan.ancestor_row_ptr, torch.int32),
            ancestor_columns=move(plan.ancestor_columns, torch.int64),
            logical_rows=logical_rows,
            max_request_rows=max_request_rows,
            max_tree_depth=max_tree_depth,
            request_slots=move(plan.request_slots, torch.int64),
            retrieve_next_token=move(retrieve_next_token, torch.int64),
            retrieve_next_sibling=move(retrieve_next_sibling, torch.int64),
            gdn_packed_to_dense=move(gdn_packed_to_dense, torch.int64),
            gdn_retrieve_parent_token=move(gdn_retrieve_parent_token, torch.int64),
            ragged_verify_layout=ragged_verify_layout,
        )

    def build_tree_attention_layout(
        self,
        out_cache_loc: torch.Tensor,
        *,
        workspace: ForestTreeAttentionWorkspace | None = None,
        max_tree_depth: int | None = None,
    ) -> ForestTreeAttentionLayout:


        if not isinstance(out_cache_loc, torch.Tensor):
            raise TypeError("out_cache_loc must be a torch.Tensor")
        if out_cache_loc.dim() != 1:
            raise ValueError("out_cache_loc must be one-dimensional")
        if out_cache_loc.device != self.draft_token.device:
            raise ValueError("out_cache_loc must share ForestVerifyInput's device")
        if out_cache_loc.dtype not in (torch.int32, torch.int64):
            raise TypeError("out_cache_loc must have an integer index dtype")
        if out_cache_loc.numel() < self.physical_rows:
            raise ValueError("out_cache_loc needs one KV location per physical row")

        if max_tree_depth is None:
            max_tree_depth = self.max_tree_depth
        elif type(max_tree_depth) is not int or max_tree_depth < 1:
            raise ValueError("max_tree_depth must be a positive integer")
        if max_tree_depth < self.max_tree_depth:
            raise ValueError(
                "max_tree_depth cannot be smaller than the live Forest depth"
            )
        if workspace is None:
            workspace = self.tree_attention_workspace

        native_operation = (
            _native_build_tree_attention()
            if out_cache_loc.device.type == "cuda"
            else None
        )
        if native_operation is not None:
            if workspace is None:
                page_table = torch.empty(
                    (self.physical_rows, max_tree_depth),
                    dtype=torch.int32,
                    device=out_cache_loc.device,
                )
                cache_seqlens = torch.empty(
                    self.physical_rows,
                    dtype=torch.int32,
                    device=out_cache_loc.device,
                )
                cu_seqlens_q = torch.empty(
                    self.physical_rows + 1,
                    dtype=torch.int32,
                    device=out_cache_loc.device,
                )
            else:
                page_table, cache_seqlens, cu_seqlens_q = workspace.views(
                    logical_rows=self.physical_rows,
                    max_tree_depth=max_tree_depth,
                    device=out_cache_loc.device,
                )
            native_operation(
                self.ancestor_row_ptr,
                self.ancestor_columns,
                out_cache_loc,
                page_table,
                cache_seqlens,
                cu_seqlens_q,
            )
            return ForestTreeAttentionLayout(
                page_table=page_table,
                cache_seqlens_int32=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=self.ancestor_row_ptr,
                max_seq_len_q=1,
                max_seq_len_k=max_tree_depth,
            )

        return self._build_tree_attention_layout_torch(
            out_cache_loc,
            workspace=workspace,
            max_tree_depth=max_tree_depth,
        )

    def prepare_for_verify(
        self,
        batch: ScheduleBatch,
        target_worker: TpModelWorker,
        *,
        transaction: ForestKVScratchTransaction | None = None,
    ):


        from sglang.srt.configs.hybrid_arch import mambaish_config
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
            ForwardMode,
        )
        from sglang.srt.speculative.spec_utils import (
            prepare_mamba_track_for_verify,
        )

        from .gpu_runtime import ForestKVScratchTransaction

        if batch.forward_mode.is_idle():
            raise NotImplementedError(
                "Request Forest eager verification does not need idle-rank "
                "participation in its single-card phase"
            )
        if batch.seq_lens.device != self.draft_token.device:
            raise ValueError(
                "ForestVerifyInput and ScheduleBatch must share one device"
            )

        if transaction is None:
            transaction = ForestKVScratchTransaction.reserve_for_batch(
                batch,
                self,
            )
        else:
            transaction.validate_for(batch, self)

        previous = (
            batch.input_ids,
            batch.out_cache_loc,
            batch.spec_info,
            batch.forward_mode,
        )
        try:
            batch.input_ids = self.draft_token
            batch.out_cache_loc = transaction.scratch_slots
            batch.spec_info = self
            batch.forward_mode = ForwardMode.TARGET_VERIFY
            model_config = getattr(target_worker.model_runner, "model_config", None)
            if model_config is not None and mambaish_config(model_config) is not None:
                prepare_mamba_track_for_verify(batch)
            forward_batch = ForwardBatch.init_new(
                batch,
                target_worker.model_runner,
                capture_hidden_mode=CaptureHiddenMode.FULL,
                return_hidden_states_before_norm=False,
            )
        except Exception:
            (
                batch.input_ids,
                batch.out_cache_loc,
                batch.spec_info,
                batch.forward_mode,
            ) = previous
            if transaction.is_active:
                transaction.rollback()
            raise

        return forward_batch, transaction

    def _build_tree_attention_layout_torch(
        self,
        out_cache_loc: torch.Tensor,
        *,
        workspace: ForestTreeAttentionWorkspace | None = None,
        max_tree_depth: int | None = None,
    ) -> ForestTreeAttentionLayout:


        if max_tree_depth is None:
            max_tree_depth = self.max_tree_depth
        row_ptr = self.ancestor_row_ptr
        cache_seqlens = row_ptr[1:] - row_ptr[:-1]
        if workspace is None:
            page_table = torch.zeros(
                (self.physical_rows, max_tree_depth),
                dtype=torch.int32,
                device=out_cache_loc.device,
            )
            cu_seqlens_q = torch.arange(
                self.physical_rows + 1,
                dtype=torch.int32,
                device=out_cache_loc.device,
            )
        else:
            page_table, staged_cache_seqlens, cu_seqlens_q = workspace.views(
                logical_rows=self.physical_rows,
                max_tree_depth=max_tree_depth,
                device=out_cache_loc.device,
            )
            page_table.zero_()
            staged_cache_seqlens.copy_(cache_seqlens)
            cache_seqlens = staged_cache_seqlens
            torch.arange(
                self.physical_rows + 1,
                dtype=torch.int32,
                device=out_cache_loc.device,
                out=cu_seqlens_q,
            )

        ancestor_nnz = self.ancestor_columns.numel()
        if ancestor_nnz:
            repeats = cache_seqlens.to(torch.int64)
            row_ids = torch.repeat_interleave(
                torch.arange(
                    self.physical_rows,
                    dtype=torch.int64,
                    device=out_cache_loc.device,
                ),
                repeats,
                output_size=ancestor_nnz,
            )
            row_starts = torch.repeat_interleave(
                row_ptr[:-1].to(torch.int64),
                repeats,
                output_size=ancestor_nnz,
            )
            column_ids = (
                torch.arange(
                    ancestor_nnz,
                    dtype=torch.int64,
                    device=out_cache_loc.device,
                )
                - row_starts
            )
            ancestor_locations = out_cache_loc.index_select(
                0, self.ancestor_columns
            ).to(torch.int32)
            page_table[row_ids, column_ids] = ancestor_locations

        return ForestTreeAttentionLayout(
            page_table=page_table,
            cache_seqlens_int32=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=row_ptr,
            max_seq_len_q=1,
            max_seq_len_k=max_tree_depth,
        )


__all__ = [
    "ForestTreeAttentionLayout",
    "ForestTreeAttentionWorkspace",
    "ForestVerifyInput",
]
