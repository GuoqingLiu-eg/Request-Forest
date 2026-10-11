from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Iterator

import torch

from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method
from sglang.srt.lora.layers import unwrap_lora_layer
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.allocation import alloc_token_slots
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.runtime_context import get_schedule, mamba_track_grid
from sglang.srt.speculative.dflash_utils import is_dense_head_weight
from sglang.srt.speculative.dflash_worker_v2 import (
    DFlashWorkerV2,
    _SelectorLatticeView,
    _build_selector_lattice_proposal,
    _resolve_dflash_embedding_module,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

from .acceptance_calibrator import (
    AcceptanceFeedbackMetadata,
    DepthConfidenceBetaEMA,
    confidence_bin,
)
from .classic_chain_runtime import (
    ClassicChainVerifyTemplate,
    run_classic_chain_target_verify_eager,
)
from .crossover_profile import load_crossover_profile
from .budget_controller import (
    BudgetDecisionSource,
    PreDraftBudgetDecision,
    RequestForestBudgetController,
    VerificationBudgetDecision,
)
from .branch_labels import (
    BranchLabelCollector,
    BranchRoundOutcome,
)
from .dflash_adapter import (
    DFlash2LatticeLowerer,
    DFlashChainLowerer,
    DFlashLoweringResult,
)
from .draft_info import RequestForestDraftInput
from .eager_runtime import run_forest_target_verify_eager
from .external_adapter import ExternalPolicyAdapter
from .external_marginals import build_marginal_proposal
from .forest_verify_input import ForestTreeAttentionWorkspace, ForestVerifyInput
from .gpu_accept import ForestGreedyResult
from .gpu_runtime import ForestKVScratchTransaction
from .ir import RequestRef
from .native import NativeVerifyPlan, TensorForestArena
from .native_loader import load_request_forest_native_ops
from .profile import (
    AcceptanceCorrectionTable,
    VerifyShape,
    VerifyShapeProfileDB,
    build_runtime_fingerprint,
)
from .target_timing import (
    AsyncTargetVerificationTimer,
    TargetTimingKey,
    TargetTimingObservation,
    TargetTimingScope,
)


logger = logging.getLogger(__name__)


_NATIVE_PLAN_BOOL_OUTPUTS = frozenset((7, 8, 9))


@dataclass(frozen=True, slots=True)
class _CompiledForestRound:


    plan: NativeVerifyPlan
    budget_decision: VerificationBudgetDecision
    tail_evidence: _TailRoundEvidence | None = None
    acceptance_feedback: AcceptanceFeedbackMetadata | None = None
    acceptance_calibrator: DepthConfidenceBetaEMA | None = None


@dataclass(frozen=True, slots=True)
class _PreparedForestRound:


    verify_input: ForestVerifyInput
    budget_decision: VerificationBudgetDecision
    tail_evidence: _TailRoundEvidence | None = None
    acceptance_feedback: AcceptanceFeedbackMetadata | None = None
    acceptance_calibrator: DepthConfidenceBetaEMA | None = None


@dataclass(slots=True)
class _PendingAcceptanceFeedback:


    metadata: AcceptanceFeedbackMetadata
    calibrator: DepthConfidenceBetaEMA
    eos_token_ids: frozenset[int]
    request_ignore_eos: tuple[bool, ...]
    source: torch.Tensor


@dataclass(frozen=True, slots=True)
class _TailRequestEvidence:


    request_id: str
    path_values: tuple[float, ...]
    depths: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _TailRoundEvidence:


    requests: tuple[_TailRequestEvidence, ...]


def _native_plan_outputs(plan: NativeVerifyPlan) -> tuple[torch.Tensor, ...]:
    return (
        plan.token_ids,
        plan.parent_rows,
        plan.first_child,
        plan.next_sibling,
        plan.depths,
        plan.positions,
        plan.request_slots,
        plan.valid_mask,
        plan.candidate_mask,
        plan.has_unverified_children,
        plan.packed_to_node,
        plan.node_to_packed,
        plan.request_offsets,
        plan.anchor_rows,
        plan.ancestor_row_ptr,
        plan.ancestor_columns,
        torch.tensor((plan.logical_rows,), dtype=torch.int64, device="cpu"),
    )


def _pack_native_plan(plan: NativeVerifyPlan) -> dict[str, Any]:


    outputs = _native_plan_outputs(plan)
    lengths = tuple(output.numel() for output in outputs)
    integer_values = torch.cat(
        tuple(
            output.reshape(-1)
            for index, output in enumerate(outputs)
            if index not in _NATIVE_PLAN_BOOL_OUTPUTS
        )
    )
    boolean_values = torch.cat(
        tuple(
            outputs[index].reshape(-1)
            for index in sorted(_NATIVE_PLAN_BOOL_OUTPUTS)
        )
    )
    return {
        "error": None,
        "lengths": lengths,
        "integer_values": integer_values,
        "boolean_values": boolean_values,
    }


def _unpack_native_plan(
    payload: dict[str, Any],
    *,
    request_refs: tuple[RequestRef, ...],
) -> NativeVerifyPlan:
    error = payload.get("error")
    if error is not None:
        raise RuntimeError(
            f"Request Forest plan compilation failed on TP rank 0: {error}"
        )
    lengths = payload["lengths"]
    if len(lengths) != 17:
        raise RuntimeError("Request Forest TP plan payload has an invalid schema")
    integer_values = payload["integer_values"]
    boolean_values = payload["boolean_values"]
    integer_offset = 0
    boolean_offset = 0
    outputs = []
    for index, length in enumerate(lengths):
        length = int(length)
        if index in _NATIVE_PLAN_BOOL_OUTPUTS:
            outputs.append(boolean_values[boolean_offset : boolean_offset + length])
            boolean_offset += length
        else:
            outputs.append(integer_values[integer_offset : integer_offset + length])
            integer_offset += length
    if (
        integer_offset != integer_values.numel()
        or boolean_offset != boolean_values.numel()
    ):
        raise RuntimeError("Request Forest TP plan payload lengths do not match its data")
    return NativeVerifyPlan.from_kernel_outputs(
        outputs,
        request_refs=request_refs,
    )


def _pack_compiled_round(compiled: _CompiledForestRound) -> dict[str, Any]:
    payload = _pack_native_plan(compiled.plan)
    payload["budget_decision"] = compiled.budget_decision
    return payload


def _unpack_compiled_round(
    payload: dict[str, Any],
    *,
    request_refs: tuple[RequestRef, ...],
) -> _CompiledForestRound:
    plan = _unpack_native_plan(payload, request_refs=request_refs)
    decision = payload.get("budget_decision")
    if not isinstance(decision, VerificationBudgetDecision):
        raise RuntimeError(
            "Request Forest TP plan payload has no verification budget decision"
        )
    return _CompiledForestRound(
        plan=plan,
        budget_decision=decision,


        tail_evidence=None,
    )


def _cuda_graph_capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _replicated_plans_agree(
    tp_group,
    *,
    ok: bool,
    device: torch.device,
) -> bool:


    local = torch.tensor((0 if ok else 1,), dtype=torch.int64, device=device)
    world = int(tp_group.world_size)
    if world == 1:
        return ok
    gathered = torch.empty((world,), dtype=torch.int64, device=device)
    torch.distributed.all_gather_into_tensor(
        gathered,
        local,
        group=tp_group.device_group,
    )
    return bool(torch.equal(gathered, gathered[:1].expand_as(gathered)) and ok)


def _mark_truncated_frontier(
    plan: NativeVerifyPlan,
    lowering: DFlashLoweringResult,
) -> NativeVerifyPlan:


    boundary_rows: list[torch.Tensor] = []
    if lowering.truncated_root_request_slots:
        root_slots = torch.tensor(
            lowering.truncated_root_request_slots,
            dtype=torch.int64,
            device="cpu",
        )
        boundary_rows.append(plan.anchor_rows.index_select(0, root_slots))
    if lowering.truncated_parent_node_ids:
        parent_ids = torch.tensor(
            lowering.truncated_parent_node_ids,
            dtype=torch.int64,
            device="cpu",
        )
        packed_parent_rows = plan.node_to_packed.index_select(0, parent_ids)


        packed_parent_rows = packed_parent_rows[packed_parent_rows >= 0]
        if packed_parent_rows.numel():
            boundary_rows.append(packed_parent_rows)
    for rows in boundary_rows:
        plan.has_unverified_children.index_fill_(0, rows, True)
    return plan

def _build_acceptance_feedback_metadata(
    *,
    plan: NativeVerifyPlan,
    lowering: DFlashLoweringResult,
    bin_count: int,
) -> AcceptanceFeedbackMetadata | None:


    if plan.logical_rows <= len(plan.request_refs):
        return None
    if lowering.conditional_probabilities.numel() != lowering.node_count:
        raise RuntimeError(
            "calibrated DFlash2 lowering must retain one raw score per node"
        )

    logical_rows = plan.logical_rows
    packed_nodes = plan.packed_to_node[:logical_rows].tolist()
    raw_probabilities = lowering.conditional_probabilities.tolist()
    start_node_id = lowering.node_ids.start
    confidence_bins = tuple(
        -1
        if node_id < 0
        else confidence_bin(
            float(raw_probabilities[node_id - start_node_id]),
            bin_count,
        )
        for node_id in packed_nodes
    )
    return AcceptanceFeedbackMetadata(
        token_ids=tuple(plan.token_ids[:logical_rows].tolist()),
        first_child=tuple(plan.first_child[:logical_rows].tolist()),
        next_sibling=tuple(plan.next_sibling[:logical_rows].tolist()),
        depths=tuple(plan.depths[:logical_rows].tolist()),
        confidence_bins=confidence_bins,
        anchor_rows=tuple(plan.anchor_rows.tolist()),
        logical_rows=logical_rows,
    )


def _make_forest_lowerer(
    *,
    selector,
    max_nodes: int,
    max_nodes_per_request: int | None = None,
    selection_mode: str = "global-forest",
):


    if selector is None:
        if selection_mode != "global-forest":
            raise ValueError(
                "Request Forest selection ablations require a DFlash2 "
                "selector lattice"
            )
        return DFlashChainLowerer(max_nodes=max_nodes)
    return DFlash2LatticeLowerer(
        max_nodes=max_nodes,
        top_k=int(selector.top_k),
        max_nodes_per_request=max_nodes_per_request,
        selection_mode=selection_mode,
    )


def _forest_mamba_commit_steps(
    *,
    request_offsets: torch.Tensor,
    accept_index: torch.Tensor,
    accept_lens: torch.Tensor,
    seq_lens_pre_verify: torch.Tensor,
    seq_lens_post_verify: torch.Tensor,
    request_rows: torch.Tensor,
    track_interval: int | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:


    request_count = accept_lens.numel()
    rows = request_rows[:request_count]
    starts = request_offsets[:request_count].to(torch.int64)
    last_columns = (accept_lens - 1).to(torch.int64)
    last_steps = accept_index[rows, last_columns].to(torch.int64) - starts
    if track_interval is None:
        return last_steps, None

    tracking_points = (
        seq_lens_post_verify // track_interval * track_interval
    )
    tracking_columns = torch.clamp(
        tracking_points - seq_lens_pre_verify - 1,
        min=0,
    ).to(torch.int64)
    crosses_boundary = (
        seq_lens_pre_verify // track_interval
        != seq_lens_post_verify // track_interval
    )
    can_track = crosses_boundary & (
        tracking_columns < accept_lens.to(torch.int64)
    )
    safe_columns = tracking_columns.clamp_max(accept_index.shape[1] - 1)
    candidate_steps = (
        accept_index[rows, safe_columns].to(torch.int64) - starts
    )
    track_steps = torch.where(
        can_track,
        candidate_steps,
        torch.full_like(candidate_steps, -1),
    )
    return last_steps, track_steps


def _lower_forest_proposal(
    *,
    lowerer,
    selector,
    arena: TensorForestArena,
    proposal,
    capacity: int,
    calibration_multipliers: torch.Tensor | None = None,
    d2_mix_lambda: float = 0.0,
) -> DFlashLoweringResult:


    if selector is None:
        return lowerer.lower(
            arena=arena,
            candidate_token_ids=proposal,
            capacity=capacity,
        )
    if calibration_multipliers is None:
        return lowerer.lower(
            arena=arena,
            candidate_ids=proposal.candidate_ids,
            scores=proposal.scores,
            capacity=capacity,
            collect_features=False,
            d2_mix_lambda=d2_mix_lambda,
        )
    return lowerer.lower(
        arena=arena,
        candidate_ids=proposal.candidate_ids,
        scores=proposal.scores,
        capacity=capacity,
        collect_features=False,
        calibration_multipliers=calibration_multipliers,
        d2_mix_lambda=d2_mix_lambda,
    )


def _classic_chain_verify_lens(
    *,
    batch_size: int,
    depth_count: int,
    capacity: int,
) -> tuple[int, ...]:


    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if depth_count < 0 or capacity < 0:
        raise ValueError("depth_count and capacity must be non-negative")
    selected = min(capacity, batch_size * depth_count)
    full_depths, partial_requests = divmod(selected, batch_size)
    return tuple(
        1 + full_depths + int(request_slot < partial_requests)
        for request_slot in range(batch_size)
    )


def _cpu_seq_lens(batch: ScheduleBatch) -> torch.Tensor:

    seq_lens_cpu = getattr(batch, "seq_lens_cpu", None)
    if seq_lens_cpu is None:


        seq_lens_cpu = batch.seq_lens.detach().to(device="cpu")
    if not isinstance(seq_lens_cpu, torch.Tensor):
        seq_lens_cpu = torch.as_tensor(seq_lens_cpu, device="cpu")
    return seq_lens_cpu


def _context_token_shape(
    batch: ScheduleBatch,
    seq_lens_cpu: torch.Tensor | None = None,
) -> tuple[int, int]:


    if seq_lens_cpu is None:
        seq_lens_cpu = _cpu_seq_lens(batch)
    if seq_lens_cpu.numel() == 0:
        return 0, 0
    seq_lens_sum = getattr(batch, "seq_lens_sum", None)
    total = (
        int(seq_lens_cpu.sum().item())
        if seq_lens_sum is None
        else int(seq_lens_sum)
    )
    return total, int(seq_lens_cpu.max().item())


def _acceptance_scene_key(
    *, batch_size: int, max_context_tokens: int
) -> tuple[int, int]:


    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if max_context_tokens < 0:
        raise ValueError("max_context_tokens must be non-negative")
    context_bucket = max(1, (max_context_tokens // 1024) * 1024)
    return batch_size, context_bucket


def _tail_round_evidence(
    *,
    arena: TensorForestArena,
    decision: VerificationBudgetDecision,
) -> _TailRoundEvidence | None:


    selected_count = decision.num_candidate_nodes
    if selected_count == 0:
        return None


    request_slots = arena.node_request_slots[:selected_count].tolist()
    path_values = arena.node_values[:selected_count].tolist()
    depths = arena.node_depths[:selected_count].tolist()
    values_by_request: list[list[float]] = [
        [] for _ in range(arena.request_count)
    ]
    depths_by_request: list[list[int]] = [
        [] for _ in range(arena.request_count)
    ]
    for request_slot, path_value, depth in zip(
        request_slots,
        path_values,
        depths,
        strict=True,
    ):
        values_by_request[int(request_slot)].append(float(path_value))
        depths_by_request[int(request_slot)].append(int(depth))
    return _TailRoundEvidence(
        requests=tuple(
            _TailRequestEvidence(
                request_id=request_ref.request_id,
                path_values=tuple(values_by_request[request_slot]),
                depths=tuple(depths_by_request[request_slot]),
            )
            for request_slot, request_ref in enumerate(arena.request_refs)
        )
    )


def _static_budget_decision(
    *,
    selected_node_ids: tuple[int, ...],
    path_values: torch.Tensor,
    num_requests: int,
    total_context_tokens: int,
    max_context_tokens: int,
    max_tree_depth: int,
    ancestor_edges: int,
) -> VerificationBudgetDecision:
    return VerificationBudgetDecision(
        selected_node_ids=selected_node_ids,
        shape=VerifyShape(
            physical_row_bucket=num_requests + len(selected_node_ids),
            execution_mode="eager",
            num_requests=num_requests,
            total_context_tokens=total_context_tokens,
            max_context_tokens=max_context_tokens,
            max_tree_depth=max_tree_depth,
            ancestor_edges=ancestor_edges,
        ),
        expected_progress=float(num_requests) + float(path_values.sum().item()),
        source=BudgetDecisionSource.STATIC_FALLBACK,
    )


def _target_only_budget_decision(
    decision: PreDraftBudgetDecision,
) -> VerificationBudgetDecision:


    return VerificationBudgetDecision(
        selected_node_ids=(),
        shape=decision.target_only_shape,
        expected_progress=float(decision.target_only_shape.num_requests),
        source=decision.target_only_source,
        profiled_shape=decision.target_only_profiled_shape,
        fixed_draft_gpu_ms=0.0,
        estimated_gpu_ms=decision.target_only_total_gpu_ms,
        estimated_total_gpu_ms=decision.target_only_total_gpu_ms,
        goodput=decision.target_only_goodput,
    )


class _RequestForestSelectorCapture:


    def __init__(
        self,
        *,
        draft_model,
        block_size: int,
        max_bs: int,
        device,
        capture_features: bool = False,
        hidden_size: int | None = None,
        hidden_dtype: torch.dtype | None = None,
    ) -> None:
        self.draft_model = draft_model
        self.block_size = int(block_size)
        max_bs = int(max_bs)
        gamma = self.block_size - 1
        top_k = int(draft_model.candidate_selector.top_k)
        self.candidate_out = torch.empty(
            (max_bs, gamma, top_k),
            dtype=torch.int64,
            device=device,
        )
        self.score_out = torch.empty(
            (max_bs, gamma, top_k, top_k),
            dtype=torch.float32,
            device=device,
        )
        if capture_features:
            if hidden_size is None or hidden_size < 1 or hidden_dtype is None:
                raise ValueError(
                    "selector feature capture requires hidden_size and hidden_dtype"
                )
            self.edge_hidden_out = torch.empty(
                (max_bs, gamma, hidden_size),
                dtype=hidden_dtype,
                device=device,
            )
            self.unary_score_out = torch.empty(
                (max_bs, gamma, top_k),
                dtype=torch.float32,
                device=device,
            )
        else:
            self.edge_hidden_out = None
            self.unary_score_out = None

    def __call__(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> None:
        graph_bs = hidden_states.shape[0] // self.block_size
        pred_hidden = hidden_states.view(graph_bs, self.block_size, -1)[:, 1:, :]
        block_ids = input_ids.view(graph_bs, self.block_size)
        selector = self.draft_model.candidate_selector
        lambda_mix = (
            getattr(selector, "lambda_mix_value", None)
            if getattr(selector, "lambda_conditioned", False)
            else None
        )
        lattice = _build_selector_lattice_proposal(
            draft_model=self.draft_model,
            pred_hidden=pred_hidden,
            anchor_token_ids=block_ids[:, 0],
            capture_features=self.edge_hidden_out is not None,
            lambda_mix=lambda_mix,
        )
        self.candidate_out[:graph_bs].copy_(lattice.candidate_ids)
        self.score_out[:graph_bs].copy_(lattice.scores)
        if self.edge_hidden_out is not None:
            self.edge_hidden_out[:graph_bs].copy_(lattice.edge_hidden)
            self.unary_score_out[:graph_bs].copy_(lattice.unary_scores)

    def view(self, batch_size: int) -> _SelectorLatticeView:
        return _SelectorLatticeView(
            candidate_ids=self.candidate_out[:batch_size],
            scores=self.score_out[:batch_size],
            edge_hidden=(
                None
                if self.edge_hidden_out is None
                else self.edge_hidden_out[:batch_size]
            ),
            unary_scores=(
                None
                if self.unary_score_out is None
                else self.unary_score_out[:batch_size]
            ),
        )


class RequestForestWorkerV2(DFlashWorkerV2):


    def __init__(self, *args, **kwargs) -> None:
        load_request_forest_native_ops()
        super().__init__(*args, **kwargs)
        external_policy = getattr(
            self.server_args, "speculative_request_forest_external_policy", "rf"
        )
        if external_policy not in ("rf", "bastion", "echo"):
            raise ValueError("external policy must be rf, bastion or echo")
        external_profile = getattr(
            self.server_args, "speculative_request_forest_external_profile_path", None
        )
        if external_policy != "rf" and not external_profile:
            raise ValueError("external policy requires a calibrated profile path")
        self._external_policy = (
            None if external_policy == "rf"
            else ExternalPolicyAdapter(external_policy, external_profile)
        )
        if external_policy == "echo" and self.selector is None:
            raise ValueError("ECHO-style requires a DFlash2 conditional lattice selector")
        max_running = int(
            get_schedule().max_running_requests
            or getattr(self.server_args, "max_running_requests", 0)
            or 1
        )
        max_nodes = max(1, max_running * max(1, int(self.block_size) - 1))
        if self._external_policy is not None:
            max_nodes = max_running * self._external_policy.per_request_cap
        self._forest_lowerer = _make_forest_lowerer(
            selector=self.selector,
            max_nodes=max_nodes,
            max_nodes_per_request=(
                max(1, int(self.block_size) - 1)


                if self.selector is not None
                and mambaish_config(self.model_runner.model_config) is not None
                else None
            ),
            selection_mode=str(
                getattr(
                    self.server_args,
                    "speculative_request_forest_selection_mode",
                    "global-forest",
                )
            ),
        )
        configured_verify_nodes = getattr(
            self.server_args,
            "speculative_request_forest_max_verify_nodes",
            None,
        )
        self._forest_max_verify_nodes = (
            None
            if configured_verify_nodes is None
            else int(configured_verify_nodes)
        )
        self._forest_d2_mix_lambda = float(
            getattr(
                self.server_args,
                "speculative_request_forest_d2_mix_lambda",
                0.0,
            )
        )
        if self._external_policy is not None:
            self._forest_d2_mix_lambda = 0.0
        self._forest_base_d2_mix_lambda = self._forest_d2_mix_lambda
        crossover_profile_path = getattr(
            self.server_args,
            "speculative_request_forest_crossover_profile_path",
            None,
        )
        self._forest_crossover_profile = (
            load_crossover_profile(crossover_profile_path)
            if self._external_policy is None else None
        )
        self._forest_profile_verify_nodes: int | None = None


        static_max_verify_nodes = (
            max_nodes
            if self._forest_max_verify_nodes is None
            else min(max_nodes, self._forest_max_verify_nodes)
        )
        profile_path = getattr(
            self.server_args,
            "speculative_request_forest_target_profile_path",
            None,
        )
        profile_db = (
            None
            if profile_path is None
            else VerifyShapeProfileDB.load(profile_path)
        )
        collect_target_timings = bool(
            getattr(
                self.server_args,
                "speculative_request_forest_collect_target_timings",
                False,
            )
        )
        configured_profile_rows = getattr(
            self.server_args,
            "speculative_request_forest_target_profile_rows",
            None,
        )
        target_runner = self.target_worker.model_runner
        if bool(getattr(target_runner.model_config, "model_is_mrope", False)):
            raise NotImplementedError(
                "Request Forest does not yet support MRoPE Target models; "
                "ragged Forest rows require request-slot-aware MRoPE positions"
            )
        tp_group = target_runner.tp_group
        branch_label_dir = getattr(
            self.server_args,
            "speculative_request_forest_branch_label_dir",
            None,
        )
        self._forest_branch_label_collector = (
            BranchLabelCollector(
                branch_label_dir,
                max_parents=int(
                    self.server_args.speculative_request_forest_branch_label_max_parents
                ),
                seed=int(self.server_args.random_seed or 0),
            )
            if branch_label_dir is not None and int(tp_group.rank_in_group) == 0
            else None
        )
        self._forest_rollout_logging_enabled = bool(
            int(tp_group.rank_in_group) == 0
            and (
                collect_target_timings
                or self._forest_branch_label_collector is not None
            )
        )


        self._forest_round_sequence = (
            time.time_ns() if self._forest_rollout_logging_enabled else 0
        )
        runtime_fingerprint = None
        if profile_db is not None or collect_target_timings:
            model_config = target_runner.model_config
            if profile_db is not None:
                physical_row_grid = profile_db.fingerprint.physical_row_grid
            elif configured_profile_rows is not None:
                physical_row_grid = tuple(configured_profile_rows)
            else:


                physical_row_grid = (
                    max_running + static_max_verify_nodes,
                )
            runtime_fingerprint = build_runtime_fingerprint(
                model_id=model_config.model_path,
                model_revision=model_config.revision,
                dtype=target_runner.dtype,
                quantization=model_config.quantization,
                device_name=torch.cuda.get_device_name(self.device),


                attention_backend=(
                    getattr(target_runner, "decode_attention_backend_str", None)
                    or getattr(self.server_args, "decode_attention_backend", None)
                    or self.server_args.attention_backend
                ),
                tp_size=int(tp_group.world_size),
                physical_row_grid=physical_row_grid,
            )
        profile_matches = (
            profile_db is not None
            and runtime_fingerprint == profile_db.fingerprint
        )
        configured_correction_alpha = getattr(
            self.server_args,
            "speculative_request_forest_acceptance_correction_alpha",
            None,
        )
        self._forest_acceptance_correction_fixed_alpha = (
            None
            if configured_correction_alpha is None
            else float(configured_correction_alpha)
        )
        self._forest_acceptance_correction_table: (
            AcceptanceCorrectionTable | None
        ) = (
            profile_db.acceptance_correction_table
            if profile_db is not None
            else None
        )
        self._forest_last_acceptance_correction_alpha = 0.0
        feedback_enabled = (
            self._forest_acceptance_correction_fixed_alpha > 0.0
            if self._forest_acceptance_correction_fixed_alpha is not None
            else (
                self._forest_acceptance_correction_table is not None
                and self._forest_acceptance_correction_table.has_positive_alpha
            )
        )
        target_ema_alpha = float(
            getattr(
                self.server_args,
                "speculative_request_forest_target_ema_alpha",
                0.2,
            )
        )
        self._forest_budget_controller = RequestForestBudgetController(
            static_max_candidate_nodes=static_max_verify_nodes,
            eager_row_budgets=(
                ()
                if profile_db is None
                else profile_db.fingerprint.physical_row_grid
            ),
            profile_db=profile_db,
            runtime_fingerprint=runtime_fingerprint,
            enabled=profile_matches and self._external_policy is None,
            online_ema_alpha=target_ema_alpha,
            fixed_draft_cost_ms=float(
                getattr(
                    self.server_args,
                    "speculative_request_forest_draft_cost_ms",
                    0.0,
                )
            ),
        )
        online_timing = bool(
            getattr(
                self.server_args,
                "speculative_request_forest_online_target_timing",
                True,
            )
        )
        self._forest_collect_target_timings = collect_target_timings


        self._forest_online_target_timing_enabled = (
            profile_matches
            and self._external_policy is None
            and online_timing
            and int(tp_group.world_size) == 1
        )
        self._forest_target_timer = (
            AsyncTargetVerificationTimer(ema_alpha=target_ema_alpha)
            if (
                collect_target_timings
                or self._forest_online_target_timing_enabled
            )
            else None
        )
        self._forest_last_timing_sequence = 0
        self._forest_draft_timer = (
            AsyncTargetVerificationTimer(ema_alpha=target_ema_alpha)
            if collect_target_timings
            else None
        )
        self._forest_last_draft_timing_sequence = 0
        self._forest_draft_model_id = str(
            getattr(
                self.draft_model_runner.model_config,
                "model_path",
                getattr(
                    self.server_args,
                    "speculative_draft_model_path",
                    self.draft_model.__class__.__name__,
                ),
            )
        )
        self._forest_arenas: dict[tuple[int, int], TensorForestArena] = {}
        self._classic_verify_templates: (
            dict[tuple[int, int], ClassicChainVerifyTemplate] | None
        ) = {} if self.selector is None and self._external_policy is None else None
        self._forest_selected_ids = torch.arange(
            max_nodes,
            dtype=torch.int64,
            device="cpu",
        )
        self._forest_request_rows = torch.arange(
            max_running,
            dtype=torch.int64,
            device=self.device,
        )
        self._forest_tree_attention_workspace = ForestTreeAttentionWorkspace()
        self._forest_tail_evidence: _TailRoundEvidence | None = None
        feedback_rank = (
            self.selector is not None
            and int(self.block_size) > 1
            and int(tp_group.rank_in_group) == 0
            and feedback_enabled
            and self._external_policy is None
        )
        self._forest_acceptance_calibrators: (
            dict[tuple[int, int], DepthConfidenceBetaEMA] | None
        ) = {} if feedback_rank else None
        self._forest_acceptance_alpha_by_scene: dict[
            tuple[int, int], float
        ] = {}
        self._forest_feedback_host = (
            torch.empty(
                max_running + static_max_verify_nodes,
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            )
            if feedback_rank
            else None
        )
        self._forest_feedback_copy_stream = (
            torch.cuda.Stream(device=self.device) if feedback_rank else None
        )
        self._forest_feedback_source_event = (
            torch.cuda.Event(blocking=False, enable_timing=False)
            if feedback_rank
            else None
        )
        self._forest_feedback_done_event = (
            torch.cuda.Event(blocking=False, enable_timing=False)
            if feedback_rank
            else None
        )
        self._forest_pending_acceptance_feedback: (
            _PendingAcceptanceFeedback | None
        ) = None

    def _maybe_build_draft_sampler(self):


        if getattr(
            self.server_args, "speculative_request_forest_external_policy", "rf"
        ) == "bastion":
            return None
        sampler = super()._maybe_build_draft_sampler()
        if sampler is None or self.selector is None:
            return sampler
        server_args = self.__dict__.get("server_args")
        branch_label_dir = (
            None
            if server_args is None
            else getattr(
                server_args,
                "speculative_request_forest_branch_label_dir",
                None,
            )
        )
        capture_features = False
        if branch_label_dir is not None:
            target_worker = self.__dict__.get("_target_worker")
            capture_features = bool(
                target_worker is not None
                and int(target_worker.model_runner.tp_group.rank_in_group) == 0
            )
        hidden_size = hidden_dtype = None
        if capture_features:
            hidden_size = int(
                self.draft_model_runner.model_config.hf_config.hidden_size
            )
            hidden_dtype = self.draft_model_runner.dtype
        return _RequestForestSelectorCapture(
            draft_model=self.draft_model,
            block_size=self.block_size,
            max_bs=int(sampler.candidate_out.shape[0]),
            device=sampler.candidate_out.device,
            capture_features=capture_features,
            hidden_size=hidden_size,
            hidden_dtype=hidden_dtype,
        )

    def _verify_node_capacity(self, batch_size: int) -> int:
        generated_nodes = batch_size * max(0, int(self.block_size) - 1)


        state = self.__dict__
        external = state.get("_external_policy")
        if external is not None:
            generated_nodes = batch_size * external.per_request_cap
            upper = state.get("_forest_max_verify_nodes")
            return generated_nodes if upper is None else min(generated_nodes, upper)
        profiled_nodes = state.get("_forest_profile_verify_nodes")
        if profiled_nodes is not None:
            return min(generated_nodes, profiled_nodes)
        max_verify_nodes = state.get("_forest_max_verify_nodes")
        if max_verify_nodes is None:
            return generated_nodes
        return min(generated_nodes, max_verify_nodes)

    def _update_crossover_policy(
        self, *, batch_size: int, max_context_tokens: int
    ) -> None:


        state = self.__dict__
        if state.get("_external_policy") is not None:
            self._forest_d2_mix_lambda = 0.0
            self._forest_profile_verify_nodes = None
            return
        profile = state.get("_forest_crossover_profile")
        if profile is None:


            self._forest_d2_mix_lambda = state.get(
                "_forest_base_d2_mix_lambda", 0.0
            )
            self._forest_profile_verify_nodes = None
            return
        generated_nodes = batch_size * max(0, int(self.block_size) - 1)
        decision = profile.resolve(
            batch_size,
            generated_nodes=generated_nodes,
            configured_max_nodes=state.get("_forest_max_verify_nodes"),
        )
        self._forest_d2_mix_lambda = decision.lambda_mix
        self._forest_profile_verify_nodes = decision.verify_nodes

    def _resolve_verify_execution_shape(
        self,
        logical_rows: int,
        *,
        num_requests: int,
        max_tree_depth: int,
    ) -> tuple[int, str]:


        target_worker = getattr(self, "_target_worker", None)
        target_model_runner = getattr(target_worker, "model_runner", None)
        graph_runner = getattr(
            target_model_runner,
            "decode_cuda_graph_runner",
            None,
        )
        resolve_shape = getattr(
            graph_runner, "resolve_request_forest_graph_shape", None
        )
        if callable(resolve_shape):
            physical_rows = resolve_shape(
                logical_rows, num_requests, max_tree_depth
            )


            if physical_rows is None:
                return logical_rows, "eager"
            return int(physical_rows), "cuda_graph"

        tiers = getattr(graph_runner, "capture_num_tokens", None)
        if (
            graph_runner is None
            or not bool(getattr(graph_runner, "ragged_verify_mode", False))
            or not tiers
            or max_tree_depth + 1
            > int(getattr(graph_runner, "captured_req_width", 0))
        ):
            return logical_rows, "eager"
        for tier in tiers:
            tier = int(tier)
            if tier < logical_rows:
                continue
            capture_slots_fn = getattr(
                graph_runner,
                "_ragged_capture_slots",
                None,
            )
            capture_slots = (
                int(capture_slots_fn(tier))
                if callable(capture_slots_fn)
                else min(tier, int(getattr(graph_runner, "max_bs", 0)))
            )
            if num_requests <= capture_slots:
                return tier, "cuda_graph"
        return logical_rows, "eager"

    def _local_pre_draft_decision(
        self,
        batch: ScheduleBatch,
        *,
        node_capacity: int,
        context_shape: tuple[int, int],
    ) -> PreDraftBudgetDecision | None:


        if (
            self.__dict__.get("_external_policy") is not None
            or self.selector is None
            or node_capacity == 0
        ):
            return None
        evidence = getattr(self, "_forest_tail_evidence", None)
        if not isinstance(evidence, _TailRoundEvidence):
            return None
        evidence_by_id = {
            request.request_id: request for request in evidence.requests
        }
        request_ids = tuple(req.rid for req in batch.reqs)
        if len(set(request_ids)) != len(request_ids):
            return None
        if any(request_id not in evidence_by_id for request_id in request_ids):
            return None

        path_values: list[float] = []
        depths: list[int] = []
        for request_id in request_ids:
            request_evidence = evidence_by_id[request_id]
            if len(request_evidence.path_values) != len(request_evidence.depths):
                return None
            path_values.extend(request_evidence.path_values)
            depths.extend(request_evidence.depths)
        if len(path_values) > node_capacity:
            return None

        num_requests = len(request_ids)
        total_context_tokens, max_context_tokens = context_shape
        target_only_rows, target_only_mode = (
            self._resolve_verify_execution_shape(
                num_requests,
                num_requests=num_requests,
                max_tree_depth=0,
            )
        )
        request_forest_logical_rows = num_requests + len(path_values)
        request_forest_rows, request_forest_mode = (
            self._resolve_verify_execution_shape(
                request_forest_logical_rows,
                num_requests=num_requests,
                max_tree_depth=max(depths, default=0),
            )
        )
        target_only_shape = VerifyShape(
            physical_row_bucket=target_only_rows,
            execution_mode=target_only_mode,
            num_requests=num_requests,
            total_context_tokens=total_context_tokens,
            max_context_tokens=max_context_tokens,
            max_tree_depth=0,
            ancestor_edges=num_requests,
        )
        request_forest_shape = VerifyShape(
            physical_row_bucket=request_forest_rows,
            execution_mode=request_forest_mode,
            num_requests=num_requests,
            total_context_tokens=total_context_tokens,
            max_context_tokens=max_context_tokens,
            max_tree_depth=max(depths, default=0),
            ancestor_edges=(
                num_requests + sum(depth + 1 for depth in depths)
            ),
        )
        return self._forest_budget_controller.choose_pre_draft_mode(
            target_only_shape=target_only_shape,
            request_forest_shape=request_forest_shape,
            request_forest_path_values=tuple(path_values),
        )

    def _choose_pre_draft_decision(
        self,
        batch: ScheduleBatch,
        *,
        node_capacity: int,
        context_shape: tuple[int, int],
    ) -> tuple[bool, PreDraftBudgetDecision | None]:


        if (
            self.__dict__.get("_external_policy") is not None
            or self.selector is None
            or node_capacity == 0
        ):
            return False, None
        tp_group = self.target_worker.model_runner.tp_group
        if int(tp_group.world_size) == 1:
            decision = self._local_pre_draft_decision(
                batch,
                node_capacity=node_capacity,
                context_shape=context_shape,
            )
            return bool(decision and decision.use_target_only), decision

        state = None
        local_decision = None
        local_error: Exception | None = None
        if int(tp_group.rank_in_group) == 0:
            try:
                local_decision = self._local_pre_draft_decision(
                    batch,
                    node_capacity=node_capacity,
                    context_shape=context_shape,
                )
                state = int(
                    bool(local_decision and local_decision.use_target_only)
                )
            except Exception as exc:


                local_error = exc
                state = 2
        state = tp_group.broadcast_object(state, src=0)
        if state == 2:
            if local_error is not None:
                raise local_error
            raise RuntimeError(
                "Request Forest pre-Draft decision failed on TP rank zero"
            )
        if state not in (0, 1):
            raise RuntimeError(
                "Request Forest TP payload has invalid pre-Draft control state"
            )
        return bool(state), local_decision

    def _choose_budget_decision(
        self,
        *,
        arena: TensorForestArena,
        allow_dynamic: bool,
        context_shape: tuple[int, int],
    ) -> VerificationBudgetDecision:
        total_context_tokens, max_context_tokens = context_shape
        if allow_dynamic:
            return self._forest_budget_controller.choose(
                path_values=arena.node_values,
                num_requests=arena.request_count,
                total_context_tokens=total_context_tokens,
                max_context_tokens=max_context_tokens,
                node_depths=arena.node_depths,
                execution_shape_resolver=lambda logical_rows, prefix_depth: (
                    self._resolve_verify_execution_shape(
                        logical_rows,
                        num_requests=arena.request_count,
                        max_tree_depth=prefix_depth,
                    )
                ),
                _trusted_internal=True,
            )
        selected_node_ids = tuple(range(arena.node_count))
        if arena.node_count == 0:
            max_tree_depth = 0
            ancestor_edges = arena.request_count
        else:
            depths = arena.node_depths
            max_tree_depth = int(depths.max().item())
            ancestor_edges = (
                arena.request_count
                + int(depths.sum().item())
                + arena.node_count
            )
        return _static_budget_decision(
            selected_node_ids=selected_node_ids,
            path_values=arena.node_values,
            num_requests=arena.request_count,
            total_context_tokens=total_context_tokens,
            max_context_tokens=max_context_tokens,
            max_tree_depth=max_tree_depth,
            ancestor_edges=ancestor_edges,
        )

    def _ingest_completed_target_timings(self) -> None:


        timer = getattr(self, "_forest_target_timer", None)
        if timer is None:
            return
        tp_group = self.target_worker.model_runner.tp_group
        timer.poll()
        last_sequence = self._forest_last_timing_sequence
        for observation in timer.recent_observations():
            if observation.sequence <= last_sequence:
                continue
            if self._forest_collect_target_timings:
                logger.info(
                    "request_forest_target_timing %s",
                    json.dumps(
                        {
                            "sequence": observation.sequence,
                            "round_id": observation.round_id,
                            "tp_rank": int(tp_group.rank_in_group),
                            "fingerprint": observation.key.fingerprint.to_dict(),
                            "actual_shape": observation.key.shape.to_dict(),
                            "gpu_ms": observation.gpu_ms,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
            if (
                self._forest_online_target_timing_enabled
                and int(tp_group.rank_in_group) == 0
            ):
                controller = self._forest_budget_controller
                estimate = controller.profile_db.estimate(
                    observation.key.shape,
                    runtime_fingerprint=controller.runtime_fingerprint,
                )
                if estimate is not None:
                    controller.ingest_observation(
                        TargetTimingObservation(
                            sequence=observation.sequence,
                            key=TargetTimingKey(
                                fingerprint=observation.key.fingerprint,
                                shape=estimate.profiled_shape,
                                scope=observation.key.scope,
                            ),
                            gpu_ms=observation.gpu_ms,
                            round_id=observation.round_id,
                        )
                    )
            last_sequence = observation.sequence
        self._forest_last_timing_sequence = last_sequence

    def _begin_target_timing(
        self,
        decision: VerificationBudgetDecision,
        *,
        round_id: int,
    ):
        timer = getattr(self, "_forest_target_timer", None)
        controller = getattr(self, "_forest_budget_controller", None)
        if (
            timer is None
            or controller is None
            or controller.runtime_fingerprint is None
        ):
            return None
        return timer.begin(
            TargetTimingKey(
                fingerprint=controller.runtime_fingerprint,
                shape=decision.shape,
                scope=TargetTimingScope.VERIFICATION_STAGE,
            ),
            round_id=round_id,
        )

    def _begin_draft_timing(
        self,
        batch: ScheduleBatch,
        *,
        context_shape: tuple[int, int],
        round_id: int,
    ):
        timer = getattr(self, "_forest_draft_timer", None)
        controller = getattr(self, "_forest_budget_controller", None)
        if (
            timer is None
            or controller is None
            or controller.runtime_fingerprint is None
        ):
            return None
        batch_size = len(batch.reqs)
        total_context_tokens, max_context_tokens = context_shape
        return timer.begin(
            TargetTimingKey(
                fingerprint=controller.runtime_fingerprint,
                shape=VerifyShape(
                    physical_row_bucket=batch_size,
                    execution_mode="draft",
                    num_requests=batch_size,
                    total_context_tokens=total_context_tokens,
                    max_context_tokens=max_context_tokens,
                    max_tree_depth=max(0, int(self.block_size) - 1),
                ),
                scope=TargetTimingScope.DRAFT_STAGE,
            ),
            round_id=round_id,
        )

    def _timed_draft_proposal(
        self,
        batch: ScheduleBatch,
        draft_input: RequestForestDraftInput,
        *,
        context_shape: tuple[int, int],
        round_id: int,
    ):


        timing_handle = self._begin_draft_timing(
            batch,
            context_shape=context_shape,
            round_id=round_id,
        )
        try:
            return self._draft_proposal(batch, draft_input)
        finally:
            timer = getattr(self, "_forest_draft_timer", None)
            if timer is not None and timing_handle is not None:
                try:
                    timer.end(timing_handle)
                except Exception:


                    logger.warning(
                        "Failed to finish Request Forest Draft timing",
                        exc_info=True,
                    )

    def _ingest_completed_draft_timings(self) -> None:


        timer = getattr(self, "_forest_draft_timer", None)
        if timer is None:
            return
        tp_group = self.target_worker.model_runner.tp_group
        timer.poll()
        last_sequence = self._forest_last_draft_timing_sequence
        for observation in timer.recent_observations():
            if observation.sequence <= last_sequence:
                continue
            if observation.key.scope == TargetTimingScope.DRAFT_STAGE:
                shape = observation.key.shape
                logger.info(
                    "request_forest_draft_timing %s",
                    json.dumps(
                        {
                            "request_forest_draft_timing": {
                                "sequence": observation.sequence,
                                "round_id": observation.round_id,
                                "tp_rank": int(tp_group.rank_in_group),
                                "draft_model": self._forest_draft_model_id,
                                "batch_size": shape.num_requests,
                                "block_size": int(self.block_size),
                                "context_shape": {
                                    "total_context_tokens": (
                                        shape.total_context_tokens
                                    ),
                                    "max_context_tokens": (
                                        shape.max_context_tokens
                                    ),
                                },
                                "gpu_ms": observation.gpu_ms,
                            }
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
            last_sequence = observation.sequence
        self._forest_last_draft_timing_sequence = last_sequence

    def _acceptance_calibrator_for(
        self,
        *,
        batch_size: int,
        max_context_tokens: int,
    ) -> DepthConfidenceBetaEMA | None:


        registry = getattr(self, "_forest_acceptance_calibrators", None)
        if registry is None:
            self._forest_last_acceptance_correction_alpha = 0.0
            return None

        key = _acceptance_scene_key(
            batch_size=batch_size, max_context_tokens=max_context_tokens
        )
        fixed_alpha = self._forest_acceptance_correction_fixed_alpha
        if fixed_alpha is not None:
            alpha = fixed_alpha
        else:
            alpha_by_scene = self._forest_acceptance_alpha_by_scene
            if key in alpha_by_scene:
                alpha = alpha_by_scene[key]
            else:
                table = self._forest_acceptance_correction_table
                alpha = (
                    0.0
                    if table is None
                    else table.estimate(
                        batch_size=key[0],
                        context_length=key[1],
                    )
                )
                alpha_by_scene[key] = alpha
        self._forest_last_acceptance_correction_alpha = alpha
        if alpha <= 0.0:
            return None

        calibrator = registry.get(key)
        if calibrator is None:
            calibrator = DepthConfidenceBetaEMA(
                depth_count=int(self.block_size) - 1,
                correction_strength=alpha,
            )
            registry[key] = calibrator
        return calibrator

    def _poll_acceptance_feedback(self) -> None:


        pending = getattr(self, "_forest_pending_acceptance_feedback", None)
        done_event = getattr(self, "_forest_feedback_done_event", None)
        if (
            pending is None
            or done_event is None
            or not done_event.query()
        ):
            return
        try:
            pending.calibrator.ingest(
                pending.metadata,
                self._forest_feedback_host,
                eos_token_ids=pending.eos_token_ids,
                request_ignore_eos=pending.request_ignore_eos,
            )
        except Exception:
            logger.exception("dropping invalid Request Forest acceptance feedback")
        finally:
            self._forest_pending_acceptance_feedback = None

    def _stage_acceptance_feedback(
        self,
        metadata: AcceptanceFeedbackMetadata | None,
        calibrator: DepthConfidenceBetaEMA | None,
        target_predict: torch.Tensor,
        *,
        eos_token_ids: frozenset[int],
        request_ignore_eos: tuple[bool, ...],
    ) -> None:


        if (
            metadata is None
            or calibrator is None
            or getattr(self, "_forest_pending_acceptance_feedback", None)
            is not None
        ):
            return
        copy_stream = self._forest_feedback_copy_stream
        source_event = self._forest_feedback_source_event
        done_event = self._forest_feedback_done_event
        host = self._forest_feedback_host
        try:
            producer = torch.cuda.current_stream(target_predict.device)
            source_event.record(producer)
            copy_stream.wait_event(source_event)
            with torch.cuda.stream(copy_stream):
                host[: metadata.logical_rows].copy_(
                    target_predict[: metadata.logical_rows],
                    non_blocking=True,
                )
                done_event.record(copy_stream)
            self._forest_pending_acceptance_feedback = (
                _PendingAcceptanceFeedback(
                    metadata=metadata,
                    calibrator=calibrator,
                    eos_token_ids=eos_token_ids,
                    request_ignore_eos=request_ignore_eos,
                    source=target_predict,
                )
            )
        except Exception:


            logger.exception("dropping Request Forest acceptance feedback copy")

    def _log_budget_decision(
        self,
        decision: VerificationBudgetDecision,
        *,
        round_id: int,
        request_ids: tuple[str, ...],
    ) -> None:


        if not getattr(self, "_forest_rollout_logging_enabled", False):
            return
        tp_group = self.target_worker.model_runner.tp_group
        if int(tp_group.rank_in_group) != 0:
            return
        logger.info(
            "request_forest_budget_decision %s",
            json.dumps(
                {
                    "round_id": round_id,
                    "request_ids": request_ids,
                    "selected_node_ids": decision.selected_node_ids,
                    "candidate_nodes": decision.num_candidate_nodes,
                    "logical_rows": decision.logical_num_tokens,
                    "physical_rows": decision.physical_row_bucket,
                    "padding_rows": (
                        decision.physical_row_bucket
                        - decision.logical_num_tokens
                    ),
                    "source": decision.source.value,
                    "expected_progress": decision.expected_progress,
                    "d2_mix_lambda": self._forest_d2_mix_lambda,
                    "acceptance_correction_alpha": getattr(
                        self, "_forest_last_acceptance_correction_alpha", 0.0
                    ),
                    "profiled_shape": (
                        None
                        if decision.profiled_shape is None
                        else decision.profiled_shape.to_dict()
                    ),
                    "target_gpu_ms": decision.estimated_gpu_ms,
                    "fixed_draft_gpu_ms": decision.fixed_draft_gpu_ms,
                    "total_gpu_ms": decision.estimated_total_gpu_ms,
                    "goodput": decision.goodput,
                    "marginal_target_gpu_ms": decision.marginal_gpu_ms,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

    def _log_round_outcome(
        self,
        *,
        outcome: BranchRoundOutcome,
        decision: VerificationBudgetDecision,
        request_ids: tuple[str, ...],
    ) -> None:


        if not getattr(self, "_forest_rollout_logging_enabled", False):
            return
        tp_group = self.target_worker.model_runner.tp_group
        if int(tp_group.rank_in_group) != 0:
            return
        requests = tuple(
            {
                "request_id": request_ids[index],
                "accept_length": outcome.accept_lens[index],
                "accepted_draft_length": outcome.accepted_draft_lens[index],
                "accepted_token_ids": outcome.accepted_token_ids[index],
                "committed_token_ids": outcome.committed_token_ids[index],
                "first_reject_position": outcome.first_miss_positions[index],
                "termination_reason": outcome.termination_reasons[index],
            }
            for index in range(outcome.request_count)
        )
        logger.info(
            "request_forest_round_outcome %s",
            json.dumps(
                {
                    "round_id": outcome.round_id,
                    "request_count": outcome.request_count,
                    "emitted_committed_tokens": outcome.committed_token_count,
                    "accepted_draft_tokens": (
                        outcome.accepted_draft_token_count
                    ),
                    "candidate_nodes": decision.num_candidate_nodes,
                    "logical_rows": decision.logical_num_tokens,
                    "physical_rows": decision.physical_row_bucket,
                    "requests": requests,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

    def _next_rollout_round_id(self) -> int:
        self._forest_round_sequence = (
            self.__dict__.get("_forest_round_sequence", 0) + 1
        )
        return self._forest_round_sequence

    def _forest_arena(
        self,
        *,
        request_refs: tuple[RequestRef, ...],
        anchor_token_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        node_capacity: int,
    ) -> TensorForestArena:
        key = (len(request_refs), node_capacity)
        arena = self._forest_arenas.get(key)
        if arena is None:
            arena = TensorForestArena.empty(
                request_refs=request_refs,
                anchor_token_ids=anchor_token_ids,
                anchor_positions=anchor_positions,
                node_capacity=node_capacity,
            )
            self._forest_arenas[key] = arena
        else:
            arena.reset_round(
                request_refs=request_refs,
                anchor_token_ids=anchor_token_ids,
                anchor_positions=anchor_positions,
            )
        return arena

    def _make_next_draft_input_prefill(
        self,
        *,
        bonus_tokens: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> RequestForestDraftInput:
        return RequestForestDraftInput(
            bonus_tokens=bonus_tokens.to(dtype=torch.int64),
            new_seq_lens=seq_lens.to(dtype=torch.int64),
        )

    def _make_next_draft_input_decode(
        self,
        *,
        bonus_tokens: torch.Tensor,
        new_seq_lens: torch.Tensor,
    ) -> RequestForestDraftInput:
        return RequestForestDraftInput(
            bonus_tokens=bonus_tokens.to(dtype=torch.int64),
            new_seq_lens=new_seq_lens.to(dtype=torch.int64),
        )

    @contextmanager
    def _temporary_proposal_kv(
        self,
        *,
        batch: ScheduleBatch,
        token_positions: torch.Tensor,
    ) -> Iterator[torch.Tensor]:


        batch_size, block_size = token_positions.shape
        flat_slots = alloc_token_slots(batch.tree_cache, batch_size * block_size)
        allocator = batch.tree_cache.token_to_kv_pool_allocator
        target_table = self.model_runner.req_to_token_pool.req_to_token
        request_rows = batch.req_pool_indices.to(torch.int64).view(-1, 1).expand_as(
            token_positions
        )
        old_locations = None
        try:
            old_locations = target_table[
                request_rows,
                token_positions.to(torch.int64),
            ]
            target_table[
                request_rows,
                token_positions.to(torch.int64),
            ] = flat_slots.view(batch_size, block_size).to(target_table.dtype)
            yield flat_slots.view(batch_size, block_size)
        finally:
            if old_locations is not None:
                target_table[
                    request_rows,
                    token_positions.to(torch.int64),
                ] = old_locations
            allocator.free(flat_slots)

    def _draft_proposal(
        self,
        batch: ScheduleBatch,
        draft_input: RequestForestDraftInput,
    ):


        batch_size = len(batch.seq_lens)
        block_size = int(self.block_size)
        self._ensure_draft_block_buffers(batch_size)
        assert self._draft_block_ids_buf is not None
        assert self._draft_block_positions_buf is not None
        assert self._draft_verify_out_cache_loc_buf is not None
        assert self._draft_seq_lens_cpu_buf is not None

        block_ids = self._draft_block_ids_buf[:batch_size]
        positions_2d = self._draft_block_positions_buf[:batch_size]
        proposal_locs_2d = self._draft_verify_out_cache_loc_buf[:batch_size]
        prefix_lens = batch.seq_lens

        block_ids.fill_(int(self._mask_token_id))
        block_ids[:, 0].copy_(draft_input.bonus_tokens.view(-1))
        torch.add(
            prefix_lens.unsqueeze(1),
            self._block_pos_offsets,
            out=positions_2d,
        )

        target_model = self.target_worker.model_runner.model
        embed_module = unwrap_lora_layer(
            _resolve_dflash_embedding_module(self.draft_model, target_model)
        )
        lm_head = unwrap_lora_layer(getattr(target_model, "lm_head", None))
        if lm_head is None:
            raise RuntimeError("Request Forest requires the Target lm_head")
        weight = getattr(lm_head, "weight", None)
        quant_method = getattr(lm_head, "quant_method", None)
        if not (
            is_dense_head_weight(weight)
            or should_apply_lm_head_quant_method(lm_head, quant_method)
        ):
            raise RuntimeError(
                "Request Forest's DFlash drafter requires a dense Target "
                "lm_head or a supported lm_head quantization method."
            )
        noise_embedding = embed_module(block_ids)
        if self._noise_embed_scale != 1.0:
            noise_embedding = noise_embedding * self._noise_embed_scale
        input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])

        with self._temporary_proposal_kv(
            batch=batch,
            token_positions=positions_2d,
        ) as proposal_locs:
            proposal_locs_2d.copy_(proposal_locs)
            seq_lens_cpu = self._draft_seq_lens_cpu_buf[:batch_size]
            if self.use_compact_draft_cache:
                draft_prefix_lens = self._compute_compact_draft_seq_lens(prefix_lens)
                self._fill_compact_seq_lens_cpu_bound(
                    batch_seq_lens_cpu=batch.seq_lens_cpu,
                    nxt_kv_lens_cpu=None,
                    draft_prefix_lens=draft_prefix_lens,
                    out=seq_lens_cpu,
                )
                self._rebuild_compact_draft_cache(
                    req_pool_indices=batch.req_pool_indices,
                    prefix_lens=prefix_lens,
                    draft_prefix_lens=draft_prefix_lens,
                    verify_out_cache_loc_2d=proposal_locs_2d,
                    bs=batch_size,
                    block_size=block_size,
                )
                draft_seq_lens = draft_prefix_lens
                draft_seq_lens_sum = int(seq_lens_cpu.sum().item())
            else:
                draft_seq_lens = prefix_lens
                if batch.seq_lens_cpu is not None:
                    seq_lens_cpu.copy_(batch.seq_lens_cpu)
                    seq_lens_cpu.add_(block_size)
                    draft_seq_lens_sum = int(seq_lens_cpu.sum())
                else:
                    seq_lens_cpu.copy_(prefix_lens.to("cpu", dtype=torch.int32))
                    seq_lens_cpu.add_(block_size)
                    draft_seq_lens_sum = int(seq_lens_cpu.sum())

            forward_batch = ForwardBatch(
                forward_mode=ForwardMode.TARGET_VERIFY,
                batch_size=batch_size,
                input_ids=block_ids.reshape(-1),
                req_pool_indices=batch.req_pool_indices,
                seq_lens=draft_seq_lens,
                out_cache_loc=proposal_locs_2d.reshape(-1),
                seq_lens_sum=draft_seq_lens_sum,
                seq_lens_cpu=seq_lens_cpu,
                positions=positions_2d.reshape(-1),
                input_embeds=input_embeds,
                spec_algorithm=SpeculativeAlgorithm.REQUEST_FOREST,
                spec_info=self._draft_block_spec_info,
                capture_hidden_mode=CaptureHiddenMode.NULL,
            )
            selector = self.selector
            if selector is not None and getattr(selector, "lambda_conditioned", False):


                selector.set_lambda_mix(self._forest_d2_mix_lambda)

            with torch.inference_mode():
                draft_output = self.draft_model_runner.forward(forward_batch)

            return self._proposal_from_draft_output(
                draft_output=draft_output,
                batch_size=batch_size,
                block_ids=block_ids,
                lm_head=lm_head,
            )

    def _proposal_from_draft_output(
        self,
        *,
        draft_output,
        batch_size: int,
        block_ids: torch.Tensor,
        lm_head,
    ):


        sampler = self._draft_sampler
        graph_hit = sampler is not None and bool(draft_output.can_run_graph)
        if graph_hit:
            if self.selector is None:
                token_count = batch_size * (int(self.block_size) - 1)
                return sampler.out[:token_count].view(batch_size, -1)
            if isinstance(sampler, _RequestForestSelectorCapture):
                return sampler.view(batch_size)

        hidden = draft_output.logits_output.hidden_states
        if hidden is None:
            raise RuntimeError(
                "Request Forest's DFlash draft returned no hidden states"
            )
        hidden = hidden.view(batch_size, int(self.block_size), -1)
        pred_hidden = hidden[:, 1:, :]
        external = self.__dict__.get("_external_policy")
        if external is not None and external.policy == "bastion":
            return build_marginal_proposal(
                self.draft_model, pred_hidden, lm_head,
                k=int(external.profile.get("candidate_top_k", 128)),
            )
        if self.selector is None:
            return self._greedy_sample_from_vocab_parallel_head(
                hidden_states=pred_hidden.reshape(-1, pred_hidden.shape[-1]),
                lm_head=lm_head,
            ).view(batch_size, int(self.block_size) - 1)
        self.draft_model.lm_head = lm_head
        lambda_mix = (
            getattr(self.selector, "lambda_mix_value", None)
            if self.selector is not None
            and getattr(self.selector, "lambda_conditioned", False)
            else None
        )
        return _build_selector_lattice_proposal(
            draft_model=self.draft_model,
            pred_hidden=pred_hidden,
            anchor_token_ids=block_ids[:, 0],
            capture_features=(
                self.__dict__.get("_forest_branch_label_collector") is not None
            ),
            lambda_mix=lambda_mix,
        )

    @staticmethod
    def _request_refs(batch: ScheduleBatch) -> tuple[RequestRef, ...]:
        epoch = max(0, int(getattr(batch, "forward_iter", 0)))
        return tuple(
            RequestRef(request_id=req.rid, generation=0, forest_epoch=epoch)
            for req in batch.reqs
        )

    def _compile_native_plan(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: RequestForestDraftInput,
        proposal,
        node_capacity: int,
        request_refs: tuple[RequestRef, ...] | None = None,
        pre_draft_decision: PreDraftBudgetDecision | None = None,
        context_shape: tuple[int, int],
    ) -> _CompiledForestRound:
        batch_size = len(batch.reqs)
        if request_refs is None:
            request_refs = self._request_refs(batch)
        anchor_token_ids = draft_input.bonus_tokens.detach().to(
            device="cpu", dtype=torch.int64
        )
        anchor_positions = _cpu_seq_lens(batch).to(dtype=torch.int64)
        arena = self._forest_arena(
            request_refs=request_refs,
            anchor_token_ids=anchor_token_ids,
            anchor_positions=anchor_positions,
            node_capacity=node_capacity,
        )
        external = self.__dict__.get("_external_policy")
        acceptance_calibrator = (
            self._acceptance_calibrator_for(
                batch_size=batch_size, max_context_tokens=context_shape[1]
            ) if external is None else None
        )
        capture_acceptance_feedback = (
            acceptance_calibrator is not None
            and getattr(self, "_forest_pending_acceptance_feedback", None) is None
        )
        if node_capacity:
            if proposal is None:
                raise RuntimeError(
                    "Request Forest requires a Draft proposal for a nonzero "
                    "verification budget"
                )
            if external is not None:
                lowering = external.lower(
                    arena=arena, proposal=proposal, capacity=node_capacity,
                    context_lengths=anchor_positions.tolist(),
                )
                if external.stats["rounds"] % 128 == 1:
                    logger.info("External policy statistics: %s", external.stats)
            else:
                lowering = _lower_forest_proposal(
                    lowerer=self._forest_lowerer,
                    selector=self.selector,
                    arena=arena,
                    proposal=proposal,
                    capacity=node_capacity,
                    calibration_multipliers=(
                        None
                        if acceptance_calibrator is None
                        else acceptance_calibrator.multipliers
                    ),


                    d2_mix_lambda=(
                        0.0
                        if getattr(self.selector, "lambda_conditioned", False)
                        else self._forest_d2_mix_lambda
                    ),
                )
        else:
            empty_features = torch.empty(
                (0,), dtype=torch.float64, device="cpu"
            )
            lowering = DFlashLoweringResult(
                node_ids=range(0),
                conditional_probabilities=empty_features,
                max_probabilities=empty_features,
                margins=empty_features,
                entropies=empty_features,
            )
        if pre_draft_decision is not None:
            if node_capacity != 0 or not pre_draft_decision.use_target_only:
                raise ValueError(
                    "pre-Draft Target-only decision requires zero node capacity"
                )
            if pre_draft_decision.target_only_shape.num_requests != batch_size:
                raise ValueError(
                    "pre-Draft Target-only shape disagrees with the batch"
                )
            decision = _target_only_budget_decision(pre_draft_decision)
        else:
            decision = self._choose_budget_decision(
                arena=arena,
                allow_dynamic=self.selector is not None and external is None,
                context_shape=context_shape,
            )
        selected = self._forest_selected_ids[: decision.num_candidate_nodes]
        plan = _mark_truncated_frontier(
            arena.compile_verify_plan(
                selected,
                physical_rows=decision.physical_row_bucket,
                padding_token_id=0,
            ),
            lowering,
        )
        acceptance_feedback = (
            None
            if not capture_acceptance_feedback
            else _build_acceptance_feedback_metadata(
                plan=plan,
                lowering=lowering,
                bin_count=acceptance_calibrator.bin_count,
            )
        )
        tail_evidence = (
            _tail_round_evidence(arena=arena, decision=decision)
            if self.selector is not None and node_capacity > 0 and external is None
            else None
        )
        return _CompiledForestRound(
            plan=plan,
            budget_decision=decision,
            tail_evidence=tail_evidence,
            acceptance_feedback=acceptance_feedback,
            acceptance_calibrator=acceptance_calibrator,
        )

    def _replicated_forest_plan_enabled(self) -> bool:


        if not envs.SGLANG_ENABLE_REQUEST_FOREST_REPLICATED_PLAN.get():
            return False
        state = self.__dict__
        if state.get("_external_policy") is not None:
            return False
        controller = state.get("_forest_budget_controller")
        if controller is not None and getattr(controller, "profile_db", None) is not None:
            return False
        if state.get("_forest_crossover_profile") is not None:
            return False
        return True

    def _compile_verify_plan_across_tp(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: RequestForestDraftInput,
        proposal,
        node_capacity: int,
        request_refs: tuple[RequestRef, ...],
        pre_draft_decision: PreDraftBudgetDecision | None,
        context_shape: tuple[int, int],
        tp_group,
    ) -> _CompiledForestRound:
        if self._replicated_forest_plan_enabled() and not _cuda_graph_capturing():
            compiled = None
            ok = True
            try:
                compiled = self._compile_native_plan(
                    batch=batch,
                    draft_input=draft_input,
                    proposal=proposal,
                    node_capacity=node_capacity,
                    request_refs=request_refs,
                    pre_draft_decision=pre_draft_decision,
                    context_shape=context_shape,
                )
            except Exception:
                ok = False
                compiled = None
            if _replicated_plans_agree(
                tp_group,
                ok=ok,
                device=self.device,
            ):
                assert compiled is not None
                return compiled
        return self._broadcast_rank0_compiled_plan(
            batch=batch,
            draft_input=draft_input,
            proposal=proposal,
            node_capacity=node_capacity,
            request_refs=request_refs,
            pre_draft_decision=pre_draft_decision,
            context_shape=context_shape,
            tp_group=tp_group,
        )

    def _broadcast_rank0_compiled_plan(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: RequestForestDraftInput,
        proposal,
        node_capacity: int,
        request_refs: tuple[RequestRef, ...],
        pre_draft_decision: PreDraftBudgetDecision | None,
        context_shape: tuple[int, int],
        tp_group,
    ) -> _CompiledForestRound:
        compile_error: Exception | None = None
        local_compiled = None
        payload = None
        if int(tp_group.rank_in_group) == 0:
            try:
                local_compiled = self._compile_native_plan(
                    batch=batch,
                    draft_input=draft_input,
                    proposal=proposal,
                    node_capacity=node_capacity,
                    request_refs=request_refs,
                    pre_draft_decision=pre_draft_decision,
                    context_shape=context_shape,
                )
                payload = _pack_compiled_round(local_compiled)
            except Exception as exc:
                compile_error = exc
                payload = {
                    "error": f"{type(exc).__name__}: {exc}",
                }
        payload = tp_group.broadcast_tensor_dict(payload, src=0)
        if payload.get("error") is not None:
            if compile_error is not None:
                raise compile_error
            raise RuntimeError(
                "Request Forest plan compilation failed on TP rank zero: "
                f"{payload['error']}"
            )
        if int(tp_group.rank_in_group) == 0:
            assert local_compiled is not None
            return local_compiled
        return _unpack_compiled_round(
            payload,
            request_refs=request_refs,
        )

    def _compile_verify_input(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: RequestForestDraftInput,
        proposal,
        node_capacity: int,
        pre_draft_decision: PreDraftBudgetDecision | None = None,
        context_shape: tuple[int, int],
    ) -> _PreparedForestRound:
        batch_size = len(batch.reqs)
        classic_key = (batch_size, node_capacity)
        if (
            self._classic_verify_templates is not None
            and proposal is not None
            and (template := self._classic_verify_templates.get(classic_key))
            is not None
        ):
            verify_input = template.materialize(
                candidate_token_ids=proposal,
                anchor_token_ids=draft_input.bonus_tokens.to(torch.int64),
                anchor_positions=batch.seq_lens.to(torch.int64),
            )
            verify_input.tree_attention_workspace = (
                self._forest_tree_attention_workspace
            )
            total_context_tokens, max_context_tokens = context_shape
            verify_lens = template.verify_lens_cpu
            decision = _static_budget_decision(
                selected_node_ids=tuple(range(node_capacity)),
                path_values=torch.ones(
                    node_capacity,
                    dtype=torch.float64,
                    device="cpu",
                ),
                num_requests=batch_size,
                total_context_tokens=total_context_tokens,
                max_context_tokens=max_context_tokens,
                max_tree_depth=max(verify_lens, default=1) - 1,
                ancestor_edges=sum(
                    length * (length + 1) // 2 for length in verify_lens
                ),
            )
            return _PreparedForestRound(
                verify_input=verify_input,
                budget_decision=decision,
            )

        request_refs = self._request_refs(batch)
        tp_group = self.target_worker.model_runner.tp_group
        if int(tp_group.world_size) == 1:
            compiled = self._compile_native_plan(
                batch=batch,
                draft_input=draft_input,
                proposal=proposal,
                node_capacity=node_capacity,
                request_refs=request_refs,
                pre_draft_decision=pre_draft_decision,
                context_shape=context_shape,
            )
        else:
            compiled = self._compile_verify_plan_across_tp(
                batch=batch,
                draft_input=draft_input,
                proposal=proposal,
                node_capacity=node_capacity,
                request_refs=request_refs,
                pre_draft_decision=pre_draft_decision,
                context_shape=context_shape,
                tp_group=tp_group,
            )
        verify_input = ForestVerifyInput.from_native_plan(
            compiled.plan,
            device=self.device,
        )
        verify_input.tree_attention_workspace = self._forest_tree_attention_workspace
        if self._classic_verify_templates is not None and proposal is not None:
            self._classic_verify_templates[classic_key] = (
                ClassicChainVerifyTemplate.from_verify_input(
                    verify_input,
                    verify_lens_cpu=_classic_chain_verify_lens(
                        batch_size=batch_size,
                        depth_count=max(0, int(self.block_size) - 1),
                        capacity=node_capacity,
                    ),
                )
            )
        return _PreparedForestRound(
            verify_input=verify_input,
            budget_decision=compiled.budget_decision,
            tail_evidence=compiled.tail_evidence,
            acceptance_feedback=compiled.acceptance_feedback,
            acceptance_calibrator=compiled.acceptance_calibrator,
        )

    def _batch_eos_config(
        self,
        batch: ScheduleBatch,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor,
        frozenset[int],
        tuple[bool, ...],
    ]:
        common: set[int] | None = None
        for req in batch.reqs:
            token_ids = set(req.eos_token_ids or ())
            tokenizer = getattr(req, "tokenizer", None)
            if tokenizer is not None:
                eos_token_id = getattr(tokenizer, "eos_token_id", None)
                if eos_token_id is not None:
                    token_ids.add(int(eos_token_id))
                token_ids.update(getattr(tokenizer, "additional_stop_token_ids", ()) or ())
            if common is None:
                common = token_ids
            elif token_ids != common:
                raise NotImplementedError(
                    "Request Forest currently requires one common EOS token set "
                    "per verification forest"
                )
        cpu_eos_token_ids = frozenset(common or ())
        eos_token_ids = (
            None
            if not cpu_eos_token_ids
            else torch.tensor(
                sorted(cpu_eos_token_ids),
                dtype=torch.int64,
                device=self.device,
            )
        )
        cpu_request_ignore_eos = tuple(
            bool(req.sampling_params.ignore_eos) for req in batch.reqs
        )
        request_ignore_eos = torch.tensor(
            cpu_request_ignore_eos,
            dtype=torch.bool,
            device=self.device,
        )
        return (
            eos_token_ids,
            request_ignore_eos,
            cpu_eos_token_ids,
            cpu_request_ignore_eos,
        )

    def _materialize_accepted_draft_kv(
        self,
        verify_input: ForestVerifyInput,
        generation_result: GenerationBatchResult,
        acceptance: ForestGreedyResult,
        transaction: ForestKVScratchTransaction,
    ) -> None:
        logits_output = generation_result.logits_output
        hidden = None if logits_output is None else logits_output.hidden_states
        if hidden is None:
            raise RuntimeError(
                "Request Forest Target verification returned no DFlash context hidden states"
            )


        accepted_rows_2d = acceptance.accept_index.clamp_min(0)
        accepted_rows = accepted_rows_2d.reshape(-1)
        cache_loc_2d = transaction.scratch_slots.index_select(
            0,
            accepted_rows,
        ).view_as(accepted_rows_2d)
        self._append_target_hidden_to_draft_kv_by_loc(
            target_hidden=hidden.index_select(0, accepted_rows),
            cache_loc=cache_loc_2d.reshape(-1),
            cache_loc_2d=cache_loc_2d,
            commit_lens=acceptance.accept_lens,
            positions=verify_input.positions.index_select(0, accepted_rows),
        )

        logits_output.hidden_states = None

    def _prepare_accepted_draft_state(
        self,
        verify_input: ForestVerifyInput,
        generation_result: GenerationBatchResult,
        acceptance: ForestGreedyResult,
        transaction: ForestKVScratchTransaction,
    ) -> None:


        self._materialize_accepted_draft_kv(
            verify_input,
            generation_result,
            acceptance,
            transaction,
        )
        row_ids = self._forest_request_rows[: acceptance.accept_lens.numel()]
        bonus_tokens = acceptance.output_token_ids[
            row_ids,
            acceptance.accept_lens - 1,
        ]
        generation_result.next_draft_input = self._make_next_draft_input_decode(
            bonus_tokens=bonus_tokens,
            new_seq_lens=generation_result.new_seq_lens,
        )

    def _commit_target_mamba_state(
        self,
        *,
        batch: ScheduleBatch,
        verify_input: ForestVerifyInput,
        generation_result: GenerationBatchResult,
        acceptance: ForestGreedyResult,
        seq_lens_pre_verify: torch.Tensor,
    ) -> None:


        attn_backend = self.target_worker.model_runner.attn_backend
        track_interval = (
            mamba_track_grid(batch.tree_cache.page_size)
            if batch.mamba_track_indices is not None
            else None
        )
        last_steps, track_steps = _forest_mamba_commit_steps(
            request_offsets=verify_input.request_offsets,
            accept_index=acceptance.accept_index,
            accept_lens=acceptance.accept_lens,
            seq_lens_pre_verify=seq_lens_pre_verify,
            seq_lens_post_verify=generation_result.new_seq_lens,
            request_rows=self._forest_request_rows,
            track_interval=track_interval,
        )
        attn_backend.update_mamba_state_after_mtp_verify(
            last_correct_step_indices=last_steps,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=track_steps,
            model=self.target_worker.model_runner.model,
            req_pool_indices=batch.req_pool_indices[: acceptance.accept_lens.numel()],
        )

    def _forward_decode(
        self,
        batch: ScheduleBatch,
        draft_input: RequestForestDraftInput,
        *,
        on_publish=None,
    ) -> GenerationBatchResult:


        if batch.seq_lens.device.type == "cuda":
            batch.seq_lens.record_stream(
                torch.get_device_module(self.device).current_stream()
            )
        self._ingest_completed_target_timings()
        self._ingest_completed_draft_timings()
        self._poll_acceptance_feedback()
        if self.__dict__.get("_forest_rollout_logging_enabled", False):
            round_id = self._next_rollout_round_id()
            request_ids = tuple(req.rid for req in batch.reqs)
        else:
            round_id = 0
            request_ids = ()
        seq_lens_cpu = _cpu_seq_lens(batch)
        context_shape = _context_token_shape(batch, seq_lens_cpu)
        self._update_crossover_policy(
            batch_size=len(batch.reqs),
            max_context_tokens=context_shape[1],
        )
        max_node_capacity = self._verify_node_capacity(len(batch.reqs))
        use_target_only, pre_draft_decision = self._choose_pre_draft_decision(
            batch,
            node_capacity=max_node_capacity,
            context_shape=context_shape,
        )
        node_capacity = (
            0
            if use_target_only
            else max_node_capacity
        )


        (
            eos_token_ids,
            request_ignore_eos,
            cpu_eos_token_ids,
            cpu_request_ignore_eos,
        ) = self._batch_eos_config(batch)
        proposal = (
            None
            if node_capacity == 0
            else self._timed_draft_proposal(
                batch,
                draft_input,
                context_shape=context_shape,
                round_id=round_id,
            )
        )
        prepared_round = self._compile_verify_input(
            batch=batch,
            draft_input=draft_input,
            proposal=proposal,
            node_capacity=node_capacity,
            pre_draft_decision=(
                pre_draft_decision if node_capacity == 0 else None
            ),
            context_shape=context_shape,
        )
        self._log_budget_decision(
            prepared_round.budget_decision,
            round_id=round_id,
            request_ids=request_ids,
        )
        verify_input = prepared_round.verify_input
        need_mamba_verify_commit = bool(
            self.__dict__.get("_need_mamba_verify_commit", False)
        )
        if need_mamba_verify_commit:
            mamba_pool = (
                self.target_worker.model_runner.req_to_token_pool.mamba_pool
            )
            scratch = mamba_pool.mamba_cache.intermediate_ssm
            if getattr(mamba_pool, "spec_ssm_row_capacity", None) is not None:

                row_cap = mamba_pool.mamba_cache.intermediate_conv_window[0].shape[
                    2
                ]
                arena_too_small = (
                    scratch is None or verify_input.logical_rows > scratch.shape[1]
                )
            else:
                row_cap = 0 if scratch is None else scratch.shape[2]
                arena_too_small = scratch is None
            if arena_too_small or verify_input.max_request_rows > row_cap:
                raise ValueError(
                    "Request Forest rows exceed Target Mamba snapshot capacity"
                )
        seq_lens_pre_verify = (
            batch.seq_lens.clone() if need_mamba_verify_commit else None
        )

        def before_kv_publish(result, acceptance, transaction):
            self._prepare_accepted_draft_state(
                verify_input,
                result,
                acceptance,
                transaction,
            )

        def before_target_kv_publish(result, acceptance, transaction):
            del transaction
            if seq_lens_pre_verify is not None:
                self._commit_target_mamba_state(
                    batch=batch,
                    verify_input=verify_input,
                    generation_result=result,
                    acceptance=acceptance,
                    seq_lens_pre_verify=seq_lens_pre_verify,
                )

        if self.selector is None and self.__dict__.get("_external_policy") is None:
            assert self._classic_verify_templates is not None
            classic_template = self._classic_verify_templates.get(
                (
                    len(batch.reqs),
                    node_capacity,
                )
            )
        else:
            classic_template = None

        timing_handle = self._begin_target_timing(
            prepared_round.budget_decision,
            round_id=round_id,
        )
        try:
            if classic_template is None:
                eager_result = run_forest_target_verify_eager(
                    batch,
                    verify_input,
                    self.target_worker,
                    eos_token_ids=eos_token_ids,
                    request_ignore_eos=request_ignore_eos,
                    before_kv_publish=before_kv_publish,
                    before_target_kv_publish=(
                        before_target_kv_publish
                        if need_mamba_verify_commit
                        else None
                    ),
                    tp_group=self.target_worker.model_runner.tp_group,
                )
            else:
                eager_result = run_classic_chain_target_verify_eager(
                    batch,
                    verify_input,
                    self.target_worker,
                    template=classic_template,
                    eos_token_ids=eos_token_ids,
                    request_ignore_eos=request_ignore_eos,
                    before_kv_publish=before_kv_publish,
                    tp_group=self.target_worker.model_runner.tp_group,
                )
            if timing_handle is not None:
                actual_mode = (
                    "cuda_graph"
                    if eager_result.generation_result.can_run_cuda_graph
                    else "eager"
                )
                key = timing_handle.key
                if key.shape.execution_mode != actual_mode:


                    timing_handle.key = replace(
                        key, shape=replace(key.shape, execution_mode=actual_mode)
                    )
        finally:
            timer = getattr(self, "_forest_target_timer", None)
            if timer is not None and timing_handle is not None:
                try:
                    timer.end(timing_handle)
                except Exception:


                    logger.warning(
                        "Failed to finish Request Forest Target timing",
                        exc_info=True,
                    )
        acceptance_feedback = getattr(
            prepared_round, "acceptance_feedback", None
        )
        if acceptance_feedback is not None:
            self._stage_acceptance_feedback(
                acceptance_feedback,
                prepared_round.acceptance_calibrator,
                eager_result.target_predict,
                eos_token_ids=cpu_eos_token_ids,
                request_ignore_eos=cpu_request_ignore_eos,
            )
        branch_label_collector = self.__dict__.get(
            "_forest_branch_label_collector"
        )
        round_outcome = None
        if branch_label_collector is not None:
            try:
                logits_output = eager_result.generation_result.logits_output
                branch_label_collector.collect_successful_round(
                    verify_input=verify_input,
                    acceptance=eager_result.acceptance,
                    logits=logits_output.next_token_logits,
                    round_id=round_id,
                    request_ids=request_ids,
                    selector_features=(
                        proposal
                        if isinstance(proposal, _SelectorLatticeView)
                        else None
                    ),
                )
                round_outcome = branch_label_collector.last_outcome
            except Exception:


                logger.warning(
                    "Failed to collect Request Forest branch labels",
                    exc_info=True,
                )
                round_outcome = branch_label_collector.last_outcome
        try:
            if round_outcome is not None:
                self._log_round_outcome(
                    outcome=round_outcome,
                    decision=prepared_round.budget_decision,
                    request_ids=request_ids,
                )
        except Exception:


            logger.warning(
                "Failed to record Request Forest round outcome",
                exc_info=True,
            )
        tail_evidence = getattr(prepared_round, "tail_evidence", None)
        if tail_evidence is not None:
            self._forest_tail_evidence = tail_evidence
        generation_result = eager_result.generation_result
        if on_publish is not None:
            on_publish(generation_result.new_seq_lens)


        target_runner = self.target_worker.model_runner
        if target_runner.shared_read_done_event is not None:
            read_done = torch.get_device_module(self.device).Event()
            read_done.record()
            target_runner.shared_read_done_event = read_done
        return generation_result

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish=None,
        grammar_barrier=None,
    ) -> GenerationBatchResult:
        del grammar_barrier
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            return super().forward_batch_generation(batch, on_publish=on_publish)

        if batch.spec_info is None:
            batch.spec_info = RequestForestDraftInput(
                bonus_tokens=torch.empty((0,), dtype=torch.int64, device=self.device),
                new_seq_lens=torch.empty((0,), dtype=torch.int64, device=self.device),
            )
        draft_input = batch.spec_info
        if not isinstance(draft_input, RequestForestDraftInput):
            raise RuntimeError(
                "Request Forest expected RequestForestDraftInput on decode"
            )
        if batch.forward_mode.is_idle():
            empty_ids = torch.empty((0,), dtype=torch.int64, device=self.device)
            empty_lens = torch.empty((0,), dtype=torch.int32, device=self.device)
            next_draft_input = self._make_next_draft_input_decode(
                bonus_tokens=empty_ids,
                new_seq_lens=empty_ids,
            )
            if on_publish is not None:
                on_publish(next_draft_input.new_seq_lens)
            return GenerationBatchResult(
                next_token_ids=empty_ids,
                accept_lens=empty_lens,
                next_draft_input=next_draft_input,
                new_seq_lens=next_draft_input.new_seq_lens,
                can_run_cuda_graph=False,
                speculative_num_draft_tokens=int(self.block_size),
            )
        return self._forward_decode(
            batch,
            draft_input,
            on_publish=on_publish,
        )


__all__ = ["RequestForestWorkerV2"]
