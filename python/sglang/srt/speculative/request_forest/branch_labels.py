from __future__ import annotations

import atexit
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import torch
from safetensors.torch import save_file

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX

if TYPE_CHECKING:
    from sglang.srt.speculative.dflash_worker_v2 import _SelectorLatticeView

    from .forest_verify_input import ForestVerifyInput
    from .gpu_accept import ForestGreedyResult


@dataclass(frozen=True, slots=True)
class _ForestTopology:
    parent_rows: torch.Tensor
    depths: torch.Tensor
    children: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class _PendingRecord:
    tensors: dict[str, torch.Tensor]
    request_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BranchRoundOutcome:


    round_id: int
    accepted_draft_lens: tuple[int, ...]
    accept_lens: tuple[int, ...]
    first_miss_positions: tuple[int, ...]
    termination_reasons: tuple[int, ...]
    accepted_token_ids: tuple[tuple[int, ...], ...]
    committed_token_ids: tuple[tuple[int, ...], ...]

    @property
    def request_count(self) -> int:
        return len(self.accept_lens)

    @property
    def accepted_draft_token_count(self) -> int:
        return sum(self.accepted_draft_lens)

    @property
    def committed_token_count(self) -> int:
        return sum(self.accept_lens)


def summarize_round_outcome(
    acceptance: ForestGreedyResult,
    *,
    round_id: int,
) -> BranchRoundOutcome:


    accepted_draft_lens = tuple(
        int(value)
        for value in acceptance.accepted_draft_lens.detach().to("cpu").tolist()
    )
    accept_lens = tuple(
        int(value) for value in acceptance.accept_lens.detach().to("cpu").tolist()
    )
    output_token_ids = acceptance.output_token_ids.detach().to(
        device="cpu", dtype=torch.int64
    )
    accepted_token_ids = tuple(
        tuple(int(token) for token in output_token_ids[row, :length].tolist())
        for row, length in enumerate(accepted_draft_lens)
    )
    committed_token_ids = tuple(
        tuple(int(token) for token in output_token_ids[row, :length].tolist())
        for row, length in enumerate(accept_lens)
    )
    return BranchRoundOutcome(
        round_id=round_id,
        accepted_draft_lens=accepted_draft_lens,
        accept_lens=accept_lens,
        first_miss_positions=tuple(
            int(value)
            for value in acceptance.first_miss_positions.detach().to("cpu").tolist()
        ),
        termination_reasons=tuple(
            int(value)
            for value in acceptance.termination_reasons.detach().to("cpu").tolist()
        ),
        accepted_token_ids=accepted_token_ids,
        committed_token_ids=committed_token_ids,
    )


def _forest_topology(
    first_child: torch.Tensor,
    next_sibling: torch.Tensor,
) -> _ForestTopology:


    logical_rows = first_child.numel()
    parents = torch.full((logical_rows,), -1, dtype=torch.int64)
    children: list[tuple[int, ...]] = []
    for parent in range(logical_rows):
        row_children: list[int] = []
        child = int(first_child[parent])
        while child >= 0:
            row_children.append(child)
            parents[child] = parent
            child = int(next_sibling[child])
        children.append(tuple(row_children))

    depths = torch.zeros(logical_rows, dtype=torch.int64)
    for row in range(logical_rows):
        parent = int(parents[row])
        if parent >= 0:
            depths[row] = depths[parent] + 1
    return _ForestTopology(
        parent_rows=parents,
        depths=depths,
        children=tuple(children),
    )


def _sample_prefix_closed_parents(
    topology: _ForestTopology,
    *,
    max_parents: int,
    generator: torch.Generator,
) -> torch.Tensor:


    labelable = [row for row, children in enumerate(topology.children) if children]
    if len(labelable) <= max_parents:
        return torch.tensor(labelable, dtype=torch.int64)

    selected: set[int] = set()
    order = torch.randperm(len(labelable), generator=generator).tolist()
    for candidate_index in order:
        row = labelable[candidate_index]
        closure: list[int] = []
        cursor = row
        while cursor >= 0 and topology.children[cursor]:
            if cursor in selected:
                break
            closure.append(cursor)
            cursor = int(topology.parent_rows[cursor])
        closure.reverse()
        if len(selected) + len(closure) <= max_parents:
            selected.update(closure)
        if len(selected) == max_parents:
            break
    return torch.tensor(sorted(selected), dtype=torch.int64)


def _candidate_matrix(
    topology: _ForestTopology,
    selected_parents: torch.Tensor,
    token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    parent_list = selected_parents.tolist()
    width = max(len(topology.children[parent]) for parent in parent_list)
    candidate_ids = torch.zeros(
        (len(parent_list), width),
        dtype=torch.int64,
    )
    candidate_mask = torch.zeros_like(candidate_ids, dtype=torch.bool)
    for output_row, parent in enumerate(parent_list):
        child_rows = topology.children[parent]
        child_tokens = token_ids[list(child_rows)]
        candidate_ids[output_row, : len(child_rows)] = child_tokens
        candidate_mask[output_row, : len(child_rows)] = True
    return candidate_ids, candidate_mask


def _ancestor_matrix(
    topology: _ForestTopology,
    selected_parents: torch.Tensor,
    token_ids: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:


    paths: list[list[int]] = []
    for parent in selected_parents.tolist():
        path: list[int] = []
        cursor = parent
        while cursor >= 0:
            path.append(cursor)
            cursor = int(topology.parent_rows[cursor])
        paths.append(list(reversed(path)))

    width = max(len(path) for path in paths)
    ancestor_token_ids = torch.zeros(
        (len(paths), width),
        dtype=torch.int64,
    )
    ancestor_positions = torch.zeros_like(ancestor_token_ids)
    ancestor_mask = torch.zeros_like(ancestor_token_ids, dtype=torch.bool)
    for output_row, path in enumerate(paths):
        path_rows = torch.tensor(path, dtype=torch.int64)
        ancestor_token_ids[output_row, : len(path)] = token_ids.index_select(
            0, path_rows
        )
        ancestor_positions[output_row, : len(path)] = positions.index_select(
            0, path_rows
        )
        ancestor_mask[output_row, : len(path)] = True
    return ancestor_token_ids, ancestor_positions, ancestor_mask


def _reached_rows(
    acceptance: ForestGreedyResult,
    *,
    logical_rows: int,
) -> torch.Tensor:
    width = acceptance.accept_index.shape[1]
    columns = torch.arange(
        width,
        dtype=acceptance.accept_lens.dtype,
        device=acceptance.accept_lens.device,
    )
    valid = columns.unsqueeze(0) < acceptance.accept_lens.unsqueeze(1)
    accepted_rows = acceptance.accept_index.masked_select(valid).to(torch.int64)
    reached = torch.zeros(
        logical_rows,
        dtype=torch.bool,
        device=acceptance.accept_index.device,
    )
    if accepted_rows.numel():
        reached[accepted_rows] = True
    return reached


def _gather_target_statistics(
    logits: torch.Tensor,
    *,
    parent_rows: torch.Tensor,
    candidate_ids: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:


    device = logits.device
    row_index = parent_rows.to(device=device)
    candidate_index = candidate_ids.to(device=device)
    mask = candidate_mask.to(device=device)
    safe_index = candidate_index.masked_fill(~mask, 0)
    parent_logits = logits.index_select(0, row_index).to(torch.float32)
    restricted = parent_logits.gather(1, safe_index).masked_fill(~mask, 0)
    target_logsumexp = torch.logsumexp(parent_logits, dim=-1)
    target_top1_token_ids = parent_logits.argmax(dim=-1).to(torch.int64)
    target_top1_in_candidates = (
        candidate_index.eq(target_top1_token_ids.unsqueeze(-1)) & mask
    ).any(dim=-1)
    return (
        restricted,
        target_logsumexp,
        target_top1_token_ids,
        target_top1_in_candidates,
    )


def _gather_selector_features(
    selector_features: _SelectorLatticeView,
    *,
    topology: _ForestTopology,
    parent_rows: torch.Tensor,
    request_slots: torch.Tensor,
    token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:


    edge_hidden = selector_features.edge_hidden
    unary_scores = selector_features.unary_scores
    lattice_candidates = selector_features.candidate_ids
    if edge_hidden is None or unary_scores is None:
        raise ValueError("selector feature view is incomplete")
    if lattice_candidates.ndim != 3:
        raise ValueError("selector candidate_ids must have shape [B, E, K]")
    batch_size, edge_count, top_k = lattice_candidates.shape
    if unary_scores.shape != (batch_size, edge_count, top_k):
        raise ValueError("selector unary_scores disagree with candidate_ids")
    if edge_hidden.ndim != 3 or edge_hidden.shape[:2] != (
        batch_size,
        edge_count,
    ):
        raise ValueError("selector edge_hidden must have shape [B, E, H]")

    selected_slots = request_slots.index_select(0, parent_rows)
    selected_depths = topology.depths.index_select(0, parent_rows)
    if bool(
        selected_slots.lt(0).any()
        or selected_slots.ge(batch_size).any()
        or selected_depths.lt(0).any()
        or selected_depths.ge(edge_count).any()
    ):
        raise ValueError("Forest parent is outside the selector lattice")

    device = lattice_candidates.device
    device_slots = selected_slots.to(device=device)
    device_depths = selected_depths.to(device=device)
    fixed_candidates = lattice_candidates[device_slots, device_depths]
    candidate_ids = fixed_candidates.detach().to(device="cpu").contiguous()
    candidate_mask = torch.ones_like(candidate_ids, dtype=torch.bool)
    return (
        candidate_ids,
        candidate_mask,
        {
            "edge_hidden": edge_hidden[device_slots, device_depths]
            .detach()
            .to(device="cpu")
            .contiguous(),
            "predecessor_ids": token_ids.index_select(0, parent_rows).contiguous(),
            "unary_scores": unary_scores[device_slots, device_depths]
            .detach()
            .to(device="cpu")
            .contiguous(),
        },
    )


def _pad_columns(tensor: torch.Tensor, width: int) -> torch.Tensor:
    if tensor.shape[1] == width:
        return tensor
    output = torch.zeros(
        (tensor.shape[0], width), dtype=tensor.dtype, device=tensor.device
    )
    output[:, : tensor.shape[1]] = tensor
    return output


def _merge_pending_records(
    records: Sequence[_PendingRecord],
) -> tuple[dict[str, torch.Tensor], list[str]]:


    candidate_width = max(
        record.tensors["candidate_ids"].shape[1] for record in records
    )
    ancestor_width = max(
        record.tensors["ancestor_token_ids"].shape[1] for record in records
    )
    padded_keys = {
        "candidate_ids": candidate_width,
        "target_logits": candidate_width,
        "candidate_mask": candidate_width,
        "unary_scores": candidate_width,
        "ancestor_token_ids": ancestor_width,
        "ancestor_positions": ancestor_width,
        "ancestor_mask": ancestor_width,
    }
    pieces: dict[str, list[torch.Tensor]] = {
        key: [] for key in records[0].tensors
    }
    shard_request_ids: list[str] = []
    request_id_to_slot: dict[str, int] = {}

    for record in records:
        request_slots = record.tensors["request_slots"]
        remapped_slots = torch.empty_like(request_slots)
        for source_slot in torch.unique(request_slots).tolist():
            if source_slot < 0 or source_slot >= len(record.request_ids):
                raise ValueError(
                    f"request slot {source_slot} is outside the round request IDs"
                )
            request_id = record.request_ids[source_slot]
            target_slot = request_id_to_slot.get(request_id)
            if target_slot is None:
                target_slot = len(shard_request_ids)
                request_id_to_slot[request_id] = target_slot
                shard_request_ids.append(request_id)
            remapped_slots[request_slots.eq(source_slot)] = target_slot

        for key, tensor in record.tensors.items():
            if key == "request_slots":
                tensor = remapped_slots
            width = padded_keys.get(key)


            tensor = tensor.detach().to(device="cpu")
            pieces[key].append(_pad_columns(tensor, width) if width else tensor)

    return (
        {key: torch.cat(tensors, dim=0).contiguous() for key, tensors in pieces.items()},
        shard_request_ids,
    )


class BranchLabelCollector:


    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        *,
        max_parents: int,
        seed: int,
        shard_rounds: int = 128,
        shard_parents: int = 8192,
    ) -> None:
        if max_parents < 1:
            raise ValueError("max_parents must be positive")
        if shard_rounds < 1 or shard_parents < 1:
            raise ValueError("branch-label shard limits must be positive")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_parents = max_parents
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)
        self.shard_rounds = shard_rounds
        self.shard_parents = shard_parents
        self._pending: list[_PendingRecord] = []
        self._pending_parents = 0
        self._last_outcome: BranchRoundOutcome | None = None
        self._shard_sequence = 0
        self._closed = False
        atexit.register(self.close)

    def collect_successful_round(
        self,
        *,
        verify_input: ForestVerifyInput,
        acceptance: ForestGreedyResult,
        logits: torch.Tensor,
        round_id: int,
        request_ids: Sequence[str] = (),
        selector_features: _SelectorLatticeView | None = None,
    ) -> Path | None:


        if self._closed:
            raise RuntimeError("cannot collect labels after the collector is closed")

        self._last_outcome = None

        if request_ids and all(
            request_id.startswith(HEALTH_CHECK_RID_PREFIX)
            for request_id in request_ids
        ):
            return None


        self._last_outcome = summarize_round_outcome(
            acceptance,
            round_id=round_id,
        )

        logical_rows = verify_input.logical_rows
        first_child = verify_input.first_child[:logical_rows].to(
            device="cpu", dtype=torch.int64
        )
        next_sibling = verify_input.next_sibling[:logical_rows].to(
            device="cpu", dtype=torch.int64
        )
        topology = _forest_topology(first_child, next_sibling)
        parent_rows = _sample_prefix_closed_parents(
            topology,
            max_parents=self.max_parents,
            generator=self.generator,
        )
        if parent_rows.numel() == 0:
            return None

        token_ids = verify_input.draft_token[:logical_rows].to(
            device="cpu", dtype=torch.int64
        )
        reached = _reached_rows(acceptance, logical_rows=logical_rows)
        device_parent_rows = parent_rows.to(device=reached.device)
        parent_mask = reached.index_select(0, device_parent_rows).to(device="cpu")
        if not bool(parent_mask.any()):
            return None

        if verify_input.request_slots is None:
            offsets = verify_input.request_offsets.to(device="cpu", dtype=torch.int64)
            rows = torch.arange(logical_rows, dtype=torch.int64)
            request_slots = torch.bucketize(rows, offsets[1:], right=True)
        else:
            request_slots = verify_input.request_slots[:logical_rows].to(
                device="cpu", dtype=torch.int64
            )
        selector_tensors: dict[str, torch.Tensor] | None = None
        if selector_features is None:
            candidate_ids, candidate_mask = _candidate_matrix(
                topology,
                parent_rows,
                token_ids,
            )
        else:
            candidate_ids, candidate_mask, selector_tensors = (
                _gather_selector_features(
                    selector_features,
                    topology=topology,
                    parent_rows=parent_rows,
                    request_slots=request_slots,
                    token_ids=token_ids,
                )
            )

        (
            target_logits,
            target_logsumexp,
            target_top1_token_ids,
            target_top1_in_candidates,
        ) = _gather_target_statistics(
            logits,
            parent_rows=parent_rows,
            candidate_ids=candidate_ids,
            candidate_mask=candidate_mask,
        )

        positions = verify_input.positions[:logical_rows].to(
            device="cpu", dtype=torch.int64
        )
        (
            ancestor_token_ids,
            ancestor_positions,
            ancestor_mask,
        ) = _ancestor_matrix(
            topology,
            parent_rows,
            token_ids,
            positions,
        )
        tensors = {
            "candidate_ids": candidate_ids.contiguous(),
            "target_logits": target_logits.to(device="cpu").contiguous(),
            "target_logsumexp": target_logsumexp.to(device="cpu").contiguous(),
            "target_top1_token_ids": target_top1_token_ids.to(
                device="cpu"
            ).contiguous(),
            "target_top1_in_candidates": target_top1_in_candidates.to(
                device="cpu"
            ).contiguous(),
            "candidate_mask": candidate_mask.contiguous(),
            "parent_mask": parent_mask.contiguous(),
            "ancestor_token_ids": ancestor_token_ids.contiguous(),
            "ancestor_positions": ancestor_positions.contiguous(),
            "ancestor_mask": ancestor_mask.contiguous(),
            "parent_rows": parent_rows.contiguous(),
            "parent_positions": positions.index_select(0, parent_rows).contiguous(),
            "parent_token_ids": token_ids.index_select(0, parent_rows).contiguous(),
            "parent_depths": topology.depths.index_select(0, parent_rows).contiguous(),
            "request_slots": request_slots.index_select(0, parent_rows).contiguous(),
            "round_ids": torch.full_like(parent_rows, round_id),
        }
        if selector_tensors is not None:
            tensors.update(selector_tensors)
        self._pending.append(
            _PendingRecord(tensors=tensors, request_ids=tuple(request_ids))
        )
        self._pending_parents += parent_rows.numel()
        if (
            len(self._pending) >= self.shard_rounds
            or self._pending_parents >= self.shard_parents
        ):
            return self.flush()
        return None

    @property
    def last_outcome(self) -> BranchRoundOutcome | None:
        return self._last_outcome

    def flush(self) -> Path | None:


        if not self._pending:
            return None
        tensors, request_ids = _merge_pending_records(self._pending)
        first_round_id = int(tensors["round_ids"][0])
        last_round_id = int(tensors["round_ids"][-1])
        metadata = {
            "request_ids": json.dumps(request_ids, ensure_ascii=False),
            "target_logits_kind": "raw_with_full_vocab_logsumexp",
            "branch_label_schema": "2",
        }
        if "edge_hidden" in tensors:
            metadata["selector_feature_schema"] = "1"
        self._shard_sequence += 1
        filename = (
            f"branch-shard-{first_round_id}-{last_round_id}-"
            f"{os.getpid()}-{self._shard_sequence}.safetensors"
        )
        final_path = self.output_dir / filename
        temporary_path = self.output_dir / f".{filename}.tmp"
        save_file(tensors, str(temporary_path), metadata=metadata)
        os.replace(temporary_path, final_path)
        self._pending.clear()
        self._pending_parents = 0
        return final_path

    def close(self) -> Path | None:


        if self._closed:
            return None
        output = self.flush()
        self._closed = True
        atexit.unregister(self.close)
        return output


__all__ = [
    "BranchLabelCollector",
    "BranchRoundOutcome",
    "summarize_round_outcome",
]
