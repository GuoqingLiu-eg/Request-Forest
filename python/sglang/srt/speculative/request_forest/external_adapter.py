from __future__ import annotations

import bisect
import json
import math
import time
from pathlib import Path

import torch

from .external_policy_types import PolicySelection


class MeasuredBastionCosts:

    def __init__(self, rows):
        self.curves = {}
        for row in rows:
            b, c, n = (row[x] for x in ("batch_size", "context_tokens", "tree_nodes"))
            t = row["cycle_seconds"]
            if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in (b, c, n)):
                raise ValueError("cost dimensions must be positive integers")
            if not isinstance(t, (int, float)) or isinstance(t, bool) or not math.isfinite(t) or t <= 0:
                raise ValueError("cycle_seconds must be measured, finite and positive")
            curve = self.curves.setdefault((b, c), {})
            if n in curve:
                raise ValueError("duplicate cost measurement cell")
            curve[n] = float(t) / b
        if not self.curves:
            raise ValueError("BASTION requires measured cycle costs")
        for curve in self.curves.values():
            sizes = sorted(curve)
            if sizes[0] != 1 or len(sizes) < 2:
                raise ValueError("each cost curve must include root-only and larger trees")
            if any(curve[a] > curve[b] for a, b in zip(sizes, sizes[1:])):
                raise ValueError("fit a monotone cost curve before using it")

    def for_batch(self, batch_size):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch size must be a positive integer")
        bs = sorted({b for b, _ in self.curves if b >= batch_size})
        if not bs:
            raise ValueError("batch size is outside the measured BASTION profile")
        b = bs[0]

        def cost(request_index, tree_nodes, context_length):
            if any(isinstance(v, bool) or not isinstance(v, int) for v in (request_index, tree_nodes, context_length)):
                raise ValueError("cost query dimensions must be integers")
            if not 0 <= request_index < batch_size or tree_nodes < 1 or context_length < 0:
                raise ValueError("invalid cost query dimensions")
            contexts = sorted(c for bb, c in self.curves if bb == b and c >= context_length)
            if not contexts:
                raise ValueError("context is outside the measured BASTION profile")
            curve = self.curves[b, contexts[0]]
            sizes = sorted(curve)
            at = bisect.bisect_left(sizes, tree_nodes)
            if at == len(sizes) or tree_nodes < 1:
                raise ValueError("tree size is outside the measured BASTION profile")
            if sizes[at] == tree_nodes:
                return curve[tree_nodes]
            lo, hi = sizes[at - 1], sizes[at]
            return curve[lo] + (curve[hi] - curve[lo]) * (tree_nodes - lo) / (hi - lo)
        return cost


def append_policy_selection(arena, selection: PolicySelection):

    from .dflash_adapter import DFlashLoweringResult

    offset = arena.node_count
    nodes = selection.nodes
    if len(nodes) > arena.capacity - offset:
        raise ValueError("external policy exceeded arena capacity")
    siblings = set(zip(
        arena.node_request_slots.tolist(),
        arena.node_parent_ids.tolist(),
        arena.node_token_ids.tolist(),
    )) if offset else set()
    for i, node in enumerate(nodes):
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (
            node.request_index, node.parent_index, node.token_id, node.depth, node.candidate_rank
        )):
            raise ValueError("external policy indices and token IDs must be integers")
        if node.candidate_rank < 0:
            raise ValueError("candidate rank must be nonnegative")
        global_parent = -1 if node.parent_index == -1 else node.parent_index + offset
        sibling = (node.request_index, global_parent, node.token_id)
        if sibling in siblings:
            raise ValueError("external policy emitted duplicate sibling tokens")
        siblings.add(sibling)
        if not 0 <= node.request_index < arena.request_count or node.token_id < 0:
            raise ValueError("invalid external policy request or token")
        if node.parent_index == -1:
            if node.depth != 1:
                raise ValueError("anchor child must have depth one")
            parent_probability = 1.0
        else:
            if not 0 <= node.parent_index < i:
                raise ValueError("external policy is not ancestor closed")
            parent = nodes[node.parent_index]
            if parent.request_index != node.request_index or parent.depth + 1 != node.depth:
                raise ValueError("external policy mixed requests or depths")
            parent_probability = parent.path_probability
        q, p = node.conditional_probability, node.path_probability
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (q, p)):
            raise ValueError("external policy emitted invalid probability")
        if not math.isclose(parent_probability * q, p, rel_tol=1e-6, abs_tol=1e-12):
            raise ValueError("external policy path probability is inconsistent")
    parents = [-1 if n.parent_index == -1 else n.parent_index + offset for n in nodes]
    ids = arena.append_tensors(torch.tensor(parents, dtype=torch.int64),
                              torch.tensor([n.request_index for n in nodes], dtype=torch.int64),
                              torch.tensor([n.token_id for n in nodes], dtype=torch.int64),
                              torch.tensor([n.depth for n in nodes], dtype=torch.int64),
                              torch.tensor([n.path_probability for n in nodes], dtype=torch.float64))
    empty = torch.empty(0, dtype=torch.float64)
    return DFlashLoweringResult(node_ids=ids, conditional_probabilities=empty,
                                max_probabilities=empty, margins=empty, entropies=empty)


class ExternalPolicyAdapter:
    def __init__(self, policy, profile_path):
        self.policy = policy
        self.profile = json.loads(Path(profile_path).read_text())
        if self.profile.get("policy") != ("echo-style" if policy == "echo" else policy):
            raise ValueError("external policy and profile disagree")
        self.per_request_cap = self.profile["max_candidates_per_request"]
        if isinstance(self.per_request_cap, bool) or not isinstance(self.per_request_cap, int) or self.per_request_cap < 1:
            raise ValueError("max_candidates_per_request must be a positive integer")
        self.stats = {"rounds": 0, "selection_seconds": 0.0, "selected_nodes": 0}
        if policy == "bastion":
            if self.profile.get("cost_scope") != "balanced_batch_cycle_per_request":
                raise ValueError("BASTION profile must declare balanced batch cost attribution")
            self.costs = MeasuredBastionCosts(self.profile["measurements"])
        elif policy == "echo":
            from .echo_policy import load_echo_profile
            self.checkpoints = load_echo_profile(profile_path)
        else:
            raise ValueError("unknown external policy")

    def lower(self, *, arena, proposal, capacity, context_lengths):
        started = time.perf_counter()
        ids = proposal.candidate_ids.detach().cpu().tolist()
        if self.policy == "bastion":
            from .bastion_policy import select_bastion_batch
            probabilities = proposal.probabilities.detach().float().cpu().tolist()
            selection = select_bastion_batch(ids, probabilities, capacity=capacity,
                context_lengths=context_lengths, cost_model=self.costs.for_batch(len(ids)),
                min_nodes_per_request=self.profile.get("min_tree_nodes", 32),
                max_nodes_per_request=self.per_request_cap + 1,
                bucket_step=self.profile.get("bucket_step", 32))
        else:
            from .echo_policy import select_echo_batch
            probabilities = torch.softmax(proposal.scores.detach().float(), dim=-1).cpu().tolist()
            selection = select_echo_batch(ids, probabilities, capacity=capacity,
                checkpoints=self.checkpoints, per_request_cap=self.per_request_cap,
                initial_width=self.profile.get("initial_width", 1),
                max_width=self.profile.get("max_width"))
        output = append_policy_selection(arena, selection)
        self.stats["rounds"] += 1
        self.stats["selected_nodes"] += len(selection.nodes)
        self.stats["selection_seconds"] += time.perf_counter() - started
        self.stats["last_round"] = selection.stats
        return output
