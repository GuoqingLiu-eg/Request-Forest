from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GraphLane:
    spec_algorithm: Any
    attn_backend: Any
    graph_runner: Any
    capture_tail_hooks: tuple
    sampler: Any = None
    prefill_graph_runner: Any = None


def capture_lane(runner, sampler=None):
    return GraphLane(runner.spec_algorithm, runner.attn_backend,
                     runner.decode_cuda_graph_runner,
                     tuple(runner.capture_tail_hooks), sampler,
                     getattr(runner, "prefill_cuda_graph_runner", None))


def activate_lane(runner, lane):
    runner.spec_algorithm = lane.spec_algorithm
    runner.attn_backend = lane.attn_backend
    runner.decode_cuda_graph_runner = lane.graph_runner
    runner.prefill_cuda_graph_runner = lane.prefill_graph_runner
    runner.capture_tail_hooks = list(lane.capture_tail_hooks)


def capture_prefill_lane(runner, *, capture_graph=None):

    from sglang.srt.model_executor.graph_memory_usage import (
        merge_graph_memory_usage,
        merge_graph_time_usage,
    )

    if capture_graph is None:
        from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
            capture_prefill_graph,
        )
        capture_graph = capture_prefill_graph

    capture = capture_graph(model_runner=runner, eager_runner=runner.eager_runner)
    runner.prefill_cuda_graph_runner = capture.runner


    runner.graph_memory_usage = merge_graph_memory_usage(
        runner.graph_memory_usage, capture.memory_usage
    )
    runner.graph_time_usage = merge_graph_time_usage(
        runner.graph_time_usage, capture.time_usage
    )
