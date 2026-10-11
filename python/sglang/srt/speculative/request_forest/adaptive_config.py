from __future__ import annotations

import os


_QWEN35_MOE_TARGETS = frozenset(
    {"Qwen3_5MoeForCausalLM", "Qwen3_5MoeForConditionalGeneration"}
)


def adaptive_rf_enabled() -> bool:
    value = os.environ.get("SGLANG_REQUEST_FOREST_ADAPTIVE", "0")
    if value not in ("0", "1"):
        raise ValueError("SGLANG_REQUEST_FOREST_ADAPTIVE must be 0 or 1")
    return value == "1"


def adaptive_native_max_batch_size(max_requests: int) -> int:

    enter_rf = int(os.environ.get("SGLANG_REQUEST_FOREST_ADAPTIVE_ENTER", "64"))
    if enter_rf < 2:
        raise ValueError("SGLANG_REQUEST_FOREST_ADAPTIVE_ENTER must be at least 2")
    return min(max_requests, enter_rf - 1)


def validate_adaptive_config(cfg) -> None:

    required = {
        "pp_size": 1,
        "dp_size": 1,
        "attn_cp_size": 1,
        "page_size": 1,
        "speculative_request_forest_d2_mix_lambda": 0.0,
        "speculative_request_forest_external_policy": "rf",
    }
    for key, value in required.items():
        if getattr(cfg, key, None) != value:
            raise ValueError(f"Adaptive RF prototype requires {key}={value!r}")
    if getattr(cfg, "speculative_num_draft_tokens", None) not in (8, 9):
        raise ValueError("Adaptive RF supports original DFlash2 W8 or W9")
    if getattr(cfg, "tp_size", None) not in (1, 2, 4):
        raise ValueError("Adaptive RF supports TP1, TP2, or TP4")
    graph = getattr(cfg, "cuda_graph_config", None)
    if graph is None or getattr(getattr(graph, "prefill", None), "backend", None) not in ("disabled", "breakable"):
        raise ValueError("Adaptive RF supports disabled or breakable prefill graphs")
    if getattr(getattr(graph, "decode", None), "backend", None) not in ("disabled", "full"):
        raise ValueError("Adaptive RF supports disabled or full decode graphs")
    for key in ("enable_pdmux", "enable_two_batch_overlap", "enable_dp_attention"):
        if getattr(cfg, key, False):
            raise ValueError(f"Adaptive RF does not support {key}")


    for key in (
        "speculative_request_forest_crossover_profile_path",
        "speculative_request_forest_target_profile_path",
    ):
        if getattr(cfg, key, None) is not None:
            raise ValueError(f"Adaptive RF prototype does not support {key}")


def validate_adaptive_target(target_worker) -> None:
    runner = target_worker.model_runner
    config = runner.model_config.hf_config
    supported = {
        "Qwen3ForCausalLM", "Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration"
    } | _QWEN35_MOE_TARGETS
    architectures = list(getattr(config, "architectures", ()) or ())
    if len(architectures) != 1 or architectures[0] not in supported or runner.model.__class__.__name__ not in supported:
        raise ValueError("Adaptive RF supports dense Qwen3 and Qwen3.5 dense/MoE GDN models only")


def validate_adaptive_selector(selector) -> None:
    if selector is None:
        raise ValueError("Adaptive RF requires an original DFlash2 selector")


    if getattr(selector, "lambda_conditioned", False):
        raise ValueError(
            "This no-additional-training prototype requires the original DFlash2 selector"
        )


def validate_adaptive_draft_window(
    draft_config, configured_width: int, *, target_architecture: str | None = None
) -> None:

    checkpoint_width = draft_config.block_size


    if (
        target_architecture in _QWEN35_MOE_TARGETS
        and checkpoint_width == 16
        and configured_width == 8
    ):
        return
    if checkpoint_width not in (8, 9):
        raise ValueError("Adaptive RF requires an explicit original DFlash2 W8 or W9 checkpoint window")
    if checkpoint_width != configured_width:
        raise ValueError(
            "Adaptive RF draft window must match the original checkpoint: "
            f"checkpoint W{checkpoint_width}, configured W{configured_width}"
        )
