from __future__ import annotations

from typing import Any, Iterable


ATTENTION_API_ATTRS = ("q_proj", "k_proj", "v_proj", "o_proj", "rope")

SCALE_ATTRS = ("scale", "sm_scale")
NUM_HEAD_ATTRS = ("n_heads", "num_heads", "num_attention_heads")
NUM_KV_HEAD_ATTRS = ("n_kv_heads", "num_k_heads", "num_kv_heads", "num_key_value_heads")
SLIDING_ATTENTION_ATTRS = (
    "is_sliding",
    "use_sliding",
    "is_sliding_window",
    "use_sliding_window",
    "is_swa",
)


WINDOW_SIZE_ATTRS = ("window_size", "sliding_window")


def first_present_attr(module: Any, names: Iterable[str]) -> Any | None:

    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    return None


def get_num_heads(module: Any) -> int | None:
    return first_present_attr(module, NUM_HEAD_ATTRS)


def get_num_kv_heads(module: Any) -> int | None:
    return first_present_attr(module, NUM_KV_HEAD_ATTRS)


def get_head_dim(module: Any) -> int | None:
    head_dim = first_present_attr(module, ("head_dim",))
    if head_dim is not None:
        return head_dim

    n_kv_heads = get_num_kv_heads(module)
    if n_kv_heads is None:
        return None
    if hasattr(module, "hidden_size") and hasattr(module, "num_k_heads"):
        return module.hidden_size // module.num_k_heads
    if hasattr(module, "k_proj") and hasattr(module.k_proj, "weight"):
        return module.k_proj.weight.shape[0] // n_kv_heads
    return None


def get_attention_scale(module: Any) -> float | None:
    return first_present_attr(module, SCALE_ATTRS)


def is_attention_module(module: Any) -> bool:
    return (
        all(hasattr(module, attr) for attr in ATTENTION_API_ATTRS)
        and any(hasattr(module, attr) for attr in SCALE_ATTRS)
        and get_num_heads(module) is not None
        and get_num_kv_heads(module) is not None
    )


def language_model_container(model: Any) -> Any:

    root = getattr(model, "language_model", model)
    return getattr(root, "model", root)


def get_container_window_size(model: Any) -> int | None:

    return first_present_attr(language_model_container(model), WINDOW_SIZE_ATTRS)


def get_layer_window_sizes(model: Any) -> dict[int, int | None]:

    container = language_model_container(model)
    layer_types = getattr(container, "layer_types", None)
    window_size = first_present_attr(container, WINDOW_SIZE_ATTRS)
    if not layer_types or window_size is None:
        return {}
    return {
        idx: window_size if layer_type == "sliding_attention" else None
        for idx, layer_type in enumerate(layer_types)
    }


def uses_sliding_window_attention(*modules: Any) -> bool:
    return any(
        bool(getattr(module, attr, False))
        for module in modules
        for attr in SLIDING_ATTENTION_ATTRS
    )
