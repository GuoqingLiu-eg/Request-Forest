from importlib import import_module


_GROUPS = (
    "activation",
    "attention",
    "communication",
    "diffusion",
    "elementwise",
    "embeddings",
    "gemm",
    "grammar",
    "kimi_k3",
    "kvcache",
    "layernorm",
    "mamba",
    "memory",
    "mm",
    "moe",
    "quantization",
    "sampling",
    "speculative",
    "lplb",
    "kv_canary",
)

for _group in _GROUPS:
    import_module(f"{__name__}.{_group}")

del import_module, _group

__all__ = list(_GROUPS)
