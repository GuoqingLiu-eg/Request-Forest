import os

from sglang.srt.environ import envs


def experimental_lora_enabled() -> bool:

    return envs.SGLANG_EXPERIMENTAL_LORA_OPTI.get()


_TRUE = {"1", "true", "yes", "on"}


class _GatedBool:
    def __init__(self, name: str, default: bool):
        self._name = name
        self._default = default

    def get(self) -> bool:
        if not experimental_lora_enabled():
            return False
        raw = os.environ.get(self._name)
        if raw is None:
            return self._default
        return raw.strip().lower() in _TRUE


class _GatedInt:
    def __init__(self, name: str, default: int):
        self._name = name
        self._default = default

    def get(self) -> int:

        if not experimental_lora_enabled():
            return self._default
        raw = os.environ.get(self._name)
        return int(raw) if raw is not None else self._default


class _LoraEnvs:

    SGLANG_ENABLE_LORA_SHRINK_SPLIT_K = _GatedBool(
        "SGLANG_ENABLE_LORA_SHRINK_SPLIT_K", True
    )
    SGLANG_OPT_LORA_FUSED_MERGED_ALIGN = _GatedBool(
        "SGLANG_OPT_LORA_FUSED_MERGED_ALIGN", True
    )
    SGLANG_OPT_LORA_FUSED_TOPK_PACK = _GatedBool(
        "SGLANG_OPT_LORA_FUSED_TOPK_PACK", True
    )
    SGLANG_OPT_LORA_QKV_B_STORE = _GatedBool("SGLANG_OPT_LORA_QKV_B_STORE", True)


    SGLANG_OPT_LORA_PREFILL_ROUTING_REUSE = _GatedBool(
        "SGLANG_OPT_LORA_PREFILL_ROUTING_REUSE", True
    )


    SGLANG_OPT_KIMI_GATE_BF16_INPUT = _GatedBool(
        "SGLANG_OPT_KIMI_GATE_BF16_INPUT", True
    )


    SGLANG_OPT_USE_JIT_KERNEL_KIMI_GATE = _GatedBool(
        "SGLANG_OPT_USE_JIT_KERNEL_KIMI_GATE", False
    )
    SGLANG_OPT_USE_JIT_KERNEL_MOE_ALIGN = _GatedBool(
        "SGLANG_OPT_USE_JIT_KERNEL_MOE_ALIGN", False
    )

    SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC = _GatedBool(
        "SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC", False
    )

    SGLANG_OPT_LORA_SHARED_ADD_OVERLAP = _GatedBool(
        "SGLANG_OPT_LORA_SHARED_ADD_OVERLAP", False
    )
    SGLANG_OPT_LORA_CUBLAS = _GatedBool("SGLANG_OPT_LORA_CUBLAS", False)
    SGLANG_OPT_LORA_CUBLAS_A = _GatedBool("SGLANG_OPT_LORA_CUBLAS_A", False)
    SGLANG_OPT_LORA_CUBLAS_B = _GatedBool("SGLANG_OPT_LORA_CUBLAS_B", False)
    SGLANG_OPT_LORA_CUBLAS_GATE_UP = _GatedBool("SGLANG_OPT_LORA_CUBLAS_GATE_UP", False)
    SGLANG_OPT_LORA_CUBLAS_QKV = _GatedBool("SGLANG_OPT_LORA_CUBLAS_QKV", False)
    SGLANG_OPT_LORA_CUBLAS_KV_B = _GatedBool("SGLANG_OPT_LORA_CUBLAS_KV_B", False)

    SGLANG_OPT_LORA_SHRINK_TUNE = _GatedBool("SGLANG_OPT_LORA_SHRINK_TUNE", False)


    SGLANG_OPT_FUSED_PERMUTE_QUANT = _GatedBool("SGLANG_OPT_FUSED_PERMUTE_QUANT", False)


    SGLANG_TWO_STREAM_MAX_TOKENS = _GatedInt("SGLANG_TWO_STREAM_MAX_TOKENS", 256)


lora_envs = _LoraEnvs()
