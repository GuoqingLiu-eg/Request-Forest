from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)


class MlxQuantizationConfig(QuantizationConfig):


    _ERR = (
        "MLX on-the-fly quantization (--quantization mlx_q4 / mlx_q8) is "
        "handled by the MLX backend at model-load time via mlx_lm.utils."
        "quantize_model, not by this QuantizationConfig class. If you "
        "reached this error, SGLANG_USE_MLX=1 is likely not set."
    )

    def __init__(self, preset: str):
        super().__init__()
        self.preset = preset

    @classmethod
    def get_name(cls) -> str:
        return "mlx"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return []

    @classmethod
    def get_min_capability(cls) -> int:

        return 0

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> MlxQuantizationConfig:
        raise NotImplementedError(cls._ERR)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:

        if user_quant is not None:


            return None
        if not isinstance(hf_quant_cfg, dict):
            return None
        if "quant_method" in hf_quant_cfg:


            return None
        bits = hf_quant_cfg.get("bits")
        group_size = hf_quant_cfg.get("group_size")
        if not isinstance(bits, int) or not isinstance(group_size, int):
            return None
        if bits == 4:
            return "mlx_q4"
        if bits == 8:
            return "mlx_q8"
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        raise NotImplementedError(self._ERR)
