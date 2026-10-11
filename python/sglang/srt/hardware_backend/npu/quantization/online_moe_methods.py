from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.hardware_backend.npu.quantization.moe_methods import NPUMXFP8MoEMethod
from sglang.srt.layers.moe.moe_runner import MoeRunner
from sglang.srt.layers.moe.utils import MoeRunnerBackend, get_moe_runner_backend
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.quantization.base_config import QuantizationConfig


class NPUMXFP8OnlineMoEMethod(UnquantizedFusedMoEMethod):


    def __init__(self, quant_config: Optional["QuantizationConfig"] = None):
        super().__init__()
        self.quant_config = quant_config

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"
    ):
        backend = get_moe_runner_backend()
        if not (backend.is_auto() or backend.is_ascend()):


            raise ValueError(
                "MXFP8 MoE on Ascend requires --moe-runner-backend 'auto' or "
                f"'ascend', got {backend.value!r}."
            )


        layer.w13_kernel = NPUMXFP8MoEMethod("w13")
        layer.w2_kernel = NPUMXFP8MoEMethod("w2")
        moe_runner_config.layer = layer
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.ASCEND, moe_runner_config)

        self._aiter_runner = None
