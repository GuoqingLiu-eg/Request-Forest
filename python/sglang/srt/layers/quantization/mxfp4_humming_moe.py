from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch.nn import Module, Parameter

from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.utils import log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput

logger = logging.getLogger(__name__)


class Mxfp4HummingMoEMethod:


    def __init__(self, fp8_method, prefix: str):
        self._fp8 = fp8_method
        self.prefix = prefix

    def create_moe_runner(self, layer, moe_runner_config):
        from sglang.srt.layers.moe.moe_runner import MoeRunner

        self.runner = MoeRunner(MoeRunnerBackend.HUMMING, moe_runner_config)

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self._fp8.create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        from sglang.srt.layers.quantization.humming_utils import (
            prepare_humming_moe_layer,
        )


        self._fp8.process_weights_after_loading(layer)

        if getattr(layer, "_mega_moe_weights_built", False):
            return

        log_info_on_rank0(
            logger,
            f"Preparing DeepSeekV4 MXFP4 experts for Humming backend "
            f"(layer: {self.prefix})...",
        )

        layer.register_parameter(
            "w13_weight_scale",
            Parameter(
                layer.w13_weight_scale_inv.to(torch.float8_e8m0fnu),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "w2_weight_scale",
            Parameter(
                layer.w2_weight_scale_inv.to(torch.float8_e8m0fnu),
                requires_grad=False,
            ),
        )
        del layer.w13_weight_scale_inv
        del layer.w2_weight_scale_inv

        prepare_humming_moe_layer(layer, {"quant_method": "mxfp4"})
        layer._dsv4_mxfp4_backend = "humming"

    def apply(
        self,
        layer: Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.moe_runner.humming import HummingMoeQuantInfo

        quant_info = HummingMoeQuantInfo(layer=layer)
        return self.runner.run(dispatch_output, quant_info)
