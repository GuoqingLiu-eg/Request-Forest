# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.10.0/vllm/compilation/pass_manager.py

import logging

from torch import fx as fx

from sglang.srt.compilation.fix_functionalization import FixFunctionalizationPass
from sglang.srt.compilation.inductor_pass import (
    CustomGraphPass,
    InductorPass,
    SGLangInductorPass,
    get_pass_context,
)

logger = logging.getLogger(__name__)


class PostGradPassManager(CustomGraphPass):


    def __init__(self):
        self.passes: list[SGLangInductorPass] = []

    def __call__(self, graph: fx.Graph):
        shape = get_pass_context().runtime_shape
        for pass_ in self.passes:
            if pass_.is_applicable_for_shape(shape):
                pass_(graph)


        self.fix_functionalization(graph)

    def configure(
        self,
    ):
        self.pass_config = dict()
        self.fix_functionalization = FixFunctionalizationPass()

    def add(self, pass_: InductorPass):
        assert isinstance(pass_, InductorPass)
        self.passes.append(pass_)

    def uuid(self):

        pass_manager_uuid = "fshdakhsa"
        state = {"pass_config": pass_manager_uuid, "passes": []}
        for pass_ in self.passes:
            state["passes"].append(pass_.uuid())
        state["passes"].append(self.fix_functionalization.uuid())
        return InductorPass.hash_dict(state)
