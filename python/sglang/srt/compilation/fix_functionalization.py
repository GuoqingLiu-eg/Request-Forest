# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.10.0/vllm/compilation/fix_functionalization.py

import logging
import operator
from collections.abc import Iterable
from typing import Optional, Union

import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized

from sglang.srt.compilation.fx_utils import is_func
from sglang.srt.compilation.inductor_pass import SGLangInductorPass

logger = logging.getLogger(__name__)


class FixFunctionalizationPass(SGLangInductorPass):


    def __call__(self, graph: torch.fx.Graph):
        self.begin()
        self.dump_graph(graph, "before_fix_functionalization")

        self.nodes_to_remove: list[torch.fx.Node] = []
        count = 0
        for node in graph.nodes:
            if not is_func(node, auto_functionalized):
                continue
            count += 1

        self.dump_graph(graph, "before_fix_functionalization_cleanup")


        count_removed = len(self.nodes_to_remove)
        for node in self.nodes_to_remove:
            graph.erase_node(node)

        logger.debug(
            "De-functionalized %s nodes, removed %s nodes", count, count_removed
        )
        self.dump_graph(graph, "after_fix_functionalization")
        self.end_and_log()

    def _remove(self, node_or_nodes: Union[torch.fx.Node, Iterable[torch.fx.Node]]):

        if isinstance(node_or_nodes, torch.fx.Node):
            self.nodes_to_remove.append(node_or_nodes)
        else:
            self.nodes_to_remove.extend(node_or_nodes)

    def defunctionalize(
        self,
        graph: torch.fx.Graph,
        node: torch.fx.Node,
        mutated_args: dict[int, Union[torch.fx.Node, str]],
        args: Optional[tuple[Union[torch.fx.Node, str], ...]] = None,
    ):

        self.replace_users_with_mutated_args(node, mutated_args)
        self.insert_defunctionalized(graph, node, args=args)
        self._remove(node)

    def replace_users_with_mutated_args(
        self, node: torch.fx.Node, mutated_args: dict[int, Union[torch.fx.Node, str]]
    ):

        for idx, user in self.getitem_users(node).items():
            arg = mutated_args[idx]
            arg = node.kwargs[arg] if isinstance(arg, str) else arg
            user.replace_all_uses_with(arg)
            self._remove(user)

    def getitem_users(self, node: torch.fx.Node) -> dict[int, torch.fx.Node]:

        users = {}
        for user in node.users:
            if is_func(user, operator.getitem):
                idx = user.args[1]
                users[idx] = user
        return users

    def insert_defunctionalized(
        self,
        graph: torch.fx.Graph,
        node: torch.fx.Node,
        args: Optional[tuple[Union[torch.fx.Node, str], ...]] = None,
    ):

        assert is_func(
            node, auto_functionalized
        ), f"node must be auto-functionalized, is {node} instead"


        with graph.inserting_before(node):
            function = node.args[0]
            if args is None:
                graph.call_function(function, kwargs=node.kwargs)
            else:

                args = tuple(
                    node.kwargs[arg] if isinstance(arg, str) else arg for arg in args
                )
                graph.call_function(function, args=args)
