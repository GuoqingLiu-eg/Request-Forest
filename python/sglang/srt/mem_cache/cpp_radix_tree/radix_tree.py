from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
from torch.utils.cpp_extension import load

_abs_path = os.path.dirname(os.path.abspath(__file__))
radix_tree_cpp = load(
    name="radix_tree_cpp",
    sources=[
        f"{_abs_path}/tree_v2_binding.cpp",
        f"{_abs_path}/tree_v2_debug.cpp",
        f"{_abs_path}/tree_v2.cpp",
    ],
    extra_cflags=["-O3", "-std=c++20"],
)

if TYPE_CHECKING:

    class TreeNodeCpp:
        pass

    class IOHandle:
        pass

    class RadixTreeCpp:
        def __init__(
            self,
            disabled: bool,
            host_size: Optional[int],
            page_size: int,
            write_through_threshold: int,
        ):

            self.tree = radix_tree_cpp.RadixTree(  # type: ignore
                disabled, host_size, page_size, write_through_threshold
            )

        def match_prefix(
            self, prefix: List[int]
        ) -> Tuple[List[torch.Tensor], int, TreeNodeCpp, TreeNodeCpp]:

            return self.tree.match_prefix(prefix)

        def evict(self, num_tokens: int) -> List[torch.Tensor]:

            return self.tree.evict(num_tokens)

        def lock_ref(self, handle: TreeNodeCpp, lock: bool) -> None:

            return self.tree.lock_ref(handle, lock)

        def writing_through(
            self, key: List[int], indices: torch.Tensor
        ) -> Tuple[List[Tuple[IOHandle, torch.Tensor, torch.Tensor]], int]:

            return self.tree.writing_through(key, indices)

        def loading_onboard(
            self,
            host_node: TreeNodeCpp,
            new_device_indices: torch.Tensor,
        ) -> Tuple[IOHandle, List[torch.Tensor]]:

            return self.tree.loading_onboard(host_node, new_device_indices)

        def commit_writing_through(self, handle: IOHandle, success: bool) -> None:

            return self.tree.commit_writing_through(handle, success)

        def commit_loading_onboard(self, handle: IOHandle, success: bool) -> None:

            return self.tree.commit_loading_onboard(handle, success)

        def evictable_size(self) -> int:

            return self.tree.evictable_size()

        def protected_size(self) -> int:

            return self.tree.protected_size()

        def total_size(self) -> int:

            return self.tree.total_size()

        def reset(self) -> None:

            return self.tree.reset()

        def debug_print(self) -> None:

            return self.tree.debug_print()

else:

    RadixTreeCpp = radix_tree_cpp.RadixTree
    TreeNodeCpp = object
    IOHandle = object
