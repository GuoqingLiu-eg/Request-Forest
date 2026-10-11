import logging
from contextlib import contextmanager
from typing import List, NamedTuple, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.kernels.ops.communication.all_reduce import (
    AllReduceAlgo,
    Communicator,
    IPCManager,
    PullPlane,
    PushPlane,
    custom_all_reduce,
)
from sglang.srt.distributed.parallel_state import in_the_same_node_as
from sglang.srt.environ import envs
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.utils.cuda_vmm_utils import (
    VmmGraphInputManager,
    compute_graph_capture_bases,
    is_vmm_pointer,
)

from .configs.custom_all_reduce_v2 import (
    get_all_reduce_config,
    get_supported_world_sizes,
)
from .custom_all_reduce_utils import (
    can_use_custom_all_reduce_with_nvlink,
    is_one_nvlink_clique,
    is_weak_contiguous,
)

logger = logging.getLogger(__name__)

MB = 1024 * 1024

_ALIGN_BYTES = 1024
_SEMAPHORE_BYTES = 128
_MAX_GRAPH_INPUTS = 131072

_DEFAULT_MAX_SIZE = envs.SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB.get() * 1024


_FORCE_PULL_SIZE_KB = envs.SGLANG_FORCE_CUSTOM_ALL_REDUCE_V2_PULL_SIZE_KB.get()
_FORCE_PUSH_SIZE_KB = envs.SGLANG_FORCE_CUSTOM_ALL_REDUCE_V2_PUSH_SIZE_KB.get()


def _ceil_align(nbytes: int, align: int) -> int:
    return (nbytes + align - 1) // align * align


def _allocate_symmetric_memory(nbytes: int, device: torch.device, group: ProcessGroup):
    from torch._C._distributed_c10d import _SymmetricMemory

    if torch.__version__ < "2.11.0":
        import torch.distributed._symmetric_memory as torch_symm_mem

        torch_symm_mem.enable_symm_mem_for_group(group.group_name)
    tensor = _SymmetricMemory.empty_strided_p2p(
        (nbytes,),
        [1],
        torch.uint8,
        device,
        group.group_name,
    )
    symm_mem = _SymmetricMemory.rendezvous(tensor)
    return tensor, symm_mem


class AllReduceConfig(NamedTuple):
    algo: AllReduceAlgo
    use_graph: bool = False
    use_multicast: bool = False


class CustomAllReduceV2:
    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        max_size: int = _DEFAULT_MAX_SIZE,
        *,
        max_pull_size: Optional[int] = None,
        max_push_size: Optional[int] = None,
        max_pull_blocks: Optional[int] = None,
        max_push_blocks: Optional[int] = None,
    ) -> None:

        self.disabled = True
        if not can_use_custom_all_reduce_v2(group=group, device=device):
            return

        self.group = group
        self.device = device
        self.rank = dist.get_rank(group=self.group)
        self.world_size = dist.get_world_size(group=self.group)
        base_config = get_all_reduce_config(self.world_size)
        if max_pull_size is None:
            max_pull_size = min(base_config.max_pull_bytes, max_size)
        if max_push_size is None:
            max_push_size = min(base_config.max_push_bytes, max_size)
        if _FORCE_PULL_SIZE_KB is not None:
            max_pull_size = int(_FORCE_PULL_SIZE_KB) * 1024
        if _FORCE_PUSH_SIZE_KB is not None:
            max_push_size = int(_FORCE_PUSH_SIZE_KB) * 1024

        def force_thresholds(heuristic):


            if _FORCE_PULL_SIZE_KB is not None:
                heuristic = heuristic._replace(two_shot_pull_threshold=max_pull_size)
            if _FORCE_PUSH_SIZE_KB is not None:
                heuristic = heuristic._replace(one_shot_push_threshold=max_push_size)
            return heuristic

        base_config = base_config._replace(
            graph=force_thresholds(base_config.graph),
            eager=force_thresholds(base_config.eager),
        )


        self.pull_enabled = max_pull_blocks != 0 and max_pull_size > 0
        self.max_pull_size = (
            _ceil_align(max(max_pull_size, _ALIGN_BYTES), _ALIGN_BYTES)
            if self.pull_enabled
            else 0
        )
        self.max_push_size = _ceil_align(max(max_push_size, _ALIGN_BYTES), _ALIGN_BYTES)
        self.max_size = max(self.max_pull_size, self.max_push_size)
        num_pull_blocks = base_config.num_pull_blocks
        num_push_blocks = base_config.num_push_blocks
        if max_pull_blocks:
            num_pull_blocks = max(min(num_pull_blocks, max_pull_blocks), 1)
        if max_push_blocks is not None:
            num_push_blocks = max(max_push_blocks, 1)
        self.config = base_config.clip(
            max_push_bytes=self.max_push_size, max_pull_bytes=self.max_pull_size
        )._replace(num_pull_blocks=num_pull_blocks, num_push_blocks=num_push_blocks)
        self.override_algo: Optional[AllReduceAlgo] = None


        is_multinode = not all(in_the_same_node_as(group, source_rank=0))
        tms_cudagraph = envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.get()
        self._is_graph_mode_supported = not tms_cudagraph and not is_multinode


        self.graph_params = torch.zeros(
            (_MAX_GRAPH_INPUTS, self.world_size),
            dtype=torch.uint64,
            device=self.device,
        )
        self._init_workspace()
        self._ipc_manager = IPCManager()
        self._vmm_graph_input_manager = VmmGraphInputManager(
            obj=self,
            group=self.group,
            rank=self.rank,
            world_size=self.world_size,
        )
        self._graph_inputs: List[Tuple[int, int]] = []
        self._graph_counter = 0
        self._graph_mode_allowed = False
        self.disabled = False

    def _init_workspace(self) -> None:

        cfg = self.config
        push_num_slots = 2 * self.world_size
        push_bytes = push_num_slots * self.max_push_size
        pull_bytes = self.max_pull_size
        sem_bytes = _SEMAPHORE_BYTES * cfg.num_pull_blocks if self.pull_enabled else 0
        total_bytes = push_bytes + pull_bytes + sem_bytes
        pull_offset = push_bytes
        sem_offset = push_bytes + pull_bytes

        self._symm_tensor, symm_mem = _allocate_symmetric_memory(
            total_bytes, device=self.device, group=self.group
        )
        slabs = [
            symm_mem.get_buffer(i, [total_bytes], torch.uint8)
            for i in range(self.world_size)
        ]


        slabs[self.rank].zero_()
        torch.cuda.synchronize()
        dist.barrier(group=self.group)

        def slice_all(shape: List[int], offset: int) -> List[torch.Tensor]:

            nbytes = 1
            for dim in shape:
                nbytes *= dim
            assert offset + nbytes <= total_bytes
            return [slab[offset : offset + nbytes].view(shape) for slab in slabs]

        multicast_ptr = int(symm_mem.multicast_ptr)
        self.has_multicast = multicast_ptr != 0

        def mc_at(offset: int) -> Optional[int]:
            return multicast_ptr + offset if self.has_multicast else None

        self._push_counter = torch.zeros(
            (cfg.num_push_blocks,), dtype=torch.uint32, device=self.device
        )
        push_plane = PushPlane(
            self.rank,
            self.world_size,
            workspaces=slice_all([push_num_slots, self.max_push_size], 0),
            counter=self._push_counter.view(-1, 1).view(torch.uint8),
            mc_workspace=mc_at(0),
        )
        pull_plane = None
        if self.pull_enabled:
            pull_plane = PullPlane(
                self.rank,
                self.world_size,
                workspaces=slice_all([pull_bytes], pull_offset),
                semaphores=slice_all(
                    [cfg.num_pull_blocks, _SEMAPHORE_BYTES], sem_offset
                ),
                mc_workspace=mc_at(pull_offset),
                mc_semaphore=mc_at(sem_offset),
            )
        if not self.has_multicast or not self.pull_enabled:
            self.config = self.config._replace(num_mc_blocks=None)

        self.obj = Communicator(push=push_plane, pull=pull_plane)
        if self.config.num_mc_blocks is not None:
            self.obj.set_pull_multicast_blocks(self.config.num_mc_blocks)
        if self.rank == 0:
            logger.info(
                "All Reduce config: symmetric_memory = %.2f MB, "
                "local_buffer = %.2f MB, multicast = %s, pull = %s",
                total_bytes / MB,
                (self.graph_params.nbytes + self._push_counter.nbytes) / MB,
                self.config.num_mc_blocks is not None,
                self.pull_enabled,
            )
        dist.barrier(group=self.group)


    def uncap_pull_thresholds(self) -> None:


        if not self.pull_enabled:
            return

        def uncap(heuristic):
            return heuristic._replace(two_shot_pull_threshold=self.max_pull_size)

        self.config = self.config._replace(
            graph=uncap(self.config.graph),
            eager=uncap(self.config.eager),
        )

    def _can_use_graph(self) -> bool:


        return (
            self._graph_mode_allowed
            and not is_in_tc_piecewise_cuda_graph()
            and torch.cuda.is_current_stream_capturing()
        )

    def _pick_config(self, nbytes: int, can_use_graph: bool) -> AllReduceConfig | None:

        heuristic = self.config.graph if can_use_graph else self.config.eager
        can_use_multicast = self.config.num_mc_blocks is not None
        if nbytes <= heuristic.one_shot_push_threshold:
            return AllReduceConfig(AllReduceAlgo.ONE_SHOT_PUSH)
        if nbytes <= heuristic.one_shot_pull_threshold:
            return AllReduceConfig(AllReduceAlgo.ONE_SHOT_PULL, use_graph=can_use_graph)
        if can_use_multicast and heuristic.mc.contains(nbytes):
            return AllReduceConfig(AllReduceAlgo.TWO_SHOT_PULL, use_multicast=True)
        if nbytes <= heuristic.two_shot_pull_threshold:
            return AllReduceConfig(AllReduceAlgo.TWO_SHOT_PULL, use_graph=can_use_graph)
        return None

    def should_custom_ar(self, inp: torch.Tensor) -> bool:

        if self.disabled:
            return False
        inp_size = inp.numel() * inp.element_size()

        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        if self.override_algo is not None:
            return inp_size <= self.max_size
        return self._pick_config(inp_size, self._can_use_graph()) is not None


    def custom_all_reduce(self, input: torch.Tensor) -> torch.Tensor:
        nbytes = input.numel() * input.element_size()
        if self.override_algo is not None:

            algo = self.override_algo
            use_graph = self._can_use_graph() and not algo.is_push()
            use_multicast = False
        else:
            config = self._pick_config(nbytes, self._can_use_graph())
            assert config is not None, f"No config for {nbytes = }"
            algo, use_graph, use_multicast = config
        graph_params = self._allocate_graph_row(input, nbytes) if use_graph else None
        return custom_all_reduce(
            self.obj,
            input,
            algo=algo,
            graph_params=graph_params,
            use_multicast=use_multicast,
        )

    def _allocate_graph_row(self, input: torch.Tensor, nbytes: int) -> torch.Tensor:
        index = self._graph_counter + len(self._graph_inputs)
        assert index < _MAX_GRAPH_INPUTS, "Graph input table overflow"
        self._graph_inputs.append((input.data_ptr(), nbytes))
        return self.graph_params[index]


    @contextmanager
    def capture(self):
        if self.disabled:
            yield
            return
        try:
            self._graph_mode_allowed = self._is_graph_mode_supported
            yield
        finally:
            self._graph_mode_allowed = False
        assert (
            not torch.cuda.is_current_stream_capturing()
        ), "Cannot register graph inputs while capturing CUDA graph"
        self._register_graph_inputs()

    def _register_graph_inputs(self) -> None:
        if not self._graph_inputs:
            return
        first_ptr = self._graph_inputs[0][0]
        if is_vmm_pointer(first_ptr):

            self._vmm_graph_input_manager.register_graph_inputs()
        else:
            self._register_graph_inputs_ipc()

    def _register_graph_inputs_ipc(self) -> None:

        ptrs = [ptr for ptr, _ in self._graph_inputs]
        handles = self._ipc_manager.batch_get_handles(ptrs)
        local = [(list(handle), int(offset)) for handle, offset in handles]
        gathered: List[Optional[list]] = [None] * self.world_size
        dist.all_gather_object(gathered, local, group=self.group)
        ptrs_per_rank: List[List[int]] = []
        for rank, remote in enumerate(gathered):
            if rank == self.rank:
                ptrs_per_rank.append(ptrs)
            else:
                ptrs_per_rank.append(list(self._ipc_manager.batch_open_handles(remote)))
        peer_ptrs = [
            [ptrs_per_rank[rank][i] for rank in range(self.world_size)]
            for i in range(len(ptrs))
        ]
        self.register_peer_mapped_inputs(peer_ptrs)

    def get_graph_capture_bases(self):

        return compute_graph_capture_bases(self._graph_inputs)

    def register_peer_mapped_inputs(self, peer_ptrs: List[List[int]]) -> None:

        assert len(peer_ptrs) == len(self._graph_inputs)
        count = len(peer_ptrs)
        rows = torch.tensor(peer_ptrs, dtype=torch.uint64, device=self.device)
        self.graph_params[self._graph_counter : self._graph_counter + count].copy_(rows)

        torch.cuda.synchronize()
        self._graph_counter += count
        self._graph_inputs.clear()


    def close(self):
        if not self.disabled and hasattr(self, "obj"):
            self._ipc_manager.destroy()
            dist.barrier(group=self.group)
            del self.obj
        if hasattr(self, "_vmm_graph_input_manager"):
            self._vmm_graph_input_manager.close()

    def __del__(self):
        self.close()


def _is_vmm_backed_allocator(device: torch.device) -> bool:

    probe = torch.empty(1, dtype=torch.uint8, device=device)
    return is_vmm_pointer(probe.data_ptr())


def can_use_custom_all_reduce_v2(
    group: ProcessGroup,
    device: torch.device,
) -> bool:
    supported = get_supported_world_sizes()
    if dist.get_world_size(group=group) not in supported:
        return False
    if not all(in_the_same_node_as(group, source_rank=0)):
        return is_one_nvlink_clique(group, device) and _is_vmm_backed_allocator(device)
    full_nvlink = can_use_custom_all_reduce_with_nvlink(
        group=group,
        device=device,
        supported_world_size=list(supported),
        cls_name="CustomAllReduceV2",
    )
    return full_nvlink is True
