# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/device_communicators/custom_all_reduce.py

import ctypes
import logging
from contextlib import contextmanager
from functools import partial
from typing import Any, List, Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import sglang.srt.distributed.device_communicators.custom_all_reduce_ops as ops
from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
    can_use_custom_all_reduce_with_nvlink,
    is_weak_contiguous,
)
from sglang.srt.environ import envs
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.utils import (
    get_bool_env_var,
    is_cuda,
    is_hip,
    is_musa,
    log_info_on_rank0,
)

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_musa = is_musa()

logger = logging.getLogger(__name__)


class CustomAllreduce:
    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]
    _MAX_CAR_SIZE = 8192 * 1024
    if _is_hip:

        _MAX_CAR_SIZE = 2 * 8192 * 1024
    if _is_musa:

        _MAX_CAR_SIZE = 16 * 8196 * 1024


    def __init__(
        self,
        group: ProcessGroup,
        device: Union[int, str, torch.device],
        max_size=_MAX_CAR_SIZE,
    ) -> None:

        self._IS_CAPTURING = False
        self.disabled = True
        self.original_disabled = True
        self.use_amd_deterministic_impl = _use_amd_deterministic_impl()

        if not ops.IS_CUSTOM_AR_AVAILABLE:


            return

        rank = dist.get_rank(group=group)
        world_size = dist.get_world_size(group=group)

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)

        assert isinstance(device, torch.device)
        self.device = device
        full_nvlink = can_use_custom_all_reduce_with_nvlink(
            group=group,
            device=device,
            supported_world_size=self._SUPPORTED_WORLD_SIZES,
            cls_name="CustomAllreduce",
        )
        if full_nvlink is None:
            return

        self.group = group
        self.max_size = max_size
        self.rank = rank
        self.world_size = world_size
        self.full_nvlink = full_nvlink

        if not _is_hip:


            self.meta_ptrs = self.create_shared_buffer(
                ops.meta_size() + max_size, group=group
            )


            self.buffer_ptrs = self.create_shared_buffer(max_size, group=group)


            self.rank_data = torch.empty(
                max_size, dtype=torch.uint8, device=self.device
            )
            self._ptr = ops.init_custom_ar(
                self.meta_ptrs, self.rank_data, rank, self.full_nvlink
            )
            ops.register_buffer(self._ptr, self.buffer_ptrs)
        else:

            self.meta = ops.allocate_meta_buffer(ops.meta_size() + max_size)
            self.buffer = torch.empty(max_size, dtype=torch.uint8, device=self.device)
            handle = ops.get_meta_buffer_ipc_handle(self.meta)
            shard_data = (
                bytes(handle),
                0,
            )
            handles, offsets = self._gather_ipc_meta(shard_data)
            self.rank_data = torch.empty(
                max_size, dtype=torch.uint8, device=self.device
            )
            self._ptr = ops.init_custom_ar(
                self.meta, self.rank_data, handles, offsets, rank, self.full_nvlink
            )
            self.register_buffer(self.buffer)

        self.disabled = False
        self.original_disabled = False
        self.tms_cudagraph = envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.get()

    @staticmethod
    def create_shared_buffer(
        size_in_bytes: int, group: Optional[ProcessGroup] = None
    ) -> List[int]:

        lib = CudaRTLibrary()
        pointer = lib.cudaMalloc(size_in_bytes)
        if _is_musa:
            lib.cudaMemset(pointer, 0, size_in_bytes)
        handle = lib.cudaIpcGetMemHandle(pointer)
        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        handles = [None] * world_size
        dist.all_gather_object(handles, handle, group=group)

        pointers: List[int] = []
        for i, h in enumerate(handles):
            if i == rank:
                pointers.append(pointer.value)  # type: ignore
            else:
                pointers.append(lib.cudaIpcOpenMemHandle(h).value)  # type: ignore

        return pointers

    @staticmethod
    def free_shared_buffer(
        pointers: List[int], group: Optional[ProcessGroup] = None
    ) -> None:
        rank = dist.get_rank(group=group)
        lib = CudaRTLibrary()
        lib.cudaFree(ctypes.c_void_p(pointers[rank]))

    @contextmanager
    def capture(self):

        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False
            if not self.disabled:
                self.register_graph_buffers()

    def _get_ipc_meta(self, inp: torch.Tensor):


        handle = ops.get_meta_buffer_ipc_handle(inp)
        shard_data = (
            bytes(handle),
            0,
        )
        return self._gather_ipc_meta(shard_data)

    def _gather_ipc_meta(self, shard_data):


        all_data: List[Optional[Any]] = [[None] for i in range(self.world_size)]
        all_data[self.rank][0] = shard_data

        ranks = dist.get_process_group_ranks(group=self.group)
        ranks.sort()
        for i, rank in enumerate(ranks):
            dist.broadcast_object_list(
                all_data[i], src=rank, group=self.group, device="cpu"
            )


        handles = []
        offsets = []
        for i in range(len(all_data)):
            handles.append(all_data[i][0][0])  # type: ignore
            offsets.append(all_data[i][0][1])  # type: ignore
        return handles, offsets

    def register_buffer(self, inp: torch.Tensor):
        handles, offsets = self._get_ipc_meta(inp)
        ops.register_buffer(self._ptr, inp, handles, offsets)

    def register_graph_buffers(self):
        if _is_hip:
            handle, offset = ops.get_graph_buffer_ipc_meta(self._ptr)
            handles, offsets = self._gather_ipc_meta((bytes(handle), offset))
            log_info_on_rank0(logger, f"Registering {len(offset)} cuda graph addresses")
            ops.register_graph_buffers(self._ptr, handles, offsets)
        else:
            handle, offset = ops.get_graph_buffer_ipc_meta(self._ptr)
            log_info_on_rank0(logger, f"Registering {len(offset)} cuda graph addresses")


            all_data = [
                [None, None] for _ in range(dist.get_world_size(group=self.group))
            ]
            all_data[self.rank] = [handle, offset]
            ranks = sorted(dist.get_process_group_ranks(group=self.group))
            for i, rank in enumerate(ranks):
                dist.broadcast_object_list(
                    all_data[i], src=rank, group=self.group, device="cpu"
                )

            handles = [d[0] for d in all_data]  # type: ignore
            offsets = [d[1] for d in all_data]  # type: ignore
            ops.register_graph_buffers(self._ptr, handles, offsets)

    def should_custom_ar(self, inp: torch.Tensor):
        if self.disabled:
            return False
        inp_size = inp.numel() * inp.element_size()

        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False


        if not _is_hip:
            if self.world_size == 2 or self.full_nvlink:
                return inp_size <= self.max_size
            return False

        if _is_hip:
            if self.use_amd_deterministic_impl:
                return True
            if self.full_nvlink:
                return inp_size <= self.max_size
            return False

        return False

    def _all_reduce_impl(self, inp: torch.Tensor, registered: bool):
        out = torch.empty_like(inp)
        if not _is_hip:
            if registered:
                ops.all_reduce(self._ptr, inp, out, 0, 0)
            else:
                ops.all_reduce(
                    self._ptr, inp, out, self.buffer_ptrs[self.rank], self.max_size
                )
        elif self.use_amd_deterministic_impl:
            inp_size = inp.numel() * inp.element_size()
            if inp_size < self.max_size:
                reg_buffer = self.buffer.view(inp.dtype)[: inp.numel()]
                ops.deterministic_all_reduce_unreg(self._ptr, inp, reg_buffer, out)
            else:
                self.register_buffer(inp)
                ops.deterministic_all_reduce_reg(self._ptr, inp, out)
        else:
            if registered:
                ops.all_reduce_reg(self._ptr, inp, out)
            else:
                ops.all_reduce_unreg(self._ptr, inp, self.buffer, out)
        return out

    def custom_all_reduce(self, input: torch.Tensor) -> Optional[torch.Tensor]:


        if self.disabled or not self.should_custom_ar(input):
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self._all_reduce_impl(input, registered=not self.tms_cudagraph)
            else:


                if is_in_tc_piecewise_cuda_graph():

                    return self._all_reduce_impl(input, registered=False)
                else:


                    return torch.zeros_like(input)
        else:
            return self._all_reduce_impl(input, registered=False)

    def close(self):
        if not self.disabled and self._ptr:
            if ops is not None:
                ops.dispose(self._ptr)
            if _is_cuda:
                self.free_shared_buffer(self.meta_ptrs)
                self.free_shared_buffer(self.buffer_ptrs)
            self._ptr = 0

    def __del__(self):
        self.close()


def dispatch_custom_allreduce(
    group: ProcessGroup,
    device: torch.device,
):

    if _is_cuda and envs.SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2.get():
        from .custom_all_reduce_v2 import (
            CustomAllReduceV2,
            can_use_custom_all_reduce_v2,
        )

        if can_use_custom_all_reduce_v2(group=group, device=device):
            logger.debug("[AR] Using CustomAllReduceV2 (JIT-compiled)")
            return CustomAllReduceV2

    if _is_cuda or _is_musa:
        return CustomAllreduce

    assert _is_hip

    if envs.SGLANG_USE_1STAGE_ALLREDUCE.is_set():
        if envs.SGLANG_USE_1STAGE_ALLREDUCE.get():
            logger.debug(
                "[AR] All-reduce: 1-stage kernel (SGLANG_USE_1STAGE_ALLREDUCE=1)"
            )
        else:
            logger.debug("[AR] All-reduce: default (SGLANG_USE_1STAGE_ALLREDUCE=0)")
    elif envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get():
        logger.debug(
            "[AR] All-reduce: 1-stage kernel (deterministic inference enabled)"
        )
    else:
        logger.debug("[AR] All-reduce: default")


    if _use_amd_deterministic_impl():
        return CustomAllreduce

    if get_bool_env_var("SGLANG_USE_AITER_AR", default="true"):
        try:
            from aiter.dist.device_communicators.custom_all_reduce import (
                CustomAllreduce as AiterCustomAllreduce,
            )

            logger.info("[AR] Using AiterCustomAllreduce (AMD default)")
            tms_cudagraph = envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.get()
            return partial(
                AiterCustomAllreduce,
                enable_register_for_capturing=not tms_cudagraph,
            )
        except ImportError as e:
            logger.warning(
                "[AR] Aiter custom all-reduce not available; "
                "falling back to sglang CustomAllreduce. Details: %s",
                e,
            )
            return CustomAllreduce

    return CustomAllreduce


def _use_amd_deterministic_impl() -> bool:
    if not _is_hip:
        return False
    if envs.SGLANG_USE_1STAGE_ALLREDUCE.is_set():
        return envs.SGLANG_USE_1STAGE_ALLREDUCE.get()
    else:
        return envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get()
