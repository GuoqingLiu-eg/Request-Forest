# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/device_communicators/custom_all_reduce_utils.py

import ctypes
import json
import logging
import os
import pickle
import subprocess
import sys
import tempfile
from functools import wraps
from itertools import product
from typing import Callable, Dict, List, Optional, Sequence, TypeVar

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing_extensions import ParamSpec

from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from sglang.srt.distributed.parallel_state import in_the_same_node_as
from sglang.srt.environ import envs as sglang_envs
from sglang.srt.utils import is_cuda, is_hip, is_musa
from sglang.srt.utils.cuda_vmm_utils import _gpu_fabric_clique

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_musa = is_musa()

if _is_cuda:
    try:
        import pynvml
    except ImportError as e:
        logger.warning("Failed to import pynvml with %r", e)

if _is_musa:
    try:
        import pymtml as pynvml
    except ImportError as e:
        logger.warning("Failed to import pymtml with %r", e)

if _is_hip:
    try:
        from amdsmi import (
            AmdSmiException,
            amdsmi_get_processor_handles,
            amdsmi_init,
            amdsmi_shut_down,
            amdsmi_topo_get_link_type,
        )
    except ImportError as e:
        logger.warning("Failed to import amdsmi with %r", e)

_P = ParamSpec("_P")
_R = TypeVar("_R")


def update_environment_variables(envs: Dict[str, str]):
    for k, v in envs.items():
        if k in os.environ and os.environ[k] != v:
            logger.warning(
                "Overwriting environment variable %s " "from '%s' to '%s'",
                k,
                os.environ[k],
                v,
            )
        os.environ[k] = v


def producer(
    batch_src: Sequence[int],
    producer_queue,
    consumer_queue,
    result_queue,
    cuda_visible_devices: Optional[str] = None,
):
    if cuda_visible_devices is not None:
        update_environment_variables({"CUDA_VISIBLE_DEVICES": cuda_visible_devices})

    lib = CudaRTLibrary()
    for i in batch_src:
        lib.cudaSetDevice(i)
        pointer = lib.cudaMalloc(1024)
        lib.cudaMemset(pointer, 1, 1024)
        lib.cudaDeviceSynchronize()
        handle = lib.cudaIpcGetMemHandle(pointer)
        producer_queue.put(handle)
        open_success = consumer_queue.get()
        if open_success:

            producer_queue.put(0)
            consumer_queue.get()

            host_data = (ctypes.c_char * 1024)()
            lib.cudaMemcpy(host_data, pointer, 1024)  # type: ignore
            for i in range(1024):
                if ord(host_data[i]) != 2:
                    open_success = False
                    break
        result_queue.put(open_success)
        lib.cudaDeviceReset()


def consumer(
    batch_tgt: Sequence[int],
    producer_queue,
    consumer_queue,
    result_queue,
    cuda_visible_devices: Optional[str] = None,
):
    if cuda_visible_devices is not None:
        update_environment_variables({"CUDA_VISIBLE_DEVICES": cuda_visible_devices})

    lib = CudaRTLibrary()
    for j in batch_tgt:
        lib.cudaSetDevice(j)
        handle = producer_queue.get()
        open_success = False
        try:
            pointer = lib.cudaIpcOpenMemHandle(handle)  # type: ignore
            open_success = True
        except RuntimeError:


            pass
        consumer_queue.put(open_success)
        if open_success:

            lib.cudaMemset(pointer, 2, 1024)
            lib.cudaDeviceSynchronize()

            producer_queue.get()
            consumer_queue.put(0)

            host_data = (ctypes.c_char * 1024)()
            lib.cudaMemcpy(host_data, pointer, 1024)  # type: ignore
            for i in range(1024):
                if ord(host_data[i]) != 2:
                    open_success = False
                    break
        result_queue.put(open_success)
        lib.cudaDeviceReset()


def can_actually_p2p(
    batch_src: Sequence[int],
    batch_tgt: Sequence[int],
) -> Sequence[bool]:

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)


    smp = mp.get_context("spawn")
    producer_queue = smp.Queue()
    consumer_queue = smp.Queue()
    result_queue = smp.Queue()
    p_src = smp.Process(
        target=producer,
        args=(
            batch_src,
            producer_queue,
            consumer_queue,
            result_queue,
            cuda_visible_devices,
        ),
    )
    p_tgt = smp.Process(
        target=consumer,
        args=(
            batch_tgt,
            producer_queue,
            consumer_queue,
            result_queue,
            cuda_visible_devices,
        ),
    )
    p_src.start()
    p_tgt.start()
    p_src.join()
    p_tgt.join()
    assert p_src.exitcode == 0 and p_tgt.exitcode == 0
    result: List[bool] = []
    for src, tgt in zip(batch_src, batch_tgt):
        a = result_queue.get()
        b = result_queue.get()
        if a != b:
            logger.warning(
                "Two processes do not agree on the P2P access"
                " status on %d -> %d, treat as disabled.",
                src,
                tgt,
            )
            result.append(False)
        else:
            result.append(a)
    return result


_gpu_p2p_access_cache: Optional[Dict[str, bool]] = None


def gpu_p2p_access_check(src: int, tgt: int) -> bool:


    global _gpu_p2p_access_cache
    if _gpu_p2p_access_cache is not None:
        return _gpu_p2p_access_cache[f"{src}->{tgt}"]

    is_distributed = dist.is_initialized()

    num_dev = torch.cuda.device_count()
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible_devices is None:
        cuda_visible_devices = ",".join(str(i) for i in range(num_dev))


    SGLANG_CACHE_ROOT = os.path.expanduser(sglang_envs.SGLANG_CACHE_DIR.get())
    path = os.path.join(
        SGLANG_CACHE_ROOT, f"gpu_p2p_access_cache_for_{cuda_visible_devices}.json"
    )
    cache_dir = os.path.dirname(path)
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except (FileExistsError, NotADirectoryError):
        if not os.path.isdir(cache_dir):

            try:
                os.remove(cache_dir)
            except OSError:
                pass
            os.makedirs(cache_dir, exist_ok=True)
    from sglang.srt.distributed.parallel_state import get_world_group

    if (not is_distributed or get_world_group().local_rank == 0) and (
        not os.path.exists(path)
    ):


        logger.info("generating GPU P2P access cache in %s", path)
        cache: Dict[str, bool] = {}
        ids = list(range(num_dev))

        batch_src, batch_tgt = zip(*list(product(ids, ids)))


        with tempfile.NamedTemporaryFile() as output_file:
            input_bytes = pickle.dumps((batch_src, batch_tgt, output_file.name))
            returned = subprocess.run(
                [sys.executable, __file__], input=input_bytes, capture_output=True
            )

            try:
                returned.check_returncode()
            except Exception as e:

                raise RuntimeError(
                    f"Error happened when batch testing "
                    f"peer-to-peer access from {batch_src} to {batch_tgt}:\n"
                    f"{returned.stderr.decode()}"
                ) from e
            with open(output_file.name, "rb") as f:
                result = pickle.load(f)
        for _i, _j, r in zip(batch_src, batch_tgt, result):
            cache[f"{_i}->{_j}"] = r
        with open(path, "w") as f:
            json.dump(cache, f, indent=4)
    if is_distributed:
        get_world_group().barrier()
    logger.info("reading GPU P2P access cache from %s", path)
    with open(path) as f:
        cache = json.load(f)
    _gpu_p2p_access_cache = cache
    return _gpu_p2p_access_cache[f"{src}->{tgt}"]


def with_nvml_context(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    @wraps(fn)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        if _is_hip:
            try:
                amdsmi_init()
                return fn(*args, **kwargs)
            finally:
                amdsmi_shut_down()
        else:
            pynvml.nvmlInit()
            try:
                return fn(*args, **kwargs)
            finally:
                pynvml.nvmlShutdown()

    return wrapper


@with_nvml_context
def is_full_nvlink(physical_device_ids: List[int], world_size: int) -> bool:
    if _is_hip:
        """
        query if the set of gpus are fully connected by xgmi (1 hop)
        """
        handles = [amdsmi_get_processor_handles()[i] for i in physical_device_ids]
        for i, handle in enumerate(handles):
            for j, peer_handle in enumerate(handles):
                if i < j:
                    try:
                        link_type = amdsmi_topo_get_link_type(handle, peer_handle)

                        if link_type["hops"] != 1 or link_type["type"] != 2:
                            return False
                    except AmdSmiException as error:
                        logger.error("AMD 1 hop XGMI detection failed.", exc_info=error)
                        return False
        return True
    else:
        """
        query if the set of gpus are fully connected by nvlink (1 hop)
        """
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_ids]
        for i, handle in enumerate(handles):
            for j, peer_handle in enumerate(handles):
                if i < j:
                    try:
                        p2p_status = pynvml.nvmlDeviceGetP2PStatus(
                            handle, peer_handle, pynvml.NVML_P2P_CAPS_INDEX_NVLINK
                        )
                        if p2p_status != pynvml.NVML_P2P_STATUS_OK:
                            return False
                    except pynvml.NVMLError:
                        logger.exception(
                            "NVLink detection failed. This is normal if your"
                            " machine has no NVLink equipped."
                        )
                        return False
        return True


@with_nvml_context
def is_one_nvlink_clique(
    group: torch.distributed.ProcessGroup, device: torch.device
) -> bool:

    if _is_hip:
        return False
    try:
        clique = _gpu_fabric_clique(device)
    except Exception as e:
        logger.warning(
            "GPU fabric clique query failed (%r); custom-AR stays intra-node.", e
        )
        clique = None


    world_size = dist.get_world_size(group=group)
    gathered: List[object] = [None] * world_size
    dist.all_gather_object(gathered, clique, group=group)
    if any(c is None for c in gathered):
        return False
    return len(set(gathered)) == 1


def is_weak_contiguous(inp: torch.Tensor):
    return inp.is_contiguous() or (
        inp.storage().nbytes() - inp.storage_offset() * inp.element_size()
        == inp.numel() * inp.element_size()
    )


def can_p2p(rank: int, world_size: int) -> bool:

    SGLANG_SKIP_P2P_CHECK = os.getenv("SGLANG_SKIP_P2P_CHECK", "0") == "1"
    for i in range(world_size):
        if i == rank:
            continue
        if SGLANG_SKIP_P2P_CHECK:
            logger.info("Skipping P2P check and trusting the driver's P2P report.")
            return torch.cuda.can_device_access_peer(rank, i)
        if not gpu_p2p_access_check(rank, i):
            return False
    return True


def can_use_custom_all_reduce_with_nvlink(
    group: torch.distributed.ProcessGroup,
    device: torch.device,
    supported_world_size: List[int],
    cls_name: str,
) -> Optional[bool]:
    assert (
        dist.get_backend(group) != dist.Backend.NCCL
    ), f"{cls_name} should be attached to a non-NCCL group."

    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)


    if world_size == 1:
        return


    if not all(in_the_same_node_as(group, source_rank=0)):
        logger.warning(
            f"{cls_name} is disabled because this process group" " spans across nodes."
        )
        return


    if world_size not in supported_world_size:
        logger.warning(
            f"{cls_name} is disabled due to an unsupported world"
            f" size: {world_size}. Supported world sizes: {supported_world_size}. "
            "To silence this warning, specify disable_custom_all_reduce=True explicitly.",
        )
        return

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible_devices:
        device_ids = list(map(int, cuda_visible_devices.split(",")))
    else:
        device_ids = list(range(torch.cuda.device_count()))
    physical_device_id = device_ids[device.index]
    tensor = torch.tensor([physical_device_id], dtype=torch.int, device="cpu")
    gather_list = [
        torch.tensor([0], dtype=torch.int, device="cpu") for _ in range(world_size)
    ]
    dist.all_gather(gather_list, tensor, group=group)
    physical_device_ids = [int(t) for t in gather_list]
    full_nvlink = is_full_nvlink(physical_device_ids, world_size)


    if world_size > 2 and not full_nvlink:
        logger.warning(
            f"{cls_name} is disabled because it's not supported on"
            " more than two PCIe-only GPUs. To silence this warning, "
            "specify disable_custom_all_reduce=True explicitly."
        )
        return


    if not _is_hip and not can_p2p(rank, world_size):
        logger.warning(
            f"{cls_name} is disabled because your platform lacks "
            "GPU P2P capability or P2P test failed. To silence this "
            "warning, specify disable_custom_all_reduce=True explicitly."
        )
        return

    return full_nvlink


if __name__ == "__main__":
    batch_src, batch_tgt, output_file = pickle.loads(sys.stdin.buffer.read())
    result = can_actually_p2p(batch_src, batch_tgt)
    with open(output_file, "wb") as f:
        f.write(pickle.dumps(result))
