import inspect
import logging
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed import (
    get_attn_tp_group,
    get_moe_ep_group,
    get_moe_tp_group,
    get_tp_group,
)
from sglang.srt.distributed.parallel_state import in_the_same_node_as
from sglang.srt.runtime_context import get_exec, get_parallel
from sglang.srt.utils import (
    ceil_align,
    get_cuda_driver_bindings,
    is_flashinfer_available,
    is_sm90_supported,
    is_sm100_supported,
)
from sglang.srt.utils.custom_op import register_custom_op

logger = logging.getLogger(__name__)


_flashinfer_comm = None
_TorchDistBackend = None
_mnnvl_comm_backend = None
_create_allreduce_fusion_workspace = None
_flashinfer_allreduce_unavailable = False
_flashinfer_create_workspace_supports_group = False
_flashinfer_create_workspace_supports_comm_backend = False
_flashinfer_allreduce_supports_trigger_completion = False


def _mnnvl_supported(is_multi_node: bool) -> bool:

    if is_sm100_supported():
        return True
    return is_sm90_supported() and not is_multi_node


def _resolve_backend(backend: str, is_multi_node: bool = False) -> str:

    if not (is_sm90_supported() or is_sm100_supported()):
        raise ValueError(
            "FlashInfer allreduce fusion requires SM90 or SM10X NVIDIA GPUs."
        )

    if backend == "auto":
        if is_multi_node:
            if is_sm100_supported():
                return "mnnvl"
            raise ValueError(
                "FlashInfer allreduce fusion does not support multi-node on "
                "non-Blackwell systems."
            )
        if is_sm100_supported():
            return "mnnvl"
        return "trtllm"

    if backend == "trtllm" and is_multi_node:
        raise ValueError(
            "FlashInfer allreduce fusion trtllm backend supports single-node only."
        )

    if backend == "mnnvl" and not _mnnvl_supported(is_multi_node):
        raise ValueError(
            "FlashInfer allreduce fusion mnnvl backend requires a Blackwell "
            "system, or SM90 single-node."
        )
    return backend


def resolve_flashinfer_allreduce_fusion_backend() -> Optional[str]:

    backend = get_exec().comm.flashinfer_allreduce_fusion_backend
    if backend is None:
        return None
    return _resolve_backend(backend, get_parallel().config.nnodes > 1)


if is_flashinfer_available():
    try:
        import flashinfer.comm as comm

        if hasattr(comm, "allreduce_fusion") and hasattr(
            comm, "create_allreduce_fusion_workspace"
        ):
            _flashinfer_comm = comm
            _create_allreduce_fusion_workspace = comm.create_allreduce_fusion_workspace
            workspace_params = inspect.signature(
                comm.create_allreduce_fusion_workspace
            ).parameters
            allreduce_params = inspect.signature(comm.allreduce_fusion).parameters
            _flashinfer_create_workspace_supports_group = "group" in workspace_params
            _flashinfer_create_workspace_supports_comm_backend = (
                "comm_backend" in workspace_params
            )
            _flashinfer_allreduce_supports_trigger_completion = (
                "trigger_completion_at_end" in allreduce_params
            )
        else:
            _flashinfer_allreduce_unavailable = True
            logger.warning(
                "flashinfer.comm unified allreduce_fusion API is not available, "
                "falling back to standard implementation"
            )
    except (ImportError, AttributeError) as e:
        _flashinfer_allreduce_unavailable = True
        logger.warning(
            "flashinfer.comm allreduce_fusion API is not available (%s), "
            "falling back to standard implementation",
            e,
        )

    try:
        from flashinfer.comm.mnnvl import TorchDistBackend

        class _FixedTorchDistBackend(TorchDistBackend):


            def __init__(self, device_group, cpu_group):
                super().__init__(group=device_group)
                self._cpu_group = cpu_group

            def bcast(self, data, root):
                import torch.distributed as dist

                group_ranks = dist.get_process_group_ranks(self._cpu_group)
                global_root = group_ranks[root]
                object_list = [data]
                dist.broadcast_object_list(
                    object_list, src=global_root, group=self._cpu_group
                )
                return object_list[0]

        _TorchDistBackend = _FixedTorchDistBackend
    except ImportError:
        logger.debug(
            "flashinfer.comm.mnnvl.TorchDistBackend is not available, "
            "allreduce fusion will use the default process group"
        )

    try:
        from flashinfer.comm.mnnvl import CommBackend

        class TorchDistributedCommBackend(CommBackend):


            def __init__(self, group: ProcessGroup):
                self._group = group

            def Get_rank(self) -> int:
                return self._group.rank()

            def Get_size(self) -> int:
                return self._group.size()

            def allgather(self, data: int):
                gathered = [None] * self.Get_size()
                dist.all_gather_object(gathered, data, group=self._group)
                return gathered

            def bcast(self, data, root: int = 0):

                obj_list = [data]
                dist.broadcast_object_list(obj_list, src=root, group=self._group)
                return obj_list[0]

            def barrier(self):
                dist.barrier(group=self._group)

            def Split(self, color: int, key: int):

                return self._group

        _mnnvl_comm_backend = TorchDistributedCommBackend
    except ImportError:
        _mnnvl_comm_backend = None


def is_flashinfer_allreduce_unavailable() -> bool:
    return _flashinfer_allreduce_unavailable


def _make_flashinfer_workspace_allocation_prop(cuda_driver):
    from flashinfer.comm.mnnvl import is_mnnvl_fabric_supported

    handle_types = cuda_driver.CUmemAllocationHandleType
    if is_mnnvl_fabric_supported(torch.cuda.current_device()):
        handle_type = handle_types.CU_MEM_HANDLE_TYPE_FABRIC
    else:
        handle_type = handle_types.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR

    prop = cuda_driver.CUmemAllocationProp()
    prop.requestedHandleTypes = handle_type
    prop.type = cuda_driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location = cuda_driver.CUmemLocation()
    prop.location.type = cuda_driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = torch.cuda.current_device()
    prop.allocFlags.gpuDirectRDMACapable = 1
    return prop


def _flashinfer_trtllm_workspace_allocation_sizes(
    cuda_driver,
    prop,
    world_size: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
) -> list[int]:

    elem_size = 4 if dtype == torch.float32 else 2
    buffer_size = world_size * max_token_num * hidden_dim * 2
    flag_size = world_size * 256 * 4

    max_comm_size = 2147483647 & ~((1 << 21) - 1)
    lamport_comm_size = min(
        world_size * max_token_num * hidden_dim * elem_size,
        max_comm_size,
    )
    lamport_buffer_size = lamport_comm_size * 3


    buffer_sizes = (
        ceil_align(size, 1 << 21)
        for size in (buffer_size, flag_size, lamport_buffer_size)
    )

    signal_pad_size = 2048
    allocation_sizes = []
    for buffer_size in buffer_sizes:
        err, alloc_granularity = cuda_driver.cuMemGetAllocationGranularity(
            prop,
            cuda_driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
        )
        if err != cuda_driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(
                "cuMemGetAllocationGranularity failed for FlashInfer "
                f"workspace preflight: {err}"
            )

        allocation_size = ceil_align(buffer_size + signal_pad_size, alloc_granularity)

        mc_prop = cuda_driver.CUmulticastObjectProp()
        mc_prop.numDevices = world_size
        mc_prop.size = allocation_size
        mc_prop.handleTypes = prop.requestedHandleTypes

        err, mc_granularity = cuda_driver.cuMulticastGetGranularity(
            mc_prop,
            cuda_driver.CUmulticastGranularity_flags.CU_MULTICAST_GRANULARITY_RECOMMENDED,
        )
        if err != cuda_driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(
                "cuMulticastGetGranularity failed for FlashInfer "
                f"workspace preflight: {err}"
            )

        allocation_size = ceil_align(allocation_size, mc_granularity)
        allocation_sizes.append(allocation_size)
    return allocation_sizes


def _probe_cumem_create_sequence(cuda_driver, allocation_sizes, prop) -> bool:
    handles = []
    try:
        for allocation_size in allocation_sizes:
            err, handle = cuda_driver.cuMemCreate(allocation_size, prop, 0)
            if err != cuda_driver.CUresult.CUDA_SUCCESS:
                return False
            handles.append(handle)
        return True
    finally:
        for handle in reversed(handles):
            cuda_driver.cuMemRelease(handle)


def _preflight_check_workspace_memory(
    world_size: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
    cpu_group: Optional["torch.distributed.ProcessGroup"] = None,
) -> bool:

    import torch.distributed as dist

    group = cpu_group
    if group is None:
        tp_group = get_tp_group()
        if tp_group.world_size <= 1:
            return True
        group = tp_group.cpu_group

    allocation_sizes = []
    try:
        cuda_driver = get_cuda_driver_bindings()
        prop = _make_flashinfer_workspace_allocation_prop(cuda_driver)
        allocation_sizes = _flashinfer_trtllm_workspace_allocation_sizes(
            cuda_driver,
            prop,
            world_size,
            max_token_num,
            hidden_dim,
            dtype,
        )
        local_ok = _probe_cumem_create_sequence(cuda_driver, allocation_sizes, prop)
    except Exception as e:
        logger.warning(
            "FlashInfer workspace preflight probe failed (%s). "
            "Skipping allreduce fusion.",
            e,
        )
        local_ok = False

    flag = torch.tensor([1 if local_ok else 0], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.BAND, group=group)

    logger.debug(
        "FlashInfer workspace preflight [rank %s]: probe=%.2f GB, "
        "local_probe=%s, vote=%s",
        dist.get_rank(group=group),
        sum(allocation_sizes) / 1e9,
        "OK" if local_ok else "FAIL",
        "PROCEED" if flag.item() == 1 else "SKIP",
    )
    if flag.item() == 0:
        logger.warning(
            "FlashInfer workspace preflight: cuMemCreate probe failed on at "
            "least one rank. Skipping allreduce fusion to avoid cross-rank "
            "desync inside the flashinfer collective."
        )
        return False
    return True


class FlashInferWorkspaceManager:


    def __init__(self):
        self.workspace = None
        self.world_size = None
        self.rank = None
        self.group = None
        self.max_token_num = None
        self.hidden_dim = None
        self.dtype = None
        self.backend = None
        self.use_fp32_lamport = None
        self.initialized = False

        self._max_token_num_seen: Optional[int] = None
        self._max_hidden_dim_seen: Optional[int] = None
        self._logged_init = False
        self._workspace_size_check_kwarg = None
        self._workspace_size_check_strategy_type = None

    def _configure_workspace_size_check(self):

        size_check_params = inspect.signature(
            self.workspace.is_buffer_size_sufficient
        ).parameters
        if "use_oneshot" in size_check_params:
            self._workspace_size_check_kwarg = "use_oneshot"
            self._workspace_size_check_strategy_type = None
        elif "strategy" in size_check_params:
            strategy_default = size_check_params["strategy"].default
            self._workspace_size_check_kwarg = "strategy"
            self._workspace_size_check_strategy_type = type(strategy_default)
        else:
            self._workspace_size_check_kwarg = None
            self._workspace_size_check_strategy_type = None

    def initialize(
        self,
        world_size: int,
        rank: int,
        max_token_num: int,
        hidden_dim: int,
        backend: str = "auto",
        group: Optional[ProcessGroup] = None,
        use_fp32_lamport: bool = False,
        dtype: Optional[torch.dtype] = None,
        use_oneshot: Optional[bool] = None,
        device_group: Optional["torch.distributed.ProcessGroup"] = None,
        cpu_group: Optional["torch.distributed.ProcessGroup"] = None,
    ):

        global _flashinfer_allreduce_unavailable


        self._max_token_num_seen = max(max_token_num, self._max_token_num_seen or 0)
        self._max_hidden_dim_seen = max(hidden_dim, self._max_hidden_dim_seen or 0)


        if (
            self.initialized
            and self.world_size == world_size
            and self.is_buffer_size_sufficient(
                token_num=max_token_num,
                hidden_dim=hidden_dim,
                dtype=dtype or torch.bfloat16,
                use_oneshot=use_oneshot,
            )
        ):
            return


        if self.initialized and self.world_size == world_size:
            self.cleanup()

        if _flashinfer_comm is None or _create_allreduce_fusion_workspace is None:
            logger.warning(
                "FlashInfer comm not available, skipping workspace initialization"
            )
            return

        self.cleanup()

        if not _preflight_check_workspace_memory(
            world_size=world_size,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            dtype=dtype,
            cpu_group=cpu_group,
        ):
            _flashinfer_allreduce_unavailable = True
            self.workspace = None
            self.initialized = False
            return


        gpus_per_node = None
        node_pg = cpu_group if cpu_group is not None else group
        if node_pg is not None:
            gpus_per_node = sum(in_the_same_node_as(node_pg, source_rank=0))
        comm_backend = None
        if (
            _TorchDistBackend is not None
            and device_group is not None
            and cpu_group is not None
        ):
            comm_backend = _TorchDistBackend(
                device_group=device_group, cpu_group=cpu_group
            )
        elif _mnnvl_comm_backend is not None and group is not None:
            comm_backend = _mnnvl_comm_backend(group)

        try:
            alloc_token_num = max(max_token_num, self._max_token_num_seen or 0)
            alloc_hidden_dim = max(hidden_dim, self._max_hidden_dim_seen or 0)
            create_kw = dict(
                backend=backend,
                world_size=world_size,
                rank=rank,
                max_token_num=alloc_token_num,
                hidden_dim=alloc_hidden_dim,
                dtype=dtype or torch.bfloat16,
                gpus_per_node=gpus_per_node,
            )
            if (
                _flashinfer_create_workspace_supports_comm_backend
                and comm_backend is not None
            ):
                create_kw["comm_backend"] = comm_backend
            if _flashinfer_create_workspace_supports_group:


                create_kw["group"] = device_group
            if use_oneshot is not None:
                create_kw["force_oneshot_support"] = bool(use_oneshot)
            if use_fp32_lamport:
                create_kw["use_fp32_lamport"] = True
            self.workspace = _create_allreduce_fusion_workspace(**create_kw)
            self._configure_workspace_size_check()
            self.world_size = world_size
            self.rank = rank
            self.group = (device_group, cpu_group)
            self.max_token_num = alloc_token_num
            self.hidden_dim = alloc_hidden_dim
            self.dtype = dtype or torch.bfloat16
            self.backend = self.workspace.backend
            self.use_fp32_lamport = (
                self.workspace.metadata["use_fp32_lamport"]
                if self.backend == "trtllm"
                else None
            )
            self.initialized = True

            if not self._logged_init:
                logger.info(
                    f"FlashInfer AllReduce Fusion enabled and workspace initialized: "
                    f"backend={self.backend}, rank={rank}, world_size={world_size}, "
                    f"max_token_num={self.max_token_num}, hidden_dim={self.hidden_dim}"
                )
                self._logged_init = True
            else:
                logger.debug(
                    f"FlashInfer workspace re-initialized: backend={self.backend}, "
                    f"rank={rank}, world_size={world_size}"
                )
        except Exception as e:
            _flashinfer_allreduce_unavailable = True
            logger.warning(
                f"Failed to initialize FlashInfer workspace (backend={backend}): {e}. "
                "Disabling flashinfer allreduce fusion permanently."
            )
            self.workspace = None
            self._workspace_size_check_kwarg = None
            self._workspace_size_check_strategy_type = None
            self.initialized = False
            return

    def is_buffer_size_sufficient(
        self,
        token_num: int,
        hidden_dim: int,
        dtype: torch.dtype,
        use_oneshot: Optional[bool] = None,
    ) -> bool:
        if not self.initialized or self.workspace is None:
            return False
        try:
            check_kw = dict(
                tp_size=self.world_size,
                num_tokens=token_num,
                hidden_dim=hidden_dim,
                dtype=dtype,
            )
            if self._workspace_size_check_kwarg == "use_oneshot":
                check_kw["use_oneshot"] = use_oneshot
            elif (
                self._workspace_size_check_kwarg == "strategy"
                and use_oneshot is not None
            ):


                check_kw["strategy"] = getattr(
                    self._workspace_size_check_strategy_type,
                    "ONESHOT" if use_oneshot else "TWOSHOT",
                )
            return self.workspace.is_buffer_size_sufficient(**check_kw)
        except Exception as e:
            logger.debug(f"FlashInfer workspace size check failed: {e}")


            if (
                self.max_token_num is not None
                and self.hidden_dim is not None
                and token_num <= self.max_token_num
                and hidden_dim <= self.hidden_dim
            ):
                return True
            return False

    def cleanup(self):

        if self.workspace is not None:
            try:
                if hasattr(self.workspace, "destroy"):
                    self.workspace.destroy()
            except Exception as e:
                logger.warning(f"Failed to cleanup FlashInfer workspace: {e}")
            finally:
                self.workspace = None
                self._workspace_size_check_kwarg = None
                self._workspace_size_check_strategy_type = None
                self.initialized = False
                self.world_size = None
                self.rank = None
                self.group = None
                self.max_token_num = None
                self.hidden_dim = None
                self.dtype = None
                self.backend = None
                self.use_fp32_lamport = None
                self._logged_init = False


def _get_workspace_manager(use_attn_tp_group: bool) -> FlashInferWorkspaceManager:

    from sglang.srt.runtime_context import get_resources

    buffers = get_resources().buffers
    name = (
        "flashinfer_fusion_attn_tp_workspace"
        if use_attn_tp_group
        else "flashinfer_fusion_moe_tp_workspace"
    )
    manager = buffers.get(name)
    if manager is None:
        manager = FlashInferWorkspaceManager()
        buffers[name] = manager
    return manager


def _sync_allreduce_unavailable_across_tp():

    global _flashinfer_allreduce_unavailable
    try:
        import torch.distributed as dist

        tp_group = get_tp_group()
        if tp_group.world_size <= 1:
            return
        flag = torch.tensor(
            [1 if _flashinfer_allreduce_unavailable else 0],
            dtype=torch.int32,
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=tp_group.cpu_group)
        if flag.item() > 0 and not _flashinfer_allreduce_unavailable:
            _flashinfer_allreduce_unavailable = True
            logger.warning(
                "FlashInfer allreduce fusion disabled globally because "
                "workspace initialization failed on at least one rank."
            )
    except Exception as e:
        logger.debug(f"Failed to sync flashinfer unavailable flag: {e}")


def ensure_workspace_initialized(
    max_token_num: int = 2048,
    hidden_dim: int = 4096,
    use_fp32_lamport: bool = False,
    dtype: Optional[torch.dtype] = None,
    token_num: Optional[int] = None,
    use_oneshot: Optional[bool] = None,
    use_attn_tp_group: bool = True,
):

    if _flashinfer_allreduce_unavailable:
        return False

    if not is_flashinfer_available() or _flashinfer_comm is None:
        return False

    if use_attn_tp_group:
        world_size = get_parallel().attn_tp_size
        rank = get_parallel().attn_tp_rank
        coordinator = get_attn_tp_group()
    else:
        if get_parallel().moe_ep_size > 1:
            world_size = get_parallel().moe_ep_size
            rank = get_parallel().moe_ep_rank
            coordinator = get_moe_ep_group()
        else:
            world_size = get_parallel().moe_tp_size
            rank = get_parallel().moe_tp_rank
            coordinator = get_moe_tp_group()


    device_group = coordinator.device_group
    cpu_group = coordinator.cpu_group

    if world_size <= 1:
        return False

    workspace_manager = _get_workspace_manager(use_attn_tp_group)
    token_num = token_num or max_token_num
    group_key = (device_group, cpu_group)
    effective_dtype = dtype or torch.bfloat16
    backend = resolve_flashinfer_allreduce_fusion_backend()
    if backend is None:
        return False

    if (
        not workspace_manager.initialized
        or workspace_manager.world_size != world_size
        or workspace_manager.rank != rank
        or workspace_manager.group != group_key
        or not workspace_manager.is_buffer_size_sufficient(
            token_num=token_num,
            hidden_dim=hidden_dim,
            dtype=effective_dtype,
            use_oneshot=use_oneshot,
        )
    ):
        workspace_manager.initialize(
            world_size=world_size,
            rank=rank,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            backend=backend,
            group=cpu_group,
            use_fp32_lamport=use_fp32_lamport,
            dtype=dtype,
            use_oneshot=use_oneshot,
            device_group=device_group,
            cpu_group=cpu_group,
        )

        _sync_allreduce_unavailable_across_tp()

    return workspace_manager.initialized


def fake_flashinfer_allreduce_residual_rmsnorm(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    max_token_num: int = 16384,
    use_oneshot: Optional[bool] = None,
    trigger_completion_at_end: bool = False,
    fp32_acc: bool = False,
    use_attn_tp_group: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    residual_out = torch.empty_like(residual)
    norm_out = torch.empty_like(input_tensor)
    return norm_out, residual_out


@register_custom_op(
    mutates_args=["input_tensor", "residual", "weight"],
    fake_impl=fake_flashinfer_allreduce_residual_rmsnorm,
)
def flashinfer_allreduce_residual_rmsnorm(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    max_token_num: int = 2048,
    use_oneshot: Optional[bool] = None,
    trigger_completion_at_end: bool = False,
    fp32_acc: bool = False,
    use_attn_tp_group: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:

    if not is_flashinfer_available() or _flashinfer_comm is None:
        logger.debug(
            "FlashInfer not available, falling back to standard implementation"
        )
        return None, None

    if use_attn_tp_group:
        world_size = get_parallel().attn_tp_size
    else:
        if get_parallel().moe_ep_size > 1:
            world_size = get_parallel().moe_ep_size
        else:
            world_size = get_parallel().moe_tp_size

    if world_size <= 1:
        logger.debug("Single GPU, no need for allreduce fusion")
        return None, None

    assert input_tensor.shape[0] <= max_token_num
    if (
        not input_tensor.is_contiguous()
        or not residual.is_contiguous()
        or not weight.is_contiguous()
    ):
        logger.debug("Non-contiguous tensors, skipping FlashInfer allreduce fusion")
        return None, None

    if not ensure_workspace_initialized(
        max_token_num=max_token_num,
        hidden_dim=input_tensor.shape[-1],
        use_fp32_lamport=(input_tensor.dtype == torch.float32),
        dtype=input_tensor.dtype,
        token_num=input_tensor.shape[0],
        use_oneshot=use_oneshot,
        use_attn_tp_group=use_attn_tp_group,
    ):
        logger.debug("FlashInfer workspace not available")
        return None, None

    workspace_manager = _get_workspace_manager(use_attn_tp_group)
    if workspace_manager.workspace is None:
        logger.debug("FlashInfer workspace is None")
        return None, None

    residual_out = torch.empty_like(residual)
    norm_out = torch.empty_like(input_tensor)

    kwargs = dict(
        input=input_tensor,
        workspace=workspace_manager.workspace,
        pattern=_flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm,
        launch_with_pdl=True,
        residual_out=residual_out,
        norm_out=norm_out,
        residual_in=residual,
        rms_gamma=weight,
        rms_eps=eps,
        use_oneshot=use_oneshot,
        fp32_acc=fp32_acc,
    )
    if _flashinfer_allreduce_supports_trigger_completion:
        kwargs["trigger_completion_at_end"] = trigger_completion_at_end
    _flashinfer_comm.allreduce_fusion(**kwargs)

    return norm_out, residual_out


def can_use_flashinfer_allreduce(
    input_: torch.Tensor,
    *,
    use_attn_tp_group: bool,
    expected_world_size: int,
    expected_group_key: Tuple[Optional[ProcessGroup], Optional[ProcessGroup]],
) -> bool:

    if _flashinfer_allreduce_unavailable or _flashinfer_comm is None:
        return False

    if input_.ndim != 2 or not input_.is_contiguous():
        return False

    workspace_manager = _get_workspace_manager(use_attn_tp_group)
    if not workspace_manager.initialized or workspace_manager.workspace is None:
        return False


    if (
        workspace_manager.world_size != expected_world_size
        or workspace_manager.group != expected_group_key
    ):
        return False


    token_num, hidden_dim = input_.shape
    if torch.compiler.is_compiling():


        return (
            workspace_manager.max_token_num is not None
            and workspace_manager.hidden_dim is not None
            and workspace_manager.dtype is not None
            and token_num <= workspace_manager.max_token_num
            and hidden_dim <= workspace_manager.hidden_dim
            and workspace_manager.dtype == input_.dtype
        )


    if workspace_manager.backend == "trtllm" and (
        token_num * hidden_dim
        > workspace_manager.max_token_num * workspace_manager.hidden_dim
        or workspace_manager.use_fp32_lamport != (input_.dtype == torch.float32)
    ):
        return False

    return workspace_manager.is_buffer_size_sufficient(
        token_num=token_num,
        hidden_dim=hidden_dim,
        dtype=input_.dtype,
    )


def flashinfer_allreduce(
    input_: torch.Tensor,
    *,
    use_attn_tp_group: bool,
) -> torch.Tensor:

    workspace_manager = _get_workspace_manager(use_attn_tp_group)

    output = torch.empty_like(input_)
    kwargs = dict(
        input=input_,
        workspace=workspace_manager.workspace,
        pattern=_flashinfer_comm.AllReduceFusionPattern.kAllReduce,
        launch_with_pdl=True,
        fp32_acc=False,
        output=output,
    )
    if _flashinfer_allreduce_supports_trigger_completion:
        kwargs["trigger_completion_at_end"] = False
    _flashinfer_comm.allreduce_fusion(**kwargs)
    return output


def pre_initialize_workspaces(
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
    use_oneshot: Optional[bool] = None,
):

    if _flashinfer_allreduce_unavailable or _flashinfer_comm is None:
        return


    ensure_workspace_initialized(
        max_token_num=max_token_num,
        hidden_dim=hidden_dim,
        dtype=dtype,
        use_oneshot=use_oneshot,
        use_attn_tp_group=False,
    )


    ensure_workspace_initialized(
        max_token_num=max_token_num,
        hidden_dim=hidden_dim,
        dtype=dtype,
        use_oneshot=use_oneshot,
        use_attn_tp_group=True,
    )


def cleanup_flashinfer_workspace():
    from sglang.srt.runtime_context import get_resources

    buffers = get_resources().buffers
    for name in (
        "flashinfer_fusion_attn_tp_workspace",
        "flashinfer_fusion_moe_tp_workspace",
    ):
        manager = buffers.get(name)
        if manager is not None:
            manager.cleanup()
