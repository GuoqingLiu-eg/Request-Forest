import logging
from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

logger = logging.getLogger(__name__)

try:
    import torch.distributed.tensor as dt
except ImportError:

    import torch.distributed._tensor as dt

from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)


def _shard_tensor(
    full_tensor: torch.Tensor,
    device_mesh: DeviceMesh,
    placements: Sequence[dt.Shard],
) -> "dt.DTensor":

    shape, offset = dt._utils.compute_local_shape_and_global_offset(
        full_tensor.shape, device_mesh, placements
    )
    slices = [
        slice(cur_offset, cur_offset + cur_shape)
        for cur_shape, cur_offset in zip(shape, offset)
    ]
    local_tensor = full_tensor[slices]
    return dt.DTensor.from_local(local_tensor, device_mesh, placements)


class ColwiseParallelSharded(ColwiseParallel):


    def _partition_linear_fn(self, name, module, device_mesh):


        for name, param in module.named_parameters():
            dtensor = dt.DTensor.from_local(param, device_mesh, [dt.Shard(0)])
            dist_param = torch.nn.Parameter(dtensor, requires_grad=False)
            module.register_parameter(name, dist_param)


class RowwiseParallelMaybeWait(RowwiseParallel):


    def _partition_linear_fn(self, name, module, device_mesh):


        module.register_parameter(
            "weight",
            nn.Parameter(_shard_tensor(module.weight, device_mesh, [dt.Shard(1)])),
        )
        if getattr(module, "bias", None) is not None:

            module.register_parameter(
                "bias",
                nn.Parameter(
                    dt.distribute_tensor(module.bias, device_mesh, [dt.Replicate()])
                ),
            )

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        outputs = super(
            RowwiseParallelMaybeWait, RowwiseParallelMaybeWait
        )._prepare_output_fn(
            output_layouts, use_local_output, mod, outputs, device_mesh
        )
        return torch.distributed._functional_collectives.wait_tensor(outputs)


def tensor_parallel(
    module: torch.nn.Module,
    device_mesh: Optional[DeviceMesh] = None,
):


    def tplize(mod: torch.nn.Module) -> None:
        tp_plan = getattr(mod, "_tp_plan", None)
        if tp_plan is None:
            return
        for child_name, tp_style in tp_plan.items():
            submod = mod.get_submodule(child_name)
            if tp_style == "Colwise":
                parallelize_module(submod, device_mesh, ColwiseParallel())
            elif tp_style == "Rowwise":
                parallelize_module(submod, device_mesh, RowwiseParallelMaybeWait())
            elif tp_style == "Colwise_Sharded":
                parallelize_module(submod, device_mesh, ColwiseParallelSharded())
            else:
                raise ValueError(f"Unknown TP style {tp_style}")


    module.apply(tplize)


def apply_torch_tp(
    *,
    model: nn.Module,
    device: str,
    tp_size: int,
):
    logger.info(f"Enabling torch tensor parallelism on {tp_size} devices.")
    device_mesh = torch.distributed.init_device_mesh(device, (tp_size,))
    tensor_parallel(model, device_mesh)
