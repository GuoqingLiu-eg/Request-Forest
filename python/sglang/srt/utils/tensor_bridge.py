# Copied and adapted from: https://github.com/vllm-project/vllm-metal
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from functools import lru_cache, wraps
from threading import RLock
from typing import TYPE_CHECKING, Any, Callable, Literal

import torch

if TYPE_CHECKING:
    import mlx.core as mx


_BRIDGE_LOCK = RLock()


def _serialized_bridge(function: Callable[..., Any]) -> Callable[..., Any]:


    @wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with _BRIDGE_LOCK:
            return function(*args, **kwargs)

    return wrapper


@lru_cache(maxsize=1)
def _mlx_core():
    try:
        import mlx.core as mx
    except ImportError:
        raise RuntimeError("The MLX tensor bridge requires MLX >= 0.32.0") from None
    return mx


def _get_torch_device() -> torch.device:

    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _torch_to_mlx(
    tensor: torch.Tensor,
    *,
    copy: bool,
    synchronize: bool = True,
) -> mx.array:

    mx = _mlx_core()
    tensor = tensor.detach()

    if tensor.device.type == "mps":
        if synchronize:

            torch.mps.synchronize()
        return mx.asarray(tensor, copy=copy)
    if tensor.device.type == "cpu":


        if tensor.dtype == torch.complex128:
            raise ValueError(
                "MLX 0.32 does not support complex128; convert the Torch tensor "
                "to complex64 explicitly"
            )


        if tensor.dtype == torch.float64:
            with mx.stream(mx.cpu):
                return mx.array(tensor, dtype=mx.float64)
        return mx.array(tensor)
    raise ValueError(
        f"The MLX tensor bridge supports CPU and MPS tensors, got {tensor.device}"
    )


class MlxTensorView:


    __slots__ = ("torch_tensor", "array")

    def __init__(self, tensor: torch.Tensor, *, synchronize: bool = True):
        with _BRIDGE_LOCK:
            owner = tensor.detach()
            if owner.device.type != "mps":
                raise ValueError(
                    f"MlxTensorView requires a Torch MPS tensor, got {owner.device}"
                )
            if synchronize:
                torch.mps.synchronize()
            self.torch_tensor = owner
            self.array = _torch_to_mlx(owner, copy=False, synchronize=False)

    @classmethod
    def _from_synchronized(cls, tensor: torch.Tensor) -> MlxTensorView:
        view = object.__new__(cls)
        owner = tensor.detach()
        if owner.device.type != "mps":
            raise ValueError(
                f"MlxTensorView requires a Torch MPS tensor, got {owner.device}"
            )
        view.torch_tensor = owner
        view.array = _torch_to_mlx(owner, copy=False, synchronize=False)
        return view

    def matches(self, tensor: torch.Tensor) -> bool:

        owner = tensor.detach()
        return (
            owner.device == self.torch_tensor.device
            and owner.dtype == self.torch_tensor.dtype
            and owner.shape == self.torch_tensor.shape
            and owner.stride() == self.torch_tensor.stride()
            and owner.data_ptr() == self.torch_tensor.data_ptr()
        )


@_serialized_bridge
def borrow_torch_tensors(
    *tensors: torch.Tensor, synchronize: bool = True
) -> tuple[MlxTensorView, ...]:

    detached = tuple(tensor.detach() for tensor in tensors)
    if any(tensor.device.type != "mps" for tensor in detached):
        devices = ", ".join(str(tensor.device) for tensor in detached)
        raise ValueError(f"borrow_torch_tensors requires MPS tensors, got {devices}")
    if synchronize and detached:
        torch.mps.synchronize()
    return tuple(MlxTensorView._from_synchronized(tensor) for tensor in detached)


@_serialized_bridge
def torch_to_mlx(tensor: torch.Tensor) -> mx.array:

    array = _torch_to_mlx(tensor, copy=True)
    if tensor.device.type == "mps":


        _mlx_core().eval(array)
    return array


@_serialized_bridge
def mlx_call(
    operation: Callable[..., mx.array],
    *tensors: torch.Tensor | MlxTensorView,
    device: torch.device | Literal["mps", "cpu"] | None = None,
) -> torch.Tensor:

    mx = _mlx_core()
    target_device = _get_torch_device() if device is None else torch.device(device)
    if target_device.type not in {"cpu", "mps"}:
        raise ValueError(
            f"The MLX tensor bridge supports CPU and MPS targets, got {target_device}"
        )
    detached = tuple(
        tensor.detach() for tensor in tensors if isinstance(tensor, torch.Tensor)
    )
    if any(tensor.device.type == "mps" for tensor in detached) or any(
        isinstance(tensor, MlxTensorView) for tensor in tensors
    ):
        torch.mps.synchronize()
    borrowed: tuple[Any, ...] = tuple(
        (
            tensor.array
            if isinstance(tensor, MlxTensorView)
            else _torch_to_mlx(tensor.detach(), copy=False, synchronize=False)
        )
        for tensor in tensors
    )


    if target_device.type == "cpu" and any(
        array.dtype == mx.float64 for array in borrowed
    ):
        with mx.stream(mx.cpu):
            result = operation(*borrowed)
    else:
        result = operation(*borrowed)
    output = mlx_to_torch(result, device=target_device)


    _ = borrowed
    return output


def _prepare_mlx_export(
    array: mx.array,
    target_device: torch.device,
    mx: Any,
) -> mx.array:

    if target_device.type not in {"cpu", "mps"}:
        raise ValueError(
            f"The MLX tensor bridge supports CPU and MPS targets, got {target_device}"
        )

    if target_device.type == "mps" and array.dtype == mx.float64:
        raise ValueError(
            "MLX float64 arrays cannot be exported to a Torch MPS tensor; "
            "use float32/bfloat16 or request device='cpu'"
        )

    return array


def _has_negative_stride(array: mx.array) -> bool:


    with memoryview(array) as view:
        return any(stride < 0 for stride in (view.strides or ()))


def _export_evaluated_mlx(
    array: mx.array,
    target_device: torch.device,
    mx: Any,
    *,
    materialize_negative: bool = True,
) -> torch.Tensor:

    if materialize_negative and _has_negative_stride(array):
        materialize_stream = mx.cpu if target_device.type == "cpu" else mx.gpu
        array = mx.contiguous(array, stream=materialize_stream)
        mx.eval(array)

    if target_device.type == "cpu":


        dlpack = array.__dlpack__(dl_device=(1, 0), copy=False)
        return torch.utils.dlpack.from_dlpack(dlpack)
    return torch.utils.dlpack.from_dlpack(array)


@_serialized_bridge
def mlx_call_multi(
    operation: Callable[..., tuple[mx.array, ...]],
    *tensors: torch.Tensor | MlxTensorView,
    device: torch.device | Literal["mps", "cpu"] | None = None,
) -> tuple[torch.Tensor, ...]:

    mx = _mlx_core()
    target_device = _get_torch_device() if device is None else torch.device(device)
    if target_device.type not in {"cpu", "mps"}:
        raise ValueError(
            f"The MLX tensor bridge supports CPU and MPS targets, got {target_device}"
        )

    detached = tuple(
        tensor.detach() for tensor in tensors if isinstance(tensor, torch.Tensor)
    )
    needs_mps_fence = any(tensor.device.type == "mps" for tensor in detached) or any(
        isinstance(tensor, MlxTensorView) for tensor in tensors
    )
    if needs_mps_fence:
        torch.mps.synchronize()

    borrowed: tuple[Any, ...] = tuple(
        (
            tensor.array
            if isinstance(tensor, MlxTensorView)
            else _torch_to_mlx(tensor.detach(), copy=False, synchronize=False)
        )
        for tensor in tensors
    )

    if target_device.type == "cpu" and any(
        array.dtype == mx.float64 for array in borrowed
    ):
        with mx.stream(mx.cpu):
            result = operation(*borrowed)
    else:
        result = operation(*borrowed)
    if not isinstance(result, (tuple, list)) or not result:
        raise TypeError(
            "mlx_call_multi operation must return a non-empty tuple or list of MLX arrays"
        )
    arrays = tuple(result)
    if any(not isinstance(array, mx.array) for array in arrays):
        raise TypeError("mlx_call_multi outputs must be MLX arrays")


    arrays = tuple(_prepare_mlx_export(array, target_device, mx) for array in arrays)
    mx.eval(*arrays)


    negative = tuple(_has_negative_stride(array) for array in arrays)
    if any(negative):
        materialized = []
        for array, needs_materialization in zip(arrays, negative):
            if needs_materialization:
                stream = mx.cpu if target_device.type == "cpu" else mx.gpu
                array = mx.contiguous(array, stream=stream)
            materialized.append(array)
        arrays = tuple(materialized)
        mx.eval(*(array for array, needs in zip(arrays, negative) if needs))

    outputs = tuple(
        _export_evaluated_mlx(array, target_device, mx, materialize_negative=False)
        for array in arrays
    )


    _ = borrowed
    return outputs


@_serialized_bridge
def mlx_to_torch(
    array: mx.array,
    device: torch.device | Literal["mps", "cpu"] | None = None,
) -> torch.Tensor:

    mx = _mlx_core()
    target_device = _get_torch_device() if device is None else torch.device(device)
    array = _prepare_mlx_export(array, target_device, mx)
    mx.eval(array)
    return _export_evaluated_mlx(array, target_device, mx)


__all__ = [
    "MlxTensorView",
    "borrow_torch_tensors",
    "mlx_call",
    "mlx_call_multi",
    "mlx_to_torch",
    "torch_to_mlx",
]
