from __future__ import annotations

import inspect
from typing import Any, Callable, List, Optional, TypeVar, Union, overload

import torch
import torch.library

from sglang.kernels.kernel_api_logging import debug_torch_op

F = TypeVar("F", bound=Callable)


@overload
def register_custom_op(
    fn: F,
    *,
    op_name: Optional[str] = None,
    mutates_args: Optional[List[str]] = None,
    out_shape: Optional[Union[int, str]] = None,
    eager: bool = True,
) -> F: ...


@overload
def register_custom_op(
    fn: F,
    *,
    op_name: Optional[str] = None,
    mutates_args: Optional[List[str]] = None,
    fake_impl: Optional[Callable],
    eager: bool = True,
) -> F: ...


@overload
def register_custom_op(
    *,
    op_name: Optional[str] = None,
    mutates_args: Optional[List[str]] = None,
    out_shape: Optional[Union[int, str]] = None,
    eager: bool = True,
) -> Callable[[F], F]: ...


@overload
def register_custom_op(
    *,
    op_name: Optional[str] = None,
    mutates_args: Optional[List[str]] = None,
    fake_impl: Optional[Callable],
    eager: bool = True,
) -> Callable[[F], F]: ...


def register_custom_op(
    fn: Optional[Callable] = None,
    *,
    op_name: Optional[str] = None,
    mutates_args: Optional[List[str]] = None,
    eager: bool = True,
    **extra_kwargs,
) -> Any:

    extra_kwarg_keys = set(extra_kwargs.keys())
    expected_kwarg_keys = set({"out_shape", "fake_impl"})
    assert (
        expected_kwarg_keys >= extra_kwarg_keys
    ), f"Unexpected extra kwargs: {extra_kwarg_keys - expected_kwarg_keys}"

    has_out_shape = "out_shape" in extra_kwargs
    has_fake_impl = "fake_impl" in extra_kwargs
    assert not (
        has_out_shape and has_fake_impl
    ), "Only one of `out_shape` or `fake_impl` should be provided."

    if not (has_out_shape or has_fake_impl):
        extra_kwargs["out_shape"] = None

    def decorator(op_func: Callable) -> Callable:
        wrapper = CustomOpWrapper(
            op_name=op_name or op_func.__name__,
            op_func=op_func,
            mutates_args=mutates_args or [],
            **extra_kwargs,
        )
        return wrapper.real_impl if eager else wrapper

    if fn is not None:
        return decorator(fn)
    return decorator


class CustomOpWrapper:
    def __init__(
        self,
        op_name: str,
        op_func: Callable,
        mutates_args: List[str],
        **extra_kwargs,
    ):
        self.op_name = op_name
        self.op_func = op_func
        self.mutates_args = mutates_args
        self.extra_kwargs = extra_kwargs
        self._impl: Optional[Callable] = None

    def __call__(self, *args, **kwargs):
        return self.real_impl(*args, **kwargs)

    @property
    def real_impl(self) -> Callable:
        if self._impl is None:
            if not hasattr(torch.ops.sglang, self.op_name):
                from sglang.srt.utils.common import direct_register_custom_op


                direct_register_custom_op(
                    op_name=self.op_name,
                    op_func=self.op_func,
                    mutates_args=self.mutates_args,
                    fake_impl=self.fake_impl,
                )
            self._impl = debug_torch_op(self.op_func, self.op_name)
            assert self._impl is not None
        return self._impl

    @property
    def fake_impl(self) -> Callable:
        if "fake_impl" in self.extra_kwargs:
            return self.extra_kwargs["fake_impl"]
        assert "out_shape" in self.extra_kwargs
        signature = inspect.signature(self.op_func)
        out_shape = self.extra_kwargs["out_shape"]


        def fake_impl(*args, **kwargs):
            if out_shape is None:
                return None
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            try:
                return torch.empty_like(
                    bound.args[out_shape]
                    if isinstance(out_shape, int)
                    else bound.arguments[out_shape]
                )
            except (IndexError, KeyError):
                raise RuntimeError(
                    f"Cannot find output argument at position `{out_shape}` for "
                    f"custom operator `{self.op_name}` with signature `{signature}`."
                )

        return fake_impl


def register_custom_op_from_extern(
    fn: Callable,
    *,
    op_name: Optional[str] = None,
    mutates_args: Optional[List[str]] = None,
    out_shape: Optional[Union[int, str]] = None,
    out_dtype: Optional[torch.dtype] = None,
    fake_impl: Optional[Callable] = None,
    computed_args: Optional[dict] = None,
) -> Callable:

    name = op_name or fn.__name__
    computed_args = computed_args or {}

    assert not (
        out_shape is not None and fake_impl is not None
    ), "Only one of `out_shape` or `fake_impl` should be provided."


    if computed_args:
        original_fn = fn
        original_sig = inspect.signature(fn)


        new_params = [
            p
            for param_name, p in original_sig.parameters.items()
            if param_name not in computed_args
        ]
        new_sig = original_sig.replace(parameters=new_params)

        def wrapper(*args, **kwargs):
            bound = new_sig.bind(*args, **kwargs)
            bound.apply_defaults()

            for arg_name, compute_fn in computed_args.items():
                bound.arguments[arg_name] = compute_fn(**bound.arguments)
            return original_fn(**bound.arguments)

        wrapper.__name__ = fn.__name__
        wrapper.__qualname__ = fn.__qualname__
        wrapper.__module__ = fn.__module__
        wrapper.__signature__ = new_sig  # type: ignore[attr-defined]

        wrapper.__annotations__ = {
            k: v
            for k, v in getattr(fn, "__annotations__", {}).items()
            if k not in computed_args
        }
        fn = wrapper


    fake_sig = inspect.signature(fn)
    if fake_impl is None and out_shape is not None:

        def _fake_impl(*args, **kwargs):
            bound = fake_sig.bind(*args, **kwargs)
            bound.apply_defaults()
            try:
                ref = (
                    bound.args[out_shape]
                    if isinstance(out_shape, int)
                    else bound.arguments[out_shape]
                )
            except (IndexError, KeyError):
                raise RuntimeError(
                    f"Cannot find output argument at position `{out_shape}` for "
                    f"external function `{name}` with signature `{fake_sig}`."
                )
            if out_dtype is not None:
                return torch.empty(ref.shape, dtype=out_dtype, device=ref.device)
            return torch.empty_like(ref)

        fake_impl = _fake_impl
    elif fake_impl is None:
        fake_impl = lambda *args, **kwargs: None

    from sglang.srt.utils.common import direct_register_custom_op

    direct_register_custom_op(
        op_name=name,
        op_func=fn,
        mutates_args=mutates_args or [],
        fake_impl=fake_impl,
    )

    return debug_torch_op(fn, name)
