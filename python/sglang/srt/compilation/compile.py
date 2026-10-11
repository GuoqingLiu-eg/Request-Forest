import inspect
import logging
import os
import sys
import types
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch

from sglang.srt.compilation.compilation_config import CompilationConfig
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)

logger = logging.getLogger(__name__)


@dataclass
class IntermediateTensors:


    tensors: dict[str, torch.Tensor]

    finished_sending: Optional[set[str]] = None
    finished_recving: Optional[set[str]] = None

    def __init__(self, tensors):


        self.tensors = tensors

    def __getitem__(self, key: Union[str, slice]):
        if isinstance(key, str):
            return self.tensors[key]
        elif isinstance(key, slice):
            return self.__class__({k: v[key] for k, v in self.tensors.items()})

    def __setitem__(self, key: str, value: torch.Tensor):
        self.tensors[key] = value

    def items(self):
        return self.tensors.items()

    def __len__(self):
        return len(self.tensors)

    def __eq__(self, other: object):
        return isinstance(other, self.__class__) and self

    def __repr__(self) -> str:
        return f"IntermediateTensors(tensors={self.tensors})"


def _normalize_dims(dims, ndim: int):
    dims = [dims] if isinstance(dims, int) else list(dims)
    return [d if d >= 0 else ndim + d for d in dims]


class _MaybeIntermediateTensors:


    def __init__(self, obj):
        self.is_intermediate = hasattr(obj, "tensors") and isinstance(
            getattr(obj, "tensors"), dict
        )
        self.obj = obj


def _mark_dynamic_on_value(val, dims):
    if isinstance(val, torch.Tensor):
        torch._dynamo.maybe_mark_dynamic(val, _normalize_dims(dims, val.ndim))
    else:
        mit = _MaybeIntermediateTensors(val)
        if mit.is_intermediate:
            for t in mit.obj.tensors.values():
                torch._dynamo.maybe_mark_dynamic(t, _normalize_dims(dims, t.ndim))


_MROPE_TOKEN_AXIS_ARGUMENTS = frozenset(
    {"positions", "position_ids", "mrope_positions"}
)


def _runtime_dynamic_dim_for_argument(name: str) -> int:

    return -1 if name in _MROPE_TOKEN_AXIS_ARGUMENTS else 0


def _infer_dynamic_arg_dims_from_annotations(forward_fn):
    sig = inspect.signature(forward_fn)
    dyn = {}
    for name, p in sig.parameters.items():
        ann = p.annotation

        if (
            ann is torch.Tensor
            or getattr(getattr(ann, "__args__", [None])[0], "__name__", "") == "Tensor"
        ):
            dyn[name] = _runtime_dynamic_dim_for_argument(name)
        elif getattr(ann, "__name__", "") in ("IntermediateTensors",) or any(
            getattr(a, "__name__", "") == "IntermediateTensors"
            for a in getattr(ann, "__args__", [])
        ):
            dyn[name] = 0
        elif ann == "torch.Tensor" or ann == "Optional[torch.Tensor]":

            dyn[name] = _runtime_dynamic_dim_for_argument(name)
    if not dyn:
        raise ValueError("No dynamic dims inferred; pass dynamic_arg_dims explicitly.")
    return dyn


def _mark_dynamic_forward_batch(forward_batch) -> None:

    if forward_batch is None:
        return

    for name, value in vars(forward_batch).items():
        if not isinstance(value, torch.Tensor) or value.ndim == 0:
            continue
        dims = _runtime_dynamic_dim_for_argument(name)
        _mark_dynamic_on_value(value, dims)


def install_torch_compiled(
    module: torch.nn.Module,
    *,
    dynamic_arg_dims: dict[str, Union[int, list[int]]] | None = None,
    backend_factory: Optional[Callable[[torch.fx.GraphModule, list], Callable]] = None,
    compile_config: CompilationConfig = None,
    fullgraph: bool = True,
    graph_pool: Any = None,
):
    unbound_fwd = module.__class__.forward
    if not callable(unbound_fwd):
        raise TypeError("module.__class__.forward must be callable")
    original_code = unbound_fwd.__code__

    dyn_map = dynamic_arg_dims or _infer_dynamic_arg_dims_from_annotations(unbound_fwd)

    if backend_factory is None:
        from sglang.srt.compilation.backend import SGLangBackend

        def backend_factory(gm, ex):
            return SGLangBackend(compile_config, graph_pool)(gm, ex)

    compiled_codes: list[type(original_code)] = []
    state = {"compiled": False, "compiled_callable": None}

    def bytecode_hook(old_code, new_code):
        if old_code is not original_code:
            return
        frame = sys._getframe()
        while frame and frame.f_back:
            frame = frame.f_back
            if (
                frame.f_code.co_name == "_compile"
                and os.path.basename(frame.f_code.co_filename) == "convert_frame.py"
            ):
                break
        try:
            dynamo_frame = frame.f_locals["frame"]
        except Exception:
            return
        if dynamo_frame.f_code is not old_code:
            return
        if dynamo_frame.f_locals.get("self") is not module:
            return
        compiled_codes.append(new_code)

    torch._dynamo.convert_frame.register_bytecode_hook(bytecode_hook)

    def _ensure_compiled(self, *args, **kwargs):

        if state["compiled"]:
            return

        sig = inspect.signature(unbound_fwd)
        ba = sig.bind(self, *args, **kwargs)
        ba.apply_defaults()
        for name, dims in (dyn_map or {}).items():
            if name in ba.arguments:
                val = ba.arguments[name]
                if val is not None:
                    _mark_dynamic_on_value(val, dims)
        _mark_dynamic_forward_batch(ba.arguments.get("forward_batch"))


        torch._dynamo.eval_frame.remove_from_cache(unbound_fwd.__code__)

        bound = types.MethodType(unbound_fwd, self)
        compiled_callable = torch.compile(
            bound,
            fullgraph=fullgraph,
            backend=backend_factory,
        )


        compiled_callable(*args, **kwargs)

        state["compiled"] = True
        state["compiled_callable"] = compiled_callable

    def trampoline(self, *args, **kwargs):
        use_compiled = is_in_tc_piecewise_cuda_graph()
        if use_compiled:
            if not state["compiled"]:
                _ensure_compiled(self, *args, **kwargs)

            compiled_callable = state["compiled_callable"]
            return compiled_callable(*args, **kwargs)
        else:

            return unbound_fwd(self, *args, **kwargs)

    module.forward = types.MethodType(trampoline, module)
    return module
