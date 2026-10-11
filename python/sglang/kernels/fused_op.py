from __future__ import annotations

import functools
import logging
from abc import ABC, abstractmethod
from typing import (
    AbstractSet,
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
)

import msgspec
import torch
from torch import nn

from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.kernels.registry import register_kernel
from sglang.kernels.spec import (
    CapabilityRequirement,
    FormatSignature,
    KernelBackend,
    KernelSpec,
    PlatformInfo,
    capabilities_satisfied,
)

logger = logging.getLogger(__name__)


BACKEND_METHODS: Dict[KernelBackend, str] = {
    KernelBackend.TORCH: "forward_native",
    KernelBackend.TORCH_COMPILE: "forward_torch_compile",
    KernelBackend.TRITON: "forward_triton",
    KernelBackend.JIT: "forward_jit",
    KernelBackend.AOT: "forward_aot",
    KernelBackend.CUTE_DSL: "forward_cute_dsl",
    KernelBackend.FLYDSL: "forward_flydsl",
    KernelBackend.FLASHINFER: "forward_flashinfer",
    KernelBackend.DEEPGEMM: "forward_deepgemm",
    KernelBackend.AITER: "forward_aiter",
    KernelBackend.TORCH_NPU: "forward_torch_npu",
}

_METHOD_BACKEND_LABELS: Dict[str, str] = {
    name: backend.value for backend, name in BACKEND_METHODS.items()
}


DEFAULT_PRIORITY: Tuple[KernelBackend, ...] = (
    KernelBackend.AOT,
    KernelBackend.JIT,
    KernelBackend.FLASHINFER,
    KernelBackend.DEEPGEMM,
    KernelBackend.CUTE_DSL,
    KernelBackend.AITER,
    KernelBackend.TORCH_NPU,
    KernelBackend.TRITON,
    KernelBackend.TORCH,
)


_ALWAYS_AVAILABLE = (KernelBackend.TORCH, KernelBackend.TORCH_COMPILE)


_PLATFORM_METHODS: Dict[str, Tuple[str, ...]] = {
    "cuda": ("forward_cuda",),
    "hip": ("forward_hip", "forward_cuda"),
    "musa": ("forward_musa",),
    "npu": ("forward_npu",),
    "xpu": ("forward_xpu",),
    "cpu": ("forward_cpu",),
}


@functools.lru_cache(maxsize=1)
def _platform() -> PlatformInfo:
    return PlatformInfo.detect()


@functools.lru_cache(maxsize=1)
def _platform_key() -> str:

    from sglang.srt.utils import (
        cpu_has_amx_support,
        is_cpu,
        is_cuda,
        is_hip,
        is_musa,
        is_npu,
        is_xpu,
    )

    if is_cuda():
        return "cuda"
    if is_hip():
        return "hip"
    if is_cpu() and cpu_has_amx_support():
        return "cpu"
    if is_npu():
        return "npu"
    if is_xpu():
        return "xpu"
    if is_musa():
        return "musa"
    return ""


@functools.lru_cache(maxsize=1)
def _oot_dispatch_key() -> Optional[str]:

    from sglang.srt.platforms import current_platform

    if current_platform.is_out_of_tree():
        return current_platform.get_dispatch_key_name()
    return None


def clear_platform_caches() -> None:

    _platform.cache_clear()
    _platform_key.cache_clear()
    _oot_dispatch_key.cache_clear()


_UNRESOLVED = object()
_forced_backend: Any = _UNRESOLVED


def get_fused_op_backend() -> Optional[KernelBackend]:

    global _forced_backend
    if _forced_backend is _UNRESOLVED:
        from sglang.srt.environ import envs

        value = envs.SGLANG_FORCE_FUSED_OP_BACKEND.get()
        _forced_backend = KernelBackend(value) if value is not None else None
    return _forced_backend


def set_fused_op_backend(backend: Optional[KernelBackend]) -> None:

    global _forced_backend
    _forced_backend = backend


class FusedOpTraceRecord(msgspec.Struct, frozen=True):


    op: str
    backend: str
    tensor_args: Tuple[str, ...]


_trace_enabled: bool = False
_trace_records: List[FusedOpTraceRecord] = []


def enable_fused_op_trace() -> None:

    global _trace_enabled
    _trace_enabled = True


def disable_fused_op_trace() -> None:
    global _trace_enabled
    _trace_enabled = False


def get_fused_op_trace() -> List[FusedOpTraceRecord]:
    return list(_trace_records)


def clear_fused_op_trace() -> None:
    _trace_records.clear()


def _describe_tensors(args: tuple, kwargs: dict) -> Tuple[str, ...]:
    described = []
    for value in (*args, *kwargs.values()):
        if isinstance(value, torch.Tensor):
            described.append(f"{value.dtype}[{', '.join(map(str, value.shape))}]")
    return tuple(described)


def _dispatch_label(method: Callable) -> str:

    name = getattr(method, "__name__", "")
    label = _METHOD_BACKEND_LABELS.get(name)
    if label is not None:
        return label
    if name.startswith("forward_"):
        return name[len("forward_") :]
    return name or "unknown"


def _record_trace(op: BaseFusedOp, label: str, args: tuple, kwargs: dict) -> None:
    _trace_records.append(
        FusedOpTraceRecord(
            op=op.op or type(op).__name__,
            backend=label,
            tensor_args=_describe_tensors(args, kwargs),
        )
    )


_warned_forced_fallbacks: set = set()


def _warn_forced_backend_unavailable(*, op: BaseFusedOp, backend: KernelBackend):
    key = (type(op), backend)
    if key not in _warned_forced_fallbacks:
        _warned_forced_fallbacks.add(key)
        logger.warning(
            "Forced fused-op backend %r is not implemented by %s; "
            "falling back to normal dispatch for this op.",
            backend.value,
            op._op_label(),
        )


class BaseFusedOp(nn.Module, ABC):


    op: ClassVar[str] = ""
    priority: ClassVar[Tuple[KernelBackend, ...]] = DEFAULT_PRIORITY
    capabilities: ClassVar[
        Mapping[KernelBackend, AbstractSet[CapabilityRequirement]]
    ] = {}
    format_signature: ClassVar[FormatSignature] = FormatSignature()
    descriptions: ClassVar[Mapping[KernelBackend, str]] = {}


    _oot_forward_registry: ClassVar[Dict[str, Dict[type, Callable]]] = {}

    @classmethod
    def register_oot_forward(cls, op_cls: type, fn: Callable, platform_key: str):

        cls._oot_forward_registry.setdefault(platform_key, {})[op_cls] = fn

    def __init__(self) -> None:
        super().__init__()


        self._forward_method: Optional[Callable] = None

        self._original_forward_method: Optional[Callable] = None
        self.is_torch_compile = False
        self._compiled_native = None

    def _defined_method(self, method_name: str) -> Optional[Callable]:

        for klass in type(self).__mro__:
            if klass is BaseFusedOp:
                return None
            if method_name in klass.__dict__:
                return getattr(self, method_name)
        return None


    @abstractmethod
    def forward_native(self, *args, **kwargs):
        pass

    def forward_torch_compile(self, *args, **kwargs):
        if self._compiled_native is None:
            self._compiled_native = torch.compile(self.forward_native)
        return self._compiled_native(*args, **kwargs)

    def forward_triton(self, *args, **kwargs):
        raise NotImplementedError(f"{self._op_label()}: no triton backend")

    def forward_jit(self, *args, **kwargs):
        raise NotImplementedError(f"{self._op_label()}: no jit backend")

    def forward_aot(self, *args, **kwargs):
        raise NotImplementedError(f"{self._op_label()}: no aot backend")

    def forward_cute_dsl(self, *args, **kwargs):
        raise NotImplementedError(f"{self._op_label()}: no cute_dsl backend")

    def forward_flashinfer(self, *args, **kwargs):
        raise NotImplementedError(f"{self._op_label()}: no flashinfer backend")

    def forward_deepgemm(self, *args, **kwargs):
        raise NotImplementedError(f"{self._op_label()}: no deepgemm backend")

    def forward_aiter(self, *args, **kwargs):
        raise NotImplementedError(f"{self._op_label()}: no aiter backend")

    def forward_torch_npu(self, *args, **kwargs):
        raise NotImplementedError(f"{self._op_label()}: no torch_npu backend")

    def _op_label(self) -> str:
        return self.op or type(self).__name__


    def available_backends(self) -> List[KernelBackend]:

        return [
            backend
            for backend in KernelBackend
            if backend in _ALWAYS_AVAILABLE
            or self._defined_method(BACKEND_METHODS[backend]) is not None
        ]

    def backend_eligible(self, backend: KernelBackend, *args, **kwargs) -> bool:

        return capabilities_satisfied(
            self.capabilities.get(backend, frozenset()), _platform()
        )

    def _auto_backend_candidates(self) -> Tuple[KernelBackend, ...]:

        candidates = []
        for backend in self.priority:
            if backend is KernelBackend.TORCH:
                continue
            if backend is KernelBackend.TORCH_COMPILE:
                candidates.append(backend)
                continue
            if (
                backend in self.capabilities
                and self._defined_method(BACKEND_METHODS[backend]) is not None
            ):
                candidates.append(backend)
        return tuple(candidates)

    def auto_selected_backend(self) -> Optional[KernelBackend]:

        platform = _platform()
        for backend in self._auto_backend_candidates():
            if capabilities_satisfied(
                self.capabilities.get(backend, frozenset()), platform
            ):
                return backend
        return None

    def _platform_method(self, platform_key: str) -> Optional[Callable]:

        if not platform_key:
            return None
        names = _PLATFORM_METHODS.get(platform_key, (f"forward_{platform_key}",))
        for name in names:
            method = self._defined_method(name)
            if method is not None:
                return method
        return None

    def _resolve_forward_method(self) -> Callable:


        oot_key = _oot_dispatch_key()
        if oot_key is not None:
            registered = self._oot_forward_registry.get(oot_key, {}).get(type(self))
            if registered is not None:
                return registered.__get__(self)
            method = self._platform_method(oot_key)
            if method is not None:
                return method
            return self.forward_native


        candidates = self._auto_backend_candidates()
        if candidates:
            if self._defined_method("backend_eligible") is not None:

                self._dynamic_backend_candidates = candidates
                return self._forward_backend_dynamic
            platform = _platform()
            for backend in candidates:
                if capabilities_satisfied(
                    self.capabilities.get(backend, frozenset()), platform
                ):
                    return getattr(self, BACKEND_METHODS[backend])


        method = self._platform_method(_platform_key())
        return method if method is not None else self.forward_native

    def _forward_backend_dynamic(self, *args, **kwargs):

        for backend in self._dynamic_backend_candidates:
            if self.backend_eligible(backend, *args, **kwargs):
                return getattr(self, BACKEND_METHODS[backend])(*args, **kwargs)
        method = self._platform_method(_platform_key())
        return (method or self.forward_native)(*args, **kwargs)

    def dispatch_forward(self) -> Callable:

        return self._resolve_forward_method()


    def _torch_compile_forward(self, num_tokens: int) -> Optional[Callable]:

        return self.forward_native

    def enter_torch_compile(self, num_tokens: int) -> None:

        if self.is_torch_compile:
            return


        get_fused_op_backend()
        self._original_forward_method = self._forward_method
        compile_forward = self._torch_compile_forward(num_tokens=num_tokens)
        if compile_forward is not None:
            self._forward_method = compile_forward
        elif self._forward_method is None:
            self._forward_method = self._resolve_forward_method()
        self.is_torch_compile = True

    def leave_torch_compile(self) -> None:

        if not self.is_torch_compile:
            return
        self._forward_method = self._original_forward_method
        self._original_forward_method = None
        self.is_torch_compile = False


    @debug_kernel_api
    def forward(self, *args, backend: Optional[KernelBackend] = None, **kwargs):

        if backend is not None:


            result = getattr(self, BACKEND_METHODS[backend])(*args, **kwargs)
            if _trace_enabled:
                _record_trace(self, backend.value, args, kwargs)
            return result
        forced = _forced_backend
        if forced is _UNRESOLVED:
            forced = get_fused_op_backend()
        if forced is not None:


            try:
                result = getattr(self, BACKEND_METHODS[forced])(*args, **kwargs)
            except NotImplementedError:
                _warn_forced_backend_unavailable(op=self, backend=forced)
            else:
                if _trace_enabled:
                    _record_trace(self, forced.value, args, kwargs)
                return result
        method = self._forward_method
        if method is None:
            method = self._forward_method = self._resolve_forward_method()
        result = method(*args, **kwargs)
        if _trace_enabled:
            _record_trace(self, _dispatch_label(method), args, kwargs)
        return result


def register_fused_op(instance: BaseFusedOp, module: str, attr: str) -> BaseFusedOp:

    for backend in instance.available_backends():
        register_kernel(
            KernelSpec(
                op=instance.op,
                backend=backend,
                target=f"{module}:{attr}.{BACKEND_METHODS[backend]}",
                capabilities=frozenset(instance.capabilities.get(backend, ())),
                format_signature=instance.format_signature,
                description=instance.descriptions.get(backend, ""),
            )
        )
    return instance
