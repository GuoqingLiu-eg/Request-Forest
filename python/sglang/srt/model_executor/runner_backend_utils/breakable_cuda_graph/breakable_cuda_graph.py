# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import threading
from contextvars import ContextVar
from typing import Any, Callable, Optional

import torch

try:
    from cuda.bindings import runtime as rt
except ImportError:
    rt = None

from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.cuda_utils import (
    checkCudaErrors,
)
from sglang.srt.utils import get_device_module, is_hip, is_xpu

_is_xpu = is_xpu()

__all__ = [
    "eager_on_graph",
    "BreakableCUDAGraph",
    "BreakableCUDAGraphCapture",
    "break_graph",
]


def _check_cuda_bindings():
    if rt is None:
        raise ImportError(
            "Breakable CUDA graph on NVIDIA requires the 'cuda-python' package. "
            "Install it with: pip install cuda-python"
        )


_current_capture_var: ContextVar["BreakableCUDAGraphCapture | None"] = ContextVar(
    "current_capture", default=None
)
_current_stream_var: ContextVar[torch.Stream | None] = ContextVar(
    "current_stream", default=None
)
_forked_streams_var: ContextVar[set[torch.Stream] | None] = ContextVar(
    "forked_streams", default=None
)


def get_current_stream(device: torch.device | None = None) -> torch.Stream:
    stream = _current_stream_var.get()
    if stream is None:
        return get_device_module().current_stream(device)
    return stream


def _capture_status(stream_ptr: int) -> "rt.cudaStreamCaptureStatus":
    _check_cuda_bindings()
    status, *_ = checkCudaErrors(rt.cudaStreamGetCaptureInfo(stream_ptr))
    return status


def _is_stream_capturing(stream: torch.Stream) -> bool:


    if is_hip() or _is_xpu:
        with get_device_module().stream(stream):
            return get_device_module().is_current_stream_capturing()
    return (
        _capture_status(stream.cuda_stream)
        == rt.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive
    )


_original_wait_stream: Callable | None = None
_hook_lock = threading.Lock()
_hook_refcount = 0


def _hooked_wait_stream(self: torch.Stream, other: torch.Stream):
    assert _original_wait_stream is not None
    forked = _forked_streams_var.get()
    if forked is None:
        _original_wait_stream(self, other)
        return
    capturing = _current_stream_var.get()
    if capturing is None:
        _original_wait_stream(self, other)
        return

    cap_id = capturing.stream_id
    is_self_cap = self is capturing or self.stream_id == cap_id
    is_other_cap = other is capturing or other.stream_id == cap_id

    if is_self_cap and not is_other_cap:
        if not _is_stream_capturing(other):
            return
        _original_wait_stream(self, other)
        forked.discard(other)
    elif is_other_cap and not is_self_cap:
        _original_wait_stream(self, other)
        forked.add(self)
    else:
        _original_wait_stream(self, other)


def _install_wait_stream_hook():
    global _original_wait_stream, _hook_refcount
    with _hook_lock:
        if _hook_refcount == 0:
            _original_wait_stream = get_device_module().Stream.wait_stream
            get_device_module().Stream.wait_stream = _hooked_wait_stream  # type: ignore[assignment]
        _hook_refcount += 1


def _uninstall_wait_stream_hook():
    global _original_wait_stream, _hook_refcount
    with _hook_lock:
        _hook_refcount -= 1
        if _hook_refcount == 0:
            assert _original_wait_stream is not None, "wait_stream hook not installed"
            get_device_module().Stream.wait_stream = _original_wait_stream  # type: ignore[assignment]
            _original_wait_stream = None


def _weak_ref_if_tensor(x):

    if torch.is_tensor(x):
        from sglang.srt.compilation.weak_ref_tensor import weak_ref_tensors

        return weak_ref_tensors(x)
    if isinstance(x, tuple):
        return tuple(_weak_ref_if_tensor(e) for e in x)
    if isinstance(x, list):
        return [_weak_ref_if_tensor(e) for e in x]
    return x


def _copy_output(dst: Any, src: Any) -> Any:

    if torch.is_tensor(dst) and torch.is_tensor(src):
        dst.copy_(src)
        return dst

    if (
        isinstance(dst, (tuple, list))
        and isinstance(src, (tuple, list))
        and len(dst) == len(src)
    ):
        copied = [_copy_output(d, s) for d, s in zip(dst, src)]
        return tuple(copied) if isinstance(dst, tuple) else copied

    if hasattr(dst, "__dict__") and hasattr(src, "__dict__"):
        for key, src_val in src.__dict__.items():
            dst_val = getattr(dst, key, None)
            if torch.is_tensor(dst_val) and torch.is_tensor(src_val):
                dst_val.copy_(src_val)
            else:
                setattr(dst, key, src_val)
        return dst

    if isinstance(dst, dict) and isinstance(src, dict):
        for key, src_val in src.items():
            dst_val = dst.get(key)
            if torch.is_tensor(dst_val) and torch.is_tensor(src_val):
                dst_val.copy_(src_val)
            else:
                dst[key] = src_val
        return dst

    return src


def eager_on_graph(enable: bool, capture_stub: Optional[Callable] = None):
    def decorator(inner: Callable):
        if not enable:
            return inner

        def wrapper(*args, **kwargs):
            capture = _current_capture_var.get()
            if capture is None:
                return inner(*args, **kwargs)


            capture._end_current_segment()


            if capture._barrier_fn is not None:
                capture._barrier_fn()


            if capture_stub is not None:
                output = capture_stub(*args, **kwargs)
            else:
                output = inner(*args, **kwargs)


            captured_inner = inner
            captured_args = tuple(_weak_ref_if_tensor(a) for a in args)
            captured_kwargs = {k: _weak_ref_if_tensor(v) for k, v in kwargs.items()}


            captured_output = output

            def replay_fn():
                new_out = captured_inner(*captured_args, **captured_kwargs)
                return _copy_output(captured_output, new_out)

            capture.cuda_graph._break_fns.append(replay_fn)


            capture._begin_new_segment()
            return output

        return wrapper

    return decorator


class BreakableCUDAGraph:


    def __init__(self, deduped_cuda_graph=None) -> None:
        self._segments: list[Any] = []
        self._break_fns: list[Callable[[], Any]] = []
        self._deduped_cuda_graph = deduped_cuda_graph

    def replay(self) -> None:
        stream = get_device_module().current_stream()
        token = _current_stream_var.set(stream)
        try:
            for i, seg in enumerate(self._segments):
                seg.replay()
                if i < len(self._break_fns):
                    self._break_fns[i]()
        finally:
            _current_stream_var.reset(token)

    def _append_segment(self, graph, needs_instantiate: bool) -> None:
        if self._deduped_cuda_graph is not None:
            self._segments.append(self._deduped_cuda_graph.register(graph))
            return
        if needs_instantiate:
            graph.instantiate()
        self._segments.append(graph)


class BreakableCUDAGraphCapture:


    def __init__(
        self,
        cuda_graph: BreakableCUDAGraph,
        pool=None,
        stream: torch.Stream | None = None,
        capture_error_mode: str = "global",
        barrier_fn: Callable[[], None] | None = None,
    ):
        assert isinstance(
            cuda_graph, BreakableCUDAGraph
        ), "cuda_graph must be a BreakableCUDAGraph"
        self.cuda_graph = cuda_graph
        self._pool = pool if pool is not None else (0, 0)
        self._stream = stream
        self._capture_error_mode = capture_error_mode
        self._barrier_fn = barrier_fn
        self._stream_ctx = None
        self._capture_token = None
        self._stream_token = None
        self._forked_token = None
        self._current_graph = None
        self._current_graph_needs_instantiate = False

    def __enter__(self):
        _install_wait_stream_hook()
        if self._stream is not None:
            self._stream_ctx = get_device_module().stream(self._stream)
            self._stream_ctx.__enter__()
        self._capture_token = _current_capture_var.set(self)
        self._stream_token = _current_stream_var.set(
            self._stream or get_device_module().current_stream()
        )
        self._forked_token = _forked_streams_var.set(set())
        self._begin_new_segment()
        return self

    def __exit__(self, *args: object):
        try:
            self._end_current_segment()
        finally:
            _forked_streams_var.reset(self._forked_token)
            _current_stream_var.reset(self._stream_token)
            _current_capture_var.reset(self._capture_token)
            if self._stream_ctx is not None:
                self._stream_ctx.__exit__(*args)
                self._stream_ctx = None
            _uninstall_wait_stream_hook()
        return False

    def _begin_new_segment(self) -> None:
        graph_cls = torch.xpu.XPUGraph if _is_xpu else torch.cuda.CUDAGraph


        if self.cuda_graph._deduped_cuda_graph is not None:
            try:
                graph = graph_cls(keep_graph=True)
                self._current_graph_needs_instantiate = True
            except TypeError:
                graph = graph_cls()
                self._current_graph_needs_instantiate = False
        else:
            graph = graph_cls()
            self._current_graph_needs_instantiate = False
        if _is_xpu:

            graph.capture_begin(pool=self._pool)
        else:
            graph.capture_begin(
                pool=self._pool, capture_error_mode=self._capture_error_mode
            )
        self._current_graph = graph

    def _end_current_segment(self) -> None:

        main_stream = get_current_stream()
        forked = _forked_streams_var.get()
        if forked:
            assert _original_wait_stream is not None
            for side in list(forked):
                if _is_stream_capturing(side):
                    _original_wait_stream(main_stream, side)
            forked.clear()
        graph = self._current_graph
        assert graph is not None
        graph.capture_end()
        self.cuda_graph._append_segment(graph, self._current_graph_needs_instantiate)
        self._current_graph = None
        self._current_graph_needs_instantiate = False


@eager_on_graph(True)
def break_graph() -> None:

    pass
