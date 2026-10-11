from __future__ import annotations

from contextlib import contextmanager

import torch

_in_torch_compile_warmup = False
_pcg_capture_stream: torch.cuda.Stream | None = None


def is_in_torch_compile_warmup() -> bool:

    return _in_torch_compile_warmup


@contextmanager
def enable_torch_compile_warmup():

    global _in_torch_compile_warmup
    _in_torch_compile_warmup = True
    try:
        yield
    finally:
        _in_torch_compile_warmup = False


def get_pcg_capture_stream() -> torch.cuda.Stream | None:
    return _pcg_capture_stream


@contextmanager
def set_pcg_capture_stream(stream: torch.cuda.Stream):
    global _pcg_capture_stream
    _pcg_capture_stream = stream
    try:
        yield
    finally:
        _pcg_capture_stream = None
