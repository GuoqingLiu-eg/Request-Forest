import logging
from typing import Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


class _AsyncNanWarner:


    def __init__(self):
        self._dev = None
        self._host = None
        self._warned = False

    def check(self, tensor: torch.Tensor, msg: str):
        if self._warned or not tensor.is_cuda:
            return
        if self._dev is None:
            self._dev = torch.zeros(1, dtype=torch.int32, device=tensor.device)
            self._host = torch.zeros(1, dtype=torch.int32, pin_memory=True)


        if int(self._host[0]):
            logger.warning(
                "NaN detected in %s; values were sanitized before sampling. "
                "This usually indicates numerical overflow (e.g. fp16 "
                "activations) or an upstream bug producing NaN. "
                "Logged once; further occurrences are silent.",
                msg,
            )
            self._warned = True
            return


        self._dev.add_(torch.isnan(tensor).any().to(torch.int32))
        self._host.copy_(self._dev, non_blocking=True)


_nan_warner = _AsyncNanWarner()


def maybe_warn_nan(tensor: Optional[torch.Tensor], msg: str = ""):

    if envs.SGLANG_ENABLE_ASYNC_ASSERT.get():

        return
    if tensor is None:
        return
    _nan_warner.check(tensor, msg)


def sanitize_nan_logits(logits: torch.Tensor, msg: str = ""):

    maybe_detect_nan(logits, msg)
    if not envs.SGLANG_SANITIZE_NAN_LOGITS.get():
        return
    maybe_warn_nan(logits, msg)
    torch.nan_to_num_(logits, nan=-1e30, posinf=1e30, neginf=-1e30)


def maybe_assert_async(cond: torch.Tensor, msg: str = ""):
    if not envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        return
    torch._assert_async(cond, msg)


def maybe_assert_sum(tensor: torch.Tensor, expected: int, msg: str = "") -> None:
    if not envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        return
    torch._assert_async(tensor.sum() == expected, msg)


def maybe_detect_nan(tensor: Optional[torch.Tensor], msg: str = ""):

    if not envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        return


    if tensor is None:
        return
    torch._assert_async(~torch.any(torch.isnan(tensor)), f"NaN detected! {msg}")


def maybe_detect_inf(tensor: Optional[torch.Tensor], msg: str = ""):

    if not envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        return
    if tensor is None:
        return
    torch._assert_async(~torch.any(torch.isinf(tensor)), f"Inf detected! {msg}")


def maybe_detect_in_closed_range(
    tensor: Optional[torch.Tensor], low: float, high: float, msg: str = ""
):
    if not envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        return
    if tensor is None or tensor.numel() == 0:
        return
    torch._assert_async(
        ((tensor >= low) & (tensor <= high)).all(),
        f"value outside [{low}, {high}]: {msg}",
    )


def maybe_detect_oob(indices: Optional[torch.Tensor], low: int, high: int, msg: str):

    if not envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        return
    if indices is None or indices.numel() == 0:
        return
    torch._assert_async(
        indices.min() >= low,
        f"index < {low} (negative / unmasked sentinel?): {msg}",
    )
    torch._assert_async(
        indices.max() < high,
        f"index >= {high} (out of range): {msg}",
    )


def maybe_detect_page_aligned(
    indices: Optional[torch.Tensor], page_size: int, msg: str
):

    if not envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        return
    if indices is None or indices.numel() == 0 or page_size <= 1:
        return
    torch._assert_async(
        (indices % page_size == 0).all(),
        f"page-misaligned indices (page_size={page_size}): {msg}",
    )
