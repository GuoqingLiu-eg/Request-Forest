from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel, get_schedule

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch


class SchedulerRecvSkipper:
    @staticmethod
    def maybe_create():
        if get_schedule().scheduler_recv_interval <= 1:
            return None
        return SchedulerRecvSkipper()

    @staticmethod
    def derive_forward_mode(gathered_modes: List[int]) -> Optional[ForwardMode]:

        active = set(gathered_modes) - {
            ForwardMode.IDLE.value,
            ForwardMode.PREBUILT.value,
        }
        if not active:
            return None
        if active - {ForwardMode.DECODE.value, ForwardMode.TARGET_VERIFY.value}:
            return ForwardMode.EXTEND
        if active == {ForwardMode.TARGET_VERIFY.value}:
            return ForwardMode.TARGET_VERIFY
        return ForwardMode.DECODE

    def __init__(self):
        self._use_synced_mode = get_parallel().config.enable_dp_attention
        self._counter = 0
        self._threshold = get_schedule().scheduler_recv_interval

        self._default_weight = envs.SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DEFAULT.get()
        self._weight_of_forward_mode = {
            ForwardMode.DECODE: envs.SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DECODE.get(),
            ForwardMode.TARGET_VERIFY: envs.SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_TARGET_VERIFY.get(),
            None: envs.SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_NONE.get(),
        }

    def _pick_mode(self, last_batch: Optional[ScheduleBatch]) -> Optional[ForwardMode]:


        if last_batch is None:
            return None
        if self._use_synced_mode:
            return last_batch.recv_skipper_forward_mode
        return last_batch.forward_mode

    def handle(self, last_batch: Optional[ScheduleBatch]) -> bool:
        should_recv = False

        last_weight = self._weight_of_forward_mode.get(
            self._pick_mode(last_batch), self._default_weight
        )
        self._counter += last_weight

        if self._counter >= self._threshold:
            self._counter = 0
            should_recv = True

        return should_recv
