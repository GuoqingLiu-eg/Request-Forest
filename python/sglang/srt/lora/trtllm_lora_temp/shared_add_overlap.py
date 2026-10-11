from typing import Optional, Tuple

import torch

from sglang.srt.lora.trtllm_lora_temp.environ import lora_envs

_PENDING: Optional[Tuple[torch.Tensor, torch.cuda.Stream]] = None


_SHARED_ADD_EVENTS: list = []


def shared_add_overlap_enabled() -> bool:
    return lora_envs.SGLANG_OPT_LORA_SHARED_ADD_OVERLAP.get()


def stage_shared_expert_add(
    shared_output: torch.Tensor, producer_stream: torch.cuda.Stream
) -> None:

    global _PENDING
    _PENDING = (shared_output, producer_stream)


def unstage_shared_expert_add() -> Optional[torch.Tensor]:

    global _PENDING
    if _PENDING is None:
        return None
    shared_output, _ = _PENDING
    _PENDING = None
    return shared_output


def maybe_overlap_staged_shared_add(output: torch.Tensor) -> Optional[torch.cuda.Event]:

    global _PENDING
    if _PENDING is None:
        return None
    shared_output, producer_stream = _PENDING
    current_stream = torch.cuda.current_stream()
    if producer_stream == current_stream:


        return None
    if torch.cuda.is_current_stream_capturing():


        return None
    _PENDING = None

    base_ready = torch.cuda.Event()
    base_ready.record(current_stream)
    add_done = torch.cuda.Event()
    with torch.cuda.stream(producer_stream):
        producer_stream.wait_event(base_ready)
        output.add_(shared_output)
        add_done.record(producer_stream)

    if torch.cuda.is_current_stream_capturing():
        _SHARED_ADD_EVENTS.append(base_ready)
        _SHARED_ADD_EVENTS.append(add_done)
    return add_done
