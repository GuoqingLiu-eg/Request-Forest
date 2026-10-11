from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol, Sequence

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo


@dataclass(frozen=True)
class CommittedTokens:
    output_index: int
    token_ids: tuple[int, ...]


class DeviceAuxiliaryOutput(Protocol):


    def copy_to_host(
        self, copy_tensor: Callable[[torch.Tensor], torch.Tensor]
    ) -> HostAuxiliaryOutput: ...


class HostAuxiliaryOutput(Protocol):


    def consume(
        self,
        batch: ScheduleBatch,
        commits: Sequence[Optional[CommittedTokens]],
    ) -> None: ...


class SamplingObserver(Protocol):


    def is_active(self, sampling_info: SamplingBatchInfo) -> bool: ...

    def before_grammar(
        self,
        logits: torch.Tensor,
        sampling_info: SamplingBatchInfo,
    ) -> Any: ...

    def after_sample(
        self, state: Any, token_ids: torch.Tensor
    ) -> Optional[DeviceAuxiliaryOutput]:

        ...
