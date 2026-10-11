from __future__ import annotations

import abc
import weakref
from typing import TYPE_CHECKING, Optional, Set, Type

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch


class BatchedPenalizerOrchestrator:
    def __init__(
        self,
        vocab_size: int,
        batch: ScheduleBatch,
        penalizers: Set[Type[_BatchedPenalizer]],
    ):
        self.vocab_size = vocab_size
        self._batch_ref = weakref.ref(batch)
        self.device = batch.device
        self.penalizers = {Penalizer: Penalizer(self) for Penalizer in penalizers}

        is_required = False
        for penalizer in self.penalizers.values():
            pen_is_required = penalizer.prepare_if_required()
            is_required |= pen_is_required
        self.is_required = is_required

    @property
    def batch(self) -> ScheduleBatch | None:
        return self._batch_ref()

    @batch.setter
    def batch(self, value: Optional[ScheduleBatch]):
        if value is None:
            self._batch_ref = lambda: None
        else:
            self._batch_ref = weakref.ref(value)

    def reqs(self):
        return self.batch.reqs

    def cumulate_output_tokens(self, output_ids: torch.Tensor):

        for penalizer in self.penalizers.values():
            penalizer.cumulate_output_tokens(output_ids=output_ids)

    def apply(self, logits: torch.Tensor, repeat: Optional[int] = None):

        if repeat is None:
            for penalizer in self.penalizers.values():
                penalizer.apply(logits)
        else:

            bs = logits.shape[0] // repeat
            additive = torch.zeros(
                (bs, logits.shape[1]), dtype=torch.float32, device=logits.device
            )
            self.accumulate_additive_penalties(additive)
            logits.add_(torch.repeat_interleave(additive, repeat, dim=0))

            accumulated = self.accumulate_scaling_penalties()
            if accumulated is not None:
                from sglang.srt.sampling.penaltylib.repetition_penalty import (
                    apply_scaling_penalties,
                )

                expanded = torch.repeat_interleave(accumulated, repeat, dim=0)
                apply_scaling_penalties(logits, expanded)

    def accumulate_additive_penalties(self, logits: torch.Tensor):

        for penalizer in self.penalizers.values():
            if not penalizer.is_multiplicative:
                penalizer.apply(logits)

    def accumulate_scaling_penalties(self) -> Optional[torch.Tensor]:

        result = None
        for penalizer in self.penalizers.values():
            if not penalizer._is_prepared or not penalizer.is_multiplicative:
                continue
            if result is None:
                result = penalizer.get_scaling_penalties().clone()
            else:
                result *= penalizer.get_scaling_penalties()
        return result

    def filter(self, keep_indices: torch.Tensor):

        if not self.is_required:
            return

        if len(keep_indices) == 0:

            self.release()
            return

        is_required = False
        for penalizer in self.penalizers.values():
            tmp_is_required = penalizer.is_required()
            is_required |= tmp_is_required
            if tmp_is_required:
                penalizer.filter(keep_indices=keep_indices)
            else:
                penalizer.teardown()
        self.is_required = is_required


    def release(self) -> None:

        for penalizer in self.penalizers.values():
            penalizer.teardown()
        self.penalizers.clear()

        self._batch_ref = None
        self.is_required = False


    def __enter__(self) -> BatchedPenalizerOrchestrator:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def merge(self, their: BatchedPenalizerOrchestrator):

        if not self.is_required and not their.is_required:
            return

        self.is_required = True
        for penalizer, their_penalizer in their.penalizers.items():
            self.penalizers[penalizer].merge(their_penalizer)


class _BatchedPenalizer(abc.ABC):


    is_multiplicative: bool = False

    def __init__(self, orchestrator: BatchedPenalizerOrchestrator):
        self._orchestrator_ref: weakref.ReferenceType[BatchedPenalizerOrchestrator] = (
            weakref.ref(orchestrator)
        )
        self._is_prepared = False

    @property
    def orchestrator(self) -> BatchedPenalizerOrchestrator:
        orch: Optional[BatchedPenalizerOrchestrator] = self._orchestrator_ref()

        if orch is None:
            raise RuntimeError(
                "BatchedPenalizerOrchestrator has been garbage-collected"
            )
        return orch

    def is_prepared(self) -> bool:
        return self._is_prepared

    def is_required(self) -> bool:
        return self._is_required()

    def prepare(self):
        if not self._is_prepared:
            self._prepare()
            self._is_prepared = True

    def prepare_if_required(self):
        if self._is_required():
            self.prepare()
            return True
        else:
            return False

    def teardown(self):
        self._teardown()
        self._is_prepared = False

    def cumulate_output_tokens(self, output_ids: torch.Tensor):
        if not self._is_prepared:
            return

        self._cumulate_output_tokens(output_ids=output_ids)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self._is_prepared:
            return

        self._apply(logits=logits)

    def filter(self, keep_indices: torch.Tensor):
        if not self._is_prepared:
            return

        self._filter(keep_indices=keep_indices)

    def merge(self, their: _BatchedPenalizer):
        if not self._is_prepared and not their._is_prepared:
            return

        self.prepare()
        their.prepare()
        self._merge(their)

    @abc.abstractmethod
    def _is_required(self) -> bool:

        pass

    @abc.abstractmethod
    def _prepare(self):

        pass

    @abc.abstractmethod
    def _cumulate_output_tokens(self, output_ids: torch.Tensor):

        pass

    @abc.abstractmethod
    def _apply(self, logits: torch.Tensor) -> torch.Tensor:

        pass

    def get_scaling_penalties(self) -> torch.Tensor:

        raise NotImplementedError

    @abc.abstractmethod
    def _filter(self, keep_indices: torch.Tensor):

        pass

    @abc.abstractmethod
    def _merge(self, their: _BatchedPenalizer):

        pass

    @abc.abstractmethod
    def _teardown(self):

        pass
