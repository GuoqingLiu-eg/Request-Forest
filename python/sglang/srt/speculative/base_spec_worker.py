from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.model_executor.graph_memory_usage import (
    merge_graph_memory_usage,
    merge_graph_time_usage,
)
from sglang.srt.runtime_context import get_disagg, get_exec, get_memory, get_schedule

if TYPE_CHECKING:
    from sglang.srt.managers.io_struct import (
        UpdateWeightFromDiskReqInput,
        UpdateWeightsFromIPCReqInput,
    )
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


class HiCacheDraftMode(str, Enum):
    NONE = "none"
    PACKED = "packed"
    SIDECAR = "sidecar"


@dataclass(frozen=True, slots=True)
class HiCacheDraftPlan:
    mode: HiCacheDraftMode = HiCacheDraftMode.NONE
    device_pools: tuple[object, ...] = ()


def _can_pack_hicache_mtp(
    spec_algorithm: SpeculativeAlgorithm,
    draft_runners: tuple[ModelRunner, ...],
) -> bool:
    is_nextn_mtp = (
        spec_algorithm.is_eagle()
        and not spec_algorithm.is_eagle3()
        and all(
            runner.model_config.num_nextn_predict_layers for runner in draft_runners
        )
    )
    is_dspark_dsv4 = (
        spec_algorithm.is_dspark()
        and draft_runners[0].model_config.hf_config.architectures[0]
        == "DeepseekV4ForCausalLMDSpark"
    )
    return is_nextn_mtp or is_dspark_dsv4


class EagleDraftWorkerBase(ABC):

    _topk1_parents_prealloc: Optional[torch.Tensor] = None
    _topk1_score_indices_prealloc: Optional[torch.Tensor] = None

    def __init__(self) -> None:
        self._specialized_graph_memory_usage: dict[str, float] = {}
        self._specialized_graph_time_usage: dict[str, float] = {}

    @abstractmethod
    def draft():
        pass

    @abstractmethod
    def draft_extend():
        pass

    @property
    def draft_runners(self) -> list[ModelRunner]:

        return [self.draft_runner]

    @property
    def graph_memory_usage(self) -> dict[str, float]:
        return merge_graph_memory_usage(
            *(runner.graph_memory_usage for runner in self.draft_runners),
            self._specialized_graph_memory_usage,
        )

    @property
    def graph_time_usage(self) -> dict[str, float]:
        return merge_graph_time_usage(
            *(runner.graph_time_usage for runner in self.draft_runners),
            self._specialized_graph_time_usage,
        )

    @property
    def weight_load_time(self) -> float:
        return sum(runner.weight_load_time for runner in self.draft_runners)

    def alloc_memory_pool(self, **kwargs):
        pass

    def init_attention_backends(self):

        self.draft_worker.init_attention_backends()
        self.init_attention_backend()

    def init_cuda_graphs(self):

        self.draft_worker.init_cuda_graphs(capture_decode_cuda_graph=False)
        self._capture_cuda_graphs()

    def _rebuild_topk1_chain_buffers(self) -> None:


        if self.topk != 1:
            return


        assert self.speculative_num_draft_tokens == self.speculative_num_steps + 1, (
            "topk=1 requires speculative_num_draft_tokens == speculative_num_steps + 1, "
            f"got {self.speculative_num_draft_tokens} and {self.speculative_num_steps}"
        )
        num_steps = self.speculative_num_steps
        sa = self.server_args
        decode_max_bs = (
            get_exec().graph.cuda_graph_config.decode.max_bs
            if get_exec().graph.cuda_graph_config is not None
            else None
        )
        max_bs = max(
            decode_max_bs or 0,
            get_schedule().max_running_requests or 0,
            1,
        )


        parent_width = num_steps if num_steps > 1 else 0
        self._topk1_parents_prealloc = torch.arange(
            -1, parent_width - 1, dtype=torch.long, device=self.device
        ).repeat(max_bs, 1)
        self._topk1_score_indices_prealloc = torch.arange(
            num_steps, dtype=torch.long, device=self.device
        ).repeat(max_bs, 1)


class BaseSpecWorker(ABC):
    _hicache_draft_plan = HiCacheDraftPlan()

    def __init__(self) -> None:
        self._additional_graph_memory_usage: dict[str, float] = {}
        self._additional_graph_time_usage: dict[str, float] = {}

    @property
    def hicache_draft_plan(self) -> HiCacheDraftPlan:
        return self._hicache_draft_plan

    def _draft_model_runners(self) -> tuple[ModelRunner, ...]:
        spec_algorithm = self.target_worker.model_runner.spec_algorithm
        draft_worker = self.draft_worker
        if (
            draft_worker is None
            or spec_algorithm.is_ngram()
            or spec_algorithm.is_frozen_kv_mtp()
        ):
            return ()
        if spec_algorithm.uses_dflash_drafter():
            return (draft_worker.model_runner,)
        return tuple(draft_worker.draft_runners)

    @property
    def primary_draft_kv_pool(self) -> Optional[object]:
        draft_runners = self._draft_model_runners()
        return draft_runners[0].token_to_kv_pool if draft_runners else None

    @property
    def target_worker(self) -> TpModelWorker:
        return self._target_worker

    @property
    def draft_worker(self) -> Optional[EagleDraftWorkerBase | TpModelWorker]:


        return self._draft_worker

    @property
    def graph_memory_usage(self) -> dict[str, float]:
        if self.draft_worker is None:
            draft_memory_usage = None
        else:
            draft_memory_usage = self.draft_worker.graph_memory_usage
        return merge_graph_memory_usage(
            draft_memory_usage,
            self._additional_graph_memory_usage,
        )

    @property
    def graph_time_usage(self) -> dict[str, float]:
        if self.draft_worker is None:
            draft_time_usage = None
        else:
            draft_time_usage = self.draft_worker.graph_time_usage
        return merge_graph_time_usage(
            draft_time_usage,
            self._additional_graph_time_usage,
        )

    @property
    def weight_load_time(self) -> float:
        if self.draft_worker is None:
            return 0.0
        return self.draft_worker.weight_load_time

    @property
    def last_shared_read_runner(self):


        return self.target_worker.model_runner

    @property
    def spec_v2_attn_backends(self) -> tuple:

        return (self.target_worker.model_runner.attn_backend,)

    def clear_cache_pool(self):


        pass

    def _build_hicache_draft_plan(self) -> HiCacheDraftPlan:
        target_model_runner = self.target_worker.model_runner
        target_model_runner.mtp_draft_device_pools = ()
        spec_algorithm = target_model_runner.spec_algorithm
        if not (
            get_memory().enable_hierarchical_cache
            or get_disagg().disaggregation_decode_retraction_backup == "host_pool"
        ):
            return HiCacheDraftPlan()

        draft_runners = self._draft_model_runners()
        if not draft_runners:
            return HiCacheDraftPlan()
        draft_pools = tuple(runner.token_to_kv_pool for runner in draft_runners)
        if (
            "InklingForConditionalGenerationMTP"
            in draft_runners[0].model_config.hf_config.architectures
        ):
            raise NotImplementedError(
                "HiCache does not support Inkling MTP draft state yet."
            )

        if _can_pack_hicache_mtp(spec_algorithm, draft_runners):
            target_model_runner.mtp_draft_device_pools = draft_pools
            return HiCacheDraftPlan(
                mode=HiCacheDraftMode.PACKED,
                device_pools=draft_pools,
            )

        return HiCacheDraftPlan(
            mode=HiCacheDraftMode.SIDECAR,


            device_pools=draft_pools[:1],
        )

    def init_hicache_draft_plan(self) -> None:
        self._hicache_draft_plan = self._build_hicache_draft_plan()

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        if self.draft_worker is not None:
            self.draft_worker.alloc_memory_pool(
                memory_pool_config=memory_pool_config,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            )
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator

    def init_attention_backends(self):
        if self.draft_worker is not None:
            self.draft_worker.init_attention_backends()

    def init_cuda_graphs(self):
        if self.draft_worker is not None:
            self.draft_worker.init_cuda_graphs()

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        for runner in self.draft_worker.draft_runners:
            success, message = runner.weight_updater.update_weights_from_disk(
                recv_req.model_path,
                recv_req.load_format,
                recapture_cuda_graph=recv_req.recapture_cuda_graph,
            )
            if not success:
                return success, message
        return True, "Succeeded to update model weights."

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        for runner in self.draft_worker.draft_runners:
            success, message = runner.weight_updater.update_weights_from_ipc(recv_req)
            if not success:
                return success, message
        return True, "Succeeded to update model weights."

    def on_verify_complete_cpu(
        self, num_correct_drafts_per_req: list[int], batch_size: int = 0
    ) -> None:

        pass

    def note_request_finished(self, *, rid: str, natural_stop: bool) -> None:

        pass

    def activate_step_by_batch(self, batch_size: int) -> None:

        pass
