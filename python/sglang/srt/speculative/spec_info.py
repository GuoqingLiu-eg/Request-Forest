from __future__ import annotations

import warnings
from abc import ABC
from enum import Enum, IntEnum, auto
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Type, Union

import torch

from sglang.srt.runtime_context import get_spec as get_spec_config
from sglang.srt.speculative.spec_registry import (
    CustomSpecAlgo,
    ServerArgsValidator,
    WorkerFactory,
)
from sglang.srt.speculative.spec_registry import get_spec as _get_registered_spec
from sglang.srt.speculative.spec_registry import (
    register_algorithm as _register_algorithm,
)

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
    from sglang.srt.speculative.ngram_worker import NGRAMWorker
    from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout


class SpeculativeAlgorithm(Enum):


    DFLASH = auto()
    DSPARK = auto()
    EAGLE = auto()
    EAGLE3 = auto()
    FROZEN_KV_MTP = auto()
    STANDALONE = auto()
    NGRAM = auto()
    REQUEST_FOREST = auto()
    NONE = auto()

    @classmethod
    def from_string(
        cls, name: Optional[str]
    ) -> Union[SpeculativeAlgorithm, CustomSpecAlgo]:
        if name is None:
            return cls.NONE
        upper = name.upper()
        try:
            return cls[upper]
        except KeyError:
            pass
        spec = _get_registered_spec(upper)
        if spec is not None:
            return spec
        raise ValueError(f"Unknown speculative algorithm name: {name}")

    @classmethod
    def register(
        cls,
        name: str,
        *,
        supports_overlap: bool = False,
        validate_server_args: Optional[ServerArgsValidator] = None,
        spec_class: Type[CustomSpecAlgo] = CustomSpecAlgo,
    ) -> Callable[[WorkerFactory], WorkerFactory]:

        return _register_algorithm(
            name,
            supports_overlap=supports_overlap,
            validate_server_args=validate_server_args,
            spec_class=spec_class,
        )

    def is_some(self) -> bool:
        return self != SpeculativeAlgorithm.NONE

    def is_none(self) -> bool:
        return self == SpeculativeAlgorithm.NONE

    def is_speculative(self) -> bool:
        return self != SpeculativeAlgorithm.NONE

    def is_eagle(self) -> bool:


        return self in (
            SpeculativeAlgorithm.EAGLE,
            SpeculativeAlgorithm.EAGLE3,
            SpeculativeAlgorithm.FROZEN_KV_MTP,
        )

    def is_eagle3(self) -> bool:
        return self == SpeculativeAlgorithm.EAGLE3

    def is_frozen_kv_mtp(self) -> bool:
        return self == SpeculativeAlgorithm.FROZEN_KV_MTP

    def is_dflash(self) -> bool:
        return self == SpeculativeAlgorithm.DFLASH

    def is_dspark(self) -> bool:
        return self == SpeculativeAlgorithm.DSPARK

    def is_dflash_family(self) -> bool:
        return self.is_dflash() or self.is_dspark()

    def is_request_forest(self) -> bool:
        return self == SpeculativeAlgorithm.REQUEST_FOREST

    def uses_dflash_drafter(self) -> bool:


        return self.is_dflash_family() or self.is_request_forest()

    def is_standalone(self) -> bool:
        return self == SpeculativeAlgorithm.STANDALONE

    def is_ngram(self) -> bool:
        return self == SpeculativeAlgorithm.NGRAM

    def supports_target_verify_for_draft(self) -> bool:
        return self.uses_dflash_drafter()

    def supports_ragged_verify(self) -> bool:

        return self.is_dspark() or self.is_request_forest()

    def supports_grammar_overlap(self) -> bool:


        return self.is_eagle() or self.is_standalone() or self.is_dflash_family()

    def has_draft_kv(self) -> bool:

        return not self.is_ngram()

    def carries_draft_hidden_states(self) -> bool:

        return self.is_eagle()

    def create_future_map(
        self,
        device: torch.device,
        req_to_token_pool,
        needs_cpu_seq_lens: bool = True,
        needs_confidence_relay: bool = False,
    ) -> FutureMap:
        from sglang.srt.managers.overlap_utils import FutureMap

        return FutureMap(
            device,
            self,
            req_to_token_pool,
            needs_cpu_seq_lens,
            needs_confidence_relay,
        )

    def build_disagg_draft_input(
        self,
        batch: ScheduleBatch,
        last_tokens_tensor: torch.Tensor,
        future_map: FutureMap,
    ) -> Optional[SpecInput]:
        if self.is_eagle():
            from sglang.srt.speculative.eagle_disaggregation import (
                build_eagle_disagg_draft_input,
            )

            return build_eagle_disagg_draft_input(batch, last_tokens_tensor, future_map)
        if self.is_dspark():
            from sglang.srt.speculative.dspark_disaggregation import (
                build_dspark_disagg_draft_input,
            )

            return build_dspark_disagg_draft_input(
                batch, last_tokens_tensor, future_map
            )
        return None

    def need_topk(self) -> bool:
        return self.is_eagle() or self.is_standalone()

    def handle_server_args(self, server_args: ServerArgs) -> None:

        from sglang.srt.arg_groups.speculative_hook import (
            _handle_dflash,
            _handle_dspark,
            _handle_eagle_family,
            _handle_frozen_kv_mtp,
            _handle_ngram,
            _handle_request_forest,
        )


        from sglang.srt.speculative.ragged_verify import read_ragged_verify_mode

        read_ragged_verify_mode()

        if self.is_dflash():
            _handle_dflash(server_args)
        elif self.is_dspark():
            _handle_dspark(server_args)
        elif self.is_frozen_kv_mtp():
            _handle_frozen_kv_mtp(server_args)
        elif self.is_eagle() or self.is_standalone():
            _handle_eagle_family(server_args)
        elif self.is_ngram():
            _handle_ngram(server_args)
        elif self.is_request_forest():
            _handle_request_forest(server_args)

    def get_num_tokens_per_req_for_target_verify(
        self, num_draft_tokens: int, is_draft_worker: bool
    ) -> int:


        if self.is_dspark() and is_draft_worker:
            return num_draft_tokens - 1
        return num_draft_tokens

    def get_num_tokens_per_bs_for_target_verify(
        self, num_draft_tokens: int, is_draft_worker: bool
    ) -> int:

        warnings.warn(
            "get_num_tokens_per_bs_for_target_verify is deprecated; use "
            "get_num_tokens_per_req_for_target_verify instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_num_tokens_per_req_for_target_verify(
            num_draft_tokens, is_draft_worker
        )

    def create_worker(
        self, server_args: ServerArgs
    ) -> Optional[Union[Type[BaseSpecWorker], Type[TpModelWorker], Type[NGRAMWorker]]]:
        from sglang.srt.arg_groups.overrides import resolving_view

        cfg = resolving_view(server_args)
        assert (
            not self.is_none()
        ), "Cannot create worker for NONE speculative algorithm."

        if self.is_request_forest():
            from sglang.srt.speculative.request_forest.adaptive_config import (
                adaptive_rf_enabled,
                validate_adaptive_config,
            )

            if adaptive_rf_enabled():
                validate_adaptive_config(cfg)
                from sglang.srt.speculative.request_forest.adaptive_worker import (
                    AdaptiveRequestForestWorker,
                )

                return AdaptiveRequestForestWorker
            from sglang.srt.speculative.request_forest.worker import (
                RequestForestWorkerV2,
            )

            return RequestForestWorkerV2

        if self.is_dflash():


            from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

            return DFlashWorkerV2

        if self.is_dspark():
            from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
                DSparkWorkerV2,
            )

            return DSparkWorkerV2

        if self.is_frozen_kv_mtp():


            from sglang.srt.speculative.frozen_kv_mtp_worker_v2 import (
                FrozenKVMTPWorkerV2,
            )

            return FrozenKVMTPWorkerV2


        if self.is_eagle() and cfg.enable_multi_layer_eagle:
            from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
                MultiLayerEagleWorkerV2,
            )

            return MultiLayerEagleWorkerV2

        elif self.is_eagle():
            from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2

            return EAGLEWorkerV2
        elif self.is_standalone():
            from sglang.srt.speculative.standalone_worker_v2 import StandaloneWorkerV2

            return StandaloneWorkerV2
        elif self.is_ngram():
            from sglang.srt.speculative.ngram_worker import NGRAMWorker

            return NGRAMWorker

        raise ValueError("Unreachable code path in create_worker.")


class SpecInputType(IntEnum):


    EAGLE_DRAFT = auto()
    EAGLE_DRAFT_EXTEND = auto()
    EAGLE_VERIFY = auto()
    FROZEN_KV_MTP_DRAFT = auto()
    FROZEN_KV_MTP_VERIFY = auto()
    DFLASH_DRAFT = auto()
    DFLASH_VERIFY = auto()
    NGRAM_VERIFY = auto()
    REQUEST_FOREST_DRAFT = auto()
    REQUEST_FOREST_VERIFY = auto()


class SpecInput(ABC):


    ragged_verify_layout: Optional[RaggedVerifyLayout] = None


    num_tokens_per_req: int = -1
    num_tokens_for_logprob_per_req: int = -1


    dsa_topk_indices: Optional[torch.Tensor] = None
    future_dsa_topk_indices_available: bool = False
    dsa_seed_topk_capture: Optional[torch.Tensor] = None

    def __init__(self, spec_input_type: SpecInputType):
        self.spec_input_type = spec_input_type


    def is_draft_input(self) -> bool:
        return self.spec_input_type in {
            SpecInputType.EAGLE_DRAFT,
            SpecInputType.EAGLE_DRAFT_EXTEND,
            SpecInputType.FROZEN_KV_MTP_DRAFT,
            SpecInputType.DFLASH_DRAFT,
            SpecInputType.REQUEST_FOREST_DRAFT,
        }

    def is_verify_input(self) -> bool:
        return self.spec_input_type in {
            SpecInputType.EAGLE_VERIFY,
            SpecInputType.FROZEN_KV_MTP_VERIFY,
            SpecInputType.DFLASH_VERIFY,
            SpecInputType.NGRAM_VERIFY,
            SpecInputType.REQUEST_FOREST_VERIFY,
        }

    def is_request_forest_verify_input(self) -> bool:
        return self.spec_input_type == SpecInputType.REQUEST_FOREST_VERIFY

    def is_dflash_verify_input(self) -> bool:
        return self.spec_input_type == SpecInputType.DFLASH_VERIFY


def spec_scale_global_num_tokens(
    spec_info: SpecInput,
    global_num_tokens: List[int],
    global_num_tokens_for_logprob: List[int],
) -> Tuple[List[int], List[int]]:

    return (
        [x * spec_info.num_tokens_per_req for x in global_num_tokens],
        [
            x * spec_info.num_tokens_for_logprob_per_req
            for x in global_num_tokens_for_logprob
        ],
    )


def create_dummy_verify_input(
    spec_algorithm: SpeculativeAlgorithm,
    custom_mask: torch.Tensor,
    num_tokens_per_req: int,
    is_draft_worker: bool,
) -> Optional[SpecInput]:

    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

    spec = get_spec_config()

    spec_info = None
    if spec_algorithm.is_eagle() or spec_algorithm.is_standalone():
        from sglang.srt.speculative.eagle_info import EagleVerifyInput

        if is_draft_worker:
            raise RuntimeError("This should not happen.")
        else:
            spec_info = EagleVerifyInput(
                draft_token=None,
                custom_mask=custom_mask,
                positions=None,
                retrieve_index=None,
                retrieve_next_token=None,
                retrieve_next_sibling=None,
                retrieve_cum_len=None,
                spec_steps=spec.speculative_num_steps,
                topk=spec.speculative_eagle_topk,
                draft_token_num=spec.speculative_num_draft_tokens,
                capture_hidden_mode=CaptureHiddenMode.FULL,
                seq_lens_sum=None,
                seq_lens_cpu=None,
            )
    elif spec_algorithm.uses_dflash_drafter():
        from sglang.srt.speculative.dflash_info import DFlashVerifyInput


        spec_info = DFlashVerifyInput(
            draft_token=None,
            positions=None,
            draft_token_num=spec.speculative_num_draft_tokens,
            custom_mask=None,
            capture_hidden_mode=(
                CaptureHiddenMode.NULL if is_draft_worker else CaptureHiddenMode.FULL
            ),
        )

    elif spec_algorithm.is_ngram():
        from sglang.srt.speculative.ngram_info import NgramVerifyInput

        spec_info = NgramVerifyInput(
            draft_token=None,
            custom_mask=custom_mask,
            positions=None,
            retrieve_index=None,
            retrieve_next_token=None,
            retrieve_next_sibling=None,
            draft_token_num=num_tokens_per_req,
        )
        spec_info.capture_hidden_mode = CaptureHiddenMode.NULL

    return spec_info
