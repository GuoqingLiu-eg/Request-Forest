from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Callable, Dict, Optional, Type

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.spec_info import SpecInput

WorkerFactory = Callable[["ServerArgs"], Type]
ServerArgsValidator = Callable[["ServerArgs"], None]

logger = logging.getLogger(__name__)


class CustomSpecAlgo:


    def __init__(
        self,
        name: str,
        factory: WorkerFactory,
        *,
        supports_overlap: bool = False,
        validate_server_args: Optional[ServerArgsValidator] = None,
    ):
        self.name = name
        self.factory = factory
        self.supports_overlap = supports_overlap
        self.validate_server_args = validate_server_args

    def __repr__(self) -> str:
        return f"CustomSpecAlgo({self.name!r})"

    def is_some(self) -> bool:
        return True

    def is_none(self) -> bool:
        return False

    def is_speculative(self) -> bool:
        return True

    def is_eagle(self) -> bool:
        return False

    def is_eagle3(self) -> bool:
        return False

    def is_frozen_kv_mtp(self) -> bool:
        return False

    def is_dflash(self) -> bool:
        return False

    def is_dspark(self) -> bool:
        return False

    def is_dflash_family(self) -> bool:
        return False

    def is_request_forest(self) -> bool:
        return False

    def uses_dflash_drafter(self) -> bool:
        return False

    def is_standalone(self) -> bool:
        return False

    def is_ngram(self) -> bool:
        return False

    def supports_target_verify_for_draft(self) -> bool:
        return False

    def supports_ragged_verify(self) -> bool:
        return False

    def supports_grammar_overlap(self) -> bool:


        return False

    def has_draft_kv(self) -> bool:

        return True

    def handle_server_args(self, server_args: ServerArgs) -> None:
        pass

    def create_worker(self, server_args: ServerArgs) -> Type:
        from sglang.srt.arg_groups.overrides import resolving_view

        cfg = resolving_view(server_args)
        if not cfg.disable_overlap_schedule and not self.supports_overlap:
            raise ValueError(
                f"Speculative algorithm {self.name} does not support overlap scheduling."
            )
        if not self.supports_overlap:


            logger.warning(
                "Speculative algorithm %s is registered with "
                "supports_overlap=False, which is deprecated: the spec V1 "
                "worker path has been removed, and the algorithm now runs on "
                "the V2 scheduler schema with overlap disabled (synchronous). "
                "Migrate the plugin worker to support overlap scheduling.",
                self.name,
            )
        return self.factory(server_args)

    def get_num_tokens_per_req_for_target_verify(
        self, num_draft_tokens: int, is_draft_worker: bool
    ) -> int:


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

    def build_disagg_draft_input(
        self,
        batch: ScheduleBatch,
        last_tokens_tensor: torch.Tensor,
        future_map: FutureMap,
    ) -> Optional[SpecInput]:

        return None


_REGISTRY: Dict[str, CustomSpecAlgo] = {}


_RESERVED_ALIASES = frozenset({"NEXTN"})


def _reserved_names() -> frozenset:

    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    return frozenset(algo.name for algo in SpeculativeAlgorithm) | _RESERVED_ALIASES


def _assert_custom_spec_algo_conforms(spec_class: Type[CustomSpecAlgo]) -> None:


    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    interface = {
        name
        for name in vars(SpeculativeAlgorithm)
        if name.startswith(("is_", "supports_"))
    }
    missing = sorted(interface - set(dir(spec_class)))
    if missing:
        raise TypeError(
            f"{spec_class.__name__} is missing duck-typed methods from "
            f"SpeculativeAlgorithm: {missing}. Add them to {spec_class.__name__} "
            "so plugin-registered algorithms stay dispatchable."
        )


def register_algorithm(
    name: str,
    *,
    supports_overlap: bool = False,
    validate_server_args: Optional[ServerArgsValidator] = None,
    spec_class: Type[CustomSpecAlgo] = CustomSpecAlgo,
) -> Callable[[WorkerFactory], WorkerFactory]:

    upper = name.upper()
    if upper in _reserved_names():
        raise ValueError(
            f"'{upper}' is a reserved speculative algorithm name; cannot be re-registered."
        )
    if upper in _REGISTRY:
        raise ValueError(f"Speculative algorithm '{upper}' already registered.")
    _assert_custom_spec_algo_conforms(spec_class)

    def decorator(factory: WorkerFactory) -> WorkerFactory:
        _REGISTRY[upper] = spec_class(
            name=upper,
            factory=factory,
            supports_overlap=supports_overlap,
            validate_server_args=validate_server_args,
        )
        return factory

    return decorator


def get_spec(name: Optional[str]) -> Optional[CustomSpecAlgo]:

    if name is None:
        return None
    return _REGISTRY.get(name.upper())
