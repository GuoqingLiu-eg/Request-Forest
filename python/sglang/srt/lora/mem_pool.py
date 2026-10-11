import logging
import re
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    overload,
)

import torch

from sglang.srt.distributed import (
    divide,
    get_pp_group,
)
from sglang.srt.environ import envs
from sglang.srt.lora.eviction_policy import get_eviction_policy
from sglang.srt.lora.layers import BaseLayerWithLoRA
from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.lora.utils import (
    ATTN_TP_LORA_MODULE_NAMES,
    EMBEDDING_NAMES,
    REPLICATED_LINEAR_LORA_NAMES,
    ROW_PARALLELISM_LINEAR_LORA_NAMES,
    LoRAType,
    copy_weight_into_buffer,
    get_hidden_dim,
    get_lm_head_lora_b_shard_size,
    get_normalized_target_modules,
    get_stacked_multiply,
    get_target_module_name,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_pin_memory_available
from sglang.srt.utils.hf_transformers_utils import AutoConfig

_SGLANG_EXPERIMENTAL_LORA_OPTI = envs.SGLANG_EXPERIMENTAL_LORA_OPTI.get()

logger = logging.getLogger(__name__)


class EmptySlot:


    __slots__ = ()

    def __repr__(self):
        return "|EMPTY|"

    def __new__(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance


EMPTY_SLOT = EmptySlot()


@overload
def append_cache_key_suffix(cache_keys: str, suffix: str) -> str: ...


@overload
def append_cache_key_suffix(
    cache_keys: Dict[int, str], suffix: str
) -> Dict[int, str]: ...


def append_cache_key_suffix(
    cache_keys: Union[str, Dict[int, str]],
    suffix: str,
) -> Union[str, Dict[int, str]]:
    if isinstance(cache_keys, dict):
        return {
            expert_id: f"{cache_key}#{suffix}"
            for expert_id, cache_key in cache_keys.items()
        }
    return f"{cache_keys}#{suffix}"


def _get_moe_ep_context() -> Tuple[int, int]:

    try:
        return get_parallel().moe_ep_size, get_parallel().moe_ep_rank
    except Exception:
        return 1, 0


def _get_moe_tp_context() -> Tuple[int, int]:

    try:
        return get_parallel().moe_tp_size, get_parallel().moe_tp_rank
    except Exception:
        return 1, 0


def _moe_runner_keeps_global_expert_ids() -> bool:

    try:
        from sglang.srt.layers.moe.utils import get_moe_runner_backend

        b = get_moe_runner_backend()
        return (
            b.is_flashinfer_cutlass()
            or b.is_flashinfer_cutedsl()
            or b.is_flashinfer_trtllm()
            or b.is_flashinfer_trtllm_routed()
            or b.is_flashinfer_mxfp4()
            or b.is_hpc_ops()
        )
    except Exception:
        return False


class LoRAMemoryPool:


    def __init__(
        self,
        base_hf_config: AutoConfig,
        max_loras_per_batch: int,
        dtype: torch.dtype,
        tp_size: int,
        tp_rank: int,
        attn_tp_size: int,
        max_lora_rank: int,
        target_modules: Set[str],
        base_model: torch.nn.Module,
        eviction_policy: str,
        lora_added_tokens_size: int,
        experts_shared_outer_loras: bool = False,
        strict_loading: bool = False,
        enable_lora_overlap_loading: bool = False,
    ):
        self.base_hf_config: AutoConfig = base_hf_config
        self.num_layer: int = base_hf_config.num_hidden_layers
        self.max_loras_per_batch: int = max_loras_per_batch
        self.dtype: torch.dtype = dtype
        self.tp_size: int = tp_size
        self.tp_rank: int = tp_rank
        self.lora_added_tokens_size: int = lora_added_tokens_size
        self.max_lora_rank: int = max_lora_rank
        self.target_modules: Set[str] = target_modules
        self.experts_shared_outer_loras: bool = experts_shared_outer_loras
        self.strict_loading: bool = strict_loading
        self.enable_lora_overlap_loading: bool = enable_lora_overlap_loading
        self.pin_memory_available: bool = is_pin_memory_available()


        self.moe_ep_size, self.moe_ep_rank = _get_moe_ep_context()
        num_experts_global = self._get_num_experts(base_model)
        self.moe_use_local_expert_ids = (
            self.moe_ep_size > 1
            and not _moe_runner_keeps_global_expert_ids()
            and num_experts_global % self.moe_ep_size == 0
        )


        self.moe_tp_size, self.moe_tp_rank = _get_moe_tp_context()


        self.attn_tp_size: int = attn_tp_size


        self.eviction_policy = get_eviction_policy(eviction_policy)


        self.A_buffer: Dict[str, List[torch.Tensor]] = {}
        self.B_buffer: Dict[str, List[torch.Tensor]] = {}

        self.embedding_A_buffer: Dict[str, torch.Tensor] = {}
        self.embedding_B_buffer: Dict[str, torch.Tensor] = {}

        self.lm_head_A_buffer: Dict[str, torch.Tensor] = {}
        self.lm_head_B_buffer: Dict[str, torch.Tensor] = {}
        self.new_embeddings_buffer: Dict[str, torch.Tensor] = {}

        self.embedding_dim: int = self.base_hf_config.hidden_size


        self.uid_to_buffer_id: Dict[Optional[str], int] = {}


        self.buffer_id_to_uid: List[Union[str, None, EmptySlot]] = [
            EMPTY_SLOT
        ] * self.max_loras_per_batch


        self.lm_head_shard_indices = None
        if "lm_head" in target_modules and tp_size > 1:
            from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

            for _, module in base_model.named_modules():
                if isinstance(module, ParallelLMHead):
                    self.lm_head_shard_indices = module.shard_indices
                    break

        self.init_buffers(base_model)

    def can_support(self, config: Union[LoRAConfig, Iterable[LoRAConfig]]) -> bool:


        def _can_support(config: LoRAConfig) -> bool:

            if config.r > self.max_lora_rank:
                return False
            if config.lora_added_tokens_size > self.lora_added_tokens_size:
                return False
            target_module_names = get_normalized_target_modules(config.target_modules)
            if "all" in target_module_names:
                return True
            return target_module_names.issubset(self.target_modules)

        if isinstance(config, LoRAConfig):
            return _can_support(config)
        else:
            return all(_can_support(x) for x in config)

    def is_moe_module(self, module_name: str) -> bool:

        return "moe" in module_name

    @staticmethod
    def is_shared_moe_module(module_name: str) -> bool:

        return module_name.endswith("_shared_moe")

    def _effective_tp_size(self, module_name: str) -> int:

        if self.is_moe_module(module_name) and not self.is_shared_moe_module(
            module_name
        ):
            return self.moe_tp_size
        if module_name in ATTN_TP_LORA_MODULE_NAMES:
            return self.attn_tp_size
        return self.tp_size

    @staticmethod
    def _get_num_experts(base_model: torch.nn.Module) -> int:
        cfg = base_model.config
        if hasattr(cfg, "get_text_config"):
            cfg = cfg.get_text_config()
        return (
            getattr(cfg, "num_experts", None)
            or getattr(cfg, "num_local_experts", None)
            or getattr(cfg, "n_routed_experts", None)
            or 1
        )

    @staticmethod
    def _get_num_shared_experts(base_model: torch.nn.Module) -> int:
        cfg = base_model.config
        if hasattr(cfg, "get_text_config"):
            cfg = cfg.get_text_config()
        return getattr(cfg, "n_shared_experts", 0) or 1

    @staticmethod
    def _has_shared_fused_moe(base_model: torch.nn.Module) -> bool:

        return any(
            getattr(m, "is_shared_fused_moe", False) for m in base_model.modules()
        )

    @staticmethod
    def _has_moe_module(base_model: torch.nn.Module) -> bool:


        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        return any(isinstance(m, FusedMoE) for m in base_model.modules())

    def _get_num_local_experts(self, base_model: torch.nn.Module) -> int:

        total = self._get_num_experts(base_model)
        if not self.moe_use_local_expert_ids:
            return total
        return total // self.moe_ep_size

    def _global_to_local_expert_id(self, global_eid: int) -> Optional[int]:

        if not self.moe_use_local_expert_ids:
            return global_eid
        local = global_eid - self.moe_ep_rank * self._num_experts_local
        return local if 0 <= local < self._num_experts_local else None

    def _iter_local_expert_weights(
        self,
        weights: Union[torch.Tensor, Dict[int, torch.Tensor]],
        cache_keys: Union[str, Dict[int, str]],
        *,
        localize: bool = True,
    ) -> Iterator[Tuple[int, torch.Tensor, str]]:

        if isinstance(weights, dict):
            assert isinstance(cache_keys, dict)
            for gid, w in weights.items():
                lid = self._global_to_local_expert_id(gid) if localize else gid
                if lid is not None:
                    yield lid, w, cache_keys[gid]
            return

        if isinstance(weights, torch.Tensor) and weights.dim() == 3:
            assert isinstance(cache_keys, str)
            total = weights.shape[0]
            if self.moe_use_local_expert_ids and localize:
                start = self.moe_ep_rank * self._num_experts_local
                count = max(0, min(self._num_experts_local, total - start))
            else:
                start, count = 0, total
            for i in range(count):
                yield (
                    i,
                    weights[start + i],
                    append_cache_key_suffix(cache_keys, f"expert{start + i}"),
                )
            return

        raise TypeError(
            f"Expected dict or 3D torch.Tensor, got {type(weights).__name__}."
        )

    def _get_standard_shape(
        self,
        module_name: str,
        base_model: torch.nn.Module,
        max_lora_dim: int,
        layer_idx: int,
    ) -> Tuple[int]:

        input_dim, _ = get_hidden_dim(
            module_name, self.base_hf_config, base_model, layer_idx
        )
        c = get_stacked_multiply(module_name, base_model)
        if self.tp_size > 1 and module_name in ROW_PARALLELISM_LINEAR_LORA_NAMES:
            input_dim = divide(input_dim, self.tp_size)
        return (self.max_loras_per_batch, max_lora_dim * c, input_dim)

    def get_lora_A_shape(
        self,
        module_name: str,
        base_model: torch.nn.Module,
        max_lora_dim: int,
        layer_idx: int,
    ) -> Tuple[int]:

        input_dim, _ = get_hidden_dim(
            module_name, self.base_hf_config, base_model, layer_idx
        )
        c = get_stacked_multiply(module_name, base_model)
        effective_tp_size = self._effective_tp_size(module_name)
        if (
            effective_tp_size > 1
            and module_name in ROW_PARALLELISM_LINEAR_LORA_NAMES
            and module_name not in REPLICATED_LINEAR_LORA_NAMES
        ):
            input_dim = divide(input_dim, effective_tp_size)

        if self.is_moe_module(module_name):
            if self.is_shared_moe_module(module_name):
                expert_dim = self._get_num_shared_experts(base_model)
            else:
                expert_dim = self._get_num_local_experts(base_model)
            if self.experts_shared_outer_loras and module_name in (
                "gate_up_proj_moe",
                "gate_up_proj_shared_moe",
            ):
                expert_dim = 1
            return (
                self.max_loras_per_batch,
                expert_dim,
                max_lora_dim * c,
                input_dim,
            )
        else:
            return (self.max_loras_per_batch, max_lora_dim * c, input_dim)

    def get_embedding_lora_A_shape(
        self,
        module_name: str,
        base_model: torch.nn.Module,
        max_lora_dim: int,
        layer_idx: int,
    ) -> Tuple[int]:
        input_dim, _ = get_hidden_dim(
            module_name, self.base_hf_config, base_model, 0, self.lora_added_tokens_size
        )


        return (
            self.max_loras_per_batch,
            max_lora_dim,
            input_dim,
        )

    def _column_parallel_lora_b_per_rank_dim(
        self,
        module_name: str,
        total_output_dim: int,
        effective_tp_size: int,
    ) -> int:

        if module_name != "qkv_proj":
            return divide(total_output_dim, effective_tp_size)

        cfg = self.base_hf_config
        if hasattr(cfg, "get_text_config"):
            cfg = cfg.get_text_config()
        num_kv_heads = getattr(cfg, "num_key_value_heads", None)
        if num_kv_heads is None or num_kv_heads >= effective_tp_size:
            return divide(total_output_dim, effective_tp_size)

        head_dim = getattr(cfg, "head_dim", None) or (
            cfg.hidden_size // cfg.num_attention_heads
        )
        kv_dim_total = 2 * num_kv_heads * head_dim
        q_dim_total = total_output_dim - kv_dim_total
        q_per_rank = divide(q_dim_total, effective_tp_size)
        return q_per_rank + 2 * head_dim

    def get_lora_B_shape(
        self,
        module_name: str,
        base_model: torch.nn.Module,
        max_lora_dim: int,
        layer_idx: int,
    ) -> Tuple[int]:

        _, output_dim = get_hidden_dim(
            module_name, self.base_hf_config, base_model, layer_idx
        )

        effective_tp_size = self._effective_tp_size(module_name)
        if (
            effective_tp_size > 1
            and module_name not in ROW_PARALLELISM_LINEAR_LORA_NAMES
            and module_name not in REPLICATED_LINEAR_LORA_NAMES
        ):
            output_dim = self._column_parallel_lora_b_per_rank_dim(
                module_name, output_dim, effective_tp_size
            )


        if self.is_moe_module(module_name):
            if self.is_shared_moe_module(module_name):
                expert_dim = self._get_num_shared_experts(base_model)
            else:
                expert_dim = self._get_num_local_experts(base_model)
            if self.experts_shared_outer_loras and module_name in (
                "down_proj_moe",
                "down_proj_shared_moe",
            ):
                expert_dim = 1
            return (self.max_loras_per_batch, expert_dim, output_dim, max_lora_dim)
        else:
            return (self.max_loras_per_batch, output_dim, max_lora_dim)

    def get_embedding_lora_B_shape(
        self,
        module_name: str,
        base_model: torch.nn.Module,
        max_lora_dim: int,
        layer_idx: int,
    ) -> Tuple[int]:
        _, output_dim = get_hidden_dim(
            module_name, self.base_hf_config, base_model, 0, self.lora_added_tokens_size
        )


        if module_name == "lm_head":
            output_dim = get_lm_head_lora_b_shard_size(
                output_dim,
                shard_indices=self.lm_head_shard_indices,
            )
        return (
            self.max_loras_per_batch,
            output_dim,
            max_lora_dim,
        )

    def init_buffers(self, base_model: torch.nn.Module):
        self.base_model = base_model
        device = next(base_model.parameters()).device


        self._num_experts_local: int = self._get_num_local_experts(base_model)

        def init_buffer(
            buffer: Dict[str, List[torch.Tensor]],
            target_modules: Set[str],
            get_lora_shape_fn: Callable[[str, torch.nn.Module, int, int], Tuple[int]],
        ):
            cfg = base_model.config
            if hasattr(cfg, "get_text_config"):
                cfg = cfg.get_text_config()
            has_shared_experts = (
                hasattr(cfg, "shared_expert_intermediate_size")
                and cfg.shared_expert_intermediate_size > 0
            ) or (getattr(cfg, "n_shared_experts", 0) or 0) > 0
            has_moe = self._has_moe_module(base_model)


            target_modules = target_modules - set(EMBEDDING_NAMES)
            for module_name in target_modules:

                ambiguous_modules = {"gate_up_proj", "down_proj"}
                if module_name in ambiguous_modules and has_moe:

                    if has_shared_experts:
                        buffer[module_name] = [
                            torch.zeros(
                                get_lora_shape_fn(
                                    module_name, base_model, self.max_lora_rank, idx
                                ),
                                dtype=self.dtype,
                                device=device,
                            )
                            for idx in range(self.num_layer)
                        ]


                    moe_key = f"{module_name}_moe"
                    buffer[moe_key] = [
                        torch.zeros(
                            get_lora_shape_fn(
                                moe_key, base_model, self.max_lora_rank, idx
                            ),
                            dtype=self.dtype,
                            device=device,
                        )
                        for idx in range(self.num_layer)
                    ]


                    if self._has_shared_fused_moe(base_model):
                        shared_moe_key = f"{module_name}_shared_moe"
                        buffer[shared_moe_key] = [
                            torch.zeros(
                                get_lora_shape_fn(
                                    shared_moe_key,
                                    base_model,
                                    self.max_lora_rank,
                                    idx,
                                ),
                                dtype=self.dtype,
                                device=device,
                            )
                            for idx in range(self.num_layer)
                        ]
                else:

                    buffer[module_name] = [
                        torch.zeros(
                            get_lora_shape_fn(
                                module_name,
                                base_model,
                                self.max_lora_rank,
                                idx,
                            ),
                            dtype=self.dtype,
                            device=device,
                        )
                        for idx in range(self.num_layer)
                    ]

        def init_embedding_buffer(
            buffer: Dict[str, torch.Tensor],
            target_modules: Set[str],
            get_lora_shape_fn: Callable[[int], Tuple[int]],
        ):
            target_modules = target_modules & set(EMBEDDING_NAMES)
            for module_name in target_modules:
                buffer[module_name] = torch.zeros(
                    get_lora_shape_fn(
                        module_name,
                        base_model,
                        self.max_lora_rank,
                        0,
                    ),
                    dtype=self.dtype,
                    device=device,
                )

        if self.lora_added_tokens_size > 0:
            self.new_embeddings_buffer["input_embeddings"] = torch.zeros(
                (
                    self.max_loras_per_batch,
                    self.lora_added_tokens_size,
                    self.embedding_dim,
                ),
                dtype=self.dtype,
                device=device,
            )

        if "embed_tokens" in self.target_modules:
            init_embedding_buffer(
                self.embedding_A_buffer,
                self.target_modules,
                self.get_embedding_lora_A_shape,
            )

            init_embedding_buffer(
                self.embedding_B_buffer,
                self.target_modules,
                self.get_embedding_lora_B_shape,
            )

        if "lm_head" in self.target_modules:
            init_embedding_buffer(
                self.lm_head_A_buffer,
                self.target_modules,
                self.get_embedding_lora_A_shape,
            )

            init_embedding_buffer(
                self.lm_head_B_buffer,
                self.target_modules,
                self.get_embedding_lora_B_shape,
            )

        init_buffer(
            self.A_buffer,
            self.target_modules,
            self.get_lora_A_shape,
        )

        init_buffer(
            self.B_buffer,
            self.target_modules,
            self.get_lora_B_shape,
        )

    def _get_maybe_cached_weight_for_transfer(
        self,
        pinned_weight_store: Dict[str, torch.Tensor],
        cache_key: str,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not self.pin_memory_available
            or weight.device.type != "cpu"
            or weight.is_pinned()
        ):
            return weight

        if not self.enable_lora_overlap_loading:
            return weight.pin_memory()

        cached_weight = pinned_weight_store.get(cache_key)
        if cached_weight is None:
            cached_weight = weight.pin_memory()
            pinned_weight_store[cache_key] = cached_weight
        elif cached_weight.shape != weight.shape or cached_weight.dtype != weight.dtype:
            raise ValueError(
                f"LoRA pinned weight cache key collision for {cache_key!r}: "
                f"cached shape={cached_weight.shape}, dtype={cached_weight.dtype}; "
                f"new shape={weight.shape}, dtype={weight.dtype}."
            )

        return cached_weight

    def prepare_lora_batch(
        self,
        cur_uids: Set[Optional[str]],
        lora_adapters: Dict[str, LoRAAdapter],
        lora_modules: List[Dict[str, torch.nn.Module]],
        lora_refs: Dict[str, LoRARef],
        lora_embed_tokens_module: Optional[BaseLayerWithLoRA],
        lora_lm_head_module: Optional[BaseLayerWithLoRA],
    ):

        ordered_uids = sorted(cur_uids, key=lambda uid: (uid is not None, uid or ""))

        def get_available_buffer_slot():

            for buffer_id in range(self.max_loras_per_batch):
                if self.buffer_id_to_uid[buffer_id] == EMPTY_SLOT:
                    return buffer_id


            candidates = set()

            for buffer_id in range(self.max_loras_per_batch):
                uid = self.buffer_id_to_uid[buffer_id]


                if uid in cur_uids:
                    continue


                if uid is not None:
                    lora_ref = lora_refs.get(uid)
                    if lora_ref and lora_ref.pinned:
                        continue

                candidates.add(uid)

            if not candidates:
                raise ValueError(
                    "No available buffer slots found. Please ensure the number of active (pinned) loras is less than max_loras_per_batch."
                )


            non_none_candidates = candidates - {None}
            if non_none_candidates:

                candidates_to_use = non_none_candidates
            else:

                candidates_to_use = candidates


            victim_uid = self.eviction_policy.select_victim(candidates_to_use)


            victim_buffer_id = self.uid_to_buffer_id[victim_uid]
            self.uid_to_buffer_id.pop(victim_uid)
            self.eviction_policy.remove(victim_uid)
            self.buffer_id_to_uid[victim_buffer_id] = EMPTY_SLOT
            logger.debug(
                f"Evicting LoRA {victim_uid} from buffer slot {victim_buffer_id}."
            )
            return victim_buffer_id


        for uid in ordered_uids:
            self.eviction_policy.mark_used(uid)

        for uid in ordered_uids:
            if uid not in self.uid_to_buffer_id:
                buffer_id = get_available_buffer_slot()
                lora_adapter = lora_adapters.get(uid, None)
                self.load_lora_weight_to_buffer(
                    uid,
                    buffer_id,
                    lora_adapter,
                    lora_modules,
                    lora_embed_tokens_module,
                    lora_lm_head_module,
                )
                self.uid_to_buffer_id[uid] = buffer_id
                self.buffer_id_to_uid[buffer_id] = uid

    def _clear_buffer_slot_for_base(self, buffer_id: int) -> None:

        for buffers in (*self.A_buffer.values(), *self.B_buffer.values()):
            for tensor in buffers:
                tensor[buffer_id].zero_()
        for buffers in (
            self.embedding_A_buffer,
            self.embedding_B_buffer,
            self.lm_head_A_buffer,
            self.lm_head_B_buffer,
            self.new_embeddings_buffer,
        ):
            for tensor in buffers.values():
                tensor[buffer_id].zero_()

    def remove_lora(self, uid: str) -> Optional[int]:

        buffer_id = self.uid_to_buffer_id.get(uid)
        if buffer_id is None:
            return None

        self._clear_buffer_slot_for_base(buffer_id)
        del self.uid_to_buffer_id[uid]
        self.buffer_id_to_uid[buffer_id] = EMPTY_SLOT
        self.eviction_policy.remove(uid)
        return buffer_id

    def load_lora_weight_to_buffer(
        self,
        uid: str,
        buffer_id: int,
        lora_adapter: LoRAAdapter,
        lora_modules: List[Dict[str, torch.nn.Module]],
        lora_embed_tokens_module: Optional[BaseLayerWithLoRA],
        lora_lm_head_module: Optional[BaseLayerWithLoRA],
    ):
        def load_lora_weight_tensor(
            buffer_view: torch.Tensor, weight: Optional[torch.Tensor]
        ):
            if weight is None:


                buffer_view.zero_()
            else:
                assert (
                    buffer_view.shape == weight.shape
                ), f"LoRA buffer shape {buffer_view.shape} does not match weight shape {weight.shape}."
                copy_weight_into_buffer(buffer_view, weight)

        if uid is None:
            self._clear_buffer_slot_for_base(buffer_id)
            return

        assert lora_adapter is not None
        lora_rank = lora_adapter.config.r


        skipped_weight_names: set = set()
        matched_modules: set = set()
        all_weight_names: list = []
        for layer in lora_adapter.layers:
            all_weight_names.extend(layer.weights.keys())
        if lora_adapter.embedding_layers:
            all_weight_names.extend(lora_adapter.embedding_layers.keys())
        for name in all_weight_names:
            try:
                target_module = get_target_module_name(name, self.target_modules)
                matched_modules.add(target_module)
            except ValueError:
                skipped_weight_names.add(name)
        if matched_modules:
            logger.info(
                "LoRA adapter '%s': loaded weights for target modules %s.",
                uid,
                sorted(matched_modules),
            )
        if skipped_weight_names:
            msg = (
                f"LoRA adapter '{uid}': {len(skipped_weight_names)} weight(s) "
                f"skipped because they did not match any target module in "
                f"{sorted(self.target_modules)}. Skipped weights: "
                f"{sorted(skipped_weight_names)}. This likely indicates a "
                f"mismatch between the adapter's target modules and the base "
                f"model architecture."
            )
            if self.strict_loading:
                raise ValueError(msg)
            else:
                logger.warning(msg)

        for layer_id in range(self.num_layer):
            layer = lora_adapter.layers[layer_id]
            layer_weights = layer.weights
            pinned_layer_weights = layer.pinned_weights
            cur_layer_modules = lora_modules[layer_id]
            has_shared_moe_module = any(
                getattr(
                    getattr(module, "base_layer", module), "is_shared_fused_moe", False
                )
                for module in cur_layer_modules.values()
            )


            temp_A_buffer: Dict[str, Union[torch.Tensor, Dict[int, torch.Tensor]]] = {
                target_module: None for target_module in self.A_buffer
            }
            temp_B_buffer: Dict[str, Union[torch.Tensor, Dict[int, torch.Tensor]]] = {
                target_module: None for target_module in self.B_buffer
            }
            temp_A_cache_keys: Dict[str, Optional[Union[str, Dict[int, str]]]] = {
                target_module: None for target_module in self.A_buffer
            }
            temp_B_cache_keys: Dict[str, Optional[Union[str, Dict[int, str]]]] = {
                target_module: None for target_module in self.B_buffer
            }

            for name, weights in layer_weights.items():
                target_module = get_target_module_name(name, self.target_modules)


                expert_match = re.search(r"experts\.(\d+)\.", name)
                is_shared_expert = "shared_experts." in name
                shared_moe_target = f"{target_module}_shared_moe"

                if is_shared_expert and (
                    weights.dim() == 3
                    or (has_shared_moe_module and shared_moe_target in temp_A_buffer)
                ):


                    target_module = shared_moe_target
                    if expert_match:
                        if temp_A_buffer[target_module] is None:
                            temp_A_buffer[target_module] = {}
                            temp_B_buffer[target_module] = {}
                            temp_A_cache_keys[target_module] = {}
                            temp_B_cache_keys[target_module] = {}
                        expert_id = int(expert_match.group(1))
                        if "lora_A" in name:
                            temp_A_buffer[target_module][expert_id] = weights
                            temp_A_cache_keys[target_module][expert_id] = name
                        else:
                            temp_B_buffer[target_module][expert_id] = weights
                            temp_B_cache_keys[target_module][expert_id] = name
                    elif "lora_A" in name:
                        temp_A_buffer[target_module] = weights
                        temp_A_cache_keys[target_module] = name
                    else:
                        temp_B_buffer[target_module] = weights
                        temp_B_cache_keys[target_module] = name
                elif expert_match:

                    target_module = target_module + "_moe"
                    if temp_A_buffer[target_module] is None:
                        temp_A_buffer[target_module] = {}
                        temp_B_buffer[target_module] = {}
                        temp_A_cache_keys[target_module] = {}
                        temp_B_cache_keys[target_module] = {}

                    expert_id = int(expert_match.group(1))
                    if "lora_A" in name:
                        temp_A_buffer[target_module][expert_id] = weights
                        temp_A_cache_keys[target_module][expert_id] = name
                    else:
                        temp_B_buffer[target_module][expert_id] = weights
                        temp_B_cache_keys[target_module][expert_id] = name
                elif "experts" in name and weights.dim() == 3:

                    target_module = target_module + "_moe"
                    if "lora_A" in name:
                        temp_A_buffer[target_module] = weights
                        temp_A_cache_keys[target_module] = name
                    else:
                        temp_B_buffer[target_module] = weights
                        temp_B_cache_keys[target_module] = name
                else:

                    if "lora_A" in name:
                        temp_A_buffer[target_module] = weights
                        temp_A_cache_keys[target_module] = name
                    else:
                        temp_B_buffer[target_module] = weights
                        temp_B_cache_keys[target_module] = name


            active_target_modules: Set[str] = set()
            from sglang.srt.lora.layers import FusedMoEWithLoRA

            for module_name, module in cur_layer_modules.items():
                if isinstance(module, FusedMoEWithLoRA) or getattr(
                    module, "is_shared_fused_moe", False
                ):
                    base_layer = getattr(module, "base_layer", module)
                    key_suffix = (
                        "_shared_moe" if base_layer.is_shared_fused_moe else "_moe"
                    )
                    moe_tp_rank = base_layer.moe_tp_rank
                    for canonical in ("gate_up_proj_moe", "down_proj_moe"):


                        target_module = canonical.replace("_moe", key_suffix)
                        active_target_modules.add(target_module)
                        if temp_A_buffer.get(target_module) is not None:
                            temp_A_buffer[target_module] = (
                                module.slice_moe_lora_a_weights(
                                    temp_A_buffer[target_module],
                                    moe_tp_rank,
                                    canonical,
                                )
                            )
                            cache_keys = temp_A_cache_keys[target_module]
                            assert cache_keys is not None
                            temp_A_cache_keys[target_module] = append_cache_key_suffix(
                                cache_keys,
                                f"moe_tp{moe_tp_rank}",
                            )
                        if temp_B_buffer.get(target_module) is not None:
                            temp_B_buffer[target_module] = (
                                module.slice_moe_lora_b_weights(
                                    temp_B_buffer[target_module],
                                    moe_tp_rank,
                                    canonical,
                                )
                            )
                            cache_keys = temp_B_cache_keys[target_module]
                            assert cache_keys is not None
                            temp_B_cache_keys[target_module] = append_cache_key_suffix(
                                cache_keys,
                                f"moe_tp{moe_tp_rank}",
                            )

                    continue


                target_module = get_target_module_name(module_name, self.target_modules)


                active_target_modules.add(target_module)

                if temp_A_buffer[target_module] is None:

                    continue


                temp_A_buffer[target_module] = module.slice_lora_a_weights(
                    temp_A_buffer[target_module]
                )
                cache_keys = temp_A_cache_keys[target_module]
                assert cache_keys is not None
                temp_A_cache_keys[target_module] = append_cache_key_suffix(
                    cache_keys,
                    f"tp{self.tp_rank}",
                )

                temp_B_buffer[target_module] = module.slice_lora_b_weights(
                    temp_B_buffer[target_module]
                )
                cache_keys = temp_B_cache_keys[target_module]
                assert cache_keys is not None
                temp_B_cache_keys[target_module] = append_cache_key_suffix(
                    cache_keys,
                    f"tp{self.tp_rank}",
                )

            for name, weights in temp_A_buffer.items():
                if name not in active_target_modules:
                    continue
                c = get_stacked_multiply(name, self.base_model)
                max_r = self.max_lora_rank
                target_buffer = self.A_buffer[name][layer_id]
                weights_cache_key = temp_A_cache_keys[name]

                if name in [
                    "gate_up_proj_moe",
                    "down_proj_moe",
                    "gate_up_proj_shared_moe",
                    "down_proj_shared_moe",
                ]:
                    if self.experts_shared_outer_loras and name in (
                        "gate_up_proj_moe",
                        "gate_up_proj_shared_moe",
                    ):
                        if weights is None or (
                            isinstance(weights, dict) and not weights
                        ):
                            representative_weight = None
                        elif isinstance(weights, torch.Tensor) and weights.dim() == 3:
                            if weights.shape[0] != 1:
                                raise ValueError(
                                    f"experts_shared_outer_loras is enabled but "
                                    f"gate_up_proj_moe lora_A has expert_dim="
                                    f"{weights.shape[0]} (expected 1)."
                                )
                            assert isinstance(weights_cache_key, str)
                            weights = self._get_maybe_cached_weight_for_transfer(
                                pinned_layer_weights,
                                weights_cache_key,
                                weights,
                            )
                            representative_weight = weights[0]
                        elif isinstance(weights, dict) and len(weights) > 0:
                            if len(weights) != 1:
                                raise ValueError(
                                    f"experts_shared_outer_loras is enabled but "
                                    f"gate_up_proj_moe lora_A dict has "
                                    f"{len(weights)} entries (expected 1)."
                                )
                            rep = next(iter(weights.values()))
                            assert isinstance(weights_cache_key, dict)
                            rep_cache_key = next(iter(weights_cache_key.values()))
                            rep = self._get_maybe_cached_weight_for_transfer(
                                pinned_layer_weights, rep_cache_key, rep
                            )
                            representative_weight = rep
                        else:
                            raise ValueError(
                                f"Unexpected weight format for shared outer gate_up_proj_moe lora_A: "
                                f"type={type(weights)}, "
                                f"shape={weights.shape if isinstance(weights, torch.Tensor) else 'N/A'}"
                            )
                        if representative_weight is not None:
                            expected_shape = target_buffer[
                                buffer_id, 0, : lora_rank * c, :
                            ].shape
                            assert representative_weight.shape == expected_shape, (
                                f"LoRA buffer shape {expected_shape} does not match "
                                f"weight shape {representative_weight.shape}."
                            )


                        target_buffer[buffer_id, 0].zero_()
                        if representative_weight is not None:
                            for ci in range(c):
                                buffer_view = target_buffer[
                                    buffer_id, 0, ci * max_r : ci * max_r + lora_rank, :
                                ]
                                load_lora_weight_tensor(
                                    buffer_view,
                                    representative_weight[
                                        ci * lora_rank : (ci + 1) * lora_rank, :
                                    ],
                                )
                    elif weights is None:
                        target_buffer[buffer_id].zero_()
                    elif isinstance(weights, (torch.Tensor, dict)):


                        target_buffer[buffer_id].zero_()
                        assert isinstance(weights_cache_key, (str, dict))
                        for (
                            local_eid,
                            expert_weight,
                            expert_cache_key,
                        ) in self._iter_local_expert_weights(
                            weights,
                            weights_cache_key,
                            localize=not self.is_shared_moe_module(name),
                        ):
                            if expert_weight is None:
                                continue
                            expert_weight = self._get_maybe_cached_weight_for_transfer(
                                pinned_layer_weights,
                                expert_cache_key,
                                expert_weight,
                            )
                            for ci in range(c):
                                buffer_view = target_buffer[
                                    buffer_id,
                                    local_eid,
                                    ci * max_r : ci * max_r + lora_rank,
                                    :,
                                ]
                                load_lora_weight_tensor(
                                    buffer_view,
                                    expert_weight[
                                        ci * lora_rank : (ci + 1) * lora_rank, :
                                    ],
                                )
                else:
                    buffer_view = target_buffer[buffer_id, : lora_rank * c, :]
                    if weights is not None:
                        assert isinstance(weights_cache_key, str)
                        weights = self._get_maybe_cached_weight_for_transfer(
                            pinned_layer_weights,
                            weights_cache_key,
                            weights,
                        )
                    load_lora_weight_tensor(buffer_view, weights)

            for name, weights in temp_B_buffer.items():
                if name not in active_target_modules:
                    continue
                target_buffer = self.B_buffer[name][layer_id]
                weights_cache_key = temp_B_cache_keys[name]

                if name in [
                    "gate_up_proj_moe",
                    "down_proj_moe",
                    "gate_up_proj_shared_moe",
                    "down_proj_shared_moe",
                ]:
                    if self.experts_shared_outer_loras and name in (
                        "down_proj_moe",
                        "down_proj_shared_moe",
                    ):
                        if weights is None or (
                            isinstance(weights, dict) and not weights
                        ):
                            buffer_view = target_buffer[buffer_id, 0, :, :lora_rank]
                            load_lora_weight_tensor(buffer_view, None)
                        elif isinstance(weights, torch.Tensor) and weights.dim() == 3:
                            if weights.shape[0] != 1:
                                raise ValueError(
                                    f"experts_shared_outer_loras is enabled but "
                                    f"down_proj_moe lora_B has expert_dim="
                                    f"{weights.shape[0]} (expected 1)."
                                )
                            buffer_view = target_buffer[buffer_id, 0, :, :lora_rank]
                            w = weights[0]
                            assert isinstance(weights_cache_key, str)
                            if w is not None:
                                w = w * lora_adapter.scaling
                                w = self._get_maybe_cached_weight_for_transfer(
                                    pinned_layer_weights,
                                    append_cache_key_suffix(
                                        weights_cache_key, "expert0"
                                    ),
                                    w,
                                )
                            load_lora_weight_tensor(buffer_view, w)
                        elif isinstance(weights, dict) and len(weights) > 0:
                            if len(weights) != 1:
                                raise ValueError(
                                    f"experts_shared_outer_loras is enabled but "
                                    f"down_proj_moe lora_B dict has "
                                    f"{len(weights)} entries (expected 1)."
                                )
                            rep = next(iter(weights.values()))
                            assert isinstance(weights_cache_key, dict)
                            rep_cache_key = next(iter(weights_cache_key.values()))
                            buffer_view = target_buffer[buffer_id, 0, :, :lora_rank]
                            if rep is not None:
                                rep = rep * lora_adapter.scaling
                                rep = self._get_maybe_cached_weight_for_transfer(
                                    pinned_layer_weights,
                                    rep_cache_key,
                                    rep,
                                )
                            load_lora_weight_tensor(buffer_view, rep)
                        else:
                            raise ValueError(
                                f"Unexpected weight format for shared outer down_proj_moe lora_B: "
                                f"type={type(weights)}, "
                                f"shape={weights.shape if isinstance(weights, torch.Tensor) else 'N/A'}"
                            )

                        target_buffer[buffer_id, 0, :, lora_rank:].zero_()
                    elif weights is None:
                        target_buffer[buffer_id].zero_()
                    elif isinstance(weights, (torch.Tensor, dict)):


                        target_buffer[buffer_id].zero_()
                        assert isinstance(weights_cache_key, (str, dict))
                        for (
                            local_eid,
                            w,
                            w_cache_key,
                        ) in self._iter_local_expert_weights(
                            weights,
                            weights_cache_key,
                            localize=not self.is_shared_moe_module(name),
                        ):
                            if w is not None:
                                w = w * lora_adapter.scaling
                                w = self._get_maybe_cached_weight_for_transfer(
                                    pinned_layer_weights,
                                    w_cache_key,
                                    w,
                                )
                            buffer_view = target_buffer[
                                buffer_id, local_eid, :, :lora_rank
                            ]
                            load_lora_weight_tensor(buffer_view, w)
                else:
                    buffer_view = target_buffer[buffer_id, :, :lora_rank]
                    if weights is not None:
                        assert isinstance(weights_cache_key, str)
                        weights = self._get_maybe_cached_weight_for_transfer(
                            pinned_layer_weights,
                            weights_cache_key,
                            weights,
                        )
                    load_lora_weight_tensor(buffer_view, weights)
                    if _SGLANG_EXPERIMENTAL_LORA_OPTI:


                        target_buffer[buffer_id, :, lora_rank:].zero_()

        if lora_adapter.embedding_layers:
            org_vocab_size = self.base_hf_config.vocab_size
            lora_added_tokens_size = lora_adapter.config.lora_added_tokens_size
            pinned_embedding_layers = lora_adapter.pinned_embedding_layers
            pinned_added_tokens_embeddings = lora_adapter.pinned_added_tokens_embeddings


            if lora_adapter.added_tokens_embeddings:
                for name, weights in lora_adapter.added_tokens_embeddings.items():
                    if "input_embeddings" in name:
                        buffer_view = self.new_embeddings_buffer["input_embeddings"][
                            buffer_id, :lora_added_tokens_size
                        ]
                        weights = self._get_maybe_cached_weight_for_transfer(
                            pinned_added_tokens_embeddings,
                            name,
                            weights,
                        )
                        load_lora_weight_tensor(buffer_view, weights)


            for name, weights in lora_adapter.embedding_layers.items():
                target_module = get_target_module_name(name, self.target_modules)
                if (
                    target_module == "embed_tokens"
                    and "embed_tokens" in name
                    and ("lora_embedding_A" in name or "lora_A" in name)
                ):
                    buffer_view = self.embedding_A_buffer[target_module][
                        buffer_id,
                        :lora_rank,
                        : (org_vocab_size + lora_added_tokens_size),
                    ]
                    weights = self._get_maybe_cached_weight_for_transfer(
                        pinned_embedding_layers,
                        name,
                        weights,
                    )
                    load_lora_weight_tensor(buffer_view, weights)
                elif (
                    target_module == "embed_tokens"
                    and "embed_tokens" in name
                    and ("lora_embedding_B" in name or "lora_B" in name)
                ):
                    lora_b_weights = weights


                    buffer_view = self.embedding_B_buffer[target_module][
                        buffer_id, :, :lora_rank
                    ]
                    lora_b_weights = self._get_maybe_cached_weight_for_transfer(
                        pinned_embedding_layers,
                        name,
                        lora_b_weights,
                    )
                    load_lora_weight_tensor(buffer_view, lora_b_weights)

                elif (
                    target_module == "lm_head"
                    and lora_lm_head_module is not None
                    and "lm_head" in name
                    and ("lora_embedding_A" in name or "lora_A" in name)
                ):
                    buffer_view = self.lm_head_A_buffer[target_module][

                        buffer_id,
                        :lora_rank,
                        :,
                    ]
                    weights = self._get_maybe_cached_weight_for_transfer(
                        pinned_embedding_layers,
                        name,
                        weights,
                    )
                    load_lora_weight_tensor(buffer_view, weights)
                elif (
                    target_module == "lm_head"
                    and lora_lm_head_module is not None
                    and "lm_head" in name
                    and ("lora_embedding_B" in name or "lora_B" in name)
                ):
                    assert lora_lm_head_module is not None
                    lora_b_weights = weights

                    if self.tp_size > 1:
                        lora_b_weights = lora_lm_head_module.slice_lora_b_weights(
                            lora_b_weights
                        )
                        cache_key = append_cache_key_suffix(name, f"tp{self.tp_rank}")
                    else:
                        cache_key = name

                    buffer_view = self.lm_head_B_buffer[target_module][
                        buffer_id,
                        : lora_b_weights.shape[0],
                        :lora_rank,
                    ]
                    lora_b_weights = self._get_maybe_cached_weight_for_transfer(
                        pinned_embedding_layers,
                        cache_key,
                        lora_b_weights,
                    )
                    load_lora_weight_tensor(buffer_view, lora_b_weights)
                elif (
                    target_module == "lm_head"
                    and "lm_head" in name
                    and (
                        "lora_embedding_A" in name
                        or "lora_A" in name
                        or "lora_embedding_B" in name
                        or "lora_B" in name
                    )
                ):


                    assert (
                        not get_pp_group().is_last_rank
                    ), f"Failed to load lm_head LoRA weight: {name}, this is only expected to happen on non-last PP stages."
                    continue
        else:


            for k in self.embedding_A_buffer.keys():
                self.embedding_A_buffer[k][buffer_id].zero_()
            for k in self.embedding_B_buffer.keys():
                self.embedding_B_buffer[k][buffer_id].zero_()
            for k in self.lm_head_A_buffer.keys():
                self.lm_head_A_buffer[k][buffer_id].zero_()
            for k in self.lm_head_B_buffer.keys():
                self.lm_head_B_buffer[k][buffer_id].zero_()
            if (
                self.lora_added_tokens_size > 0
                and "input_embeddings" in self.new_embeddings_buffer
            ):
                self.new_embeddings_buffer["input_embeddings"][buffer_id].zero_()

    def get_embedding_tensor(
        self, target_module: str, lora_type: LoRAType
    ) -> Optional[torch.Tensor]:


        if target_module == "added_tokens":
            if (
                self.lora_added_tokens_size is not None
                and self.lora_added_tokens_size > 0
            ):
                return self.new_embeddings_buffer["input_embeddings"]
            return None
        elif target_module == "embed_tokens":
            if lora_type == LoRAType.LORA_A:
                return self.embedding_A_buffer[target_module]
            return self.embedding_B_buffer[target_module]
        elif target_module == "lm_head":
            if lora_type == LoRAType.LORA_A:
                return self.lm_head_A_buffer[target_module]
            return self.lm_head_B_buffer[target_module]

        raise ValueError(
            f"Invalid target_module '{target_module}'. "
            f"Expected 'embed_tokens' or 'lm_head'."
        )

    def get_tensor(
        self, target_module: str, layer_id: int, lora_type: LoRAType
    ) -> torch.Tensor:

        buffer_dict = self.A_buffer if lora_type == LoRAType.LORA_A else self.B_buffer

        return buffer_dict[target_module][layer_id]

    def get_buffer_id(self, lora_uid: str):
        return self.uid_to_buffer_id[lora_uid]
