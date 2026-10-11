import logging
from typing import Tuple

import torch

from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.hardware_backend.mlx.kv_cache.auxiliary_state import (
    MlxAuxiliaryStateReqToTokenPool,
)
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.model_runner_components.layer_setup import (
    ModelLayerInfo,
)
from sglang.srt.runtime_context import get_exec, get_memory, get_schedule

logger = logging.getLogger(__name__)


MLX_AUX_STATE_SIZE_MAX_RUNNING_REQUESTS_RATIO = 4


class _DummyKVCache(KVCache):


    def __init__(self, size: int, dtype: torch.dtype, device: str):


        self.size = size
        self.page_size = 1
        self.dtype = dtype
        self.store_dtype = dtype
        self.device = device
        self.layer_num = 0
        self.start_layer = 0
        self.end_layer = 0
        self.mem_usage = 0
        self.cpu_offloading_chunk_size = 8192
        self.layer_transfer_counter = None
        self.enable_custom_mem_pool = False
        self.custom_mem_pool = None

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise RuntimeError("_DummyKVCache has no key buffer (MLX manages cache)")

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise RuntimeError("_DummyKVCache has no value buffer (MLX manages cache)")

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("_DummyKVCache has no kv buffer (MLX manages cache)")

    def set_kv_buffer(self, layer, loc, cache_k, cache_v) -> None:
        raise RuntimeError("_DummyKVCache cannot set kv buffer (MLX manages cache)")

    def get_kv_size_bytes(self):
        return 0, 0


class _DummyModel:


    @staticmethod
    def forward():
        pass


class MlxModelRunnerStub(ModelRunner):


    canary_manager = None


    prefill_aware_swa = False

    @staticmethod
    def validate_startup_weight_load_mode(server_args) -> None:
        if server_args.is_startup_weight_load_overlap:
            raise ValueError(
                "--startup-weight-load-mode=overlap is not supported: CUDA only"
            )

    def __init__(self, *args, mlx_pool_size: int | None = None, **kwargs):
        self.validate_startup_weight_load_mode(kwargs["server_args"])
        self._mlx_pool_size = mlx_pool_size
        super().__init__(*args, **kwargs)

    def load_model(self):

        logger.info(
            "MLX stub: skipping PyTorch model weight loading "
            "(inference runs through MLX)"
        )

        self.model = _DummyModel()

        self.sliding_window_size = None
        if (
            self.model_config.is_hybrid_swa
            and self.model_config.sliding_window_size is not None
        ):
            self.sliding_window_size = self.model_config.sliding_window_size
        elif self.model_config.attention_chunk_size is not None:
            self.sliding_window_size = self.model_config.attention_chunk_size

        self.dtype = self.model_config.dtype
        self.weight_load_mem_usage = 0

    def _aux_state_slots_per_request(self) -> int:

        if get_memory().disable_radix_cache:
            return 1
        return MLX_AUX_STATE_SIZE_MAX_RUNNING_REQUESTS_RATIO

    def _explicit_aux_state_size_per_worker(self) -> int | None:

        aux_state_size = get_schedule().max_mamba_cache_size
        if aux_state_size is None:
            return None
        return aux_state_size // self.ps.attn_dp_size

    def _resolve_max_running_requests(self) -> int:

        capacity_cap = self.max_total_num_tokens // 2
        requested = get_schedule().max_running_requests
        if requested is None:
            requested_per_worker = None
            resolved = min(capacity_cap, 4096)
        else:
            requested_per_worker = requested // self.ps.attn_dp_size
            resolved = min(requested_per_worker, capacity_cap)

        aux_state_size = self._explicit_aux_state_size_per_worker()
        if (
            mambaish_config(self.model_config) is not None
            and aux_state_size is not None
        ):
            ratio = self._aux_state_slots_per_request()
            resolved = min(resolved, aux_state_size // ratio)
            if resolved <= 0:
                global_aux_state_size = get_schedule().max_mamba_cache_size
                min_global_aux_state_size = ratio * self.ps.attn_dp_size
                raise RuntimeError(
                    f"MLX auxiliary-state cache is too small to serve any "
                    f"requests: max_mamba_cache_size={global_aux_state_size} "
                    f"backs only {aux_state_size // ratio} concurrent requests "
                    f"per attention-DP worker (per-worker auxiliary-state "
                    f"cap={aux_state_size}, {ratio} slots per request). "
                    f"Increase --max-mamba-cache-size to at least "
                    f"{min_global_aux_state_size}, or leave it unset to size "
                    f"the pool from the concurrency cap."
                )

        if requested_per_worker is not None and resolved < requested_per_worker:
            logger.warning(
                "max_running_requests was reduced from the requested %d to %d "
                "(per attention-DP worker) due to the available KV cache or "
                "auxiliary-state capacity.",
                requested_per_worker,
                resolved,
            )
        return resolved

    def initialize(self):

        from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=get_exec().features.enable_memory_saver
        )


        self.sampler = None
        self.load_model()


        model_num_layers = max(
            self.model_config.num_hidden_layers,
            self.model_config.num_attention_layers,
        )
        self.layer_info = ModelLayerInfo(
            start_layer=0,
            end_layer=model_num_layers,
            num_effective_layers=model_num_layers,
        )


        self.kv_cache_dtype = self.dtype


        if self._mlx_pool_size is not None:
            self.max_total_num_tokens = self._mlx_pool_size
        else:
            self.max_total_num_tokens = self.model_config.context_len
        self.max_running_requests = self._resolve_max_running_requests()
        self.is_hybrid_swa = False


        if mambaish_config(self.model_config) is not None:
            auxiliary_state_size = self._explicit_aux_state_size_per_worker()
            if auxiliary_state_size is None:
                auxiliary_state_size = (
                    self.max_running_requests * self._aux_state_slots_per_request()
                )
            self.req_to_token_pool = MlxAuxiliaryStateReqToTokenPool(
                size=self.max_running_requests,
                max_context_len=self.model_config.context_len,
                device="cpu",
                enable_memory_saver=False,
                auxiliary_state_size=auxiliary_state_size,


                owns_auxiliary_state_release=get_memory().disable_radix_cache,
            )
        else:
            self.req_to_token_pool = ReqToTokenPool(
                size=self.max_running_requests,
                max_context_len=self.model_config.context_len,
                device="cpu",
                enable_memory_saver=False,
            )

        dummy_kv = _DummyKVCache(
            size=self.max_total_num_tokens,
            dtype=self.kv_cache_dtype,
            device="cpu",
        )
        self.token_to_kv_pool = dummy_kv
        self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
            size=self.max_total_num_tokens,
            dtype=self.kv_cache_dtype,
            device="cpu",
            kvcache=dummy_kv,
            need_sort=False,
        )


        self.decode_cuda_graph_runner = None
        self.attn_backend = None

        self.init_ngram_embedding_manager()

        logger.info(
            f"MLX stub: initialized minimal pools "
            f"(max_total_num_tokens={self.max_total_num_tokens}, "
            f"max_running_requests={self.max_running_requests}, "
            f"zero GPU KV cache allocation)"
        )

    def alloc_memory_pool(self, memory_pool_config=None):

        pass

    def init_attention_backends(self):

        self.attn_backend = None
