from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import msgspec
import torch

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.configs.hybrid_arch import (
    hybrid_gdn_config,
    kimi_linear_config,
    mambaish_config,
)
from sglang.srt.configs.model_config import (
    ModelConfig,
    dsa_layer_skips_topk,
    get_dsa_index_head_dim,
    get_minimax_sparse_attention_config,
    get_minimax_sparse_disable_value_layer_ids,
    get_minimax_sparse_layer_ids,
    is_deepseek_dsa,
    is_deepseek_v4,
    is_minimax_sparse,
)
from sglang.srt.distributed.parallel_state import get_world_group
from sglang.srt.distributed.utils import get_pp_indices
from sglang.srt.environ import envs
from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
    get_kv_cache_quant_method,
    resolve_kv_cache_quant,
)
from sglang.srt.mem_cache.allocation_sizing import get_req_to_token_extra_context_len
from sglang.srt.mem_cache.allocator import (
    BaseTokenToKVPoolAllocator,
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.swa import (
    PureSWATokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    HybridLinearKVPool,
    HybridReqToTokenPool,
    KVCache,
    MHATokenToKVPool,
    MHATokenToKVPoolFP4,
    MHATokenToKVPoolMXFP8,
    MiniMaxSparseKVPool,
    MLATokenToKVPool,
    MLATokenToKVPoolFP4,
    NoOpMHATokenToKVPool,
    PageMajorMHATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import (
    get_context,
    get_disagg,
    get_exec,
    get_memory,
    get_mm,
    get_parallel,
    get_schedule,
    get_spec,
    mamba_extra_buffer_enabled,
    mamba_extra_buffer_lazy_enabled,
    max_speculative_num_draft_tokens,
    pre_capture_activation_reserve_mb,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils.common import (
    get_available_gpu_memory,
    get_cuda_graph_max_batch_size,
    get_device_memory_capacity,
    is_float4_e2m1fn_x2,
    is_hip,
    is_npu,
)

logger = logging.getLogger(__name__)


def _should_elide_dsa_index_k(*, is_draft_worker: bool) -> bool:
    memory_config = get_memory()
    return (
        not memory_config.enable_hisparse
        and not is_draft_worker
        and not memory_config.enable_hierarchical_cache
        and get_disagg().disaggregation_mode == "null"
    )


_is_hip = is_hip()


def _get_dsv4_compress_state_dtypes() -> tuple[torch.dtype, torch.dtype]:
    dtype_name = envs.SGLANG_DSV4_COMPRESS_STATE_DTYPE.get().strip().lower()
    if dtype_name in ("float32", "fp32"):
        return torch.float32, torch.float32
    if dtype_name in ("bfloat16", "bf16"):
        return torch.bfloat16, torch.bfloat16
    raise ValueError(
        "Unsupported SGLANG_DSV4_COMPRESS_STATE_DTYPE="
        f"{dtype_name!r}. Expected one of: float32, fp32, bfloat16, bf16."
    )


_is_npu = is_npu()


def _should_enable_lazy_compaction() -> bool:

    return not envs.SGLANG_DISABLE_LAZY_COMPACTION.get()


def mm_runtime_reservation_gb(
    *, is_multimodal: bool, mm_feature_transport: Optional[str]
) -> float:

    if not is_multimodal:
        return 0.0
    reserved_mb = envs.SGLANG_VLM_CACHE_SIZE_MB.get()
    if mm_feature_transport in ("cuda_ipc", "cuda_vmm"):
        reserved_mb += envs.SGLANG_MM_FEATURE_CACHE_MB.get()
    if reserved_mb > 0:
        logger.info(
            "Reserving %.2f GB of the KV budget for post-sizing multimodal "
            "allocations (feature-transport pools + embedding cache).",
            reserved_mb / 1024,
        )
    return reserved_mb / 1024


MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO = 3
MAMBA_CACHE_BASE_RATIO_DROP_ON_SKIP = 1
MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP = 2
MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP_LAZY = 1
MAMBA_CACHE_V2_ADDITIONAL_RATIO_NO_OVERLAP = 1
MAMBA_CACHE_V2_ADDITIONAL_RATIO_NO_BUFFER = 1

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.mem_cache.unified_memory_pool import (
        UnifiedKVPool,
        UnifiedPoolBundle,
    )
    from sglang.srt.model_executor.model_runner_components.layer_setup import (
        ModelLayerInfo,
    )
    from sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state import (
        SpecAuxHiddenStateConfig,
    )
    from sglang.srt.model_executor.pool_configurator import (
        MemoryPoolConfig,
    )


class KVCacheConfigResult(msgspec.Struct, frozen=True, kw_only=True):
    max_total_num_tokens: int
    max_running_requests: int
    full_max_total_num_tokens: Optional[int]
    swa_max_total_num_tokens: Optional[int]
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool: KVCache
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    memory_pool_config: MemoryPoolConfig
    unified_memory_pool: Optional[UnifiedKVPool] = None


class _InitializedPools(msgspec.Struct, frozen=True, kw_only=True):
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool: KVCache
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    unified_memory_pool: Optional[UnifiedKVPool] = None


class _PoolSizes(msgspec.Struct, frozen=True, kw_only=True):
    max_total_num_tokens: int
    max_running_requests: int
    full_max_total_num_tokens: Optional[int]
    swa_max_total_num_tokens: Optional[int]
    c4_max_total_num_tokens: int
    c128_max_total_num_tokens: int
    c4_state_pool_size: int
    c128_state_pool_size: int
    c4_state_dtype: Optional[torch.dtype]
    c128_state_dtype: Optional[torch.dtype]


@dataclass(slots=True, kw_only=True)
class KVCacheConfigurator:
    device: str
    gpu_id: int
    ps: ParallelState
    pp_group: Any
    model: Any
    model_config: ModelConfig
    server_args: ServerArgs
    kv_cache_dtype: torch.dtype
    model_dtype: torch.dtype
    page_size: int
    sliding_window_size: Optional[int]
    spec_algorithm: SpeculativeAlgorithm
    is_draft_worker: bool
    post_capture_kv_active: bool
    spec_aux_config: SpecAuxHiddenStateConfig
    is_hybrid_swa: bool
    is_hybrid_swa_compress: bool
    use_mla_backend: bool
    layer_info: ModelLayerInfo
    forward_stream: Any
    req_to_token_pool: Optional[ReqToTokenPool]
    token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator]
    memory_pool_config: Optional[MemoryPoolConfig]
    draft_model_idx: Optional[int] = None
    kv_cache_dtype_str: Optional[str] = None
    mambaish_config: Optional[Any] = field(init=False)
    hybrid_gdn_config: Optional[Any] = field(init=False)
    is_inkling_mtp_draft: bool = field(init=False)
    draft_swa_full_capacity: bool = field(init=False)

    def __post_init__(self) -> None:
        self.mambaish_config = mambaish_config(self.model_config)
        self.hybrid_gdn_config = hybrid_gdn_config(self.model_config)


        self.is_inkling_mtp_draft = (
            self.is_draft_worker
            and self.draft_model_idx is not None
            and self.model_config.hf_config.architectures[0]
            == "InklingForConditionalGenerationMTP"
        )
        self.draft_swa_full_capacity = self.is_inkling_mtp_draft and (
            self.draft_model_idx
            in set(self.model_config.hf_text_config.mtp_local_layer_ids)
        )

    def _build_fp4_quant_method(self, *, num_layers: int):
        if not is_float4_e2m1fn_x2(self.kv_cache_dtype):
            return None
        quant_name = resolve_kv_cache_quant(self.kv_cache_dtype_str)
        if quant_name is None:
            return None
        quant_method = get_kv_cache_quant_method(
            quant_name,
            num_layers=num_layers,
            device=self.device,
        )
        quant_method.load_scales_from_model(self.model)
        return quant_method

    def configure(self, *, pre_model_load_memory: int) -> KVCacheConfigResult:

        if not self.spec_algorithm.is_none() and self.is_draft_worker:
            assert (
                self.memory_pool_config is not None
            ), "Draft worker requires memory_pool_config"
            config = self.memory_pool_config
        else:
            config = self._resolve_memory_pool_config(pre_model_load_memory)

        sizes = self._derive_pool_sizes(config=config)

        pools = self._init_pools(
            sizes=sizes,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
        )

        logger.info(
            f"Memory pool end. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        return KVCacheConfigResult(
            max_total_num_tokens=sizes.max_total_num_tokens,
            max_running_requests=sizes.max_running_requests,
            full_max_total_num_tokens=sizes.full_max_total_num_tokens,
            swa_max_total_num_tokens=sizes.swa_max_total_num_tokens,
            req_to_token_pool=pools.req_to_token_pool,
            token_to_kv_pool=pools.token_to_kv_pool,
            token_to_kv_pool_allocator=pools.token_to_kv_pool_allocator,
            memory_pool_config=config,
            unified_memory_pool=pools.unified_memory_pool,
        )


    @property
    def loc_space_scale(self) -> int:
        dcp_size = get_parallel().attn_dcp_size
        return dcp_size if (self.is_draft_worker and dcp_size > 1) else 1

    @property
    def pool_page_size(self) -> int:
        return get_schedule().page_size * self.loc_space_scale

    def _derive_pool_sizes(self, *, config: MemoryPoolConfig) -> _PoolSizes:
        max_total_num_tokens = config.max_total_num_tokens
        max_running_requests = config.max_running_requests
        full_max_total_num_tokens = None
        swa_max_total_num_tokens = None
        if self.is_hybrid_swa:
            full_max_total_num_tokens = config.full_max_total_num_tokens
            swa_max_total_num_tokens = config.swa_max_total_num_tokens


        loc_scale = self.loc_space_scale
        max_total_num_tokens *= loc_scale
        if full_max_total_num_tokens is not None:
            full_max_total_num_tokens *= loc_scale
        if swa_max_total_num_tokens is not None:
            swa_max_total_num_tokens *= loc_scale


        if self.is_draft_worker:
            c4_max_total_num_tokens = 0
            c128_max_total_num_tokens = 0
            c4_state_pool_size = 0
            c128_state_pool_size = 0
        else:
            c4_max_total_num_tokens = config.c4_max_total_num_tokens
            c128_max_total_num_tokens = config.c128_max_total_num_tokens
            c4_state_pool_size = config.c4_state_pool_size
            c128_state_pool_size = config.c128_state_pool_size


        c4_state_dtype: Optional[torch.dtype] = None
        c128_state_dtype: Optional[torch.dtype] = None
        if is_deepseek_v4(self.model_config.hf_config):
            c4_state_dtype, c128_state_dtype = _get_dsv4_compress_state_dtypes()

        return _PoolSizes(
            max_total_num_tokens=max_total_num_tokens,
            max_running_requests=max_running_requests,
            full_max_total_num_tokens=full_max_total_num_tokens,
            swa_max_total_num_tokens=swa_max_total_num_tokens,
            c4_max_total_num_tokens=c4_max_total_num_tokens,
            c128_max_total_num_tokens=c128_max_total_num_tokens,
            c4_state_pool_size=c4_state_pool_size,
            c128_state_pool_size=c128_state_pool_size,
            c4_state_dtype=c4_state_dtype,
            c128_state_dtype=c128_state_dtype,
        )

    def _init_pools(
        self,
        *,
        sizes: _PoolSizes,
        req_to_token_pool: Optional[ReqToTokenPool],
        token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator],
    ) -> _InitializedPools:

        token_to_kv_pool = None


        if get_memory().enable_unified_memory and req_to_token_pool is None:
            pd_enabled = get_disagg().disaggregation_mode != "null"
            if self.mambaish_config is not None:
                if pd_enabled and not self.use_mla_backend:
                    raise ValueError(
                        "--enable-unified-memory with PD disaggregation "
                        "currently supports only MLA hybrid-Mamba models "
                        "(e.g. kimi-linear); this model uses the MHA full-"
                        "attention pool. Drop --enable-unified-memory or run "
                        "without PD disaggregation."
                    )
                bundle = self._init_unified_mamba_pools(
                    max_num_reqs=sizes.max_running_requests,
                    max_total_num_tokens=sizes.max_total_num_tokens,
                )
            elif self.is_hybrid_swa and not is_deepseek_v4(self.model_config.hf_config):
                if pd_enabled:
                    raise ValueError(
                        "--enable-unified-memory with PD disaggregation does "
                        "not support hybrid-SWA models yet (no whole-envelope "
                        "transfer scheme for the SWA sub-pool). Drop "
                        "--enable-unified-memory or run without PD."
                    )
                bundle = self._init_unified_swa_pools(
                    max_num_reqs=sizes.max_running_requests,
                    full_max_total_num_tokens=sizes.full_max_total_num_tokens,
                    swa_max_total_num_tokens=sizes.swa_max_total_num_tokens,
                )
            else:


                raise ValueError(
                    "--enable-unified-memory only supports hybrid Mamba and "
                    "hybrid sliding-window-attention models (DeepSeek-V4 excluded); "
                    f"the current model ({self.model_config.hf_config.architectures}) "
                    "is neither, so the unified memory pool cannot be built. Drop "
                    "--enable-unified-memory for this model."
                )
            return _InitializedPools(
                req_to_token_pool=bundle.req_to_token_pool,
                token_to_kv_pool=bundle.token_to_kv_pool,
                token_to_kv_pool_allocator=bundle.token_to_kv_pool_allocator,
                unified_memory_pool=bundle.unified_memory_pool,
            )


        draft_virtual_id_space: Optional[int] = None
        if self.is_draft_worker and token_to_kv_pool_allocator is not None:
            from sglang.srt.mem_cache.multi_ended_allocator import (
                UnifiedMambaTokenToKVPoolAllocator,
                UnifiedSWATokenToKVPoolAllocator,
            )

            if isinstance(token_to_kv_pool_allocator, UnifiedSWATokenToKVPoolAllocator):
                raise ValueError(
                    "Speculative decoding with --enable-unified-memory is only "
                    "supported for hybrid-Mamba targets; the unified hybrid-SWA "
                    "pool's draft sizing (virtual-id space) is not wired yet."
                )
            if isinstance(
                token_to_kv_pool_allocator, UnifiedMambaTokenToKVPoolAllocator
            ):
                draft_virtual_id_space = token_to_kv_pool_allocator.size_full
                assert draft_virtual_id_space >= sizes.max_total_num_tokens, (
                    "unified allocator virtual space smaller than the token "
                    f"budget: size_full={draft_virtual_id_space} < "
                    f"max_total_num_tokens={sizes.max_total_num_tokens}"
                )


                page = max(int(self.pool_page_size or 1), 1)
                draft_virtual_id_space = (
                    (draft_virtual_id_space + page - 1) // page * page
                )
                sizes = msgspec.structs.replace(
                    sizes, max_total_num_tokens=draft_virtual_id_space
                )


        if req_to_token_pool is None:
            req_to_token_pool = self._build_req_to_token_pool(
                max_num_reqs=sizes.max_running_requests
            )
        else:

            assert self.is_draft_worker


            if self.is_inkling_mtp_draft and isinstance(
                req_to_token_pool, HybridReqToTokenPool
            ):


                req_to_token_pool = req_to_token_pool.clone_with_new_mamba(
                    mamba_size=get_schedule().max_mamba_cache_size,
                    mamba_spec_state_size=sizes.max_running_requests,
                    cache_params=self.mambaish_config.mamba2_cache_params,
                    device=self.device,
                    enable_mamba_extra_buffer=mamba_extra_buffer_enabled(),
                    draft_model_idx=self.draft_model_idx,
                    speculative_eagle_topk=get_spec().speculative_eagle_topk,
                )


        is_dsa_model = is_deepseek_dsa(self.model_config.hf_config)
        is_dsv4_model = is_deepseek_v4(self.model_config.hf_config)

        self._validate_prefill_only_disable_kv_cache_pool_family(
            is_dsa_model, is_dsv4_model, current_platform
        )

        token_to_kv_pool = self._build_token_to_kv_pool(
            sizes=sizes,
            is_dsa_model=is_dsa_model,
            is_dsv4_model=is_dsv4_model,
            req_to_token_pool=req_to_token_pool,
        )

        if draft_virtual_id_space is not None:
            assert token_to_kv_pool.size >= draft_virtual_id_space, (
                "draft token_to_kv_pool smaller than the shared unified "
                f"allocator's virtual-id space: pool size="
                f"{token_to_kv_pool.size} < size_full={draft_virtual_id_space}; "
                "verify-window writes at high virtual ids would go out of "
                "bounds."
            )

        token_to_kv_pool_allocator = self._build_token_to_kv_pool_allocator(
            sizes=sizes,
            token_to_kv_pool=token_to_kv_pool,
            is_dsv4_model=is_dsv4_model,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )


        if (
            get_schedule().prefill_only_disable_kv_cache
            and not self.is_draft_worker
            and not isinstance(token_to_kv_pool, NoOpMHATokenToKVPool)
        ):
            raise RuntimeError(
                "--prefill-only-disable-kv-cache expected NoOpMHATokenToKVPool but the "
                f"runtime pool is {type(token_to_kv_pool).__name__}. This pool "
                "family is not yet supported by --prefill-only-disable-kv-cache. "
                "Supported configurations today: plain MHA models on CUDA with the FA "
                "(fa3/fa4) prefill backend, --is-embedding, --chunked-prefill-size=-1, "
                "--disable-radix-cache, no context-parallel attention, no HiSparse, "
                "and --kv-cache-dtype not in {nvfp4, fp4_mx_block16}."
            )
        return _InitializedPools(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool=token_to_kv_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )

    def _init_unified_mamba_pools(
        self, *, max_num_reqs: int, max_total_num_tokens: int
    ) -> UnifiedPoolBundle:

        from sglang.srt.mem_cache.unified_memory_pool import init_unified_mamba_pools

        config = self.mambaish_config
        assert config is not None


        assert self.page_size >= 1, f"page_size must be >= 1, got {self.page_size}"

        extra_max_context_len = 4
        if get_spec().speculative_num_draft_tokens is not None:
            extra_max_context_len += get_spec().speculative_num_draft_tokens

        mamba_layer_ids = [
            i
            for i in config.mamba2_cache_params.layers
            if self.layer_info.start_layer <= i < self.layer_info.end_layer
        ]
        full_attention_layer_ids = [
            i
            for i in config.full_attention_layer_ids
            if self.layer_info.start_layer <= i < self.layer_info.end_layer
        ]

        bundle = init_unified_mamba_pools(
            device=self.device,
            kv_cache_dtype=self.kv_cache_dtype,
            head_num=self.model_config.get_num_kv_heads(
                get_parallel().attn_tp_size, get_parallel().attn_dcp_size
            ),
            head_dim=self.model_config.head_dim,
            page_size=self.page_size,
            start_layer=self.layer_info.start_layer,
            end_layer=self.layer_info.end_layer,
            is_draft_worker=self.is_draft_worker,
            use_mla_backend=self.use_mla_backend,
            kv_lora_rank=(
                self.model_config.kv_lora_rank if self.use_mla_backend else None
            ),
            qk_rope_head_dim=(
                self.model_config.qk_rope_head_dim if self.use_mla_backend else None
            ),
            mamba_layer_ids=mamba_layer_ids,
            full_attention_layer_ids=full_attention_layer_ids,
            mamba2_cache_params=config.mamba2_cache_params,
            model_context_len=self.model_config.context_len,
            extra_max_context_len=extra_max_context_len,
            max_total_num_tokens=max_total_num_tokens,
            max_mamba_cache_size=get_schedule().max_mamba_cache_size,
            max_num_reqs=max_num_reqs,
            enable_memory_saver=get_exec().features.enable_memory_saver,
            enable_mamba_extra_buffer=mamba_extra_buffer_enabled(),
            speculative_num_draft_tokens=get_spec().speculative_num_draft_tokens,
            disable_overlap_schedule=get_schedule().disable_overlap_schedule,
            need_sort=get_disagg().disaggregation_mode in ("decode", "prefill"),
            decode_pre_alloc_size=(
                get_disagg().disaggregation_decode_extra_slots
                if get_disagg().disaggregation_mode == "decode"
                else 0
            ),
            mamba_full_memory_ratio=get_schedule().mamba_full_memory_ratio,


            forward_stream=self.forward_stream,

            lazy_compaction=_should_enable_lazy_compaction(),
        )
        return bundle

    def _init_unified_swa_pools(
        self,
        *,
        max_num_reqs: int,
        full_max_total_num_tokens: Optional[int],
        swa_max_total_num_tokens: Optional[int],
    ) -> UnifiedPoolBundle:

        from sglang.srt.mem_cache.unified_memory_pool import (
            UnifiedPoolBundle,
            init_unified_swa_pools,
        )

        assert self.is_hybrid_swa, "_init_unified_swa_pools called on a non-SWA model"


        assert self.page_size >= 1, f"page_size must be >= 1, got {self.page_size}"
        assert (
            not self.use_mla_backend
        ), "unified memory pool does not support MLA-SWA hybrid yet"

        extra_max_context_len = 4
        if get_spec().speculative_num_draft_tokens is not None:
            extra_max_context_len += get_spec().speculative_num_draft_tokens
        req_to_token_pool = ReqToTokenPool(
            size=max_num_reqs,
            max_context_len=self.model_config.context_len + extra_max_context_len,
            device=self.device,
            enable_memory_saver=get_exec().features.enable_memory_saver,
        )

        head_num = self.model_config.get_num_kv_heads(
            get_parallel().attn_tp_size, get_parallel().attn_dcp_size
        )
        head_dim = self.model_config.head_dim
        if self.is_hybrid_swa_compress:


            v_head_dim = self.model_config.hf_text_config.v_head_dim
            swa_head_num = max(
                1,
                self.model_config.hf_text_config.swa_num_key_value_heads
                // get_parallel().attn_tp_size,
            )
            swa_head_dim = self.model_config.hf_text_config.swa_head_dim
            swa_v_head_dim = self.model_config.hf_text_config.swa_v_head_dim
        else:
            v_head_dim = head_dim
            swa_head_num = head_num
            swa_head_dim = head_dim
            swa_v_head_dim = head_dim


        swa_attention_layer_ids = [
            i
            for i in self.model_config.swa_attention_layer_ids
            if self.layer_info.start_layer <= i < self.layer_info.end_layer
        ]
        full_attention_layer_ids = [
            i
            for i in self.model_config.full_attention_layer_ids
            if self.layer_info.start_layer <= i < self.layer_info.end_layer
        ]

        bundle = init_unified_swa_pools(
            device=self.device,
            kv_cache_dtype=self.kv_cache_dtype,
            head_num=head_num,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
            swa_head_num=swa_head_num,
            swa_head_dim=swa_head_dim,
            swa_v_head_dim=swa_v_head_dim,
            page_size=self.page_size,
            start_layer=self.layer_info.start_layer,
            end_layer=self.layer_info.end_layer,
            swa_attention_layer_ids=swa_attention_layer_ids,
            full_attention_layer_ids=full_attention_layer_ids,
            full_max_total_num_tokens=full_max_total_num_tokens,
            swa_max_total_num_tokens=swa_max_total_num_tokens,
            enable_memory_saver=get_exec().features.enable_memory_saver,
            need_sort=get_disagg().disaggregation_mode in ("decode", "prefill"),


            forward_stream=self.forward_stream,

            lazy_compaction=_should_enable_lazy_compaction(),
        )
        return UnifiedPoolBundle(
            unified_memory_pool=bundle.unified_memory_pool,
            token_to_kv_pool=bundle.token_to_kv_pool,
            token_to_kv_pool_allocator=bundle.token_to_kv_pool_allocator,
            req_to_token_pool=req_to_token_pool,
        )

    def _validate_prefill_only_disable_kv_cache_pool_family(
        self,
        is_dsa_model: bool,
        is_dsv4_model: bool,
        current_platform,
    ):
        if not get_schedule().prefill_only_disable_kv_cache or self.is_draft_worker:
            return

        unsupported_pool_family = None
        if is_dsv4_model:
            unsupported_pool_family = "DeepSeekV4TokenToKVPool"
        elif current_platform.is_out_of_tree() and not self.mambaish_config:
            unsupported_pool_family = "out-of-tree platform KV pool"
        elif (
            get_exec().kernel.attention_backend == "ascend" and not self.mambaish_config
        ):
            unsupported_pool_family = "NPU/Ascend KV pool"
        elif self.use_mla_backend and self.is_hybrid_swa:
            unsupported_pool_family = "hybrid DSA/MLA-SWA KV pool"
        elif self.use_mla_backend and is_dsa_model:
            unsupported_pool_family = "DSA/MLA KV pool"
        elif self.use_mla_backend and not self.mambaish_config:
            unsupported_pool_family = "MLA KV pool"
        elif self.is_hybrid_swa:
            unsupported_pool_family = "SWA KV pool"
        elif self.mambaish_config:
            unsupported_pool_family = "hybrid linear/Mamba KV pool"
        elif is_float4_e2m1fn_x2(self.kv_cache_dtype):
            unsupported_pool_family = "FP4 MHA KV pool"

        if unsupported_pool_family is not None:
            raise RuntimeError(
                "--prefill-only-disable-kv-cache is not supported for "
                f"{unsupported_pool_family}. Supported configurations today: plain MHA "
                "models on CUDA with the FA (fa3/fa4) prefill backend, --is-embedding, "
                "--chunked-prefill-size=-1, --disable-radix-cache, no context-parallel "
                "attention, no HiSparse, and --kv-cache-dtype not in {nvfp4, fp4_mx_block16}."
            )

    def _build_req_to_token_pool(self, *, max_num_reqs: int) -> ReqToTokenPool:
        extra_max_context_len = get_req_to_token_extra_context_len()

        if get_disagg().disaggregation_mode == "decode":

            pre_alloc_size = get_disagg().disaggregation_decode_extra_slots
            if self.mambaish_config:
                req_to_token_pool = self._build_hybrid_mamba_decode_req_pool(
                    max_num_reqs=max_num_reqs,
                    extra_max_context_len=extra_max_context_len,
                    pre_alloc_size=pre_alloc_size,
                )
            else:
                req_to_token_pool = self._build_decode_req_pool(
                    max_num_reqs=max_num_reqs,
                    extra_max_context_len=extra_max_context_len,
                    pre_alloc_size=pre_alloc_size,
                )
        elif self.mambaish_config:
            req_to_token_pool = self._build_hybrid_req_pool(
                max_num_reqs=max_num_reqs,
                extra_max_context_len=extra_max_context_len,
            )
        else:
            req_to_token_pool = self._build_default_req_pool(
                max_num_reqs=max_num_reqs,
                extra_max_context_len=extra_max_context_len,
            )
        return req_to_token_pool

    def _build_hybrid_mamba_decode_req_pool(
        self,
        *,
        max_num_reqs: int,
        extra_max_context_len: int,
        pre_alloc_size: int,
    ) -> ReqToTokenPool:
        from sglang.srt.disaggregation.decode import (
            HybridMambaDecodeReqToTokenPool,
        )

        req_to_token_pool = HybridMambaDecodeReqToTokenPool(
            size=max_num_reqs,
            max_context_len=self.model_config.context_len + extra_max_context_len,
            device=self.device,
            enable_memory_saver=get_exec().features.enable_memory_saver,
            cache_params=self.mambaish_config.mamba2_cache_params,
            mamba_layer_ids=(
                [
                    i
                    for i in self.mambaish_config.mamba2_cache_params.layers
                    if self.layer_info.start_layer <= i < self.layer_info.end_layer
                ]
            ),
            speculative_num_draft_tokens=max_speculative_num_draft_tokens(),
            speculative_eagle_topk=get_spec().speculative_eagle_topk,
            enable_mamba_extra_buffer=mamba_extra_buffer_enabled(),
            pre_alloc_size=pre_alloc_size,
            enable_overlap_schedule=not get_schedule().disable_overlap_schedule,
            mamba_size=get_schedule().max_mamba_cache_size,
            start_layer=self.layer_info.start_layer,
            linear_replayssm_cache_len=get_exec().mamba.linear_replayssm_cache_len,
            mamba_envelope_layout=get_memory().enable_page_major_kv_layout,


            enable_linear_replayssm_spec=(
                get_exec().mamba.enable_linear_replayssm_spec
                and (
                    self.hybrid_gdn_config is not None
                    or kimi_linear_config(self.model_config) is not None
                )
            ),
        )
        return req_to_token_pool

    def _build_decode_req_pool(
        self,
        *,
        max_num_reqs: int,
        extra_max_context_len: int,
        pre_alloc_size: int,
    ) -> ReqToTokenPool:
        if _is_npu and is_deepseek_v4(self.model_config.hf_config):
            from sglang.srt.hardware_backend.npu.dsv4.dsv4_req_to_token_pool import (
                DSV4NPUDecodeReqToTokenPool as DecodeReqToTokenPool,
            )
        else:
            from sglang.srt.disaggregation.decode import DecodeReqToTokenPool

        req_to_token_pool = DecodeReqToTokenPool(
            size=max_num_reqs,
            max_context_len=self.model_config.context_len + extra_max_context_len,
            device=self.device,
            enable_memory_saver=get_exec().features.enable_memory_saver,
            pre_alloc_size=pre_alloc_size,
        )
        return req_to_token_pool

    def _build_hybrid_req_pool(
        self,
        *,
        max_num_reqs: int,
        extra_max_context_len: int,
    ) -> ReqToTokenPool:


        _algo = (get_spec().speculative_algorithm or "").upper()
        if (
            get_exec().mamba.enable_linear_replayssm_spec
            and _algo in ("DSPARK", "DFLASH")
            and kimi_linear_config(self.model_config) is None
        ):
            raise ValueError(
                "--enable-linear-replayssm-spec with DSPARK/DFLASH requires a KDA "
                "(kimi_linear) model; got a non-KDA model."
            )
        req_to_token_pool = HybridReqToTokenPool(
            size=max_num_reqs,
            mamba_size=get_schedule().max_mamba_cache_size,
            mamba_spec_state_size=max_num_reqs,
            max_context_len=self.model_config.context_len + extra_max_context_len,
            device=self.device,
            enable_memory_saver=get_exec().features.enable_memory_saver,
            cache_params=self.mambaish_config.mamba2_cache_params,
            mamba_layer_ids=(
                [
                    i
                    for i in self.mambaish_config.mamba2_cache_params.layers
                    if self.layer_info.start_layer <= i < self.layer_info.end_layer
                ]
            ),
            enable_mamba_extra_buffer=mamba_extra_buffer_enabled(),
            enable_mamba_extra_buffer_lazy=mamba_extra_buffer_lazy_enabled(),


            speculative_num_draft_tokens=(
                None
                if get_disagg().disaggregation_mode == "prefill"
                else max_speculative_num_draft_tokens()
            ),
            speculative_eagle_topk=get_spec().speculative_eagle_topk,
            spec_ssm_row_capacity=self._request_forest_ssm_row_capacity(max_num_reqs),
            enable_overlap_schedule=not get_schedule().disable_overlap_schedule,
            start_layer=self.layer_info.start_layer,
            enable_linear_replayssm=get_exec().mamba.enable_linear_replayssm,
            linear_replayssm_cache_len=get_exec().mamba.linear_replayssm_cache_len,
            mamba_envelope_layout=get_memory().enable_page_major_kv_layout,


            enable_linear_replayssm_spec=(
                get_exec().mamba.enable_linear_replayssm_spec
                and (
                    self.hybrid_gdn_config is not None
                    or kimi_linear_config(self.model_config) is not None
                )
            ),
        )
        return req_to_token_pool

    def _build_default_req_pool(
        self,
        *,
        max_num_reqs: int,
        extra_max_context_len: int,
    ) -> ReqToTokenPool:


        req_to_token_pool_cls = ReqToTokenPool
        if _is_npu and is_deepseek_v4(self.model_config.hf_config):
            from sglang.srt.hardware_backend.npu.dsv4.dsv4_req_to_token_pool import (
                DSV4NPUReqToTokenPool,
            )

            req_to_token_pool_cls = DSV4NPUReqToTokenPool

        req_to_token_pool = req_to_token_pool_cls(
            size=max_num_reqs,
            max_context_len=self.model_config.context_len + extra_max_context_len,
            device=self.device,
            enable_memory_saver=get_exec().features.enable_memory_saver,
        )
        return req_to_token_pool

    def _build_token_to_kv_pool(
        self,
        *,
        sizes: _PoolSizes,
        is_dsa_model: bool,
        is_dsv4_model: bool,
        req_to_token_pool: ReqToTokenPool,
    ) -> KVCache:


        enable_page_major = get_memory().enable_page_major_kv_layout
        if self.is_draft_worker and get_memory().enable_unified_memory:


            enable_page_major = False
        mha_pool_class = (
            PageMajorMHATokenToKVPool if enable_page_major else MHATokenToKVPool
        )

        if is_dsv4_model:
            token_to_kv_pool = self._build_dsv4_kv_pool(
                max_running_requests=sizes.max_running_requests,
                swa_max_total_num_tokens=sizes.swa_max_total_num_tokens,
                c4_max_total_num_tokens=sizes.c4_max_total_num_tokens,
                c128_max_total_num_tokens=sizes.c128_max_total_num_tokens,
                c4_state_pool_size=sizes.c4_state_pool_size,
                c128_state_pool_size=sizes.c128_state_pool_size,
                c4_state_dtype=sizes.c4_state_dtype,
                c128_state_dtype=sizes.c128_state_dtype,
                req_to_token_pool=req_to_token_pool,
            )
        elif current_platform.is_out_of_tree() and not self.mambaish_config:
            if self.use_mla_backend and is_dsa_model:
                token_to_kv_pool = self._build_oot_dsa_kv_pool(
                    max_total_num_tokens=sizes.max_total_num_tokens,
                )
            elif self.use_mla_backend:
                token_to_kv_pool = self._build_oot_mla_kv_pool(
                    max_total_num_tokens=sizes.max_total_num_tokens,
                    is_dsa_model=is_dsa_model,
                )
            else:
                token_to_kv_pool = self._build_oot_mha_kv_pool(
                    max_total_num_tokens=sizes.max_total_num_tokens,
                )
        elif (
            get_exec().kernel.attention_backend == "ascend" and not self.mambaish_config
        ):
            if self.is_hybrid_swa:
                token_to_kv_pool = self._build_ascend_swa_kv_pool(
                    full_max_total_num_tokens=sizes.full_max_total_num_tokens,
                    swa_max_total_num_tokens=sizes.swa_max_total_num_tokens,
                )
            elif is_minimax_sparse(self.model_config.hf_config):
                token_to_kv_pool = self._build_ascend_minimax_sparse_kv_pool(
                    max_total_num_tokens=sizes.max_total_num_tokens,
                )
            elif self.use_mla_backend:
                token_to_kv_pool = self._build_ascend_mla_kv_pool(
                    max_total_num_tokens=sizes.max_total_num_tokens,
                    is_dsa_model=is_dsa_model,
                )
            else:
                token_to_kv_pool = self._build_ascend_mha_kv_pool(
                    max_total_num_tokens=sizes.max_total_num_tokens,
                )
        elif self.use_mla_backend and self.is_hybrid_swa:
            token_to_kv_pool = self._build_hybrid_mla_swa_kv_pool(
                full_max_total_num_tokens=sizes.full_max_total_num_tokens,
                swa_max_total_num_tokens=sizes.swa_max_total_num_tokens,
                is_dsa_model=is_dsa_model,
            )
        elif self.use_mla_backend and is_dsa_model:
            token_to_kv_pool = self._build_dsa_kv_pool(
                max_total_num_tokens=sizes.max_total_num_tokens,
            )
        elif self.use_mla_backend and not self.mambaish_config:
            assert not is_dsa_model
            if is_float4_e2m1fn_x2(self.kv_cache_dtype):
                token_to_kv_pool = self._build_mla_fp4_kv_pool(
                    max_total_num_tokens=sizes.max_total_num_tokens,
                )
            else:
                token_to_kv_pool = self._build_mla_kv_pool(
                    max_total_num_tokens=sizes.max_total_num_tokens,
                )
        else:
            if self.is_hybrid_swa:
                token_to_kv_pool = self._build_hybrid_swa_kv_pool(
                    full_max_total_num_tokens=sizes.full_max_total_num_tokens,
                    swa_max_total_num_tokens=sizes.swa_max_total_num_tokens,
                    mha_pool_class=mha_pool_class,
                )
            elif is_minimax_sparse(self.model_config.hf_config):
                token_to_kv_pool = self._build_minimax_sparse_kv_pool(
                    max_total_num_tokens=sizes.max_total_num_tokens,
                )
            elif self.mambaish_config:
                token_to_kv_pool = self._build_hybrid_linear_kv_pool(
                    max_total_num_tokens=sizes.max_total_num_tokens,
                    req_to_token_pool=req_to_token_pool,
                    mha_pool_class=mha_pool_class,
                )
            else:
                quant_method = None
                if is_float4_e2m1fn_x2(self.kv_cache_dtype):
                    assert (
                        not enable_page_major
                    ), "page-major KV layout is not supported with fp4 KV cache"
                    quant_method = self._build_fp4_quant_method(
                        num_layers=self.layer_info.num_effective_layers
                    )
                token_to_kv_pool = self._build_mha_kv_pool(
                    max_total_num_tokens=sizes.max_total_num_tokens,
                    mha_pool_class=mha_pool_class,
                    quant_method=quant_method,
                )
        return token_to_kv_pool

    def _build_dsv4_kv_pool(
        self,
        *,
        max_running_requests: int,
        swa_max_total_num_tokens: Optional[int],
        c4_max_total_num_tokens: int,
        c128_max_total_num_tokens: int,
        c4_state_pool_size: int,
        c128_state_pool_size: int,
        c4_state_dtype: Optional[torch.dtype],
        c128_state_dtype: Optional[torch.dtype],
        req_to_token_pool: ReqToTokenPool,
    ) -> KVCache:
        swa_page_size = get_schedule().page_size
        if not _is_npu:
            assert swa_page_size == 256, "In paged swa mode, page_size must be 256."

        if self.is_draft_worker:
            from sglang.srt.models.deepseek_v4_nextn import (
                COMPRESS_RATIO_NEXTN_LAYER,
            )

            compression_ratios = [
                COMPRESS_RATIO_NEXTN_LAYER
            ] * self.layer_info.num_effective_layers
        else:
            compression_ratios = self.model_config.compress_ratios


        if _is_npu:
            from sglang.srt.hardware_backend.npu.dsv4.dsv4_memory_pool import (
                DSV4NPUTokenToKVPool,
            )

            pool_cls = DSV4NPUTokenToKVPool
        else:
            pool_cls = DeepSeekV4TokenToKVPool

        token_to_kv_pool = pool_cls(
            max_num_reqs=max_running_requests,


            num_req_slots=req_to_token_pool.req_to_token.shape[0],
            swa_size=swa_max_total_num_tokens,
            c4_size=c4_max_total_num_tokens,
            c128_size=c128_max_total_num_tokens,
            c4_state_pool_size=c4_state_pool_size,
            c128_state_pool_size=c128_state_pool_size,
            page_size=self.pool_page_size,
            swa_page_size=swa_page_size,
            sliding_window=self.model_config.window_size,
            dtype=self.kv_cache_dtype,
            c4_state_dtype=c4_state_dtype,
            c128_state_dtype=c128_state_dtype,
            qk_nope_head_dim=self.model_config.qk_nope_head_dim,
            qk_rope_head_dim=self.model_config.qk_rope_head_dim,
            indexer_head_dim=self.model_config.index_head_dim,
            layer_num=self.layer_info.num_effective_layers,
            device=self.device,
            enable_memory_saver=get_exec().features.enable_memory_saver,
            compression_ratios=compression_ratios,
            start_layer=self.layer_info.start_layer,
            end_layer=self.layer_info.end_layer,
            enable_hisparse=get_memory().enable_hisparse,
            online_mtp_max_draft_tokens=(max_speculative_num_draft_tokens() or 0),
        )
        return token_to_kv_pool

    def _build_oot_dsa_kv_pool(self, *, max_total_num_tokens: int) -> KVCache:
        PoolCls = current_platform.get_dsa_kv_pool_cls()
        token_to_kv_pool = PoolCls(
            max_total_num_tokens,
            page_size=self.pool_page_size,
            dtype=self.kv_cache_dtype,
            kv_lora_rank=self.model_config.kv_lora_rank,
            qk_rope_head_dim=self.model_config.qk_rope_head_dim,
            layer_num=self.layer_info.num_effective_layers,
            device=self.device,
            kv_cache_dim=calculate_mla_kv_cache_dim(
                model_config=self.model_config,
                kv_cache_dtype=self.kv_cache_dtype,
            ),
            enable_memory_saver=get_exec().features.enable_memory_saver,
            start_layer=self.layer_info.start_layer,
            end_layer=self.layer_info.end_layer,
            index_head_dim=get_dsa_index_head_dim(self.model_config.hf_config),
        )
        return token_to_kv_pool

    def _build_oot_mla_kv_pool(
        self, *, max_total_num_tokens: int, is_dsa_model: bool
    ) -> KVCache:
        PoolCls = current_platform.get_mla_kv_pool_cls()
        token_to_kv_pool = PoolCls(
            max_total_num_tokens,
            page_size=self.pool_page_size,
            dtype=self.kv_cache_dtype,
            kv_lora_rank=self.model_config.kv_lora_rank,
            qk_rope_head_dim=self.model_config.qk_rope_head_dim,
            index_head_dim=(self.model_config.index_head_dim if is_dsa_model else None),
            layer_num=self.layer_info.num_effective_layers,
            device=self.device,
            enable_memory_saver=get_exec().features.enable_memory_saver,
            start_layer=self.layer_info.start_layer,
            end_layer=self.layer_info.end_layer,
        )
        return token_to_kv_pool

    def _build_oot_mha_kv_pool(self, *, max_total_num_tokens: int) -> KVCache:
        PoolCls = current_platform.get_mha_kv_pool_cls()
        token_to_kv_pool = PoolCls(
            max_total_num_tokens,
            page_size=self.pool_page_size,
            dtype=self.kv_cache_dtype,
            head_num=self.model_config.get_num_kv_heads(
                get_parallel().attn_tp_size, get_parallel().attn_dcp_size
            ),
            head_dim=self.model_config.head_dim,
            layer_num=self.layer_info.num_effective_layers,
            device=self.device,
            enable_memory_saver=get_exec().features.enable_memory_saver,
            start_layer=self.layer_info.start_layer,
            end_layer=self.layer_info.end_layer,
        )
        return token_to_kv_pool

    def _build_ascend_swa_kv_pool(
        self,
        *,
        full_max_total_num_tokens: Optional[int],
        swa_max_total_num_tokens: Optional[int],
    ) -> KVCache:
        from sglang.srt.hardware_backend.npu.memory_pool_npu import (
            NPUMHATokenToKVPool,
        )

        kwargs = {}
        if self.is_hybrid_swa_compress:
            kwargs = {
                "swa_head_num": max(
                    1,
                    self.model_config.hf_text_config.swa_num_key_value_heads
                    // get_parallel().attn_tp_size,
                ),
                "swa_head_dim": self.model_config.swa_head_dim,
                "swa_v_head_dim": self.model_config.swa_v_head_dim,
                "v_head_dim": self.model_config.v_head_dim,
            }
        token_to_kv_pool = SWAKVPool(
            size=full_max_total_num_tokens,
            size_swa=swa_max_total_num_tokens,
            page_size=self.pool_page_size,
            dtype=self.kv_cache_dtype,
            post_capture_active=self.post_capture_kv_active,
            head_num=self.model_config.get_num_kv_heads(
                get_parallel().attn_tp_size, get_parallel().attn_dcp_size
            ),
            head_dim=self.model_config.head_dim,
            swa_attention_layer_ids=self.model_config.swa_attention_layer_ids,
            full_attention_layer_ids=self.model_config.full_attention_layer_ids,
            device=self.device,
            token_to_kv_pool_class=NPUMHATokenToKVPool,
            **kwargs,
        )
        return token_to_kv_pool

    def _build_ascend_minimax_sparse_kv_pool(
        self, *, max_total_num_tokens: int
    ) -> KVCache:
        _hf_config = self.model_config.hf_config
        sparse_cfg = get_minimax_sparse_attention_config(_hf_config)
        dense_layer_ids, sparse_layer_ids = get_minimax_sparse_layer_ids(sparse_cfg)
        disable_value_sparse_layer_ids = get_minimax_sparse_disable_value_layer_ids(
            sparse_cfg
        )
        from sglang.srt.hardware_backend.npu.memory_pool_npu import (
            NPUMiniMaxSparseKVPool,
        )

        token_to_kv_pool = NPUMiniMaxSparseKVPool(
            size=max_total_num_tokens,
            page_size=get_schedule().page_size,
            dtype=self.kv_cache_dtype,
            index_dtype=self.model_dtype,
            head_num=self.model_config.get_num_kv_heads(get_parallel().attn_tp_size),
            head_dim=self.model_config.head_dim,
            idx_head_dim=sparse_cfg["sparse_index_dim"],
            dense_layer_ids=dense_layer_ids,
            sparse_layer_ids=sparse_layer_ids,
            disable_value_sparse_layer_ids=disable_value_sparse_layer_ids,
            device=self.device,
            enable_memory_saver=get_exec().features.enable_memory_saver,
            start_layer=self.layer_info.start_layer,
            end_layer=self.layer_info.end_layer,
        )
        return token_to_kv_pool

    def _build_ascend_mla_kv_pool(
        self, *, max_total_num_tokens: int, is_dsa_model: bool
    ) -> KVCache:
        from sglang.srt.hardware_backend.npu.memory_pool_npu import (
            NPUMLATokenToKVPool,
        )

        token_to_kv_pool = NPUMLATokenToKVPool(
            max_total_num_tokens,
            page_size=self.pool_page_size,
            dtype=self.kv_cache_dtype,
            kv_lora_rank=self.model_config.kv_lora_rank,
            qk_rope_head_dim=self.model_config.qk_rope_head_dim,
            index_head_dim=(self.model_config.index_head_dim if is_dsa_model else None),
            layer_num=self.layer_info.num_effective_layers,
            device=self.device,
            enable_memory_saver=get_exec().features.enable_memory_saver,
            start_layer=self.layer_info.start_layer,
            end_layer=self.layer_info.end_layer,
        )
        return token_to_kv_pool

    def _build_ascend_mha_kv_pool(self, *, max_total_num_tokens: int) -> KVCache:
        from sglang.srt.hardware_backend.npu.memory_pool_npu import (
            NPUMHATokenToKVPool,
        )

        token_to_kv_pool = NPUMHATokenToKVPool(
            max_total_num_tokens,
            page_size=self.pool_page_size,
            dtype=self.kv_cache_dtype,
            head_num=self.model_config.get_num_kv_heads(
                get_parallel().attn_tp_size, get_parallel().attn_dcp_size
            ),
            head_dim=self.model_config.head_dim,
            layer_num=self.layer_info.num_effective_layers,
            device=self.device,
            enable_memory_saver=get_exec().features.enable_memory_saver,
            start_layer=self.layer_info.start_layer,
            end_layer=self.layer_info.end_layer,
        )
        return token_to_kv_pool

    def _build_dsa_kv_pool(self, *, max_total_num_tokens: int) -> KVCache:
        from sglang.srt.layers.cp.utils import get_glm_dsa_cp_layer_shard_info

        (
            dsa_cp_layer_shard_rank,
            dsa_cp_layer_shard_size,
        ) = get_glm_dsa_cp_layer_shard_info(self)
        pool_kwargs = {}
        if get_memory().enable_hisparse:
            PoolCls = HiSparseDSATokenToKVPool
            from sglang.srt.mem_cache.sparsity import parse_hisparse_config

            pool_kwargs["host_to_device_ratio"] = parse_hisparse_config(
                self.server_args
            ).host_to_device_ratio
        elif dsa_cp_layer_shard_rank is not None:

            from sglang.srt.mem_cache.dsa_cache_layer_split import (
                LayerSplitDSATokenToKVPool,
            )

            PoolCls = LayerSplitDSATokenToKVPool
            pool_kwargs["layer_shard_rank"] = dsa_cp_layer_shard_rank
            pool_kwargs["layer_shard_size"] = dsa_cp_layer_shard_size
        else:
            PoolCls = DSATokenToKVPool
        if _should_elide_dsa_index_k(is_draft_worker=self.is_draft_worker):
            pool_kwargs["skip_topk_layers"] = [
                dsa_layer_skips_topk(self.model_config.hf_config, layer_id)
                for layer_id in range(
                    self.layer_info.start_layer, self.layer_info.end_layer
                )
            ]
        token_to_kv_pool = PoolCls(
            max_total_num_tokens,
            page_size=self.pool_page_size,
            dtype=self.kv_cache_dtype,
            kv_lora_rank=self.model_config.kv_lora_rank,
            qk_rope_head_dim=self.model_config.qk_rope_head_dim,
            layer_num=self.layer_info.num_effective_layers,
            device=self.device,
            kv_cache_dim=calculate_mla_kv_cache_dim(
                model_config=self.model_config,
                kv_cache_dtype=self.kv_cache_dtype,
            ),
            enable_memory_saver=get_exec().features.enable_memory_saver,
            start_layer=self.layer_info.start_layer,
            end_layer=self.layer_info.end_layer,
            index_head_dim=get_dsa_index_head_dim(self.model_config.hf_config),
            **pool_kwargs,
        )
        return token_to_kv_pool

    def _build_hybrid_mla_swa_kv_pool(
        self,
        *,
        full_max_total_num_tokens: int,
        swa_max_total_num_tokens: int,
        is_dsa_model: bool,
    ) -> KVCache:

        full_pool_class = DSATokenToKVPool if is_dsa_model else MLATokenToKVPool
        common = {
            "page_size": get_schedule().page_size,
            "device": self.device,
            "enable_memory_saver": False,
        }
        full_pool_kwargs = {
            **common,
            "kv_lora_rank": self.model_config.kv_lora_rank,
            "qk_rope_head_dim": self.model_config.qk_rope_head_dim,
        }
        if is_dsa_model:
            full_pool_kwargs.update(
                index_head_dim=get_dsa_index_head_dim(self.model_config.hf_config),
                kv_cache_dim=calculate_mla_kv_cache_dim(
                    model_config=self.model_config,
                    kv_cache_dtype=self.kv_cache_dtype,
                ),
            )

        return SWAKVPool(
            size=full_max_total_num_tokens,
            size_swa=swa_max_total_num_tokens,
            page_size=get_schedule().page_size,
            dtype=self.kv_cache_dtype,
            head_num=0,
            head_dim=0,
            swa_attention_layer_ids=self.model_config.swa_attention_layer_ids,
            full_attention_layer_ids=self.model_config.full_attention_layer_ids,
            device=self.device,
            full_kv_pool_class=full_pool_class,
            swa_kv_pool_class=MLATokenToKVPool,
            full_kv_pool_kwargs=full_pool_kwargs,
            swa_kv_pool_kwargs={
                **common,
                "kv_lora_rank": self.model_config.swa_kv_lora_rank,
                "qk_rope_head_dim": self.model_config.swa_qk_rope_head_dim,
            },
        )

    def _build_mla_fp4_kv_pool(self, *, max_total_num_tokens: int) -> KVCache:
        token_to_kv_pool = MLATokenToKVPoolFP4(
            max_total_num_tokens,
            page_size=self.pool_page_size,
            dtype=self.kv_cache_dtype,
            kv_lora_rank=self.model_config.kv_lora_rank,
            qk_rope_head_dim=self.model_config.qk_rope_head_dim,
            layer_num=self.layer_info.num_effective_layers,
            device=self.device,
            enable_memory_saver=get_exec().features.enable_memory_saver,
            start_layer=self.layer_info.start_layer,
            end_layer=self.layer_info.end_layer,
        )
        return token_to_kv_pool

    def _build_mla_kv_pool(self, *, max_total_num_tokens: int) -> KVCache:
        token_to_kv_pool = MLATokenToKVPool(
            max_total_num_tokens,
            page_size=self.pool_page_size,
            dtype=self.kv_cache_dtype,
            kv_lora_rank=self.model_config.kv_lora_rank,
            qk_rope_head_dim=self.model_config.qk_rope_head_dim,
            layer_num=self.layer_info.num_effective_layers,
            device=self.device,
            enable_memory_saver=get_exec().features.enable_memory_saver,
            start_layer=self.layer_info.start_layer,
            end_layer=self.layer_info.end_layer,
        )
        return token_to_kv_pool

    def _build_hybrid_swa_kv_pool(
        self,
        *,
        full_max_total_num_tokens: Optional[int],
        swa_max_total_num_tokens: Optional[int],
        mha_pool_class: type,
    ) -> KVCache:
        kwargs = {}
        if self.is_hybrid_swa_compress:
            kwargs = {
                "swa_head_num": max(
                    1,
                    self.model_config.hf_text_config.swa_num_key_value_heads
                    // get_parallel().attn_tp_size,
                ),
                "swa_head_dim": self.model_config.swa_head_dim,
                "swa_v_head_dim": self.model_config.swa_v_head_dim,
                "v_head_dim": self.model_config.v_head_dim,
            }
        swa_pool_class = (
            MHATokenToKVPoolMXFP8
            if self.kv_cache_dtype_str == "mxfp8"
            else mha_pool_class
        )
        swa_attention_layer_ids = self.model_config.swa_attention_layer_ids
        full_attention_layer_ids = self.model_config.full_attention_layer_ids
        if self.is_inkling_mtp_draft:
            if self.draft_swa_full_capacity:


                swa_attention_layer_ids = [self.draft_model_idx]
                full_attention_layer_ids = []
            else:
                swa_attention_layer_ids = []
                full_attention_layer_ids = [self.draft_model_idx]


        size_swa = (
            full_max_total_num_tokens
            if self.draft_swa_full_capacity
            else swa_max_total_num_tokens
        )
        token_to_kv_pool = SWAKVPool(
            size=full_max_total_num_tokens,
            size_swa=size_swa,
            page_size=self.pool_page_size,
            dtype=self.kv_cache_dtype,
            post_capture_active=self.post_capture_kv_active,
            head_num=self.model_config.get_num_kv_heads(
                get_parallel().attn_tp_size, get_parallel().attn_dcp_size
            ),
            head_dim=self.model_config.head_dim,
            swa_attention_layer_ids=swa_attention_layer_ids,
            full_attention_layer_ids=full_attention_layer_ids,
            device=self.device,
            enable_kv_cache_copy=(get_spec().speculative_algorithm is not None),
            token_to_kv_pool_class=swa_pool_class,
            **kwargs,
        )
        return token_to_kv_pool

    def _build_minimax_sparse_kv_pool(self, *, max_total_num_tokens: int) -> KVCache:
        from sglang.srt.server_args import m3_fp8_attn_gemm_enabled

        _hf_config = self.model_config.hf_config
        sparse_cfg = get_minimax_sparse_attention_config(_hf_config)
        dense_layer_ids, sparse_layer_ids = get_minimax_sparse_layer_ids(sparse_cfg)
        disable_value_sparse_layer_ids = get_minimax_sparse_disable_value_layer_ids(
            sparse_cfg
        )
        token_to_kv_pool = MiniMaxSparseKVPool(
            size=max_total_num_tokens,
            page_size=self.pool_page_size,
            dtype=self.kv_cache_dtype,


            index_dtype=(
                self.kv_cache_dtype
                if m3_fp8_attn_gemm_enabled(resolving_view(self.server_args))
                else self.model_dtype
            ),
            head_num=self.model_config.get_num_kv_heads(
                get_parallel().attn_tp_size, get_parallel().attn_dcp_size
            ),
            head_dim=self.model_config.head_dim,
            idx_head_dim=sparse_cfg["sparse_index_dim"],
            dense_layer_ids=dense_layer_ids,
            sparse_layer_ids=sparse_layer_ids,
            disable_value_sparse_layer_ids=disable_value_sparse_layer_ids,
            device=self.device,
            enable_memory_saver=get_exec().features.enable_memory_saver,
            start_layer=self.layer_info.start_layer,
            end_layer=self.layer_info.end_layer,
        )
        return token_to_kv_pool

    def _build_hybrid_linear_kv_pool(
        self,
        *,
        max_total_num_tokens: int,
        req_to_token_pool: ReqToTokenPool,
        mha_pool_class: type,
    ) -> KVCache:
        extra_args = {}
        if self.use_mla_backend:
            extra_args = {
                "kv_lora_rank": self.model_config.kv_lora_rank,
                "qk_rope_head_dim": self.model_config.qk_rope_head_dim,
            }
        full_attention_layer_ids = (
            [0]
            if self.is_draft_worker
            else [
                i
                for i in self.mambaish_config.full_attention_layer_ids
                if self.layer_info.start_layer <= i < self.layer_info.end_layer
            ]
        )
        quant_method = self._build_fp4_quant_method(
            num_layers=len(full_attention_layer_ids)
        )


        full_pool_class = (
            MHATokenToKVPoolMXFP8
            if self.kv_cache_dtype_str == "mxfp8" and not self.use_mla_backend
            else mha_pool_class
        )
        token_to_kv_pool = HybridLinearKVPool(
            page_size=self.pool_page_size,
            size=max_total_num_tokens,
            dtype=self.kv_cache_dtype,
            head_num=self.model_config.get_num_kv_heads(
                get_parallel().attn_tp_size, get_parallel().attn_dcp_size
            ),
            head_dim=self.model_config.head_dim,

            full_attention_layer_ids=full_attention_layer_ids,
            device=self.device,
            mamba_pool=req_to_token_pool.mamba_pool,
            enable_memory_saver=get_exec().features.enable_memory_saver,
            enable_kv_cache_copy=(get_spec().speculative_algorithm is not None),
            use_mla=self.use_mla_backend,
            start_layer=self.layer_info.start_layer,
            full_kv_pool_class=full_pool_class,
            quant_method=quant_method,
            post_capture_active=self.post_capture_kv_active and quant_method is None,
            **extra_args,
        )
        return token_to_kv_pool

    def _build_mha_fp4_kv_pool(self, *, max_total_num_tokens: int) -> KVCache:
        token_to_kv_pool = MHATokenToKVPoolFP4(
            max_total_num_tokens,
            page_size=self.pool_page_size,
            dtype=self.kv_cache_dtype,
            head_num=self.model_config.get_num_kv_heads(
                get_parallel().attn_tp_size, get_parallel().attn_dcp_size
            ),
            head_dim=self.model_config.head_dim,
            v_head_dim=self.model_config.v_head_dim,
            layer_num=self.layer_info.num_effective_layers,
            device=self.device,
            enable_memory_saver=get_exec().features.enable_memory_saver,
            start_layer=self.layer_info.start_layer,
            end_layer=self.layer_info.end_layer,
            enable_alt_stream=not get_disagg().enable_pdmux,
            enable_kv_cache_copy=(get_spec().speculative_algorithm is not None),
        )
        return token_to_kv_pool

    def _build_mha_kv_pool(
        self, *, max_total_num_tokens: int, mha_pool_class: type, quant_method=None
    ) -> KVCache:
        if self.kv_cache_dtype_str == "mxfp8":
            pool_cls = MHATokenToKVPoolMXFP8
        else:
            pool_cls = (
                NoOpMHATokenToKVPool
                if get_schedule().prefill_only_disable_kv_cache
                else mha_pool_class
            )
        pool_kwargs = {}
        if quant_method is not None:
            pool_kwargs["quant_method"] = quant_method
        else:
            pool_kwargs["post_capture_active"] = self.post_capture_kv_active
        token_to_kv_pool = pool_cls(
            max_total_num_tokens,
            page_size=self.pool_page_size,
            dtype=self.kv_cache_dtype,
            head_num=self.model_config.get_num_kv_heads(
                get_parallel().attn_tp_size, get_parallel().attn_dcp_size
            ),
            head_dim=self.model_config.head_dim,
            v_head_dim=self.model_config.v_head_dim,
            layer_num=self.layer_info.num_effective_layers,
            device=self.device,
            enable_memory_saver=get_exec().features.enable_memory_saver,
            start_layer=self.layer_info.start_layer,
            end_layer=self.layer_info.end_layer,
            enable_alt_stream=not get_disagg().enable_pdmux,
            enable_kv_cache_copy=(get_spec().speculative_algorithm is not None),
            **pool_kwargs,
        )
        return token_to_kv_pool

    def _build_token_to_kv_pool_allocator(
        self,
        *,
        sizes: _PoolSizes,
        token_to_kv_pool: KVCache,
        is_dsv4_model: bool,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator],
    ) -> BaseTokenToKVPoolAllocator:

        need_sort = get_disagg().disaggregation_mode in ("decode", "prefill")
        if token_to_kv_pool_allocator is None:
            if current_platform.is_out_of_tree():
                AllocatorCls = current_platform.get_paged_allocator_cls()
                token_to_kv_pool_allocator = AllocatorCls(
                    sizes.max_total_num_tokens,
                    page_size=get_schedule().page_size,
                    dtype=self.kv_cache_dtype,
                    device=self.device,
                    kvcache=token_to_kv_pool,
                    need_sort=need_sort,
                )
            elif _is_npu and (
                get_exec().kernel.attention_backend == "ascend"
                or is_dsv4_model
                or self.hybrid_gdn_config is not None
            ):
                if self.is_hybrid_swa:


                    if is_dsv4_model:
                        from sglang.srt.hardware_backend.npu.dsv4.dsv4_allocator import (
                            DSV4NPUTokenToKVPoolAllocator,
                        )

                        swa_allocator_cls = DSV4NPUTokenToKVPoolAllocator
                    else:
                        swa_allocator_cls = SWATokenToKVPoolAllocator
                    token_to_kv_pool_allocator = swa_allocator_cls(
                        sizes.full_max_total_num_tokens,
                        sizes.swa_max_total_num_tokens,
                        page_size=get_schedule().page_size,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=token_to_kv_pool,
                        need_sort=need_sort,
                    )
                else:
                    from sglang.srt.hardware_backend.npu.allocator_npu import (
                        NPUPagedTokenToKVPoolAllocator,
                    )

                    token_to_kv_pool_allocator = NPUPagedTokenToKVPoolAllocator(
                        sizes.max_total_num_tokens,
                        page_size=get_schedule().page_size,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=token_to_kv_pool,
                        need_sort=need_sort,
                    )
            else:
                if self.is_hybrid_swa and sizes.full_max_total_num_tokens == 0:
                    token_to_kv_pool_allocator = PureSWATokenToKVPoolAllocator(
                        sizes.swa_max_total_num_tokens,
                        page_size=get_schedule().page_size,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=token_to_kv_pool,
                        need_sort=need_sort,
                    )
                elif self.is_hybrid_swa:
                    token_to_kv_pool_allocator = SWATokenToKVPoolAllocator(
                        sizes.full_max_total_num_tokens,
                        sizes.swa_max_total_num_tokens,
                        page_size=get_schedule().page_size,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=token_to_kv_pool,
                        need_sort=need_sort,
                    )
                else:
                    if get_memory().enable_hisparse:
                        from sglang.srt.mem_cache.sparsity import (
                            parse_hisparse_config,
                        )

                        hisparse_cfg = parse_hisparse_config(self.server_args)
                        token_to_kv_pool_allocator = HiSparseTokenToKVPoolAllocator(
                            sizes.max_total_num_tokens,
                            page_size=get_schedule().page_size,
                            dtype=self.kv_cache_dtype,
                            device=self.device,
                            kvcache=token_to_kv_pool,
                            need_sort=need_sort,
                            host_to_device_ratio=hisparse_cfg.host_to_device_ratio,
                        )
                    elif (
                        get_schedule().page_size == 1 and not get_parallel().dcp_enabled
                    ):
                        token_to_kv_pool_allocator = TokenToKVPoolAllocator(
                            sizes.max_total_num_tokens,
                            dtype=self.kv_cache_dtype,
                            device=self.device,
                            kvcache=token_to_kv_pool,
                            need_sort=need_sort,
                        )
                    else:
                        token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
                            sizes.max_total_num_tokens * get_parallel().attn_dcp_size,
                            page_size=get_schedule().page_size
                            * get_parallel().attn_dcp_size,
                            dtype=self.kv_cache_dtype,
                            device=self.device,
                            kvcache=token_to_kv_pool,
                            need_sort=need_sort,
                        )

            if get_memory().enable_hisparse and is_dsv4_model:
                assert self.is_hybrid_swa, "DeepSeek V4 HiSparse requires SWA mode."
                token_to_kv_pool_allocator = DeepSeekV4HiSparseTokenToKVPoolAllocator(
                    token_to_kv_pool_allocator
                )


            if hasattr(req_to_token_pool, "register_dsv4_allocator"):
                req_to_token_pool.register_dsv4_allocator(token_to_kv_pool_allocator)

        else:
            assert self.is_draft_worker
            if self.is_hybrid_swa:
                if self.draft_swa_full_capacity:


                    n = sizes.full_max_total_num_tokens + self.page_size
                    identity_mapping = torch.arange(
                        n + 1, dtype=torch.int64, device=self.device
                    )
                    identity_mapping[-1] = -1
                    token_to_kv_pool.register_mapping(identity_mapping)
                else:
                    swa_allocator = getattr(
                        token_to_kv_pool_allocator,
                        "logical_attn_allocator",
                        token_to_kv_pool_allocator,
                    )
                    assert isinstance(swa_allocator, SWATokenToKVPoolAllocator)
                    token_to_kv_pool.register_mapping(
                        swa_allocator.full_to_swa_index_mapping
                    )
        return token_to_kv_pool_allocator

    def _profile_available_bytes(self, pre_model_load_memory: int) -> int:


        available_gpu_memory = get_available_gpu_memory(
            self.device,
            self.gpu_id,
            distributed=get_world_group().world_size > 1,
            cpu_group=get_world_group().cpu_group,
        )

        slack_gb = pre_model_load_memory * (1 - get_schedule().mem_fraction_static)
        if self.mambaish_config is not None and self.post_capture_kv_active:

            slack_gb = max(
                slack_gb,
                pre_capture_activation_reserve_mb(
                    get_device_memory_capacity(self.device)
                )
                / 1024,
            )
        mm_reservation_gb = mm_runtime_reservation_gb(
            is_multimodal=self.model_config.is_multimodal,
            mm_feature_transport=get_mm().mm_feature_transport,
        )
        rest_memory = available_gpu_memory - slack_gb - mm_reservation_gb
        if self.mambaish_config is not None:
            rest_memory = self._handle_max_mamba_cache(rest_memory)


        if rest_memory <= 0:
            minimum_mem_fraction_static = (
                1 - available_gpu_memory / pre_model_load_memory
            )
            suggested_mem_fraction_static = (
                math.ceil(minimum_mem_fraction_static * 1000) / 1000
            )
            raise ValueError(
                f"Loaded weights leave no GPU memory for the KV cache under "
                f"--mem-fraction-static={get_schedule().mem_fraction_static}. "
                f"Raise --mem-fraction-static above "
                f"{suggested_mem_fraction_static:.3f} "
                f"(minimum viable = 1 - available/pre = "
                f"{minimum_mem_fraction_static:.4f}). If using speculative "
                f"decoding, draft weights are now counted."
            )

        return int(rest_memory * (1 << 30))

    def _calculate_mamba_ratio(self) -> int:
        if get_memory().disable_radix_cache:
            return 1

        skip_decode_lock = envs.SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK.get()
        base = MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO - (
            MAMBA_CACHE_BASE_RATIO_DROP_ON_SKIP if skip_decode_lock else 0
        )

        additional_ratio = 0
        if mamba_extra_buffer_enabled():


            if not get_schedule().disable_overlap_schedule:
                if mamba_extra_buffer_lazy_enabled():
                    additional_ratio = MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP_LAZY
                else:
                    additional_ratio = MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP
            else:
                assert (
                    not mamba_extra_buffer_lazy_enabled()
                ), "Lazy extra buffer requires overlap schedule (--disable-overlap-schedule is incompatible)"
                additional_ratio = MAMBA_CACHE_V2_ADDITIONAL_RATIO_NO_OVERLAP
        elif skip_decode_lock:


            additional_ratio = MAMBA_CACHE_V2_ADDITIONAL_RATIO_NO_BUFFER

        return base + additional_ratio

    def _apply_token_constraints(self, token_capacity: int) -> int:

        user_limit = get_schedule().max_total_tokens


        if user_limit is not None:
            if user_limit > token_capacity:
                logging.warning(
                    f"max_total_tokens={user_limit} is larger than the profiled value "
                    f"{token_capacity}. Use the profiled value instead."
                )
            token_capacity = min(token_capacity, user_limit)


        if get_parallel().config.pp_size > 1:
            tensor = torch.tensor(token_capacity, dtype=torch.int64)
            torch.distributed.all_reduce(
                tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=get_world_group().cpu_group,
            )
            token_capacity = tensor.item()

        return token_capacity

    def resolve_max_num_reqs(self, token_capacity: int) -> int:


        estimated = int(token_capacity / self.model_config.context_len * 512)
        estimated = max(min(estimated, 4096), 2048)

        max_num_reqs = get_schedule().max_running_requests
        if max_num_reqs is not None:
            requested_per_worker = max_num_reqs // self.ps.attn_dp_size
            max_num_reqs = min(requested_per_worker, token_capacity // 2)
        else:
            requested_per_worker = None
            max_num_reqs = min(estimated, token_capacity // 2)

        capped_by_mamba = False
        if self.mambaish_config is not None:
            ratio = self._calculate_mamba_ratio()
            mamba_cap = get_schedule().max_mamba_cache_size // ratio
            if mamba_cap < max_num_reqs:
                capped_by_mamba = True
                logger.warning(
                    "max_running_requests is capped to %d by the mamba state "
                    "cache (max_mamba_cache_size=%d, %d state slots per "
                    "request). To raise it: increase --mamba-full-memory-ratio "
                    "or --max-mamba-cache-size, or halve the state size with "
                    "--mamba-ssm-dtype bfloat16.",
                    mamba_cap,
                    get_schedule().max_mamba_cache_size,
                    ratio,
                )
            max_num_reqs = min(max_num_reqs, mamba_cap)

            if max_num_reqs <= 0:
                raise RuntimeError(
                    f"Hybrid (mamba/linear-attention) state cache is too small to serve "
                    f"any requests. max_mamba_cache_size={get_schedule().max_mamba_cache_size}, "
                    f"mamba_ratio={ratio}, resulting max_num_reqs={max_num_reqs}. "
                    f"Try: (1) reduce --max-running-requests, "
                    f"(2) increase --mem-fraction-static, or "
                    f"(3) use GPUs with more memory."
                )
        if (
            requested_per_worker is not None
            and max_num_reqs < requested_per_worker
            and not capped_by_mamba
        ):
            logger.warning(
                "max_running_requests was reduced from the requested %d to %d "
                "(per dp worker) due to the available KV cache capacity.",
                requested_per_worker,
                max_num_reqs,
            )
        return max_num_reqs

    def _resolve_memory_pool_config(
        self, pre_model_load_memory: int
    ) -> MemoryPoolConfig:

        from sglang.srt.model_executor.pool_configurator import (
            create_memory_pool_configurator,
        )

        available_bytes = self._profile_available_bytes(pre_model_load_memory)
        config = self.config_from_budget(available_bytes)
        config.max_running_requests = self.resolve_max_num_reqs(
            config.max_total_num_tokens
        )
        configurator = create_memory_pool_configurator(self)
        config = configurator.finalize_with_max_running_requests(config)
        config.mem_fraction_static = get_schedule().mem_fraction_static
        return config

    def config_from_budget(
        self, budget_bytes: int, *, cap_tokens: Optional[int] = None
    ) -> MemoryPoolConfig:


        from sglang.srt.model_executor.pool_configurator import (
            create_memory_pool_configurator,
        )

        configurator = create_memory_pool_configurator(self)
        config = configurator.calculate_pool_sizes(
            budget_bytes, get_schedule().page_size
        )
        max_tokens = self._apply_token_constraints(config.max_total_num_tokens)
        if cap_tokens is not None:
            max_tokens = min(max_tokens, cap_tokens)
        if max_tokens != config.max_total_num_tokens:
            config = configurator.calculate_pool_sizes_from_max_tokens(
                max_tokens, get_schedule().page_size
            )
        return config

    def _request_forest_ssm_row_capacity(self, max_num_reqs: int) -> Optional[int]:

        if (
            not self.spec_algorithm.is_request_forest()
            or self.is_draft_worker
            or self.hybrid_gdn_config is None
        ):
            return None
        budget = getattr(
            self.server_args, "speculative_request_forest_max_verify_nodes", None
        )
        width = max_speculative_num_draft_tokens()
        if (
            budget is None
            or width is None
            or get_exec().mamba.enable_linear_replayssm_spec
            or get_memory().enable_unified_memory
        ):
            return None

        from sglang.srt.speculative.ragged_verify import ragged_verify_compact_enabled
        from sglang.srt.speculative.request_forest.adaptive_config import (
            adaptive_native_max_batch_size,
            adaptive_rf_enabled,
        )


        adaptive_w9 = width == 9 and adaptive_rf_enabled()
        if (
            (width not in (8, 16) and not adaptive_w9)
            or get_exec().mamba.linear_attn_verify_backend != "triton"
            or not ragged_verify_compact_enabled()
        ):
            return None
        slots = get_cuda_graph_max_batch_size(max_num_reqs)

        rows = slots + min(slots * (width - 1), int(budget))
        rows = ((rows + width - 1) // width) * width
        if adaptive_rf_enabled():


            native_rows = adaptive_native_max_batch_size(max_num_reqs) * width
            rows = max(rows, native_rows)
        return rows

    def _handle_max_mamba_cache(self, total_rest_memory):
        config = self.mambaish_config
        assert config is not None


        all_mamba_layers = config.mamba2_cache_params.layers
        if self.ps.pp_size > 1 and all_mamba_layers:
            max_stage_mamba_layers = max(
                sum(1 for i in all_mamba_layers if start <= i < end)
                for start, end in (
                    get_pp_indices(
                        self.model_config.num_hidden_layers, rank, self.ps.pp_size
                    )
                    for rank in range(self.ps.pp_size)
                )
            )
        else:
            max_stage_mamba_layers = len(all_mamba_layers)
        pp_layer_scale = max_stage_mamba_layers / max(len(all_mamba_layers), 1)
        stage_per_req = int(
            config.mamba2_cache_params.mamba_cache_per_req * pp_layer_scale
        )

        def intermediate_bytes(capped_reqs: int) -> int:
            packed_rows = self._request_forest_ssm_row_capacity(capped_reqs)
            if packed_rows is None:
                return (
                    stage_per_req
                    * (capped_reqs + 1)
                    * get_spec().speculative_num_draft_tokens
                )

            state_params = config.mamba2_cache_params
            stage_ssm_per_row = (
                math.prod(state_params.shape.temporal)
                * state_params.dtype.temporal.itemsize
                * max_stage_mamba_layers
            )
            return stage_ssm_per_row * packed_rows + (
                (stage_per_req - stage_ssm_per_row)
                * (capped_reqs + 1)
                * max_speculative_num_draft_tokens()
            )

        has_spec_dec = not self.spec_algorithm.is_none()


        replayssm_active = get_exec().mamba.enable_linear_replayssm_spec and (
            self.hybrid_gdn_config is not None
            or kimi_linear_config(self.model_config) is not None
        )
        if replayssm_active:


            max_draft_tokens = max_speculative_num_draft_tokens()
            if kimi_linear_config(self.model_config) is not None:
                record_len = get_exec().mamba.linear_replayssm_cache_len
            elif max_draft_tokens is not None:
                record_len = max_draft_tokens
            else:
                record_len = get_exec().mamba.linear_replayssm_cache_len
            replayssm_ring_per_req = (
                config.mamba2_cache_params.replayssm_ring_bytes_per_req(
                    record_len=record_len
                )
            )
        else:
            replayssm_ring_per_req = 0
        replayssm_ring_per_req = int(replayssm_ring_per_req * pp_layer_scale)
        if has_spec_dec:
            assert get_spec().speculative_num_draft_tokens is not None
            assert get_schedule().max_running_requests is not None

        if get_schedule().max_mamba_cache_size is not None:

            get_context().override(
                "mamba_pool.per_dp_shard",
                max_mamba_cache_size=get_schedule().max_mamba_cache_size
                // self.ps.attn_dp_size,
            )


            if has_spec_dec and not replayssm_active:
                ratio = self._calculate_mamba_ratio()
                capped_reqs = min(
                    get_schedule().max_running_requests // self.ps.attn_dp_size,
                    get_schedule().max_mamba_cache_size // ratio,
                )
                intermediate_size = intermediate_bytes(capped_reqs)
                total_rest_memory = total_rest_memory - (intermediate_size / (1 << 30))
        elif (
            get_memory().disable_radix_cache
            and get_schedule().max_running_requests is not None
        ):

            get_context().override(
                "mamba_pool.from_max_running_requests",
                max_mamba_cache_size=get_schedule().max_running_requests
                // self.ps.attn_dp_size,
            )


            if has_spec_dec and not replayssm_active:
                intermediate_size = intermediate_bytes(
                    get_schedule().max_mamba_cache_size
                )
                total_rest_memory = total_rest_memory - (intermediate_size / (1 << 30))
        else:

            assert stage_per_req > 0
            per_req = stage_per_req


            mamba_budget = (
                total_rest_memory
                * get_schedule().mamba_full_memory_ratio
                / (1 + get_schedule().mamba_full_memory_ratio)
            )
            mamba_budget_bytes = mamba_budget * (1 << 30)

            if has_spec_dec and not replayssm_active:
                ratio = self._calculate_mamba_ratio()
                D = get_spec().speculative_num_draft_tokens
                requested_reqs = (
                    get_schedule().max_running_requests // self.ps.attn_dp_size
                )
                if self._request_forest_ssm_row_capacity(requested_reqs) is not None:


                    low, high = 0, max(0, int(mamba_budget_bytes // per_req))
                    while low < high:
                        candidate = (low + high + 1) // 2
                        reqs = min(requested_reqs, candidate // ratio)
                        needed = (candidate + 1) * per_req + intermediate_bytes(reqs)
                        if needed <= mamba_budget_bytes:
                            low = candidate
                        else:
                            high = candidate - 1
                    max_mamba_cache_size = low
                else:
                    max_mamba_cache_size = int(
                        (mamba_budget_bytes - per_req * (1 + D))
                        // (per_req * (1 + D / ratio))
                    )
                get_context().override(
                    "mamba_pool.memory_budget_spec",
                    max_mamba_cache_size=max_mamba_cache_size,
                )


                capped_reqs = min(
                    get_schedule().max_running_requests // self.ps.attn_dp_size,
                    get_schedule().max_mamba_cache_size // ratio,
                )
                intermediate_size = intermediate_bytes(capped_reqs)
                total_rest_memory = total_rest_memory - (intermediate_size / (1 << 30))
            else:
                per_slot = per_req + replayssm_ring_per_req
                get_context().override(
                    "mamba_pool.memory_budget",
                    max_mamba_cache_size=int(
                        (mamba_budget_bytes - per_slot) // per_slot
                    ),
                )


        if get_schedule().max_mamba_cache_size <= 0:
            raise RuntimeError(
                f"Not enough GPU memory for hybrid (mamba/linear-attention) state cache. "
                f"Computed max_mamba_cache_size={get_schedule().max_mamba_cache_size} "
                f"(total_rest_memory={total_rest_memory:.2f} GB, "
                f"mamba_cache_per_req={stage_per_req / (1 << 20):.2f} MB). "
                f"Try: (1) reduce --max-running-requests, "
                f"(2) increase --mem-fraction-static, "
                f"(3) reduce --speculative-num-draft-tokens, or "
                f"(4) use GPUs with more memory."
            )


        mamba_state_memory = (
            (get_schedule().max_mamba_cache_size + 1)
            * (stage_per_req + replayssm_ring_per_req)
            / (1 << 30)
        )
        return total_rest_memory - mamba_state_memory


def calculate_mla_kv_cache_dim(
    *, model_config: ModelConfig, kv_cache_dtype: torch.dtype
) -> int:
    is_dsa_model = is_deepseek_dsa(model_config.hf_config)
    kv_cache_dtype = kv_cache_dtype
    kv_lora_rank = model_config.kv_lora_rank
    qk_rope_head_dim = model_config.qk_rope_head_dim
    kv_cache_dim = kv_lora_rank + qk_rope_head_dim


    if not is_dsa_model:
        return kv_cache_dim


    if (
        get_exec().kernel.dsa_prefill_backend == "trtllm"
        or get_exec().kernel.dsa_decode_backend == "trtllm"
    ):
        return kv_cache_dim


    if _is_hip and (
        get_exec().kernel.dsa_prefill_backend in ("tilelang", "aiter")
        or get_exec().kernel.dsa_decode_backend in ("tilelang", "aiter")
    ):
        return kv_cache_dim

    quant_block_size = DSATokenToKVPool.quant_block_size
    rope_storage_dtype = DSATokenToKVPool.rope_storage_dtype


    if kv_cache_dtype == torch.float8_e4m3fn:
        assert (
            kv_lora_rank % quant_block_size == 0
        ), f"kv_lora_rank {kv_lora_rank} must be multiple of quant_block_size {quant_block_size}"

        return (
            kv_lora_rank
            + kv_lora_rank // quant_block_size * 4
            + qk_rope_head_dim * rope_storage_dtype.itemsize
        )

    return kv_cache_dim
