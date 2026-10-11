import logging
import time
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np
import psutil
from mlx.utils import tree_flatten
from mlx_lm import load as mlx_lm_load
from mlx_lm.utils import quantize_model as mlx_lm_quantize_model

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.mlx.aot import (
    MLX_AOT_KERNEL_REGISTRY,
    MlxAOTKernelSet,
)
from sglang.srt.hardware_backend.mlx.kv_cache import (
    AttentionOffsetCache,
    BatchedDecodeContext,
    ContiguousAttentionKVCache,
    MlxAttentionKVPool,
    MLXAttentionWrapper,
    MlxModelCacheLayout,
    PoolBackedAttentionKVCache,
    WindowedAttentionKVCache,
    clear_context,
    find_attention_layers,
    get_head_dim,
    get_layer_window_sizes,
    get_num_kv_heads,
    patch_model_attention,
    set_context,
    uses_sliding_window_attention,
)
from sglang.srt.hardware_backend.mlx.remote_code_gate import (
    ensure_remote_code_allowed,
    resolve_model_directory,
)
from sglang.srt.hardware_backend.mlx.sampling import (
    GREEDY_PARAMS,
    MlxLazyLogprobs,
    MlxLogprobSpec,
    MlxSamplingParams,
    MlxStepLogprobs,
    all_greedy,
    compute_logprobs,
    lazy_logprob_arrays,
    sample_tokens,
    sanitize_logits,
    scale_by_temperature,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.runtime_context import (
    mamba_cache_chunk_size,
)

logger = logging.getLogger(__name__)


@dataclass
class MlxPendingPrefill:


    lazy_token: mx.array
    cache: list[Any]
    req_id: str
    full_token_ids: list[int]
    req_pool_idx: int
    synced_offset: int
    lazy_logprobs: MlxLazyLogprobs | None = None


@dataclass
class MlxPendingExtend:


    lazy_token: mx.array
    cache: list[Any]
    req_id: str
    new_token_ids: list[int]
    new_synced_offset: int
    lazy_logprobs: MlxLazyLogprobs | None = None


@dataclass
class MlxPendingDecode:


    lazy_tokens: mx.array
    req_ids: list[str]
    caches: list[list[Any]]
    lazy_logprobs: MlxLazyLogprobs | None = None

    logprob_spec: MlxLogprobSpec | None = None


    edit_rows: mx.array | None = None


_MLX_QUANTIZATION_PRESETS: dict[str, tuple[int, int]] = {

    "mlx_q4": (4, 64),
    "mlx_q8": (8, 64),
}
_MLX_KV_FLOAT_DTYPES = {mx.float16, mx.bfloat16, mx.float32}


class MlxModelRunner:


    _trunk = None
    _enable_sampling = False
    _sanitize_nan = False
    _deterministic_seeding = False

    def __init__(
        self,
        model_path: str,
        trust_remote_code: bool = False,
        disable_radix_cache: bool = False,
        pool_size: int | None = None,
        mem_fraction_static: float = 0.8,
        quantization: str | None = None,
        revision: str | None = None,
        enable_sampling: bool = False,
        sampling_rng_seed: int = 0,
        deterministic_seeding: bool = False,
    ):
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.revision = revision
        self.model = None
        self.disable_radix_cache = disable_radix_cache
        self._mem_fraction_static = mem_fraction_static
        self._enable_sampling = enable_sampling


        self._deterministic_seeding = deterministic_seeding
        self._sanitize_nan = envs.SGLANG_SANITIZE_NAN_LOGITS.get()


        self._rng_key: mx.array | None = (
            mx.random.key(sampling_rng_seed) if enable_sampling else None
        )

        self._decode_step_ct: int = 0
        self._clear_steps = envs.SGLANG_MLX_CLEAR_CACHE_STEPS.get()


        self._quantization: str | None = quantization


        cache_limit_gb = envs.SGLANG_MLX_CACHE_LIMIT_GB.get()
        if cache_limit_gb is not None:
            if cache_limit_gb < 0:
                raise ValueError(
                    f"SGLANG_MLX_CACHE_LIMIT_GB must be >= 0, got {cache_limit_gb}"
                )
            mx.set_cache_limit(int(cache_limit_gb * (1024**3)))
            logger.info(f"MLX buffer cache limit set to {cache_limit_gb:.1f} GB")

        self._load_model()


        device_info = mx.device_info()
        max_wired = int(device_info.get("max_recommended_working_set_size", 0))
        if max_wired > 0:
            mx.set_wired_limit(max_wired)
            logger.info(f"Wired memory limit set to {max_wired / (1024**3):.1f} GB")

        patch_model_attention(self.model)

        layer_list, attn_attrs = find_attention_layers(self.model)
        self._cache_layout = MlxModelCacheLayout.from_attention_discovery(
            layer_list,
            attn_attrs,

            layer_window_sizes=get_layer_window_sizes(self.model),
        )
        if self._cache_layout.num_attention_layers == 0:
            raise RuntimeError("MLX model has no supported attention layers")
        if self._cache_layout.has_auxiliary_state and not hasattr(
            self.model, "make_cache"
        ):
            raise RuntimeError(
                "MLX models with auxiliary cache state require model.make_cache()."
            )
        if (
            self._cache_layout.has_auxiliary_state
            and self._cache_layout.has_sliding_window_layers
        ):


            raise NotImplementedError(
                "MLX runner does not support models with both auxiliary "
                "cache state and sliding-window attention layers."
            )
        if self._cache_layout.has_auxiliary_state:
            self._model_embed, self._model_norm, self._model_lm_head = (
                self._extract_model_components()
            )
        self._max_seq_len = 4096

        self._req_caches: dict[str, list[Any]] = {}
        self._req_token_ids: dict[str, list[int]] = {}
        self._req_sampling: dict[str, MlxSamplingParams] = {}

        self._cache_pool: list[list[Any]] = []

        self._attention_kv_pool: MlxAttentionKVPool | None = None
        self._req_to_token_pool: ReqToTokenPool | None = None
        self._req_pool_idx: dict[str, int] = {}
        self._req_synced_offset: dict[str, int] = {}

        self._pool_size = self._compute_pool_size(pool_size)
        self._aot_kernels = self._build_aot_kernels()

    @staticmethod
    def _extract_logits(model_output):

        if isinstance(model_output, tuple):
            return model_output[0]
        return model_output

    def _new_cache_skeleton(self) -> list[Any]:

        if self._cache_layout.has_auxiliary_state:
            cache = self.model.make_cache()
            if len(cache) != self._cache_layout.num_layers:
                raise RuntimeError(
                    "model.make_cache() returned "
                    f"{len(cache)} entries for {self._cache_layout.num_layers} layers"
                )
        else:
            cache = [None] * self._cache_layout.num_layers
        return cache

    def _new_native_cache(self) -> list[Any]:

        cache = self._new_cache_skeleton()
        for layer_idx in self._cache_layout.attention_layer_indices:
            window = self._cache_layout.window_size(layer_idx)
            cache[layer_idx] = (
                WindowedAttentionKVCache(window)
                if window is not None
                else ContiguousAttentionKVCache(max_seq_len=self._max_seq_len)
            )
        return cache

    def _acquire_cache(self) -> list[Any]:

        if not self._cache_layout.has_auxiliary_state and self._cache_pool:
            cache = self._cache_pool.pop()
            for c in cache:
                c.reset()
            return cache
        return self._new_native_cache()

    def _release_cache(self, cache: list[Any]) -> None:

        if not self._cache_layout.has_auxiliary_state:
            self._cache_pool.append(cache)

    def _first_attention_cache(self, cache: list[Any]) -> Any:
        return cache[self._cache_layout.first_attention_layer_index]

    def _get_auxiliary_state_pool_index(self, req_pool_idx: int) -> Any | None:
        if (
            not self._cache_layout.has_auxiliary_state
            or self._req_to_token_pool is None
            or not hasattr(self._req_to_token_pool, "get_auxiliary_state_indices")
        ):
            return None
        return self._req_to_token_pool.get_auxiliary_state_indices(req_pool_idx)

    def _get_auxiliary_state_pool(self) -> Any | None:
        return getattr(self._req_to_token_pool, "auxiliary_state_pool", None)

    def _restore_auxiliary_state(self, req_pool_idx: int, cache: list[Any]) -> bool:
        pool_index = self._get_auxiliary_state_pool_index(req_pool_idx)
        pool = self._get_auxiliary_state_pool()
        if pool_index is None or not hasattr(pool, "restore_cache"):
            return False
        return pool.restore_cache(
            pool_index,
            cache,
            self._cache_layout.auxiliary_layer_indices,
        )

    def _store_auxiliary_state(self, req_pool_idx: int, cache: list[Any]) -> None:
        pool_index = self._get_auxiliary_state_pool_index(req_pool_idx)
        pool = self._get_auxiliary_state_pool()
        if pool_index is None or not hasattr(pool, "store_cache"):
            return
        pool.store_cache(
            pool_index,
            cache,
            self._cache_layout.auxiliary_layer_indices,
        )

    def store_auxiliary_state_for_request(self, req_id: str) -> None:

        req_pool_idx = self._req_pool_idx.get(req_id)
        cache = self._req_caches.get(req_id)
        if req_pool_idx is None or cache is None:
            return
        self._store_auxiliary_state(req_pool_idx, cache)

    def _select_auxiliary_state_track_len(
        self,
        *,
        prefix_len: int,
        new_token_count: int,
        full_len: int,
        req: Any | None,
    ) -> int | None:
        if (
            not self._cache_layout.has_auxiliary_state
            or req is None
            or new_token_count <= 0
        ):
            return None

        chunk_size = mamba_cache_chunk_size()
        track_len = prefix_len + (new_token_count // chunk_size) * chunk_size
        branching_len = getattr(req, "mamba_branching_seqlen", None)
        if (
            branching_len is not None
            and prefix_len < branching_len <= prefix_len + new_token_count
            and (branching_len - prefix_len) % chunk_size == 0
        ):
            track_len = branching_len

        if track_len <= prefix_len or track_len > full_len:
            return None
        return track_len

    def _store_tracked_auxiliary_state(
        self,
        req: Any | None,
        cache: list[Any],
        track_len: int | None,
    ) -> None:
        if (
            req is None
            or track_len is None
            or not self._cache_layout.has_auxiliary_state
        ):
            return
        pool = self._get_auxiliary_state_pool()
        if pool is None or not hasattr(pool, "store_cache"):
            return

        track_buffer = getattr(req, "mamba_ping_pong_track_buffer", None)
        if track_buffer is None:
            track_buffer = pool.alloc(1)
            if track_buffer is None:
                logger.warning(
                    "MLX auxiliary-state track slot allocation failed; "
                    "falling back to leaf-only auxiliary-state radix caching."
                )
                return
            req.mamba_ping_pong_track_buffer = track_buffer
            req.mamba_next_track_idx = 0
            req.mamba_last_track_idx = 0

        pool.store_cache(
            track_buffer[0],
            cache,
            self._cache_layout.auxiliary_layer_indices,
        )
        req.mamba_last_track_seqlen = track_len

    def _cache_with_pool_backed_attention(
        self, prefix_slot_ids: list[int], prefix_len: int
    ) -> list[Any]:

        assert self._attention_kv_pool is not None
        slot_ids_mx = mx.array(prefix_slot_ids, dtype=mx.int32)
        cache = self._new_cache_skeleton()
        for layer_idx in self._cache_layout.attention_layer_indices:
            cache[layer_idx] = PoolBackedAttentionKVCache(
                self._attention_kv_pool,
                self._cache_layout.full_kv_pool_index(layer_idx),
                slot_ids_mx,
                prefix_len,
            )
        return cache

    def _materialize_pool_backed_attention(self, cache: list[Any]) -> list[Any]:
        contiguous_cache = self._acquire_cache()
        for layer_idx in self._cache_layout.attention_layer_indices:
            pbc = cache[layer_idx]
            contiguous_cache[layer_idx].update_and_fetch(
                pbc._full_keys, pbc._full_values
            )
        for layer_idx in self._cache_layout.auxiliary_layer_indices:
            contiguous_cache[layer_idx] = cache[layer_idx]
        return contiguous_cache

    @staticmethod
    def _cache_arrays(cache: Any) -> list[mx.array]:

        arrays: list[mx.array] = []

        def collect(value: Any) -> None:
            if isinstance(value, mx.array):
                arrays.append(value)
            elif value is None:
                return
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)
            elif isinstance(value, dict):
                for item in value.values():
                    collect(item)

        collect(getattr(cache, "state", ()))
        return arrays

    @staticmethod
    def cache_state_arrays(caches: list[list[Any]]) -> list[mx.array]:

        return [
            s
            for cache_list in caches
            for cache in cache_list
            for s in MlxModelRunner._cache_arrays(cache)
        ]

    def _load_model(self):

        logger.info(f"Loading MLX model: {self.model_path}")
        start_time = time.time()


        model_dir = resolve_model_directory(self.model_path, revision=self.revision)
        ensure_remote_code_allowed(model_dir, self.trust_remote_code)


        loaded = mlx_lm_load(
            str(model_dir),
            tokenizer_config={"trust_remote_code": self.trust_remote_code},
            return_config=True,
        )
        self.model, _tokenizer, config = loaded

        if self._quantization in _MLX_QUANTIZATION_PRESETS:
            bits, group_size = _MLX_QUANTIZATION_PRESETS[self._quantization]


            if "quantization" in (config or {}):
                logger.info(
                    "MLX model is already quantized by the HF repo; "
                    f"ignoring --quantization={self._quantization}"
                )
            else:


                bytes_before = sum(
                    p.size * p.itemsize
                    for _, p in tree_flatten(self.model.parameters())
                )
                q_start = time.time()
                logger.info(
                    f"Quantizing MLX model on-the-fly: bits={bits} "
                    f"group_size={group_size} (preset={self._quantization})"
                )
                self.model, _new_config = mlx_lm_quantize_model(
                    self.model,
                    config or {},
                    group_size=group_size,
                    bits=bits,
                )
                bytes_after = sum(
                    p.size * p.itemsize
                    for _, p in tree_flatten(self.model.parameters())
                )
                q_time = time.time() - q_start
                pct_reduction = (1 - bytes_after / max(bytes_before, 1)) * 100
                logger.info(
                    f"Quantization complete in {q_time:.2f}s — "
                    f"weight bytes: {bytes_before / 1024**3:.2f} GB -> "
                    f"{bytes_after / 1024**3:.2f} GB ({pct_reduction:.1f}% reduction)"
                )


        mx.eval(self.model.parameters())

        load_time = time.time() - start_time
        logger.info(f"MLX model loaded in {load_time:.2f}s")


        trunk = getattr(self.model, "model", None)
        self._trunk = trunk if callable(trunk) else None
        if self._trunk is None:
            logger.info(
                "Model %s exposes no headless trunk (`.model`); non-final "
                "chunked-prefill chunks will compute full vocab logits.",
                type(self.model).__name__,
            )


        if envs.SGLANG_MLX_FUSE_SWIGLU.get():
            from sglang.srt.hardware_backend.mlx.moe.fused_swiglu import (
                patch_switch_glu_with_fused_swiglu,
            )

            n_patched = patch_switch_glu_with_fused_swiglu(self.model)
            logger.info(
                f"MLX SwiGLU activation fusion enabled: patched {n_patched} blocks"
            )

    def _attention_module_for_layer(self, layer_idx: int) -> Any:
        attn = getattr(
            self._cache_layout.layers[layer_idx],
            self._cache_layout.attention_attr(layer_idx),
        )
        if isinstance(attn, MLXAttentionWrapper):
            return attn._inner
        return attn

    def _attention_kv_config_for_layer(
        self, layer_idx: int
    ) -> tuple[int, int, mx.Dtype]:
        layer = self._cache_layout.layers[layer_idx]
        sample_attn = self._attention_module_for_layer(layer_idx)
        unsized_window = self._cache_layout.window_size(layer_idx) is None
        if unsized_window and uses_sliding_window_attention(layer, sample_attn):
            raise NotImplementedError(
                f"Attention layer {layer_idx} declares sliding-window "
                "attention but the model exposes no per-layer window map "
                "(container `layer_types` plus a scalar window), so the MLX "
                "KV cache cannot bound its sliding-window KV."
            )
        n_kv_heads = get_num_kv_heads(sample_attn)
        if n_kv_heads is None:
            raise RuntimeError(
                f"Cannot determine n_kv_heads from attention module at layer {layer_idx}"
            )
        head_dim = get_head_dim(sample_attn)
        if head_dim is None:
            raise RuntimeError(
                f"Cannot determine head_dim from attention module at layer {layer_idx}"
            )
        dtype = mx.float16
        if hasattr(sample_attn, "k_proj") and hasattr(sample_attn.k_proj, "weight"):
            dtype = sample_attn.k_proj.weight.dtype
        if dtype not in _MLX_KV_FLOAT_DTYPES:


            scales = getattr(sample_attn.k_proj, "scales", None)
            if scales is not None and scales.dtype in _MLX_KV_FLOAT_DTYPES:
                dtype = scales.dtype
            else:
                dtype = mx.float32
        return n_kv_heads, head_dim, dtype

    def _get_attn_config(self) -> tuple[int, int, mx.Dtype]:

        if self._cache_layout.num_attention_layers == 0:
            raise RuntimeError(
                "Cannot determine attention config: no attention module found"
            )
        first_layer_idx = self._cache_layout.first_attention_layer_index
        first_config = self._attention_kv_config_for_layer(first_layer_idx)
        for layer_idx in self._cache_layout.attention_layer_indices[1:]:
            config = self._attention_kv_config_for_layer(layer_idx)
            if config != first_config:
                raise NotImplementedError(
                    "MLX attention KV caching requires uniform softmax-attention "
                    "KV shape across layers. "
                    f"Layer {first_layer_idx} has {first_config}, "
                    f"but layer {layer_idx} has {config}. "
                    "Heterogeneous attention KV needs per-layer pools."
                )
        return first_config

    def _compute_pool_size(self, explicit_size: int | None) -> int:

        if explicit_size is not None:
            return explicit_size
        n_kv_heads, head_dim, dtype = self._get_attn_config()


        num_layers = (
            self._cache_layout.num_full_attention_layers
            or self._cache_layout.num_attention_layers
        )
        sys_available = psutil.virtual_memory().available
        mlx_limit = mx.device_info().get(
            "max_recommended_working_set_size",
            mx.device_info().get("memory_size", 0),
        )
        mlx_used = mx.get_active_memory()
        mlx_usable = int(mlx_limit * self._mem_fraction_static)
        kv_budget = min(
            max(mlx_usable - mlx_used, 0),
            int(sys_available * self._mem_fraction_static),
        )
        bytes_per_slot = 2 * num_layers * n_kv_heads * head_dim * dtype.size
        pool_size = max(kv_budget // bytes_per_slot, 256)
        logger.info(
            f"Auto-sized attention KV pool: "
            f"sys_available={sys_available / (1024**3):.2f} GB, "
            f"mlx_limit={mlx_limit / (1024**3):.1f} GB, "
            f"mlx_used={mlx_used / (1024**3):.2f} GB, "
            f"kv_budget={kv_budget / (1024**3):.2f} GB, "
            f"bytes_per_slot={bytes_per_slot}, pool_size={pool_size}"
        )
        return pool_size

    @property
    def pool_size(self) -> int:
        return self._pool_size

    def _build_aot_kernels(self) -> MlxAOTKernelSet:

        if self._cache_layout.num_attention_layers == 0:
            return MlxAOTKernelSet()
        layer_idx = self._cache_layout.first_attention_layer_index
        sample_attn = getattr(
            self._cache_layout.layers[layer_idx],
            self._cache_layout.attention_attr(layer_idx),
        )
        n_kv_heads, head_dim, _ = self._get_attn_config()
        return MLX_AOT_KERNEL_REGISTRY.build_kernel_set(
            sample_attn=sample_attn,
            n_kv_heads=int(n_kv_heads),
            head_dim=int(head_dim),
        )

    def init_cache_pools(self, req_to_token_pool: ReqToTokenPool | None) -> None:

        self._req_to_token_pool = req_to_token_pool
        if self.disable_radix_cache:
            return
        num_pool_layers = self._cache_layout.num_full_attention_layers
        if self._cache_layout.has_sliding_window_layers:


            logger.info(
                "Model has %d sliding-window attention layers; skipping the "
                "shared attention KV pool (an SWA prefix hit recomputes the "
                "prefix, so the pool would never be read). Per-request "
                "windowed caches only.",
                len(self._cache_layout.swa_attention_layer_indices),
            )
            return
        if num_pool_layers == 0:
            return
        n_kv_heads, head_dim, dtype = self._get_attn_config()

        self._attention_kv_pool = MlxAttentionKVPool(
            pool_size=self._pool_size + 1,
            num_layers=num_pool_layers,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        )
        logger.info(
            f"Attention KV pool initialized: pool_size={self._pool_size} "
            f"(buffer size {self._pool_size + 1} incl. padding slot 0), "
            f"{num_pool_layers} full-attention layers "
            f"({len(self._cache_layout.swa_attention_layer_indices)} "
            "sliding-window layers stay per-request), "
            f"{n_kv_heads} kv_heads, {head_dim} head_dim"
        )

    def prefill(
        self,
        req_id: str,
        new_token_ids: list[int],
        full_token_ids: list[int],
        prefix_slot_ids: list[int],
        new_slot_ids: list[int],
        req_pool_idx: int,
        req: Any | None = None,
        needs_logits: bool = True,
    ) -> int:

        pending = self.prefill_start(
            req_id=req_id,
            new_token_ids=new_token_ids,
            full_token_ids=full_token_ids,
            prefix_slot_ids=prefix_slot_ids,
            new_slot_ids=new_slot_ids,
            req_pool_idx=req_pool_idx,
            req=req,
            needs_logits=needs_logits,
        )
        self.eval_pending(pending)
        return self.prefill_finalize(pending)

    def extend(
        self,
        req_id: str,
        new_token_ids: list[int],
        new_slot_ids: list[int],
        needs_logits: bool = True,
    ) -> int:

        pending = self.extend_start(
            req_id, new_token_ids, new_slot_ids, needs_logits=needs_logits
        )
        self.eval_pending(pending)
        return self.extend_finalize(pending)

    def _sync_new_kv_to_pool(
        self,
        cache: list[Any],
        cache_start: int,
        slot_ids: list[int],
    ) -> None:

        if not slot_ids or self._attention_kv_pool is None:
            return
        full_layer_indices = self._cache_layout.full_attention_layer_indices
        end = cache_start + len(slot_ids)
        slot_ids_mx = mx.array(slot_ids, dtype=mx.int32)


        k_all = mx.stack(
            [
                cache[layer_idx].keys[0, :, cache_start:end, :].transpose(1, 0, 2)
                for layer_idx in full_layer_indices
            ]
        )
        v_all = mx.stack(
            [
                cache[layer_idx].values[0, :, cache_start:end, :].transpose(1, 0, 2)
                for layer_idx in full_layer_indices
            ]
        )
        self._attention_kv_pool.set_kv_all_layers(slot_ids_mx, k_all, v_all)

    def _sync_decode_kv_to_pool(self, req_id: str) -> None:

        if self._attention_kv_pool is None or self._req_to_token_pool is None:
            return
        cache = self._req_caches.get(req_id)
        if cache is None:
            return
        current_offset = self._first_attention_cache(cache).offset
        synced_offset = self._req_synced_offset.get(req_id, 0)
        if current_offset <= synced_offset:
            return
        req_pool_idx = self._req_pool_idx.get(req_id)
        if req_pool_idx is None:
            return

        slot_ids = (
            self._req_to_token_pool.req_to_token[
                req_pool_idx, synced_offset:current_offset
            ]
            .to(dtype=int)
            .tolist()
        )
        self._sync_new_kv_to_pool(cache, synced_offset, slot_ids)
        self._req_synced_offset[req_id] = current_offset

    def flush_all_decode_kv(self) -> None:

        if self.disable_radix_cache or self._attention_kv_pool is None:
            return
        for req_id in list(self._req_caches.keys()):
            self._sync_decode_kv_to_pool(req_id)

    def decode_batch(self, req_ids: list[str]) -> list[int]:

        pending = self.decode_batch_start(req_ids)
        self.eval_pending(pending)
        return self.decode_batch_finalize(pending)

    def prefill_start(
        self,
        req_id: str,
        new_token_ids: list[int],
        full_token_ids: list[int],
        prefix_slot_ids: list[int],
        new_slot_ids: list[int],
        req_pool_idx: int,
        req: Any | None = None,
        needs_logits: bool = True,
        logit_edit_row: mx.array | None = None,
        logprob_spec: MlxLogprobSpec | None = None,
    ) -> MlxPendingPrefill:

        prefix_len = len(prefix_slot_ids)
        if req is not None:
            req.mamba_last_track_seqlen = None
        if self._enable_sampling:
            self._req_sampling[req_id] = (
                MlxSamplingParams.from_req(
                    req, deterministic_seeding=self._deterministic_seeding
                )
                if req is not None
                else GREEDY_PARAMS
            )

        if self.disable_radix_cache:
            cache = self._acquire_cache()
            input_ids = mx.array([new_token_ids], dtype=mx.int32)
            lazy_token, lazy_logprobs = self._forward_lazy_token(
                input_ids, cache, needs_logits, req_id, logit_edit_row, logprob_spec
            )
            return MlxPendingPrefill(
                lazy_token=lazy_token,
                cache=cache,
                req_id=req_id,
                full_token_ids=list(full_token_ids),
                req_pool_idx=req_pool_idx,
                synced_offset=0,
                lazy_logprobs=lazy_logprobs,
            )


        assert (
            self._attention_kv_pool is not None
            or self._cache_layout.num_full_attention_layers == 0
            or self._cache_layout.has_sliding_window_layers
        )

        new_token_count = len(new_token_ids)
        track_len = self._select_auxiliary_state_track_len(
            prefix_len=prefix_len,
            new_token_count=new_token_count,
            full_len=len(full_token_ids),
            req=req,
        )


        if prefix_len > 0 and self._cache_layout.has_sliding_window_layers:
            trusted_prefix_len = 0
            run_token_ids = list(full_token_ids[:prefix_len]) + new_token_ids
        else:
            trusted_prefix_len = prefix_len
            run_token_ids = new_token_ids

        if trusted_prefix_len > 0:
            cache = self._cache_with_pool_backed_attention(
                prefix_slot_ids[:trusted_prefix_len], trusted_prefix_len
            )
            pool_backed_attention = True
            restored_auxiliary_state = (
                not self._cache_layout.has_auxiliary_state
                or self._restore_auxiliary_state(req_pool_idx, cache)
            )
            if self._cache_layout.has_auxiliary_state and (
                not restored_auxiliary_state or new_token_count == 0
            ):


                cache = self._acquire_cache()
                input_ids = mx.array([full_token_ids or new_token_ids], dtype=mx.int32)
                lazy_token, lazy_logprobs = self._forward_lazy_token(
                    input_ids, cache, needs_logits, req_id, logit_edit_row, logprob_spec
                )
                if new_slot_ids:
                    self._sync_new_kv_to_pool(cache, prefix_len, new_slot_ids)
                return MlxPendingPrefill(
                    lazy_token=lazy_token,
                    cache=cache,
                    req_id=req_id,
                    full_token_ids=list(full_token_ids),
                    req_pool_idx=req_pool_idx,
                    synced_offset=prefix_len + len(new_slot_ids),
                    lazy_logprobs=lazy_logprobs,
                )
        else:
            cache = self._acquire_cache()
            pool_backed_attention = False

        if run_token_ids:
            track_new_count = track_len - prefix_len if track_len is not None else None
            if track_new_count is not None and 0 < track_new_count < new_token_count:


                input_ids = mx.array([new_token_ids[:track_new_count]], dtype=mx.int32)


                if self._trunk_forward(input_ids, cache) is None:
                    self.model(input_ids, cache=cache)
                self._store_tracked_auxiliary_state(req, cache, track_len)
                if pool_backed_attention:
                    cache = self._materialize_pool_backed_attention(cache)
                    pool_backed_attention = False
                extend_tokens = new_token_ids[track_new_count:]
            else:
                extend_tokens = run_token_ids
        else:


            extend_tokens = full_token_ids[-1:]
            for c in cache:
                c.offset = max(c.offset - 1, 0)

        input_ids = mx.array([extend_tokens], dtype=mx.int32)
        lazy_token, lazy_logprobs = self._forward_lazy_token(
            input_ids, cache, needs_logits, req_id, logit_edit_row, logprob_spec
        )

        if track_len is not None and track_len == prefix_len + new_token_count:
            self._store_tracked_auxiliary_state(req, cache, track_len)


        if pool_backed_attention:
            cache = self._materialize_pool_backed_attention(cache)

        if new_slot_ids:
            self._sync_new_kv_to_pool(cache, prefix_len, new_slot_ids)

        return MlxPendingPrefill(
            lazy_token=lazy_token,
            cache=cache,
            req_id=req_id,
            full_token_ids=list(full_token_ids),
            req_pool_idx=req_pool_idx,
            synced_offset=prefix_len + len(new_slot_ids),
            lazy_logprobs=lazy_logprobs,
        )

    def prefill_finalize(self, pending: MlxPendingPrefill) -> int:

        next_token = int(pending.lazy_token.item())
        self._req_token_ids[pending.req_id] = list(pending.full_token_ids) + [
            next_token
        ]
        self._req_caches[pending.req_id] = pending.cache
        self._req_pool_idx[pending.req_id] = pending.req_pool_idx
        self._req_synced_offset[pending.req_id] = pending.synced_offset
        self._store_auxiliary_state(pending.req_pool_idx, pending.cache)
        return next_token

    def extend_start(
        self,
        req_id: str,
        new_token_ids: list[int],
        new_slot_ids: list[int],
        needs_logits: bool = True,
        logit_edit_row: mx.array | None = None,
        logprob_spec: MlxLogprobSpec | None = None,
    ) -> MlxPendingExtend:

        assert (
            req_id in self._req_caches
        ), f"extend_start called for unknown request {req_id}"

        cache = self._req_caches[req_id]

        input_ids = mx.array([new_token_ids], dtype=mx.int32)
        lazy_token, lazy_logprobs = self._forward_lazy_token(
            input_ids, cache, needs_logits, req_id, logit_edit_row, logprob_spec
        )

        if not self.disable_radix_cache and new_slot_ids:
            synced = self._req_synced_offset[req_id]
            self._sync_new_kv_to_pool(cache, synced, new_slot_ids)
            new_synced_offset = synced + len(new_slot_ids)
        else:
            new_synced_offset = self._req_synced_offset.get(req_id, 0)

        return MlxPendingExtend(
            lazy_token=lazy_token,
            cache=cache,
            req_id=req_id,
            new_token_ids=list(new_token_ids),
            new_synced_offset=new_synced_offset,
            lazy_logprobs=lazy_logprobs,
        )

    def extend_finalize(self, pending: MlxPendingExtend) -> int:

        next_token = int(pending.lazy_token.item())

        prev_tokens = self._req_token_ids[pending.req_id]
        if prev_tokens:
            prev_tokens.pop()
        prev_tokens.extend(pending.new_token_ids)
        prev_tokens.append(next_token)

        self._req_synced_offset[pending.req_id] = pending.new_synced_offset
        self._store_auxiliary_state(
            self._req_pool_idx[pending.req_id],
            self._req_caches[pending.req_id],
        )
        return next_token

    def _trunk_forward(self, input_ids: mx.array, cache: list[Any]) -> mx.array | None:

        if self._trunk is None:
            return None
        return self._trunk(input_ids, cache=cache)

    def _forward_lazy_token(
        self,
        input_ids: mx.array,
        cache: list[Any],
        needs_logits: bool,
        req_id: str,
        logit_edit_row: mx.array | None = None,
        logprob_spec: MlxLogprobSpec | None = None,
    ) -> tuple[mx.array, MlxLazyLogprobs | None]:

        if not needs_logits:
            hidden = self._trunk_forward(input_ids, cache)
            if hidden is not None:
                return self._dummy_next_token(hidden), None


            model_output = self.model(input_ids, cache=cache)
            logits = self._extract_logits(model_output)
            return mx.argmax(logits[:, -1, :], axis=-1), None
        model_output = self.model(input_ids, cache=cache)
        logits = self._extract_logits(model_output)
        edits = logit_edit_row[None, :] if logit_edit_row is not None else None
        return self._select_tokens_with_logprobs(
            logits[:, -1, :], [req_id], [cache], edits, logprob_spec
        )

    def _select_tokens_with_logprobs(
        self,
        last_logits: mx.array,
        req_ids: list[str],
        caches: list[list[Any]],
        edit_rows: mx.array | None = None,
        logprob_spec: MlxLogprobSpec | None = None,
    ) -> tuple[mx.array, MlxLazyLogprobs | None]:

        if not self._enable_sampling:
            return mx.argmax(last_logits, axis=-1), None
        params = [self._req_sampling[rid] for rid in req_ids]
        edited = self._edited_logits(last_logits, edit_rows)
        greedy = all_greedy(params)

        scaled = (
            scale_by_temperature(edited, params)
            if not greedy or logprob_spec is not None
            else None
        )
        if greedy:
            tokens = mx.argmax(edited, axis=-1)
        else:
            positions = [self._first_attention_cache(c).offset - 1 for c in caches]
            self._rng_key, key = mx.random.split(self._rng_key)
            tokens = sample_tokens(
                last_logits=edited,
                params=params,
                positions=positions,
                key=key,
                scaled=scaled,
            )
        lazy_logprobs = (
            compute_logprobs(
                last_logits=edited,
                params=params,
                tokens=tokens,
                spec=logprob_spec,
                scaled=scaled,
            )
            if logprob_spec is not None
            else None
        )
        return tokens, lazy_logprobs

    def _edited_logits(
        self, last_logits: mx.array, edit_rows: mx.array | None
    ) -> mx.array:

        edited = last_logits
        if edit_rows is not None:


            if edit_rows.shape[-1] != last_logits.shape[-1]:
                raise RuntimeError(
                    "Logit edit rows do not match the model's vocabulary: "
                    f"sampling_info.vocab_size={edit_rows.shape[-1]} vs "
                    f"lm_head width {last_logits.shape[-1]}"
                )
            edited = edited.astype(mx.float32) + edit_rows
        if self._sanitize_nan:
            edited = sanitize_logits(edited.astype(mx.float32))
        return edited

    def _run_logits_hook(self, last_logits: mx.array, logits_hook) -> mx.array:

        logits32 = last_logits.astype(mx.float32)
        mx.eval(logits32)
        return mx.array(logits_hook(np.array(logits32)))

    def collect_logprobs(
        self, lazy_logprobs: MlxLazyLogprobs | None
    ) -> MlxStepLogprobs | None:

        if lazy_logprobs is None:
            return None
        mx.eval(*lazy_logprob_arrays(lazy_logprobs))
        spec = lazy_logprobs.spec
        if lazy_logprobs.top_val is not None:
            all_val = lazy_logprobs.top_val.tolist()
            all_idx = lazy_logprobs.top_idx.tolist()
            top_val = [all_val[i][:k] for i, k in enumerate(spec.top_ks)]
            top_idx = [all_idx[i][:k] for i, k in enumerate(spec.top_ks)]
        else:
            top_val = [[] for _ in spec.top_ks]
            top_idx = [[] for _ in spec.top_ks]
        return MlxStepLogprobs(
            chosen=lazy_logprobs.chosen.tolist(),
            top_val=top_val,
            top_idx=top_idx,
            token_ids_val=[
                a.tolist() if a is not None else [] for a in lazy_logprobs.token_ids_val
            ],
            token_ids_idx=[list(ids) if ids else [] for ids in spec.token_ids],
        )

    def eval_pending(
        self, pending: MlxPendingPrefill | MlxPendingExtend | MlxPendingDecode
    ) -> None:

        if isinstance(pending, MlxPendingDecode):
            tokens, caches = pending.lazy_tokens, pending.caches
        else:
            tokens, caches = pending.lazy_token, [pending.cache]
        mx.eval(
            tokens,
            *self.cache_state_arrays(caches),
            *lazy_logprob_arrays(pending.lazy_logprobs),
        )

    @staticmethod
    def _dummy_next_token(hidden: mx.array) -> mx.array:

        return (hidden[:, -1, 0] * 0).astype(mx.int32)

    def _extract_model_components(self):

        root = getattr(self.model, "language_model", self.model)
        text_model = getattr(root, "model", root)
        embed = text_model.embed_tokens
        norm = text_model.norm
        if hasattr(root, "lm_head"):
            lm_head = root.lm_head
        elif hasattr(root, "args") and getattr(root.args, "tie_word_embeddings", False):
            lm_head = text_model.embed_tokens.as_linear
        else:
            lm_head = root.lm_head
        return embed, norm, lm_head

    def _decode_with_hybrid_batching(
        self,
        caches: list[list[Any]],
        batched_input: mx.array,
        req_ids: list[str],
    ) -> mx.array:

        batch_size = len(caches)

        hidden_states = self._model_embed(batched_input)

        ctx = self._build_batched_decode_context(caches, req_ids)
        seq_lens = ctx.seq_lens
        max_offset = max(seq_lens)

        set_context(ctx)
        try:
            for layer_idx in range(self._cache_layout.num_layers):
                layer = self._cache_layout.layers[layer_idx]

                if self._cache_layout.attention_attrs[layer_idx] is not None:
                    shim = AttentionOffsetCache(offset=max_offset)
                    hidden_states = layer(hidden_states, mask=None, cache=shim)
                else:
                    layer_caches = [caches[i][layer_idx] for i in range(batch_size)]
                    hidden_states = self._decode_auxiliary_layer(
                        layer,
                        hidden_states,
                        layer_caches,
                    )
        finally:
            clear_context()

        hidden_states = self._model_norm(hidden_states)
        logits = self._extract_logits(self._model_lm_head(hidden_states))
        return logits[:, -1, :]

    def _decode_auxiliary_layer(
        self,
        layer: Any,
        hidden_states: mx.array,
        layer_caches: list[Any],
    ) -> mx.array:

        if self._can_batch_auxiliary_layer(layer, layer_caches):
            return self._decode_auxiliary_layer_batched(
                layer,
                hidden_states,
                layer_caches,
            )

        results = []
        for i, cache in enumerate(layer_caches):
            results.append(layer(hidden_states[i : i + 1], mask=None, cache=cache))
        return mx.concatenate(results, axis=0)

    @staticmethod
    def _can_batch_auxiliary_layer(layer: Any, layer_caches: list[Any]) -> bool:

        if not layer_caches:
            return False
        if not (
            getattr(layer, "is_linear", False)
            and hasattr(layer, "input_layernorm")
            and hasattr(layer, "linear_attn")
            and hasattr(layer, "post_attention_layernorm")
            and hasattr(layer, "mlp")
        ):
            return False

        cache_type = type(layer_caches[0])
        if not callable(getattr(cache_type, "merge", None)) or not all(
            isinstance(cache, cache_type) and callable(getattr(cache, "extract", None))
            for cache in layer_caches
        ):
            return False
        return True

    @staticmethod
    def _decode_auxiliary_layer_batched(
        layer: Any,
        hidden_states: mx.array,
        layer_caches: list[Any],
    ) -> mx.array:
        residual = hidden_states
        normed = layer.input_layernorm(hidden_states)

        batched_cache = MlxModelRunner._merge_auxiliary_caches(layer_caches)
        mixed = layer.linear_attn(normed, mask=None, cache=batched_cache)

        extract = getattr(batched_cache, "extract", None)
        if not callable(extract):
            raise RuntimeError(
                f"{type(batched_cache).__name__}.merge() returned a cache "
                "without extract(); cannot split auxiliary decode state"
            )
        for i, cache in enumerate(layer_caches):
            split_cache = extract(i)
            MlxModelRunner._replace_cache_contents(cache, split_cache)

        hidden_states = residual + mixed
        return hidden_states + layer.mlp(layer.post_attention_layernorm(hidden_states))

    @staticmethod
    def _merge_auxiliary_caches(layer_caches: list[Any]) -> Any:
        if MlxModelRunner._can_fast_merge_arrays_cache(layer_caches):
            return MlxModelRunner._fast_merge_arrays_cache(layer_caches)
        return type(layer_caches[0]).merge(layer_caches)

    @staticmethod
    def _can_fast_merge_arrays_cache(layer_caches: list[Any]) -> bool:
        cache_type = type(layer_caches[0])
        if cache_type.__name__ != "ArraysCache":
            return False
        return all(
            type(cache) is cache_type
            and isinstance(getattr(cache, "cache", None), list)
            and getattr(cache, "lengths", None) is None
            and getattr(cache, "left_padding", None) is None
            for cache in layer_caches
        )

    @staticmethod
    def _fast_merge_arrays_cache(layer_caches: list[Any]) -> Any:

        cache_type = type(layer_caches[0])
        merged = cache_type(len(layer_caches[0].cache))
        slots = []
        for slot_idx in range(len(layer_caches[0].cache)):
            values = [cache.cache[slot_idx] for cache in layer_caches]
            first = next((value for value in values if value is not None), None)
            if first is None:
                slots.append(None)
                continue
            slots.append(
                mx.concatenate(
                    [
                        value if value is not None else mx.zeros_like(first)
                        for value in values
                    ],
                    axis=0,
                )
            )
        merged.cache = slots
        return merged

    @staticmethod
    def _replace_cache_contents(cache: Any, new_cache: Any) -> None:

        if type(cache) is type(new_cache) and hasattr(cache, "__dict__"):
            cache.__dict__.clear()
            cache.__dict__.update(new_cache.__dict__)
            return
        if hasattr(cache, "state") and hasattr(new_cache, "state"):
            cache.state = new_cache.state
            return
        raise RuntimeError(
            f"Cannot copy {type(new_cache).__name__} state into "
            f"{type(cache).__name__}"
        )

    def _decode_with_native_cache(
        self,
        caches: list[list[Any]],
        input_ids_by_request: list[mx.array],
    ) -> mx.array:
        lazy_logits_list = []
        for input_ids, cache in zip(input_ids_by_request, caches):
            model_output = self.model(input_ids, cache=cache)
            logits = self._extract_logits(model_output)
            lazy_logits_list.append(logits[:, -1, :])
        return (
            lazy_logits_list[0]
            if len(lazy_logits_list) == 1
            else mx.concatenate(lazy_logits_list, axis=0)
        )

    def _decode_with_batched_attention(
        self,
        caches: list[list[Any]],
        batched_input: mx.array,
        req_ids: list[str],
    ) -> mx.array:
        ctx = self._build_batched_decode_context(caches, req_ids)
        seq_lens = ctx.seq_lens
        set_context(ctx)
        try:
            max_offset = max(seq_lens)
            shim_cache = [
                AttentionOffsetCache(offset=max_offset)
                for _ in range(self._cache_layout.num_layers)
            ]
            model_output = self.model(batched_input, cache=shim_cache)
            logits = self._extract_logits(model_output)
            return logits[:, -1, :]
        finally:
            clear_context()

    def _build_batched_decode_context(
        self,
        caches: list[list[Any]],
        req_ids: list[str],
    ) -> BatchedDecodeContext:

        return BatchedDecodeContext.from_decode(
            caches=caches,
            req_ids=req_ids,
            aot_kernels=self._aot_kernels,
            kv_pool=self._attention_kv_pool,
            req_pool_idx=self._req_pool_idx,
            req_to_token_pool=self._req_to_token_pool,
            attention_layer_indices=self._cache_layout.attention_layer_indices,
            attention_pool_index_by_layer=(
                self._cache_layout.attention_pool_index_by_layer
            ),
            full_kv_pool_index_by_layer=self._cache_layout.full_kv_pool_index_by_layer,
        )

    def decode_batch_start(
        self,
        req_ids: list[str],
        edit_rows: mx.array | None = None,
        logprob_spec: MlxLogprobSpec | None = None,
        logits_hook=None,
    ) -> MlxPendingDecode:

        caches = [self._req_caches[rid] for rid in req_ids]
        last_tokens = [self._req_token_ids[rid][-1] for rid in req_ids]
        batched_input = mx.array(last_tokens, dtype=mx.int32)[:, None]

        if self._cache_layout.has_auxiliary_state:
            last_logits = self._decode_with_hybrid_batching(
                caches, batched_input, list(req_ids)
            )
        else:
            last_logits = self._decode_with_batched_attention(
                caches, batched_input, list(req_ids)
            )

        if logits_hook is not None:


            if edit_rows is not None:
                last_logits = last_logits.astype(mx.float32) + edit_rows
                edit_rows = None
            last_logits = self._run_logits_hook(last_logits, logits_hook)
        lazy_tokens, lazy_logprobs = self._select_tokens_with_logprobs(
            last_logits, list(req_ids), caches, edit_rows, logprob_spec
        )
        return MlxPendingDecode(
            lazy_tokens=lazy_tokens,
            req_ids=list(req_ids),
            caches=caches,
            lazy_logprobs=lazy_logprobs,
            logprob_spec=logprob_spec,
            edit_rows=edit_rows,
        )

    def decode_batch_start_chained(
        self,
        prev: MlxPendingDecode,
    ) -> MlxPendingDecode:

        caches = prev.caches


        batched_input = prev.lazy_tokens[:, None]
        if self._cache_layout.has_auxiliary_state:
            last_logits = self._decode_with_hybrid_batching(
                caches, batched_input, prev.req_ids
            )
        else:
            last_logits = self._decode_with_batched_attention(
                caches, batched_input, prev.req_ids
            )

        lazy_tokens, lazy_logprobs = self._select_tokens_with_logprobs(
            last_logits, prev.req_ids, caches, prev.edit_rows, prev.logprob_spec
        )
        return MlxPendingDecode(
            lazy_tokens=lazy_tokens,
            req_ids=prev.req_ids,
            caches=caches,
            lazy_logprobs=lazy_logprobs,
            logprob_spec=prev.logprob_spec,
            edit_rows=prev.edit_rows,
        )

    def decode_batch_finalize(
        self,
        pending: MlxPendingDecode,
    ) -> list[int]:

        raw = pending.lazy_tokens.tolist()
        if not isinstance(raw, list):
            raw = [raw]
        next_tokens = [int(t) for t in raw]

        for i, rid in enumerate(pending.req_ids):
            self._req_token_ids[rid].append(next_tokens[i])

        self._decode_step_ct += 1
        if self._clear_steps > 0 and self._decode_step_ct % self._clear_steps == 0:
            mx.clear_cache()

        return next_tokens

    def has_request(self, req_id: str) -> bool:

        return req_id in self._req_caches

    def remove_request(self, req_id: str):

        if not self.disable_radix_cache:
            self._sync_decode_kv_to_pool(req_id)

        self._req_token_ids.pop(req_id, None)
        self._req_sampling.pop(req_id, None)
        cache = self._req_caches.pop(req_id, None)
        if cache is not None:
            self._release_cache(cache)
        self._req_pool_idx.pop(req_id, None)
        self._req_synced_offset.pop(req_id, None)

    def clear(self):

        self._req_token_ids.clear()
        self._req_sampling.clear()
        for cache in self._req_caches.values():
            self._release_cache(cache)
        self._req_caches.clear()
        self._req_pool_idx.clear()
        self._req_synced_offset.clear()
        if self._attention_kv_pool is not None:
            self._attention_kv_pool.clear()
