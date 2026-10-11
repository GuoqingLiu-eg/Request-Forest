from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.observability.metrics_collector import (
    STAT_LOGGER_ROLE_STORAGE,
    StorageMetricsCollector,
    resolve_collector_class,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

logger = logging.getLogger(__name__)


_PREFETCH_POLICIES = ("best_effort", "wait_complete", "timeout")
_WRITE_POLICIES = ("write_back", "write_through", "write_through_selective")


class StorageAttachment:


    def __init__(self, cache: UnifiedRadixCache):
        self._cache = cache


    def attach(
        self,
        storage_backend: str,
        storage_backend_extra_config_json: Optional[str] = None,
        served_model_name: Optional[str] = None,
        hicache_storage_prefetch_policy: Optional[str] = None,
        hicache_write_policy: Optional[str] = None,
    ) -> tuple[bool, str]:

        cache = self._cache


        invalid = self._validate_policies(
            hicache_storage_prefetch_policy, hicache_write_policy
        )
        if invalid is not None:
            return False, invalid

        controller = cache.cache_controller
        if controller is None:
            return (
                False,
                "HiCache is not initialized (no cache controller); "
                "launch with --enable-hierarchical-cache to attach a backend.",
            )

        if cache.enable_storage:
            current_backend = controller.storage_backend_type
            if current_backend != storage_backend:
                return (
                    False,
                    f"HiCache storage backend is already enabled with backend "
                    f"'{current_backend}'. Cannot attach different backend "
                    f"'{storage_backend}'. Detach first.",
                )

            self._apply_policies(hicache_storage_prefetch_policy, hicache_write_policy)
            return (
                True,
                "HiCache storage backend already enabled with same backend; "
                "policies updated.",
            )


        self._apply_policies(hicache_storage_prefetch_policy, hicache_write_policy)

        logger.info(f"Attaching HiCache storage backend: {storage_backend}")
        try:
            (
                extra_config,
                prefetch_threshold,
                prefetch_timeout_base,
                prefetch_timeout_per_ki_token,
                hicache_storage_pass_prefix_keys,
            ) = HybridCacheController.parse_storage_backend_extra_config(
                storage_backend_extra_config_json
            )
        except Exception as e:
            logger.exception(f"Failed to parse storage_backend_extra_config_json: {e}")
            return (
                False,
                f"Failed to parse storage_backend_extra_config_json "
                f"'{storage_backend_extra_config_json}': {e}",
            )

        try:
            controller.attach_storage_backend(
                storage_backend=storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=served_model_name,
                storage_backend_extra_config=extra_config,
                host_pools=controller.mem_pool_host.entries,
            )
        except Exception as e:
            logger.exception(
                f"Failed to attach storage backend '{storage_backend}': {e}"
            )
            return False, f"Failed to attach storage backend '{storage_backend}': {e}"

        self.apply_runtime_config(
            storage_backend=storage_backend,
            prefetch_threshold=prefetch_threshold,
            prefetch_timeout_base=prefetch_timeout_base,
            prefetch_timeout_per_ki_token=prefetch_timeout_per_ki_token,
            hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
            enable_storage=True,
            enable_storage_metrics=cache._enable_metrics_flag,
            extra_metric_labels=cache.extra_metric_labels,
        )
        return True, "Attached HiCache storage backend successfully."

    def detach(self) -> tuple[bool, str]:

        cache = self._cache
        controller = cache.cache_controller
        if controller is None:
            return False, "HiCache storage backend is not initialized."

        try:
            cache.drain_storage_control_queues_local()


            controller.detach_storage_backend()
        except Exception as e:
            logger.exception("Failed to detach storage backend.")


            return False, f"Failed to detach HiCache storage backend: {e}"

        try:
            self._release_pending_storage_ops()
            cache.drain_storage_control_queues_local()
        except Exception:
            logger.exception("Failed post-detach cleanup of storage bookkeeping.")

        cache.enable_storage = False
        cache.enable_storage_metrics = False
        return True, "Detached HiCache storage backend successfully."

    def shutdown(self) -> None:

        try:
            if self._cache.enable_storage:
                self.detach()
        except Exception:
            logger.exception("Failed to detach storage backend on process shutdown.")

    def clear(self) -> bool:

        try:
            ok = self._cache.cache_controller.clear_storage_backend()
        except Exception as e:
            logger.error("Failed to clear hierarchical cache storage backend: %s", e)
            return False
        if ok:
            logger.info("Hierarchical cache storage backend cleared successfully!")
        return ok


    def apply_runtime_config(
        self,
        *,
        storage_backend: Optional[str],
        prefetch_threshold: int,
        prefetch_timeout_base: float,
        prefetch_timeout_per_ki_token: float,
        hicache_storage_pass_prefix_keys: bool,
        enable_storage: bool,
        enable_storage_metrics: bool,
        extra_metric_labels: Optional[dict[str, str]],
    ) -> None:

        cache = self._cache


        if enable_storage and not cache.enable_storage:
            filled = cache.tree_core.backfill_missing_hash_values()
            if filled:
                logger.info(
                    "Hashed %d radix nodes that predate the storage backend.", filled
                )

        cache.enable_storage = enable_storage
        cache.prefetch_threshold = prefetch_threshold
        cache.prefetch_timeout_base = prefetch_timeout_base
        cache.prefetch_timeout_per_page = (
            cache.page_size / 1024 * prefetch_timeout_per_ki_token
        )
        cache.hicache_storage_pass_prefix_keys = hicache_storage_pass_prefix_keys
        cache.enable_storage_metrics = enable_storage_metrics

        if enable_storage_metrics:
            cache.storage_metrics_collector = self._resolve_metrics_collector(
                storage_backend, extra_metric_labels
            )
        else:
            cache.storage_metrics_collector = None

    def _resolve_metrics_collector(
        self,
        storage_backend: Optional[str],
        extra_metric_labels: Optional[dict[str, str]],
    ) -> Optional[StorageMetricsCollector]:

        cache = self._cache
        controller = cache.cache_controller
        attn_cp_rank, attn_cp_size = controller.get_attn_cp_rank_and_size()
        labels = {
            "storage_backend": storage_backend,
            "tp_rank": controller.tp_rank,
            "dp_rank": controller.dp_rank,
            "pp_rank": controller.pp_rank,
            "pp_size": controller.pp_size,
            "attn_cp_rank": attn_cp_rank,
            "attn_cp_size": attn_cp_size,
        }
        if extra_metric_labels:
            labels.update(extra_metric_labels)

        existing_collector = cache.storage_metrics_collector
        if existing_collector is None:
            from sglang.srt.runtime_context import get_server_args

            storage_cls = resolve_collector_class(
                get_server_args(),
                STAT_LOGGER_ROLE_STORAGE,
                StorageMetricsCollector,
            )
            return storage_cls(labels=labels)

        if set(existing_collector.labels.keys()) == set(labels.keys()):
            existing_collector.labels = labels
        else:
            logger.warning(
                "Storage metrics labels changed (%s -> %s). Keep existing labels to avoid duplicate metric registration.",
                sorted(existing_collector.labels.keys()),
                sorted(labels.keys()),
            )
        return existing_collector


    @staticmethod
    def _validate_policies(
        hicache_storage_prefetch_policy: Optional[str],
        hicache_write_policy: Optional[str],
    ) -> Optional[str]:

        if (
            hicache_storage_prefetch_policy is not None
            and hicache_storage_prefetch_policy not in _PREFETCH_POLICIES
        ):
            return (
                f"Invalid hicache_storage_prefetch_policy: "
                f"{hicache_storage_prefetch_policy!r}. "
                f"Expected one of {list(_PREFETCH_POLICIES)}."
            )
        if (
            hicache_write_policy is not None
            and hicache_write_policy not in _WRITE_POLICIES
        ):
            return (
                f"Invalid hicache_write_policy: {hicache_write_policy!r}. "
                f"Expected one of {list(_WRITE_POLICIES)}."
            )
        return None

    def _apply_policies(
        self,
        hicache_storage_prefetch_policy: Optional[str],
        hicache_write_policy: Optional[str],
    ) -> None:
        cache = self._cache
        if hicache_storage_prefetch_policy is not None:
            cache.prefetch_stop_policy = hicache_storage_prefetch_policy
            logger.info(
                f"Set hicache_storage_prefetch_policy to "
                f"{hicache_storage_prefetch_policy}"
            )
        if hicache_write_policy is not None:
            cache.cache_controller.write_policy = hicache_write_policy
            cache.write_through_threshold = (
                1 if hicache_write_policy == "write_through" else 2
            )
            cache.is_write_back = hicache_write_policy == "write_back"
            logger.info(f"Set hicache_write_policy to {hicache_write_policy}")

    def _release_pending_storage_ops(self) -> None:

        cache = self._cache
        controller = cache.cache_controller

        for req_id in list(cache.ongoing_prefetch):
            info = cache.ongoing_prefetch[req_id]
            try:
                if info.host_indices is None:

                    cache.revoke_pending_prefetch(req_id)
                    continue
                completed_tokens, _ = controller.terminate_prefetch(info.operation)
                del cache.ongoing_prefetch[req_id]
                cache.dec_host_lock_ref(info.anchor_node_id, info.anchor_lock_params)
                controller.append_host_mem_release(
                    host_indices=info.host_indices[:completed_tokens],
                    extra_pools=[
                        x for xfers in info.comp_xfers.values() for x in xfers
                    ],
                )
                controller.prefetch_tokens_occupied = max(
                    0, controller.prefetch_tokens_occupied - len(info.prefetch_key)
                )
            except Exception:
                logger.exception("Failed to release pending prefetch %s", req_id)
                cache.ongoing_prefetch.pop(req_id, None)

        for ack_id in list(cache.ongoing_backup):
            node_id, lock_params = cache.ongoing_backup.pop(ack_id)
            try:
                cache.dec_host_lock_ref(node_id, lock_params)
            except Exception:
                logger.exception("Failed to release host lock for backup op %s", ack_id)

        cache.prefetch_loaded_tokens_by_reqid.clear()
