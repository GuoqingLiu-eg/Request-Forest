import asyncio
import atexit
import contextlib
import logging
import multiprocessing as mp
import os
import time
import traceback
import uuid
from collections import defaultdict
from dataclasses import dataclass
from http import HTTPStatus
from typing import Dict, List, Optional, Set, Tuple

import zmq
import zmq.asyncio

import sglang.srt.disaggregation.encoder.server as server_module
from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.disaggregation.encoder.server import (
    ENCODER_MAX_BATCH_SIZE,
    ENCODER_MAX_BATCH_SIZE_EXPLICIT,
    EncoderProfiler,
    MMEncoder,
    MMError,
    launch_encoder,
)
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import (
    ProfileReq,
    ProfileReqType,
    async_sock_recv,
    async_sock_send,
    sock_send,
    wrap_as_pickle,
)
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.observability.metrics_collector import EncoderMetricsCollector
from sglang.srt.observability.req_time_stats import EncoderReqTimeStats
from sglang.srt.observability.trace import (
    process_tracing_init,
    trace_set_thread_info,
)
from sglang.srt.runtime_context import (
    get_observability,
    get_parallel,
    get_serving,
    publish,
)
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import configure_logger, random_uuid, set_prometheus_multiproc_dir
from sglang.srt.utils.common import maybe_reindex_device_id
from sglang.srt.utils.network import NetworkAddress, get_free_port, get_zmq_socket

logger = logging.getLogger(__name__)


class PendingRequest:
    __slots__ = ("request", "future", "submit_time")

    def __init__(self, request: dict, loop: asyncio.AbstractEventLoop):
        self.request = request
        self.future: asyncio.Future = loop.create_future()
        self.submit_time = time.time()


_BATCHABLE_MODALITIES = {Modality.IMAGE, Modality.AUDIO}
_KIMI_K3_DEFAULT_ENCODER_MAX_BATCH_SIZE = 2


def _resolve_encoder_batch_policy(
    model_type: str,
    configured_max_batch_size: int,
    max_batch_size_is_explicit: bool,
) -> Tuple[int, bool]:

    max_batch_size = max(1, int(configured_max_batch_size))
    coalesce_same_turn = model_type == "kimi_k3"
    if coalesce_same_turn and not max_batch_size_is_explicit:
        max_batch_size = min(max_batch_size, _KIMI_K3_DEFAULT_ENCODER_MAX_BATCH_SIZE)
    return max_batch_size, coalesce_same_turn


MINIMUM_PNG_PICTURE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAbUlEQVRYhe3VsQ2AMAxE0Y/lIgNQULD/OqyCMgCihCKSG4yRuKuiNH6JLsoEbMACOGBcua9HOR7Y6w6swBwMy0qLTpkeI77qdEBpBFAHBBDAGH8WrwJKI4AAegUCfAKgEgpQDvh3CR3oQCuav58qlAw73kKCSgAAAABJRU5ErkJggg=="


MINIMUM_WAV_SILENCE_BASE64 = "UklGRmQBAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YUABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=="


class EncoderScheduler:


    def __init__(
        self,
        encoder: "MMEncoder",
        send_sockets: List[zmq.Socket],
        max_batch_size: int,
        coalesce_same_turn: bool = False,
        request_timeout: float = server_module.ENCODER_REQ_TIMEOUT,
    ):
        self.encoder = encoder
        self.send_sockets = send_sockets
        self.max_batch_size = max(1, int(max_batch_size))
        self.coalesce_same_turn = bool(coalesce_same_turn)
        self.request_timeout = max(1.0, float(request_timeout))
        self.pending_queue: asyncio.Queue[PendingRequest] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None

    def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._batch_worker())
            logger.info(
                "EncoderScheduler started with "
                f"max_batch_size={self.max_batch_size}, "
                f"coalesce_same_turn={self.coalesce_same_turn}"
            )

    async def stop(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
            self._worker_task = None

        while True:
            try:
                pending = self.pending_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not pending.future.done():
                pending.future.set_exception(RuntimeError("EncoderScheduler stopped"))

    async def submit(self, request: dict) -> Tuple:
        pending = PendingRequest(request, asyncio.get_running_loop())
        await self.pending_queue.put(pending)
        try:
            return await asyncio.wait_for(pending.future, timeout=self.request_timeout)
        except asyncio.TimeoutError:
            if not pending.future.done():
                pending.future.cancel()
            req_id = request.get("req_id")

            await self.encoder.release_request(req_id)
            logger.error(
                f"EncoderScheduler.submit timed out after {self.request_timeout}s "
                f"for req_id={req_id}"
            )
            raise

    async def _collect_batch(self) -> List[PendingRequest]:
        batch = [await self.pending_queue.get()]
        first_modality = Modality.from_str(batch[0].request.get("modality", "image"))
        should_yield = (
            self.coalesce_same_turn
            and self.max_batch_size > 1
            and first_modality in _BATCHABLE_MODALITIES
        )
        if should_yield:


            await asyncio.sleep(0)
        while len(batch) < self.max_batch_size:
            try:
                batch.append(self.pending_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return batch

    async def _batch_worker(self) -> None:
        while True:
            batch: List[PendingRequest] = []
            try:
                batch = await self._collect_batch()
                groups: Dict[Modality, List[PendingRequest]] = defaultdict(list)
                for p in batch:
                    groups[
                        Modality.from_str(p.request.get("modality", "image"))
                    ].append(p)
                for modality, group in groups.items():
                    await self._dispatch_group(group, modality)
            except asyncio.CancelledError:
                for p in batch:
                    if not p.future.done():
                        p.future.set_exception(RuntimeError("EncoderScheduler stopped"))
                raise
            except Exception as e:
                logger.error(
                    f"Error in EncoderScheduler batch worker: {e}", exc_info=True
                )
                for p in batch:
                    if not p.future.done():
                        p.future.set_exception(e)

    @staticmethod
    def _validate_request_shape(req: dict) -> Optional[str]:


        if not isinstance(req, dict):
            return f"request is not a dict: {type(req).__name__}"
        if not req.get("req_id"):
            return "missing req_id"
        if not req.get("mm_items"):
            return "missing or empty mm_items"
        if "num_parts" not in req or "part_idx" not in req:
            return "missing num_parts / part_idx"
        h = req.get("hashes")
        if h is not None and not isinstance(h, (list, tuple, str, int, bytes)):
            return f"hashes must be list/scalar, got {type(h).__name__}"
        return None

    async def _dispatch_group(
        self, group: List[PendingRequest], modality: Modality
    ) -> None:


        group = [pending for pending in group if not pending.future.done()]
        if not group:
            return


        if modality not in _BATCHABLE_MODALITIES:
            await self._dispatch_per_request(group, modality)
            return


        valid: List[PendingRequest] = []
        for p in group:
            err = self._validate_request_shape(p.request)
            if err is None:
                valid.append(p)
                continue
            logger.error(f"Dropping req_id={p.request.get('req_id')} from batch: {err}")
            if not p.future.done():
                p.future.set_exception(server_module.BadRequestError(err))
        if not valid:
            return
        group = valid

        requests = [p.request for p in group]
        start = time.time()
        modality_str = modality.name.lower()
        if server_module.encoder_metrics_collector is not None:
            for p in group:
                server_module.encoder_metrics_collector.observe_queue_wait(
                    max(0.0, start - p.submit_time), modality=modality_str
                )
        try:


            async with self.encoder.encode_dispatch_lock:
                for sock in self.send_sockets:
                    sock_send(
                        sock,
                        wrap_as_pickle(
                            {
                                "type": "batch_encode",
                                "modality": modality.name,
                                "requests": requests,
                                "enter_time": start,
                            }
                        ),
                    )

                logger.info(
                    f"Dispatching batch of {len(group)} {modality.name} requests"
                )
                results = await self.encoder.batch_encode(requests, modality)
            if len(group) > 1:
                logger.info(
                    f"Batch of {len(group)} {modality.name} requests completed in "
                    f"{(time.time() - start) * 1000:.1f}ms"
                )
        except Exception as e:


            logger.error(f"batch_encode raised: {e}", exc_info=True)
            for p in group:
                if not p.future.done():
                    p.future.set_exception(e)
            return

        if len(results) != len(group):
            err = RuntimeError(
                f"batch_encode returned {len(results)} results for {len(group)} requests"
            )
            logger.error(str(err))
            for p in group:
                if not p.future.done():
                    p.future.set_exception(err)
            return

        for p, result in zip(group, results):
            if not p.future.done():
                p.future.set_result(result)

    async def _dispatch_per_request(
        self,
        group: List[PendingRequest],
        modality: Modality,
    ) -> None:
        modality_str = modality.name.lower()
        for p in group:
            if p.future.done():
                continue
            req = p.request
            try:
                start = time.time()
                if server_module.encoder_metrics_collector is not None:
                    server_module.encoder_metrics_collector.observe_queue_wait(
                        max(0.0, start - p.submit_time), modality=modality_str
                    )
                for sock in self.send_sockets:
                    sock_send(sock, wrap_as_pickle(req))
                result = await self.encoder.encode(
                    mm_items=req["mm_items"],
                    modality=modality,
                    req_id=req["req_id"],
                    num_parts=req["num_parts"],
                    part_idx=req["part_idx"],
                    hashes=req.get("hashes"),
                )
                if not p.future.done():
                    p.future.set_result(result)
            except Exception as e:
                logger.error(
                    f"Per-request encode failed for req_id={req.get('req_id')}: {e}"
                )
                if not p.future.done():
                    p.future.set_exception(e)


@dataclass
class EncoderRuntime:


    encoder: MMEncoder
    scheduler: EncoderScheduler
    send_sockets: List[zmq.Socket]
    zmq_context: zmq.Context
    tp_processes: List[mp.Process]

    def start(self) -> None:
        self.scheduler.start()

    async def stop(self) -> None:


        await self.scheduler.stop()


class DPDispatcher:


    def __init__(
        self,
        dp_size: int,
        dispatch_sockets: List,
        result_socket,
        worker_processes: List[mp.Process],
        enable_metrics: bool = False,
        labels: Optional[Dict[str, str]] = None,
    ):
        self.dp_size = dp_size
        self.dispatch_sockets = dispatch_sockets
        self.result_socket = result_socket
        self.worker_processes = worker_processes


        self.pending_futures: List[Dict[str, asyncio.Future]] = [
            {} for _ in range(dp_size)
        ]
        self.req_id_to_rank: Dict[str, int] = {}
        self._mapping_condition = asyncio.Condition()
        self._rr_counter = 0
        self._broadcast_counter = 0
        self._metadata_counter = 0
        self._dead_ranks: Set[int] = set()

        self._pending_send_at: Dict[str, float] = {}

        self._listener_failed = False


        self.labels = dict(labels or {})
        self.pending_gauge = None
        if enable_metrics:
            from prometheus_client import Gauge

            self.pending_gauge = Gauge(
                name="sglang:encoder_dp_pending_requests",
                documentation="Number of pending requests per encoder DP rank.",
                labelnames=list(self.labels.keys()) + ["dp_rank"],
                multiprocess_mode="mostrecent",
            )

    @property
    def pending_counts(self) -> List[int]:
        return [len(d) for d in self.pending_futures]

    def _update_pending_gauge(self) -> None:

        if self.pending_gauge is not None:
            for i, c in enumerate(self.pending_counts):
                self.pending_gauge.labels(**self.labels, dp_rank=str(i)).set(c)

    @property
    def alive_ranks(self) -> List[int]:

        if self._listener_failed:
            return []
        return [r for r in range(self.dp_size) if r not in self._dead_ranks]

    @property
    def all_ranks_alive(self) -> bool:

        return len(self.alive_ranks) == self.dp_size

    def start(self) -> None:
        logger.info(f"DP dispatcher started: {self.dp_size} ranks (all remote)")
        asyncio.create_task(self._result_listener())
        asyncio.create_task(self._worker_watchdog())
        asyncio.create_task(self._cleanup_stale_mappings())

    def _drop_pending_and_mapping(self, rank: int, req_id: str) -> None:

        self.pending_futures[rank].pop(req_id, None)
        self.req_id_to_rank.pop(req_id, None)
        self._update_pending_gauge()

    @staticmethod
    def _send_req_key(req_id: str, request: dict) -> str:

        endpoint = NetworkAddress(
            request["prefill_host"], request["embedding_port"]
        ).to_host_port_str()
        return f"{req_id}_send_{endpoint}"

    @staticmethod
    def _register_req_key(req_id: str, request: dict) -> str:
        return f"{req_id}_register_{request['receive_url']}"

    def _metadata_req_key(self, req_id: str) -> str:
        key = f"{req_id}_metadata_{self._metadata_counter}"
        self._metadata_counter += 1
        return key

    @staticmethod
    def _pending_req_info(key: str) -> Tuple[str, str]:
        marker_index, dp_type = max(
            (
                (key.rfind("_send_"), "send"),
                (key.rfind("_metadata_"), "wait_metadata"),
                (key.rfind("_register_"), "register_destinations"),
            ),
            key=lambda item: item[0],
        )
        if marker_index >= 0:
            return key[:marker_index], dp_type
        return key, "encode"

    def _fail_pending_for_rank(self, rank: int, reason: str, error_type: str) -> None:

        pending = self.pending_futures[rank]
        for key, future in list(pending.items()):
            if not future.done():
                req_id, dp_type = self._pending_req_info(key)
                future.set_result(
                    {
                        "req_id": req_id,
                        "_dp_type": dp_type,
                        "content": None,
                        "_error": reason,
                        "_error_type": error_type,
                        "_error_code": int(HTTPStatus.SERVICE_UNAVAILABLE),
                    }
                )
            pending.pop(key, None)
        self._update_pending_gauge()

    def _fail_all_pending(self, reason: str, error_type: str) -> None:
        for rank in range(self.dp_size):
            self._fail_pending_for_rank(rank, reason, error_type)
        self.req_id_to_rank.clear()
        self._pending_send_at.clear()

    @staticmethod
    def _timeout_envelope(req_id: str, dp_type: str, reason: str) -> dict:
        return {
            "req_id": req_id,
            "_dp_type": dp_type,
            "content": None,
            "_error": reason,
            "_error_type": "TimeoutError",
            "_error_code": int(HTTPStatus.GATEWAY_TIMEOUT),
        }

    async def dispatch(self, request: dict) -> dict:
        counts = self.pending_counts

        alive_ranks = self.alive_ranks
        if not alive_ranks:
            raise server_module.MMError(
                "All encoder DP workers are dead.",
                code=HTTPStatus.SERVICE_UNAVAILABLE,
            )
        min_p = min(counts[r] for r in alive_ranks)
        candidates = [r for r in alive_ranks if counts[r] == min_p]
        rank = candidates[self._rr_counter % len(candidates)]
        self._rr_counter += 1
        req_id = request["req_id"]
        future = asyncio.get_running_loop().create_future()
        self.pending_futures[rank][req_id] = future
        self._update_pending_gauge()
        logger.info(
            f"MM-Encoder DP dispatch: req_id={req_id}, "
            f"modality={request.get('modality', 'image')}, "
            f"dp_rank={rank}, pending={self.pending_counts}"
        )

        try:


            async with self._mapping_condition:
                self.req_id_to_rank[req_id] = rank
                try:
                    await async_sock_send(
                        self.dispatch_sockets[rank], wrap_as_pickle(request)
                    )
                except BaseException:
                    self._drop_pending_and_mapping(rank, req_id)
                    self._mapping_condition.notify_all()
                    raise
                self._mapping_condition.notify_all()


            return await asyncio.wait_for(
                future, timeout=server_module.ENCODER_REQ_TIMEOUT
            )
        except asyncio.TimeoutError:
            self._drop_pending_and_mapping(rank, req_id)
            return self._timeout_envelope(
                req_id,
                "encode",
                f"Encoder DP rank={rank} timed out after {server_module.ENCODER_REQ_TIMEOUT}s",
            )
        except BaseException:
            self._drop_pending_and_mapping(rank, req_id)
            raise

    async def dispatch_register_destinations(self, request: dict) -> dict:

        req_id = request["req_id"]
        deadline = time.monotonic() + min(5.0, server_module.ENCODER_REQ_TIMEOUT)
        async with self._mapping_condition:
            while req_id not in self.req_id_to_rank:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return {
                        "req_id": req_id,
                        "_error": f"Unknown req_id: {req_id}",
                        "_error_code": int(HTTPStatus.NOT_FOUND),
                    }
                try:
                    await asyncio.wait_for(
                        self._mapping_condition.wait(), timeout=remaining
                    )
                except asyncio.TimeoutError:
                    return {
                        "req_id": req_id,
                        "_error": f"Unknown req_id: {req_id}",
                        "_error_code": int(HTTPStatus.NOT_FOUND),
                    }
            rank = self.req_id_to_rank[req_id]

        if rank in self._dead_ranks:
            return {
                "req_id": req_id,
                "_error": f"DP worker rank={rank} died before URL registration",
                "_error_code": int(HTTPStatus.SERVICE_UNAVAILABLE),
            }

        key = self._register_req_key(req_id, request)
        future = asyncio.get_running_loop().create_future()
        self.pending_futures[rank][key] = future
        self._update_pending_gauge()
        worker_request = {
            **request,
            "_dp_type": "register_destinations",
            "_dp_register_key": key,
        }
        try:
            await async_sock_send(
                self.dispatch_sockets[rank], wrap_as_pickle(worker_request)
            )
            return await asyncio.wait_for(
                future, timeout=server_module.ENCODER_REQ_TIMEOUT
            )
        except asyncio.TimeoutError:
            self.pending_futures[rank].pop(key, None)
            self._update_pending_gauge()
            return self._timeout_envelope(
                req_id,
                "register_destinations",
                f"Encoder DP rank={rank} URL registration timed out after "
                f"{server_module.ENCODER_REQ_TIMEOUT}s",
            )
        except BaseException:
            self.pending_futures[rank].pop(key, None)
            self._update_pending_gauge()
            raise

    async def dispatch_wait_metadata(self, request: dict) -> dict:

        req_id = request["req_id"]
        deadline = time.monotonic() + min(5.0, server_module.ENCODER_REQ_TIMEOUT)
        async with self._mapping_condition:
            while req_id not in self.req_id_to_rank:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return {
                        "req_id": req_id,
                        "_error": f"Unknown req_id: {req_id}",
                        "_error_code": int(HTTPStatus.NOT_FOUND),
                    }
                try:
                    await asyncio.wait_for(
                        self._mapping_condition.wait(), timeout=remaining
                    )
                except asyncio.TimeoutError:
                    return {
                        "req_id": req_id,
                        "_error": f"Unknown req_id: {req_id}",
                        "_error_code": int(HTTPStatus.NOT_FOUND),
                    }
            rank = self.req_id_to_rank[req_id]

        if rank in self._dead_ranks:
            return {
                "req_id": req_id,
                "_error": f"DP worker rank={rank} died before metadata became ready",
                "_error_code": int(HTTPStatus.SERVICE_UNAVAILABLE),
            }

        key = self._metadata_req_key(req_id)
        future = asyncio.get_running_loop().create_future()
        self.pending_futures[rank][key] = future
        self._update_pending_gauge()
        worker_request = {
            **request,
            "_dp_type": "wait_metadata",
            "_dp_metadata_key": key,
        }
        try:
            await async_sock_send(
                self.dispatch_sockets[rank], wrap_as_pickle(worker_request)
            )
            return await asyncio.wait_for(
                future, timeout=server_module.ENCODER_REQ_TIMEOUT
            )
        except asyncio.TimeoutError:
            self.pending_futures[rank].pop(key, None)
            self._update_pending_gauge()
            return self._timeout_envelope(
                req_id,
                "wait_metadata",
                f"Encoder DP rank={rank} metadata wait timed out after "
                f"{server_module.ENCODER_REQ_TIMEOUT}s",
            )
        except BaseException:
            self.pending_futures[rank].pop(key, None)
            self._update_pending_gauge()
            raise

    async def dispatch_send(self, request: dict) -> dict:
        req_id = request["req_id"]

        self._pending_send_at.pop(req_id, None)
        if self._listener_failed:
            return {
                "req_id": req_id,
                "_error": "encoder DP result listener stopped; cannot route /send",
                "_error_code": int(HTTPStatus.SERVICE_UNAVAILABLE),
            }
        rank = self.req_id_to_rank.get(req_id)
        if rank is None:
            logger.warning(
                f"MM-Encoder dispatch_send: unknown req_id={req_id}, "
                f"cannot route to worker"
            )
            return {"req_id": req_id, "_error": f"Unknown req_id: {req_id}"}
        if rank in self._dead_ranks:

            self.req_id_to_rank.pop(req_id, None)
            return {
                "req_id": req_id,
                "_error": f"DP worker rank={rank} died before /send for req_id={req_id}",
                "_error_code": int(HTTPStatus.SERVICE_UNAVAILABLE),
            }
        key = self._send_req_key(req_id, request)
        future = asyncio.get_running_loop().create_future()
        self.pending_futures[rank][key] = future
        self._update_pending_gauge()
        request["_dp_type"] = "send"
        request["_dp_send_key"] = key
        logger.info(
            f"MM-Encoder DP dispatch_send: req_id={req_id}, "
            f"dp_rank={rank}, send_key={key}, pending={self.pending_counts}"
        )
        try:
            await async_sock_send(self.dispatch_sockets[rank], wrap_as_pickle(request))
            return await asyncio.wait_for(
                future, timeout=server_module.ENCODER_REQ_TIMEOUT
            )
        except asyncio.TimeoutError:
            self.pending_futures[rank].pop(key, None)
            self._update_pending_gauge()

            self._pending_send_at[req_id] = time.monotonic()
            return self._timeout_envelope(
                req_id,
                "send",
                f"Encoder DP rank={rank} /send timed out after {server_module.ENCODER_REQ_TIMEOUT}s",
            )
        except BaseException:
            self.pending_futures[rank].pop(key, None)
            self._update_pending_gauge()
            self._pending_send_at[req_id] = time.monotonic()
            raise

    async def broadcast(
        self, request: dict, timeout: Optional[float] = None
    ) -> List[dict]:


        eff_timeout = (
            timeout if timeout is not None else server_module.ENCODER_REQ_TIMEOUT
        )
        alive_ranks = self.alive_ranks
        if not alive_ranks:
            raise server_module.MMError(
                "All encoder DP workers are dead.",
                code=HTTPStatus.SERVICE_UNAVAILABLE,
            )
        batch_id = self._broadcast_counter
        self._broadcast_counter += 1
        rank_keys: List[Tuple[int, str]] = []
        futures: List[asyncio.Future] = []
        dp_type = request.get("_dp_type", "unknown")
        try:
            for rank in alive_ranks:
                req_id = f"_broadcast_{batch_id}_{rank}"
                future = asyncio.get_running_loop().create_future()
                self.pending_futures[rank][req_id] = future
                self.req_id_to_rank[req_id] = rank
                rank_keys.append((rank, req_id))
                request_copy = {**request, "req_id": req_id}
                await async_sock_send(
                    self.dispatch_sockets[rank], wrap_as_pickle(request_copy)
                )
                futures.append(future)


            outcomes = await asyncio.gather(
                *(asyncio.wait_for(fut, timeout=eff_timeout) for fut in futures),
                return_exceptions=True,
            )
            results: List[dict] = []
            for (rank, req_id), outcome in zip(rank_keys, outcomes):
                if isinstance(outcome, asyncio.TimeoutError):
                    self._drop_pending_and_mapping(rank, req_id)
                    results.append(
                        self._timeout_envelope(
                            req_id,
                            dp_type,
                            f"Encoder DP rank={rank} broadcast timed out "
                            f"after {eff_timeout}s",
                        )
                    )
                elif isinstance(outcome, BaseException):
                    self._drop_pending_and_mapping(rank, req_id)
                    raise outcome
                else:
                    results.append(outcome)
            return results
        except BaseException:
            for rank, req_id in rank_keys:
                self._drop_pending_and_mapping(rank, req_id)
            raise

    async def _worker_watchdog(self) -> None:


        loop = asyncio.get_running_loop()
        watch: Dict[int, asyncio.Future] = {}
        for rank, proc in enumerate(self.worker_processes):
            fut: asyncio.Future = loop.create_future()


            def _on_exit(r=rank, f=fut, p=proc, lp=loop):
                try:
                    lp.remove_reader(p.sentinel)
                except (ValueError, OSError):
                    pass
                if not f.done():
                    f.set_result(r)

            try:
                loop.add_reader(proc.sentinel, _on_exit)
            except (ValueError, OSError):
                continue
            watch[rank] = fut

        while watch:
            done, _ = await asyncio.wait(
                watch.values(), return_when=asyncio.FIRST_COMPLETED
            )
            for fut in done:
                rank = fut.result()
                proc = self.worker_processes[rank]
                logger.error(
                    f"DP worker rank={rank} (pid={proc.pid}) exited "
                    f"with code={proc.exitcode}; failing pending requests"
                )
                self._dead_ranks.add(rank)
                reason = f"DP worker rank={rank} died (exitcode={proc.exitcode})"
                self._fail_pending_for_rank(rank, reason, "WorkerDied")
                self.req_id_to_rank = {
                    r: rk for r, rk in self.req_id_to_rank.items() if rk != rank
                }
                watch.pop(rank, None)

    async def _result_listener(self) -> None:


        consecutive_errors = 0
        while True:
            try:
                msg = await async_sock_recv(self.result_socket)
                consecutive_errors = 0
            except asyncio.CancelledError:
                raise
            except Exception:
                consecutive_errors += 1
                logger.error("_result_listener recv error", exc_info=True)
                if consecutive_errors >= 30:
                    logger.error(
                        "_result_listener giving up after 30 consecutive errors"
                    )
                    self._listener_failed = True
                    self._fail_all_pending(
                        "encoder DP result listener stopped after repeated "
                        "recv errors",
                        "ResultListenerStopped",
                    )
                    return
                await asyncio.sleep(min(0.1 * consecutive_errors, 1.0))
                continue
            req_id = msg.get("req_id", "")
            dp_type = msg.get("_dp_type", "encode")
            if dp_type == "send":
                key = msg.get("_dp_send_key")
                if key is None:


                    logger.warning(
                        f"_result_listener: send envelope without _dp_send_key "
                        f"for req_id={req_id}, dropping"
                    )
                    continue
            elif dp_type == "register_destinations":
                key = msg.get("_dp_register_key")
                if key is None:
                    logger.warning(
                        f"_result_listener: URL registration envelope without "
                        f"_dp_register_key for req_id={req_id}, dropping"
                    )
                    continue
            elif dp_type == "wait_metadata":
                key = msg.get("_dp_metadata_key")
                if key is None:
                    logger.warning(
                        f"_result_listener: metadata envelope without "
                        f"_dp_metadata_key for req_id={req_id}, dropping"
                    )
                    continue
            else:
                key = req_id
            rank = self.req_id_to_rank.get(req_id)
            if rank is None or key not in self.pending_futures[rank]:
                logger.warning(
                    f"_result_listener: no pending future for "
                    f"req_id={req_id}, dp_type={dp_type}, key={key}, dropping"
                )
                continue
            future = self.pending_futures[rank].pop(key)
            self._update_pending_gauge()


            register_prefix = f"{req_id}_register_"
            has_pending_registration = any(
                pending_key.startswith(register_prefix)
                for pending_key in self.pending_futures[rank]
            )
            metadata_prefix = f"{req_id}_metadata_"
            has_pending_metadata = any(
                pending_key.startswith(metadata_prefix)
                for pending_key in self.pending_futures[rank]
            )
            keep_mapping = (
                dp_type in ("send", "register_destinations", "wait_metadata")
                or (dp_type == "encode" and msg.get("content") is not None)
                or has_pending_registration
                or has_pending_metadata
            )
            if dp_type == "send" or (
                dp_type == "encode" and msg.get("content") is not None
            ):
                self._pending_send_at[req_id] = time.monotonic()
            if not keep_mapping:
                self.req_id_to_rank.pop(req_id, None)
            try:
                future.set_result(msg)

            except asyncio.InvalidStateError:
                logger.warning(
                    f"_result_listener: future already done for "
                    f"req_id={req_id}, dp_type={dp_type}, key={key}"
                )

            if dp_type == "register_destinations":
                encode_still_pending = req_id in self.pending_futures[rank]
                other_registration_pending = any(
                    pending_key.startswith(register_prefix)
                    for pending_key in self.pending_futures[rank]
                )
                if not encode_still_pending and not other_registration_pending:
                    self.req_id_to_rank.pop(req_id, None)
            elif dp_type == "wait_metadata":
                encode_still_pending = req_id in self.pending_futures[rank]
                other_metadata_pending = any(
                    pending_key.startswith(metadata_prefix)
                    for pending_key in self.pending_futures[rank]
                )
                if (
                    not encode_still_pending
                    and not other_metadata_pending
                    and req_id not in self._pending_send_at
                ):
                    self.req_id_to_rank.pop(req_id, None)

    async def _cleanup_stale_mappings(self) -> None:


        ttl = envs.SGLANG_ENCODER_SEND_TIMEOUT.get()
        interval = max(ttl / 4, 30)
        while True:
            await asyncio.sleep(interval)
            now = time.monotonic()
            stale = [rid for rid, ts in self._pending_send_at.items() if now - ts > ttl]
            for rid in stale:
                self._pending_send_at.pop(rid, None)
                self.req_id_to_rank.pop(rid, None)
            if stale:
                logger.warning(
                    f"Evicted {len(stale)} stale encoder DP /send mapping(s) "
                    f"with no /send within {ttl}s"
                )


async def _push_embedding_to_prefill(
    enc: MMEncoder,
    request: dict,
    *,
    background_url_send: bool = False,
) -> None:

    req_id = request["req_id"]
    backend = enc.transfer_backend

    if backend == "mooncake":
        return

    if backend == "zmq_to_scheduler" and request.get("embedding_port") is None:
        send_coro = enc.send_with_url(req_id=req_id)
        if background_url_send:
            task = asyncio.create_task(send_coro)
            enc.background_tasks.add(task)
            task.add_done_callback(enc.background_tasks.discard)
        else:
            await send_coro
        return

    if backend == "zmq_to_tokenizer":
        try:
            await enc.send(
                req_id=req_id,
                prefill_host=request["prefill_host"],
                embedding_port=request["embedding_port"],
            )
        finally:
            await enc.release_request(req_id)
        return

    if backend == "zmq_to_scheduler":
        ports = request["embedding_port"]
        assert isinstance(ports, list)
        try:
            await asyncio.gather(
                *(
                    enc.send(
                        req_id=req_id,
                        prefill_host=request["prefill_host"],
                        embedding_port=p,
                    )
                    for p in ports
                )
            )
        finally:
            await enc.release_request(req_id)


def _record_pipeline_result(modality: Modality, status: str) -> None:
    if server_module.encoder_metrics_collector is not None:
        server_module.encoder_metrics_collector.inc_requests_total(
            modality=modality.name.lower(), status=status
        )


async def execute_encode_pipeline(
    enc: MMEncoder,
    sched: Optional[EncoderScheduler],
    request: dict,
    *,
    send_sockets: Optional[List[zmq.Socket]] = None,
) -> Optional[dict]:

    req_id = request["req_id"]
    time_stats_json = request.pop("time_stats_json", None)
    time_stats = EncoderReqTimeStats()
    if time_stats_json:
        time_stats.decode_json(time_stats_json)
    request["enter_time"] = time.time()
    modality = Modality.from_str(request["modality"])
    modality_str = modality.name.lower()
    time_stats.modality = modality_str
    time_stats.set_metrics_collector(server_module.encoder_metrics_collector)
    backend = enc.transfer_backend

    if server_module.encoder_metrics_collector is not None:
        server_module.encoder_metrics_collector.inc_requests_received(
            modality=modality_str
        )

    time_stats.set_mm_encode_start_time()
    try:
        if sched is not None and modality in _BATCHABLE_MODALITIES:
            result = await sched.submit(request)
        elif send_sockets is not None:


            async with enc.encode_dispatch_lock:
                for socket in send_sockets:
                    sock_send(socket, wrap_as_pickle(request))
                result = await enc.encode(
                    mm_items=request["mm_items"],
                    modality=modality,
                    req_id=request["req_id"],
                    num_parts=request["num_parts"],
                    part_idx=request["part_idx"],
                    hashes=request.get("hashes"),
                )
        else:
            result = await enc.encode(
                mm_items=request["mm_items"],
                modality=modality,
                req_id=request["req_id"],
                num_parts=request["num_parts"],
                part_idx=request["part_idx"],
                hashes=request.get("hashes"),
            )
    except asyncio.TimeoutError:
        error_msg = "encoder batch timed out"
        time_stats.trace_ctx.abort(abort_info={"reason": error_msg})
        await server_module.meta_registry.publish(req_id, 0, 0, 0, error=error_msg)
        await enc.release_request(req_id, preserve_metadata=backend == "mooncake")
        _record_pipeline_result(modality, "error")
        raise
    except Exception as e:
        error_msg = str(e)
        time_stats.trace_ctx.abort(abort_info={"reason": error_msg})
        await server_module.meta_registry.publish(req_id, 0, 0, 0, error=error_msg)
        await enc.release_request(req_id, preserve_metadata=backend == "mooncake")
        _record_pipeline_result(modality, "error")
        raise

    nbytes, embedding_len, embedding_dim, error_msg, error_code = result
    if error_msg:
        time_stats.trace_ctx.abort(abort_info={"reason": error_msg})
        await server_module.meta_registry.publish(req_id, 0, 0, 0, error=error_msg)
        if backend == "mooncake":
            await enc.release_request(req_id, preserve_metadata=True)
        else:
            try:
                await _push_embedding_to_prefill(
                    enc,
                    request,
                    background_url_send=True,
                )
            except Exception as send_err:
                logger.error(
                    f"Error-send failed for req_id={req_id}: {send_err}",
                    exc_info=True,
                )
        _record_pipeline_result(modality, "error")
        raise MMError(error_msg, code=error_code or HTTPStatus.INTERNAL_SERVER_ERROR)

    time_stats.set_mm_encode_end_time()
    try:


        await server_module.meta_registry.publish(
            req_id, nbytes, embedding_len, embedding_dim
        )

        if backend == "mooncake":
            request.pop("mm_items", None)
            request.update(
                embedding_size=nbytes,
                embedding_len=embedding_len,
                embedding_dim=embedding_dim,
            )
            content = request
        else:
            await _push_embedding_to_prefill(enc, request)
            content = None
    except Exception as e:
        time_stats.trace_ctx.abort(abort_info={"reason": str(e)})
        await enc.release_request(req_id)
        _record_pipeline_result(modality, "error")
        raise

    _record_pipeline_result(modality, "success")
    return content


async def _dp_worker_health_encode(enc: MMEncoder) -> None:

    if enc.supports_modality(Modality.IMAGE):
        mm_items = [f"data:image/png;base64,{MINIMUM_PNG_PICTURE_BASE64}"]
        modality = Modality.IMAGE
    elif enc.supports_modality(Modality.AUDIO):
        mm_items = [f"data:audio/wav;base64,{MINIMUM_WAV_SILENCE_BASE64}"]
        modality = Modality.AUDIO
    else:

        return None


    req_id = f"{HEALTH_CHECK_RID_PREFIX}_{uuid.uuid4().hex}"
    try:
        async with enc.encode_dispatch_lock:

            if enc.has_pending_embeddings():
                return None
            _, _, _, error_msg, error_code = await enc.encode(
                mm_items=mm_items,
                modality=modality,
                req_id=req_id,
                num_parts=1,
                part_idx=0,
            )
    finally:

        await enc.release_request(req_id)

    if error_msg:
        raise MMError(error_msg, code=error_code or HTTPStatus.INTERNAL_SERVER_ERROR)


async def _dp_worker_handle_profile(
    enc: MMEncoder, dp_rank: int, dp_type: str, request: dict
) -> dict:
    prefix = f"dp_rank={dp_rank}: "
    if dp_type == "start_profile":
        req = request.get("profile_req") or ProfileReq()
        req.req_type = ProfileReqType.START_PROFILE
        if enc.profiler is None:
            enc.profiler = EncoderProfiler(dp_rank)
        ok, msg = enc.profiler.start(req)
        detail = (
            f"started profiling, output_dir={enc.profiler.output_dir}" if ok else msg
        )
    else:
        if enc.profiler is None:
            return {"ok": False, "msg": prefix + "profiling not initialized"}
        ok, msg = enc.profiler.stop()
        detail = "stopped profiling" if ok else msg
    return {"ok": ok, "msg": prefix + detail}


async def _dp_worker_handle_request(
    enc: MMEncoder,
    sched: EncoderScheduler,
    send_sock,
    send_lock: asyncio.Lock,
    dp_rank: int,
    request: dict,
    dp_type: str,
) -> None:
    t0 = time.time()
    try:
        if dp_type in ("start_profile", "stop_profile"):
            content = await _dp_worker_handle_profile(enc, dp_rank, dp_type, request)
        elif dp_type == "health_encode":
            content = await _dp_worker_health_encode(enc)
        elif dp_type == "register_destinations":
            await enc.register_embedding_destinations(
                request["req_id"],
                request["receive_count"],
                [request["receive_url"]],
            )
            content = None
        elif dp_type == "wait_metadata":
            try:
                content = await server_module.meta_registry.wait(request["req_id"])
            except asyncio.TimeoutError as e:
                raise MMError(
                    "encode metadata not ready", code=HTTPStatus.GATEWAY_TIMEOUT
                ) from e
        elif dp_type == "send":
            req_id = request["req_id"]
            sent = await enc.send(
                req_id=req_id,
                prefill_host=request["prefill_host"],
                embedding_port=request["embedding_port"],
                session_id=request["session_id"],
                buffer_address=request["buffer_address"],
            )
            if not sent:


                raise MMError(
                    f"no staged embedding for /send req_id={req_id} "
                    f"(already released)"
                )


            receive_count = request.get("receive_count")
            if receive_count:
                await server_module.meta_registry.note_send_done(req_id, receive_count)
            else:
                await enc.release_request(req_id)
            content = None
        else:
            content = await execute_encode_pipeline(enc, sched, request)

        logger.info(
            f"MM-Encoder [dp_rank={dp_rank}] {dp_type} done: "
            f"req_id={request.get('req_id', '?')}, "
            f"modality={request.get('modality', 'image')}, "
            f"cost={(time.time() - t0) * 1000:.1f}ms"
        )
        envelope = {
            "req_id": request.get("req_id", ""),
            "_dp_type": dp_type,
            "content": content,
        }
        if dp_type == "send" and request.get("_dp_send_key") is not None:
            envelope["_dp_send_key"] = request["_dp_send_key"]
        if (
            dp_type == "register_destinations"
            and request.get("_dp_register_key") is not None
        ):
            envelope["_dp_register_key"] = request["_dp_register_key"]
        if dp_type == "wait_metadata" and request.get("_dp_metadata_key") is not None:
            envelope["_dp_metadata_key"] = request["_dp_metadata_key"]
    except Exception as e:
        logger.error(
            f"DP worker {dp_rank} error on {dp_type} "
            f"req_id={request.get('req_id', '?')}: {e}",
            exc_info=True,
        )
        err_code = int(getattr(e, "code", None) or HTTPStatus.INTERNAL_SERVER_ERROR)
        envelope = {
            "req_id": request.get("req_id", ""),
            "_dp_type": dp_type,
            "content": None,
            "_error": str(e),
            "_error_type": type(e).__name__,
            "_error_code": err_code,
        }
        if dp_type == "send" and request.get("_dp_send_key") is not None:
            envelope["_dp_send_key"] = request["_dp_send_key"]
        if (
            dp_type == "register_destinations"
            and request.get("_dp_register_key") is not None
        ):
            envelope["_dp_register_key"] = request["_dp_register_key"]
        if dp_type == "wait_metadata" and request.get("_dp_metadata_key") is not None:
            envelope["_dp_metadata_key"] = request["_dp_metadata_key"]


    try:
        async with send_lock:
            await async_sock_send(send_sock, wrap_as_pickle(envelope))
    except Exception:
        logger.error(
            f"DP worker {dp_rank} failed to send envelope for "
            f"req_id={request.get('req_id', '?')}",
            exc_info=True,
        )


async def run_dp_worker(
    server_args: ServerArgs,
    dp_rank: int,
    gpu_id: int,
    dispatch_path: str,
    result_path: str,
):
    logger.info(
        f"DP worker {dp_rank} starting on gpu_id={gpu_id} "
        f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})"
    )


    enc = MMEncoder(
        server_args,
        dist_init_method=f"tcp://127.0.0.1:{get_free_port()}",
        rank=0,
        gpu_id=gpu_id,
    )

    if get_observability().enable_metrics:
        set_prometheus_multiproc_dir()
        labels = {
            "model_name": get_serving().served_model_name,
            "dp_rank": str(dp_rank),
        }
        if get_observability().extra_metric_labels:
            labels.update(get_observability().extra_metric_labels)
        server_module.encoder_metrics_collector = EncoderMetricsCollector(labels)
        enc.dp_rank = dp_rank

    max_batch_size, coalesce_same_turn = _resolve_encoder_batch_policy(
        enc.model_type,
        ENCODER_MAX_BATCH_SIZE,
        ENCODER_MAX_BATCH_SIZE_EXPLICIT,
    )
    sched = EncoderScheduler(
        encoder=enc,
        send_sockets=[],
        max_batch_size=max_batch_size,
        coalesce_same_turn=coalesce_same_turn,
    )

    ctx = zmq.asyncio.Context(2)
    recv_sock = get_zmq_socket(ctx, zmq.PULL, dispatch_path, False)
    send_sock = get_zmq_socket(ctx, zmq.PUSH, result_path, False)
    send_lock = asyncio.Lock()
    inflight: Set[asyncio.Task] = set()


    max_inflight = envs.SGLANG_ENCODER_DP_WORKER_MAX_INFLIGHT.get()
    if max_inflight < max_batch_size:
        logger.warning(
            f"SGLANG_ENCODER_DP_WORKER_MAX_INFLIGHT={max_inflight} is below "
            f"the effective encoder max_batch_size={max_batch_size}; the encoder "
            f"will never assemble a full batch."
        )
    inflight_sem = asyncio.Semaphore(max_inflight)
    sched.start()
    logger.info(f"DP worker {dp_rank} ready")

    try:
        while True:
            await inflight_sem.acquire()
            spawned = False
            try:
                try:
                    request = await async_sock_recv(recv_sock)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.error(f"DP worker {dp_rank} recv error", exc_info=True)
                    continue
                if not isinstance(request, dict):
                    logger.error(
                        f"DP worker {dp_rank} received non-dict request "
                        f"({type(request).__name__}); dropping"
                    )
                    continue
                dp_type = request.pop("_dp_type", "encode")

                async def _run(req=request, t=dp_type):
                    try:
                        await _dp_worker_handle_request(
                            enc, sched, send_sock, send_lock, dp_rank, req, t
                        )
                    finally:
                        inflight_sem.release()

                task = asyncio.create_task(_run())
                spawned = True
                inflight.add(task)
                task.add_done_callback(inflight.discard)
            finally:
                if not spawned:
                    inflight_sem.release()
    finally:
        for task in inflight:
            task.cancel()
        ctx.destroy(linger=0)


def launch_dp_worker(
    server_args: ServerArgs,
    dp_rank: int,
    gpu_id: int,
    dispatch_path: str,
    result_path: str,
):
    publish(server_args, role="encoder")
    try:
        configure_logger(server_args, prefix=f" encode_dp_worker[{dp_rank}]")
        asyncio.run(
            run_dp_worker(server_args, dp_rank, gpu_id, dispatch_path, result_path)
        )
    except KeyboardInterrupt:
        logger.info(f"DP worker {dp_rank} exiting")
    except Exception:
        traceback.print_exc()


def launch_local_runtime(server_args: ServerArgs) -> EncoderRuntime:

    if get_parallel().config.dp_size > 1:
        raise ValueError(
            "launch_local_runtime requires --dp-size 1; got "
            f"dp_size={get_parallel().config.dp_size}."
        )


    if get_observability().enable_metrics:
        set_prometheus_multiproc_dir()
        labels = {
            "model_name": get_serving().served_model_name,
            "dp_rank": "0",
        }
        if get_observability().extra_metric_labels:
            labels.update(get_observability().extra_metric_labels)
        server_module.encoder_metrics_collector = EncoderMetricsCollector(labels)

    process_context = mp.get_context("spawn")
    zmq_context = zmq.Context(10)
    ipc_path_prefix = random_uuid()
    port_args = PortArgs.init_new(server_args)
    if get_parallel().config.dist_init_addr:
        dist_init_method = NetworkAddress.parse(
            get_parallel().config.dist_init_addr
        ).to_tcp()
    else:
        dist_init_method = NetworkAddress(
            get_serving().host or "127.0.0.1", port_args.nccl_port
        ).to_tcp()

    if get_observability().enable_trace:
        process_tracing_init(
            get_observability().otlp_traces_endpoint,
            "sglang",
            trace_modules=get_observability().trace_modules,
        )
        trace_set_thread_info("Encoder")

    send_sockets: List[zmq.Socket] = []
    tp_processes: List[mp.Process] = []
    for rank in range(1, get_parallel().config.tp_size):
        schedule_path = f"ipc:///tmp/{ipc_path_prefix}_schedule_{rank}"
        send_sockets.append(
            get_zmq_socket(zmq_context, zmq.PUSH, schedule_path, bind=False)
        )
        process = process_context.Process(
            target=launch_encoder,
            args=(server_args, schedule_path, dist_init_method, rank),
            daemon=True,
        )
        process.start()
        tp_processes.append(process)

    encoder = MMEncoder(server_args, dist_init_method=dist_init_method)
    max_batch_size, coalesce_same_turn = _resolve_encoder_batch_policy(
        encoder.model_type,
        ENCODER_MAX_BATCH_SIZE,
        ENCODER_MAX_BATCH_SIZE_EXPLICIT,
    )
    scheduler = EncoderScheduler(
        encoder,
        send_sockets,
        max_batch_size=max_batch_size,
        coalesce_same_turn=coalesce_same_turn,
    )
    return EncoderRuntime(
        encoder=encoder,
        scheduler=scheduler,
        send_sockets=send_sockets,
        zmq_context=zmq_context,
        tp_processes=tp_processes,
    )


def launch_dp_runtime(server_args: ServerArgs) -> DPDispatcher:

    if get_parallel().config.dp_size <= 1 or get_parallel().config.tp_size != 1:
        raise ValueError(
            "Encoder DP mode requires --dp-size > 1 and --tp-size 1; got "
            f"dp_size={get_parallel().config.dp_size}, tp_size={get_parallel().config.tp_size}."
        )
    dp_size = get_parallel().config.dp_size
    logger.info(f"Launching encoder in DP mode: dp_size={dp_size}")


    if get_observability().enable_metrics:
        set_prometheus_multiproc_dir()

    ctx = mp.get_context("spawn")
    ipc_prefix = random_uuid()
    async_zmq_ctx = zmq.asyncio.Context(dp_size + 1)

    result_path = f"ipc:///tmp/{ipc_prefix}_dp_result"
    result_socket = get_zmq_socket(async_zmq_ctx, zmq.PULL, result_path, True)
    dispatch_sockets: List[zmq.asyncio.Socket] = [
        get_zmq_socket(
            async_zmq_ctx, zmq.PUSH, f"ipc:///tmp/{ipc_prefix}_dp_dispatch_{r}", True
        )
        for r in range(dp_size)
    ]

    worker_processes: List[mp.Process] = []

    def _kill_workers():
        for process in worker_processes:
            if process.is_alive():
                process.kill()
        for process in worker_processes:
            process.join(timeout=5)


    atexit.register(_kill_workers)

    for dp_rank in range(dp_size):
        gpu_id = server_args.base_gpu_id + dp_rank


        with maybe_reindex_device_id(gpu_id) as gpu_id:
            process = ctx.Process(
                target=launch_dp_worker,
                args=(
                    server_args,
                    dp_rank,
                    gpu_id,
                    f"ipc:///tmp/{ipc_prefix}_dp_dispatch_{dp_rank}",
                    result_path,
                ),
                daemon=False,
            )
            process.start()
        worker_processes.append(process)

    labels = {"model_name": get_serving().served_model_name}
    if server_args.extra_metric_labels:
        labels.update(server_args.extra_metric_labels)
    return DPDispatcher(
        dp_size,
        dispatch_sockets,
        result_socket,
        worker_processes,
        enable_metrics=get_observability().enable_metrics,
        labels=labels,
    )
