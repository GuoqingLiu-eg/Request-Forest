from __future__ import annotations

import dataclasses
import logging
import threading
import time
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from sglang.srt.runtime_context import (
    get_schedule,
)

logger = logging.getLogger(__name__)


STAGING_WATERMARK_WAIT_S = 0.001

if TYPE_CHECKING:
    from sglang.srt.disaggregation.decode import DecodeRequest


@dataclasses.dataclass
class DecodeStagingContext:


    allocator: object = None
    room_bootstrap: dict = dataclasses.field(default_factory=dict)
    room_receivers: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class PrefillStagingContext:


    buffers: list = dataclasses.field(default_factory=list)
    remote_watermarks: dict = dataclasses.field(default_factory=dict)
    watermark_cv: threading.Condition = dataclasses.field(
        default_factory=threading.Condition
    )

    prefetch_requested: set = dataclasses.field(default_factory=set)


    prefetched_rooms: set = dataclasses.field(default_factory=set)
    prefetch_sockets: dict = dataclasses.field(default_factory=dict)


class DecodeStagingHandler:


    def __init__(
        self,
        kv_manager,
        staging_allocator,
        kv_buffer_info: dict,
        decode_tp: int,
        total_kv_heads: int,
        tp_rank: int,
        scheduler,
    ):
        self.kv_manager = kv_manager
        self.staging_allocator = staging_allocator
        self.kv_buffer_info = kv_buffer_info
        self.decode_tp = decode_tp
        self.total_kv_heads = total_kv_heads
        self.tp_rank = tp_rank
        self.scheduler = scheduler


        from sglang.srt.environ import envs

        self.completion_timeout = float(
            envs.SGLANG_DISAGGREGATION_WAITING_TIMEOUT.get()
        )
        self._room_to_decode_req: dict = {}


        self._room_to_receiver: dict = {}
        self._wm_subscribers: dict = {}


        self._writer_counts: dict = {}

    def register_wm_subscriber(self, receiver, session_id: str) -> None:

        if receiver is None or not receiver.bootstrap_infos:
            return
        key = tuple(str(bi) for bi in receiver.bootstrap_infos)
        if key not in self._wm_subscribers:
            self._wm_subscribers[key] = (receiver, session_id)

    def num_writers_for(self, receiver) -> int:

        prefill_info = receiver.prefill_info
        prefill_tp = prefill_info.attn_tp_size
        if prefill_tp > self.decode_tp:
            tp_writers = prefill_tp // max(1, self.decode_tp)
        else:
            tp_writers = 1
        pp_writers = prefill_info.pp_size // self.kv_manager.pp_size
        return tp_writers * pp_writers

    @classmethod
    def create(cls, kv_manager, scheduler, tp_rank: int) -> DecodeStagingHandler:

        staging_allocator = kv_manager._staging_ctx.allocator
        if staging_allocator is None:
            raise RuntimeError(
                "Staging is enabled but kv_manager._staging_ctx.allocator is None. "
                "Check that the transfer backend correctly initializes the staging allocator."
            )
        kv_buffer_info = kv_manager.kv_buffer_tensors
        if kv_buffer_info is None:
            raise RuntimeError(
                "Staging is enabled but kv_manager.kv_buffer_tensors is None. "
                "Check that set_kv_buffer_tensors() was called during kv_manager init."
            )
        decode_tp = kv_manager.attn_tp_size

        from sglang.srt.disaggregation.common.staging_buffer import (
            resolve_total_kv_heads,
        )

        total_kv_heads = resolve_total_kv_heads(kv_manager.kv_args, decode_tp)
        return cls(
            kv_manager=kv_manager,
            staging_allocator=staging_allocator,
            kv_buffer_info=kv_buffer_info,
            decode_tp=decode_tp,
            total_kv_heads=total_kv_heads,
            tp_rank=tp_rank,
            scheduler=scheduler,
        )


    def register_decode_req(self, room: int, decode_req: DecodeRequest) -> None:

        decode_req._staging_all_success = False
        decode_req._staging_success_ts = 0.0
        decode_req._staging_failed = False
        decode_req._staging_scatter_done = False
        decode_req._chunk_events = []
        self._room_to_decode_req[room] = decode_req
        self._room_to_receiver[room] = decode_req.kv_receiver


        page_size = self.kv_buffer_info["page_size"]
        if decode_req.req.cache_protected_len % page_size != 0:
            logger.error(
                "[STAGING] decode prefix length %s is not page-aligned "
                "(page_size=%s); failing room=%s (staging scatter offsets "
                "would be wrong).",
                decode_req.req.cache_protected_len,
                page_size,
                room,
            )
            decode_req._staging_failed = True

    def unregister_decode_req(self, room: int) -> None:

        decode_req = self._room_to_decode_req.pop(room, None)
        receiver = self._room_to_receiver.pop(room, None)
        self._writer_counts.pop(room, None)
        if decode_req is not None:
            self.release_room(room, decode_req, receiver)
        self.kv_manager._staging_ctx.room_receivers.pop(room, None)
        self.kv_manager._staging_ctx.room_bootstrap.pop(room, None)

    def release_room(self, room: int, decode_req: DecodeRequest, receiver) -> None:


        stream = self.staging_allocator._scatter_stream
        if stream is not None:
            stream.synchronize()
        chunk_infos = receiver.chunk_staging_infos if receiver is not None else []
        unscattered_allocs = []
        for chunk_idx, info in enumerate(chunk_infos):
            if info[0] >= 0:
                unscattered_allocs.append((chunk_idx, info[0]))
                chunk_infos[chunk_idx] = (-1, -1, 0, -1, 0)
        for chunk_idx, alloc_id in unscattered_allocs:
            logger.warning(
                "[STAGING] releasing unscattered staging allocation "
                "room=%s chunk=%s alloc_id=%s",
                room,
                chunk_idx,
                alloc_id,
            )
            self._free_and_send_watermark(alloc_id, decode_req)
        for _event, alloc_id in decode_req._chunk_events:
            self._free_and_send_watermark(alloc_id, decode_req)
        decode_req._chunk_events.clear()


    def submit_chunk_scatter(
        self, room: int, chunk_idx: int, page_start: int, num_pages: int
    ) -> bool:

        decode_req = self._room_to_decode_req.get(room)
        if decode_req is None:
            logger.warning(
                "[STAGING] submit_chunk_scatter: room=%s not registered, "
                "chunk_idx=%s. This should not happen if register_decode_req "
                "is called at kv_receiver.init() time.",
                room,
                chunk_idx,
            )
            return False
        receiver = self._room_to_receiver.get(room)
        chunk_infos = receiver.chunk_staging_infos if receiver is not None else []
        if chunk_idx >= len(chunk_infos):
            return False
        alloc_id, staging_offset, _, _, _ = chunk_infos[chunk_idx]
        if staging_offset < 0 or alloc_id < 0:
            return False

        ok = self._scatter_region(
            staging_offset, page_start, num_pages, decode_req, receiver
        )
        if ok:
            event = torch.cuda.Event()
            event.record(self.staging_allocator._scatter_stream)


            decode_req._chunk_events.append((event, alloc_id))
            chunk_infos[chunk_idx] = (-1, -1, 0, -1, 0)
        else:
            logger.warning(
                "submit_chunk_scatter failed room=%s chunk_idx=%s tp_rank=%s",
                room,
                chunk_idx,
                self.tp_rank,
            )
        return ok

    def is_staging_room(self, room: int) -> bool:

        return room in self._room_to_decode_req

    def handle_chunk_arrived(
        self,
        room: int,
        chunk_idx: int,
        page_start: int,
        num_pages: int,
        writer_id: str,
    ) -> bool:


        receiver = self._room_to_receiver.get(room)
        if receiver is None:
            logger.warning(
                "Staging chunk arrived for unregistered room=%s chunk=%d, " "skipping",
                room,
                chunk_idx,
            )
            return False
        room_counts = self._writer_counts.setdefault(room, {})
        arrivals = room_counts.setdefault(chunk_idx, [])
        arrivals.append((page_start, num_pages, writer_id))
        num_writers = self.num_writers_for(receiver)
        if len(arrivals) >= num_writers:
            self.submit_chunk_scatter(room, chunk_idx, page_start, num_pages)
            del room_counts[chunk_idx]
            return True
        return False

    def submit_last_scatter_async(self, room: int) -> bool:

        decode_req = self._room_to_decode_req.get(room)
        if decode_req is None:
            logger.warning(
                "[STAGING] submit_last_scatter_async: room=%s not registered. "
                "This should not happen if register_decode_req is called at "
                "kv_receiver.init() time.",
                room,
            )
            return False
        if not decode_req._staging_all_success:


            decode_req._staging_success_ts = time.monotonic()
            decode_req._staging_all_success = True
        return True


    def is_done(self, decode_req: DecodeRequest) -> bool:

        return decode_req._staging_scatter_done and not decode_req._chunk_events

    def is_failed(self, decode_req: DecodeRequest) -> bool:

        return decode_req._staging_failed

    def advance_scatter(self, decode_req: DecodeRequest) -> None:

        chunk_events = decode_req._chunk_events
        if chunk_events:
            for i in range(len(chunk_events) - 1, -1, -1):
                event, alloc_id = chunk_events[i]
                if event.query():
                    chunk_events.pop(i)
                    self._free_and_send_watermark(alloc_id, decode_req)

        if not decode_req._staging_all_success:
            return
        room = decode_req.req.bootstrap_room
        receiver = self._room_to_receiver.get(room)
        chunk_infos = receiver.chunk_staging_infos if receiver is not None else []
        incomplete = bool(chunk_events) or any(info[0] >= 0 for info in chunk_infos)
        if not incomplete:
            decode_req._staging_scatter_done = True
            return
        elapsed = time.monotonic() - decode_req._staging_success_ts
        if elapsed > self.completion_timeout:
            logger.error(
                "[STAGING] room=%s not complete %.0fs after all-ranks Success "
                "(a scatter never arrived); failing the request.",
                room,
                elapsed,
            )
            decode_req._staging_failed = True


    def _scatter_region(
        self,
        staging_offset: int,
        page_start: int,
        num_pages: int,
        decode_req: DecodeRequest,
        receiver,
    ) -> bool:

        from sglang.srt.disaggregation.common.staging_buffer import (
            scatter_staging_to_kv,
        )

        k_buffers = self.kv_buffer_info["k_buffers"]
        v_buffers = self.kv_buffer_info["v_buffers"]
        page_size = self.kv_buffer_info["page_size"]
        dst_tp_rank = self.kv_manager.kv_args.engine_rank % self.decode_tp

        device = k_buffers[0].device
        torch.cuda.set_device(device)

        if self.staging_allocator._scatter_stream is None:
            self.staging_allocator._scatter_stream = torch.cuda.Stream(device=device)

        scatter_stream = self.staging_allocator._scatter_stream

        staging_view = self.staging_allocator.buffer.buffer[staging_offset:]

        req_pool_idx = decode_req.req.req_pool_idx


        prefix_tokens = decode_req.req.cache_protected_len
        token_start = prefix_tokens + page_start * page_size
        token_end = token_start + num_pages * page_size
        prefill_tp = receiver.prefill_info.attn_tp_size

        with torch.cuda.stream(scatter_stream):
            kv_indices = self.scheduler.req_to_token_pool.req_to_token[
                req_pool_idx, token_start:token_end
            ]
            if page_size > 1:
                page_idx_tensor = kv_indices[::page_size] // page_size
            else:
                page_idx_tensor = kv_indices

            scatter_staging_to_kv(
                staging_view,
                k_buffers,
                v_buffers,
                page_idx_tensor,
                page_size,
                prefill_tp,
                self.decode_tp,
                dst_tp_rank,
                self.total_kv_heads,
            )

        return True

    def _free_and_send_watermark(
        self, alloc_id: int, decode_req: DecodeRequest
    ) -> None:

        self.staging_allocator.free(alloc_id)
        post_wm = self.staging_allocator.get_watermark()
        room = decode_req.req.bootstrap_room
        wm_round, wm_tail = post_wm
        wm_round_b = str(wm_round).encode("ascii")
        wm_tail_b = str(wm_tail).encode("ascii")
        for _key, (receiver, session_id) in list(self._wm_subscribers.items()):
            sid_b = session_id.encode("ascii")
            for bootstrap_info in receiver.bootstrap_infos:
                try:
                    sock, lock = receiver._connect_to_bootstrap_server(bootstrap_info)
                    with lock:
                        sock.send_multipart(
                            [b"WATERMARK", wm_round_b, wm_tail_b, sid_b]
                        )
                except Exception:
                    pass


def is_watermark_ready(
    staging_state, session_id: str, alloc_round: int, alloc_end: int
) -> bool:

    if alloc_round <= 0:
        return True
    prev_round = alloc_round - 1
    wm_round, wm_tail = staging_state.remote_watermarks.get(session_id, (0, 0))
    return prev_round < wm_round or (prev_round == wm_round and alloc_end <= wm_tail)


def handle_watermark_msg(staging_ctx, msg_parts) -> None:

    wm_round = int(msg_parts[1].decode("ascii"))
    wm_tail = int(msg_parts[2].decode("ascii"))
    wm_session = msg_parts[3].decode("ascii") if len(msg_parts) > 3 else ""
    with staging_ctx.watermark_cv:
        prev = staging_ctx.remote_watermarks.get(wm_session, (0, 0))
        if (wm_round, wm_tail) > prev:
            staging_ctx.remote_watermarks[wm_session] = (
                wm_round,
                wm_tail,
            )
        staging_ctx.watermark_cv.notify_all()


def handle_staging_rsp(msg_parts, transfer_infos: dict) -> None:

    stg_room = int(msg_parts[1].decode("ascii"))
    stg_chunk_idx = int(msg_parts[2].decode("ascii"))
    stg_offset = int(msg_parts[3].decode("ascii"))
    stg_round = int(msg_parts[4].decode("ascii"))
    stg_end = int(msg_parts[5].decode("ascii"))
    stg_session = msg_parts[6].decode("ascii")
    room_infos = transfer_infos.get(stg_room, {})
    tinfo = room_infos.get(stg_session)
    if tinfo is not None:
        if tinfo.staging is None:
            tinfo.staging = StagingTransferInfo()
        tinfo.staging.set_chunk(stg_chunk_idx, stg_offset, stg_round, stg_end)
    else:
        logger.warning(
            "STAGING_RSP RECV but tinfo=None room=%s chunk=%d session=%s",
            stg_room,
            stg_chunk_idx,
            stg_session,
        )


@dataclasses.dataclass
class StagingTransferInfo:


    offsets: List[int] = dataclasses.field(default_factory=lambda: [-1])
    rounds: List[int] = dataclasses.field(default_factory=lambda: [0])
    ends: List[int] = dataclasses.field(default_factory=lambda: [-1])

    def set_chunk(self, idx: int, offset: int, rnd: int, end: int):
        while len(self.offsets) <= idx:
            self.offsets.append(-1)
            self.rounds.append(0)
            self.ends.append(-1)
        self.offsets[idx] = offset
        self.rounds[idx] = rnd
        self.ends[idx] = end


class PrefillStagingStrategy:


    def __init__(self, kv_manager, staging_buffer):
        from sglang.srt.disaggregation.common.staging_buffer import (
            staging_grid_tokens,
        )

        self.kv_manager = kv_manager
        self.staging_buffer = staging_buffer
        page_size = kv_manager.kv_buffer_tensors["page_size"]
        self.full_chunk_pages = (
            staging_grid_tokens(get_schedule().chunked_prefill_size, page_size)
            // page_size
        )

    def check_ready(
        self,
        req,
        kv_chunk_index_start: int,
        num_chunk_pages: int,
        session_id: Optional[str] = None,
    ) -> Tuple[bool, int, int, int, int]:

        from sglang.srt.disaggregation.common.staging_buffer import StagingAllocator

        chunk_idx = (
            kv_chunk_index_start // self.full_chunk_pages
            if self.full_chunk_pages > 0
            else 0
        )

        stg = req.staging
        if stg is None or chunk_idx >= len(stg.offsets):
            return (False, chunk_idx, -1, 0, -1)

        c_offset = stg.offsets[chunk_idx]
        if c_offset == StagingAllocator.ALLOC_OVERSIZED:
            return (False, chunk_idx, StagingAllocator.ALLOC_OVERSIZED, 0, -1)
        if c_offset < 0:
            return (False, chunk_idx, -1, 0, -1)

        c_round = stg.rounds[chunk_idx]
        c_end = stg.ends[chunk_idx]

        if session_id is None:
            session_id = req.mooncake_session_id
        if not self.kv_manager._is_watermark_ready(session_id, c_round, c_end):
            return (False, chunk_idx, c_offset, c_round, c_end)

        return (True, chunk_idx, c_offset, c_round, c_end)

    def transfer(
        self,
        session_id: str,
        prefill_kv_indices,
        dst_staging_ptr: int,
        dst_staging_size: int,
        target_info,
    ) -> int:

        try:
            return self.kv_manager.send_kvcache_staged(
                session_id,
                prefill_kv_indices,
                dst_staging_ptr,
                dst_staging_size,
                target_info.dst_tp_rank,
                target_info.dst_attn_tp_size,
                target_info.dst_kv_item_len,
                target_info.dst_kv_layer_ids,
                staging_buffer=self.staging_buffer,
            )
        except Exception as e:
            raise RuntimeError(
                f"[Staging] KV transfer via staging buffer failed: {e}. "
                f"session={session_id}"
            ) from e


def _get_custom_mem_pool(device: str):

    from sglang.srt.disaggregation.mooncake.utils import (
        init_mooncake_custom_mem_pool,
    )

    _, custom_mem_pool, pool_type = init_mooncake_custom_mem_pool(device)
    if custom_mem_pool is None:
        logger.info(
            "Staging buffer using cudaMalloc (no custom mem pool). "
            "This works for all GPU architectures. "
            "For NVLink/MNNVL transport, set SGLANG_MOONCAKE_CUSTOM_MEM_POOL."
        )
    return custom_mem_pool, pool_type


def init_staging_buffers(
    register_fn, kv_args, count: int, chunked_prefill_size: int
) -> list:

    from sglang.srt.disaggregation.common.staging_buffer import StagingBuffer

    full_chunk_pages = max(1, chunked_prefill_size // kv_args.page_size)
    size_bytes = full_chunk_pages * sum(kv_args.kv_item_lens)
    gpu_id = kv_args.gpu_id
    device = f"cuda:{gpu_id}"

    custom_mem_pool, _ = _get_custom_mem_pool(device)

    buffers = []
    for _ in range(count):
        buf = StagingBuffer(size_bytes, device, gpu_id, custom_mem_pool=custom_mem_pool)
        register_fn(buf.get_ptr(), buf.get_size())
        buffers.append(buf)
    return buffers


def init_staging_allocator(register_fn, kv_args):

    from sglang.srt.disaggregation.common.staging_buffer import StagingAllocator
    from sglang.srt.environ import envs

    pool_size_mb = envs.SGLANG_DISAGG_STAGING_POOL_SIZE_MB.get()
    pool_size_bytes = pool_size_mb * 1024 * 1024
    gpu_id = kv_args.gpu_id
    device = f"cuda:{gpu_id}"

    custom_mem_pool, _ = _get_custom_mem_pool(device)
    allocator = StagingAllocator(pool_size_bytes, device, gpu_id, custom_mem_pool)
    register_fn(allocator.get_base_ptr(), allocator.get_total_size())
    return allocator


def handle_staging_req(
    msg,
    staging_allocator,
    kv_args,
    attn_tp_size: int,
    prefill_attn_tp_size: int,
    kv_buffer_tensors,
    room_receivers: dict,
    room_bootstrap: dict,
):

    from sglang.srt.disaggregation.common.staging_buffer import StagingAllocator

    room = int(msg[1].decode("ascii"))
    chunk_idx = int(msg[2].decode("ascii"))
    chunk_num_pages = int(msg[3].decode("ascii"))
    session_id = msg[4].decode("ascii")
    requester_pp_rank = int(msg[5].decode("ascii")) if len(msg) > 5 else None

    if staging_allocator is None:
        logger.warning(
            "STAGING_REQ ignored: allocator is None room=%s chunk=%s",
            room,
            chunk_idx,
        )
        return

    receiver = room_receivers.get(room)
    if receiver is None:
        logger.warning(
            "STAGING_REQ dropped: no receiver for room=%s chunk=%s session=%s",
            room,
            chunk_idx,
            session_id,
        )
        return
    infos = receiver.chunk_staging_infos

    if chunk_idx < len(infos) and infos[chunk_idx][0] >= 0:
        _, offset, rnd, end, _ = infos[chunk_idx]
    elif (
        chunk_idx < len(infos)
        and infos[chunk_idx][1] == StagingAllocator.ALLOC_OVERSIZED
    ):
        offset, rnd, end = StagingAllocator.ALLOC_OVERSIZED, 0, -1
    else:
        from sglang.srt.disaggregation.common.staging_buffer import (
            compute_staging_layout,
            resolve_total_kv_heads,
        )

        page_size = kv_args.page_size
        kv_item_lens = kv_args.kv_item_lens
        num_kv_layers = len(kv_item_lens) // 2
        decode_bytes_per_token = kv_item_lens[0] // page_size
        total_kv_heads = resolve_total_kv_heads(kv_args, attn_tp_size)
        dst_heads_per_rank = max(1, total_kv_heads // max(1, attn_tp_size))
        bytes_per_head_per_token = decode_bytes_per_token // dst_heads_per_rank
        dst_tp_rank = kv_args.engine_rank % max(1, attn_tp_size)

        chunk_tokens = chunk_num_pages * page_size
        _, _, required = compute_staging_layout(
            prefill_attn_tp_size,
            attn_tp_size,
            dst_tp_rank,
            total_kv_heads,
            chunk_tokens,
            bytes_per_head_per_token,
            num_kv_layers,
        )
        result = staging_allocator.assign(required)
        if result is None:
            logger.error(
                "[STAGING_REQ] alloc failed room=%s chunk=%d (need %d bytes, "
                "buffer total=%d bytes). Increase SGLANG_DISAGG_STAGING_POOL_SIZE_MB.",
                room,
                chunk_idx,
                required,
                staging_allocator.total_size,
            )
            offset, rnd, end = StagingAllocator.ALLOC_OVERSIZED, 0, -1
            while len(infos) <= chunk_idx:
                infos.append((-1, -1, 0, -1, 0))
            infos[chunk_idx] = (
                -1,
                StagingAllocator.ALLOC_OVERSIZED,
                0,
                -1,
                chunk_num_pages,
            )
        else:
            alloc_id, offset, rnd = result
            end = offset + required
            while len(infos) <= chunk_idx:
                infos.append((-1, -1, 0, -1, 0))
            infos[chunk_idx] = (alloc_id, offset, rnd, end, chunk_num_pages)

    bootstrap_infos = room_bootstrap.get(room)
    if bootstrap_infos:
        for bi in bootstrap_infos:
            if requester_pp_rank is not None and bi["pp_rank"] != requester_pp_rank:
                continue
            try:
                sock, lock = receiver._connect_to_bootstrap_server(bi)
                with lock:
                    sock.send_multipart(
                        [
                            b"STAGING_RSP",
                            str(room).encode("ascii"),
                            str(chunk_idx).encode("ascii"),
                            str(offset).encode("ascii"),
                            str(rnd).encode("ascii"),
                            str(end).encode("ascii"),
                            session_id.encode("ascii"),
                        ]
                    )
            except Exception:
                pass


class StagingManagerMixin:


    def _is_watermark_ready(
        self, session_id: str, alloc_round: int, alloc_end: int
    ) -> bool:
        return is_watermark_ready(self._staging_ctx, session_id, alloc_round, alloc_end)

    def _handle_staging_req(self, msg):
        room = int(msg[1].decode("ascii"))
        session_id = msg[4].decode("ascii")
        handler = self._staging_handler
        assert (
            handler is not None
        ), "STAGING_REQ received before staging handler initialized"
        decode_req = handler._room_to_decode_req.get(room)
        if decode_req is None:
            logger.warning(
                "STAGING_REQ received for unregistered room=%s, skipping",
                room,
            )
            return
        prefill_tp = decode_req.kv_receiver.prefill_info.attn_tp_size
        handle_staging_req(
            msg,
            self._staging_ctx.allocator,
            self.kv_args,
            self.attn_tp_size,
            prefill_tp,
            getattr(self, "kv_buffer_tensors", None),
            self._staging_ctx.room_receivers,
            self._staging_ctx.room_bootstrap,
        )

        receiver = self._staging_ctx.room_receivers.get(room)
        if receiver is not None:
            handler.register_wm_subscriber(receiver, session_id)


def prefetch_staging_reqs(
    room: int,
    transfer_infos: dict,
    kv_buffer_tensors: dict,
    chunked_prefill_size: int,
    staging_requested: set,
    prefetch_sockets: dict,
    requester_pp_rank: Optional[int] = None,
) -> None:

    import zmq

    from sglang.srt.disaggregation.common.staging_buffer import staging_grid_tokens
    from sglang.srt.utils.network import NetworkAddress

    page_size = kv_buffer_tensors["page_size"]
    full_chunk_pages = staging_grid_tokens(chunked_prefill_size, page_size) // page_size

    for session_id, tinfo in transfer_infos[room].items():


        is_dummy_attr = tinfo.is_dummy
        if is_dummy_attr() if callable(is_dummy_attr) else is_dummy_attr:
            continue
        total_pages = len(tinfo.dst_kv_indices)
        if total_pages == 0:
            continue
        num_chunks = (total_pages + full_chunk_pages - 1) // full_chunk_pages

        for chunk_idx in range(num_chunks):
            stg_key = (room, chunk_idx, session_id)
            if stg_key in staging_requested:
                continue
            staging_requested.add(stg_key)

            remaining = total_pages - chunk_idx * full_chunk_pages
            chunk_pages = min(full_chunk_pages, remaining)
            try:
                na = NetworkAddress(tinfo.endpoint, tinfo.dst_port)
                ep = na.to_tcp()
                if ep not in prefetch_sockets:
                    sock = zmq.Context().socket(zmq.PUSH)
                    if na.is_ipv6:
                        sock.setsockopt(zmq.IPV6, 1)
                    sock.connect(ep)
                    prefetch_sockets[ep] = sock
                request = [
                    b"STAGING_REQ",
                    str(room).encode("ascii"),
                    str(chunk_idx).encode("ascii"),
                    str(chunk_pages).encode("ascii"),
                    session_id.encode("ascii"),
                ]
                if requester_pp_rank is not None:
                    request.append(str(requester_pp_rank).encode("ascii"))
                prefetch_sockets[ep].send_multipart(request)
            except Exception:
                staging_requested.discard(stg_key)
