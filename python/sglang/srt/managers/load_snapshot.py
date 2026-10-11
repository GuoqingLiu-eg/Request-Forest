from __future__ import annotations

import fcntl
import hashlib
import logging
import mmap
import os
import struct
from contextlib import contextmanager
from typing import Optional

import msgspec
import msgspec.msgpack
import msgspec.structs

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel, get_serving
from sglang.srt.utils.network import is_zmq_endpoint_ipv6

logger = logging.getLogger(__name__)


def should_use_zmq() -> bool:

    return (
        get_parallel().config.enable_dp_attention and get_parallel().config.nnodes > 1
    ) or envs.SGLANG_LOAD_SNAPSHOT_USE_ZMQ.get()


_LOAD_AWARE_METHODS = frozenset({"total_requests", "total_tokens"})


def _tokenizer_load_snapshot_owner_caller() -> str:

    if get_serving().tokenizer_worker_num > 1:
        return "MultiTokenizerRouter"
    return "TokenizerManager"


def zmq_reader_owner(caller: str) -> bool:

    if not should_use_zmq():
        return False
    if get_parallel().config.node_rank != 0:
        return False
    if caller == "DataParallelController":
        return (
            get_parallel().config.dp_size > 1
            and get_parallel().config.load_balance_method.lower() in _LOAD_AWARE_METHODS
        )
    if get_parallel().config.dp_size > 1 and (
        get_parallel().config.load_balance_method.lower() in _LOAD_AWARE_METHODS
    ):
        return False
    return caller == _tokenizer_load_snapshot_owner_caller()


class MemoryMetrics(msgspec.Struct, array_like=True):


    weight_gb: float
    kv_cache_gb: float
    graph_gb: float
    token_capacity: int


class SpeculativeMetrics(msgspec.Struct, array_like=True):


    accept_length: float
    accept_rate: float


class LoRAMetrics(msgspec.Struct, array_like=True):


    slots_used: int
    slots_total: int
    utilization: float


class DisaggregationMetrics(msgspec.Struct, array_like=True):


    mode: str
    prefill_bootstrap_queue_reqs: int = 0
    prefill_inflight_queue_reqs: int = 0
    decode_prealloc_queue_reqs: int = 0
    decode_transfer_queue_reqs: int = 0
    decode_retracted_queue_reqs: int = 0
    kv_transfer_speed_gb_s: float = 0.0
    kv_transfer_latency_ms: float = 0.0


class QueueMetrics(msgspec.Struct, array_like=True):


    waiting: int
    grammar: int
    paused: int
    retracted: int
    prealloc_ready: int


_SECTION_FIELDS = frozenset(
    {"memory", "speculative", "lora", "disaggregation", "queues"}
)


class LoadSnapshot(msgspec.Struct, omit_defaults=True):


    timestamp: float = 0.0
    dp_rank: int = 0
    num_running_reqs: int = 0
    num_waiting_reqs: int = 0
    num_waiting_uncached_tokens: int = 0
    num_used_tokens: int = 0
    num_total_tokens: int = 0


    num_active_tokens: int = 0
    max_total_num_tokens: int = 0
    max_running_requests: int = 0
    token_usage: float = 0.0
    gen_throughput: float = 0.0
    cache_hit_rate: float = 0.0
    utilization: float = 0.0

    total_prefill_uncached_tokens: int = 0
    total_prefill_busy_us: int = 0

    decode_moments: Optional[list[float]] = None

    memory: Optional[MemoryMetrics] = None
    speculative: Optional[SpeculativeMetrics] = None
    lora: Optional[LoRAMetrics] = None
    disaggregation: Optional[DisaggregationMetrics] = None
    queues: Optional[QueueMetrics] = None

    VALID_SECTIONS = frozenset(
        {"core", "memory", "spec", "lora", "disagg", "queues", "all"}
    )

    def to_dict(self, include: Optional[set[str]] = None) -> dict:
        load = {key: getattr(self, key) for key in _CORE_KEYS}

        if include is None or "all" in include:
            include_all = True
        else:
            if not (include <= self.VALID_SECTIONS):
                raise ValueError(
                    f"Invalid include sections: {include - self.VALID_SECTIONS}. "
                    f"Valid options: {sorted(self.VALID_SECTIONS)}"
                )
            if include == {"core"}:
                return load
            include_all = False

        for field, include_name, section in (
            ("memory", "memory", self.memory),
            ("speculative", "spec", self.speculative),
            ("lora", "lora", self.lora),
            ("disaggregation", "disagg", self.disaggregation),
            ("queues", "queues", self.queues),
        ):
            if section is None or (not include_all and include_name not in include):
                continue
            load[field] = msgspec.structs.asdict(section)

        return load


_CORE_KEYS = tuple(
    f for f in LoadSnapshot.__struct_fields__ if f not in _SECTION_FIELDS
)


def _enc_hook(obj):

    to_item = getattr(obj, "item", None)
    if to_item is not None:
        return to_item()
    raise NotImplementedError(f"cannot encode {type(obj).__name__} in load snapshot")


snapshot_encoder = msgspec.msgpack.Encoder(enc_hook=_enc_hook)
snapshot_decoder = msgspec.msgpack.Decoder(LoadSnapshot)


MAGIC = b"SLNS"
VERSION = 2
HEADER_STRUCT = struct.Struct("<4sHHI")
SLOT_LEN_STRUCT = struct.Struct("<I")
SLOT_SIZE = 16 * 1024


@contextmanager
def file_lock(fd: int, lock_type: int):
    fcntl.flock(fd, lock_type)
    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)


def shm_path_for(ipc_name: str) -> str:
    name = os.path.basename(ipc_name.rstrip("/")) or "default"
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    digest = hashlib.blake2s(ipc_name.encode(), digest_size=4).hexdigest()
    return f"/dev/shm/sglang_loads_{safe_name}_{digest}.shm"


def file_size(dp_size: int, slot_size: int = SLOT_SIZE) -> int:
    return HEADER_STRUCT.size + dp_size * slot_size


def slot_offset(dp_rank: int, slot_size: int = SLOT_SIZE) -> int:
    return HEADER_STRUCT.size + dp_rank * slot_size


class ShmLoadSnapshotWriter:
    def __init__(
        self, path: str, dp_size: int, dp_rank: int, publish_interval: int = 1
    ):
        if dp_rank < 0 or dp_rank >= dp_size:
            raise ValueError(f"invalid dp_rank={dp_rank} for dp_size={dp_size}")
        self.publish_interval = max(1, publish_interval)
        self.publish_counter = 0

        self.path = path
        self.dp_size = dp_size
        self.dp_rank = dp_rank
        self.slot_size = SLOT_SIZE
        self.fd = -1
        size = file_size(dp_size, self.slot_size)

        self.fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            with file_lock(self.fd, fcntl.LOCK_EX):
                os.ftruncate(self.fd, size)
                self.mmap = mmap.mmap(self.fd, size, access=mmap.ACCESS_WRITE)
                HEADER_STRUCT.pack_into(
                    self.mmap, 0, MAGIC, VERSION, dp_size, self.slot_size
                )
                self._write_payload(LoadSnapshot(dp_rank=dp_rank))
        except Exception:
            if self.fd >= 0:
                os.close(self.fd)
            raise

    def write(self, snapshot: LoadSnapshot) -> None:
        if snapshot.dp_rank != self.dp_rank:
            raise ValueError(
                f"snapshot dp_rank={snapshot.dp_rank} does not match writer dp_rank={self.dp_rank}"
            )

        with file_lock(self.fd, fcntl.LOCK_EX):
            self._write_payload(snapshot)

    def _write_payload(self, snapshot: LoadSnapshot) -> None:
        payload = snapshot_encoder.encode(snapshot)
        max_payload_size = self.slot_size - SLOT_LEN_STRUCT.size
        if len(payload) > max_payload_size:
            raise ValueError(
                f"load snapshot payload size {len(payload)} exceeds slot payload "
                f"capacity {max_payload_size}"
            )

        offset = slot_offset(self.dp_rank, self.slot_size)
        payload_start = offset + SLOT_LEN_STRUCT.size
        payload_end = payload_start + len(payload)
        slot_end = offset + self.slot_size

        SLOT_LEN_STRUCT.pack_into(self.mmap, offset, 0)
        self.mmap[payload_start:payload_end] = payload
        self.mmap[payload_end:slot_end] = b"\0" * (slot_end - payload_end)
        SLOT_LEN_STRUCT.pack_into(self.mmap, offset, len(payload))

    def close(self) -> None:
        self.mmap.close()
        os.close(self.fd)


class ZmqLoadSnapshotWriter:


    def __init__(
        self, endpoint: str, dp_size: int, dp_rank: int, publish_interval: int = 1
    ):
        import zmq as _zmq

        if dp_rank < 0 or dp_rank >= dp_size:
            raise ValueError(f"invalid dp_rank={dp_rank} for dp_size={dp_size}")
        self.publish_interval = max(1, publish_interval)
        self.publish_counter = 0
        self.dp_size = dp_size
        self.dp_rank = dp_rank

        self._zmq = _zmq
        self._ctx = _zmq.Context.instance()
        self._socket = self._ctx.socket(_zmq.PUSH)
        if is_zmq_endpoint_ipv6(endpoint):
            self._socket.setsockopt(_zmq.IPV6, 1)
        self._socket.setsockopt(_zmq.LINGER, 0)
        self._socket.setsockopt(_zmq.CONFLATE, 1)
        self._socket.connect(endpoint)

    def write(self, snapshot: LoadSnapshot) -> None:
        if snapshot.dp_rank != self.dp_rank:
            raise ValueError(
                f"snapshot dp_rank={snapshot.dp_rank} does not match "
                f"writer dp_rank={self.dp_rank}"
            )
        try:
            self._socket.send(snapshot_encoder.encode(snapshot), self._zmq.NOBLOCK)
        except self._zmq.Again:
            pass

    def close(self) -> None:
        self._socket.close()


class ShmLoadSnapshotReader:
    def __init__(self, path: str, dp_size: int):
        self.path = path
        self.dp_size = dp_size
        self.mmap: Optional[mmap.mmap] = None
        self.fd: Optional[int] = None
        self.slot_size = SLOT_SIZE
        self._header_warning_logged = False
        self._attach()

    def _attach(self) -> bool:
        if self.mmap is not None:
            return True

        try:
            fd = os.open(self.path, os.O_RDONLY)
        except FileNotFoundError:
            return False

        size = os.fstat(fd).st_size
        if size < HEADER_STRUCT.size:
            os.close(fd)
            return False

        try:
            with file_lock(fd, fcntl.LOCK_SH):
                mapped = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
                magic, version, dp_size, slot_size = HEADER_STRUCT.unpack_from(
                    mapped, 0
                )
        except (OSError, ValueError):
            os.close(fd)
            return False

        if (
            magic != MAGIC
            or version != VERSION
            or dp_size != self.dp_size
            or slot_size < SLOT_LEN_STRUCT.size
            or size < file_size(self.dp_size, slot_size)
        ):
            mapped.close()
            os.close(fd)
            if not self._header_warning_logged:
                logger.warning("load shm header mismatch at %s", self.path)
                self._header_warning_logged = True
            return False

        self.mmap = mapped
        self.fd = fd
        self.slot_size = slot_size
        return True

    def read(self, dp_rank: int) -> Optional[LoadSnapshot]:
        if dp_rank < 0 or dp_rank >= self.dp_size:
            return None
        if not self._attach():
            return None

        assert self.fd is not None
        with file_lock(self.fd, fcntl.LOCK_SH):
            return self._read_slot(dp_rank)

    def _read_slot(self, dp_rank: int) -> Optional[LoadSnapshot]:
        assert self.mmap is not None
        offset = slot_offset(dp_rank, self.slot_size)
        (payload_len,) = SLOT_LEN_STRUCT.unpack_from(self.mmap, offset)
        max_payload_size = self.slot_size - SLOT_LEN_STRUCT.size
        if payload_len == 0 or payload_len > max_payload_size:
            return None

        payload_start = offset + SLOT_LEN_STRUCT.size
        payload_end = payload_start + payload_len
        try:
            return snapshot_decoder.decode(self.mmap[payload_start:payload_end])
        except Exception as e:
            logger.debug("load snapshot decode failed for rank %s: %s", dp_rank, e)
            return None

    def read_all(self) -> list[LoadSnapshot]:
        if not self._attach():
            return []

        assert self.fd is not None
        with file_lock(self.fd, fcntl.LOCK_SH):
            loads = []
            for r in range(self.dp_size):
                load = self._read_slot(r)
                if load is not None:
                    loads.append(load)
            return loads

    def close(self) -> None:
        if self.mmap is not None:
            self.mmap.close()
            self.mmap = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class ZmqShmLoadSnapshotReader:


    def __init__(self, endpoint: str, shm_path: str, dp_size: int):
        import zmq as _zmq

        self._zmq = _zmq
        self._ctx = _zmq.Context.instance()
        self._socket = self._ctx.socket(_zmq.PULL)
        if is_zmq_endpoint_ipv6(endpoint):
            self._socket.setsockopt(_zmq.IPV6, 1)
        self._socket.setsockopt(_zmq.LINGER, 0)
        self._socket.setsockopt(_zmq.CONFLATE, 1)
        self._socket.bind(endpoint)

        self._endpoint = endpoint
        self._shm_path = shm_path
        self.dp_size = dp_size
        self._shm_reader = ShmLoadSnapshotReader(shm_path, dp_size)
        self._shm_writers: dict[int, ShmLoadSnapshotWriter] = {}

    def _poll(self) -> None:

        latest: dict[int, LoadSnapshot] = {}
        while True:
            try:
                data = self._socket.recv(self._zmq.NOBLOCK)
            except self._zmq.Again:
                break
            try:
                snapshot = snapshot_decoder.decode(data)
                if 0 <= snapshot.dp_rank < self.dp_size:
                    latest[snapshot.dp_rank] = snapshot
            except Exception as e:
                logger.warning("load snapshot zmq decode failed: %s", e)

        for dp_rank, snapshot in latest.items():
            if dp_rank not in self._shm_writers:
                self._shm_writers[dp_rank] = ShmLoadSnapshotWriter(
                    self._shm_path, self.dp_size, dp_rank
                )
            try:
                self._shm_writers[dp_rank].write(snapshot)
            except Exception as e:
                logger.warning(
                    "load snapshot shm write failed for rank %d: %s", dp_rank, e
                )

    def fileno(self) -> int:

        return self._socket.getsockopt(self._zmq.FD)

    def poll(self) -> None:

        self._poll()

    def read(self, dp_rank: int) -> Optional[LoadSnapshot]:
        self._poll()
        return self._shm_reader.read(dp_rank)

    def read_all(self) -> list[LoadSnapshot]:
        self._poll()
        return self._shm_reader.read_all()

    def close(self) -> None:
        for w in self._shm_writers.values():
            w.close()
        self._shm_writers.clear()
        self._shm_reader.close()
        self._socket.close()
        if self._endpoint.startswith("ipc://"):
            try:
                os.unlink(self._endpoint[len("ipc://") :])
            except OSError:
                pass


def _zmq_addr_for(port_args) -> str:

    ipc_name = getattr(port_args, "load_collector_ipc_name", "")
    if ipc_name:
        return ipc_name
    safe = "".join(
        c if c.isalnum() or c in "._-" else "_" for c in port_args.instance_id
    )
    digest = hashlib.blake2s(port_args.instance_id.encode(), digest_size=4).hexdigest()
    return f"ipc:///tmp/sglang_load_collector_{safe}_{digest}.sock"


def create_load_snapshot_writer(
    port_args,
    dp_size: int,
    dp_rank: int,
    publish_interval: int = 1,
):

    if should_use_zmq():
        return ZmqLoadSnapshotWriter(
            _zmq_addr_for(port_args), dp_size, dp_rank, publish_interval
        )
    return ShmLoadSnapshotWriter(
        shm_path_for(port_args.instance_id), dp_size, dp_rank, publish_interval
    )


def create_load_snapshot_reader(port_args, caller: str):

    dp_size = get_parallel().config.dp_size
    if zmq_reader_owner(caller):
        return ZmqShmLoadSnapshotReader(
            _zmq_addr_for(port_args), shm_path_for(port_args.instance_id), dp_size
        )
    return ShmLoadSnapshotReader(shm_path_for(port_args.instance_id), dp_size)
