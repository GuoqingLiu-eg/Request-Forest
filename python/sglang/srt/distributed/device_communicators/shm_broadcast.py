# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/device_communicators/shm_broadcast.py

import logging
import os
import pickle
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from typing import List, Optional
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from zmq import IPV6  # type: ignore
from zmq import SUB, SUBSCRIBE, XPUB, XPUB_VERBOSE, Context  # type: ignore

from sglang.srt.environ import envs
from sglang.srt.utils.network import NetworkAddress, get_local_ip_auto, get_open_port
from sglang.srt.utils.stale_shm_cleanup import make_shm_name


SGLANG_RINGBUFFER_WARNING_INTERVAL = int(
    os.environ.get("SGLANG_RINGBUFFER_WARNING_INTERVAL", "60")
)

logger = logging.getLogger(__name__)


class ShmRingBuffer:

    def __init__(
        self,
        n_reader: int,
        max_chunk_bytes: int,
        max_chunks: int,
        name: Optional[str] = None,
    ):

        self.n_reader = n_reader
        self.metadata_size = 1 + n_reader
        self.max_chunk_bytes = max_chunk_bytes
        self.max_chunks = max_chunks
        self.total_bytes_of_buffer = (
            self.max_chunk_bytes + self.metadata_size
        ) * self.max_chunks
        self.data_offset = 0
        self.metadata_offset = self.max_chunk_bytes * self.max_chunks

        if name is None:

            self.is_creator = True
            self.shared_memory = shared_memory.SharedMemory(
                create=True,
                size=self.total_bytes_of_buffer,
                name=make_shm_name("mq"),
            )

            with memoryview(
                self.shared_memory.buf[self.metadata_offset :]
            ) as metadata_buffer:
                torch.frombuffer(metadata_buffer, dtype=torch.uint8).fill_(0)
        else:

            self.is_creator = False


            with patch(
                "multiprocessing.resource_tracker.register",
                lambda *args, **kwargs: None,
            ):
                try:
                    self.shared_memory = shared_memory.SharedMemory(name=name)
                    assert self.shared_memory.size == self.total_bytes_of_buffer
                except FileNotFoundError:


                    pass

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.n_reader,
                self.max_chunk_bytes,
                self.max_chunks,
                self.shared_memory.name,
            ),
        )

    def __del__(self):
        if hasattr(self, "shared_memory"):
            self.shared_memory.close()
            if self.is_creator:
                self.shared_memory.unlink()

    @contextmanager
    def get_data(self, current_idx: int):
        start = self.data_offset + current_idx * self.max_chunk_bytes
        end = start + self.max_chunk_bytes
        with memoryview(self.shared_memory.buf[start:end]) as buf:
            yield buf

    @contextmanager
    def get_metadata(self, current_idx: int):
        start = self.metadata_offset + current_idx * self.metadata_size
        end = start + self.metadata_size
        with memoryview(self.shared_memory.buf[start:end]) as buf:
            yield buf


@dataclass
class Handle:
    connect_ip: str
    local_reader_ranks: List[int] = field(default_factory=list)

    buffer: Optional[ShmRingBuffer] = None
    local_subscribe_port: Optional[int] = None
    remote_subscribe_port: Optional[int] = None


class MessageQueue:

    def __init__(
        self,
        n_reader,
        n_local_reader,
        local_reader_ranks: Optional[List[int]] = None,
        max_chunk_bytes: int = 1024 * 1024 * 10,
        max_chunks: int = 10,
        connect_ip: Optional[str] = None,
    ):
        if local_reader_ranks is None:
            local_reader_ranks = list(range(n_local_reader))
        else:
            assert len(local_reader_ranks) == n_local_reader
        self.n_local_reader = n_local_reader
        n_remote_reader = n_reader - n_local_reader
        self.n_remote_reader = n_remote_reader

        if connect_ip is None:
            connect_ip = (
                get_local_ip_auto("0.0.0.0") if n_remote_reader > 0 else "127.0.0.1"
            )

        context = Context()

        if n_local_reader > 0:


            self.buffer = ShmRingBuffer(n_local_reader, max_chunk_bytes, max_chunks)


            self.local_socket = context.socket(XPUB)


            self.local_socket.setsockopt(XPUB_VERBOSE, True)


            sglang_port = envs.SGLANG_PORT.get()
            if sglang_port is not None:
                local_subscribe_port = self.local_socket.bind_to_random_port(
                    "tcp://127.0.0.1", min_port=sglang_port, max_port=sglang_port + 8
                )
            else:
                local_subscribe_port = self.local_socket.bind_to_random_port(
                    "tcp://127.0.0.1"
                )
            logger.debug("Bound to tcp://127.0.0.1:%d", local_subscribe_port)
            self.current_idx = 0

        else:
            self.buffer = None  # type: ignore
            local_subscribe_port = None
            self.local_socket = None
            self.current_idx = -1

        if n_remote_reader > 0:


            self.remote_socket = context.socket(XPUB)
            self.remote_socket.setsockopt(XPUB_VERBOSE, True)
            remote_subscribe_port = get_open_port()
            na = NetworkAddress(connect_ip, remote_subscribe_port)
            if na.is_ipv6:
                self.remote_socket.setsockopt(IPV6, 1)
            address = na.to_tcp()
            logger.debug(f"class MessageQueue: Binding remote socket to {address=}")
            self.remote_socket.bind(address)

        else:
            remote_subscribe_port = None
            self.remote_socket = None

        self._is_writer = True
        self._is_local_reader = False
        self.local_reader_rank = -1

        self._is_remote_reader = False

        self.handle = Handle(
            connect_ip=connect_ip,
            local_reader_ranks=local_reader_ranks,
            buffer=self.buffer,
            local_subscribe_port=local_subscribe_port,
            remote_subscribe_port=remote_subscribe_port,
        )

        logger.debug("Message queue communication handle: %s", self.handle)

    def export_handle(self) -> Handle:
        return self.handle

    @staticmethod
    def create_from_handle(handle: Handle, rank) -> "MessageQueue":
        self = MessageQueue.__new__(MessageQueue)
        self.handle = handle
        self._is_writer = False

        context = Context()

        if rank in handle.local_reader_ranks:
            assert handle.buffer is not None
            self.buffer = handle.buffer
            self.current_idx = 0
            self.local_reader_rank = handle.local_reader_ranks.index(rank)
            self._is_local_reader = True
            self._is_remote_reader = False

            self.local_socket = context.socket(SUB)
            self.local_socket.setsockopt_string(SUBSCRIBE, "")
            socket_addr = f"tcp://127.0.0.1:{handle.local_subscribe_port}"
            logger.debug("Connecting to %s", socket_addr)
            self.local_socket.connect(socket_addr)

            self.remote_socket = None
        else:
            self.buffer = None  # type: ignore
            self.current_idx = -1
            self.local_reader_rank = -1
            self._is_local_reader = False
            self._is_remote_reader = True

            self.local_socket = None

            self.remote_socket = context.socket(SUB)
            self.remote_socket.setsockopt_string(SUBSCRIBE, "")
            na = NetworkAddress(handle.connect_ip, handle.remote_subscribe_port)
            if na.is_ipv6:
                self.remote_socket.setsockopt(IPV6, 1)
            socket_addr = na.to_tcp()
            logger.debug("Connecting to %s", socket_addr)
            self.remote_socket.connect(socket_addr)

        return self

    def wait_until_ready(self):

        if self._is_writer:


            for i in range(self.n_local_reader):

                self.local_socket.recv()
            if self.n_local_reader > 0:


                self.local_socket.send(b"READY")


            for i in range(self.n_remote_reader):

                self.remote_socket.recv()
            if self.n_remote_reader > 0:


                self.remote_socket.send(b"READY")
        elif self._is_local_reader:

            recv = self.local_socket.recv()
            assert recv == b"READY"
        elif self._is_remote_reader:

            recv = self.remote_socket.recv()
            assert recv == b"READY"

    @contextmanager
    def acquire_write(self):
        assert self._is_writer, "Only writers can acquire write"
        start_time = time.monotonic()
        n_warning = 1
        while True:
            with self.buffer.get_metadata(self.current_idx) as metadata_buffer:
                read_count = sum(metadata_buffer[1:])
                written_flag = metadata_buffer[0]
                if written_flag and read_count != self.buffer.n_reader:


                    os.sched_yield()


                    if (
                        time.monotonic() - start_time
                        > SGLANG_RINGBUFFER_WARNING_INTERVAL * n_warning
                    ):
                        logger.warning(
                            "No available block found in %s second. ",
                            SGLANG_RINGBUFFER_WARNING_INTERVAL,
                        )
                        n_warning += 1

                    continue


                metadata_buffer[0] = 0

                with self.buffer.get_data(self.current_idx) as buf:
                    yield buf


                for i in range(1, self.buffer.n_reader + 1):

                    metadata_buffer[i] = 0

                metadata_buffer[0] = 1
                self.current_idx = (self.current_idx + 1) % self.buffer.max_chunks
                break

    @contextmanager
    def acquire_read(self):
        assert self._is_local_reader, "Only readers can acquire read"
        start_time = time.monotonic()
        n_warning = 1
        while True:
            with self.buffer.get_metadata(self.current_idx) as metadata_buffer:
                read_flag = metadata_buffer[self.local_reader_rank + 1]
                written_flag = metadata_buffer[0]
                if not written_flag or read_flag:


                    os.sched_yield()


                    if (
                        time.monotonic() - start_time
                        > SGLANG_RINGBUFFER_WARNING_INTERVAL * n_warning
                    ):
                        logger.warning(
                            "No available block found in %s second. ",
                            SGLANG_RINGBUFFER_WARNING_INTERVAL,
                        )
                        n_warning += 1

                    continue


                with self.buffer.get_data(self.current_idx) as buf:
                    yield buf


                metadata_buffer[self.local_reader_rank + 1] = 1
                self.current_idx = (self.current_idx + 1) % self.buffer.max_chunks
                break

    def enqueue(self, obj):
        assert self._is_writer, "Only writers can enqueue"
        serialized_obj = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        if self.n_local_reader > 0:
            if len(serialized_obj) >= self.buffer.max_chunk_bytes:
                with self.acquire_write() as buf:
                    buf[0] = 1
                self.local_socket.send(serialized_obj)
            else:
                with self.acquire_write() as buf:
                    buf[0] = 0
                    buf[1 : len(serialized_obj) + 1] = serialized_obj
        if self.n_remote_reader > 0:
            self.remote_socket.send(serialized_obj)

    def dequeue(self):
        if self._is_local_reader:
            with self.acquire_read() as buf:
                overflow = buf[0] == 1
                if not overflow:


                    obj = pickle.loads(buf[1:])
            if overflow:
                recv = self.local_socket.recv()
                obj = pickle.loads(recv)
        elif self._is_remote_reader:
            recv = self.remote_socket.recv()
            obj = pickle.loads(recv)
        else:
            raise RuntimeError("Only readers can dequeue")
        return obj

    def broadcast_object(self, obj=None):
        if self._is_writer:
            self.enqueue(obj)
            return obj
        else:
            return self.dequeue()

    @staticmethod
    def create_from_process_group(
        pg: ProcessGroup, max_chunk_bytes, max_chunks, writer_rank=0
    ) -> "MessageQueue":
        group_rank = dist.get_rank(pg)
        group_world_size = dist.get_world_size(pg)
        global_ranks = dist.get_process_group_ranks(pg)

        from sglang.srt.distributed.parallel_state import in_the_same_node_as

        status = in_the_same_node_as(pg, source_rank=writer_rank)
        same_node_ranks = [i for i, s in enumerate(status) if s]
        n_reader = group_world_size - 1
        n_local_reader = len(same_node_ranks) - 1
        local_reader_ranks = [i for i in same_node_ranks if i != writer_rank]
        buffer_io: MessageQueue
        if group_rank == writer_rank:
            buffer_io = MessageQueue(
                n_reader=n_reader,
                n_local_reader=n_local_reader,
                local_reader_ranks=local_reader_ranks,
                max_chunk_bytes=max_chunk_bytes,
                max_chunks=max_chunks,
            )
            handle = buffer_io.export_handle()
            dist.broadcast_object_list(
                [handle], src=global_ranks[writer_rank], group=pg
            )
        else:
            recv = [None]
            dist.broadcast_object_list(recv, src=global_ranks[writer_rank], group=pg)
            handle = recv[0]  # type: ignore
            buffer_io = MessageQueue.create_from_handle(handle, group_rank)
        buffer_io.wait_until_ready()
        return buffer_io
