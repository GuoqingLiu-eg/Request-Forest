from __future__ import annotations

from typing import TYPE_CHECKING

import zmq

from sglang.srt.environ import envs
from sglang.srt.observability.req_time_stats import real_time
from sglang.srt.platforms import current_platform

if TYPE_CHECKING:
    from sglang.srt.managers.rust_server import RustServer


class IdleSleeper:


    def __init__(self, sockets):
        self.poller = zmq.Poller()
        self.last_empty_time = real_time()
        for s in sockets:
            self.poller.register(s, zmq.POLLIN)

        self.empty_cache_interval = envs.SGLANG_EMPTY_CACHE_INTERVAL.get()

    def maybe_sleep(self):
        self.poller.poll(1000)
        if (
            self.empty_cache_interval > 0
            and real_time() - self.last_empty_time > self.empty_cache_interval
        ):
            self.last_empty_time = real_time()
            current_platform.empty_cache()


class RustServerIdleSleeper:


    def __init__(self, rust_server: RustServer, timeout_ms: int = 1000):
        self.rust_server = rust_server
        self.timeout_ms = timeout_ms
        self.last_empty_time = real_time()
        self.empty_cache_interval = envs.SGLANG_EMPTY_CACHE_INTERVAL.get()

    def maybe_sleep(self):
        self.rust_server.wait_request(self.timeout_ms)
        if (
            self.empty_cache_interval > 0
            and real_time() - self.last_empty_time > self.empty_cache_interval
        ):
            self.last_empty_time = real_time()
            current_platform.empty_cache()
