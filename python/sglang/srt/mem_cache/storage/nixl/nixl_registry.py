import logging
import threading
from contextlib import contextmanager
from typing import List, Optional

from .nixl_utils import NixlFileManager

logger = logging.getLogger(__name__)


def _buffer_sizes(buffers) -> Optional[List[int]]:

    if not buffers or not isinstance(buffers[0], tuple):
        return None
    return [b[1] for b in buffers]


class NixlRegistry:


    def __init__(
        self,
        agent,
        mem_type: str,
        file_manager: Optional[NixlFileManager] = None,
    ):
        self.agent = agent
        self.mem_type = mem_type
        self.file_manager = file_manager


        self._obj_devid_lock = threading.Lock()
        self._obj_devid_next = 1
        self.path_mode = mem_type == "FILE" and self._probe_path_mode()
        if mem_type == "FILE" and self.path_mode:
            logger.info("HiCacheNixl: path-mode FILE registration active.")
        elif mem_type == "FILE":

            logger.info(
                "HiCacheNixl: the installed NIXL build does not "
                "support path-mode FILE registration; using legacy "
                "fd registration."
            )

    @contextmanager
    def _open_files(self, paths: List[str], create: bool):

        fds: List[int] = []
        try:
            for path in paths:
                fd = self.file_manager.open_file(path, create=create)
                if fd is None:
                    yield None
                    return
                fds.append(fd)
            yield fds
        finally:
            for fd in fds:
                self.file_manager.close_file(fd)

    @contextmanager
    def _registered(self, items: List[tuple], mem_type: str):

        reg = None
        if items:
            reg_descs = self.agent.get_reg_descs(items, mem_type)
            if reg_descs is not None:
                try:
                    reg = self.agent.register_memory(reg_descs)
                except Exception as e:
                    logger.error(f"Failed to register memory of type {mem_type}: {e}")
        try:
            yield reg
        finally:
            if reg is not None:
                try:
                    self.agent.deregister_memory(reg)
                except Exception as e:
                    logger.debug("deregister_memory skipped: %s", e)

    def _probe_path_mode(self) -> bool:

        reg_descs = self.agent.get_reg_descs(
            [(0, 4096, 1, "rw:/nonexistent-nixl-probe")], "FILE"
        )
        if reg_descs is None:
            return False
        try:
            reg = self.agent.register_memory(reg_descs)
            if reg is not None:
                try:
                    self.agent.deregister_memory(reg)
                except Exception:
                    pass
            return False
        except Exception:
            return True

    @contextmanager
    def storage(self, buffers, keys, direction):

        sizes = _buffer_sizes(buffers)
        if sizes is None:
            yield None
            return

        if self.mem_type == "FILE":
            if self.path_mode:
                parts = ["rw", "create"] if direction == "WRITE" else ["ro"]
                if self.file_manager.use_direct_io:
                    parts.append("direct")
                spec = ",".join(parts)
                tuples = [
                    (0, sizes[i], i + 1, f"{spec}:{keys[i]}") for i in range(len(keys))
                ]
                with self._registered(tuples, "FILE") as reg:
                    if reg is None:
                        yield None
                        return
                    yield reg.trim()
            else:
                with self._open_files(keys, create=(direction == "WRITE")) as fds:
                    if fds is None:
                        yield None
                        return
                    tuples = [(0, sizes[i], fds[i], keys[i]) for i in range(len(keys))]
                    with self._registered(tuples, "FILE") as reg:
                        if reg is None:
                            yield None
                            return
                        yield self.agent.get_xfer_descs(
                            [(0, sizes[i], fds[i]) for i in range(len(fds))], "FILE"
                        )
        else:


            n = len(keys)
            with self._obj_devid_lock:
                base = self._obj_devid_next
                self._obj_devid_next += n
            dev_ids = list(range(base, base + n))
            tuples = [(0, sizes[i], dev_ids[i], keys[i]) for i in range(n)]
            with self._registered(tuples, "OBJ") as reg:
                if reg is None:
                    yield None
                    return
                yield self.agent.get_xfer_descs(
                    [(0, sizes[i], dev_ids[i]) for i in range(n)],
                    self.mem_type,
                )
