import ctypes
import fcntl
import hashlib
import os
import pickle
import sys
import tempfile
import time
from functools import lru_cache
from getpass import getuser
from pathlib import Path
from typing import Hashable, TypeAlias

import cutlass
import cutlass.cute as cute
import tvm_ffi
from cutlass.cutlass_dsl import JitCompiledFunction

from sglang.kernels.ops.attention.flash_attn.cute.fa_logging import fa_log


for _lib_path in cute.runtime.find_runtime_libraries(enable_tvm_ffi=False):
    if Path(_lib_path).exists():
        ctypes.CDLL(_lib_path, mode=ctypes.RTLD_GLOBAL)

CompileKeyType: TypeAlias = tuple[Hashable, ...]
CallableFunction: TypeAlias = JitCompiledFunction | tvm_ffi.Function


CUTE_DSL_CACHE_ENABLED: bool = (
    os.getenv("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "0") == "1"
)


CUTE_DSL_CACHE_DIR: str | None = os.getenv("FLASH_ATTENTION_CUTE_DSL_CACHE_DIR", None)


def get_cache_path() -> Path:
    if CUTE_DSL_CACHE_DIR is not None:
        cache_dir = Path(CUTE_DSL_CACHE_DIR)
    else:
        cache_dir = (
            Path(tempfile.gettempdir()) / getuser() / "flash_attention_cute_dsl_cache"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@lru_cache(maxsize=1)
def _compute_source_fingerprint() -> str:

    cute_root = Path(__file__).resolve().parent
    h = hashlib.sha256()

    h.update(f"py{sys.version_info.major}.{sys.version_info.minor}".encode())
    h.update(f"cutlass={cutlass.__version__}".encode())
    h.update(f"tvm_ffi={tvm_ffi.__version__}".encode())

    for src in sorted(cute_root.rglob("*.py")):
        if not src.is_file():
            continue
        h.update(src.relative_to(cute_root).as_posix().encode())
        content = src.read_bytes()
        h.update(len(content).to_bytes(8, "little"))
        h.update(content)

    return h.hexdigest()


class FileLock:


    def __init__(
        self,
        lock_path: Path,
        exclusive: bool,
        timeout: float = 15,
        label: str = "",
    ):

        self.lock_path: Path = lock_path
        self.exclusive: bool = exclusive
        self.timeout: float = timeout
        self.label: str = label
        self._fd: int = -1

    @property
    def _lock_label(self) -> str:
        kind = "exclusive" if self.exclusive else "shared"
        return f"{kind} {self.label}" if self.label else kind

    def __enter__(self) -> "FileLock":
        open_flags = (
            os.O_WRONLY | os.O_CREAT if self.exclusive else os.O_RDONLY | os.O_CREAT
        )
        lock_type = fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH

        self._fd = os.open(str(self.lock_path), open_flags)

        deadline = time.monotonic() + self.timeout
        acquired = False
        while time.monotonic() < deadline:
            try:
                fcntl.flock(self._fd, lock_type | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError:
                time.sleep(0.1)
        if not acquired:
            os.close(self._fd)
            self._fd = None
            raise RuntimeError(
                f"Timed out after {self.timeout}s waiting for "
                f"{self._lock_label} lock: {self.lock_path}"
            )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None


class JITCache:


    def __init__(self):
        self.cache: dict[CompileKeyType, CallableFunction] = {}

    def __setitem__(self, key: CompileKeyType, fn: JitCompiledFunction) -> None:
        self.cache[key] = fn

    def __getitem__(self, key: CompileKeyType) -> CallableFunction:
        return self.cache[key]

    def __contains__(self, key: CompileKeyType) -> bool:
        return key in self.cache

    def clear(self) -> None:

        self.cache.clear()


class JITPersistentCache(JITCache):


    EXPORT_FUNCTION_PREFIX = "func"
    LOCK_TIMEOUT_SECONDS = 15

    def __init__(self, cache_path: Path):
        super().__init__()
        cache_path.mkdir(parents=True, exist_ok=True)
        self.cache_path: Path = cache_path

    def __setitem__(self, key: CompileKeyType, fn: JitCompiledFunction) -> None:
        JITCache.__setitem__(self, key, fn)
        self._try_export_to_storage(key, fn)

    def __getitem__(self, key: CompileKeyType) -> CallableFunction:

        self.__contains__(key)
        return JITCache.__getitem__(self, key)

    def __contains__(self, key: CompileKeyType) -> bool:


        if JITCache.__contains__(self, key):
            return True
        return self._try_load_from_storage(key)

    def _try_load_from_storage(self, key: CompileKeyType) -> bool:

        sha256_hex = self._key_to_hash(key)
        obj_path = self.cache_path / f"{sha256_hex}.o"
        with FileLock(
            self._lock_path(sha256_hex),
            exclusive=False,
            timeout=self.LOCK_TIMEOUT_SECONDS,
            label=sha256_hex,
        ):
            if obj_path.exists():
                fa_log(1, f"Loading compiled function from disk: {obj_path}")
                m = cute.runtime.load_module(str(obj_path), enable_tvm_ffi=True)
                fn = getattr(m, self.EXPORT_FUNCTION_PREFIX)
                JITCache.__setitem__(self, key, fn)
                return True
            else:
                fa_log(1, f"Cache miss on disk for key hash {sha256_hex}")
        return False

    def _try_export_to_storage(
        self, key: CompileKeyType, fn: JitCompiledFunction
    ) -> None:

        sha256_hex = self._key_to_hash(key)
        with FileLock(
            self._lock_path(sha256_hex),
            exclusive=True,
            timeout=self.LOCK_TIMEOUT_SECONDS,
            label=sha256_hex,
        ):
            obj_path = self.cache_path / f"{sha256_hex}.o"
            if obj_path.exists():

                fa_log(1, f"Skipping export, already on disk: {obj_path}")
                return
            fa_log(1, f"Exporting compiled function to disk: {obj_path}")
            fn.export_to_c(
                object_file_path=str(obj_path),
                function_name=self.EXPORT_FUNCTION_PREFIX,
            )
            fa_log(1, f"Successfully exported compiled function to disk: {obj_path}")

    def _key_to_hash(self, key: CompileKeyType) -> str:
        return hashlib.sha256(pickle.dumps(key)).hexdigest()

    def _lock_path(self, sha256_hex: str) -> Path:
        return self.cache_path / f"{sha256_hex}.lock"

    def clear(self) -> None:

        fa_log(1, f"Clearing persistent cache at {self.cache_path}")
        super().clear()
        for child in self.cache_path.iterdir():
            child.unlink()


def get_jit_cache(name: str | None = None) -> JITCache:

    if CUTE_DSL_CACHE_ENABLED:
        path = get_cache_path() / _compute_source_fingerprint()
        if name:
            path = path / name
        fa_log(1, f"Creating persistent JIT cache at {path}")
        return JITPersistentCache(path)
    else:
        fa_log(1, "Persistent cache disabled, using in-memory JIT cache")
        return JITCache()
