from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import pathlib
import shutil
import subprocess
import sysconfig
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as dist_version
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import msgspec

from sglang.kernels.jit.utils.arch import get_jit_cuda_arch
from sglang.kernels.jit.utils.common import cache_once, is_hip_runtime, is_musa_runtime
from sglang.kernels.jit.utils.compile import toolchain
from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH
from sglang.kernels.jit.utils.compile.spec import BuildSpec
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_DEPS_FILE = "sgl_deps.json"
_BUILD_KEY_PREFIX = "build-"
_DEPS_KEY_PREFIX = "deps-"
_KEY_HEX_LEN = 16


_MAX_LEAVES_SCANNED = 100


class _DepEntry(msgspec.Struct, frozen=True, array_like=True):


    root: str
    relpath: str
    digest: str


def _package_dir(package: str) -> Optional[pathlib.Path]:
    try:
        spec = importlib.util.find_spec(package)
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    if spec.origin is not None:
        return pathlib.Path(spec.origin).resolve().parent
    locations = list(spec.submodule_search_locations or [])
    return pathlib.Path(locations[0]).resolve() if locations else None


@cache_once
def _anchor_roots() -> Tuple[Tuple[str, pathlib.Path], ...]:

    candidates: List[Tuple[str, Optional[pathlib.Path]]] = [
        ("kernels", KERNEL_PATH),
        ("tvm_ffi", _package_dir("tvm_ffi")),
        ("pkg:flashinfer", _package_dir("flashinfer")),
        ("pkg:deep_gemm", _package_dir("deep_gemm")),
        ("pkg:nvidia", _package_dir("nvidia")),
        ("toolkit", toolchain.toolkit_home()),
        ("sitepkgs", pathlib.Path(sysconfig.get_paths()["purelib"])),
        ("sys", pathlib.Path("/usr")),
    ]


    roots = [
        (token, path.resolve())
        for token, path in candidates
        if path is not None and path.exists()
    ]
    roots.sort(key=lambda item: len(str(item[1])), reverse=True)
    return tuple(roots)


def _normalize_path(path: pathlib.Path) -> Tuple[str, str]:

    for token, root in _anchor_roots():
        try:
            return token, str(path.relative_to(root))
        except ValueError:
            continue
    return "abs", str(path)


def _resolve_path(*, root: str, relpath: str) -> Optional[pathlib.Path]:
    if root == "abs":
        return pathlib.Path(relpath)
    for token, base in _anchor_roots():
        if token == root:
            return base / relpath
    return None


_digest_cache: Dict[pathlib.Path, Optional[str]] = {}


def _file_digest(path: pathlib.Path) -> Optional[str]:

    cached = _digest_cache.get(path)
    if cached is not None or path in _digest_cache:
        return cached
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        digest = None
    _digest_cache[path] = digest
    return digest


def clear_digest_cache() -> None:

    _digest_cache.clear()


def _hash_parts(parts: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(repr(part).encode())
        digest.update(b"\0")
    return digest.hexdigest()[:_KEY_HEX_LEN]


_VERSIONED_PACKAGES = (
    "apache-tvm-ffi",
    "flashinfer-python",
    "deep_gemm",
    "nvidia-mathdx",
    "torch",
)


@cache_once
def _target_tag() -> str:

    if is_hip_runtime():
        return toolchain.gpu_arch_name().split(":")[0] or "unknown"
    arch = get_jit_cuda_arch()
    prefix = "mp" if is_musa_runtime() else "sm"
    return f"{prefix}{arch.major}{arch.minor}{arch.suffix}"


@cache_once
def _environment_fingerprint() -> str:

    if is_hip_runtime():
        target = f"hip:{toolchain.gpu_arch_name()}"
    else:
        arch = get_jit_cuda_arch()
        target = f"{'musa' if is_musa_runtime() else 'cuda'}:{arch.target_name}"

    compilers = []
    for path in (toolchain.device_compiler_path(), toolchain.host_compiler_path()):
        try:
            compilers.append(subprocess.check_output([path, "--version"], text=True))
        except (OSError, subprocess.SubprocessError) as error:
            logger.warning("Cannot fingerprint compiler %s: %s", path, error)
            compilers.append("unknown")

    versions: List[Tuple[str, str]] = []
    for name in _VERSIONED_PACKAGES:
        try:
            versions.append((name, dist_version(name)))
        except (PackageNotFoundError, ValueError):
            versions.append((name, "absent"))

    return _hash_parts([target, compilers, versions])


def compute_build_key(spec: BuildSpec, *, build_file: str) -> str:

    source_digests = [
        (
            _normalize_path(pathlib.Path(path).resolve()),
            _file_digest(pathlib.Path(path).resolve()),
        )
        for path in sorted(spec.sources)
    ]
    units = [
        (unit.filename if unit.source is None else "", unit.is_cuda, unit.source)
        for unit in spec.translation_units()
    ]
    return _hash_parts(
        [
            spec.module_args,
            tuple(source_digests),
            _normalize_text(build_file),
            _normalize_text(repr(units)),
            _environment_fingerprint(),
        ]
    )


def _normalize_text(text: str) -> str:

    for token, root in _anchor_roots():
        text = text.replace(str(root), f"<{token}>")
    return text


def build_key_dir(*, module_name: str, build_key: str) -> pathlib.Path:
    configured = envs.SGLANG_JIT_CACHE_DIR.get() or "~/.cache/sglang/jit"
    root = pathlib.Path(configured).expanduser()
    return root / _target_tag() / module_name / f"{_BUILD_KEY_PREFIX}{build_key}"


def _deps_key(entries: Sequence[_DepEntry]) -> str:
    return _hash_parts([(e.root, e.relpath, e.digest) for e in entries])


def _read_deps(leaf: pathlib.Path) -> Optional[List[_DepEntry]]:
    try:
        raw = (leaf / _DEPS_FILE).read_bytes()
    except OSError:
        return None
    try:
        return msgspec.json.decode(raw, type=List[_DepEntry])
    except msgspec.DecodeError:
        return None


def _refresh(
    entries: Sequence[_DepEntry],
) -> Tuple[Optional[List[_DepEntry]], List[str]]:

    current: List[_DepEntry] = []
    changed: List[str] = []
    for entry in entries:
        path = _resolve_path(root=entry.root, relpath=entry.relpath)
        digest = _file_digest(path) if path is not None else None
        if digest is None:
            return None, [f"{entry.root}:{entry.relpath}"]
        if digest != entry.digest:
            changed.append(f"{entry.root}:{entry.relpath}")
        current.append(_DepEntry(root=entry.root, relpath=entry.relpath, digest=digest))
    return current, changed


def find_prebuilt(*, scope: pathlib.Path, module_name: str) -> Optional[pathlib.Path]:

    try:
        leaves = sorted(
            (p for p in scope.iterdir() if p.name.startswith(_DEPS_KEY_PREFIX)),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return None
    if len(leaves) > _MAX_LEAVES_SCANNED:
        logger.debug(
            "JIT module %s has %d cached leaves; only the newest %d are considered.",
            module_name,
            len(leaves),
            _MAX_LEAVES_SCANNED,
        )

    reason: List[str] = []
    for leaf in leaves[:_MAX_LEAVES_SCANNED]:
        entries = _read_deps(leaf)
        if entries is None:
            continue
        current, changed = _refresh(entries)
        if current is None:
            reason = reason or changed
            continue
        if f"{_DEPS_KEY_PREFIX}{_deps_key(current)}" != leaf.name:
            reason = reason or changed
            continue
        candidate = leaf / f"{module_name}.so"
        if candidate.is_file():


            try:
                os.utime(leaf, None)
            except OSError:
                pass
            return candidate

    if reason:
        log = logger.info if envs.SGLANG_JIT_CACHE_DEBUG.get() else logger.debug
        log("Rebuilding JIT module %s: %s changed", module_name, reason[0])
    return None


def commit_build(
    spec: BuildSpec,
    *,
    scope: pathlib.Path,
    staging: pathlib.Path,
    dependencies: Sequence[pathlib.Path],
) -> Optional[pathlib.Path]:

    entries = _to_entries(dependencies=dependencies, build_dir=staging)
    if not _covers_direct_sources(entries=entries, direct_sources=spec.sources):
        logger.warning(
            "JIT module %s produced no usable dependency information; it will be "
            "rebuilt on every load. This is a build-rule problem, not a cache problem.",
            spec.module_name,
        )
        return None


    (staging / _DEPS_FILE).write_bytes(msgspec.json.encode(entries))
    leaf = _publish(
        staging=staging, leaf=scope / f"{_DEPS_KEY_PREFIX}{_deps_key(entries)}"
    )
    _prune(scope=scope, keep_newest=leaf)
    return leaf / f"{spec.module_name}.so"


def _prune(*, scope: pathlib.Path, keep_newest: pathlib.Path) -> None:

    keep = envs.SGLANG_JIT_CACHE_KEEP.get()
    if keep is None:
        return
    leaves = sorted(
        (
            path
            for path in scope.iterdir()
            if path.name.startswith(_DEPS_KEY_PREFIX) and path != keep_newest
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for stale in leaves[max(keep - 1, 0) :]:
        logger.debug("Pruning JIT build %s/%s", scope.name, stale.name)
        shutil.rmtree(stale, ignore_errors=True)


def _to_entries(
    *, dependencies: Sequence[pathlib.Path], build_dir: pathlib.Path
) -> List[_DepEntry]:

    seen: Dict[Tuple[str, str], _DepEntry] = {}
    for candidate in dependencies:
        path = (
            candidate.resolve()
            if candidate.is_absolute()
            else (build_dir / candidate).resolve()
        )
        if path.is_relative_to(build_dir) or not path.is_file():
            continue
        digest = _file_digest(path)
        if digest is None:
            continue
        root, relpath = _normalize_path(path)
        seen[(root, relpath)] = _DepEntry(root=root, relpath=relpath, digest=digest)
    return [seen[key] for key in sorted(seen)]


def _covers_direct_sources(
    *, entries: Sequence[_DepEntry], direct_sources: Sequence[str]
) -> bool:

    if not entries:
        return False
    recorded = {(entry.root, entry.relpath) for entry in entries}
    return all(
        _normalize_path(pathlib.Path(source).resolve()) in recorded
        for source in direct_sources
    )


def _publish(*, staging: pathlib.Path, leaf: pathlib.Path) -> pathlib.Path:

    leaf.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(staging, leaf)
    except OSError:
        if not leaf.is_dir():
            raise
        logger.debug("JIT leaf %s already published by another process", leaf.name)
    return leaf
