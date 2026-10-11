# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from importlib.metadata import EntryPoint, entry_points

logger = logging.getLogger(__name__)

SERVE_BACKENDS_GROUP = "sglang.serve_backends"
SERVE_BACKEND_API_VERSION = 1
RESERVED_SERVE_BACKEND_NAMES = frozenset({"auto"})


class ServeBackendDetection(str, Enum):


    MATCH = "match"
    NO_MATCH = "no_match"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ServeRequest:


    argv: tuple[str, ...]
    model_path: str | None
    model_path_is_positional: bool = False


ServeBackendRunner = Callable[[ServeRequest], None]
ServeBackendDetector = Callable[[ServeRequest], ServeBackendDetection]


@dataclass(frozen=True)
class ServeBackend:


    api_version: int
    run: ServeBackendRunner
    detect: ServeBackendDetector | None = None
    requires_model_path: bool = True


@dataclass(frozen=True)
class RegisteredServeBackend:


    name: str
    backend: ServeBackend
    distribution: str | None = None


class ServeBackendRegistry:


    def __init__(self, builtins: Mapping[str, ServeBackend]) -> None:
        invalid_builtin_names = set(builtins) & RESERVED_SERVE_BACKEND_NAMES
        if invalid_builtin_names:
            names = ", ".join(sorted(invalid_builtin_names))
            raise ValueError(f"Reserved serve backend names cannot be used: {names}")

        self._builtins = dict(builtins)
        self._entry_points = self._discover_entry_points()
        self._loaded: dict[str, RegisteredServeBackend] = {
            name: RegisteredServeBackend(name=name, backend=backend)
            for name, backend in self._builtins.items()
        }

        reserved = (set(self._builtins) | RESERVED_SERVE_BACKEND_NAMES) & set(
            self._entry_points
        )
        if reserved:
            names = ", ".join(sorted(reserved))
            raise RuntimeError(
                "Out-of-tree serve backends cannot replace reserved or built-in "
                f"backends: {names}"
            )

    @staticmethod
    def _discover_entry_points() -> dict[str, list[EntryPoint]]:
        discovered: dict[str, list[EntryPoint]] = {}
        for entry_point in entry_points(group=SERVE_BACKENDS_GROUP):
            discovered.setdefault(entry_point.name, []).append(entry_point)
        return discovered

    @property
    def available_names(self) -> tuple[str, ...]:


        external_names = sorted(set(self._entry_points) - set(self._builtins))
        return (*self._builtins, *external_names)

    def get(self, name: str) -> RegisteredServeBackend:


        if name in self._loaded:
            return self._loaded[name]

        candidates = self._entry_points.get(name, [])
        if not candidates:
            available = ", ".join(("auto", *self.available_names))
            raise ValueError(
                f"Unknown serve backend {name!r}. Available values: {available}."
            )
        if len(candidates) > 1:
            providers = ", ".join(
                sorted(
                    self._entry_point_provider(candidate) for candidate in candidates
                )
            )
            raise RuntimeError(
                f"Multiple distributions register serve backend {name!r}: "
                f"{providers}. Uninstall one provider or choose another backend name."
            )

        entry_point = candidates[0]
        try:
            factory = entry_point.load()
            if not callable(factory):
                raise TypeError("the entry point must resolve to a callable factory")
            backend = factory()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load serve backend {name!r} from "
                f"{self._entry_point_provider(entry_point)}: {exc}"
            ) from exc

        if not isinstance(backend, ServeBackend):
            raise TypeError(
                f"Serve backend {name!r} factory returned {type(backend).__name__}; "
                "expected sglang.cli.serve_backends.ServeBackend."
            )
        if backend.api_version != SERVE_BACKEND_API_VERSION:
            raise RuntimeError(
                f"Serve backend {name!r} uses API version {backend.api_version}; "
                f"this SGLang release requires version {SERVE_BACKEND_API_VERSION}."
            )

        registered = RegisteredServeBackend(
            name=name,
            backend=backend,
            distribution=self._entry_point_distribution(entry_point),
        )
        self._loaded[name] = registered
        return registered

    def auto_detect(self, request: ServeRequest) -> RegisteredServeBackend:


        matches: list[RegisteredServeBackend] = []
        for name in self.available_names:
            if name == "llm":


                continue
            try:
                registered = self.get(name)
                detector = registered.backend.detect
                if detector is None:
                    continue
                result = detector(request)
                if not isinstance(result, ServeBackendDetection):
                    raise TypeError(
                        "detector must return ServeBackendDetection, got "
                        f"{type(result).__name__}"
                    )
                if result is ServeBackendDetection.MATCH:
                    matches.append(registered)
            except Exception as exc:


                logger.warning(
                    "Skipping automatic detection for serve backend %r: %s",
                    name,
                    exc,
                )

        if len(matches) > 1:
            names = ", ".join(match.name for match in matches)
            raise RuntimeError(
                f"Multiple serve backends matched this request: {names}. "
                "Select one explicitly with --model-type BACKEND."
            )
        if matches:
            return matches[0]
        return self.get("llm")

    @classmethod
    def _entry_point_distribution(cls, entry_point: EntryPoint) -> str | None:
        distribution = getattr(entry_point, "dist", None)
        return getattr(distribution, "name", None)

    @classmethod
    def _entry_point_provider(cls, entry_point: EntryPoint) -> str:
        return cls._entry_point_distribution(entry_point) or entry_point.value


__all__ = [
    "SERVE_BACKEND_API_VERSION",
    "SERVE_BACKENDS_GROUP",
    "ServeBackend",
    "ServeBackendDetection",
    "ServeBackendRegistry",
    "ServeRequest",
]
