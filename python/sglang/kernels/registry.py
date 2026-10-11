from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from sglang.kernels.spec import KernelBackend, KernelSpec


class KernelRegistry:


    def __init__(self) -> None:
        self._by_op: Dict[str, List[KernelSpec]] = defaultdict(list)

    def register(self, spec: KernelSpec) -> KernelSpec:

        existing = self._by_op[spec.op]
        for other in existing:
            if other.backend == spec.backend:
                if other != spec:
                    raise ValueError(
                        f"Conflicting kernel registration for op {spec.op!r}, "
                        f"backend {spec.backend.value!r}: "
                        f"{other.target!r} != {spec.target!r}"
                    )
                return spec
        existing.append(spec)
        return spec

    def get(self, op: str) -> List[KernelSpec]:

        return list(self._by_op.get(op, ()))

    def get_backend(self, op: str, backend: KernelBackend) -> KernelSpec:

        for spec in self._by_op.get(op, ()):
            if spec.backend == backend:
                return spec
        raise KeyError(f"No '{backend.value}' backend registered for op {op!r}")

    def has(self, op: str) -> bool:
        return bool(self._by_op.get(op))

    def ops(self) -> List[str]:

        return sorted(self._by_op.keys())

    def all_specs(self) -> List[KernelSpec]:
        specs: List[KernelSpec] = []
        for op in self.ops():
            specs.extend(self._by_op[op])
        return specs


registry = KernelRegistry()


def register_kernel(spec: KernelSpec) -> KernelSpec:

    return registry.register(spec)
