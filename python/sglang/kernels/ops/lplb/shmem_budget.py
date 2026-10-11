from __future__ import annotations

from dataclasses import dataclass


_RUNTIME_PAD_BYTES = 256


_BYTES_PER_ELEM = 4


GPU_BUDGETS_BYTES: dict[str, int] = {
    "a100": 160 * 1024,
    "h100": 223 * 1024,
    "h200": 223 * 1024,
    "h20": 223 * 1024,
    "b200": 224 * 1024,
}


@dataclass(frozen=True)
class ShmemBreakdown:
    nc: int
    nv: int
    a_bytes: int
    c_bytes: int
    x_bytes: int
    ata_bytes: int
    rhs_bytes: int
    d_bytes: int
    pad_bytes: int

    @property
    def total_bytes(self) -> int:
        return (
            self.a_bytes
            + self.c_bytes
            + self.x_bytes
            + self.ata_bytes
            + self.rhs_bytes
            + self.d_bytes
            + self.pad_bytes
        )

    def as_kib(self) -> float:
        return self.total_bytes / 1024.0


def shmem_bytes(nc: int, nv: int, bytes_per_elem: int = _BYTES_PER_ELEM) -> int:

    return bytes_per_elem * (nc * nv + nc * nc + 3 * nv + nc) + _RUNTIME_PAD_BYTES


def breakdown(
    nc: int, nv: int, bytes_per_elem: int = _BYTES_PER_ELEM
) -> ShmemBreakdown:

    b = bytes_per_elem
    return ShmemBreakdown(
        nc=nc,
        nv=nv,
        a_bytes=b * nc * nv,
        c_bytes=b * nv,
        x_bytes=b * nv,
        ata_bytes=b * nc * nc,
        rhs_bytes=b * nc,
        d_bytes=b * nv,
        pad_bytes=_RUNTIME_PAD_BYTES,
    )


def gpu_budget_bytes(gpu: str) -> int:
    key = gpu.lower()
    if key not in GPU_BUDGETS_BYTES:
        raise ValueError(
            f"unknown gpu '{gpu}', expected one of {sorted(GPU_BUDGETS_BYTES)}"
        )
    return GPU_BUDGETS_BYTES[key]


def fits(nc: int, nv: int, gpu: str = "h100") -> bool:
    return shmem_bytes(nc, nv) <= gpu_budget_bytes(gpu)


def assert_fits(nc: int, nv: int, gpu: str = "h100") -> None:

    used = shmem_bytes(nc, nv)
    cap = gpu_budget_bytes(gpu)
    if used > cap:
        raise ValueError(
            f"fused IPM kernel needs {used/1024:.1f} KiB of shared memory for "
            f"NC={nc}, NV={nv}, but {gpu} allows {cap/1024:.1f} KiB/block. "
            f"Either reduce problem size or switch to a tiled design."
        )


def max_nc_for_nv(nv: int, gpu: str = "h100") -> int:

    cap = gpu_budget_bytes(gpu)
    b = _BYTES_PER_ELEM

    rhs = (cap - _RUNTIME_PAD_BYTES) / b - 3 * nv
    if rhs <= 0:
        return 0

    import math

    disc = (nv + 1) ** 2 + 4 * rhs
    nc_max = int((-(nv + 1) + math.sqrt(disc)) / 2.0)
    while nc_max > 0 and shmem_bytes(nc_max, nv) > cap:
        nc_max -= 1
    return max(nc_max, 0)


def report(nc: int, nv: int, gpu: str = "h100") -> str:

    bd = breakdown(nc, nv)
    cap = gpu_budget_bytes(gpu)
    status = "FITS" if bd.total_bytes <= cap else "OVER BUDGET"
    return (
        f"[shmem] NC={nc} NV={nv} gpu={gpu} | "
        f"A={bd.a_bytes/1024:.1f}K "
        f"ata={bd.ata_bytes/1024:.1f}K "
        f"rest={(bd.c_bytes+bd.x_bytes+bd.rhs_bytes+bd.d_bytes)/1024:.1f}K | "
        f"total={bd.total_bytes/1024:.1f}K / {cap/1024:.1f}K  {status}"
    )
