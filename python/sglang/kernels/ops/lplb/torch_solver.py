import logging

import torch

logger = logging.getLogger(__name__)


_BACKEND_CHECKED = False
_FUSED_AVAILABLE = False
_FUSED_SOLVE_IPM = None  # type: ignore[assignment]
_FUSED_WARMUP = None  # type: ignore[assignment]
_FUSED_ASSERT_FITS = None  # type: ignore[assignment]


def _init_fused_backend() -> None:

    global _BACKEND_CHECKED, _FUSED_AVAILABLE
    global _FUSED_SOLVE_IPM, _FUSED_WARMUP, _FUSED_ASSERT_FITS

    if _BACKEND_CHECKED:
        return
    _BACKEND_CHECKED = True

    if not torch.cuda.is_available():
        logger.info("LPLB fused solver disabled: CUDA not available")
        return

    cap = torch.cuda.get_device_capability()
    if cap[0] < 9:
        logger.info(
            f"LPLB fused solver disabled: GPU SM {cap[0]}.{cap[1]} < 9.0 "
            "(requires Hopper or newer)"
        )
        return

    try:
        from sglang.kernels.ops.lplb.cuda_solver import solve_ipm as fused_solve_ipm
        from sglang.kernels.ops.lplb.cuda_solver import warmup as fused_warmup
        from sglang.kernels.ops.lplb.shmem_budget import assert_fits
    except ImportError as e:
        logger.info(
            f"LPLB fused solver disabled: {e}. "
            "Install Math-DX cuBLASDx via `pip install nvidia-mathdx` "
            "or set MATHDX_HOME to an extracted archive."
        )
        return

    _FUSED_SOLVE_IPM = fused_solve_ipm
    _FUSED_WARMUP = fused_warmup
    _FUSED_ASSERT_FITS = assert_fits
    _FUSED_AVAILABLE = True
    logger.info("LPLB fused solver enabled (CUDA C++ via load_jit, cuBLASDx)")


def _unavailable_reason() -> str:
    if not torch.cuda.is_available():
        return "CUDA is not available"
    cap = torch.cuda.get_device_capability()
    if cap[0] < 9:
        return f"GPU SM {cap[0]}.{cap[1]} < 9.0 (requires Hopper or newer)"
    return (
        "Math-DX cuBLASDx headers not found — install via "
        "`pip install nvidia-mathdx` or set MATHDX_HOME"
    )


def warmup(nc: int, nv: int, num_iters: int = 5, device: str = "cuda") -> None:

    _init_fused_backend()
    if not _FUSED_AVAILABLE:
        raise RuntimeError(f"LPLB fused solver unavailable: {_unavailable_reason()}")
    _FUSED_ASSERT_FITS(nc, nv, gpu="h100")
    _FUSED_WARMUP(nc, nv, num_iters=num_iters, device=device)


def solve_ipm(
    A: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    num_iters: int = 5,
) -> torch.Tensor:

    nc, nv = A.shape
    assert b.shape == (nc,), f"b shape mismatch: {b.shape} vs ({nc},)"
    assert c.shape == (nv,), f"c shape mismatch: {c.shape} vs ({nv},)"

    _init_fused_backend()
    if not _FUSED_AVAILABLE:
        raise RuntimeError(f"LPLB fused solver unavailable: {_unavailable_reason()}")
    if not A.is_cuda:
        raise RuntimeError(
            f"LPLB fused solver requires CUDA tensors; got A on {A.device}."
        )
    if A.dtype != torch.float32:
        raise RuntimeError(
            f"LPLB fused solver requires float32; got A.dtype={A.dtype}."
        )
    return _FUSED_SOLVE_IPM(A, b, c, num_iters=num_iters)


def solve_ipm_torch_reference(
    A: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    num_iters: int = 5,
) -> torch.Tensor:

    nc, nv = A.shape
    assert b.shape == (nc,), f"b shape mismatch: {b.shape} vs ({nc},)"
    assert c.shape == (nv,), f"c shape mismatch: {c.shape} vs ({nv},)"

    x = torch.ones(nv, device=A.device, dtype=torch.float32)
    d_max = torch.tensor(0.0, device=A.device, dtype=torch.float32)
    for _ in range(num_iters):
        ax2 = A * (x * x).unsqueeze(0)
        ax2a = ax2 @ A.t()
        ax2c = ax2 @ c


        ax2a = ax2a + 1e-12 * torch.eye(nc, device=A.device, dtype=torch.float32)
        delta = torch.linalg.solve(ax2a, ax2c)
        r = A.t() @ delta
        d = x * (c - r)
        d_max = d.max()
        alpha = 0.999 / d_max if d_max > 1e-9 else torch.tensor(1.0, device=A.device)
        x = x * (1.0 - alpha * d)

    max_residual = (A @ x - b).abs().max()
    converged = (d_max < 0.1) and (0 <= x[-1] < 1e-4) and (max_residual < 0.05)
    if not converged:
        return torch.full((nv,), 0.5, device=A.device, dtype=torch.float32)
    return x
