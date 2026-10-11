from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.kernels.fused_op import BaseFusedOp, register_fused_op
from sglang.kernels.spec import (
    CapabilityRequirement,
    FormatSignature,
    KernelBackend,
)

if TYPE_CHECKING:
    import torch

_NORM_DTYPES = ("float16", "bfloat16")
_CUDA = frozenset({CapabilityRequirement.CUDA})
_HIP = frozenset({CapabilityRequirement.HIP})
_NPU = frozenset({CapabilityRequirement.NPU})


_NORM_PRIORITY = (
    KernelBackend.AOT,
    KernelBackend.JIT,
    KernelBackend.AITER,
    KernelBackend.TORCH_NPU,
    KernelBackend.TORCH,
)


class RMSNormOp(BaseFusedOp):


    op = "layernorm.rmsnorm"
    priority = _NORM_PRIORITY
    capabilities = {
        KernelBackend.AOT: _CUDA,
        KernelBackend.JIT: _CUDA,
        KernelBackend.AITER: _HIP,
        KernelBackend.TORCH_NPU: _NPU,
    }
    format_signature = FormatSignature(
        supported_dtypes=_NORM_DTYPES,
        description="out = (x / RMS(x)) * weight; returns tensor",
    )
    descriptions = {
        KernelBackend.AOT: "RMS normalization (sgl_kernel wheel).",
        KernelBackend.JIT: "RMS normalization (sglang.kernels.jit).",
        KernelBackend.AITER: "RMS normalization (aiter rmsnorm2d_fwd, ROCm).",
        KernelBackend.TORCH_NPU: "RMS normalization (torch_npu, Ascend).",
        KernelBackend.TORCH: "RMS normalization (pure-torch reference).",
    }

    def forward_native(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        out: Optional[torch.Tensor] = None,
        enable_pdl: Optional[bool] = None,
    ) -> torch.Tensor:
        import torch

        x = input.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        result = (x * weight).to(input.dtype)
        if out is None:
            return result
        out.copy_(result)
        return out

    def forward_aot(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        out: Optional[torch.Tensor] = None,
        enable_pdl: Optional[bool] = None,
    ) -> torch.Tensor:
        import sgl_kernel

        return sgl_kernel.rmsnorm(input, weight, eps, out, enable_pdl)

    def forward_jit(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        out: Optional[torch.Tensor] = None,
        enable_pdl: Optional[bool] = None,
    ) -> torch.Tensor:
        import torch

        from sglang.kernels.ops.layernorm.norm import rmsnorm as jit_rmsnorm

        if out is None:
            out = torch.empty_like(input)
        jit_rmsnorm(input, weight, out, eps)
        return out

    def forward_aiter(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        out: Optional[torch.Tensor] = None,
        enable_pdl: Optional[bool] = None,
    ) -> torch.Tensor:
        import torch
        from aiter import rmsnorm2d_fwd


        if out is None:
            out = torch.empty_like(input)
        rmsnorm2d_fwd(out, input, weight, eps)
        return out

    def forward_torch_npu(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        out: Optional[torch.Tensor] = None,
        enable_pdl: Optional[bool] = None,
    ) -> torch.Tensor:
        import torch_npu

        result = torch_npu.npu_rms_norm(input, weight, eps)[0]
        if out is None:
            return result
        out.copy_(result)
        return out


class FusedAddRMSNormOp(BaseFusedOp):


    op = "layernorm.fused_add_rmsnorm"
    priority = _NORM_PRIORITY
    capabilities = {
        KernelBackend.AOT: _CUDA,
        KernelBackend.JIT: _CUDA,
        KernelBackend.AITER: _HIP,
        KernelBackend.TORCH_NPU: _NPU,
    }
    format_signature = FormatSignature(
        supported_dtypes=_NORM_DTYPES,
        in_place=True,
        description="residual += x; x = RMSNorm(residual) * weight",
    )
    descriptions = {
        KernelBackend.AOT: (
            "Fused residual-add + RMS normalization (sgl_kernel wheel)."
        ),
        KernelBackend.JIT: (
            "Fused residual-add + RMS normalization (sglang.kernels.jit)."
        ),
        KernelBackend.AITER: ("Fused residual-add + RMS normalization (aiter, ROCm)."),
        KernelBackend.TORCH_NPU: (
            "Fused residual-add + RMS normalization (torch_npu, Ascend)."
        ),
        KernelBackend.TORCH: (
            "Fused residual-add + RMS normalization (pure-torch reference)."
        ),
    }

    def forward_native(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        enable_pdl: Optional[bool] = None,
    ) -> None:
        import torch

        acc = input.to(torch.float32) + residual.to(torch.float32)
        residual.copy_(acc.to(residual.dtype))
        variance = acc.pow(2).mean(dim=-1, keepdim=True)
        normed = acc * torch.rsqrt(variance + eps)
        input.copy_((normed * weight).to(input.dtype))

    def forward_aot(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        enable_pdl: Optional[bool] = None,
    ) -> None:
        import sgl_kernel

        return sgl_kernel.fused_add_rmsnorm(input, residual, weight, eps, enable_pdl)

    def forward_jit(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        enable_pdl: Optional[bool] = None,
    ) -> None:
        from sglang.kernels.ops.layernorm.norm import (
            fused_add_rmsnorm as jit_fused_add_rmsnorm,
        )

        return jit_fused_add_rmsnorm(input, residual, weight, eps)

    def forward_aiter(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        enable_pdl: Optional[bool] = None,
    ) -> None:
        import torch
        from aiter import rmsnorm2d_fwd_with_add


        out = torch.empty_like(input)
        residual_out = torch.empty_like(residual)
        rmsnorm2d_fwd_with_add(out, input, residual_out, residual, weight, eps)
        input.copy_(out)
        residual.copy_(residual_out)

    def forward_torch_npu(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        enable_pdl: Optional[bool] = None,
    ) -> None:
        import torch_npu


        out, _, residual_out = torch_npu.npu_add_rms_norm(residual, input, weight, eps)
        input.copy_(out)
        residual.copy_(residual_out)


class GemmaRMSNormOp(BaseFusedOp):


    op = "layernorm.gemma_rmsnorm"
    priority = _NORM_PRIORITY


    capabilities = {
        KernelBackend.AOT: _CUDA,
        KernelBackend.JIT: _HIP,
        KernelBackend.TORCH_NPU: _NPU,
    }
    format_signature = FormatSignature(
        supported_dtypes=_NORM_DTYPES,
        description="out = (x / RMS(x)) * (weight + 1); returns tensor",
    )
    descriptions = {
        KernelBackend.AOT: "Gemma-style RMS normalization (sgl_kernel wheel).",
        KernelBackend.JIT: (
            "Gemma-style RMS normalization (rocm-triton, sglang.kernels.jit)."
        ),
        KernelBackend.TORCH_NPU: ("Gemma-style RMS normalization (torch_npu, Ascend)."),
        KernelBackend.TORCH: "Gemma-style RMS normalization (pure-torch reference).",
    }

    def forward_native(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        out: Optional[torch.Tensor] = None,
        enable_pdl: Optional[bool] = None,
    ) -> torch.Tensor:
        import torch

        x = input.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        result = (x * (1.0 + weight.to(torch.float32))).to(input.dtype)
        if out is None:
            return result
        out.copy_(result)
        return out

    def forward_aot(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        out: Optional[torch.Tensor] = None,
        enable_pdl: Optional[bool] = None,
    ) -> torch.Tensor:
        import sgl_kernel

        return sgl_kernel.gemma_rmsnorm(input, weight, eps, out, enable_pdl)

    def forward_jit(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        out: Optional[torch.Tensor] = None,
        enable_pdl: Optional[bool] = None,
    ) -> torch.Tensor:
        from sglang.kernels.ops.layernorm.minimax_m3_rmsnorm import (
            gemma_rmsnorm as rocm_triton_gemma_rmsnorm,
        )

        result = rocm_triton_gemma_rmsnorm(input, weight, eps)
        if out is None:
            return result
        out.copy_(result)
        return out

    def forward_torch_npu(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        out: Optional[torch.Tensor] = None,
        enable_pdl: Optional[bool] = None,
    ) -> torch.Tensor:
        import torch_npu

        result = torch_npu.npu_gemma_rms_norm(input, weight, eps)[0]
        if out is None:
            return result
        out.copy_(result)
        return out


class GemmaFusedAddRMSNormOp(BaseFusedOp):


    op = "layernorm.gemma_fused_add_rmsnorm"
    priority = _NORM_PRIORITY


    capabilities = {
        KernelBackend.AOT: _CUDA,
        KernelBackend.JIT: _HIP,
    }
    format_signature = FormatSignature(
        supported_dtypes=_NORM_DTYPES,
        in_place=True,
        description="residual += x; x = GemmaRMSNorm(residual) * (weight + 1)",
    )
    descriptions = {
        KernelBackend.AOT: ("Gemma-style fused residual-add + RMS normalization."),
        KernelBackend.JIT: (
            "Gemma-style fused residual-add + RMS normalization "
            "(rocm-triton, sglang.kernels.jit)."
        ),
        KernelBackend.TORCH: (
            "Gemma-style fused residual-add + RMS normalization "
            "(pure-torch reference)."
        ),
    }

    def forward_native(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        enable_pdl: Optional[bool] = None,
    ) -> None:
        import torch

        acc = input.to(torch.float32) + residual.to(torch.float32)
        residual.copy_(acc.to(residual.dtype))
        variance = acc.pow(2).mean(dim=-1, keepdim=True)
        normed = acc * torch.rsqrt(variance + eps)
        input.copy_((normed * (1.0 + weight.to(torch.float32))).to(input.dtype))

    def forward_aot(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        enable_pdl: Optional[bool] = None,
    ) -> None:
        import sgl_kernel

        return sgl_kernel.gemma_fused_add_rmsnorm(
            input, residual, weight, eps, enable_pdl
        )

    def forward_jit(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        enable_pdl: Optional[bool] = None,
    ) -> None:
        from sglang.kernels.ops.layernorm.minimax_m3_rmsnorm import (
            gemma_fused_add_rmsnorm as rocm_triton_gemma_fused_add_rmsnorm,
        )


        norm_out, residual_out = rocm_triton_gemma_fused_add_rmsnorm(
            input, residual, weight, eps
        )
        input.copy_(norm_out)
        residual.copy_(residual_out)


_RMSNORM = register_fused_op(RMSNormOp(), __name__, "_RMSNORM")
_FUSED_ADD_RMSNORM = register_fused_op(
    FusedAddRMSNormOp(), __name__, "_FUSED_ADD_RMSNORM"
)
_GEMMA_RMSNORM = register_fused_op(GemmaRMSNormOp(), __name__, "_GEMMA_RMSNORM")
_GEMMA_FUSED_ADD_RMSNORM = register_fused_op(
    GemmaFusedAddRMSNormOp(), __name__, "_GEMMA_FUSED_ADD_RMSNORM"
)


def rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:

    return _RMSNORM(input, weight, eps, out, enable_pdl)


def fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:

    return _FUSED_ADD_RMSNORM(input, residual, weight, eps, enable_pdl)


def gemma_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:

    return _GEMMA_RMSNORM(input, weight, eps, out, enable_pdl)


def gemma_fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:

    return _GEMMA_FUSED_ADD_RMSNORM(input, residual, weight, eps, enable_pdl)


__all__ = [
    "RMSNormOp",
    "FusedAddRMSNormOp",
    "GemmaRMSNormOp",
    "GemmaFusedAddRMSNormOp",
    "rmsnorm",
    "fused_add_rmsnorm",
    "gemma_rmsnorm",
    "gemma_fused_add_rmsnorm",
]


from sglang.kernels.registry import register_kernel
from sglang.kernels.spec import KernelSpec


_PHASE25_KERNELS = [
    ("gemma4_fused_ops", "gemma4_fused_routing", "triton"),
    ("gemma4_fused_ops", "gemma_qkv_rmsnorm", "triton"),
    ("mhc_head", "fused_hc_head", "triton"),
]
for _mod, _fn, _bk in _PHASE25_KERNELS:
    register_kernel(
        KernelSpec(
            op=f"layernorm.{_fn}",
            backend=KernelBackend(_bk),
            target=f"sglang.kernels.ops.layernorm.{_mod}:{_fn}",
        )
    )
del _mod, _fn, _bk


for _fn in ("fused_dual_residual_rmsnorm", "fused_rmsnorm"):
    register_kernel(
        KernelSpec(
            op=f"layernorm.{_fn}",
            backend=KernelBackend.TRITON,
            target=f"sglang.kernels.ops.elementwise.elementwise:{_fn}",
        )
    )
del _fn
