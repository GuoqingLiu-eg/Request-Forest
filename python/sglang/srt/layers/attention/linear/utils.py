from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Optional

import msgspec

from sglang.srt.runtime_context import get_exec
from sglang.srt.utils.common import rank0_log

if TYPE_CHECKING:
    pass


class LinearAttnKernelBackend(Enum):
    TRITON = "triton"
    CUTEDSL = "cutedsl"
    NV_CUTEDSL = "nv_cutedsl"
    FLASHINFER = "flashinfer"
    FLASHKDA = "flashkda"
    NVIDIA_KDA = "nvidia_kda"
    PTX_KDA = "ptx_kda"
    HELION = "helion"
    INTEL_XPU = "intel_xpu"
    CUSTOM = "custom"

    @classmethod
    def _missing_(cls, value):
        return cls.CUSTOM

    def is_triton(self):
        return self == LinearAttnKernelBackend.TRITON

    def is_cutedsl(self):
        return self == LinearAttnKernelBackend.CUTEDSL

    def is_nv_cutedsl(self):
        return self == LinearAttnKernelBackend.NV_CUTEDSL

    def is_flashinfer(self):
        return self == LinearAttnKernelBackend.FLASHINFER

    def is_flashkda(self):
        return self == LinearAttnKernelBackend.FLASHKDA

    def is_nvidia_kda(self):
        return self == LinearAttnKernelBackend.NVIDIA_KDA

    def is_ptx_kda(self):
        return self == LinearAttnKernelBackend.PTX_KDA

    def is_helion(self):
        return self == LinearAttnKernelBackend.HELION

    def is_intel_xpu(self):
        return self == LinearAttnKernelBackend.INTEL_XPU

    def is_custom(self):
        return self == LinearAttnKernelBackend.CUSTOM


class LinearAttnBackends(msgspec.Struct, frozen=True):


    decode: LinearAttnKernelBackend
    prefill: LinearAttnKernelBackend
    verify: LinearAttnKernelBackend


def resolve_linear_attn_backends(
    prefill_default: Optional[str] = None,
) -> LinearAttnBackends:

    mamba = get_exec().mamba
    base = mamba.linear_attn_backend
    decode = LinearAttnKernelBackend(mamba.linear_attn_decode_backend or base)
    prefill = LinearAttnKernelBackend(
        mamba.linear_attn_prefill_backend or prefill_default or base
    )


    verify = mamba.linear_attn_verify_backend
    if verify is None:
        verify = decode.value if decode.is_flashinfer() else "triton"

    backends = LinearAttnBackends(
        decode=decode, prefill=prefill, verify=LinearAttnKernelBackend(verify)
    )
    rank0_log(
        f"Linear attention kernel backend: decode={backends.decode.value}, "
        f"prefill={backends.prefill.value}, verify={backends.verify.value}"
    )
    return backends


def build_verify_intermediate_state_indices(pool_size: int, device):

    import torch

    from sglang.srt.utils.common import get_eager_max_batch_size

    padded_bs = max(get_eager_max_batch_size(pool_size), pool_size)
    indices = torch.arange(pool_size, dtype=torch.int32, device=device)
    if padded_bs > pool_size:
        indices = torch.cat(
            [
                indices,
                torch.full(
                    (padded_bs - pool_size,),
                    pool_size,
                    dtype=torch.int32,
                    device=device,
                ),
            ]
        )
    return indices
