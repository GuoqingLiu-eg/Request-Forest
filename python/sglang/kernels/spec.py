from __future__ import annotations

import importlib
from enum import Enum
from typing import Callable, ClassVar, FrozenSet, Optional, Tuple, Union

import msgspec


class KernelBackend(str, Enum):


    TORCH = "torch"
    TORCH_COMPILE = "torch_compile"
    TRITON = "triton"
    JIT = "jit"
    AOT = "aot"
    CUTE_DSL = "cute_dsl"
    FLYDSL = "flydsl"
    FLASHINFER = "flashinfer"
    DEEPGEMM = "deepgemm"
    AITER = "aiter"
    TORCH_NPU = "torch_npu"


class DeviceType(str, Enum):


    CUDA = "cuda"
    HIP = "hip"
    NPU = "npu"
    CPU = "cpu"


class PlatformInfo(msgspec.Struct, frozen=True):


    device_type: str = "cpu"
    cuda_arch_major: Optional[int] = None
    cuda_arch_minor: Optional[int] = None

    @property
    def device(self) -> DeviceType:
        try:
            return DeviceType(self.device_type)
        except (ValueError, TypeError):
            return DeviceType.CPU

    @property
    def is_cuda(self) -> bool:
        return self.device_type == "cuda"

    @property
    def is_hip(self) -> bool:
        return self.device_type == "hip"

    @classmethod
    def detect(cls) -> PlatformInfo:

        try:
            import torch
        except Exception:
            return cls()

        try:
            if torch.version.hip is not None and torch.cuda.is_available():
                return cls(device_type="hip")
            npu = getattr(torch, "npu", None)
            if npu is not None and npu.is_available():
                return cls(device_type="npu")
            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability()
                return cls(
                    device_type="cuda",
                    cuda_arch_major=major,
                    cuda_arch_minor=minor,
                )
        except Exception:
            pass
        return cls()


class CapabilityRequirement(msgspec.Struct, frozen=True):


    device: DeviceType
    min_cuda_arch: Optional[Tuple[int, int]] = None
    max_cuda_arch: Optional[Tuple[int, int]] = None


    CUDA: ClassVar[CapabilityRequirement]
    HIP: ClassVar[CapabilityRequirement]
    NPU: ClassVar[CapabilityRequirement]

    @classmethod
    def cuda(
        cls,
        min_sm: Optional[Tuple[int, int]] = None,
        max_sm: Optional[Tuple[int, int]] = None,
    ) -> CapabilityRequirement:

        return cls(device=DeviceType.CUDA, min_cuda_arch=min_sm, max_cuda_arch=max_sm)

    def is_satisfied_by(self, platform: PlatformInfo) -> bool:
        if self.device != platform.device:
            return False
        if self.device == DeviceType.CUDA and platform.cuda_arch_major is not None:
            arch = (platform.cuda_arch_major, platform.cuda_arch_minor or 0)
            if self.min_cuda_arch is not None and arch < self.min_cuda_arch:
                return False
            if self.max_cuda_arch is not None and arch > self.max_cuda_arch:
                return False
        return True


CapabilityRequirement.CUDA = CapabilityRequirement(device=DeviceType.CUDA)
CapabilityRequirement.HIP = CapabilityRequirement(device=DeviceType.HIP)
CapabilityRequirement.NPU = CapabilityRequirement(device=DeviceType.NPU)


def capabilities_satisfied(
    capabilities: Union[
        FrozenSet[CapabilityRequirement],
        Tuple[CapabilityRequirement, ...],
        CapabilityRequirement,
    ],
    platform: PlatformInfo,
) -> bool:

    if isinstance(capabilities, CapabilityRequirement):
        capabilities = (capabilities,)
    return (not capabilities) or any(c.is_satisfied_by(platform) for c in capabilities)


class FormatSignature(msgspec.Struct, frozen=True):


    supported_dtypes: Tuple[str, ...] = ()
    in_place: bool = False
    description: str = ""


class KernelSpec(msgspec.Struct, frozen=True):


    op: str
    backend: KernelBackend
    target: str
    capabilities: FrozenSet[CapabilityRequirement] = frozenset()
    format_signature: FormatSignature = msgspec.field(default_factory=FormatSignature)
    description: str = ""

    @property
    def group(self) -> str:
        return self.op.split(".", 1)[0]

    @property
    def name(self) -> str:
        return self.op.split(".", 1)[1] if "." in self.op else self.op

    def is_available(self, platform: PlatformInfo) -> bool:

        return capabilities_satisfied(self.capabilities, platform)

    def load(self) -> Callable:

        module_path, sep, attr = self.target.partition(":")
        if not sep or not attr:
            raise ValueError(
                f"KernelSpec.target must be 'module:attr', got {self.target!r}"
            )
        obj = importlib.import_module(module_path)
        for part in attr.split("."):
            obj = getattr(obj, part)
        return obj
