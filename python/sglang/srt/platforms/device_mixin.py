import enum
import random
from typing import NamedTuple, Optional

import numpy as np
import torch

from sglang.srt.environ import envs


class PlatformEnum(enum.Enum):


    CUDA = enum.auto()
    ROCM = enum.auto()
    CPU = enum.auto()
    XPU = enum.auto()
    MUSA = enum.auto()
    NPU = enum.auto()
    TPU = enum.auto()
    MPS = enum.auto()
    OOT = enum.auto()
    UNSPECIFIED = enum.auto()


class CpuArchEnum(enum.Enum):


    X86 = enum.auto()
    ARM = enum.auto()
    UNSPECIFIED = enum.auto()


class DeviceCapability(NamedTuple):


    major: int
    minor: int

    def as_version_str(self) -> str:
        return f"{self.major}.{self.minor}"

    def to_int(self) -> int:

        assert 0 <= self.minor < 10
        return self.major * 10 + self.minor


_DEVICE_TO_DISTRIBUTED_BACKEND: dict[str, str] = {
    "cuda": "nccl",
    "xpu": "xccl",
    "hpu": "hccl",
    "cpu": "gloo",
    "npu": "hccl" if not envs.SGLANG_ZBAL_LOCAL_MEM_SIZE.get() > 0 else "zbal",
    "musa": "mccl",
}


class DeviceMixin:


    _enum: PlatformEnum = PlatformEnum.UNSPECIFIED
    device_name: str = "unknown"
    device_type: str = "cpu"


    def is_cuda(self) -> bool:
        return self._enum == PlatformEnum.CUDA

    def is_rocm(self) -> bool:
        return self._enum == PlatformEnum.ROCM

    def is_cpu(self) -> bool:
        return self._enum == PlatformEnum.CPU

    def is_xpu(self) -> bool:
        return self._enum == PlatformEnum.XPU

    def is_musa(self) -> bool:
        return self._enum == PlatformEnum.MUSA

    def is_npu(self) -> bool:
        return self._enum == PlatformEnum.NPU

    def is_tpu(self) -> bool:
        return self._enum == PlatformEnum.TPU

    def is_mps(self) -> bool:
        return self._enum == PlatformEnum.MPS

    def is_cuda_alike(self) -> bool:

        return self._enum in (
            PlatformEnum.CUDA,
            PlatformEnum.ROCM,
            PlatformEnum.MUSA,
        )

    def is_out_of_tree(self) -> bool:

        return self._enum == PlatformEnum.OOT


    def get_device_total_memory(self, device_id: int = 0) -> int:

        raise NotImplementedError

    def get_current_memory_usage(
        self, device: Optional["torch.device"] = None
    ) -> float:

        raise NotImplementedError

    def is_pin_memory_available(self, device=None) -> bool:

        return False


    def get_device(self, device_id: int = 0) -> str:

        raise NotImplementedError

    def set_device(self, device: "torch.device") -> None:

        raise NotImplementedError

    def get_device_name(self, device_id: int = 0) -> str:

        raise NotImplementedError

    def get_device_uuid(self, device_id: int = 0) -> str:

        raise NotImplementedError

    def get_device_capability(self, device_id: int = 0) -> Optional["DeviceCapability"]:

        raise NotImplementedError

    def empty_cache(self) -> None:

        pass

    def synchronize(self) -> None:

        pass


    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:

        raise NotImplementedError


    def get_torch_distributed_backend_str(self) -> str:

        return _DEVICE_TO_DISTRIBUTED_BACKEND.get(self.device_type, "gloo")

    def get_communicator_class(self) -> type | None:

        return None


    @classmethod
    def inference_mode(cls):

        return torch.inference_mode(mode=True)

    @classmethod
    def seed_everything(cls, seed: int | None = None) -> None:

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    def verify_quantization(self, quant: str) -> None:

        pass

    @classmethod
    def get_cpu_architecture(cls) -> "CpuArchEnum":

        import platform as _platform

        machine = _platform.machine().lower()
        if machine in ("x86_64", "amd64", "i386", "i686"):
            return CpuArchEnum.X86
        elif machine in ("arm64", "aarch64"):
            return CpuArchEnum.ARM
        return CpuArchEnum.UNSPECIFIED

    def get_torch_profiler_activity_str(self) -> str:

        raise NotImplementedError

    def get_torch_profiler_activity(self) -> torch.profiler.ProfilerActivity:

        raise NotImplementedError


    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device_name})"
