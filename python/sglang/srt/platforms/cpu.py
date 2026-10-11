import gc
import platform as _platform
from functools import cached_property
from typing import Optional

import psutil
import torch

from sglang.srt.platforms.device_mixin import (
    CpuArchEnum,
    DeviceCapability,
    DeviceMixin,
    PlatformEnum,
)
from sglang.srt.platforms.interface import SRTPlatform


class CpuDeviceMixin(DeviceMixin):


    _enum: PlatformEnum = PlatformEnum.CPU
    device_name: str = "cpu"
    device_type: str = "cpu"

    @cached_property
    def cpu_arch(self) -> CpuArchEnum:

        return self.get_cpu_architecture()

    def get_device_total_memory(self, device_id: int = 0) -> int:
        return int(psutil.virtual_memory().total)

    def get_current_memory_usage(
        self, device: Optional["torch.device"] = None
    ) -> float:

        vm = psutil.virtual_memory()
        return float(vm.total - vm.available)

    def get_device(self, local_rank: int) -> "torch.device":


        return torch.device("cpu")

    def set_device(self, device: "torch.device") -> None:


        torch.cpu.set_device(device)

    def get_device_name(self, device_id: int = 0) -> str:


        if self.cpu_arch == CpuArchEnum.ARM:
            return "cpu (aarch64)"
        if self.cpu_arch == CpuArchEnum.X86:
            return "cpu (x86_64)"
        return "cpu"

    def get_device_uuid(self, device_id: int = 0) -> str:


        return _platform.machine()

    def get_device_capability(self, device_id: int = 0) -> Optional[DeviceCapability]:
        return None

    def empty_cache(self) -> None:


        gc.collect()

    def synchronize(self) -> None:


        torch.cpu.synchronize()

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        vm = psutil.virtual_memory()
        return (vm.available, vm.total)

    def get_torch_distributed_backend_str(self) -> str:
        return "gloo"


class CpuSRTPlatform(CpuDeviceMixin, SRTPlatform):


    def is_pin_memory_available(self, device=None) -> bool:
        return False
