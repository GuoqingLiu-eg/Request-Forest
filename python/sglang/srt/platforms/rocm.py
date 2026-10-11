from sglang.srt.platforms.cuda import CudaDeviceMixin
from sglang.srt.platforms.device_mixin import PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform


class RocmDeviceMixin(CudaDeviceMixin):


    _enum: PlatformEnum = PlatformEnum.ROCM
    device_name: str = "rocm"


class RocmSRTPlatform(RocmDeviceMixin, SRTPlatform):
    pass
