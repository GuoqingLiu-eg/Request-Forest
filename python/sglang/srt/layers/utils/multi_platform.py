import warnings

from sglang.kernels.fused_op import BaseFusedOp


class MultiPlatformOp(BaseFusedOp):


    def __init_subclass__(cls, **kwargs):
        warnings.warn(
            "MultiPlatformOp is deprecated; subclass "
            "sglang.kernels.fused_op.BaseFusedOp instead (RFC #29630).",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init_subclass__(**kwargs)

    def forward_native(self, *args, **kwargs):
        raise NotImplementedError

    def forward_cuda(self, *args, **kwargs):
        raise NotImplementedError

    def forward_hip(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)

    def forward_musa(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)

    def forward_npu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_xpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_hpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)
