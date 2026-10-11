import warnings

warnings.warn(
    "sglang.srt.layers.attention.nsa.triton_kernel is deprecated; "
    "use sglang.kernels.ops.attention.dsa.triton_kernel instead.",
    DeprecationWarning,
    stacklevel=2,
)
from sglang.kernels.ops.attention.dsa.triton_kernel import *
