import warnings

warnings.warn(
    "sglang.srt.layers.attention.nsa.dequant_k_cache is deprecated; "
    "use sglang.kernels.ops.attention.dsa.dequant_k_cache instead.",
    DeprecationWarning,
    stacklevel=2,
)
from sglang.kernels.ops.attention.dsa.dequant_k_cache import *
