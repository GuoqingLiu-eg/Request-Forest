import warnings

warnings.warn(
    "sglang.srt.layers.attention.nsa.nsa_backend_mtp_precompute is deprecated; "
    "use sglang.srt.layers.attention.dsa.dsa_backend_mtp_precompute instead.",
    DeprecationWarning,
    stacklevel=2,
)
from sglang.srt.layers.attention.dsa.dsa_backend_mtp_precompute import *
