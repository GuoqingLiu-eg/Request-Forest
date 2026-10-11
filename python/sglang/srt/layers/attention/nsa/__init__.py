import warnings

warnings.warn(
    "sglang.srt.layers.attention.nsa is deprecated; "
    "use sglang.srt.layers.attention.dsa instead.",
    DeprecationWarning,
    stacklevel=2,
)
from sglang.srt.layers.attention.dsa import *
