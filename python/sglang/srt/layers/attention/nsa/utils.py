import warnings

warnings.warn(
    "sglang.srt.layers.attention.nsa.utils is deprecated; "
    "use sglang.srt.layers.attention.dsa.utils instead.",
    DeprecationWarning,
    stacklevel=2,
)
from sglang.srt.layers.attention.dsa.utils import *
