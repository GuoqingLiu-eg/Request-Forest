import warnings

warnings.warn(
    "sglang.srt.layers.attention.nsa_backend is deprecated; "
    "use sglang.srt.layers.attention.dsa_backend instead.",
    DeprecationWarning,
    stacklevel=2,
)
from sglang.srt.layers.attention.dsa_backend import *
from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend,
    DeepseekSparseAttnMultiStepBackend,
    DSAFlashMLAMetadata,
    DSAIndexerMetadata,
    DSAMetadata,
    NativeSparseAttnBackend,
    NativeSparseAttnMultiStepBackend,
    NSAFlashMLAMetadata,
    NSAIndexerMetadata,
    NSAMetadata,
)
