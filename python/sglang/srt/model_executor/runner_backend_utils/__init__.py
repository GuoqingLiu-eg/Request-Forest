CUDA_GRAPH_CAPTURE_FAILED_MSG = (
    "Possible solutions:\n"
    "1. set --mem-fraction-static to a smaller value (e.g., 0.8 or 0.7)\n"
    "2. set --cuda-graph-max-bs-decode to a smaller value (e.g., 16)\n"
    "3. disable decode CUDA graph by --cuda-graph-backend-decode=disabled. "
    "(Not recommended. Huge performance loss)\n"
    "Open an issue on GitHub https://github.com/sgl-project/sglang/issues/new/choose \n"
)

PREFILL_CUDA_GRAPH_CAPTURE_FAILED_MSG = (
    "Fail when using backend: {backend} for prefill runner.\n"
    "Possible suggestions:\n"
    "{suggestions}"
    "Open an issue on GitHub https://github.com/sgl-project/sglang/issues/new/choose \n"
)
