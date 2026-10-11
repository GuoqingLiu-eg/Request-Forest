from typing import Literal

from sglang.srt.runtime_context import get_parallel

TensorTransportMode = Literal["cuda_ipc", "auto", "default"]


def determine_tensor_transport_mode() -> TensorTransportMode:

    if get_parallel().config.nnodes > 1:

        return "default"
    return "cuda_ipc"
