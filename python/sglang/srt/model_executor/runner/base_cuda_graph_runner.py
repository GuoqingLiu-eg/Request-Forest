# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from __future__ import annotations

import bisect
import gc
import logging
from abc import abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, List, Sequence, Tuple

from sglang.srt.model_executor.runner.base_runner import BaseRunner
from sglang.srt.runtime_context import (
    get_exec,
    get_flags,
)
from sglang.srt.utils import (
    get_cuda_graph_batch_size_alignment,
    get_cuda_graph_max_batch_size,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.input_buffers import ForwardInputBuffers
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.runner_backend.base_cuda_graph_backend import (
        BaseCudaGraphBackend,
    )

logger = logging.getLogger(__name__)


@contextmanager
def freeze_gc(enable_cudagraph_gc: bool):

    gc.collect()
    should_freeze = not enable_cudagraph_gc
    if should_freeze:
        gc.freeze()
    try:
        yield
    finally:
        if should_freeze:
            gc.unfreeze()
            gc.collect()


def get_batch_sizes_to_capture(
    model_runner: ModelRunner, captured_req_width: int = 1
) -> Tuple[List[int], List[int]]:


    capture_bs = list(get_exec().graph.cuda_graph_config.decode.bs)
    num_max_requests = model_runner.req_to_token_pool.size

    mul_base = get_cuda_graph_batch_size_alignment()


    alignment_width = captured_req_width
    if get_exec().overlap.enable_two_batch_overlap:
        alignment_width = 1


    num_max_requests = get_cuda_graph_max_batch_size(num_max_requests)
    if max(capture_bs) > num_max_requests:


        capture_bs += [num_max_requests]


    from sglang.srt.speculative.request_forest.adaptive_config import (
        adaptive_native_max_batch_size,
        adaptive_rf_enabled,
    )

    if model_runner.spec_algorithm.is_dflash() and adaptive_rf_enabled():
        num_max_requests = adaptive_native_max_batch_size(
            model_runner.req_to_token_pool.size
        )

        capture_bs += [num_max_requests]


    capture_bs = [bs for bs in capture_bs if bs * alignment_width % mul_base == 0]
    capture_bs = [bs for bs in capture_bs if bs <= num_max_requests]
    capture_bs = list(sorted(set(capture_bs)))

    assert len(capture_bs) > 0 and capture_bs[0] > 0, f"{capture_bs=}"
    compile_bs = (
        [bs for bs in capture_bs if bs <= get_exec().graph.torch_compile_max_bs]
        if get_flags().capture.enable_torch_compile
        else []
    )
    return capture_bs, compile_bs


class BaseCudaGraphRunner(BaseRunner):


    buffers: ForwardInputBuffers
    backend: BaseCudaGraphBackend

    @staticmethod
    def _pad_to_bucket(raw_size: int, buckets: Sequence[int]) -> int:

        assert raw_size <= buckets[-1], (
            f"size {raw_size} exceeds max captured bucket {buckets[-1]}; "
            f"can_run_graph should have rejected this batch"
        )
        index = bisect.bisect_left(buckets, raw_size)
        return buckets[index]

    @abstractmethod
    def capture_prepare(self, size: int, *args, **kwargs) -> Any: ...

    @abstractmethod
    def capture(self) -> None: ...

    @abstractmethod
    def capture_one_shape(self, size: int, *args, **kwargs) -> Any: ...
