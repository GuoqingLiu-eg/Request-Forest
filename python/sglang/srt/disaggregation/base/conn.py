from __future__ import annotations

import dataclasses
import enum
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import numpy.typing as npt

from sglang.srt.server_args import ServerArgs

if TYPE_CHECKING:
    from sglang.srt.disaggregation.utils import DisaggregationMode


class StateType(str, enum.Enum):
    MAMBA = "mamba"
    SWA = "swa"
    DSA = "dsa"
    MINIMAX_INDEX_K = "minimax_index_k"


    SWA_RING = "swa_ring"

    C128_STATE = "c128_state"


    BLOCK_SCALE = "block_scale"
    BLOCK_SCALE_SWA = "block_scale_swa"


@dataclasses.dataclass
class KVTransferMetric:

    transfer_latency_s: Optional[float] = None

    alloc_latency_s: Optional[float] = None
    transfer_total_bytes: Optional[int] = None


class KVArgs:
    engine_rank: int
    kv_data_ptrs: List[int]
    kv_data_lens: List[int]
    kv_item_lens: List[int]
    kv_layer_ids: List[int]
    kv_cache_dtype_str: str
    aux_data_ptrs: List[int]
    aux_data_lens: List[int]
    aux_item_lens: List[int]
    state_types: List[StateType]
    state_data_ptrs: List[List[int]]
    state_data_lens: List[List[int]]
    state_item_lens: List[List[int]]
    state_layer_ids: List[List[int]]

    state_dim_per_tensor: List[List[int]]

    state_slice_outer_counts: List[List[int]]
    is_hybrid_mla_backend: bool


    state_conv_shard_groups: List[List[Optional[List[int]]]]
    ib_device: str
    gpu_id: int
    kv_head_num: int
    total_kv_head_num: int
    page_size: int

    system_dp_rank: int

    pp_rank: int
    prefill_start_layer: int


    prefill_end_layer: Optional[int]


    mla_compression_ratios: Optional[List[int]]

    kv_buf_groups: int

    total_kv_layers: int


class KVPoll:
    Failed = 0
    Bootstrapping = 1
    WaitingForInput = 2
    Transferring = 3
    Success = 4


class BaseKVManager(ABC):


    @abstractmethod
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ): ...

    @abstractmethod
    def register_to_bootstrap(self):

        ...


class BaseKVSender(ABC):
    @abstractmethod
    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
        req_has_disagg_prefill_dp_rank: bool = False,
    ): ...

    @abstractmethod
    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):

        ...

    @abstractmethod
    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List] = None,
        num_kv_tokens: Optional[int] = None,
    ):

        ...

    def pop_decode_prefix_len(self) -> int:
        return 0

    def should_send_kv_chunk(self, num_pages: int, last_chunk: bool) -> bool:
        return num_pages > 0

    @abstractmethod
    def get_transfer_metric(self) -> KVTransferMetric:

        ...

    @abstractmethod
    def poll(self) -> KVPoll:

        ...

    @abstractmethod
    def failure_exception(self):

        ...

    def clear(self):

        pass

    def abort(self):

        pass


class BaseKVReceiver(ABC):
    @abstractmethod
    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ): ...

    @abstractmethod
    def init(
        self,
        prefill_dp_rank: int,
    ):

        ...

    @abstractmethod
    def send_metadata(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List] = None,
        decode_prefix_len: Optional[int] = None,
    ):

        ...

    @abstractmethod
    def poll(self) -> KVPoll:

        ...

    @abstractmethod
    def failure_exception(self):

        ...

    def clear(self):

        pass

    def abort(self):

        pass


class BaseKVBootstrapServer(ABC):
    @abstractmethod
    def __init__(self, host: str, port: int): ...
