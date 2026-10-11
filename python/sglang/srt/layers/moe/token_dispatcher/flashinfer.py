from __future__ import annotations

import logging
from typing import NamedTuple, Optional

import torch

from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    get_dp_global_num_tokens,
    is_dp_attention_enabled,
)
from sglang.srt.layers.moe.token_dispatcher import (
    BaseDispatcher,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.token_dispatcher.flashinfer_utils import (
    TorchDistributedCommBackend,
)
from sglang.srt.layers.moe.topk import (
    StandardTopKOutput,
    TopKOutput,
    TopKOutputChecker,
)
from sglang.srt.layers.moe.utils import get_moe_runner_backend
from sglang.srt.runtime_context import get_schedule, get_spec
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

try:
    from flashinfer import nvfp4_block_scale_interleave
    from flashinfer.comm import MoeAlltoAll, moe_a2a_get_workspace_size_per_rank
    from flashinfer.comm.mapping import Mapping
    from flashinfer.comm.mnnvl import MnnvlConfig

    from sglang.srt.layers.quantization.fp4_utils import fp4_quantize

    use_flashinfer = True
except ImportError:
    use_flashinfer = False

logger = logging.getLogger(__name__)

MOE_NVFP4_DISPATCH = envs.SGLANG_MOE_NVFP4_DISPATCH.get()


class FlashinferDispatchOutput(NamedTuple):


    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_output: StandardTopKOutput

    moe_output: Optional[torch.Tensor] = None

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.FLASHINFER


assert isinstance(FlashinferDispatchOutput, DispatchOutput)


class FlashinferCombineInput(NamedTuple):


    hidden_states: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.FLASHINFER


assert isinstance(FlashinferCombineInput, CombineInput)


class FlashinferDispatcher(BaseDispatcher):


    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
    ):
        super().__init__()
        if not use_flashinfer:
            raise ImportError(
                "Flashinfer is not installed or does not support A2A. "
                "Please install the appropriate version of Flashinfer."
            )

        self.ep_size = group.size()
        self.ep_rank = group.rank()
        self.router_topk = router_topk
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.invalid_token_expert_id = (
            -1
            if get_moe_runner_backend().is_flashinfer_trtllm_routed()
            else self.num_experts
        )

        self.payload_in_workspace = get_moe_runner_backend().is_flashinfer_cutlass()


        cps = get_schedule().chunked_prefill_size
        default_max_tokens = max(cps if cps and cps > 0 else 4096, 4096)
        configured_max_tokens = (
            envs.SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
        )
        self.max_num_tokens = (
            configured_max_tokens
            if configured_max_tokens is not None
            else default_max_tokens
        )


        speculative_algo = SpeculativeAlgorithm.from_string(
            get_spec().speculative_algorithm
        )
        if MOE_NVFP4_DISPATCH and not speculative_algo.is_eagle():
            total_dispatch_payload_size_per_token = (
                hidden_size // 2
                + hidden_size // 16
                + self.router_topk * 4
                + self.router_topk * 4
            )
        else:
            total_dispatch_payload_size_per_token = (
                hidden_size * 2
                + self.router_topk * 4
                + self.router_topk * 4
            )
        combine_payload_size_per_token = hidden_size * 2
        self.workspace_size = moe_a2a_get_workspace_size_per_rank(
            ep_size=self.ep_size,
            max_num_tokens=self.max_num_tokens,
            total_dispatch_payload_size_per_token=total_dispatch_payload_size_per_token,
            combine_payload_size_per_token=combine_payload_size_per_token,
        )

        self.mapping = Mapping(
            rank=self.ep_rank,
            tp_size=self.ep_size,
            moe_ep_size=self.ep_size,
            world_size=self.ep_size,
            gpus_per_node=torch.cuda.device_count(),
            pp_size=1,
            cp_size=1,
        )
        self.moe_a2a = MoeAlltoAll(
            mapping=self.mapping,
            max_num_tokens=self.max_num_tokens,
            top_k=self.router_topk,
            num_experts=self.num_experts,
            workspace_size_per_rank=self.workspace_size,
            mnnvl_config=MnnvlConfig(comm_backend=TorchDistributedCommBackend(group)),
        )

    @debug_kernel_api
    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> FlashinferDispatchOutput:
        output_dtype = hidden_states.dtype
        x = hidden_states
        x_sf = None

        if TopKOutputChecker.format_is_bypassed(topk_output):
            topk_output = topk_output.to_standard()


        topk_ids = topk_output.topk_ids.to(torch.int32)
        topk_weights = topk_output.topk_weights

        global_scale = self.quant_config.get("input_global_scale", None)
        if global_scale is not None:
            if x.shape[0] > 0:
                x, x_sf = fp4_quantize(x, global_scale, is_sf_swizzled_layout=False)
            else:
                x_col = x.shape[1]
                x = torch.zeros(0, x_col // 2, dtype=torch.uint8, device=x.device)
                x_sf = torch.zeros(0, x_col // 16, dtype=torch.uint8, device=x.device)

        payloads = []
        payloads.append(x)
        if x_sf is not None:
            payloads.append(x_sf)
            expert_id_payload_index = 2
        else:
            expert_id_payload_index = 1
        payloads.append(topk_ids)
        payloads.append(topk_weights)


        dp_global = get_dp_global_num_tokens()
        if dp_global is not None and len(dp_global) > 1:

            self.runtime_max_tokens_per_rank = max(dp_global)
        else:


            assert not is_dp_attention_enabled() or self.ep_size == 1, (
                "FlashInfer A2A: DP attention reached the x.shape[0] fallback "
                f"with ep_size={self.ep_size} > 1 (dp_global={dp_global}); "
                "runtime_max_tokens_per_rank would not be rank-invariant."
            )
            self.runtime_max_tokens_per_rank = x.shape[0]


        assert self.runtime_max_tokens_per_rank >= x.shape[0], (
            f"runtime_max_tokens_per_rank={self.runtime_max_tokens_per_rank} < "
            f"x.shape[0]={x.shape[0]}: MoeAlltoAll recv buffer would overflow."
        )


        recv_tensors = self.moe_a2a.dispatch(
            topk_ids,
            payloads,
            self.runtime_max_tokens_per_rank,
            invalid_token_expert_id=self.invalid_token_expert_id,
            expert_id_payload_index=expert_id_payload_index,
        )
        if x_sf is not None:
            x_recv, x_sf_recv, topk_ids_recv, topk_weights_recv = recv_tensors
            x_sf = x_sf_recv.view(-1, x_sf_recv.shape[-1])

            if get_moe_runner_backend().is_flashinfer_cutlass():
                x_sf = nvfp4_block_scale_interleave(x_sf)
        else:
            x_recv, topk_ids_recv, topk_weights_recv = recv_tensors
        x = x_recv.view(-1, x_recv.shape[-1])
        topk_ids = topk_ids_recv.view(-1, topk_ids_recv.shape[-1])
        topk_weights = topk_weights_recv.view(-1, topk_weights_recv.shape[-1])


        moe_output = None
        if self.payload_in_workspace:
            moe_output = self.moe_a2a.get_combine_payload_tensor_in_workspace(
                self.runtime_max_tokens_per_rank, self.hidden_size, output_dtype
            ).view(-1, self.hidden_size)
        return FlashinferDispatchOutput(
            x,
            x_sf,
            StandardTopKOutput(topk_weights, topk_ids, topk_output.router_logits),
            moe_output,
        )

    @debug_kernel_api
    def combine(self, combine_input: FlashinferCombineInput) -> torch.Tensor:
        hidden_states = combine_input.hidden_states
        output_hidden_size = hidden_states.shape[-1]
        hidden_states = self.moe_a2a.combine(
            hidden_states.view(
                self.ep_size, self.runtime_max_tokens_per_rank, output_hidden_size
            ),
            self.runtime_max_tokens_per_rank,
            payload_in_workspace=self.payload_in_workspace,
        )

        del self.runtime_max_tokens_per_rank
        return hidden_states
