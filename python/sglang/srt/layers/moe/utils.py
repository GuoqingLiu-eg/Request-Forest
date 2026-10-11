from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from enum import Enum, IntEnum

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    is_dp_attention_enabled,
)
from sglang.srt.runtime_context import (
    get_exec,
    get_flags,
    get_forward,
    get_model,
    get_parallel,
    get_spec,
)
from sglang.srt.utils import is_cuda, is_npu

_is_npu = is_npu()

from sglang.srt.runtime_context import get_server_args
from sglang.srt.utils.common import log_info_on_rank0

logger = logging.getLogger(__name__)


class MoeA2ABackend(Enum):

    NONE = "none"
    DEEPEP = "deepep"
    MOONCAKE = "mooncake"
    NIXL = "nixl"
    MORI = "mori"
    ASCEND_FUSEEP = "ascend_fuseep"
    ASCEND_TP = "ascend_tp"
    FLASHINFER = "flashinfer"
    MEGAMOE = "megamoe"
    PPLX = "pplx"
    CUSTOMIZED = "customized"

    @classmethod
    def _missing_(cls, value):
        if value is None:
            return cls.NONE
        for member in cls:
            if value == member.value:
                return member
        raise ValueError(f"No {cls.__name__} member for value {value}")

    def is_none(self):
        return self == MoeA2ABackend.NONE

    def is_deepep(self):
        return self == MoeA2ABackend.DEEPEP

    def is_mooncake(self):
        return self == MoeA2ABackend.MOONCAKE

    def is_nixl(self):
        return self == MoeA2ABackend.NIXL

    def is_flashinfer(self):
        return self == MoeA2ABackend.FLASHINFER

    def is_ascend_fuseep(self):
        return self == MoeA2ABackend.ASCEND_FUSEEP

    def is_ascend_tp(self):
        return self == MoeA2ABackend.ASCEND_TP

    def is_mori(self):
        return self == MoeA2ABackend.MORI

    def is_megamoe(self):
        return self == MoeA2ABackend.MEGAMOE

    def is_pplx(self):
        return self == MoeA2ABackend.PPLX

    def is_customized(self):
        return self == MoeA2ABackend.CUSTOMIZED

    def supports_aiter(self) -> bool:
        return self in (
            MoeA2ABackend.NONE,
            MoeA2ABackend.DEEPEP,
            MoeA2ABackend.MOONCAKE,
            MoeA2ABackend.NIXL,
            MoeA2ABackend.MORI,
        )


class MoeRunnerBackend(Enum):

    AUTO = "auto"
    DEEP_GEMM = "deep_gemm"
    TRITON = "triton"
    TRITON_KERNELS = "triton_kernel"
    ASCEND = "ascend"
    FLASHINFER_TRTLLM = "flashinfer_trtllm"
    EXPERIMENTAL_SGL_TRTLLM = "experimental_sgl_trtllm"
    FLASHINFER_TRTLLM_ROUTED = "flashinfer_trtllm_routed"
    FLASHINFER_CUTLASS = "flashinfer_cutlass"
    FLASHINFER_MXFP4 = "flashinfer_mxfp4"
    FLASHINFER_CUTEDSL = "flashinfer_cutedsl"
    CUTLASS = "cutlass"
    MARLIN = "marlin"
    HUMMING = "humming"
    EXPERIMENTAL_SGL_MARLIN = "experimental_sgl_marlin"
    AITER = "aiter"
    HPC_OPS = "hpc_ops"

    def is_auto(self):
        return self == MoeRunnerBackend.AUTO

    def is_hpc_ops(self):
        return self == MoeRunnerBackend.HPC_OPS

    def is_deep_gemm(self):
        return self == MoeRunnerBackend.DEEP_GEMM

    def is_triton(self):
        return self == MoeRunnerBackend.TRITON

    def is_ascend(self):
        return self == MoeRunnerBackend.ASCEND

    def is_triton_kernels(self):
        return self == MoeRunnerBackend.TRITON_KERNELS

    def is_flashinfer_trtllm(self):


        return self in (
            MoeRunnerBackend.FLASHINFER_TRTLLM,
            MoeRunnerBackend.EXPERIMENTAL_SGL_TRTLLM,
        )

    def is_experimental_sgl_trtllm(self):
        return self == MoeRunnerBackend.EXPERIMENTAL_SGL_TRTLLM

    def is_flashinfer_trtllm_routed(self):
        return self == MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED

    def is_flashinfer_cutlass(self):
        return self == MoeRunnerBackend.FLASHINFER_CUTLASS

    def is_flashinfer_cutedsl(self):
        return self == MoeRunnerBackend.FLASHINFER_CUTEDSL

    def is_flashinfer_mxfp4(self):
        return self == MoeRunnerBackend.FLASHINFER_MXFP4

    def is_cutlass(self):
        return self == MoeRunnerBackend.CUTLASS

    def is_marlin(self):


        return self in (
            MoeRunnerBackend.MARLIN,
            MoeRunnerBackend.EXPERIMENTAL_SGL_MARLIN,
        )

    def is_experimental_sgl_marlin(self):
        return self == MoeRunnerBackend.EXPERIMENTAL_SGL_MARLIN

    def is_humming(self):
        return self == MoeRunnerBackend.HUMMING

    def is_aiter(self):
        return self == MoeRunnerBackend.AITER


class DeepEPMode(Enum):

    NORMAL = "normal"
    LOW_LATENCY = "low_latency"
    AUTO = "auto"

    def enable_normal(self) -> bool:
        return self in [DeepEPMode.NORMAL, DeepEPMode.AUTO]

    def enable_low_latency(self) -> bool:
        return self in [DeepEPMode.LOW_LATENCY, DeepEPMode.AUTO]

    def resolve(self, is_extend_in_batch: bool) -> DeepEPMode:
        if self != DeepEPMode.AUTO:
            return self

        if is_extend_in_batch:
            return DeepEPMode.NORMAL
        else:
            return DeepEPMode.LOW_LATENCY

    def is_normal(self) -> bool:
        return self == DeepEPMode.NORMAL

    def is_low_latency(self) -> bool:
        return self == DeepEPMode.LOW_LATENCY

    def is_auto(self) -> bool:
        return self == DeepEPMode.AUTO


class DispatcherOutputDtype(Enum):


    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"
    NVFP4 = "nvfp4"
    MXFP8 = "mxfp8"


def get_deepep_output_dtype(self) -> DispatcherOutputDtype:


    server_args = get_server_args()
    if server_args and get_exec().moe.deepep_dispatcher_output_dtype != "auto":
        return DispatcherOutputDtype(get_exec().moe.deepep_dispatcher_output_dtype)


    if envs.SGLANG_DEEPEP_BF16_DISPATCH.get():
        logger.warning_once(
            "Warning: The env variable SGLANG_DEEPEP_BF16_DISPATCH deprecated "
            "and will be removed in future releases. Please use a new "
            "`--deepep-dispatcher-output-dtype bf16` argument instead."
        )
        return DispatcherOutputDtype.BF16


    if self.quant_config is not None:
        input_global_scale = self.quant_config.get("input_global_scale", None)
        if input_global_scale is not None:
            return DispatcherOutputDtype.NVFP4


        dispatch_mode = getattr(self, "dispatch_mode", None)
        if dispatch_mode is not None:
            mode_dispatcher_output_dtype = self.quant_config.get(
                f"{dispatch_mode.value}_dispatcher_output_dtype", None
            )
            if mode_dispatcher_output_dtype is not None:
                return DispatcherOutputDtype(mode_dispatcher_output_dtype)


        dispatcher_output_dtype = self.quant_config.get("dispatcher_output_dtype", None)
        if dispatcher_output_dtype is not None:
            return DispatcherOutputDtype(dispatcher_output_dtype)


    if (
        get_moe_runner_backend().is_flashinfer_cutedsl()
        or get_moe_runner_backend().is_cutlass()
        or get_moe_runner_backend().is_humming()
    ):
        return DispatcherOutputDtype.BF16


    if _is_npu:
        return DispatcherOutputDtype.BF16


    return DispatcherOutputDtype.FP8


def get_ascend_dispatcher_output_dtype(dispatcher):


    if dispatcher.quant_config is not None:
        dispatcher_output_dtype = dispatcher.quant_config.get(
            "dispatcher_output_dtype", None
        )
        if dispatcher_output_dtype is not None:
            return DispatcherOutputDtype(dispatcher_output_dtype)


    return DispatcherOutputDtype.BF16


def initialize_moe_config():

    exec_moe = get_exec().moe
    overlap = get_exec().overlap
    spec = get_spec()
    moe = get_flags().moe
    moe.a2a_backend = MoeA2ABackend(exec_moe.moe_a2a_backend)
    moe.runner_backend = MoeRunnerBackend(exec_moe.moe_runner_backend)
    moe.speculative_runner_backend = (
        MoeRunnerBackend(spec.speculative_moe_runner_backend)
        if spec.speculative_moe_runner_backend is not None
        else moe.runner_backend
    )
    moe.speculative_a2a_backend = (
        MoeA2ABackend(spec.speculative_moe_a2a_backend)
        if spec.speculative_moe_a2a_backend is not None
        else moe.a2a_backend
    )
    moe.deepep_mode = DeepEPMode(exec_moe.deepep_mode)
    moe.deepep_config = exec_moe.deepep_config or ""
    moe.tbo_enabled = overlap.enable_two_batch_overlap
    moe.sbo_enabled = overlap.enable_single_batch_overlap
    if moe.sbo_enabled and is_cuda():
        if torch.cuda.get_device_capability()[0] == 9:
            raise ValueError(
                "SBO (single batch overlap) is not supported on SM90 GPUs with latest sgl-deep-gemm wheel. Please try removing --enable-single-batch-overlap argument."
            )
    moe.tbo_token_distribution_threshold = overlap.tbo_token_distribution_threshold
    moe.disable_fp4_allgather = exec_moe.disable_flashinfer_cutlass_moe_fp4_allgather
    moe.quantization = get_model().quantization


    moe.disable_shared_experts_fusion = exec_moe.disable_shared_experts_fusion
    moe.speculative_disable_shared_experts_fusion = (
        exec_moe.disable_shared_experts_fusion
    )


def get_moe_a2a_backend() -> MoeA2ABackend:
    moe = get_flags().moe
    if moe.a2a_backend is None:
        moe.a2a_backend = MoeA2ABackend.NONE
    return moe.a2a_backend


def get_moe_runner_backend() -> MoeRunnerBackend:
    moe = get_flags().moe
    if moe.runner_backend is None:
        moe.runner_backend = MoeRunnerBackend.AUTO
    return moe.runner_backend


def get_speculative_moe_runner_backend() -> MoeRunnerBackend:
    moe = get_flags().moe
    if moe.speculative_runner_backend is None:
        logger.warning(
            "SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend"
        )
        moe.speculative_runner_backend = MoeRunnerBackend.AUTO
    return moe.speculative_runner_backend


def get_speculative_moe_a2a_backend() -> MoeA2ABackend:
    moe = get_flags().moe
    if moe.speculative_a2a_backend is None:
        logger.warning(
            "SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend"
        )
        moe.speculative_a2a_backend = MoeA2ABackend.NONE
    return moe.speculative_a2a_backend


def is_shared_experts_fusion_disabled() -> bool:

    from sglang.srt.model_executor.forward_context import has_forward_context

    if has_forward_context():
        raise AssertionError(
            "is_shared_experts_fusion_disabled() called inside a forward: the "
            "fusion decision is construction-time state (it can hold the draft's "
            "value while a draft builds). Read the value your build baked in, "
            "e.g. the layer's num_fused_shared_experts."
        )
    moe = get_flags().moe
    if moe.disable_shared_experts_fusion is None:
        from sglang.srt.runtime_context import get_exec

        return get_exec().moe.disable_shared_experts_fusion
    return moe.disable_shared_experts_fusion


@contextmanager
def draft_model_build_scope():

    moe = get_flags().moe
    original_fusion = moe.disable_shared_experts_fusion
    original_scope = moe.in_speculative_scope
    try:
        moe.in_speculative_scope = True
        yield
    finally:
        moe.in_speculative_scope = original_scope
        moe.disable_shared_experts_fusion = original_fusion


def install_shared_experts_fusion_decision(
    model_class, hf_config, quant_config
) -> None:

    from sglang.srt.runtime_context import get_exec

    disabled = get_exec().moe.disable_shared_experts_fusion
    if not disabled:
        gate = getattr(model_class, "shared_experts_fusion_disable_reason", None)
        reason = gate(hf_config, quant_config) if gate is not None else None
        if reason:
            log_info_on_rank0(
                logger, f"{reason} Shared experts fusion optimization is disabled."
            )
            disabled = True
    moe = get_flags().moe
    moe.disable_shared_experts_fusion = disabled
    if moe.in_speculative_scope:
        moe.speculative_disable_shared_experts_fusion = disabled


def get_deepep_mode() -> DeepEPMode:
    moe = get_flags().moe
    if moe.deepep_mode is None:
        logger.warning("DEEPEP_MODE is not initialized, using auto mode")
        moe.deepep_mode = DeepEPMode.AUTO
    return moe.deepep_mode


def get_deepep_config() -> str:
    moe = get_flags().moe
    if moe.deepep_config is None:
        logger.warning("DEEPEP_CONFIG is not initialized, using default config")
        moe.deepep_config = ""
    return moe.deepep_config


def is_tbo_enabled() -> bool:
    moe = get_flags().moe
    if moe.tbo_enabled is None:
        moe.tbo_enabled = False
    return moe.tbo_enabled


def is_sbo_enabled() -> bool:
    moe = get_flags().moe
    if moe.sbo_enabled is None:
        moe.sbo_enabled = False
    return moe.sbo_enabled


def is_deepep_class_backend() -> bool:

    b = get_moe_a2a_backend()
    return b.is_deepep() or b.is_mooncake() or b.is_mori() or b.is_pplx()


def uses_per_rank_fused_shared_slots() -> bool:

    return is_deepep_class_backend() or get_moe_a2a_backend().is_megamoe()


def has_per_rank_fused_shared_slots(num_fused_shared_experts: int) -> bool:

    return num_fused_shared_experts > 0 and uses_per_rank_fused_shared_slots()


def is_flashinfer_cutedsl_v1_path() -> bool:

    return (
        get_moe_runner_backend().is_flashinfer_cutedsl()
        and get_moe_a2a_backend().is_deepep()
    )


def get_tbo_token_distribution_threshold() -> float:
    moe = get_flags().moe
    if moe.tbo_token_distribution_threshold is None:
        logger.warning(
            "TBO_TOKEN_DISTRIBUTION_THRESHOLD is not initialized, using 0.48"
        )
        moe.tbo_token_distribution_threshold = 0.48
    return moe.tbo_token_distribution_threshold


def filter_moe_weight_param_global_expert(name, x, num_local_experts):

    return (
        not getattr(x, "_sglang_require_global_experts", False)
        and x.data.ndim > 0
        and x.data.shape[0] == num_local_experts
    )


def should_use_flashinfer_cutlass_moe_fp4_allgather():

    return (
        not get_flags().moe.disable_fp4_allgather
        and get_moe_a2a_backend().is_none()
        and get_moe_runner_backend().is_flashinfer_cutlass()
        and is_dp_attention_enabled()
        and get_flags().moe.quantization == "modelopt_fp4"
        and get_parallel().moe_ep_size == get_parallel().attn_dp_size
    )


def should_use_dp_reduce_scatterv():

    return (
        not should_use_flashinfer_cutlass_moe_fp4_allgather()
        and get_moe_a2a_backend().is_none()
        and is_dp_attention_enabled()
        and get_parallel().attn_dp_size > 1
        and get_parallel().tp_size == get_parallel().attn_dp_size
        and get_parallel().moe_ep_size == get_parallel().attn_dp_size
    )


def should_skip_mlp_all_reduce() -> bool:

    f = get_forward()
    return f.fuse_mlp_allreduce or f.mlp_reduce_scatter


def should_skip_post_experts_all_reduce(*, is_tp_path: bool) -> bool:

    if should_skip_mlp_all_reduce():
        return True
    if get_parallel().config.dwdp_size > 1:
        return True
    if should_use_dp_reduce_scatterv():
        return True
    if is_tp_path and should_use_flashinfer_cutlass_moe_fp4_allgather():
        return True
    if get_moe_a2a_backend().is_flashinfer():
        return True
    if get_moe_a2a_backend().is_pplx():


        return True
    return False


@contextmanager
def speculative_moe_backend_context():

    moe = get_flags().moe
    original_backend = moe.runner_backend
    try:
        moe.runner_backend = get_speculative_moe_runner_backend()
        yield
    finally:
        moe.runner_backend = original_backend


@contextmanager
def speculative_moe_a2a_backend_context():

    moe = get_flags().moe
    original_backend = moe.a2a_backend
    original_disable_fp4_allgather = moe.disable_fp4_allgather
    try:
        moe.a2a_backend = get_speculative_moe_a2a_backend()

        moe.disable_fp4_allgather = True
        yield
    finally:
        moe.a2a_backend = original_backend
        moe.disable_fp4_allgather = original_disable_fp4_allgather


class RoutingMethodType(IntEnum):

    Default = (0,)

    Renormalize = (1,)

    DeepSeekV3 = (2,)

    Llama4 = (3,)

    RenormalizeNaive = (4,)

    TopK = (5,)

    SigmoidRenorm = (6,)

    MiniMax2 = (7,)

    Sigmoid = (8,)

    Unspecified = 9


AITER_PADDING_SIZE = 128
TRITON_PADDING_SIZE = 128


XPU_MOE_LD_PADDING_BYTES = 64


def xpu_moe_ld_padding_elems(k_dim: int, itemsize: int) -> int:

    row_bytes = k_dim * itemsize
    if row_bytes <= 0 or XPU_MOE_LD_PADDING_BYTES % itemsize != 0:
        return 0
    trailing_zeros = (row_bytes & -row_bytes).bit_length() - 1
    odd_cofactor = row_bytes >> trailing_zeros
    if trailing_zeros >= 11 and odd_cofactor >= 3:
        return XPU_MOE_LD_PADDING_BYTES // itemsize
    return 0


def get_moe_padding_size(is_aiter_moe):
    if is_aiter_moe:
        return AITER_PADDING_SIZE
    else:
        return (
            TRITON_PADDING_SIZE
            if bool(int(os.getenv("SGLANG_MOE_PADDING", "0")))
            else 0
        )


def get_moe_weight_sizes(inter_dim, is_concat, is_packed, is_aiter_moe):


    w13_up_dim = 2 * inter_dim if is_concat else inter_dim
    w2_down_dim = inter_dim // 2 if is_packed else inter_dim

    if is_aiter_moe:
        padding_size = get_moe_padding_size(True)
        align_aiter = lambda n: ((n + padding_size - 1) // padding_size) * padding_size
        is_padded = (w2_down_dim % padding_size) > 0
        if is_padded:

            w2_down_dim = align_aiter(w2_down_dim)

        if is_concat:
            w13_up_dim = w2_down_dim * 2

        if hasattr(torch, "float4_e2m1fn_x2") and is_packed:

            w13_up_dim *= 2

    return (w13_up_dim, w2_down_dim, False if not is_aiter_moe else is_padded)
