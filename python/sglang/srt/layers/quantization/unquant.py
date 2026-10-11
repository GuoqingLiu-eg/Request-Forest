from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Callable, List, Optional

logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from sglang.kernels.fused_op import BaseFusedOp
from sglang.srt.environ import envs
from sglang.srt.layers.amx_utils import (
    CPUQuantMethod,
    _amx_process_weight_after_loading,
)
from sglang.srt.layers.moe import (
    MoeRunner,
    MoeRunnerBackend,
    MoeRunnerConfig,
    get_moe_a2a_backend,
    get_moe_runner_backend,
)
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.moe.utils import xpu_moe_ld_padding_elems
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizeMethodBase,
)
from sglang.srt.layers.utils import copy_or_rebind_param
from sglang.srt.runtime_context import (
    get_exec,
    get_lora,
)
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_cuda,
    is_hip,
    is_npu,
    set_weight_attrs,
    use_intel_amx_backend,
    use_intel_xpu_backend,
)
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        DispatchOutput,
        StandardDispatchOutput,
    )
    from sglang.srt.server_args import ServerArgs

from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
    NPUUnquantMoEMethod,
)

_is_cpu_amx_available = cpu_has_amx_support()
_is_cuda = is_cuda()
_is_hip = is_hip()
_is_cpu = is_cpu()
_is_npu = is_npu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    from aiter.ops.shuffle import shuffle_weight
    from aiter.tuned_gemm import tgemm


class Bf16GemmBackend(Enum):
    AUTO = "auto"
    CUTEDSL = "cutedsl"
    GEMV = "gemv"
    TORCH = "torch"

    def is_auto(self) -> bool:
        return self == Bf16GemmBackend.AUTO

    def is_cutedsl(self) -> bool:
        return self == Bf16GemmBackend.CUTEDSL

    def is_gemv(self) -> bool:
        return self == Bf16GemmBackend.GEMV


_BF16_GEMM_BACKEND: Optional[Bf16GemmBackend] = None
_cutedsl_bf16_gemm = None
_use_cutedsl_bf16_gemm = None
_hopper_bf16_gemv = None
_use_hopper_bf16_gemv = None


def initialize_bf16_gemm_config(server_args: ServerArgs) -> None:
    global _BF16_GEMM_BACKEND, _cutedsl_bf16_gemm, _use_cutedsl_bf16_gemm

    from sglang.srt.utils import is_sm100_supported

    backend_str = server_args.bf16_gemm_backend
    if backend_str == "auto" and is_sm100_supported():
        backend_str = (
            "torch"
            if get_exec().deterministic.enable_deterministic_inference
            else "cutedsl"
        )

    backend = Bf16GemmBackend(backend_str)

    if backend.is_gemv():
        if torch.cuda.get_device_capability()[0] != 9:
            raise ValueError("--bf16-gemm-backend gemv requires SM90 (Hopper)")

        global _hopper_bf16_gemv, _use_hopper_bf16_gemv
        from sglang.kernels.ops.gemm.hopper_bf16_gemv import (
            hopper_bf16_gemv,
            use_hopper_bf16_gemv,
        )

        _hopper_bf16_gemv = hopper_bf16_gemv
        _use_hopper_bf16_gemv = use_hopper_bf16_gemv
    elif backend.is_cutedsl():
        if get_exec().deterministic.enable_deterministic_inference:
            raise ValueError(
                "--bf16-gemm-backend cutedsl is batch-size dependent and cannot "
                "be combined with --enable-deterministic-inference"
            )
        if not is_sm100_supported():
            raise ValueError("--bf16-gemm-backend cutedsl requires an SM10x GPU")

        from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import (
            cutedsl_bf16_gemm,
            use_cutedsl_bf16_gemm,
        )

        _cutedsl_bf16_gemm = cutedsl_bf16_gemm
        _use_cutedsl_bf16_gemm = use_cutedsl_bf16_gemm

    _BF16_GEMM_BACKEND = backend


def _bf16_gemm_dispatch_fake(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


@register_custom_op(fake_impl=_bf16_gemm_dispatch_fake)
def bf16_gemm_dispatch(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    if (
        _use_hopper_bf16_gemv is not None
        and bias is None
        and _use_hopper_bf16_gemv(
            x.numel() // x.shape[-1], weight.shape[0], weight.shape[1]
        )
    ):
        return _hopper_bf16_gemv(x.view(-1, x.shape[-1]), weight).view(
            *x.shape[:-1], -1
        )
    if _use_cutedsl_bf16_gemm is not None and _use_cutedsl_bf16_gemm(
        x.numel() // x.shape[-1], weight.shape[0], weight.shape[1]
    ):
        return _cutedsl_bf16_gemm(x.view(-1, x.shape[-1]), weight, bias).view(
            *x.shape[:-1], -1
        )
    return F.linear(x, weight, bias)


def get_bf16_gemm_backend() -> Bf16GemmBackend:
    global _BF16_GEMM_BACKEND
    if _BF16_GEMM_BACKEND is None:
        _BF16_GEMM_BACKEND = Bf16GemmBackend.AUTO
    return _BF16_GEMM_BACKEND


class UnquantizedEmbeddingMethod(QuantizeMethodBase):


    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):

        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return F.linear(x, layer.weight, bias)

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_, layer.weight)


class UnquantizedLinearMethod(LinearMethodBase):


    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if _is_cpu and _is_cpu_amx_available:
            _amx_process_weight_after_loading(layer, ["weight"])

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if use_intel_amx_backend(layer):
            x_shapes = x.shape
            if len(x_shapes) == 3:
                x = x.view(-1, x.shape[-1])
            output = torch.ops.sgl_kernel.weight_packed_linear(
                x,
                layer.weight,
                bias,
                True,
            )
            if len(x_shapes) == 3:
                output = output.view(x_shapes[0], x_shapes[1], -1)
            return output

        elif _use_aiter and type(layer.weight.data) is torch.Tensor:
            return tgemm.mm(x, layer.weight, bias, otype=x.dtype)

        elif (
            get_bf16_gemm_backend().is_cutedsl()
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and layer.weight.dtype == torch.bfloat16
            and (bias is None or bias.dtype == torch.bfloat16)
            and not layer.weight.requires_grad
            and (bias is None or not bias.requires_grad)
        ):
            if torch.compiler.is_compiling():


                return bf16_gemm_dispatch(x, layer.weight, bias)
            if _use_cutedsl_bf16_gemm(
                x.numel() // x.shape[-1],
                layer.weight.shape[0],
                layer.weight.shape[1],
            ):
                x_shapes = x.shape
                output = _cutedsl_bf16_gemm(
                    x.view(-1, x_shapes[-1]), layer.weight, bias
                )
                return output.view(*x_shapes[:-1], -1)
            return F.linear(x, layer.weight, bias)

        return F.linear(x, layer.weight, bias)

    def apply_into(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        output: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if (
            get_bf16_gemm_backend().is_cutedsl()
            and x.is_cuda
            and x.ndim == 2
            and x.dtype == torch.bfloat16
            and layer.weight.dtype == torch.bfloat16
            and output.dtype == torch.bfloat16
            and output.is_contiguous()
            and output.shape == (x.shape[0], layer.weight.shape[0])
            and (bias is None or bias.dtype == torch.bfloat16)
            and not layer.weight.requires_grad
            and (bias is None or not bias.requires_grad)
            and _use_cutedsl_bf16_gemm(
                x.shape[0], layer.weight.shape[0], layer.weight.shape[1]
            )
        ):
            from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import (
                cutedsl_bf16_gemm_out,
            )

            return cutedsl_bf16_gemm_out(x, layer.weight, output, bias)

        if x.ndim != 2:
            raise ValueError("caller-owned linear output currently requires a 2D input")
        if output.shape != (x.shape[0], layer.weight.shape[0]):
            raise ValueError(
                f"linear output has shape {output.shape}, expected "
                f"{(x.shape[0], layer.weight.shape[0])}"
            )
        torch.mm(x, layer.weight.t(), out=output)
        if bias is not None:
            output.add_(bias)
        return output


def _use_xpu_moe_ld_padding(use_triton_kernels: bool) -> bool:

    return (
        use_intel_xpu_backend()
        and torch.get_default_device().type == "xpu"
        and not use_triton_kernels
    )


def _empty_xpu_moe_expert_weight(
    num_experts: int,
    n_dim: int,
    k_dim: int,
    dtype: torch.dtype,
) -> torch.Tensor:

    pad = xpu_moe_ld_padding_elems(k_dim, dtype.itemsize)
    if pad == 0:
        return torch.empty(num_experts, n_dim, k_dim, dtype=dtype)

    return torch.empty(num_experts, n_dim, k_dim + pad, dtype=dtype)[:, :, :k_dim]


class UnquantizedFusedMoEMethod(FusedMoEMethodBase, BaseFusedOp):


    def __init__(
        self,
        use_triton_kernels: bool = False,
        use_flashinfer_trtllm_moe: bool = False,
        use_deep_gemm: bool = False,
    ):
        super().__init__()
        self.use_flashinfer_cutlass = get_moe_runner_backend().is_flashinfer_cutlass()
        self.use_triton_kernels = use_triton_kernels
        self.with_bias = False
        self.use_flashinfer_trtllm_moe = use_flashinfer_trtllm_moe
        self.use_deep_gemm = use_deep_gemm
        self._cache_permute_indices = dict({})


        self.w13_swiglu_interleaved = False

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        with_bias: bool = False,
        **extra_weight_attrs,
    ):
        self.with_bias = with_bias


        pad_ld_for_xpu = _use_xpu_moe_ld_padding(self.use_triton_kernels)


        w13_up_dim = (
            2 * intermediate_size_per_partition
            if layer.moe_runner_config.is_gated
            else intermediate_size_per_partition
        )
        w13_weight_n, w13_weight_k = (w13_up_dim, hidden_size)
        if self.use_triton_kernels:
            w13_weight_n, w13_weight_k = w13_weight_k, w13_weight_n
        if pad_ld_for_xpu:
            w13_weight_data = _empty_xpu_moe_expert_weight(
                num_experts, w13_weight_n, w13_weight_k, params_dtype
            )
        else:
            w13_weight_data = torch.empty(
                num_experts, w13_weight_n, w13_weight_k, dtype=params_dtype
            )
        w13_weight = torch.nn.Parameter(w13_weight_data, requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        if self.with_bias:
            w13_weight_bias = torch.nn.Parameter(
                torch.empty(num_experts, w13_up_dim, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_bias", w13_weight_bias)
            set_weight_attrs(w13_weight_bias, extra_weight_attrs)


        w2_weight_n, w2_weight_k = (
            hidden_size,
            intermediate_size_per_partition,
        )
        if self.use_triton_kernels:
            w2_weight_n, w2_weight_k = w2_weight_k, w2_weight_n
        if pad_ld_for_xpu:
            w2_weight_data = _empty_xpu_moe_expert_weight(
                num_experts, w2_weight_n, w2_weight_k, params_dtype
            )
        else:
            w2_weight_data = torch.empty(
                num_experts, w2_weight_n, w2_weight_k, dtype=params_dtype
            )
        w2_weight = torch.nn.Parameter(w2_weight_data, requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        if self.with_bias:
            w2_weight_bias = torch.nn.Parameter(
                torch.empty(num_experts, hidden_size, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_bias", w2_weight_bias)
            set_weight_attrs(w2_weight_bias, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        _should_use_aiter_moe = (
            _use_aiter
            and (
                get_moe_runner_backend().is_auto()
                or get_moe_runner_backend().is_aiter()
            )
            and self._aiter_ck_moe_supported(layer)
            and not layer._skip_aiter_moe_shuffle
        )
        if _should_use_aiter_moe:
            copy_or_rebind_param(
                layer, "w13_weight", shuffle_weight(layer.w13_weight.data, (16, 16))
            )
            torch.cuda.empty_cache()
            copy_or_rebind_param(
                layer, "w2_weight", shuffle_weight(layer.w2_weight.data, (16, 16))
            )
            torch.cuda.empty_cache()


        if _is_cpu and _is_cpu_amx_available:
            _amx_process_weight_after_loading(layer, ["w13_weight", "w2_weight"])
            if hasattr(layer, "w13_weight_bias"):
                layer.w13_weight_bias = Parameter(
                    layer.w13_weight_bias.float(), requires_grad=False
                )
            if hasattr(layer, "w2_weight_bias"):
                layer.w2_weight_bias = Parameter(
                    layer.w2_weight_bias.float(), requires_grad=False
                )

        if (
            self.use_deep_gemm
            and layer.w13_weight.dtype == torch.bfloat16
            and (get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_pplx())
            and not _is_npu
            and not _is_hip
            and hasattr(layer, "dispatcher")
        ):
            layer.dispatcher.set_quant_config({"dispatcher_output_dtype": "bf16"})


        if self.use_flashinfer_trtllm_moe:


            self._cache_permute_indices.clear()

            from flashinfer.fused_moe.core import (
                _maybe_get_cached_w3_w1_permute_indices,
                convert_to_block_layout,
                get_w2_permute_indices_with_cache,
            )


            epilogue_tile_m = 128
            block_k = 128
            old_shape_w13 = layer.w13_weight.data[0].shape
            old_shape_w2 = layer.w2_weight.data[0].shape
            new_shape_w13 = None
            new_shape_w2 = None
            for i in range(layer.num_local_experts):
                permute_indices = _maybe_get_cached_w3_w1_permute_indices(
                    self._cache_permute_indices,
                    layer.w13_weight.data[i].view(torch.uint8),
                    epilogue_tile_m,
                    is_gated_act_gemm=layer.moe_runner_config.is_gated,
                )
                tmp_weights1 = (
                    layer.w13_weight.data[i]
                    .clone()
                    .view(torch.uint8)[permute_indices.to(layer.w13_weight.data.device)]
                    .contiguous()
                )

                permute_indices = get_w2_permute_indices_with_cache(
                    self._cache_permute_indices,
                    layer.w2_weight.data[i].view(torch.uint8),
                    epilogue_tile_m,
                )
                tmp_weights2 = (
                    layer.w2_weight.data[i]
                    .clone()
                    .view(torch.uint8)[permute_indices.to(layer.w2_weight.data.device)]
                    .contiguous()
                )

                tmp_weights1 = convert_to_block_layout(
                    tmp_weights1.view(torch.uint8), block_k
                )
                tmp_weights2 = convert_to_block_layout(
                    tmp_weights2.view(torch.uint8), block_k
                )

                new_shape_w13 = tmp_weights1.view(torch.bfloat16).shape
                new_shape_w2 = tmp_weights2.view(torch.bfloat16).shape
                layer.w13_weight.data[i] = (
                    tmp_weights1.view(torch.bfloat16)
                    .contiguous()
                    .reshape(old_shape_w13)
                )
                layer.w2_weight.data[i] = (
                    tmp_weights2.view(torch.bfloat16).contiguous().reshape(old_shape_w2)
                )

            layer.w13_weight.data = layer.w13_weight.data.reshape(
                layer.num_local_experts, *new_shape_w13
            )
            layer.w2_weight.data = layer.w2_weight.data.reshape(
                layer.num_local_experts, *new_shape_w2
            )
        if _is_npu:


            layer.w13_kernel.process_weights_after_loading(layer, "w13")
            layer.w2_kernel.process_weights_after_loading(layer, "w2")

        self._maybe_interleave_w13_for_fused_swiglu(layer)

        return

    def _maybe_interleave_w13_for_fused_swiglu(self, layer: torch.nn.Module) -> None:

        if not envs.SGLANG_OPT_FUSE_SWIGLU_INTERLEAVED.get():
            return

        moe_runner_config = layer.moe_runner_config
        if not (
            _is_cuda
            and self._aiter_runner is None
            and self.runner.runner_backend.is_triton()
            and get_moe_a2a_backend().is_none()
            and not self.with_bias
            and layer.w13_weight.dtype == torch.bfloat16
            and moe_runner_config.activation == "silu"
            and moe_runner_config.is_gated
            and moe_runner_config.gemm1_alpha is None
            and moe_runner_config.gemm1_clamp_limit is None
            and moe_runner_config.swiglu_limit is None


            and not get_lora().enable_lora
            and not get_lora().lora_paths

            and not get_exec().moe.enable_eplb
        ):
            return

        w13 = layer.w13_weight.data
        inter = w13.shape[1] // 2
        idx = torch.empty(w13.shape[1], dtype=torch.long, device=w13.device)
        idx[0::2] = torch.arange(0, inter, device=w13.device)
        idx[1::2] = torch.arange(inter, 2 * inter, device=w13.device)

        for e in range(w13.shape[0]):
            w13[e] = w13[e][idx]
        self.w13_swiglu_interleaved = True
        logger.info_once(
            "Interleaved w13 gate/up: the SwiGLU is applied by the MoE up-GEMM epilogue."
        )

    def maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load(
        self,
        layer: torch.nn.Module,
        param: torch.nn.Parameter,
        weight_name: str,
    ) -> None:

        if not get_moe_runner_backend().is_flashinfer_trtllm_routed():
            return

        expected_shape = None
        if weight_name.endswith(".experts.w13_weight"):
            w13_rows = (
                2 * layer.intermediate_size_per_partition
                if layer.moe_runner_config.is_gated
                else layer.intermediate_size_per_partition
            )
            expected_shape = (layer.num_local_experts, w13_rows, layer.hidden_size)
        elif weight_name.endswith(".experts.w2_weight"):
            expected_shape = (
                layer.num_local_experts,
                layer.hidden_size,
                layer.intermediate_size_per_partition,
            )

        if expected_shape is None or tuple(param.data.shape) == expected_shape:
            return

        expected_numel = expected_shape[0] * expected_shape[1] * expected_shape[2]
        if param.data.numel() != expected_numel:
            raise RuntimeError(
                f"Cannot restore flashinfer TRT-LLM BF16 MoE weight shape for {weight_name}: "
                f"current shape={tuple(param.data.shape)}, expected shape={expected_shape}."
            )

        param.data = param.data.reshape(expected_shape)

    def _aiter_ck_moe_supported(self, layer) -> bool:


        return layer.intermediate_size_per_partition % 128 == 0

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        if self.use_flashinfer_trtllm_moe:
            backend = (
                MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED
                if get_moe_runner_backend().is_flashinfer_trtllm_routed()
                else MoeRunnerBackend.FLASHINFER_TRTLLM
            )
        elif self.use_flashinfer_cutlass:
            import sglang.srt.layers.moe.moe_runner.flashinfer_cutlass

            backend = MoeRunnerBackend.FLASHINFER_CUTLASS
        elif self.use_deep_gemm:
            backend = MoeRunnerBackend.DEEP_GEMM
        elif self.use_triton_kernels:
            backend = MoeRunnerBackend.TRITON_KERNELS
        elif _is_npu:
            layer.w13_kernel = NPUUnquantMoEMethod()
            layer.w2_kernel = NPUUnquantMoEMethod()
            moe_runner_config.layer = layer
            backend = MoeRunnerBackend.ASCEND
        else:
            backend = MoeRunnerBackend.TRITON
        self.runner = MoeRunner(backend, moe_runner_config)


        self._aiter_runner: Optional[MoeRunner] = None
        if (
            _use_aiter
            and (
                get_moe_runner_backend().is_auto()
                or get_moe_runner_backend().is_aiter()
            )
            and get_moe_a2a_backend().supports_aiter()
        ):
            if self._aiter_ck_moe_supported(layer):
                self._aiter_runner = MoeRunner(
                    MoeRunnerBackend.AITER, moe_runner_config
                )
            elif get_moe_runner_backend().is_aiter():
                raise ValueError(
                    "moe_runner_backend=aiter is not supported for "
                    f"intermediate_size_per_partition={layer.intermediate_size_per_partition}; "
                    "use --moe-runner-backend triton."
                )
            else:
                logger.warning_once(
                    "aiter CK fused-MoE does not support "
                    f"intermediate_size_per_partition={layer.intermediate_size_per_partition}; "
                    "using triton MoE runner."
                )

    @property
    def load_up_proj_weight_first(self) -> bool:

        return self.use_flashinfer_cutlass

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        return self.forward(
            layer=layer,
            dispatch_output=dispatch_output,
        )


    def _torch_compile_forward(self, num_tokens: int) -> Optional[Callable]:


        if num_tokens == 1:
            from sglang.srt.layers.moe.fused_moe_native import (
                fused_moe_forward_native,
            )

            return fused_moe_forward_native
        return None

    def forward_cuda(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        x = dispatch_output.hidden_states

        backend = self.runner.runner_backend
        if backend.is_triton_kernels():
            from sglang.srt.layers.moe.moe_runner.triton_kernels import (
                TritonKernelsQuantInfo,
            )

            quant_info = TritonKernelsQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                w13_bias=getattr(layer, "w13_weight_bias", None),
                w2_bias=getattr(layer, "w2_weight_bias", None),
            )
            return self.runner.run(dispatch_output, quant_info)
        elif self.runner.runner_backend.is_deep_gemm():
            w13_weight = layer.w13_weight
            w2_weight = layer.w2_weight
            from sglang.srt.layers.moe.moe_runner.deep_gemm import DeepGemmMoeQuantInfo


            use_fp8 = not envs.SGLANG_DEEPEP_BF16_DISPATCH.get()
            quant_info = DeepGemmMoeQuantInfo(
                w13_weight=w13_weight,
                w2_weight=w2_weight,
                use_fp8=use_fp8,
            )
            return self.runner.run(dispatch_output, quant_info)
        elif self.use_flashinfer_cutlass:
            from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
                FlashInferCutlassMoeQuantInfo,
            )

            quant_info = FlashInferCutlassMoeQuantInfo(
                quant_type="bf16",
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                output_dtype=x.dtype,
                moe_ep_size=layer.moe_ep_size,
                moe_ep_rank=layer.moe_ep_rank,
                moe_tp_size=layer.moe_tp_size,
                moe_tp_rank=layer.moe_tp_rank,
                apply_routed_scaling_factor=not layer.should_fuse_routed_scaling_factor_in_topk,
            )
            return self.runner.run(dispatch_output, quant_info)
        elif self.use_flashinfer_trtllm_moe:
            from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
                FlashInferTrtllmBf16MoeQuantInfo,
            )

            quant_info = FlashInferTrtllmBf16MoeQuantInfo(
                gemm1_weights=layer.w13_weight,
                gemm2_weights=layer.w2_weight,
                global_num_experts=layer.num_experts,
                local_expert_offset=layer.moe_ep_rank * layer.num_local_experts,
            )
            return self.runner.run(dispatch_output, quant_info)
        else:
            if self._aiter_runner is not None:
                from sglang.srt.layers.moe.moe_runner.aiter import (
                    AiterMoeQuantInfo,
                )

                quant_info = AiterMoeQuantInfo(
                    w13_weight=layer.w13_weight,
                    w2_weight=layer.w2_weight,
                    expert_mask=layer.dispatcher.expert_mask_gpu,
                )
                return self._aiter_runner.run(dispatch_output, quant_info)

            quant_info = TritonMoeQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                b13=getattr(layer, "w13_weight_bias", None),
                b2=getattr(layer, "w2_weight_bias", None),
                fuse_swiglu_interleaved=self.w13_swiglu_interleaved,
            )
            return self.runner.run(dispatch_output, quant_info)

    def forward_cpu(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        moe_runner_config = self.moe_runner_config

        if use_intel_amx_backend(layer):
            from sglang.srt.layers.moe.topk import apply_topk_weights_cpu

            topk_weights, topk_ids, _ = topk_output
            x, topk_weights = apply_topk_weights_cpu(
                moe_runner_config.apply_router_weight_on_input, topk_weights, x
            )
            output = torch.ops.sgl_kernel.fused_experts_cpu(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                False,
                CPUQuantMethod.UNQUANT,
                None,
                None,
                None,
                None,
                None,
                getattr(layer, "w13_weight_bias", None),
                getattr(layer, "w2_weight_bias", None),
                layer.moe_runner_config.gemm1_alpha,
                layer.moe_runner_config.gemm1_clamp_limit,
                True,
                moe_runner_config.activation,
            )
            return StandardCombineInput(hidden_states=output)
        else:
            from sglang.srt.layers.moe.fused_moe_native import moe_forward_native

            output = moe_forward_native(
                layer,
                x,
                topk_output,
                moe_runner_config,
            )
            return StandardCombineInput(hidden_states=output)

    def get_triton_quant_info(self, layer: torch.nn.Module) -> TritonMoeQuantInfo:
        return TritonMoeQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            b13=getattr(layer, "w13_weight_bias", None),
            b2=getattr(layer, "w2_weight_bias", None),
            fuse_swiglu_interleaved=self.w13_swiglu_interleaved,
        )

    def forward_xpu(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        moe_runner_config = self.moe_runner_config
        assert moe_runner_config.activation in [
            "silu",
            "gelu",
            "relu2",
        ], f"activation = {moe_runner_config.activation} is not supported."

        backend = self.runner.runner_backend
        if use_intel_xpu_backend():

            from sgl_kernel import fused_experts

            topk_weights, topk_ids, _ = topk_output
            if moe_runner_config.apply_router_weight_on_input:
                x = x * topk_weights.to(x.dtype)
                topk_weights = torch.ones_like(topk_weights)
            output = fused_experts(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                b1=getattr(layer, "w13_weight_bias", None),
                b2=getattr(layer, "w2_weight_bias", None),
                activation=moe_runner_config.activation,
                gemm1_alpha=moe_runner_config.gemm1_alpha,
                gemm1_limit=moe_runner_config.gemm1_clamp_limit,
            )
            return StandardCombineInput(hidden_states=output)
        else:
            assert backend.is_triton()
            assert (
                moe_runner_config.activation == "silu"
            ), f"activation = {moe_runner_config.activation} is not supported \
            for Triton PATH, please set ENV SGLANG_USE_SGL_XPU=1."

            quant_info = self.get_triton_quant_info(layer)
            return self.runner.run(dispatch_output, quant_info)

    def forward_npu(
        self,
        layer: torch.nn.Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:

        return self.runner.run(dispatch_output, layer)

    def forward_tpu(self, *args, **kwargs) -> CombineInput:
        raise NotImplementedError("The TPU backend currently does not support MoE.")

    def forward_musa(self, *args, **kwargs) -> CombineInput:
        return self.forward_cuda(*args, **kwargs)

    forward_native = forward_cpu
