from __future__ import annotations

import concurrent.futures
import functools
import logging
import time
from contextlib import contextmanager, nullcontext
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
import torch.nn.functional as F

import sglang.srt.models.deepseek_v2 as deepseek_v2
from sglang.kernels.ops.attention.dsv4 import (
    fused_norm_rope_inplace,
    fused_q_norm_rope,
    fused_rope_inplace,
    sglang_per_token_group_quant_fp8_dsv4_wo_a,
)
from sglang.kernels.ops.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
)
from sglang.srt.compilation.compilation_config import register_split_op
from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.distributed import (
    get_pp_group,
    get_tp_group,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.hardware_backend.npu.dsv4.dsv4_rope import Dsv4NpuRoPE
from sglang.srt.layers.attention.dsa.utils import (
    can_dsa_cp_split,
    dsa_use_prefill_cp,
    is_dsa_enable_prefill_cp,
    is_dsa_prefill_cp_round_robin_split,
)
from sglang.srt.layers.attention.dsv4.compressor import Compressor
from sglang.srt.layers.attention.dsv4.indexer import C4Indexer
from sglang.srt.layers.communicator import get_attn_tp_context
from sglang.srt.layers.communicator_dsa_cp import (
    dsa_cp_gather_hidden_states,
    dsa_cp_reduce_scatter_hidden_states,
)
from sglang.srt.layers.cp.cp_decode_attn_tp import get_cp_decode_attn_tp_ctx
from sglang.srt.layers.cp.utils import (
    cp_materialize_global_token_order,
    cp_round_robin_input_ids_v2,
    is_cp_v2_active,
)
from sglang.srt.layers.dp_attention import (
    _tbo_event,
    attn_cp_overlap_all_gather_into_tensor,
    attn_cp_overlap_reduce_scatter_tensor,
    attn_tp_all_gather,
    attn_tp_all_reduce,
    dp_gather_partial,
    dp_gather_replicate,
    dp_reduce_scatter_tensor,
    dp_reduce_scatterv_async,
    dp_scatter,
    get_dp_global_num_tokens,
    get_dp_tbo_comm_stream,
    get_global_dp_buffer,
    get_global_dp_buffer_len,
    get_local_dp_buffer,
    get_local_dp_buffer_len,
    get_tbo_persistent_buffer,
    is_allocation_symmetric,
    is_dp_attention_enabled,
    is_dp_gatherv_active,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import get_moe_a2a_backend, should_use_dp_reduce_scatterv
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.utils import (
    is_shared_experts_fusion_disabled,
    uses_per_rank_fused_shared_slots,
)
from sglang.srt.layers.quantization.fp8_utils import (
    view_aiter_fused_rms_transposed_fp8_scale,
)
from sglang.srt.layers.rotary_embedding import get_rope_wrapper
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.utils.cp_utils import (
    cp_all_gather_rerange_finish,
    cp_all_gather_rerange_launch,
    cp_all_gather_rerange_output,
    cp_round_robin_input_ids,
    cp_split_and_rebuild_data,
    cp_split_and_rebuild_position,
    prepare_context_parallel_metadata,
)
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.mem_cache.memory_pool import RadixAttention
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.model_executor.runner import (
    compile_in_capture_mode,
    get_is_capture_mode,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
    eager_on_graph,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    get_tc_piecewise_forward_context,
)
from sglang.srt.model_loader.utils import maybe_executor_submit, should_async_load
from sglang.srt.model_loader.weight_utils import (
    RUNAI_STREAMER_TENSOR_ATTR,
    default_weight_loader,
)
from sglang.srt.models.dbrx import ReplicatedLinear
from sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc import (
    apply_mhc_post_pre_boundary,
    is_cross_layer_mhc_fusion_enabled,
)
from sglang.srt.models.deepseek_common.utils import (
    _use_aiter_bpreshuffle_gfx95,
    is_wint4afp8_or_wint4a16_config,
    quant_blocks_shared_experts_fusion,
)
from sglang.srt.models.deepseek_v2 import (
    ParallelLMHead,
    _is_cuda,
    _is_hip,
    _is_npu,
    _is_xpu,
)
from sglang.srt.runtime_context import get_device, get_exec, get_forward, get_parallel

if not _is_hip:
    from sglang.srt.layers.utils.cp_utils import (
        prepare_context_parallel_metadata,
    )

from sglang.srt.utils import (
    LazyValue,
    add_prefix,
    get_bool_env_var,
    is_gfx95_supported,
    is_gfx942_supported,
    log_info_on_rank0,
    make_layers,
)
from sglang.srt.utils.custom_op import register_custom_op
from sglang.srt.utils.hf_transformers_utils import get_rope_config


if _is_npu:
    import torch_npu


class MhcOps(NamedTuple):
    hc_split_sinkhorn: Callable[..., Any]
    mhc_fused_post_pre: Optional[Callable[..., Any]]
    npu_hc_pre: Optional[Callable[..., Any]]


@functools.cache
def _get_mhc_ops() -> MhcOps:

    if _is_xpu:
        from sgl_kernel import hc_split_sinkhorn

        return MhcOps(hc_split_sinkhorn, None, None)

    from sglang.kernels.ops.layernorm.mhc import (
        hc_split_sinkhorn,
        mhc_fused_post_pre,
        npu_hc_pre,
    )

    return MhcOps(hc_split_sinkhorn, mhc_fused_post_pre, npu_hc_pre)


logger = logging.getLogger(__name__)

_FP8_WO_A_GEMM = envs.SGLANG_OPT_FP8_WO_A_GEMM.get()
_MHC_POST_MULT_VALUE = 2.0

DEEPSEEK_V4_STACKED_PARAMS_MAPPING: List[Tuple[str, str, int]] = [
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
]


_FLASHINFER_MHC_PRE_SPLITS = (1, 2, 4, 8, 16)


@functools.cache
def _cuda_sm_count() -> int:
    return torch.cuda.get_device_properties(0).multi_processor_count


def _flashinfer_mhc_pre_num_splits(num_tokens: int, hc_hidden_size: int) -> int:
    block_m = block_k = 64
    grid_m = (num_tokens + block_m - 1) // block_m
    num_block_k = (hc_hidden_size + block_k - 1) // block_k
    raw = max(1, min(_cuda_sm_count() // max(grid_m, 1), num_block_k // 4))
    best = 1
    for split in _FLASHINFER_MHC_PRE_SPLITS:
        if split <= raw:
            best = split
    return best


def _flashinfer_hc_pre(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from flashinfer.mhc import mhc_pre_big_fuse

    from sglang.srt.layers.deep_gemm_wrapper.entrypoint import tf32_hc_prenorm_gemm

    num_tokens, hc_mult, hidden_size = x.shape
    hc_hidden_size = hc_mult * hidden_size
    mix_dim = hc_fn.shape[0]
    n_splits = _flashinfer_mhc_pre_num_splits(num_tokens, hc_hidden_size)

    dot_mix = torch.empty(
        (n_splits, num_tokens, mix_dim), dtype=torch.float32, device=x.device
    )
    sqrsum = torch.empty((n_splits, num_tokens), dtype=torch.float32, device=x.device)
    tf32_hc_prenorm_gemm(
        x.reshape(num_tokens, hc_hidden_size), hc_fn, dot_mix, sqrsum, n_splits
    )
    if n_splits == 1:
        dot_mix = dot_mix.squeeze(0)
        sqrsum = sqrsum.squeeze(0)

    post, comb, layer_input = mhc_pre_big_fuse(
        dot_mix,
        sqrsum,
        x,
        hc_scale,
        hc_base,
        hc_hidden_size,
        rms_eps=rms_eps,
        mhc_pre_eps=hc_eps,
        mhc_sinkhorn_eps=hc_eps,
        mhc_post_mult_value=_MHC_POST_MULT_VALUE,
        sinkhorn_repeat=sinkhorn_iters,
        num_splits=n_splits,
    )
    return layer_input, post.squeeze(-1), comb


_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip


_SHARED_EXPERT_LOCAL = get_bool_env_var("SGLANG_DP_SHARED_EXPERT_LOCAL")
_is_gfx95_supported = is_gfx95_supported()
_is_gfx942_supported = is_gfx942_supported()

if _use_aiter:
    if _is_gfx95_supported:
        from aiter.ops.triton.fused_fp8_quant import fused_rms_fp8_group_quant


def _wo_a_aiter_gemm_eligible(
    flag: bool, use_aiter: bool, is_hip: bool, is_gfx95: bool
) -> bool:

    return bool(flag and use_aiter and is_hip and is_gfx95)


_wo_a_aiter_batched_gemm_enabled = _wo_a_aiter_gemm_eligible(
    envs.SGLANG_OPT_USE_AITER_BATCHED_GEMM.get(),
    _use_aiter,
    _is_hip,
    _is_gfx95_supported,
)
_wo_a_batched_gemm_bf16 = None
if _wo_a_aiter_batched_gemm_enabled:
    try:
        from aiter.ops.triton.gemm.batched.batched_gemm_bf16 import (
            batched_gemm_bf16 as _wo_a_batched_gemm_bf16,
        )
    except Exception as err:
        _wo_a_aiter_batched_gemm_enabled = False
        logger.warning(
            "aiter wo_a batched_gemm_bf16 import failed; using einsum for wo_a "
            "for the rest of this process: %s",
            err,
        )


_wo_a_aiter_batched_gemm_disabled = False


def _apply_wo_a_bf16_matmul(
    o: torch.Tensor, wo_a: torch.Tensor, is_decode: bool
) -> torch.Tensor:

    global _wo_a_aiter_batched_gemm_disabled
    if (
        is_decode
        and _wo_a_aiter_batched_gemm_enabled
        and not _wo_a_aiter_batched_gemm_disabled
    ):
        try:


            xq = o.transpose(0, 1).contiguous()
            y = _wo_a_batched_gemm_bf16(xq, wo_a, dtype=torch.bfloat16)
            return y.transpose(0, 1).contiguous()
        except Exception as err:
            _wo_a_aiter_batched_gemm_disabled = True
            logger.warning(
                "aiter wo_a batched_gemm_bf16 failed; disabling the reroute and "
                "falling back to einsum for the rest of this process: %s",
                err,
            )
    return torch.einsum("tgd,grd->tgr", o, wo_a)


def _fused_rmsnorm_fp8_quant(hidden_states, weight, eps):
    x_quant, x_bf16, _, _ = fused_rms_fp8_group_quant(
        hidden_states,
        weight,
        eps,
        inp2=None,
        inp2_weight=None,
        inp2_epsilon=None,
        group_size=128,
        dtype_quant=torch.float8_e4m3fn,
        res1=None,
        output_unquantized_inp1=True,
        transpose_scale=_use_aiter_bpreshuffle_gfx95,
    )
    if _use_aiter_bpreshuffle_gfx95:
        x_quant = (
            x_quant[0],
            view_aiter_fused_rms_transposed_fp8_scale(x_quant[1]),
        )
    return x_quant, x_bf16


def make_hc_mixing_params(
    hc_mult: int, hidden_size: int
) -> Tuple[
    nn.Parameter, nn.Parameter, nn.Parameter, nn.Parameter, nn.Parameter, nn.Parameter
]:
    mix_hc = (2 + hc_mult) * hc_mult
    hc_dim = hc_mult * hidden_size
    return (
        nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32)),
        nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32)),
        nn.Parameter(torch.empty(mix_hc, dtype=torch.float32)),
        nn.Parameter(torch.empty(mix_hc, dtype=torch.float32)),
        nn.Parameter(torch.empty(3, dtype=torch.float32)),
        nn.Parameter(torch.empty(3, dtype=torch.float32)),
    )


def make_hc_head_params(
    hc_mult: int, hidden_size: int
) -> Tuple[nn.Parameter, nn.Parameter, nn.Parameter]:
    hc_dim = hc_mult * hidden_size
    return (
        nn.Parameter(torch.empty(hc_mult, hc_dim, dtype=torch.float32)),
        nn.Parameter(torch.empty(hc_mult, dtype=torch.float32)),
        nn.Parameter(torch.empty(1, dtype=torch.float32)),
    )


def hc_head_torch(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    shape, dtype = x.size(), x.dtype
    x = x.flatten(-2).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x, hc_fn) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=-2)
    return y.to(dtype)


_FREQS_CIS_TO_COS_SIN: dict[
    Tuple[int, torch.dtype, torch.device], Tuple[torch.Tensor, torch.Tensor]
] = {}


def _freqs_cis_to_cos_sin(
    freqs_cis: torch.Tensor, dtype: torch.dtype, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:

    key = (id(freqs_cis), dtype, device)
    cached = _FREQS_CIS_TO_COS_SIN.get(key)
    if cached is not None:
        return cached
    fr = torch.view_as_real(freqs_cis)
    cos = fr[..., 0].to(device=device, dtype=dtype).contiguous()
    sin = fr[..., 1].to(device=device, dtype=dtype).contiguous()
    _FREQS_CIS_TO_COS_SIN[key] = (cos, sin)
    return cos, sin


def _apply_gguf_grouped_wo_a(
    o: torch.Tensor,
    qweight: torch.Tensor,
    qweight_type: int,
    o_lora_rank: int,
    matmul_fn: Optional[Callable] = None,
) -> torch.Tensor:
    if matmul_fn is None:
        from sglang.srt.layers.quantization.gguf import fused_mul_mat_gguf

        matmul_fn = fused_mul_mat_gguf

    group_outputs = []
    for group_id in range(o.shape[1]):
        start = group_id * o_lora_rank
        group_outputs.append(
            matmul_fn(
                o[:, group_id, :].contiguous(),
                qweight[start : start + o_lora_rank],
                qweight_type,
            )
        )
    return torch.stack(group_outputs, dim=1)


if TYPE_CHECKING:
    from sglang.srt.layers.attention.deepseek_v4_backend import (
        DeepseekV4AttnBackend,
    )
    from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
        DeepseekV4HipRadixBackend,
    )
    from sglang.srt.layers.quantization import QuantizationConfig
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@register_custom_op(mutates_args=["output"])
@register_split_op()
def deepseek_v4_attention_with_output(
    query: torch.Tensor,
    key_value: torch.Tensor,
    output: torch.Tensor,
    layer_id: int,
    compress_ratio: int,
    attn_sink: torch.Tensor,
    save_kv_cache: bool,
) -> None:
    context = get_tc_piecewise_forward_context()
    forward_batch = context.forward_batch
    attention_layers = context.attention_layers
    attention_layer = attention_layers[layer_id]
    real_num_tokens = forward_batch.num_token_non_padded_cpu

    query = query[:real_num_tokens]
    key_value = key_value[:real_num_tokens]

    original_out_cache_loc = forward_batch.out_cache_loc
    forward_batch.out_cache_loc = original_out_cache_loc[:real_num_tokens]

    attn_backend = get_attn_backend()
    try:
        ret = attn_backend.forward(
            q=query,
            k=key_value,
            v=key_value,
            layer=attention_layer,
            forward_batch=forward_batch,
            compress_ratio=compress_ratio,
            attn_sink=attn_sink,
            save_kv_cache=save_kv_cache,
        )
    finally:
        forward_batch.out_cache_loc = original_out_cache_loc

    assert (
        output[:real_num_tokens].numel() == ret.numel()
    ), f"Output tensor element mismatch: {output[:real_num_tokens].numel()} != {ret.numel()}"

    output[:real_num_tokens].view(ret.shape).copy_(ret)
    return


bcg_deepseek_v4_attention_with_output = eager_on_graph(True)(
    deepseek_v4_attention_with_output
)


class MqaAttentionBase(nn.Module):

    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
        *,
        attn_tp_rank: Optional[int] = None,
        attn_tp_size: Optional[int] = None,
        compress_ratio: Optional[int] = None,
        fuse_wqa_wkv: Optional[bool] = None,
        wo_a_fp8: Optional[bool] = None,
        wo_a_keeps_quant_config: Optional[bool] = None,
        wo_b_reduce_results: Optional[bool] = None,
        rope_original_seq_len: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
        if attn_tp_rank is None or attn_tp_size is None:
            attn_tp_rank = get_parallel().attn_tp_rank
            attn_tp_size = get_parallel().attn_tp_size
            if self.dsa_enable_prefill_cp:
                self.cp_size = get_parallel().attn_cp_size
                attn_tp_rank, attn_tp_size = 0, 1
        self.attn_tp_rank: int = attn_tp_rank
        self.attn_tp_size: int = attn_tp_size

        self.layer_id = layer_id
        self.dim = config.hidden_size
        self.hidden_size = config.hidden_size
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.head_dim - config.qk_rope_head_dim
        self.head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.n_heads = config.num_attention_heads
        self.n_local_heads = self.n_heads // self.attn_tp_size
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // self.attn_tp_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.eps = config.rms_norm_eps
        self.softmax_scale = self.head_dim**-0.5

        self.compress_ratio: int = (
            compress_ratio
            if compress_ratio is not None
            else config.compress_ratios[layer_id]
        )
        assert self.compress_ratio in (
            0,
            4,
            128,
        ), f"V4 compress_ratio: expected one of (0, 4, 128), got {self.compress_ratio}"

        assert self.head_dim == config.head_dim
        assert config.num_key_value_heads == 1

        fuse: bool = (
            envs.SGLANG_OPT_FUSE_WQA_WKV.get() if fuse_wqa_wkv is None else fuse_wqa_wkv
        )
        fp8: bool = _FP8_WO_A_GEMM if wo_a_fp8 is None else wo_a_fp8
        reduce_results: bool = (
            (self.attn_tp_size == get_parallel().tp_size and self.attn_tp_size > 1)
            if wo_b_reduce_results is None
            else wo_b_reduce_results
        )
        if wo_a_keeps_quant_config is None:
            keep_source_quant = (
                quant_config is not None and quant_config.get_name() == "expert_pack"
            )
            wo_a_quant_config: Optional[QuantizationConfig] = (
                quant_config if fp8 or keep_source_quant else None
            )
        elif wo_a_keeps_quant_config:
            wo_a_quant_config = quant_config
        else:
            wo_a_quant_config = None

        self.fuse_wqa_wkv = fuse

        self.attn_sink = nn.Parameter(torch.empty(self.n_heads, dtype=torch.float32))
        self._attn_sink_local: Optional[torch.Tensor] = None
        if fuse:
            self.wqkv_a = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank + self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("wqkv_a", prefix),
            )
        else:
            self.wq_a = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("wq_a", prefix),
            )
            self.wkv = ReplicatedLinear(
                self.hidden_size,
                self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("wkv", prefix),
            )
        self.q_norm = RMSNorm(self.q_lora_rank, eps=self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
        )
        self.kv_norm = RMSNorm(self.head_dim, eps=self.eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=wo_a_quant_config,
            prefix=add_prefix("wo_a", prefix),
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            **({} if fp8 else {"params_dtype": torch.bfloat16}),
        )
        if fp8:
            from sglang.srt.layers import deep_gemm_wrapper

            assert hasattr(
                self.wo_a, "weight_scale_inv"
            ), "FP8 quant_config must create weight_scale_inv"
            self.wo_a.weight_scale_inv.format_ue8m0 = (
                deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
            )
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("wo_b", prefix),
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
        )

        from sglang.kernels.ops.attention.deepseek_v4_rope import precompute_freqs_cis

        rope_theta, rope_scaling = get_rope_config(config)
        self.rope_scaling = dict(rope_scaling) if rope_scaling else None
        scaling = self.rope_scaling or {}


        self.rope_base = (
            config.compress_rope_theta if self.compress_ratio else rope_theta
        )
        original_seq_len: int = (
            rope_original_seq_len
            if rope_original_seq_len is not None
            else (
                scaling["original_max_position_embeddings"]
                if self.compress_ratio
                else 0
            )
        )
        freqs_cis = precompute_freqs_cis(
            dim=self.qk_rope_head_dim,
            seqlen=config.max_position_embeddings,
            original_seq_len=original_seq_len,
            base=self.rope_base,
            factor=scaling.get("factor", 1.0),
            beta_fast=scaling.get("beta_fast", 32),
            beta_slow=scaling.get("beta_slow", 1),
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        self.freqs_cis: torch.Tensor

    def _local_attn_sink(self) -> torch.Tensor:
        if self.attn_tp_size == 1:
            return self.attn_sink
        if self._attn_sink_local is None:
            rank = self.attn_tp_rank
            num_heads = self.n_local_heads
            padded_num_heads = 64 if num_heads <= 64 else self.n_heads
            sink = self.attn_sink.new_zeros(padded_num_heads)
            sink[:num_heads] = self.attn_sink[rank * num_heads : (rank + 1) * num_heads]
            self._attn_sink_local = sink
        return self._attn_sink_local

    @contextmanager
    def maybe_use_decode_attn_tp(self, forward_batch: ForwardBatch):
        ctx = get_cp_decode_attn_tp_ctx()
        attn = self.attn_mqa if isinstance(self, MQALayer) else self.attn
        with ctx.maybe_use_decode_attn_tp(
            forward_batch,
            [self.wq_b, self.wo_a, self.wo_b],
            radix_attn=attn,
        ):
            if ctx.use_decode_attn_tp:
                orig = (
                    self.n_local_heads,
                    self.n_local_groups,
                    self.attn_tp_rank,
                    self.attn_tp_size,
                )
                decode_tp_size = ctx.decode_tp_size
                self.n_local_heads = self.n_heads // decode_tp_size
                self.n_local_groups = self.n_groups // decode_tp_size
                self.attn_tp_rank = ctx.decode_tp_rank
                self.attn_tp_size = decode_tp_size
                try:
                    yield
                finally:
                    (
                        self.n_local_heads,
                        self.n_local_groups,
                        self.attn_tp_rank,
                        self.attn_tp_size,
                    ) = orig
            else:
                yield


class MQALayer(MqaAttentionBase):
    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        compress_ratio_override: Optional[int] = None,
    ) -> None:
        super().__init__(
            config,
            layer_id,
            quant_config,
            prefix,
            compress_ratio=compress_ratio_override,
        )

        active_rope_scaling = None
        if self.compress_ratio in (4, 128):
            active_rope_scaling = dict(self.rope_scaling or {})
            active_rope_scaling["rope_type"] = "deepseek_yarn"
        self.rotary_emb = get_rope_wrapper(
            head_size=self.rope_head_dim,
            rotary_dim=self.rope_head_dim,
            max_position=config.max_position_embeddings,
            base=self.rope_base,
            rope_scaling=active_rope_scaling,
            is_neox_style=False,
            device=get_device().device,
        )

        if _is_npu:
            Dsv4NpuRoPE.for_freqs(
                self.freqs_cis, getattr(self, "rotary_emb", None)
            ).ensure_tables(torch.float32)

        if _is_hip:
            cos_cache = (
                self.freqs_cis.real.to(torch.bfloat16).unsqueeze(-2).unsqueeze(-2)
            )
            sin_cache = (
                self.freqs_cis.imag.to(torch.bfloat16).unsqueeze(-2).unsqueeze(-2)
            )
            self.register_buffer("cos_cache", cos_cache, persistent=False)
            self.register_buffer("sin_cache", sin_cache, persistent=False)

        if alt_streams is not None and (
            (_is_cuda and envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get())
            or (_is_npu and envs.SGLANG_NPU_USE_MULTI_STREAM.get())
        ):
            self.alt_streams = alt_streams[:3]
            self.alt_streams_indexer = alt_streams[-2:]
        else:
            self.alt_streams = None
            self.alt_streams_indexer = None

        from sglang.srt.utils import is_blackwell_supported

        self._multi_stream_bs_limit = 128 if is_blackwell_supported() else 64

        self.compressor = None
        self.indexer = None
        if self.compress_ratio in (4, 128):
            expert_pack_quant_config = (
                quant_config
                if quant_config is not None and quant_config.get_name() == "expert_pack"
                else None
            )
            self.compressor = Compressor(
                config,
                layer_id=self.layer_id,
                is_in_indexer=False,
                freqs_cis=self.freqs_cis,
                compress_ratio=self.compress_ratio,
                head_dim=self.head_dim,
                rotate=False,
                prefix=add_prefix("compressor", prefix),
                quant_config=expert_pack_quant_config,
                rotary_emb=self.rotary_emb,
            )
            if self.compress_ratio == 4:
                self.indexer = C4Indexer(
                    config,
                    freqs_cis=self.freqs_cis,
                    layer_id=layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix("indexer", prefix),
                    alt_streams=self.alt_streams_indexer,
                    rotary_emb=self.rotary_emb,
                )

        self.attn_mqa = RadixAttention(
            self.n_local_heads,
            self.head_dim,
            self.softmax_scale,
            num_kv_heads=1,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn_mqa", prefix),
        )

        self.use_fused_qk_norm_rope = (
            _is_hip and envs.SGLANG_OPT_USE_FUSED_QK_NORM_ROPE.get()
        )


    def _get_npu_rope_position_cache(
        self, positions: torch.Tensor, dtype: torch.dtype, inverse: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:


        return Dsv4NpuRoPE.for_freqs(
            self.freqs_cis, getattr(self, "rotary_emb", None)
        ).get_cos_sin(
            positions,
            dtype,
            view_4d=True,
            inverse=inverse,
            allow_build=False,
            cache_dtype=torch.float32,
        )

    def _compute_q_a(
        self,
        x: torch.Tensor,
        qkv_a: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if qkv_a is not None:
            q = qkv_a[..., : self.q_lora_rank]
        else:
            q, _ = self.wq_a(x)
        return self.q_norm(q)

    def _compute_q_b(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        q_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q, _ = self.wq_b(q)
        q = q.view(-1, self.n_local_heads, self.head_dim)
        if q_out is None:
            q_out = torch.empty_like(q)

        fused_q_norm_rope(q, q_out, self.eps, self.freqs_cis, positions)
        return q_out

    def _compute_kv_to_cache(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend,
        qkv_a: Optional[torch.Tensor] = None,
    ) -> None:

        if envs.SGLANG_DSV4_USE_BF16_KV_QUANT_SOURCE.get():


            kv = self._compute_kv_bf16(x, positions, qkv_a=qkv_a)
            attn_backend.store_cache(
                layer_id=self.layer_id, swa_k=kv, forward_batch=forward_batch
            )
            return
        if qkv_a is not None:
            kv = qkv_a[..., self.q_lora_rank :]
        else:
            kv, _ = self.wkv(x)
        token_to_kv_pool = get_token_to_kv_pool()
        if TYPE_CHECKING:
            assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
        token_to_kv_pool.set_swa_key_buffer_radix_fused_norm_rope(
            layer_id=self.layer_id,
            swa_loc=attn_backend.get_swa_out_cache_loc(forward_batch),
            kv=kv,
            kv_weight=self.kv_norm.weight.data,
            eps=self.eps,
            freqs_cis=self.freqs_cis,
            positions=positions,
        )

    def _compute_kv_bf16(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        qkv_a: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if qkv_a is not None:
            kv = qkv_a[..., self.q_lora_rank :]
        else:
            kv, _ = self.wkv(x)
        kv = kv.contiguous()
        fused_norm_rope_inplace(
            kv,
            self.kv_norm.weight.data,
            self.eps,
            self.freqs_cis,
            positions,
        )
        return kv

    def _forward_prepare_multi_stream(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend,
        q_out: Optional[torch.Tensor] = None,
        x_quant=None,
    ) -> torch.Tensor:
        assert self.alt_streams is not None
        assert len(self.alt_streams) >= 3

        current_stream = torch.cuda.current_stream()
        stream_kv = self.alt_streams[0]
        stream_compressor = self.alt_streams[1]
        stream_indexer = self.alt_streams[2]

        stream_kv.wait_stream(current_stream)
        stream_compressor.wait_stream(current_stream)
        stream_indexer.wait_stream(current_stream)

        x_linear = x_quant if x_quant is not None else x
        qkv_a: Optional[torch.Tensor] = None
        qkv_a_ready: Optional[torch.cuda.Event] = None
        if self.fuse_wqa_wkv:
            qkv_a, _ = self.wqkv_a(x_linear)
            qkv_a_ready = current_stream.record_event()

        q_lora = self._compute_q_a(x_linear, qkv_a=qkv_a)
        q_lora_ready = current_stream.record_event()

        if self.indexer is not None:
            with torch.cuda.stream(stream_indexer):
                self.indexer(
                    x=x,
                    q_lora=q_lora,
                    forward_batch=forward_batch,
                    attn_backend=attn_backend,
                    enable_multi_stream=True,
                    q_lora_ready=q_lora_ready,
                )

        with torch.cuda.stream(stream_kv):
            if qkv_a_ready is not None:
                stream_kv.wait_event(qkv_a_ready)

            self._compute_kv_to_cache(
                x_linear, positions, forward_batch, attn_backend, qkv_a=qkv_a
            )

        del qkv_a

        if self.compressor is not None:
            with torch.cuda.stream(stream_compressor):
                attn_backend.forward_core_compressor(
                    x, forward_batch, self.layer_id, self.compressor
                )

        q = self._compute_q_b(q_lora, positions, q_out)
        current_stream.wait_stream(stream_kv)
        current_stream.wait_stream(stream_compressor)
        current_stream.wait_stream(stream_indexer)

        return q

    def _forward_prepare_multi_stream_npu(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend,
        q_out: Optional[torch.Tensor] = None,
        x_quant=None,
    ) -> torch.Tensor:


        assert self.alt_streams is not None
        current_stream = torch.npu.current_stream()
        stream_kv = self.alt_streams[0]
        stream_q = self.alt_streams[1]
        stream_kv.wait_stream(current_stream)
        stream_q.wait_stream(current_stream)

        x_linear = x_quant if x_quant is not None else x
        qkv_a: Optional[torch.Tensor] = None
        qkv_a_ready = None
        if self.fuse_wqa_wkv:
            qkv_a, _ = self.wqkv_a(x_linear)
            qkv_a_ready = current_stream.record_event()
        if qkv_a is not None:
            q_lora = qkv_a[..., : self.q_lora_rank]
        else:
            q_lora, _ = self.wq_a(x_linear)
        q_lora = self.q_norm(q_lora)
        q_lora_ready = current_stream.record_event()


        with torch.npu.stream(stream_kv):
            if qkv_a_ready is not None:
                stream_kv.wait_event(qkv_a_ready)
            if qkv_a is not None:
                kv = qkv_a[..., self.q_lora_rank :]
            else:
                kv, _ = self.wkv(x)
            kv = self.kv_norm(kv)
            cos4_k, sin4_k = self._get_npu_rope_position_cache(
                positions, kv.dtype, inverse=False
            )
            Dsv4NpuRoPE.apply_rotary_mul_inplace(
                kv.unsqueeze(1),
                None,
                cos4_k,
                sin4_k,
                qk_nope_dim=self.qk_nope_head_dim,
            )
            attn_backend.store_cache(
                layer_id=self.layer_id,
                swa_k=kv,
                forward_batch=forward_batch,
            )


        with torch.npu.stream(stream_q):
            stream_q.wait_event(q_lora_ready)
            q, _ = self.wq_b(q_lora)
            q = q.view(-1, self.n_local_heads, self.head_dim)
            _dummy = q.new_ones(q.shape[-1])
            q = torch_npu.npu_rms_norm(q, _dummy, self.eps)[0]
            cos4_q, sin4_q = self._get_npu_rope_position_cache(
                positions, q.dtype, inverse=False
            )
            Dsv4NpuRoPE.apply_rotary_mul_inplace(
                q,
                None,
                cos4_q,
                sin4_q,
                qk_nope_dim=self.qk_nope_head_dim,
            )
            if q_out is not None:
                q_out.copy_(q)
            q.record_stream(stream_q)

        del qkv_a


        if self.indexer is not None:
            self.indexer(
                x=x,
                q_lora=q_lora,
                forward_batch=forward_batch,
                attn_backend=attn_backend,
            )
        if self.compressor is not None:
            attn_backend.forward_core_compressor(
                x,
                forward_batch,
                self.layer_id,
                self.compressor,
            )


        current_stream.wait_stream(stream_kv)
        current_stream.wait_stream(stream_q)
        return q

    def _forward_prepare_multi_stream_hip(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend,
        q_out: Optional[torch.Tensor] = None,
        x_quant=None,
    ) -> torch.Tensor:

        assert self.alt_streams is not None
        assert len(self.alt_streams) >= 1

        current_stream = torch.cuda.current_stream()
        stream_compressor = self.alt_streams[0]
        stream_indexer_compressor = (
            self.alt_streams[1] if len(self.alt_streams) > 1 else None
        )

        if self.compressor is not None:
            stream_compressor.wait_stream(current_stream)
            with torch.cuda.stream(stream_compressor):
                attn_backend.forward_core_compressor(
                    x, forward_batch, self.layer_id, self.compressor
                )

        if self.indexer is not None and stream_indexer_compressor is not None:
            stream_indexer_compressor.wait_stream(current_stream)
            with torch.cuda.stream(stream_indexer_compressor):
                attn_backend.forward_indexer_compressor(
                    x=x,
                    forward_batch=forward_batch,
                    layer_id=self.indexer.layer_id,
                    compressor=self.indexer.compressor,
                )

        x_linear = x_quant if x_quant is not None else x
        if self.fuse_wqa_wkv:
            qkv_a, _ = self.wqkv_a(x_linear)
            q_lora = qkv_a[..., : self.q_lora_rank]
        else:
            q_lora, _ = self.wq_a(x_linear)
            qkv_a = None

        if self.use_fused_qk_norm_rope:
            if _is_gfx95_supported:
                q_for_wqb, q_lora = _fused_rmsnorm_fp8_quant(
                    q_lora,
                    self.q_norm.weight,
                    self.q_norm.variance_epsilon,
                )
                q, _ = self.wq_b(q_for_wqb)
            else:
                q_lora = self.q_norm(q_lora)
                q, _ = self.wq_b(q_lora)

            kv = (
                qkv_a[..., self.q_lora_rank :]
                if qkv_a is not None
                else self.wkv(x_linear)[0]
            )

            from sglang.kernels.ops.attention.fused_qk_norm_rope_store import (
                fused_qk_norm_rope_swa_store,
            )

            token_to_kv_pool = get_token_to_kv_pool()
            swa_loc = attn_backend.get_swa_out_cache_loc(forward_batch)
            swa_cache = token_to_kv_pool.get_swa_raw_buffer(self.layer_id)
            swa_page_size = token_to_kv_pool.swa_kv_pool.page_size

            q = fused_qk_norm_rope_swa_store(
                q=q,
                kv=kv,
                q_norm_weight=None,
                kv_norm_weight=self.kv_norm.weight,
                q_rms_eps=self.eps,
                kv_rms_eps=self.eps,
                rope_head_dim=self.qk_rope_head_dim,
                cos_cache=self.cos_cache,
                sin_cache=self.sin_cache,
                positions=positions,
                swa_cache=swa_cache,
                swa_loc=swa_loc,
                swa_page_size=swa_page_size,
                q_out=q_out,
                dtype=x.dtype,
            )
        else:
            q_lora = self.q_norm(q_lora)
            q = self._compute_q_b(q_lora, positions, q_out)
            self._compute_kv_to_cache(
                x_linear, positions, forward_batch, attn_backend, qkv_a=qkv_a
            )

        del qkv_a

        if self.indexer is not None:
            current_stream.wait_stream(stream_compressor)
            if stream_indexer_compressor is not None:
                current_stream.wait_stream(stream_indexer_compressor)
            self.indexer(
                x=x,
                q_lora=q_lora,
                forward_batch=forward_batch,
                attn_backend=attn_backend,
                skip_compressor=True,
            )
        elif self.compressor is not None:
            current_stream.wait_stream(stream_compressor)

        return q

    def _forward_prepare(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend,
        q_out: Optional[torch.Tensor] = None,
        x_quant=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x_linear = x_quant if x_quant is not None else x


        if _is_hip and self.compressor is not None:
            self.compressor.prelaunch_kv_score(x, forward_batch)
            if self.indexer is not None:
                self.indexer.compressor.prelaunch_kv_score(x, forward_batch)

        if self.fuse_wqa_wkv:
            qkv_a, _ = self.wqkv_a(x_linear)
            q_lora = qkv_a[..., : self.q_lora_rank]
        else:
            q_lora, _ = self.wq_a(x_linear)
            qkv_a = None

        use_cp = self.dsa_enable_prefill_cp and dsa_use_prefill_cp(forward_batch)
        kv: Optional[torch.Tensor]
        kv_handle = None

        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
            is_unified_kv_triton,
        )

        unified = is_unified_kv_triton()
        is_decode = forward_batch.forward_mode.is_decode_or_idle()


        fuse_verify = (
            envs.SGLANG_OPT_FUSED_QK_NORM_ROPE_VERIFY.get()
            and forward_batch.forward_mode.is_target_verify()
        )
        do_fused_qk_norm_rope = (unified and (is_decode or fuse_verify)) or (
            not unified and self.use_fused_qk_norm_rope
        )

        if do_fused_qk_norm_rope:
            if _is_gfx95_supported:
                q_for_wqb, q_lora = _fused_rmsnorm_fp8_quant(
                    q_lora,
                    self.q_norm.weight,
                    self.q_norm.variance_epsilon,
                )
                q, _ = self.wq_b(q_for_wqb)
            else:
                q_lora = self.q_norm(q_lora)
                q, _ = self.wq_b(q_lora)

            kv = (
                qkv_a[..., self.q_lora_rank :]
                if qkv_a is not None
                else self.wkv(x_linear)[0]
            )

            token_to_kv_pool = get_token_to_kv_pool()
            if unified and fuse_verify:


                kv = kv.contiguous()
                swa_cache, swa_loc = None, None
                swa_page_size, bf16_store = 1, True
            elif unified:
                swa_cache = token_to_kv_pool.get_unified_kv(self.layer_id)


                swa_loc = attn_backend.get_unified_swa_loc(forward_batch)
                swa_page_size, bf16_store = 1, True
            else:
                swa_cache = token_to_kv_pool.get_swa_raw_buffer(self.layer_id)
                swa_loc = attn_backend.get_swa_out_cache_loc(forward_batch)
                swa_page_size, bf16_store = (
                    token_to_kv_pool.swa_kv_pool.page_size,
                    False,
                )

            from sglang.kernels.ops.attention.fused_qk_norm_rope_store import (
                fused_qk_norm_rope_swa_store,
            )

            q = fused_qk_norm_rope_swa_store(
                q=q,
                kv=kv,
                q_norm_weight=None,
                kv_norm_weight=self.kv_norm.weight,
                q_rms_eps=self.eps,
                kv_rms_eps=self.eps,
                rope_head_dim=self.qk_rope_head_dim,
                cos_cache=self.cos_cache,
                sin_cache=self.sin_cache,
                positions=positions,
                swa_cache=swa_cache,
                swa_loc=swa_loc,
                swa_page_size=swa_page_size,
                q_out=q_out,
                dtype=x.dtype,
                bf16_store=bf16_store,
            )


            if not (unified and fuse_verify):
                kv = None

            if not unified and use_cp:


                kv = self._compute_kv_bf16(x, positions, qkv_a=qkv_a)
                kv = cp_materialize_global_token_order(
                    kv.contiguous(),
                    forward_batch,
                    torch.cuda.current_stream(),
                )
        elif _is_npu:
            q_lora = self.q_norm(q_lora)
            q, _ = self.wq_b(q_lora)
            q = q.view(-1, self.n_local_heads, self.head_dim)
            _dummy = q.new_ones(q.shape[-1])
            q = torch_npu.npu_rms_norm(q, _dummy, self.eps)[0]

            if qkv_a is not None:
                kv = qkv_a[..., self.q_lora_rank :]
            else:
                kv, _ = self.wkv(x)
            kv = self.kv_norm(kv)

            cos4, sin4 = self._get_npu_rope_position_cache(
                positions, q.dtype, inverse=False
            )
            Dsv4NpuRoPE.apply_rotary_mul_inplace(
                q,
                kv.unsqueeze(1),
                cos4,
                sin4,
                qk_nope_dim=self.qk_nope_head_dim,
            )
            attn_backend.store_cache(
                layer_id=self.layer_id,
                swa_k=kv,
                forward_batch=forward_batch,
            )
            kv = None
            if q_out is not None:
                q_out.copy_(q)
        else:
            q_lora = self.q_norm(q_lora)
            q = self._compute_q_b(q_lora, positions, q_out)
            if unified:


                kv = self._compute_kv_bf16(x_linear, positions, qkv_a=qkv_a)


                if use_cp and _is_hip:


                    comm_stream = getattr(
                        forward_batch, "_cp_prefetch_comm_stream", None
                    )
                    if comm_stream is not None:


                        kv_handle = cp_all_gather_rerange_launch(
                            kv, self.cp_size, comm_stream, ("kv", self.layer_id)
                        )
                        kv = None
                    else:
                        kv = cp_materialize_global_token_order(
                            kv.contiguous(),
                            forward_batch,
                            torch.cuda.current_stream(),
                        )
            elif use_cp:


                kv = self._compute_kv_bf16(x_linear, positions, qkv_a=qkv_a)
                kv = cp_materialize_global_token_order(
                    kv.contiguous(),
                    forward_batch,
                    torch.cuda.current_stream(),
                )
                attn_backend.store_cache(
                    layer_id=self.layer_id,
                    swa_k=kv,
                    forward_batch=forward_batch,
                )
            else:
                self._compute_kv_to_cache(
                    x_linear, positions, forward_batch, attn_backend, qkv_a=qkv_a
                )
                kv = None

        del qkv_a

        if self.indexer is not None:
            self.indexer(
                x=x,
                q_lora=q_lora,
                forward_batch=forward_batch,
                attn_backend=attn_backend,
            )
        if self.compressor is not None:
            attn_backend.forward_core_compressor(
                x,
                forward_batch,
                self.layer_id,
                self.compressor,
            )

        if _is_hip and kv_handle is not None:
            kv = cp_all_gather_rerange_finish(kv_handle)

        return q, kv

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        x_quant=None,
    ) -> torch.Tensor:
        if not get_attn_tp_context().input_scattered and x.shape[0] == 0:
            return x

        attn_backend = get_attn_backend()
        if TYPE_CHECKING:
            assert isinstance(
                attn_backend,
                (DeepseekV4AttnBackend, DeepseekV4HipRadixBackend),
            )

        enable_multi_stream = (
            envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get()
            and self.alt_streams is not None
            and get_is_capture_mode()
            and (
                is_in_breakable_cuda_graph()
                or x.shape[0] <= self._multi_stream_bs_limit
            )
            and not (self.dsa_enable_prefill_cp and dsa_use_prefill_cp(forward_batch))
            and not (_is_hip and self.compressor is None)
        ) or (
            _is_npu
            and envs.SGLANG_NPU_USE_MULTI_STREAM.get()
            and self.alt_streams is not None
            and x.shape[0] <= self._multi_stream_bs_limit
            and not forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()
        )

        tp_slice, q_padded, q_out = slice(None), None, None
        if self.attn_tp_size > 1:


            padded_num_heads = 64 if self.n_local_heads <= 64 else self.n_heads


            if _is_gfx942_supported:
                q_padded = x.new_zeros(x.shape[0], padded_num_heads, self.head_dim)
            else:
                q_padded = x.new_empty(x.shape[0], padded_num_heads, self.head_dim)
            tp_slice = slice(0, self.n_local_heads)
            q_out = q_padded[:, tp_slice, :]
        attn_sink = self._local_attn_sink()

        if enable_multi_stream:


            if _is_hip:
                q = self._forward_prepare_multi_stream_hip(
                    x,
                    positions,
                    forward_batch,
                    attn_backend,
                    q_out,
                    x_quant=x_quant,
                )
            elif _is_npu:
                q = self._forward_prepare_multi_stream_npu(
                    x,
                    positions,
                    forward_batch,
                    attn_backend,
                    q_out,
                    x_quant=x_quant,
                )
            else:
                q = self._forward_prepare_multi_stream(
                    x,
                    positions,
                    forward_batch,
                    attn_backend,
                    q_out,
                    x_quant=x_quant,
                )
            kv = None
        else:
            q, kv = self._forward_prepare(
                x,
                positions,
                forward_batch,
                attn_backend,
                q_out,
                x_quant=x_quant,
            )


        attn_k = kv if kv is not None else q
        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
            is_unified_kv_triton,
        )

        if is_unified_kv_triton():
            o = attn_backend.forward(
                q=q_out if q_out is not None else q,
                k=attn_k,
                v=attn_k,
                layer=self.attn_mqa,
                forward_batch=forward_batch,
                compress_ratio=self.compress_ratio,
                attn_sink=self.attn_sink,
                save_kv_cache=kv is not None,
            )
        else:
            attn_q = q_padded if q_padded is not None else q
            save_kv_cache = False
            if forward_batch.forward_mode.is_extend() and is_in_breakable_cuda_graph():
                o = attn_q.new_empty(
                    (*attn_q.shape[:-1], self.attn_mqa.v_head_dim),
                )
                bcg_deepseek_v4_attention_with_output(
                    attn_q,
                    attn_k,
                    o,
                    self.attn_mqa.layer_id,
                    self.compress_ratio,
                    attn_sink,
                    save_kv_cache,
                )
            else:
                o = attn_backend.forward(
                    q=attn_q,
                    k=attn_k,
                    v=attn_k,
                    layer=self.attn_mqa,
                    forward_batch=forward_batch,
                    compress_ratio=self.compress_ratio,
                    attn_sink=attn_sink,
                    save_kv_cache=save_kv_cache,
                )
            o = o[:, tp_slice, :]
        if _is_npu:
            cos4, sin4 = self._get_npu_rope_position_cache(
                positions, o.dtype, inverse=True
            )
            Dsv4NpuRoPE.apply_rotary_mul_inplace(
                o,
                None,
                cos4,
                sin4,
                qk_nope_dim=self.qk_nope_head_dim,
            )
        else:
            fused_rope_inplace(
                o[..., -self.qk_rope_head_dim :],
                None,
                self.freqs_cis,
                positions=positions,
                inverse=True,
            )

        o = o.view(o.shape[0], self.n_local_groups, -1)

        if _FP8_WO_A_GEMM:
            import deep_gemm

            from sglang.srt.layers import deep_gemm_wrapper

            T, G, D = o.shape
            R = self.o_lora_rank
            if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:

                o_fp8, o_s = sglang_per_token_group_quant_fp8_dsv4_wo_a(o)
                recipe = (1, 1, 128)
            else:

                o_fp8, o_s = sglang_per_token_group_quant_fp8(
                    o.reshape(T * G, D).contiguous(),
                    group_size=128,
                    scale_ue8m0=False,
                )
                o_fp8 = o_fp8.view(T, G, D)
                o_s = o_s.view(T, G, -1)
                recipe = (1, 128, 128)
            output = torch.empty(T, G, R, device=o.device, dtype=torch.bfloat16)
            deep_gemm.fp8_einsum(
                "bhr,hdr->bhd",
                (o_fp8, o_s),
                (self.wo_a.weight.view(G, R, D), self.wo_a.weight_scale_inv.data),
                output,
                recipe=recipe,
            )
            o = output
        else:
            wo_a_weight = getattr(self.wo_a, "weight", None)
            if wo_a_weight is not None:
                wo_a = wo_a_weight.view(self.n_local_groups, self.o_lora_rank, -1)
                o = _apply_wo_a_bf16_matmul(
                    o, wo_a, is_decode=forward_batch.forward_mode.is_decode()
                )
            else:
                o = _apply_gguf_grouped_wo_a(
                    o,
                    self.wo_a.qweight,
                    self.wo_a.qweight_type.weight_type,
                    self.o_lora_rank,
                )

        o, _ = self.wo_b(o.flatten(1))
        if self.attn_tp_size > 1 and self.attn_tp_size < get_parallel().tp_size:
            o = attn_tp_all_reduce(o)

        return o


    def op_attn(self, state):

        state.hidden_states_after_attn = self.forward(
            x=state.pop("hidden_states_after_input_norm"),
            positions=state.positions,
            forward_batch=state.forward_batch,
            x_quant=state.pop("attn_x_quant"),
        )


class DeepseekV4DecoderLayer(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        moe_quant_config_override: Optional[QuantizationConfig] = None,
        is_nextn: bool = False,
        prefix: str = "",
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        compress_ratio_override: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id
        self.self_attn = self._build_self_attn(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            alt_streams=alt_streams,
            compress_ratio_override=compress_ratio_override,
        )
        moe_alt_stream = (
            alt_streams[0]
            if (
                alt_streams is not None
                and (
                    _is_cuda
                    or envs.SGLANG_ROCM_USE_MULTI_STREAM.get()
                    or envs.SGLANG_NPU_USE_MULTI_STREAM.get()
                )
            )
            else None
        )
        self.mlp = deepseek_v2.DeepseekV2MoE(
            config=config,
            quant_config=moe_quant_config_override or quant_config,
            prefix=add_prefix("mlp", prefix),
            layer_id=self.layer_id,
            alt_stream=moe_alt_stream,
            is_nextn=is_nextn,
            is_deepseek_v4=True,
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.hc_mult = hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        (
            self.hc_attn_fn,
            self.hc_ffn_fn,
            self.hc_attn_base,
            self.hc_ffn_base,
            self.hc_attn_scale,
            self.hc_ffn_scale,
        ) = make_hc_mixing_params(hc_mult, config.hidden_size)
        self.rms_norm_eps = config.rms_norm_eps
        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
        self.use_fused_mhc_post_pre = is_cross_layer_mhc_fusion_enabled()
        self._input_layernorm_weight_bf16 = None
        self._post_attention_layernorm_weight_bf16 = None

    def _build_self_attn(
        self,
        *,
        config: DeepSeekV4Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
        alt_streams: Optional[List[torch.cuda.Stream]],
        compress_ratio_override: Optional[int],
    ) -> nn.Module:
        return MQALayer(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=prefix,
            alt_streams=alt_streams,
            compress_ratio_override=compress_ratio_override,
        )

    def refresh_mhc_norm_weight_cache(self):

        self._input_layernorm_weight_bf16 = (
            self.input_layernorm.weight.data.bfloat16().contiguous()
        )
        self._post_attention_layernorm_weight_bf16 = (
            self.post_attention_layernorm.weight.data.bfloat16().contiguous()
        )

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm: Optional[nn.Module] = None,
        forward_batch: Optional[ForwardBatch] = None,
    ):


        @compile_in_capture_mode
        def hc_pre_torch_impl(x, hc_fn):
            x_flat = x.flatten(1).float()
            rsqrt = torch.rsqrt(
                x_flat.square().mean(-1, keepdim=True) + self.rms_norm_eps
            )
            mixes = (F.linear(x_flat, hc_fn) * rsqrt).unsqueeze(1)
            return x_flat, mixes

        shape, dtype = x.size(), x.dtype

        if _is_npu:
            return _get_mhc_ops().npu_hc_pre(
                x,
                hc_fn,
                hc_scale,
                hc_base,
                hc_mult=self.hc_mult,
                hc_sinkhorn_iters=self.hc_sinkhorn_iters,
                rms_norm_eps=self.rms_norm_eps,
                hc_eps=self.hc_eps,
                forward_batch=forward_batch,
            )

        if x.shape[0] == 0:
            y = torch.empty((0, shape[-1]), dtype=dtype, device=x.device)
            post = torch.empty((0, self.hc_mult), dtype=torch.float32, device=x.device)
            comb = torch.empty(
                (0, self.hc_mult, self.hc_mult), dtype=torch.float32, device=x.device
            )
            return y, post, comb, False

        if envs.SGLANG_OPT_USE_FLASHINFER_MHC.get():
            y, post, comb = _flashinfer_hc_pre(
                x,
                hc_fn,
                hc_scale,
                hc_base,
                rms_eps=self.rms_norm_eps,
                hc_eps=self.hc_eps,
                sinkhorn_iters=self.hc_sinkhorn_iters,
            )
            return y, post, comb, False

        if envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.get():
            from sglang.kernels.ops.layernorm.mhc import mhc_pre

            norm_kwargs = {}
            if norm is not None:
                norm_kwargs["norm_weight"] = norm.weight.data
                norm_kwargs["norm_eps"] = norm.variance_epsilon

            post, comb, y = mhc_pre(
                residual=x,
                fn=hc_fn,
                hc_scale=hc_scale,
                hc_base=hc_base,
                rms_eps=self.rms_norm_eps,
                hc_pre_eps=self.hc_eps,
                hc_sinkhorn_eps=self.hc_eps,
                hc_post_mult_value=_MHC_POST_MULT_VALUE,
                sinkhorn_repeat=self.hc_sinkhorn_iters,
                **norm_kwargs,
            )
            return y, post.squeeze(-1), comb, norm is not None

        if _is_hip:
            from aiter.ops.mhc import mhc_pre

            post, comb, y = mhc_pre(
                residual=x,
                fn=hc_fn,
                hc_scale=hc_scale,
                hc_base=hc_base,
                rms_eps=self.rms_norm_eps,
                hc_pre_eps=self.hc_eps,
                hc_sinkhorn_eps=self.hc_eps,
                hc_post_mult_value=_MHC_POST_MULT_VALUE,
                sinkhorn_repeat=self.hc_sinkhorn_iters,
            )
            return y, post.squeeze(-1), comb, False

        if envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get():
            from sglang.srt.layers.deep_gemm_wrapper.entrypoint import (
                tf32_hc_prenorm_gemm,
            )

            x_flat = x.flatten(1).bfloat16()

            m, k = x_flat.shape
            mix_hc = hc_fn.size(0)
            d_out = torch.empty((m, mix_hc), dtype=torch.float, device=x.device)
            s_out = torch.empty((m,), dtype=torch.float, device=x.device)
            tf32_hc_prenorm_gemm(
                x_flat, hc_fn.float().contiguous(), d_out, s_out, num_splits=None
            )
            rsqrt = torch.rsqrt(s_out / k + self.rms_norm_eps)
            mixes = (d_out * rsqrt.unsqueeze(1)).unsqueeze(1)
        else:
            x_flat, mixes = hc_pre_torch_impl(x, hc_fn)

        pre, post, comb = _get_mhc_ops().hc_split_sinkhorn(
            mixes,
            hc_scale,
            hc_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )


        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            y = (pre.squeeze(1).unsqueeze(-1) * x_flat.view(shape)).sum(dim=1).to(dtype)
        return y, post.squeeze(1), comb.squeeze(1), False

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):

        if x.shape[0] == 0:
            return torch.empty(
                (0, self.hc_mult, x.shape[-1]), dtype=x.dtype, device=x.device
            )

        if _is_npu:
            return torch.ops.custom.npu_hc_post(x, residual, post, comb)

        if envs.SGLANG_OPT_USE_FLASHINFER_MHC.get():
            from flashinfer.mhc import mhc_post

            return mhc_post(x, residual, post, comb)

        if envs.SGLANG_OPT_USE_TILELANG_MHC_POST.get():
            from sglang.kernels.ops.layernorm.mhc import mhc_post

            return mhc_post(x, residual, post, comb)

        elif _is_hip:
            from aiter.ops.mhc import mhc_post

            result = torch.empty_like(residual)
            mhc_post(result, x, residual, post, comb)
            return result

        assert residual.shape == (x.shape[0], self.hc_mult, x.shape[-1])
        assert post.shape == (x.shape[0], self.hc_mult)
        assert comb.shape == (x.shape[0], self.hc_mult, self.hc_mult)

        @compile_in_capture_mode
        def hc_post_torch_impl(x, residual, post, comb):
            return (
                post.unsqueeze(-1) * x.unsqueeze(1)
                + (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(dim=1)
            ).type_as(x)

        return hc_post_torch_impl(x, residual, post, comb)

    def forward(
        self,
        positions: torch.tensor,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        input_ids_global: torch.Tensor,
        prev_residual: Optional[torch.Tensor] = None,
        prev_post: Optional[torch.Tensor] = None,
        prev_comb: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        use_fused = self.use_fused_mhc_post_pre

        if prev_residual is not None and use_fused:
            input_norm_weight = (
                self._input_layernorm_weight_bf16
                if self._input_layernorm_weight_bf16 is not None
                else self.input_layernorm.weight.data
            )
            fused = apply_mhc_post_pre_boundary(
                hidden_states,
                prev_residual,
                prev_post,
                prev_comb,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.hc_mult,
                self.rms_norm_eps,
                self.hc_eps,
                _MHC_POST_MULT_VALUE,
                self.hc_sinkhorn_iters,
                input_norm_weight,
                self.input_layernorm.variance_epsilon,
                fn_transpose=False,
            )
            if fused is not None:
                residual, hidden_states, post, comb, norm_fused = fused
                if not norm_fused:


                    if _use_aiter and _is_gfx95_supported:
                        x_quant, hidden_states = _fused_rmsnorm_fp8_quant(
                            hidden_states,
                            self.input_layernorm.weight,
                            self.rms_norm_eps,
                        )
                    else:
                        hidden_states = self.input_layernorm(hidden_states)
                        x_quant = None
                else:
                    x_quant = None
            else:


                hidden_states = self.hc_post(
                    hidden_states, prev_residual, prev_post, prev_comb
                )
                residual = hidden_states
                hidden_states, post, comb, norm_fused = self.hc_pre(
                    hidden_states,
                    self.hc_attn_fn,
                    self.hc_attn_scale,
                    self.hc_attn_base,
                    norm=self.input_layernorm,
                    forward_batch=forward_batch,
                )
                if not norm_fused:
                    if _use_aiter and _is_gfx95_supported:
                        x_quant, hidden_states = _fused_rmsnorm_fp8_quant(
                            hidden_states,
                            self.input_layernorm.weight,
                            self.rms_norm_eps,
                        )
                    else:
                        hidden_states = self.input_layernorm(hidden_states)
                        x_quant = None
                else:
                    x_quant = None
        else:
            residual = hidden_states
            hidden_states, post, comb, norm_fused = self.hc_pre(
                hidden_states,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                norm=self.input_layernorm,
                forward_batch=forward_batch,
            )
            if not norm_fused:
                if _use_aiter and _is_gfx95_supported:
                    x_quant, hidden_states = _fused_rmsnorm_fp8_quant(
                        hidden_states,
                        self.input_layernorm.weight,
                        self.rms_norm_eps,
                    )
                else:
                    hidden_states = self.input_layernorm(hidden_states)
                    x_quant = None
            else:
                x_quant = None

        with self.self_attn.maybe_use_decode_attn_tp(forward_batch):
            hidden_states = self.self_attn(
                x=hidden_states,
                positions=positions,
                forward_batch=forward_batch,
                x_quant=x_quant,
            )

        if use_fused:
            post_attn_norm_weight = (
                self._post_attention_layernorm_weight_bf16
                if self._post_attention_layernorm_weight_bf16 is not None
                else self.post_attention_layernorm.weight.data
            )
            fused = apply_mhc_post_pre_boundary(
                hidden_states,
                residual,
                post,
                comb,
                self.hc_ffn_fn,
                self.hc_ffn_scale,
                self.hc_ffn_base,
                self.hc_mult,
                self.rms_norm_eps,
                self.hc_eps,
                _MHC_POST_MULT_VALUE,
                self.hc_sinkhorn_iters,
                post_attn_norm_weight,
                self.post_attention_layernorm.variance_epsilon,
                fn_transpose=True,
            )
            if fused is not None:
                residual, hidden_states, post, comb, norm_fused = fused
                if not norm_fused:


                    hidden_states = self.post_attention_layernorm(hidden_states)
            else:
                hidden_states = self.hc_post(hidden_states, residual, post, comb)
                residual = hidden_states
                hidden_states, post, comb, norm_fused = self.hc_pre(
                    hidden_states,
                    self.hc_ffn_fn,
                    self.hc_ffn_scale,
                    self.hc_ffn_base,
                    norm=self.post_attention_layernorm,
                    forward_batch=forward_batch,
                )
                if not norm_fused:
                    hidden_states = self.post_attention_layernorm(hidden_states)
        else:
            hidden_states = self.hc_post(hidden_states, residual, post, comb)
            residual = hidden_states
            hidden_states, post, comb, norm_fused = self.hc_pre(
                hidden_states,
                self.hc_ffn_fn,
                self.hc_ffn_scale,
                self.hc_ffn_base,
                norm=self.post_attention_layernorm,
                forward_batch=forward_batch,
            )
            if not norm_fused:
                hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self._run_moe_ffn_dp_sync(
            hidden_states,
            forward_batch,
            input_ids=input_ids,
            input_ids_global=input_ids_global,
        )

        if not use_fused:
            hidden_states = self.hc_post(hidden_states, residual, post, comb)
            return hidden_states, None, None, None


        return hidden_states, residual, post, comb

    def _run_moe_ffn_dp_sync(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        *,
        input_ids: torch.Tensor,
        input_ids_global: torch.Tensor,
    ) -> torch.Tensor:
        _use_cp = self.dsa_enable_prefill_cp and dsa_use_prefill_cp(forward_batch)
        _use_tp_moe_gather = (
            not _use_cp
            and get_parallel().attn_dp_size > 1
            and get_moe_a2a_backend().is_none()
        )
        _use_tp_attn_a2a_scatter = (
            not _use_cp
            and get_parallel().attn_tp_size > 1
            and not get_moe_a2a_backend().is_none()
        )


        _use_reduce_scatterv = (
            _use_tp_moe_gather
            and is_dp_gatherv_active()
            and forward_batch.dp_padding_mode is not None
            and not forward_batch.dp_padding_mode.is_max_len()
        )


        _use_reduce_scatter = (
            envs.SGLANG_DP_USE_REDUCE_SCATTER.get()
            and _use_tp_moe_gather
            and not _use_reduce_scatterv
            and not should_use_dp_reduce_scatterv()
            and forward_batch.dp_padding_mode is not None
            and forward_batch.dp_padding_mode.is_max_len()
            and get_parallel().tp_size == get_parallel().attn_dp_size
        )
        mlp_reduce_scatter = _use_cp or _use_reduce_scatterv or _use_reduce_scatter


        _shared_local = None
        _do_shared_local = (
            _SHARED_EXPERT_LOCAL
            and _use_tp_moe_gather
            and getattr(self.mlp, "shared_experts", None) is not None
            and getattr(self.mlp, "_shared_expert_tp1", False)
        )
        if _use_cp:
            moe_a2a_backend = get_moe_a2a_backend()
            if moe_a2a_backend.is_none():
                hidden_states = dsa_cp_gather_hidden_states(hidden_states)
            else:
                assert moe_a2a_backend.is_deepep() or moe_a2a_backend.is_megamoe(), (
                    "CP requires DeepEP or megaMoE "
                    "(moe_a2a_backend == deepep or megamoe). "
                    f"Got {moe_a2a_backend.value}."
                )
        elif _use_tp_moe_gather:
            hidden_states, local_hidden_states = (
                get_global_dp_buffer(get_tp_group()),
                hidden_states,
            )
            if _do_shared_local and local_hidden_states.shape[0] > 0:
                _shared_local = self.mlp._forward_shared_experts(local_hidden_states)


            dp_gather_replicate(hidden_states, local_hidden_states, forward_batch)
        _a2a_scatter_chunks: Optional[List[torch.Tensor]] = None
        if _use_tp_attn_a2a_scatter:
            s, r = get_parallel().attn_tp_size, get_parallel().attn_tp_rank
            _a2a_scatter_chunks = list(hidden_states.tensor_split(s))
            hidden_states = _a2a_scatter_chunks[r].contiguous()
            input_ids = input_ids.tensor_split(s)[r].contiguous()
            input_ids_global = input_ids_global.tensor_split(s)[r].contiguous()


        with get_forward().scoped(mlp_reduce_scatter=mlp_reduce_scatter):
            hidden_states = self.mlp(
                hidden_states,
                forward_batch,
                input_ids=input_ids,
                input_ids_global=input_ids_global,
                skip_shared_experts=_do_shared_local,
            )
        if _use_cp and get_moe_a2a_backend().is_none():
            hidden_states = dsa_cp_reduce_scatter_hidden_states(hidden_states)
        elif _use_tp_moe_gather:
            hidden_states, global_hidden_states = (
                get_local_dp_buffer(get_tp_group()),
                hidden_states,
            )
            if should_use_dp_reduce_scatterv() or _use_reduce_scatterv:


                get_tp_group().reduce_scatterv(
                    global_hidden_states,
                    output=hidden_states,
                    sizes=get_dp_global_num_tokens(),
                )
            elif _use_reduce_scatter:


                dp_reduce_scatter_tensor(hidden_states, global_hidden_states)
            else:
                dp_scatter(hidden_states, global_hidden_states, forward_batch)


            if _shared_local is not None:
                n = hidden_states.shape[0]
                hidden_states = hidden_states + _shared_local[:n]
        if _use_tp_attn_a2a_scatter:
            assert _a2a_scatter_chunks is not None
            gathered = [torch.empty_like(t) for t in _a2a_scatter_chunks]
            attn_tp_all_gather(gathered, hidden_states.contiguous())
            hidden_states = torch.cat(gathered)
        return hidden_states


    def op_mhc_prepare_attn(
        self,
        state,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor] = None,
        tbo_subbatch_index: Optional[int] = None,
        **kwargs,
    ):

        attn_residual = hidden_states
        hidden_states, post, comb, norm_fused = self.hc_pre(
            hidden_states,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            norm=self.input_layernorm,
            forward_batch=forward_batch,
        )
        if not norm_fused:
            if _use_aiter and _is_gfx95_supported:
                x_quant, hidden_states = _fused_rmsnorm_fp8_quant(
                    hidden_states,
                    self.input_layernorm.weight,
                    self.rms_norm_eps,
                )
            else:
                hidden_states = self.input_layernorm(hidden_states)
                x_quant = None
        else:
            x_quant = None

        state.attn_residual = attn_residual
        state.attn_post = post
        state.attn_comb = comb
        state.hidden_states_after_input_norm = hidden_states
        state.attn_x_quant = x_quant

        if get_moe_a2a_backend().is_mori():
            state.num_tokens = attn_residual.shape[0]
        state.update(
            dict(
                forward_batch=forward_batch,
                positions=positions,
                tbo_subbatch_index=tbo_subbatch_index,
            )
        )

    def op_mhc_post_attn_pre_mlp(self, state):


        hidden_states_after_attn = state.pop("hidden_states_after_attn")
        attn_residual = state.pop("attn_residual")
        attn_post = state.pop("attn_post")
        attn_comb = state.pop("attn_comb")

        if self.use_fused_mhc_post_pre:
            post_attn_norm_weight = (
                self._post_attention_layernorm_weight_bf16
                if self._post_attention_layernorm_weight_bf16 is not None
                else self.post_attention_layernorm.weight.data
            )
            fused = apply_mhc_post_pre_boundary(
                hidden_states_after_attn,
                attn_residual,
                attn_post,
                attn_comb,
                self.hc_ffn_fn,
                self.hc_ffn_scale,
                self.hc_ffn_base,
                self.hc_mult,
                self.rms_norm_eps,
                self.hc_eps,
                _MHC_POST_MULT_VALUE,
                self.hc_sinkhorn_iters,
                post_attn_norm_weight,
                self.post_attention_layernorm.variance_epsilon,
                fn_transpose=True,
            )
            if fused is not None:
                ffn_residual, hidden_states, post, comb, norm_fused = fused
                if not norm_fused:


                    hidden_states = self.post_attention_layernorm(hidden_states)
                state.ffn_residual = ffn_residual
                state.ffn_post = post
                state.ffn_comb = comb
                state.hidden_states_mlp_input = hidden_states
                return

        hidden_states = self.hc_post(
            hidden_states_after_attn,
            attn_residual,
            attn_post,
            attn_comb,
        )
        ffn_residual = hidden_states
        hidden_states, post, comb, norm_fused = self.hc_pre(
            hidden_states,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            norm=self.post_attention_layernorm,
            forward_batch=state.forward_batch,
        )
        if not norm_fused:
            hidden_states = self.post_attention_layernorm(hidden_states)
        state.ffn_residual = ffn_residual
        state.ffn_post = post
        state.ffn_comb = comb
        state.hidden_states_mlp_input = hidden_states

    def op_mhc_postprocess(self, state):

        hidden_states = self.hc_post(
            state.pop("hidden_states_mlp_output"),
            state.pop("ffn_residual"),
            state.pop("ffn_post"),
            state.pop("ffn_comb"),
        )
        output = dict(
            positions=state.positions,
            hidden_states=hidden_states,


            residual=None,
            forward_batch=state.forward_batch,
            tbo_subbatch_index=state.tbo_subbatch_index,
        )
        state.clear(
            expect_keys={
                "positions",
                "forward_batch",
                "tbo_subbatch_index",
            }
        )
        return output


    def op_gather_a(self, state):


        fb = state.forward_batch
        local = state.pop("hidden_states_mlp_input")


        do_shared_local = (
            _SHARED_EXPERT_LOCAL
            and getattr(self.mlp, "shared_experts", None) is not None
            and getattr(self.mlp, "_shared_expert_tp1", False)
        )
        state.do_shared_local = do_shared_local
        state.shared_local = (
            self.mlp._forward_shared_experts(local)
            if (do_shared_local and local.shape[0] > 0)
            else None
        )


        sub = state.tbo_subbatch_index
        global_rows = get_global_dp_buffer_len()
        global_hidden = get_tbo_persistent_buffer(
            ("gh", sub), global_rows, local.shape[1], local.dtype, local.device
        )
        comm = get_dp_tbo_comm_stream()
        compute = torch.cuda.current_stream()
        with torch.cuda.stream(comm):
            comm.wait_stream(compute)
            dp_gather_partial(global_hidden, local, fb)
            state.gather_event = _tbo_event(("gather", sub))
            state.gather_event.record(comm)
        state.gather_keepalive = local
        state.global_hidden = global_hidden

    def op_gather_b(self, state):
        torch.cuda.current_stream().wait_event(state.pop("gather_event"))


        state.pop("gather_keepalive")

    def op_moe(self, state):


        fb = state.forward_batch
        global_hidden = state.pop("global_hidden")
        global_ids = fb._tbo_global_input_ids
        with get_forward().scoped(mlp_reduce_scatter=True):
            state.global_expert_out = self.mlp(
                global_hidden,
                fb,
                input_ids=global_ids,
                input_ids_global=global_ids,
                skip_shared_experts=state.do_shared_local,
            )

    def op_combine_a(self, state):


        global_out = state.pop("global_expert_out")
        local_out = get_tbo_persistent_buffer(
            ("lo", state.tbo_subbatch_index),
            get_local_dp_buffer_len(),
            global_out.shape[1],
            global_out.dtype,
            global_out.device,
        )
        state.combine_event = dp_reduce_scatterv_async(
            local_out,
            global_out,
            get_dp_global_num_tokens(),
            event_key=("combine", state.tbo_subbatch_index),
        )
        state.local_out = local_out


        state.combine_keepalive = global_out

    def op_combine_b(self, state):
        torch.cuda.current_stream().wait_event(state.pop("combine_event"))
        state.pop("combine_keepalive")
        hidden = state.pop("local_out")
        shared_local = state.pop("shared_local")
        state.pop("do_shared_local")
        if shared_local is not None:
            n = hidden.shape[0]
            hidden = hidden + shared_local[:n]
        state.hidden_states_mlp_output = hidden

    def _cp_tbo_launch(self, state, x, key, out_rows, collective):
        assert _is_hip, "CP+TBO MoE overlap is HIP-only"
        x = x.contiguous()
        sub = state.tbo_subbatch_index
        out = get_tbo_persistent_buffer(
            (key, sub), out_rows, x.shape[1], x.dtype, x.device
        )
        comm = get_dp_tbo_comm_stream()
        comm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(comm):
            collective(out, x)
            event = _tbo_event((key, sub))
            event.record(comm)
        return out, event, x

    def op_cp_gather_a(self, state):
        local = state.pop("hidden_states_mlp_input")
        out, event, keepalive = self._cp_tbo_launch(
            state,
            local,
            "cpgh",
            local.shape[0] * get_parallel().attn_cp_size,
            attn_cp_overlap_all_gather_into_tensor,
        )
        state.global_hidden = out
        state.cp_gather_event = event
        state.cp_gather_keepalive = keepalive

    def op_cp_gather_b(self, state):
        torch.cuda.current_stream().wait_event(state.pop("cp_gather_event"))
        state.pop("cp_gather_keepalive")

    def op_cp_moe(self, state):
        fb = state.forward_batch
        global_ids = fb._cp_moe_input_ids
        with get_forward().scoped(mlp_reduce_scatter=True):
            state.global_expert_out = self.mlp(
                state.pop("global_hidden"),
                fb,
                input_ids=global_ids,
                input_ids_global=global_ids,
            )

    def op_cp_combine_a(self, state):
        global_out = state.pop("global_expert_out")
        out, event, keepalive = self._cp_tbo_launch(
            state,
            global_out,
            "cplo",
            global_out.shape[0] // get_parallel().attn_cp_size,
            attn_cp_overlap_reduce_scatter_tensor,
        )
        state.local_out = out
        state.cp_combine_event = event
        state.cp_combine_keepalive = keepalive

    def op_cp_combine_b(self, state):
        torch.cuda.current_stream().wait_event(state.pop("cp_combine_event"))
        state.pop("cp_combine_keepalive")
        state.hidden_states_mlp_output = state.pop("local_out")


class DeepseekV4Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: DeepSeekV4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.hidden_size = config.hidden_size
        if self.pp_group.is_first_rank:
            embedding_quant_config = (
                quant_config
                if quant_config is not None and quant_config.get_name() == "expert_pack"
                else None
            )
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                enable_tp=not is_dp_attention_enabled(),
                quant_config=embedding_quant_config,
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.rms_norm_eps = config.rms_norm_eps
        use_stream_pool = (
            _is_cuda
            or (
                _is_hip
                and (
                    envs.SGLANG_ROCM_USE_MULTI_STREAM.get()
                    or envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get()
                )
            )
            or (_is_npu and envs.SGLANG_NPU_USE_MULTI_STREAM.get())
        )
        device_module = torch.get_device_module()
        num_alt_streams = 5 if (_is_cuda or _is_npu) else 2
        self.alt_streams = (
            [device_module.Stream() for _ in range(num_alt_streams)]
            if use_stream_pool
            else None
        )
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: DeepseekV4DecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_streams=self.alt_streams,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.gemm_output_zero_allocator_size = 0
        self.hc_eps = config.hc_eps
        self.hc_mult = hc_mult = config.hc_mult
        self.norm_eps = config.rms_norm_eps
        if self.pp_group.is_last_rank:
            (
                self.hc_head_fn,
                self.hc_head_base,
                self.hc_head_scale,
            ) = make_hc_head_params(hc_mult, config.hidden_size)

        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
        self.use_fused_mhc_post_pre = is_cross_layer_mhc_fusion_enabled()
        if self.dsa_enable_prefill_cp:
            self.cp_size = get_parallel().attn_cp_size

        self.dspark_layers_to_capture: Optional[List[int]] = None

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def hc_head(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ):
        if x.numel() > 0:
            from sglang.kernels.ops.layernorm.mhc_head import fused_hc_head

            return fused_hc_head(
                x.contiguous(),
                hc_fn,
                hc_scale,
                hc_base,
                norm_eps=self.norm_eps,
                hc_eps=self.hc_eps,
            )
        return hc_head_torch(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            norm_eps=self.norm_eps,
            hc_eps=self.hc_eps,
        )

    def _cp_children_splittable(self, forward_batch: ForwardBatch) -> bool:
        children = forward_batch.tbo_children
        if not children:
            return False
        cp_size = get_parallel().attn_cp_size
        for child in children:
            if child.batch_size <= 0 or child.extend_seq_lens_cpu is None:
                return False
            if sum(child.extend_seq_lens_cpu) < cp_size:
                return False
        return True

    def _can_run_tbo(self, forward_batch: ForwardBatch) -> bool:

        from sglang.srt.layers.moe import is_tbo_enabled

        if dsa_use_prefill_cp(forward_batch):
            path_ok = (
                _is_hip
                and not is_cp_v2_active(forward_batch)
                and is_dsa_prefill_cp_round_robin_split()
                and get_moe_a2a_backend().is_none()
                and self._cp_children_splittable(forward_batch)
            )
        else:
            path_ok = (
                not _is_hip
                or not get_moe_a2a_backend().is_none()
                or get_parallel().attn_dp_size > 1
            )
        return (
            is_tbo_enabled()
            and forward_batch.can_run_tbo
            and forward_batch.tbo_children is not None
            and forward_batch.global_forward_mode is not None


            and forward_batch.global_forward_mode.is_extend_without_speculative()
            and path_ok
            and self.pp_group.world_size == 1
        )

    def _forward_layers_tbo(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        from sglang.srt.batch_overlap.operations import execute_overlapped_operations
        from sglang.srt.batch_overlap.operations_strategy import OperationsStrategy
        from sglang.srt.batch_overlap.two_batch_overlap import (
            _model_forward_filter_inputs,
            _model_forward_tbo_merge_outputs,
        )

        if _is_hip and dsa_use_prefill_cp(forward_batch):
            return self._forward_layers_tbo_cp(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        layers = [self.layers[i] for i in range(self.start_layer, self.end_layer)]
        operations_strategy = OperationsStrategy.init_new_tbo(
            layers, forward_batch.global_forward_mode
        )


        inputs_arr = [
            _model_forward_filter_inputs(
                hidden_states=hidden_states,
                residual=None,
                positions=positions,
                output_forward_batch=child,
                tbo_subbatch_index=idx,
            )
            for idx, child in enumerate(forward_batch.tbo_children)
        ]


        if get_moe_a2a_backend().is_none() and get_parallel().attn_dp_size > 1:
            tp_group = get_tp_group()
            world = tp_group.world_size
            children = forward_batch.tbo_children
            local_lens = torch.tensor(
                [int(c.tbo_padded_len) for c in children],
                dtype=torch.int64,
                device=hidden_states.device,
            )
            gathered = torch.empty(
                (world, local_lens.shape[0]),
                dtype=torch.int64,
                device=hidden_states.device,
            )
            tp_group.all_gather_into_tensor(gathered, local_lens)
            gathered_cpu = gathered.tolist()
            rank = tp_group.rank_in_group
            for idx, child in enumerate(children):
                sizes = [gathered_cpu[r][idx] for r in range(world)]
                child.global_num_tokens_cpu = sizes
                child.global_num_tokens_gpu = gathered[:, idx].contiguous()
                child.global_dp_buffer_len = sum(sizes)


                local_ids = child.input_ids
                rows = sizes[rank]
                if local_ids.shape[0] < rows:
                    padded_ids = local_ids.new_zeros((rows,))
                    padded_ids[: local_ids.shape[0]] = local_ids
                elif local_ids.shape[0] > rows:
                    padded_ids = local_ids[:rows]
                else:
                    padded_ids = local_ids
                gids = torch.empty(
                    (sum(sizes),), dtype=local_ids.dtype, device=local_ids.device
                )
                tp_group.all_gatherv(padded_ids, sizes=sizes, output=gids)
                child._tbo_global_input_ids = gids

        outputs_arr = execute_overlapped_operations(
            inputs_arr=inputs_arr,
            operations_arr=[operations_strategy.operations] * 2,
            delta_stages=[0, operations_strategy.tbo_delta_stages],
        )

        hidden_states, _ = _model_forward_tbo_merge_outputs(
            outputs_arr[0], outputs_arr[1], hidden_states.shape[0]
        )
        return hidden_states

    def _setup_child_cp_metadata(self, child: ForwardBatch, child_backend) -> None:
        cp_rank = get_parallel().attn_cp_rank
        cp_size = get_parallel().attn_cp_size
        child.attn_cp_metadata = prepare_context_parallel_metadata(
            len(child.input_ids),
            cp_rank,
            cp_size,
            child.seq_lens_cpu.tolist(),
            extend_seqs_len=child.extend_seq_lens_cpu,
        )
        if is_dsa_prefill_cp_round_robin_split():
            metadata = child_backend.forward_metadata
            core_meta = metadata.core_attn_metadata
            core_meta.apply_cp_reindex()
            core_meta.init_flashmla_related(is_prefill=True)
            if metadata.indexer_metadata is not None:
                metadata.indexer_metadata = child_backend.init_forward_metadata_indexer(
                    core_meta
                )

    def _forward_layers_tbo_cp(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        assert _is_hip, "CP+TBO prefill path is HIP-only"

        from sglang.srt.batch_overlap.operations import execute_overlapped_operations
        from sglang.srt.batch_overlap.operations_strategy import OperationsStrategy
        from sglang.srt.batch_overlap.two_batch_overlap import (
            _model_forward_filter_inputs,
            _model_forward_tbo_merge_outputs,
        )

        original_len = hidden_states.shape[0]
        cp_size = get_parallel().attn_cp_size
        layers = [self.layers[i] for i in range(self.start_layer, self.end_layer)]
        operations_strategy = OperationsStrategy.init_new_tbo(
            layers, forward_batch.global_forward_mode, use_cp=True
        )

        attn_backend = get_attn_backend()
        children = forward_batch.tbo_children


        prefetch_comm_stream = get_dp_tbo_comm_stream()

        inputs_arr = []
        for idx, child in enumerate(children):
            child_inputs = _model_forward_filter_inputs(
                hidden_states=hidden_states,
                residual=None,
                positions=positions,
                output_forward_batch=child,
                tbo_subbatch_index=idx,
            )
            self._setup_child_cp_metadata(child, attn_backend.children[idx])
            if self.pp_group.is_first_rank:
                child_inputs["hidden_states"] = cp_split_and_rebuild_data(
                    child, child_inputs["hidden_states"]
                )
            child_inputs["positions"] = cp_split_and_rebuild_position(
                child, child_inputs["positions"]
            )
            child._cp_moe_input_ids = cp_round_robin_input_ids(child.input_ids)
            child._cp_prefetch_comm_stream = prefetch_comm_stream
            inputs_arr.append(child_inputs)

        outputs_arr = execute_overlapped_operations(
            inputs_arr=inputs_arr,
            operations_arr=[operations_strategy.operations] * 2,
            delta_stages=[0, operations_strategy.tbo_delta_stages],
        )

        if self.pp_group.is_last_rank:
            for idx, child in enumerate(children):
                outputs_arr[idx]["hidden_states"] = cp_all_gather_rerange_output(
                    outputs_arr[idx]["hidden_states"],
                    cp_size,
                    child,
                    torch.cuda.current_stream(),
                )
        hidden_states, _ = _model_forward_tbo_merge_outputs(
            outputs_arr[0], outputs_arr[1], original_len
        )
        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor],
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        cp_v2_active = is_cp_v2_active(forward_batch)
        use_prefill_cp = dsa_use_prefill_cp(forward_batch)
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]

            if hidden_states.ndim == 2:
                hidden_states = hidden_states.view(
                    hidden_states.shape[0], self.hc_mult, self.hidden_size
                )

        if get_parallel().attn_dp_size > 1 and get_moe_a2a_backend().is_none():
            input_ids_global = torch.empty(
                (get_global_dp_buffer_len(), 1),
                dtype=input_ids.dtype,
                device=input_ids.device,
            )


            dp_gather_replicate(
                input_ids_global, input_ids[:, None].clone(), forward_batch
            )
            input_ids_global = input_ids_global.squeeze(-1)
        else:
            input_ids_global = input_ids

        capture_dspark = self.dspark_layers_to_capture is not None
        dspark_aux_hidden_states: List[torch.Tensor] = []


        run_tbo = self._can_run_tbo(forward_batch) and not capture_dspark
        if use_prefill_cp and not run_tbo:
            if cp_v2_active:
                input_ids = cp_round_robin_input_ids_v2(input_ids, forward_batch)
            else:
                if self.pp_group.is_first_rank:
                    hidden_states = cp_split_and_rebuild_data(
                        forward_batch, hidden_states
                    )
                positions = cp_split_and_rebuild_position(forward_batch, positions)
                input_ids = cp_round_robin_input_ids(input_ids)
            input_ids_global = input_ids


        for _attr in ("freqs_cis_c4", "freqs_cis_c128"):
            if hasattr(forward_batch, _attr):
                delattr(forward_batch, _attr)
        if run_tbo:


            hidden_states = self._forward_layers_tbo(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        else:
            use_fused = self.use_fused_mhc_post_pre
            prev_residual, prev_post, prev_comb = None, None, None
            last_layer = None
            for i in range(self.start_layer, self.end_layer):
                layer = self.layers[i]
                last_layer = layer
                ctx = (
                    nullcontext()
                    if check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE)
                    else get_global_expert_distribution_recorder().with_current_layer(i)
                )
                with ctx:
                    hidden_states, prev_residual, prev_post, prev_comb = layer(
                        positions=positions,
                        hidden_states=hidden_states,
                        forward_batch=forward_batch,
                        input_ids=input_ids,
                        input_ids_global=input_ids_global,
                        prev_residual=prev_residual,
                        prev_post=prev_post,
                        prev_comb=prev_comb,
                    )
                if capture_dspark and i in self.dspark_layers_to_capture:
                    if use_fused:
                        completed = layer.hc_post(
                            hidden_states, prev_residual, prev_post, prev_comb
                        )
                    else:
                        completed = hidden_states
                    dspark_aux_hidden_states.append(completed.mean(dim=1))
            if use_fused and last_layer is not None:
                hidden_states = last_layer.hc_post(
                    hidden_states, prev_residual, prev_post, prev_comb
                )


        if (
            self.pp_group.is_last_rank
            and use_prefill_cp
            and not cp_v2_active
            and not run_tbo
        ):
            stream = torch.cuda.current_stream()
            hidden_states = cp_all_gather_rerange_output(
                hidden_states,
                self.cp_size,
                forward_batch,
                stream,
            )

            if capture_dspark:
                dspark_aux_hidden_states = [
                    cp_all_gather_rerange_output(
                        aux, self.cp_size, forward_batch, stream
                    )
                    for aux in dspark_aux_hidden_states
                ]

        if not self.pp_group.is_last_rank:

            return PPProxyTensors({"hidden_states": hidden_states.flatten(1)})

        pre_hc_head = hidden_states.flatten(1)

        hidden_states = self.hc_head(
            hidden_states, self.hc_head_fn, self.hc_head_scale, self.hc_head_base
        )
        hidden_states = self.norm(hidden_states)

        if capture_dspark:
            return (hidden_states, pre_hc_head), dspark_aux_hidden_states

        return hidden_states, pre_hc_head


class DeepseekV4ForCausalLM(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()


        if _is_hip:
            from sglang.kernels.ops.attention.deepseek_v4_rope import set_batched_rope
            from sglang.srt.layers.quantization.fp8_utils import set_force_ck_w8a8

            set_force_ck_w8a8(True)
            set_batched_rope(True)
        self.config = config
        self.tp_size = get_parallel().tp_size
        self.quant_config = quant_config
        self.determine_num_fused_shared_experts()
        self.model = DeepseekV4Model(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.pp_group = get_pp_group()
        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                    use_attn_tp_group=get_parallel().config.enable_dp_lm_head,
                )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = False
        get_attn_tp_context().init_context(config.q_lora_rank, is_dsa=True)

        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: self.model.layers[layer_id].mlp.get_moe_weights()
                for layer_id in range(self.model.start_layer, self.model.end_layer)
                if isinstance(
                    self.model.layers[layer_id].mlp, deepseek_v2.DeepseekV2MoE
                )
            }
        )


        self.start_layer = self.model.start_layer
        self.end_layer = self.model.end_layer

        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
        if self.dsa_enable_prefill_cp:
            self.cp_rank = get_parallel().attn_cp_rank
            self.cp_size = get_parallel().attn_cp_size


        self._mhc_prewarmed_at_load = False

    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def set_dspark_layers_to_capture(self, layer_ids: List[int]) -> None:
        if not self.pp_group.is_last_rank:
            return
        if layer_ids is None:
            raise ValueError(
                "DSPARK requires explicit layer_ids for aux hidden capture."
            )
        self.capture_aux_hidden_states = True
        self.model.dspark_layers_to_capture = list(layer_ids)

    @classmethod
    def shared_experts_fusion_disable_reason(cls, hf_config, quant_config):


        if quant_blocks_shared_experts_fusion(quant_config):
            return (
                "Quantization keeps shared experts at a higher precision than the "
                "routed experts, so they cannot be fused into the quantized "
                "routed-expert path."
            )
        if get_parallel().moe_ep_size > 1 and not uses_per_rank_fused_shared_slots():
            return (
                "Expert parallelism keeps only a slice of the routed experts on "
                "each rank, so the fused shared expert cannot be appended to the "
                "routed weight tensor (only DeepEP/MegaMOE per-rank shared slots "
                "support fusion under EP)."
            )
        if not get_exec().moe.enforce_shared_experts_fusion:
            return "Config does not support fused shared expert(s)."
        if hf_config.n_shared_experts != 1:
            raise ValueError(
                "DeepSeek V4 shared-experts fusion expects exactly one shared "
                f"expert, but got n_shared_experts={hf_config.n_shared_experts}."
            )
        return None

    def determine_num_fused_shared_experts(self):

        self.num_fused_shared_experts = (
            0 if is_shared_experts_fusion_disabled() else self.config.n_shared_experts
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if self.dsa_enable_prefill_cp:
            if can_dsa_cp_split(len(input_ids), self.cp_size, True, forward_batch):
                forward_batch.attn_cp_metadata = prepare_context_parallel_metadata(
                    len(input_ids),
                    self.cp_rank,
                    self.cp_size,
                    forward_batch.seq_lens_cpu.tolist(),
                    extend_seqs_len=forward_batch.extend_seq_lens_cpu,
                )
                if is_dsa_prefill_cp_round_robin_split():
                    attn_backend = get_attn_backend()
                    metadata = attn_backend.forward_metadata
                    core_meta = metadata.core_attn_metadata
                    core_meta.apply_cp_reindex()
                    core_meta.init_flashmla_related(is_prefill=True)
                    if metadata.indexer_metadata is not None:
                        metadata.indexer_metadata = (
                            attn_backend.init_forward_metadata_indexer(core_meta)
                        )

        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            hidden_states = self.model.forward(
                input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
            )
        if not self.pp_group.is_last_rank:
            return hidden_states

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states
        hidden_states, pre_hc_head = hidden_states

        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            aux_hidden_states,
            hidden_states_before_norm=(
                None if aux_hidden_states is not None else pre_hc_head
            ),
        )

    def _setup_fp8_wo_a_scales(self, is_nextn: bool) -> None:
        from sglang.srt.layers import deep_gemm_wrapper

        if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
            from deep_gemm import transform_sf_into_required_layout

        if is_nextn:
            layers = [self.model.decoder]
        else:
            layers = [
                self.model.layers[layer_id]
                for layer_id in range(self.model.start_layer, self.model.end_layer)
            ]
        for layer in layers:
            attn = layer.self_attn
            G = attn.n_local_groups
            R = attn.o_lora_rank
            D = attn.wo_a.weight.shape[1]

            raw_scale = attn.wo_a.weight_scale_inv.data.view(G, R // 128, D // 128)
            if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
                attn.wo_a.weight_scale_inv.data = transform_sf_into_required_layout(
                    raw_scale,
                    mn=R,
                    k=D,
                    recipe=(1, 128, 128),
                    num_groups=G,
                    is_sfa=False,
                )
                attn.wo_a.weight_scale_inv.format_ue8m0 = True
            else:
                attn.wo_a.weight_scale_inv.data = raw_scale.contiguous()
                attn.wo_a.weight_scale_inv.format_ue8m0 = False

    def post_load_weights(self, is_nextn=False, weight_names=None):
        if _FP8_WO_A_GEMM:
            self._setup_fp8_wo_a_scales(is_nextn)

        if is_nextn:
            return
        for layer_id in range(self.model.start_layer, self.model.end_layer):
            layer = self.model.layers[layer_id]
            self_attn = layer.self_attn
            if (
                self_attn.compress_ratio in (4, 128)
                and not self_attn.compressor.ape_converted
            ):
                self_attn.compressor.apply_ape_hotfix()
            if (
                self_attn.compress_ratio == 4
                and not self_attn.indexer.compressor.ape_converted
            ):
                self_attn.indexer.compressor.apply_ape_hotfix()
            layer.refresh_mhc_norm_weight_cache()

    @staticmethod
    def remap_weight_name_to_dpsk_hf_format(
        name: str,
        is_nextn: bool = False,
        num_hidden_layers: Optional[int] = None,
    ) -> str:
        if name.startswith("embed."):
            return "model.embed_tokens." + name.removeprefix("embed.")
        if name.startswith("head."):
            return "lm_head." + name.removeprefix("head.")
        if name == "norm.weight":
            return "model.norm.weight"
        if name.startswith("hc_head_"):
            return "model." + name

        if is_nextn and name.startswith("mtp."):
            parts = name.split(".", 2)
            if len(parts) >= 3:
                rest = parts[2]
                nextn_spec_prefixes = [
                    "e_proj",
                    "h_proj",
                    "emb",
                    "enorm",
                    "hnorm",
                    "norm",
                    "head",
                    "hc_head",
                ]
                is_nextn_spec = any(rest.startswith(p) for p in nextn_spec_prefixes)
                if is_nextn_spec:
                    if rest.startswith("emb.tok_emb"):
                        rest = rest.replace("emb.tok_emb", "embed_tokens")
                    elif rest == "norm.weight":
                        rest = "shared_head.norm.weight"
                    elif rest.startswith("head."):
                        rest = "shared_head.head.weight"
                    elif rest == "e_proj.scale":
                        rest = "e_proj.weight_scale_inv"
                    elif rest == "h_proj.scale":
                        rest = "h_proj.weight_scale_inv"
                name = f"model.layers.{num_hidden_layers}." + rest

        if name.startswith("layers."):
            name = "model." + name
        name = name.replace(".attn.", ".self_attn.")
        name = name.replace(".ffn.", ".mlp.")
        name = name.replace(".attn_norm.", ".input_layernorm.")
        name = name.replace(".ffn_norm.", ".post_attention_layernorm.")

        if "self_attn" in name and name.endswith(".scale"):
            name = name.removesuffix(".scale") + ".weight_scale_inv"

        name = name.replace(".gate.tid2eid", ".topk.tid2eid")
        name = name.replace(".gate.bias", ".gate.e_score_correction_bias")
        name = name.replace(".w1.", ".gate_proj.")
        name = name.replace(".w2.", ".down_proj.")
        name = name.replace(".w3.", ".up_proj.")
        if "mlp" in name and name.endswith(".scale"):
            name = name.removesuffix(".scale") + ".weight_scale_inv"

        return name

    def _prewarm_mhc_kernels(self) -> None:

        if self._mhc_prewarmed_at_load:
            return
        self._mhc_prewarmed_at_load = True
        if _is_npu or not envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.get():
            return
        layer = next(
            (m for m in self.model.layers if isinstance(m, DeepseekV4DecoderLayer)),
            None,
        )
        if layer is None:
            return

        from sglang.kernels.ops.layernorm.mhc import mhc_post, prewarm_mhc_pre

        tic = time.perf_counter()
        residual = torch.zeros(
            (1, layer.hc_mult, layer.hidden_size),
            dtype=torch.bfloat16,
            device=layer.hc_attn_fn.device,
        )
        prewarm_mhc_pre(

            residual=residual,
            fn=layer.hc_attn_fn,
            hc_scale=layer.hc_attn_scale,
            hc_base=layer.hc_attn_base,
            rms_eps=layer.rms_norm_eps,
            hc_pre_eps=layer.hc_eps,
            hc_sinkhorn_eps=layer.hc_eps,
            hc_post_mult_value=_MHC_POST_MULT_VALUE,
            sinkhorn_repeat=layer.hc_sinkhorn_iters,
            n_splits=1,
            n_splits_pre=32,
            norm_weight=layer.input_layernorm.weight.data,
            norm_eps=layer.input_layernorm.variance_epsilon,
        )
        mhc_post(
            x=residual.new_zeros((1, layer.hidden_size)),
            residual=residual,
            post_layer_mix=torch.zeros(
                (1, layer.hc_mult, 1),
                dtype=torch.float32,
                device=residual.device,
            ),
            comb_res_mix=torch.zeros(
                (1, layer.hc_mult, layer.hc_mult),
                dtype=torch.float32,
                device=residual.device,
            ),
        )
        torch.cuda.synchronize()
        compile_secs = time.perf_counter() - tic

        torch.cuda.empty_cache()
        get_tp_group().barrier()
        logger.info(
            "DeepSeek V4 MHC prewarm at load: compile %.1fs, rank sync +%.1fs",
            compile_secs,
            time.perf_counter() - tic - compile_secs,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        if is_nextn:
            if hasattr(self.config, "num_nextn_predict_layers"):
                num_nextn_layers = self.config.num_nextn_predict_layers
                assert num_nextn_layers == 1, "Only 1 nextn layer is supported"
                nextn_layer_id = (
                    0
                    if self.config.num_hidden_layers == 1
                    else self.config.num_hidden_layers
                )
            else:
                raise ValueError("num_nextn_predict_layers is not in the config")

        if not _FP8_WO_A_GEMM:
            weights = _prepare_deepseek_v4_weights(weights, self.quant_config)

        stacked_params_mapping = DEEPSEEK_V4_STACKED_PARAMS_MAPPING

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts + self.num_fused_shared_experts,
        )

        if is_wint4afp8_or_wint4a16_config(self.quant_config):
            expert_params_mapping += FusedMoE.make_expert_input_scale_params_mapping(
                num_experts=self.config.n_routed_experts
            )

        cache_compressor_weight = {}
        COMPRESSOR_PART = ".compressor.w"

        fuse_wqa_wkv = envs.SGLANG_OPT_FUSE_WQA_WKV.get()
        cache_wqkv_a_weight: dict[str, dict[str, torch.Tensor]] = {}

        def auto_weight_loader(module):
            return getattr(module, "weight_loader", default_weight_loader)

        if is_nextn:
            nextn_layer_prefix = f"model.layers.{nextn_layer_id}"
            nextn_spec_weight_names_out_of_layer = [
                "shared_head.norm",
                "shared_head.head",
                "embed_tokens",
                ".e_proj",
                "h_proj",
                "enorm",
                "hnorm",
                "hc_head_base",
                "hc_head_fn",
                "hc_head_scale",
            ]

        if self.num_fused_shared_experts > 0:
            assert self.num_fused_shared_experts == 1
            log_info_on_rank0(logger, "Shared experts fusion optimization enabled.")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            weight_names = []
            for name, loaded_weight in weights:
                if (
                    _FP8_WO_A_GEMM
                    and name.endswith(".wo_a.weight")
                    and loaded_weight.dtype != torch.float8_e4m3fn
                ):
                    raise ValueError(
                        f"SGLANG_OPT_FP8_WO_A_GEMM is enabled but {name} has "
                        f"dtype {loaded_weight.dtype}, expected "
                        "torch.float8_e4m3fn. This checkpoint does not provide "
                        "a supported fp8-quantized wo_a; rerun with "
                        "SGLANG_OPT_FP8_WO_A_GEMM=0."
                    )
                try:
                    use_async_loading = should_async_load(loaded_weight)

                    name = self.remap_weight_name_to_dpsk_hf_format(
                        name,
                        is_nextn=is_nextn,
                        num_hidden_layers=self.config.num_hidden_layers,
                    )

                    layer_id = get_layer_id(name)
                    if (
                        layer_id is not None
                        and hasattr(self.model, "start_layer")
                        and (
                            layer_id < self.model.start_layer
                            or layer_id >= self.model.end_layer
                        )
                    ):
                        continue
                    if (
                        self.num_fused_shared_experts > 0
                        and "mlp.shared_experts" in name
                    ):
                        name = name.replace(
                            "mlp.shared_experts",
                            f"mlp.experts.{self.config.n_routed_experts}",
                        )

                    weight_names.append(name)

                    if not is_nextn:
                        if hasattr(self.config, "num_nextn_predict_layers"):
                            num_nextn_layers = self.config.num_nextn_predict_layers
                            if num_nextn_layers > 0 and name.startswith("model.layers"):
                                name_list = name.split(".")
                                if (
                                    len(name_list) >= 3
                                    and int(name_list[2])
                                    >= self.config.num_hidden_layers
                                ):
                                    continue

                            if name.startswith("mtp"):
                                continue
                    else:
                        if "shared_head.head" in name or "embed_tokens" in name:
                            continue

                        if not name.startswith(nextn_layer_prefix):
                            continue

                        in_decoder = True
                        for weight_name in nextn_spec_weight_names_out_of_layer:
                            if weight_name in name:
                                in_decoder = False
                                name = name.replace(nextn_layer_prefix, "model")
                                break

                        if in_decoder:
                            name = name.replace(nextn_layer_prefix, "model.decoder")

                    if "rotary_emb.inv_freq" in name:
                        continue
                    for param_name, weight_name, shard_id in stacked_params_mapping:
                        if weight_name not in name:
                            continue
                        if _is_npu:
                            name = name.replace("weight_packed", "weight")
                        if ("mlp.experts." in name) and name not in params_dict:
                            continue
                        name = name.replace(weight_name, param_name)
                        if name.endswith(".bias") and name not in params_dict:
                            continue
                        if name not in params_dict and name.startswith("mtp"):
                            break
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        maybe_executor_submit(
                            executor=executor,
                            futures=futures,
                            use_async=use_async_loading,
                            func=weight_loader,
                            func_args=(param, loaded_weight, shard_id),
                        )
                        loaded_params.add(name)
                        break
                    else:
                        skip_unmaterialized_expert_param = False
                        for mapping in expert_params_mapping:
                            param_name, weight_name, expert_id, shard_id = mapping
                            if weight_name not in name:
                                continue
                            if _is_npu:
                                name = name.replace("weight_packed", "weight")
                            resolved_name = name.replace(weight_name, param_name)
                            if resolved_name not in params_dict:
                                skip_unmaterialized_expert_param = True
                                continue
                            param = params_dict[resolved_name]
                            weight_loader = param.weight_loader
                            maybe_executor_submit(
                                executor=executor,
                                futures=futures,
                                use_async=use_async_loading,
                                func=weight_loader,
                                func_args=(
                                    param,
                                    loaded_weight,
                                    resolved_name,
                                ),
                                func_kwargs={
                                    "shard_id": shard_id,
                                    "expert_id": expert_id,
                                },
                            )
                            loaded_params.add(resolved_name)
                            break
                        else:
                            if skip_unmaterialized_expert_param:
                                continue
                            if name.endswith(".bias") and name not in params_dict:
                                continue
                            if (
                                ".embed_tokens." in name
                                and not self.pp_group.is_first_rank
                            ):
                                continue
                            if (
                                name == "model.norm.weight"
                                and not self.pp_group.is_last_rank
                            ):
                                continue
                            if (
                                name.startswith("model.hc_head_")
                                or name == "lm_head.weight"
                            ) and not self.pp_group.is_last_rank:
                                continue
                            elif COMPRESSOR_PART in name and ".wkv_gate." not in name:
                                is_kv = name.endswith(".wkv.weight")
                                is_wgate = name.endswith(".wgate.weight")
                                assert is_kv != is_wgate
                                key = name.rsplit(".", 2)[0]
                                assert key.endswith(".compressor")
                                if key not in cache_compressor_weight:
                                    cache_compressor_weight[key] = (
                                        is_kv,
                                        _clone_if_runai_streamed_tensor(loaded_weight),
                                    )
                                else:
                                    assert key in cache_compressor_weight
                                    cached_is_kv, cached_weight = (
                                        cache_compressor_weight[key]
                                    )
                                    assert cached_is_kv != is_kv
                                    kv = loaded_weight if is_kv else cached_weight
                                    wgate = loaded_weight if is_wgate else cached_weight
                                    fused_weight = torch.cat([kv, wgate], dim=0)
                                    param_name = key + ".wkv_gate.weight"
                                    param = params_dict[param_name]
                                    weight_loader = auto_weight_loader(param)
                                    maybe_executor_submit(
                                        executor=executor,
                                        futures=futures,
                                        use_async=use_async_loading,
                                        func=weight_loader,
                                        func_args=(param, fused_weight),
                                    )
                                    loaded_params.add(param_name)
                                    cache_compressor_weight.pop(key)
                            elif fuse_wqa_wkv and (
                                name.endswith(".wq_a.weight")
                                or name.endswith(".wq_a.weight_scale_inv")
                                or name.endswith(".wkv.weight")
                                or name.endswith(".wkv.weight_scale_inv")
                                or name.endswith(".wq_a.qweight")
                                or name.endswith(".wkv.qweight")
                                or name.endswith(".wq_a.qweight_type")
                                or name.endswith(".wkv.qweight_type")
                            ):
                                is_q = ".wq_a." in name
                                param_name = name.replace(
                                    ".wq_a." if is_q else ".wkv.", ".wqkv_a."
                                )
                                bucket = cache_wqkv_a_weight.setdefault(param_name, {})
                                shard_key = "q" if is_q else "kv"
                                assert (
                                    shard_key not in bucket
                                ), f"duplicate shard {shard_key} for {param_name}"
                                bucket[shard_key] = _clone_if_runai_streamed_tensor(
                                    loaded_weight
                                )
                                if len(bucket) == 2:
                                    fused_weight = _fuse_deepseek_v4_wqkv_a_pair(
                                        param_name, bucket
                                    )
                                    param = params_dict[param_name]
                                    weight_loader = auto_weight_loader(param)
                                    maybe_executor_submit(
                                        executor=executor,
                                        futures=futures,
                                        use_async=use_async_loading,
                                        func=weight_loader,
                                        func_args=(param, fused_weight),
                                    )
                                    loaded_params.add(param_name)
                                    cache_wqkv_a_weight.pop(param_name)
                            else:
                                if (
                                    "k_scale" in name or "v_scale" in name
                                ) and name not in params_dict:
                                    for scale in ["k_scale", "v_scale"]:
                                        if scale in name:
                                            name = name.replace(
                                                f"{scale[0]}_proj", "attn_mqa"
                                            )
                                            break
                                if name not in params_dict:
                                    if not name.startswith("mtp"):
                                        logger.warning(
                                            f"{name} not found in params_dict."
                                        )
                                    continue
                                param = params_dict[name]

                                weight_loader = auto_weight_loader(param)
                                maybe_executor_submit(
                                    executor=executor,
                                    futures=futures,
                                    use_async=use_async_loading,
                                    func=weight_loader,
                                    func_args=(param, loaded_weight),
                                )
                                loaded_params.add(name)
                except Exception as e:
                    e.add_note(f"{name=} {loaded_weight.shape=}")
                    raise

            for future in concurrent.futures.as_completed(futures):
                future.result()

        assert len(cache_compressor_weight) == 0
        assert len(cache_wqkv_a_weight) == 0, cache_wqkv_a_weight.keys()
        unloaded_params = params_dict.keys() - loaded_params

        skipped_checking_patterns = [
            "attn_mqa.k_scale",
            "attn_mqa.v_scale",
            "blockscale_swizzled",
        ]
        if not self.pp_group.is_first_rank:
            skipped_checking_patterns.append("embed_tokens")
        if not self.pp_group.is_last_rank:
            skipped_checking_patterns.append("model.norm.")
            skipped_checking_patterns.extend(["lm_head", "hc_head_"])
        if is_nextn:
            skipped_checking_patterns.extend(["lm_head", "embed_tokens"])
        unloaded_params = {
            p
            for p in unloaded_params
            if all(
                skipped_checking_pattern not in p
                for skipped_checking_pattern in skipped_checking_patterns
            )
        }
        if unloaded_params:
            logger.warning(
                f"Some weights are not initialized from checkpoints: {unloaded_params}"
            )

        self.post_load_weights(is_nextn=is_nextn, weight_names=weight_names)

        if not is_nextn:
            self._prewarm_mhc_kernels()

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head


        torch.get_device_module().empty_cache()
        torch.get_device_module().synchronize()

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=None,
        )


EntryClass = [DeepseekV4ForCausalLM]


def _dequant_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    from einops import rearrange

    assert (
        weight.dtype == torch.float8_e4m3fn
    ), f"expected fp8_e4m3fn, got {weight.dtype}"
    assert scale.dtype in (
        torch.float8_e8m0fnu,
        torch.float32,
    ), f"expected fp8_e8m0fnu or float32, got {scale.dtype}"

    weight_f32 = rearrange(
        weight.float(), "(sn bn) (sk bk) -> sn bn sk bk", bn=128, bk=128
    )
    result = rearrange(
        weight_f32 * scale.float()[:, None, :, None], "sn bn sk bk -> (sn bn) (sk bk)"
    )

    return result.to(torch.bfloat16)


def _clone_if_runai_streamed_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if getattr(tensor, RUNAI_STREAMER_TENSOR_ATTR, False):
        return tensor.clone().detach()
    return tensor


def _dequant_fp8_wo_a_streaming(
    weights: Iterable[Tuple[str, torch.Tensor]],
) -> Iterable[Tuple[str, torch.Tensor]]:
    pending: dict[str, dict[str, torch.Tensor]] = {}
    saw_wo_a_scale = False
    emitted = False

    for name, tensor in weights:
        if name.endswith(".wo_a.weight"):
            prefix = name[: -len(".weight")]
            bucket = pending.setdefault(prefix, {})
            scale = bucket.pop("scale", None)
            if scale is not None:
                pending.pop(prefix, None)
                emitted = True
                yield name, _dequant_fp8(tensor, scale)
            else:
                bucket["weight"] = _clone_if_runai_streamed_tensor(tensor)
            continue

        if name.endswith(".wo_a.scale"):
            saw_wo_a_scale = True
            prefix = name[: -len(".scale")]
            bucket = pending.setdefault(prefix, {})
            weight = bucket.pop("weight", None)
            if weight is not None:
                pending.pop(prefix, None)
                emitted = True
                yield prefix + ".weight", _dequant_fp8(weight, tensor)
            else:
                bucket["scale"] = _clone_if_runai_streamed_tensor(tensor)
            continue

        yield name, tensor

    if emitted:
        logger.info("Finished streaming dequant fp8 wo_a")
    for prefix, bucket in pending.items():
        if "weight" in bucket:
            assert not saw_wo_a_scale, f"{prefix}.scale is missing"
            yield prefix + ".weight", bucket["weight"]
        if "scale" in bucket:
            yield prefix + ".scale", bucket["scale"]


def _dequant_fp8_wo_a(
    weights: Iterable[Tuple[str, torch.Tensor]],
) -> Iterable[Tuple[str, torch.Tensor]]:
    weights_dict = dict(weights)

    for name in list(weights_dict.keys()):
        if name not in weights_dict:
            continue
        if not name.endswith(".wo_a.weight"):
            continue
        scale_name = name.replace(".wo_a.weight", ".wo_a.scale")
        assert scale_name in weights_dict
        weight = weights_dict.pop(name)
        scale = weights_dict.pop(scale_name)
        yield name, _dequant_fp8(weight, scale)

    yield from weights_dict.items()


def _prepare_deepseek_v4_weights(
    weights: Iterable[Tuple[str, torch.Tensor]],
    quant_config: Optional[QuantizationConfig],
) -> Iterable[Tuple[str, torch.Tensor]]:


    if quant_config is not None and quant_config.get_name() == "expert_pack":
        logger.info("Keep Expert Pack GGUF weights on the streaming load path")
        return weights
    return _dequant_fp8_wo_a_streaming(weights)


def _fuse_deepseek_v4_wqkv_a_pair(
    param_name: str, bucket: dict[str, torch.Tensor]
) -> torch.Tensor:


    q = bucket["q"]
    kv = bucket["kv"]
    if param_name.endswith(".qweight_type"):
        if q.numel() != 1 or kv.numel() != 1 or q.item() != kv.item():
            raise ValueError(
                f"cannot fuse different GGUF qweight types for {param_name}: "
                f"q={q.tolist()} kv={kv.tolist()}"
            )
        return q
    return torch.cat([q, kv], dim=0)
