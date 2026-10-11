from enum import IntEnum

import torch
import torch.nn as nn
import triton
import triton.language as tl
from einops import rearrange
from torch.nn.parameter import Parameter

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.models.inkling_common.kernels.sconv import (
    SconvDecodeMetadata,
    SconvExtendMetadata,
    causal_conv1d,
    fused_causal_conv1d_update_decode,
    save_intermediate_conv_windows,
    update_sconv_cache,
)
from sglang.srt.runtime_context import (
    get_exec,
    get_parallel,
    mamba_extra_buffer_enabled,
)
from sglang.srt.utils import is_cuda, set_weight_attrs


class SconvType(IntEnum):
    K_FULL = 0
    V_FULL = 1
    K_LOCAL = 2
    V_LOCAL = 3
    ATTN = 4
    MLP = 5


class ShortConvolution(nn.Module):


    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        sconv_type: SconvType,
        activation: str | None = None,
        use_residual: bool = True,
        layer_id: int | None = None,
        tp_rank: int | None = None,
    ):
        super().__init__()

        self.kernel_size = (
            (kernel_size,) if isinstance(kernel_size, int) else kernel_size
        )
        self.use_residual = use_residual
        self.layer_id = layer_id
        self.sconv_type: SconvType | None = sconv_type

        if tp_rank is None:
            tp_rank = get_parallel().attn_tp_rank
        self.tp_rank = tp_rank


        self.weight = nn.Parameter(
            torch.empty(
                hidden_size,
                1,
                kernel_size,
            ),
            requires_grad=False,
        )

        set_weight_attrs(self.weight, {"weight_loader": self.weight_loader})

        self.activation = None
        if activation is not None:
            assert activation in [
                "silu",
                "swish",
            ], f"Activation `{activation}` not supported yet."
            self.activation = activation

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):

        param_data = param.data

        if loaded_weight.shape[0] != param_data.shape[0]:

            shard_size = param_data.shape[0]
            start_idx = self.tp_rank * shard_size


            loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)

        assert param_data.shape == loaded_weight.shape, (
            f"Shape mismatch after narrowing: param {param_data.shape} vs "
            f"loaded {loaded_weight.shape}"
        )
        param_data.copy_(loaded_weight)

    def _conv_state(self, forward_batch: ForwardBatch):

        return get_attn_backend().conv_state_metadata(self.layer_id, forward_batch)

    def _sconv_cache(self) -> torch.Tensor:

        return get_attn_backend().sconv_state(
            layer_id=self.layer_id, stream=self.sconv_type.value
        )

    def _intermediate_window(self) -> torch.Tensor:

        return get_attn_backend().sconv_intermediate_window(
            layer_id=self.layer_id, stream=self.sconv_type.value
        )

    def _weight_2d(self) -> torch.Tensor:
        return rearrange(self.weight, "d 1 w -> d w")

    def _apply_training_sconv_kernel(
        self,
        hidden_states: torch.Tensor,
        sconv_cache: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        has_initial_state: torch.Tensor,
        precomputed: SconvDecodeMetadata | SconvExtendMetadata,
    ) -> torch.Tensor:
        y = self._apply_causal_sconv_kernel(
            hidden_states=hidden_states,
            sconv_cache=sconv_cache,
            precomputed=precomputed,
        )
        update_sconv_cache(
            x=hidden_states,
            sconv_cache=sconv_cache,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            query_start_loc=query_start_loc,
        )
        return y

    def _apply_causal_sconv_kernel(
        self,
        hidden_states: torch.Tensor,
        sconv_cache: torch.Tensor,
        precomputed: SconvDecodeMetadata | SconvExtendMetadata,
    ) -> torch.Tensor:
        return causal_conv1d(
            x=hidden_states,
            weight=self._weight_2d(),
            sconv_cache=sconv_cache,
            activation=self.activation,
            use_residual=self.use_residual,
            **precomputed,
        )

    def _apply_decode_sconv_kernel(
        self,
        hidden_states: torch.Tensor,
        sconv_cache: torch.Tensor,
        cache_indices: torch.Tensor,
        precomputed: SconvDecodeMetadata | SconvExtendMetadata,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        return fused_causal_conv1d_update_decode(
            x=hidden_states,
            weight=self._weight_2d(),
            sconv_cache=sconv_cache,
            cache_indices=cache_indices,
            cache_mask=precomputed["cache_mask"],
            activation=self.activation,
            use_residual=self.use_residual,
            track_mask=forward_batch.mamba_track_mask,
            track_indices=forward_batch.mamba_track_indices,
        )

    def _prepare_extend_sconv_cache(
        self,
        forward_batch: ForwardBatch,
        sconv_cache: torch.Tensor,
        hidden_states: torch.Tensor,
        track_conv_indices: torch.Tensor | None,
    ):
        if track_conv_indices is not None:


            fused_gather_scatter_to_sconv_cache(
                hidden_states=hidden_states,
                sconv_cache=sconv_cache,
                track_conv_indices=track_conv_indices,
                mask=forward_batch.mamba_track_mask,
                dst_indices=forward_batch.mamba_track_indices,
            )

    def _save_intermediate_conv_windows(
        self,
        forward_batch: ForwardBatch,
        sconv_cache: torch.Tensor,
        cache_indices: torch.Tensor,
        hidden_states: torch.Tensor,
    ):

        save_intermediate_conv_windows(
            sconv_cache=sconv_cache,
            hidden_states=hidden_states,
            cache_indices=cache_indices,
            intermediate_out=self._intermediate_window(),
            batch_size=forward_batch.batch_size,
            draft_token_num=forward_batch.spec_info.draft_token_num,
        )

    def _update_sconv_cache_for_draft_extend(
        self,
        forward_batch: ForwardBatch,
        sconv_cache: torch.Tensor,
        cache_indices: torch.Tensor,
        hidden_states: torch.Tensor,
    ):

        num_accept_tokens = forward_batch.spec_info.num_accept_tokens
        batch_size = forward_batch.batch_size
        if len(hidden_states.shape) == 2:
            draft_token_num = hidden_states.shape[0] // batch_size
        else:
            draft_token_num = hidden_states.shape[1]

        mamba_track_indices = getattr(forward_batch, "mamba_track_indices", None)
        do_tracking = mamba_track_indices is not None and mamba_extra_buffer_enabled()

        crossed = track_step = None
        if do_tracking:
            mamba_track_interval = get_exec().mamba.mamba_track_interval
            pre_seqlen = forward_batch.seq_lens[:batch_size] - draft_token_num
            post_seqlen = pre_seqlen + num_accept_tokens
            crossed = (pre_seqlen // mamba_track_interval) != (
                post_seqlen // mamba_track_interval
            )
            tracking_boundary = (
                post_seqlen // mamba_track_interval
            ) * mamba_track_interval
            track_step = (tracking_boundary - pre_seqlen - 1).clamp(0, draft_token_num)

        fused_draft_extend_sconv_cache(
            hidden_states=hidden_states,
            sconv_cache=sconv_cache,
            cache_indices=cache_indices[:batch_size],
            num_accept_tokens=num_accept_tokens,
            draft_token_num=draft_token_num,
            do_tracking=do_tracking,
            crossed=crossed,
            track_step=track_step,
            mamba_track_indices=(
                mamba_track_indices[:batch_size] if do_tracking else None
            ),
        )

    def decode_fused_ar_inputs(self, forward_batch: ForwardBatch):

        meta = self._conv_state(forward_batch)
        return (
            self._sconv_cache(),
            meta.cache_indices,
            meta.precomputed["cache_mask"],
            self._weight_2d(),
        )

    def verify_fused_ar_inputs(self, forward_batch: ForwardBatch):

        meta = self._conv_state(forward_batch)
        b = forward_batch.batch_size
        return (
            self._sconv_cache(),
            meta.cache_indices[:b],
            meta.has_initial_state,
            self._weight_2d(),
            self._intermediate_window(),
        )

    def extend_fused_ar_inputs(self, forward_batch: ForwardBatch):

        meta = self._conv_state(forward_batch)
        precomputed = meta.precomputed


        dev = meta.cache_indices.device
        if meta.track_conv_indices is not None:
            track_rows = meta.track_conv_indices
            track_mask = forward_batch.mamba_track_mask
            track_dst = forward_batch.mamba_track_indices
        else:
            w1 = self.kernel_size[0] - 1
            track_rows = torch.empty((0, w1), dtype=torch.int64, device=dev)
            track_mask = torch.empty((0,), dtype=torch.bool, device=dev)
            track_dst = torch.empty((0,), dtype=torch.int64, device=dev)
        return (
            self._sconv_cache(),
            precomputed["safe_idx"],
            precomputed["cache_mask"].view(-1),
            precomputed["cu"],
            precomputed["si"],
            self._weight_2d(),
            meta.query_start_loc,
            meta.cache_indices,
            meta.has_initial_state,
            track_rows,
            track_mask,
            track_dst,
        )

    def verify_fused_ar_finish(
        self,
        forward_batch: ForwardBatch,
        x_scratch: torch.Tensor,
        cache_indices: torch.Tensor,
    ) -> None:

        meta = self._conv_state(forward_batch)
        self._save_intermediate_conv_windows(
            forward_batch=forward_batch,
            sconv_cache=self._sconv_cache(),
            cache_indices=cache_indices,
            hidden_states=x_scratch,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:

        del positions

        meta = self._conv_state(forward_batch)
        cache_indices = meta.cache_indices
        sconv_cache = self._sconv_cache()
        precomputed = meta.precomputed

        if forward_batch.forward_mode.is_target_verify():
            y = self._apply_causal_sconv_kernel(
                hidden_states=hidden_states,
                sconv_cache=sconv_cache,
                precomputed=precomputed,
            )
            self._save_intermediate_conv_windows(
                forward_batch=forward_batch,
                sconv_cache=sconv_cache,
                cache_indices=cache_indices,
                hidden_states=hidden_states,
            )

        elif forward_batch.forward_mode.is_extend(include_draft_extend_v2=True):
            self._prepare_extend_sconv_cache(
                forward_batch, sconv_cache, hidden_states, meta.track_conv_indices
            )

            if forward_batch.forward_mode.is_draft_extend_v2():
                y = self._apply_causal_sconv_kernel(
                    hidden_states=hidden_states,
                    sconv_cache=sconv_cache,
                    precomputed=precomputed,
                )
                self._update_sconv_cache_for_draft_extend(
                    forward_batch,
                    sconv_cache,
                    cache_indices,
                    hidden_states,
                )
            else:
                y = self._apply_training_sconv_kernel(
                    hidden_states=hidden_states,
                    sconv_cache=sconv_cache,
                    cache_indices=cache_indices,
                    query_start_loc=meta.query_start_loc,
                    has_initial_state=meta.has_initial_state,
                    precomputed=precomputed,
                )
        else:


            y = self._apply_decode_sconv_kernel(
                hidden_states=hidden_states,
                sconv_cache=sconv_cache,
                cache_indices=cache_indices,
                precomputed=precomputed,
                forward_batch=forward_batch,
            )

        return y


@triton.jit
def _fused_gather_scatter_to_sconv_cache_kernel(
    hidden_ptr,
    sconv_cache_ptr,
    track_idx_ptr,
    mask_ptr,
    dst_ptr,
    stride_hs_t,
    stride_cache_slot,
    stride_cache_w,
    stride_track_b,
    stride_track_w,
    stride_dst_b,
    D,
    W_MINUS_1: tl.constexpr,
    BLOCK_D: tl.constexpr,
):

    bid = tl.program_id(0)
    pid_d = tl.program_id(1)

    if not tl.load(mask_ptr + bid):
        return

    dst_slot = tl.load(dst_ptr + bid * stride_dst_b).to(tl.int64)
    d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_off < D

    for w in tl.static_range(W_MINUS_1):
        src_t = tl.load(track_idx_ptr + bid * stride_track_b + w * stride_track_w).to(
            tl.int64
        )
        val = tl.load(hidden_ptr + src_t * stride_hs_t + d_off, mask=d_mask, other=0.0)
        tl.store(
            sconv_cache_ptr + dst_slot * stride_cache_slot + w * stride_cache_w + d_off,
            val,
            mask=d_mask,
        )


def fused_gather_scatter_to_sconv_cache(
    hidden_states: torch.Tensor,
    sconv_cache: torch.Tensor,
    track_conv_indices: torch.Tensor,
    mask: torch.Tensor,
    dst_indices: torch.Tensor,
) -> None:

    D = hidden_states.shape[-1]
    W_minus_1 = sconv_cache.shape[1]
    BLOCK_D = min(triton.next_power_of_2(D), 1024)
    B = mask.shape[0]

    if (
        is_cuda()
        and hidden_states.dtype == torch.bfloat16
        and D % 2 == 0
        and hidden_states.stride(-1) == 1
        and sconv_cache.stride(2) == 1
    ):
        from sglang.kernels.ops.mamba.inkling_sconv import (
            fused_gather_scatter_to_sconv_cache as _cuda_gs,
        )

        _cuda_gs(
            hidden_states,
            sconv_cache,
            track_conv_indices.to(torch.int32),
            mask,
            dst_indices.to(torch.int64),
        )
        return

    grid = (B, triton.cdiv(D, BLOCK_D))
    _fused_gather_scatter_to_sconv_cache_kernel[grid](
        hidden_states,
        sconv_cache,
        track_conv_indices,
        mask,
        dst_indices,
        hidden_states.stride(0),
        sconv_cache.stride(0),
        sconv_cache.stride(1),
        track_conv_indices.stride(0),
        track_conv_indices.stride(1),
        dst_indices.stride(0),
        D,
        W_MINUS_1=W_minus_1,
        BLOCK_D=BLOCK_D,
    )


@triton.jit
def _fused_draft_extend_sconv_cache_kernel(
    hidden_ptr,
    sconv_cache_ptr,
    cache_indices_ptr,
    num_accept_ptr,
    crossed_ptr,
    track_step_ptr,
    mamba_track_indices_ptr,
    stride_hs_b,
    stride_hs_t,
    stride_cache_slot,
    stride_cache_w,
    D,
    W_MINUS_1: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DO_TRACKING: tl.constexpr,
):

    bid = tl.program_id(0)
    pid_d = tl.program_id(1)


    n_acc = tl.load(num_accept_ptr + bid).to(tl.int64)
    if n_acc < 0:
        return

    d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_off < D
    ci = tl.load(cache_indices_ptr + bid).to(tl.int64)


    if DO_TRACKING:
        do_track = tl.load(crossed_ptr + bid)
        if do_track:
            step = tl.load(track_step_ptr + bid).to(tl.int64)
            dst_slot = tl.load(mamba_track_indices_ptr + bid)
            for w in tl.static_range(W_MINUS_1):
                pos = step + w
                if pos < W_MINUS_1:
                    val = tl.load(
                        sconv_cache_ptr
                        + ci * stride_cache_slot
                        + pos * stride_cache_w
                        + d_off,
                        mask=d_mask,
                        other=0.0,
                    )
                else:
                    val = tl.load(
                        hidden_ptr
                        + bid * stride_hs_b
                        + (pos - W_MINUS_1) * stride_hs_t
                        + d_off,
                        mask=d_mask,
                        other=0.0,
                    )
                tl.store(
                    sconv_cache_ptr
                    + dst_slot * stride_cache_slot
                    + w * stride_cache_w
                    + d_off,
                    val,
                    mask=d_mask,
                )


    for w in tl.static_range(W_MINUS_1):
        pos = n_acc + w
        if pos < W_MINUS_1:
            val = tl.load(
                sconv_cache_ptr + ci * stride_cache_slot + pos * stride_cache_w + d_off,
                mask=d_mask,
                other=0.0,
            )
        else:
            val = tl.load(
                hidden_ptr
                + bid * stride_hs_b
                + (pos - W_MINUS_1) * stride_hs_t
                + d_off,
                mask=d_mask,
                other=0.0,
            )
        tl.store(
            sconv_cache_ptr + ci * stride_cache_slot + w * stride_cache_w + d_off,
            val,
            mask=d_mask,
        )


def fused_draft_extend_sconv_cache(
    hidden_states: torch.Tensor,
    sconv_cache: torch.Tensor,
    cache_indices: torch.Tensor,
    num_accept_tokens: torch.Tensor,
    draft_token_num: int,
    do_tracking: bool = False,
    crossed: torch.Tensor | None = None,
    track_step: torch.Tensor | None = None,
    mamba_track_indices: torch.Tensor | None = None,
) -> None:

    B = cache_indices.shape[0]
    D = sconv_cache.shape[2]
    W_minus_1 = sconv_cache.shape[1]

    if (
        is_cuda()
        and hidden_states.ndim == 2
        and hidden_states.dtype == torch.bfloat16
        and D % 2 == 0
        and hidden_states.stride(-1) == 1
        and sconv_cache.stride(2) == 1
    ):
        from sglang.kernels.ops.mamba.inkling_sconv import (
            fused_draft_extend_sconv_cache as _cuda_de,
        )

        _cuda_de(
            hidden_states,
            sconv_cache,
            cache_indices.to(torch.int32),
            num_accept_tokens.to(torch.int32),
            draft_token_num,
            do_tracking,
            crossed,
            track_step.to(torch.int32) if track_step is not None else None,
            (
                mamba_track_indices.to(torch.int64)
                if mamba_track_indices is not None
                else None
            ),
        )
        return

    BLOCK_D = min(triton.next_power_of_2(D), 1024)

    if hidden_states.ndim == 2:
        stride_hs_b = hidden_states.stride(0) * draft_token_num
        stride_hs_t = hidden_states.stride(0)
    else:
        stride_hs_b = hidden_states.stride(0)
        stride_hs_t = hidden_states.stride(1)


    _dummy_bool = torch.zeros(1, dtype=torch.bool, device=sconv_cache.device)
    _dummy_int = torch.zeros(1, dtype=torch.int32, device=sconv_cache.device)
    _dummy_int64 = torch.zeros(1, dtype=torch.int64, device=sconv_cache.device)

    grid = (B, triton.cdiv(D, BLOCK_D))
    _fused_draft_extend_sconv_cache_kernel[grid](
        hidden_states,
        sconv_cache,
        cache_indices.to(torch.int32).contiguous(),
        num_accept_tokens.to(torch.int32).contiguous(),
        crossed if do_tracking else _dummy_bool,
        track_step.to(torch.int32).contiguous() if do_tracking else _dummy_int,
        (
            mamba_track_indices.to(torch.int64).contiguous()
            if do_tracking
            else _dummy_int64
        ),
        stride_hs_b,
        stride_hs_t,
        sconv_cache.stride(0),
        sconv_cache.stride(1),
        D,
        W_MINUS_1=W_minus_1,
        BLOCK_D=BLOCK_D,
        DO_TRACKING=do_tracking,
    )
