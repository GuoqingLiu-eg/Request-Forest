# SPDX-License-Identifier: Apache-2.0
# Kimi-K3 Attention Residual: snapshot bank + aggregation.
#
# The public API is the AttnResidual class (constructed once per forward pass).
# It owns the frozen snapshot bank [T, NB, H] and the valid-row counter, and
# dispatches each aggregation point (score rows → softmax → weighted sum →
# RMSNorm) by hardware capability:
#   fast  — warp-specialized TMA kernel: cp.async.bulk producer +
#           online-softmax consumers over a double-buffered chunk ring, out
#           norm fused, per-nvb tuned launch config, one persistent CTA per
#           SM. Taken on SM100+ except SM12x with H=7168.
#   hip   — single Triton kernel, everything in one launch; taken on ROCm
#           within its register budget.
#   fused — Triton 2-kernel pipeline with full H-parallelism; the fallback
#           everywhere the fast kernel does not apply.
# aggregate_stream_torch is the eager reference (tests and the
# H % _BLOCK_H != 0 shape fallback of aggregate_stream).

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.utils import is_hip, is_npu

_BLOCK_H: int = 1024
_MAX_ROWS: int = 16

_FAST_SUPPORTED = None
_HIP_SHAPE_GATE = None


def _supports_attn_res_tma(capability: tuple[int, int]) -> bool:

    major, _ = capability
    return major >= 10 and major != 12


def _use_fast(hidden_size: int) -> bool:

    global _FAST_SUPPORTED
    if is_npu():
        return False
    if _FAST_SUPPORTED is None:
        _FAST_SUPPORTED = _supports_attn_res_tma(torch.cuda.get_device_capability())
    return _FAST_SUPPORTED and hidden_size == 7168


def _use_hip_fused(hidden_size: int, nvb: int) -> bool:

    if not is_hip():
        return False
    global _HIP_SHAPE_GATE
    if _HIP_SHAPE_GATE is None:
        from sglang.kernels.ops.kimi_k3.attn_res_hip import supports_attn_res_hip

        _HIP_SHAPE_GATE = supports_attn_res_hip
    return _HIP_SHAPE_GATE(hidden_size, nvb)


def get_cw(
    proj: ReplicatedLinear,
    norm: RMSNorm,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:

    cache = getattr(proj, "_attn_res_cw_cache", None)
    if cache is None:
        cache = {}
        proj._attn_res_cw_cache = cache
    cw = cache.get(dtype)
    if cw is None:
        cw = (norm.weight.float() * proj.weight.squeeze().float()).contiguous()
        cw = cache[dtype] = cw.to(dtype)
    return cw


def _aggregate_fast(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    nvb: int,
    score_proj: ReplicatedLinear,
    score_norm: RMSNorm,
    out_norm: RMSNorm,
    write_bank_row: bool = False,
) -> torch.Tensor:

    from sglang.kernels.ops.kimi_k3.attn_res import attn_res_fused_tma


    assert score_norm.variance_epsilon == out_norm.variance_epsilon

    cw = get_cw(score_proj, score_norm, dtype=torch.bfloat16)
    out = torch.empty_like(prefix_sum)
    attn_res_fused_tma(
        prefix_sum,
        bank,
        cw,
        out_norm.weight,
        out,
        nvb,
        score_norm.variance_epsilon,
        write_prefix=write_bank_row,
    )
    return out


@triton.jit
def _score_kernel(
    prefix_ptr,
    bank_ptr,
    cw_ptr,
    scores_ptr,
    NVB,
    eps,
    stride_pm,
    stride_bm,
    stride_bb,
    stride_sm,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):

    pid_t = tl.program_id(0)
    j = tl.program_id(1)
    if j > NVB:
        return
    sumsq = 0.0
    dotv = 0.0
    for h0 in tl.static_range(0, H, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        if j < NVB:
            v = tl.load(bank_ptr + pid_t * stride_bm + j * stride_bb + offs_h).to(
                tl.float32
            )
        else:
            v = tl.load(prefix_ptr + pid_t * stride_pm + offs_h).to(tl.float32)
        cw = tl.load(cw_ptr + offs_h)
        sumsq += tl.sum(v * v)
        dotv += tl.sum(v * cw)
    rrms = 1.0 / tl.sqrt(sumsq / H + eps)
    tl.store(scores_ptr + pid_t * stride_sm + j, dotv * rrms)


@triton.jit
def _combine_kernel(
    prefix_ptr,
    bank_ptr,
    scores_ptr,
    out_ptr,
    NVB,
    stride_pm,
    stride_bm,
    stride_bb,
    stride_sm,
    stride_om,
    BLOCK_H: tl.constexpr,
    MAX_ROWS: tl.constexpr,
):

    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    h0 = pid_h * BLOCK_H


    offs_b = tl.arange(0, MAX_ROWS)
    mask_b = offs_b <= NVB
    raw = tl.load(
        scores_ptr + pid_t * stride_sm + offs_b, mask=mask_b, other=float("-inf")
    )
    m = tl.max(raw, axis=0)
    e = tl.where(mask_b, tl.exp(raw - m), 0.0)
    p = e / tl.sum(e, axis=0)


    offs_h = h0 + tl.arange(0, BLOCK_H)
    acc = tl.zeros([BLOCK_H], tl.float32)
    for j in range(0, NVB + 1):
        if j < NVB:
            v = tl.load(bank_ptr + pid_t * stride_bm + j * stride_bb + offs_h).to(
                tl.float32
            )
        else:
            v = tl.load(prefix_ptr + pid_t * stride_pm + offs_h).to(tl.float32)
        p_j = tl.sum(tl.where(offs_b == j, p, 0.0), axis=0)
        acc += p_j * v
    tl.store(
        out_ptr + pid_t * stride_om + offs_h,
        acc.to(out_ptr.dtype.element_ty),
    )


def _mix_fused(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    nvb: int,
    score_proj: ReplicatedLinear,
    score_norm: RMSNorm,
) -> torch.Tensor:

    T, H = prefix_sum.shape
    if T == 0:
        return prefix_sum
    cw = get_cw(score_proj, score_norm)
    if is_npu():
        from sgl_kernel_npu.kimi_k3.attn_residual import mix_fused

        return mix_fused(
            prefix_sum,
            bank,
            nvb,
            cw,
            score_norm.variance_epsilon,
        )
    n_h_blocks = H // _BLOCK_H


    scores = torch.empty((T, _MAX_ROWS), dtype=torch.float32, device=prefix_sum.device)
    _score_kernel[(T, nvb + 1)](
        prefix_sum,
        bank,
        cw,
        scores,
        nvb,
        score_norm.variance_epsilon,
        prefix_sum.stride(0),
        bank.stride(0),
        bank.stride(1),
        scores.stride(0),
        H=H,
        BLOCK_H=_BLOCK_H,
        num_warps=8,
    )


    out = torch.empty_like(prefix_sum)
    _combine_kernel[(T, n_h_blocks)](
        prefix_sum,
        bank,
        scores,
        out,
        nvb,
        prefix_sum.stride(0),
        bank.stride(0),
        bank.stride(1),
        scores.stride(0),
        out.stride(0),
        BLOCK_H=_BLOCK_H,
        MAX_ROWS=_MAX_ROWS,
        num_warps=4,
    )
    return out


def _aggregate_fused(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    nvb: int,
    score_proj: ReplicatedLinear,
    score_norm: RMSNorm,
    out_norm: RMSNorm,
) -> torch.Tensor:

    return out_norm(_mix_fused(prefix_sum, bank, nvb, score_proj, score_norm))


def _aggregate_hip(
    prefix_sum: torch.Tensor,
    addend: Optional[torch.Tensor],
    bank: torch.Tensor,
    nvb: int,
    score_proj: ReplicatedLinear,
    score_norm: RMSNorm,
    out_norm: Optional[RMSNorm],
    write_bank_row: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:

    from sglang.kernels.ops.kimi_k3.attn_res_hip import attn_res_hip

    cw = get_cw(score_proj, score_norm)
    prefix = prefix_sum if addend is None else torch.empty_like(prefix_sum)
    out = torch.empty_like(prefix_sum)
    attn_res_hip(
        prefix_sum,
        bank,
        cw,
        out_norm.weight if out_norm is not None else None,
        out,
        nvb,
        score_norm.variance_epsilon,
        out_norm.variance_epsilon if out_norm is not None else 0.0,
        addend=addend,
        prefix_out=prefix,
        write_prefix=write_bank_row,
    )
    return out, prefix


def aggregate_stream_torch(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    nvb: int,
    score_proj: ReplicatedLinear,
    score_norm: RMSNorm,
) -> torch.Tensor:

    if nvb == 0:
        return prefix_sum
    T, H = prefix_sum.shape

    rows = torch.cat([bank[:, :nvb, :], prefix_sum.unsqueeze(1)], dim=1)
    R = nvb + 1
    normed = score_norm(rows.reshape(T * R, H))
    scores = score_proj(normed)[0].reshape(T, R)
    probs = torch.softmax(scores.float(), dim=-1)
    mixed = (probs.unsqueeze(-1) * rows.float()).sum(dim=1)
    return mixed.to(prefix_sum.dtype)


def aggregate_stream(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    nvb: int,
    score_proj: ReplicatedLinear,
    score_norm: RMSNorm,
) -> torch.Tensor:

    if nvb == 0:
        return prefix_sum
    if _use_hip_fused(prefix_sum.shape[1], nvb):
        return _aggregate_hip(
            prefix_sum, None, bank, nvb, score_proj, score_norm, None
        )[0]
    if prefix_sum.shape[1] % _BLOCK_H != 0:
        return aggregate_stream_torch(prefix_sum, bank, nvb, score_proj, score_norm)
    return _mix_fused(prefix_sum, bank, nvb, score_proj, score_norm)


def _aggregate_fused_add(
    prefix_a: torch.Tensor,
    prefix_b: torch.Tensor,
    bank: torch.Tensor,
    nvb: int,
    score_proj: ReplicatedLinear,
    score_norm: RMSNorm,
    out_norm: RMSNorm,
    write_bank_row: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:

    if _use_hip_fused(prefix_a.shape[1], nvb):

        return _aggregate_hip(
            prefix_a,
            prefix_b,
            bank,
            nvb,
            score_proj,
            score_norm,
            out_norm,
            write_bank_row=write_bank_row,
        )
    prefix = prefix_a + prefix_b
    return (
        _aggregate(
            prefix,
            bank,
            nvb,
            score_proj,
            score_norm,
            out_norm,
            write_bank_row=write_bank_row,
        ),
        prefix,
    )


def _aggregate(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    nvb: int,
    score_proj: ReplicatedLinear,
    score_norm: RMSNorm,
    out_norm: RMSNorm,
    write_bank_row: bool = False,
) -> torch.Tensor:

    if prefix_sum.shape[0] == 0:
        return prefix_sum
    if _use_fast(prefix_sum.shape[1]):
        return _aggregate_fast(
            prefix_sum,
            bank,
            nvb,
            score_proj,
            score_norm,
            out_norm,
            write_bank_row=write_bank_row,
        )
    if _use_hip_fused(prefix_sum.shape[1], nvb):
        return _aggregate_hip(
            prefix_sum,
            None,
            bank,
            nvb,
            score_proj,
            score_norm,
            out_norm,
            write_bank_row=write_bank_row,
        )[0]
    assert not write_bank_row, "fused bank write is fast-path only"
    return _aggregate_fused(prefix_sum, bank, nvb, score_proj, score_norm, out_norm)


class AttnResidual:


    def __init__(
        self,
        hidden_states: torch.Tensor,
        block_num: int,
        block_residual: Optional[torch.Tensor] = None,
    ) -> None:
        num_tokens, hidden_size = hidden_states.shape


        self.block_residual = hidden_states.new_empty(
            (num_tokens, block_num, hidden_size)
        )
        self.num_valid_blocks = 0
        if block_residual is not None:
            self.num_valid_blocks = block_residual.size(1)
            self.block_residual[:, : self.num_valid_blocks, :].copy_(block_residual)

    def write(self, prefix_sum: torch.Tensor, rows: Optional[slice] = None) -> None:

        bank = self.block_residual if rows is None else self.block_residual[rows]
        bank[:, self.num_valid_blocks, :].copy_(prefix_sum)
        self.num_valid_blocks += 1

    def forward(
        self,
        hidden_states: torch.Tensor,
        prefix_sum: Optional[torch.Tensor],
        score_proj: ReplicatedLinear,
        score_norm: RMSNorm,
        out_norm: RMSNorm,
        rows: Optional[slice] = None,
        write: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        nvb = self.num_valid_blocks

        if nvb == 0:
            assert prefix_sum is None
            if write:
                self.write(hidden_states, rows)
            return out_norm(hidden_states), hidden_states


        block_residual = (
            self.block_residual if rows is None else self.block_residual[rows]
        )

        fused_write = write and (
            _use_fast(hidden_states.shape[1])
            or _use_hip_fused(hidden_states.shape[1], nvb)
        )
        if prefix_sum is None:


            normed = _aggregate(
                hidden_states,
                block_residual,
                nvb,
                score_proj,
                score_norm,
                out_norm,
                write_bank_row=fused_write,
            )
            prefix = hidden_states
        else:

            normed, prefix = _aggregate_fused_add(
                prefix_sum,
                hidden_states,
                block_residual,
                nvb,
                score_proj,
                score_norm,
                out_norm,
                write_bank_row=fused_write,
            )
        if fused_write:
            self.num_valid_blocks += 1
        elif write:
            self.write(prefix, rows)
        return normed, prefix

    def forward_sp_all_gather(
        self,
        hidden_states: torch.Tensor,
        prefix_sum: Optional[torch.Tensor],
        score_proj: ReplicatedLinear,
        score_norm: RMSNorm,
        out_norm: RMSNorm,
        rows: slice,
        write: bool = False,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:

        nvb = self.num_valid_blocks
        if nvb == 0:
            return None
        if prefix_sum is not None and prefix_sum.shape != hidden_states.shape:
            prefix_sum = prefix_sum[rows]
        prefix = hidden_states if prefix_sum is None else prefix_sum.add(hidden_states)
        bank = self.block_residual[rows]
        cw = get_cw(score_proj, score_norm, dtype=torch.bfloat16)
        assert score_norm.variance_epsilon == out_norm.variance_epsilon
        from sglang.srt.layers import k3_sp_collective

        normed = k3_sp_collective.attn_res_all_gather(
            prefix,
            bank,
            cw,
            out_norm.weight,
            nvb,
            score_norm.variance_epsilon,
            write_prefix=write,
        )
        if normed is None:
            return None
        if write:
            self.num_valid_blocks += 1
        return normed, prefix

    def forward_sp_reduce_scatter(
        self,
        hidden_states: torch.Tensor,
        prefix_sum: Optional[torch.Tensor],
        score_proj: ReplicatedLinear,
        score_norm: RMSNorm,
        out_norm: RMSNorm,
        rows: slice,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:

        nvb = self.num_valid_blocks
        if nvb == 0:
            return None
        local_tokens = rows.stop - rows.start
        residual = prefix_sum
        if residual is not None and residual.shape[0] != local_tokens:
            residual = residual[rows]
        bank = self.block_residual[rows]
        cw = get_cw(score_proj, score_norm, dtype=torch.bfloat16)
        assert score_norm.variance_epsilon == out_norm.variance_epsilon
        from sglang.srt.layers import k3_sp_collective

        return k3_sp_collective.reduce_scatter_attn_res(
            hidden_states,
            residual,
            bank,
            cw,
            out_norm.weight,
            nvb,
            score_norm.variance_epsilon,
        )
