# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

_CAM_IDENTITY_CACHE: dict = {}


@dataclass(frozen=True)
class _PhaseCfg:
    nw: int
    BS: int = 0
    ns: int = 1
    use_acc: bool = False


@dataclass(frozen=True)
class _ChunkwiseCfg:
    A: _PhaseCfg
    B: _PhaseCfg
    C: _PhaseCfg

    def as_tuple(self) -> tuple:

        return (
            self.A.nw,
            self.A.BS,
            self.B.nw,
            self.B.ns,
            self.B.use_acc,
            self.C.nw,
            self.C.BS,
            self.C.ns,
        )


_CHUNKWISE_TUNING: dict[tuple[str, str], _ChunkwiseCfg] = {


    ("ampere", "bf16"): _ChunkwiseCfg(
        A=_PhaseCfg(nw=8, BS=32),
        B=_PhaseCfg(nw=8, use_acc=False, ns=1),
        C=_PhaseCfg(nw=4, BS=32, ns=1),
    ),
    ("ampere", "fp32"): _ChunkwiseCfg(


        A=_PhaseCfg(nw=16, BS=32),
        B=_PhaseCfg(nw=32, use_acc=False, ns=2),
        C=_PhaseCfg(
            nw=16, BS=32, ns=1
        ),
    ),


    ("hopper", "bf16"): _ChunkwiseCfg(
        A=_PhaseCfg(nw=8, BS=64),
        B=_PhaseCfg(nw=4, use_acc=False, ns=1),
        C=_PhaseCfg(nw=8, BS=32, ns=1),
    ),
    ("hopper", "fp32"): _ChunkwiseCfg(
        A=_PhaseCfg(nw=8, BS=32),
        B=_PhaseCfg(
            nw=32, use_acc=False, ns=1
        ),
        C=_PhaseCfg(
            nw=16, BS=32, ns=1
        ),
    ),


    ("blackwell_dc", "bf16"): _ChunkwiseCfg(
        A=_PhaseCfg(nw=4, BS=64),
        B=_PhaseCfg(nw=4, use_acc=False, ns=1),
        C=_PhaseCfg(nw=8, BS=64, ns=1),
    ),
    ("blackwell_dc", "fp32"): _ChunkwiseCfg(
        A=_PhaseCfg(
            nw=8, BS=128
        ),
        B=_PhaseCfg(
            nw=32, use_acc=False, ns=3
        ),
        C=_PhaseCfg(
            nw=4, BS=64, ns=1
        ),
    ),


    ("blackwell_spark", "bf16"): _ChunkwiseCfg(
        A=_PhaseCfg(nw=8, BS=32),
        B=_PhaseCfg(
            nw=8, use_acc=False, ns=1
        ),


        C=_PhaseCfg(nw=4, BS=32, ns=1),
    ),
    ("blackwell_spark", "fp32"): _ChunkwiseCfg(
        A=_PhaseCfg(nw=8, BS=16),


        B=_PhaseCfg(nw=8, use_acc=False, ns=1),
        C=_PhaseCfg(nw=8, BS=16, ns=1),
    ),
}


_CHUNKWISE_SHAPE_OVERRIDES: dict[tuple[str, str, str], _ChunkwiseCfg] = {}


_ARCH_OVERRIDES: dict = {}


def _arch_key(cap: tuple) -> str:

    if cap[0] == 8:
        return "ampere"
    if cap[0] == 9:
        return "hopper"
    if cap[0] >= 10:
        has_big_sram = True
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            smem = getattr(props, "shared_memory_per_multiprocessor", 228 * 1024)
            has_big_sram = smem >= 150 * 1024
        return "blackwell_dc" if has_big_sram else "blackwell_spark"
    return "ampere"


def _prec_key(dot_prec: int) -> str:
    return "fp32" if dot_prec >= 1 else "bf16"


def _auto_config(dot_prec: int, cap: tuple, shape_hint: str | None = None) -> tuple:

    arch = _arch_key(cap)
    prec = _prec_key(dot_prec)

    if shape_hint is not None:
        cfg = _CHUNKWISE_SHAPE_OVERRIDES.get((arch, prec, shape_hint))
        if cfg is not None:
            return cfg.as_tuple()

    cfg = _CHUNKWISE_TUNING.get((arch, prec)) or _CHUNKWISE_TUNING[("ampere", prec)]
    return cfg.as_tuple()


def _get_arch_config(
    dot_precision: int = 0,
    shape_hint: str | None = None,
    device: torch.device | int | None = None,
):

    if not torch.cuda.is_available():
        cap = (9, 0)
    else:
        if device is None:
            dev_idx = torch.cuda.current_device()
        elif isinstance(device, int):
            dev_idx = device
        else:
            dev_idx = (
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
        cap = torch.cuda.get_device_capability(dev_idx)
    key = (cap, dot_precision)
    if key in _ARCH_OVERRIDES:
        return _ARCH_OVERRIDES[key]
    return _auto_config(dot_precision, cap, shape_hint)


@triton.jit
def _phase_a_kv_kernel(
    qkv_ptr,
    stride_b: tl.constexpr,
    stride_n: tl.constexpr,
    stride_3: tl.constexpr,
    stride_h: tl.constexpr,
    stride_d: tl.constexpr,
    beta_ptr,
    k_inv_rms_ptr,
    k_norm_w_ptr,
    rope_cos_ptr,
    rope_sin_ptr,
    I_minus_P_kv_ptr,
    A_ptr,
    H: tl.constexpr,
    F: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    K_SCALE,
    NORM_EPS: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    SKIP_RELU: tl.constexpr = False,
):
    if DOT_PRECISION >= 1:
        dot_dtype = tl.float32
    else:
        dot_dtype = tl.bfloat16
    dot_ip: tl.constexpr = "ieee" if DOT_PRECISION == 2 else "tf32"

    pid = tl.program_id(0)
    pid_b = pid // (H * F)
    pid_hf = pid % (H * F)
    pid_h = pid_hf // F
    pid_f = pid_hf % F
    bh = pid_b * H + pid_h
    N: tl.constexpr = F * S

    qkv_bh = qkv_ptr + pid_b * stride_b + pid_h * stride_h
    beta_bhf = beta_ptr + bh * (F * S) + pid_f * S
    I_P_kv_bhf = (
        I_minus_P_kv_ptr + bh * F * BLOCK_D * BLOCK_D + pid_f * BLOCK_D * BLOCK_D
    )
    A_bhf = A_ptr + bh * F * BLOCK_D * BLOCK_D + pid_f * BLOCK_D * BLOCK_D

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    offs_d_pair = offs_d ^ 1
    mask_d_pair = offs_d_pair < D

    nw_offset = pid_h * D
    k_nw = tl.load(k_norm_w_ptr + nw_offset + offs_d, mask=mask_d, other=0.0).to(
        tl.float32
    )
    k_nw_pair = tl.load(
        k_norm_w_ptr + nw_offset + offs_d_pair, mask=mask_d_pair, other=0.0
    ).to(tl.float32)


    P_kv_acc = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)
    A_acc = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)

    k_scale = K_SCALE
    n_base = pid_f * S

    for s0 in range(0, S, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        mask_s = offs_s < S
        mask_sd = mask_s[:, None] & mask_d[None, :]
        n_idx = n_base + offs_s

        k_ptrs = (
            qkv_bh
            + n_idx[:, None] * stride_n
            + 1 * stride_3
            + offs_d[None, :] * stride_d
        )
        v_ptrs = (
            qkv_bh
            + n_idx[:, None] * stride_n
            + 2 * stride_3
            + offs_d[None, :] * stride_d
        )
        K_raw = tl.load(k_ptrs, mask=mask_sd, other=0.0).to(tl.float32)
        V_raw = tl.load(v_ptrs, mask=mask_sd, other=0.0).to(tl.float32)
        beta_t = tl.load(beta_bhf + offs_s, mask=mask_s, other=0.0).to(tl.float32)

        k_inv_rms = tl.load(
            k_inv_rms_ptr + pid_b * N + n_idx, mask=mask_s, other=1.0
        ).to(tl.float32)
        K_normed = K_raw * k_inv_rms[:, None] * k_nw[None, :]
        if SKIP_RELU:
            K = K_normed * k_scale
        else:
            K = tl.where(K_normed > 0, K_normed, 0.0) * k_scale

        K_pair_raw = tl.reshape(
            tl.flip(tl.reshape(K_raw, (BLOCK_S, BLOCK_D // 2, 2)), dim=2),
            (BLOCK_S, BLOCK_D),
        )
        K_pair_normed = K_pair_raw * k_inv_rms[:, None] * k_nw_pair[None, :]
        if SKIP_RELU:
            K_pair = K_pair_normed * k_scale
        else:
            K_pair = tl.where(K_pair_normed > 0, K_pair_normed, 0.0) * k_scale

        rope_ptrs = n_idx[:, None] * D + offs_d[None, :]
        Cos = tl.load(rope_cos_ptr + rope_ptrs, mask=mask_sd, other=1.0).to(tl.float32)
        Sin = tl.load(rope_sin_ptr + rope_ptrs, mask=mask_sd, other=0.0).to(tl.float32)
        K_rot = K * Cos + K_pair * Sin

        beta_Krot = beta_t[:, None] * K_rot
        beta_V = beta_t[:, None] * V_raw

        K_rot_T = tl.trans(K_rot)
        P_kv_acc += tl.dot(
            K_rot_T.to(dot_dtype),
            beta_Krot.to(dot_dtype),
            out_dtype=tl.float32,
            input_precision=dot_ip,
        )
        A_acc += tl.dot(
            K_rot_T.to(dot_dtype),
            beta_V.to(dot_dtype),
            out_dtype=tl.float32,
            input_precision=dot_ip,
        )


    offs_dd = offs_d[:, None] * BLOCK_D + offs_d[None, :]
    diag_in_range = (
        (offs_d[:, None] == offs_d[None, :]) & mask_d[:, None] & mask_d[None, :]
    )
    I_minus_P_kv = tl.where(diag_in_range, 1.0 - P_kv_acc, -P_kv_acc)
    if DOT_PRECISION >= 1:
        tl.store(I_P_kv_bhf + offs_dd, I_minus_P_kv)
        tl.store(A_bhf + offs_dd, A_acc)
    else:
        tl.store(I_P_kv_bhf + offs_dd, I_minus_P_kv.to(tl.bfloat16))
        tl.store(A_bhf + offs_dd, A_acc.to(tl.bfloat16))


@triton.jit
def _phase_a_z_kernel(
    qkv_ptr,
    stride_b: tl.constexpr,
    stride_n: tl.constexpr,
    stride_3: tl.constexpr,
    stride_h: tl.constexpr,
    stride_d: tl.constexpr,
    beta_ptr,
    k_inv_rms_ptr,
    k_norm_w_ptr,
    I_minus_P_z_ptr,
    B_ptr,
    H: tl.constexpr,
    F: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    K_SCALE,
    NORM_EPS: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):

    if DOT_PRECISION >= 1:
        dot_dtype = tl.float32
    else:
        dot_dtype = tl.bfloat16
    dot_ip: tl.constexpr = "ieee" if DOT_PRECISION == 2 else "tf32"

    pid = tl.program_id(0)
    pid_b = pid // (H * F)
    pid_hf = pid % (H * F)
    pid_h = pid_hf // F
    pid_f = pid_hf % F
    bh = pid_b * H + pid_h
    N: tl.constexpr = F * S

    qkv_bh = qkv_ptr + pid_b * stride_b + pid_h * stride_h
    beta_bhf = beta_ptr + bh * (F * S) + pid_f * S
    I_P_z_bhf = I_minus_P_z_ptr + bh * F * BLOCK_D * BLOCK_D + pid_f * BLOCK_D * BLOCK_D
    B_bhf = B_ptr + bh * F * BLOCK_D + pid_f * BLOCK_D

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    nw_offset = pid_h * D
    k_nw = tl.load(k_norm_w_ptr + nw_offset + offs_d, mask=mask_d, other=0.0).to(
        tl.float32
    )

    P_z_acc = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)
    B_acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    k_scale = K_SCALE
    n_base = pid_f * S

    for s0 in range(0, S, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        mask_s = offs_s < S
        mask_sd = mask_s[:, None] & mask_d[None, :]
        n_idx = n_base + offs_s

        k_ptrs = (
            qkv_bh
            + n_idx[:, None] * stride_n
            + 1 * stride_3
            + offs_d[None, :] * stride_d
        )
        K_raw = tl.load(k_ptrs, mask=mask_sd, other=0.0).to(tl.float32)
        beta_t = tl.load(beta_bhf + offs_s, mask=mask_s, other=0.0).to(tl.float32)

        k_inv_rms = tl.load(
            k_inv_rms_ptr + pid_b * N + n_idx, mask=mask_s, other=1.0
        ).to(tl.float32)
        K_normed = K_raw * k_inv_rms[:, None] * k_nw[None, :]
        K = (tl.where(K_normed > 0, K_normed, 0.0) * k_scale).to(tl.float32)

        beta_K = beta_t[:, None] * K

        K_T = tl.trans(K)
        P_z_acc += tl.dot(
            K_T.to(dot_dtype),
            beta_K.to(dot_dtype),
            out_dtype=tl.float32,
            input_precision=dot_ip,
        )
        B_acc += tl.sum(beta_K, axis=0)

    offs_dd = offs_d[:, None] * BLOCK_D + offs_d[None, :]
    diag_in_range = (
        (offs_d[:, None] == offs_d[None, :]) & mask_d[:, None] & mask_d[None, :]
    )
    I_minus_P_z = tl.where(diag_in_range, 1.0 - P_z_acc, -P_z_acc)

    if DOT_PRECISION >= 1:
        tl.store(I_P_z_bhf + offs_dd, I_minus_P_z)
    else:
        tl.store(I_P_z_bhf + offs_dd, I_minus_P_z.to(tl.bfloat16))

    tl.store(B_bhf + offs_d, B_acc)


def phase_a(
    qkv: torch.Tensor,
    beta: torch.Tensor,
    q_inv_rms: torch.Tensor,
    k_inv_rms: torch.Tensor,
    q_norm_w: torch.Tensor,
    k_norm_w: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    F: int,
    S: int,
    k_scale: float = 1.0,
    norm_eps: float = 1e-5,
    num_warps: int | None = None,
    num_stages: int = 1,
    BLOCK_S: int | None = None,
    dot_precision: int = 0,
    skip_relu: bool = False,
    skip_z: bool = False,
):

    if num_warps is None or BLOCK_S is None:
        a_w, a_bs, *_ = _get_arch_config(dot_precision, device=qkv.device)
        if num_warps is None:
            num_warps = a_w
        if BLOCK_S is None:
            BLOCK_S = a_bs
    B, N, three, H, D = qkv.shape
    assert three == 3 and N == F * S
    BLOCK_D = triton.next_power_of_2(D)
    BH = B * H


    bridge_dtype = torch.float32 if dot_precision >= 1 else torch.bfloat16
    I_P_kv = torch.empty(BH, F, BLOCK_D, BLOCK_D, device=qkv.device, dtype=bridge_dtype)
    A = torch.empty(BH, F, BLOCK_D, BLOCK_D, device=qkv.device, dtype=bridge_dtype)

    beta_c = beta.contiguous()
    grid = (BH * F,)

    _phase_a_kv_kernel[grid](
        qkv,
        qkv.stride(0),
        qkv.stride(1),
        qkv.stride(2),
        qkv.stride(3),
        qkv.stride(4),
        beta_c,
        k_inv_rms,
        k_norm_w,
        rope_cos,
        rope_sin,
        I_P_kv,
        A,
        H=H,
        F=F,
        S=S,
        D=D,
        K_SCALE=k_scale,
        NORM_EPS=norm_eps,
        DOT_PRECISION=dot_precision,
        BLOCK_D=BLOCK_D,
        BLOCK_S=BLOCK_S,
        SKIP_RELU=skip_relu,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if skip_z:


        I_P_z = torch.empty(1, device=qkv.device, dtype=bridge_dtype)
        B_z = torch.empty(1, device=qkv.device, dtype=torch.float32)
        return I_P_kv, A, I_P_z, B_z

    I_P_z = torch.empty(BH, F, BLOCK_D, BLOCK_D, device=qkv.device, dtype=bridge_dtype)

    B_z = torch.empty(BH, F, BLOCK_D, device=qkv.device, dtype=torch.float32)

    _phase_a_z_kernel[grid](
        qkv,
        qkv.stride(0),
        qkv.stride(1),
        qkv.stride(2),
        qkv.stride(3),
        qkv.stride(4),
        beta_c,
        k_inv_rms,
        k_norm_w,
        I_P_z,
        B_z,
        H=H,
        F=F,
        S=S,
        D=D,
        K_SCALE=k_scale,
        NORM_EPS=norm_eps,
        DOT_PRECISION=dot_precision,
        BLOCK_D=BLOCK_D,
        BLOCK_S=BLOCK_S,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return I_P_kv, A, I_P_z, B_z


@triton.jit
def _phase_b_kernel(
    I_P_kv_ptr,
    A_ptr,
    I_P_z_ptr,
    B_ptr,
    decay_ptr,
    M_fwd_ptr,
    z_fwd_ptr,
    M_rev_ptr,
    z_rev_ptr,
    init_state_kv_ptr,
    init_state_z_ptr,
    final_state_kv_ptr,
    final_state_z_ptr,
    BH: tl.constexpr,
    F: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    USE_ACC_FUSION: tl.constexpr,
    LOAD_INIT_STATE: tl.constexpr,
    SAVE_FINAL_STATE: tl.constexpr,
    DIRECTION: tl.constexpr,
    COMBINED_HISTORY: tl.constexpr,


    SKIP_Z: tl.constexpr,
):
    if DOT_PRECISION >= 1:
        dot_dtype = tl.float32
    else:
        dot_dtype = tl.bfloat16
    dot_ip: tl.constexpr = "ieee" if DOT_PRECISION == 2 else "tf32"

    pid = tl.program_id(0)
    bh = pid

    offs_d = tl.arange(0, BLOCK_D)
    offs_dd = offs_d[:, None] * BLOCK_D + offs_d[None, :]


    if DIRECTION != 2:
        if LOAD_INIT_STATE:
            M = tl.load(init_state_kv_ptr + bh * BLOCK_D * BLOCK_D + offs_dd).to(
                tl.float32
            )
            if not SKIP_Z:
                z = tl.load(init_state_z_ptr + bh * BLOCK_D + offs_d).to(tl.float32)
        else:
            M = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)
            if not SKIP_Z:
                z = tl.zeros([BLOCK_D], dtype=tl.float32)
        for f in range(F):
            I_P_kv_f = tl.load(
                I_P_kv_ptr
                + bh * F * BLOCK_D * BLOCK_D
                + f * BLOCK_D * BLOCK_D
                + offs_dd
            )
            A_f = tl.load(
                A_ptr + bh * F * BLOCK_D * BLOCK_D + f * BLOCK_D * BLOCK_D + offs_dd
            )
            g_f = tl.load(decay_ptr + bh * F + f).to(tl.float32)


            if USE_ACC_FUSION:


                I_P_scaled = I_P_kv_f.to(tl.float32) * g_f
                M = tl.dot(
                    I_P_scaled.to(dot_dtype),
                    M.to(dot_dtype),
                    acc=A_f.to(tl.float32),
                    out_dtype=tl.float32,
                    input_precision=dot_ip,
                )
            else:
                M_temp = tl.dot(
                    I_P_kv_f.to(dot_dtype),
                    M.to(dot_dtype),
                    out_dtype=tl.float32,
                    input_precision=dot_ip,
                )
                M = g_f * M_temp + A_f

            tl.store(
                M_fwd_ptr
                + bh * F * BLOCK_D * BLOCK_D
                + f * BLOCK_D * BLOCK_D
                + offs_dd,
                M,
            )
            if not SKIP_Z:
                I_P_z_f = tl.load(
                    I_P_z_ptr
                    + bh * F * BLOCK_D * BLOCK_D
                    + f * BLOCK_D * BLOCK_D
                    + offs_dd
                )
                B_f = tl.load(B_ptr + bh * F * BLOCK_D + f * BLOCK_D + offs_d)

                z_temp = tl.sum(I_P_z_f * z[None, :], axis=1)
                z = g_f * z_temp + B_f
                tl.store(z_fwd_ptr + bh * F * BLOCK_D + f * BLOCK_D + offs_d, z)


        if SAVE_FINAL_STATE:
            tl.store(final_state_kv_ptr + bh * BLOCK_D * BLOCK_D + offs_dd, M)
            if not SKIP_Z:
                tl.store(final_state_z_ptr + bh * BLOCK_D + offs_d, z)


    if DIRECTION != 1:
        M = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)
        if not SKIP_Z:
            z = tl.zeros([BLOCK_D], dtype=tl.float32)


        if not COMBINED_HISTORY:
            tl.store(
                M_rev_ptr
                + bh * F * BLOCK_D * BLOCK_D
                + (F - 1) * BLOCK_D * BLOCK_D
                + offs_dd,
                M,
            )
            if not SKIP_Z:
                tl.store(z_rev_ptr + bh * F * BLOCK_D + (F - 1) * BLOCK_D + offs_d, z)
        for f_iter in range(F - 1):
            f_src = F - 1 - f_iter
            f_dst = f_src - 1
            I_P_kv_f = tl.load(
                I_P_kv_ptr
                + bh * F * BLOCK_D * BLOCK_D
                + f_src * BLOCK_D * BLOCK_D
                + offs_dd
            )
            A_f = tl.load(
                A_ptr + bh * F * BLOCK_D * BLOCK_D + f_src * BLOCK_D * BLOCK_D + offs_dd
            )
            g_f = tl.load(decay_ptr + bh * F + f_src).to(tl.float32)

            if USE_ACC_FUSION:
                I_P_scaled = I_P_kv_f.to(tl.float32) * g_f
                M = tl.dot(
                    I_P_scaled.to(dot_dtype),
                    M.to(dot_dtype),
                    acc=A_f.to(tl.float32),
                    out_dtype=tl.float32,
                    input_precision=dot_ip,
                )
            else:
                M_temp = tl.dot(
                    I_P_kv_f.to(dot_dtype),
                    M.to(dot_dtype),
                    out_dtype=tl.float32,
                    input_precision=dot_ip,
                )
                M = g_f * M_temp + A_f

            if not SKIP_Z:
                I_P_z_f = tl.load(
                    I_P_z_ptr
                    + bh * F * BLOCK_D * BLOCK_D
                    + f_src * BLOCK_D * BLOCK_D
                    + offs_dd
                )
                B_f = tl.load(B_ptr + bh * F * BLOCK_D + f_src * BLOCK_D + offs_d)
                z_temp = tl.sum(I_P_z_f * z[None, :], axis=1)
                z = g_f * z_temp + B_f

            if COMBINED_HISTORY:


                M_addr = (
                    M_fwd_ptr
                    + bh * F * BLOCK_D * BLOCK_D
                    + f_dst * BLOCK_D * BLOCK_D
                    + offs_dd
                )
                tl.store(M_addr, tl.load(M_addr) + M)
                if not SKIP_Z:
                    z_addr = z_fwd_ptr + bh * F * BLOCK_D + f_dst * BLOCK_D + offs_d
                    tl.store(z_addr, tl.load(z_addr) + z)
            else:
                tl.store(
                    M_rev_ptr
                    + bh * F * BLOCK_D * BLOCK_D
                    + f_dst * BLOCK_D * BLOCK_D
                    + offs_dd,
                    M,
                )
                if not SKIP_Z:
                    tl.store(z_rev_ptr + bh * F * BLOCK_D + f_dst * BLOCK_D + offs_d, z)


def phase_b_triton(
    I_P_kv,
    A,
    I_P_z,
    B,
    decay,
    F,
    num_warps=None,
    num_stages=None,
    use_acc_fusion=None,
    dot_precision=0,
    init_state_kv=None,
    init_state_z=None,
    return_final_state=False,
    direction=0,
    combined_history=False,
    skip_z=False,
):

    BH = I_P_kv.shape[0]
    _, _, BLOCK_D, _ = A.shape
    device, fdtype = I_P_kv.device, torch.float32

    if num_warps is None or num_stages is None or use_acc_fusion is None:
        _, _, b_w, b_s, b_acc, *_ = _get_arch_config(dot_precision, device=device)
        if num_warps is None:
            num_warps = b_w
        if num_stages is None:
            num_stages = b_s
        if use_acc_fusion is None:
            use_acc_fusion = b_acc

    if combined_history and direction != 0:
        raise ValueError("combined_history=True requires direction=0 (bidi)")


    decay_flat = decay.reshape(BH, F).contiguous().float()

    load_init = init_state_kv is not None
    dummy = torch.empty(1, device=device, dtype=fdtype)

    def full_M():
        return torch.empty(BH, F, BLOCK_D, BLOCK_D, device=device, dtype=fdtype)

    def full_z():
        return torch.empty(BH, F, BLOCK_D, device=device, dtype=fdtype)

    M_fwd = dummy if direction == 2 else full_M()
    z_fwd = dummy if (direction == 2 or skip_z) else full_z()


    M_rev = dummy if (direction == 1 or combined_history) else full_M()
    z_rev = dummy if (direction == 1 or combined_history or skip_z) else full_z()
    if load_init:
        init_kv = init_state_kv.contiguous().view(BH, BLOCK_D, BLOCK_D)
        init_z = dummy if skip_z else init_state_z.contiguous().view(BH, BLOCK_D)
    else:
        init_kv = dummy
        init_z = dummy

    if return_final_state:
        final_kv = torch.empty(BH, BLOCK_D, BLOCK_D, device=device, dtype=fdtype)
        final_z = (
            dummy if skip_z else torch.empty(BH, BLOCK_D, device=device, dtype=fdtype)
        )
    else:
        final_kv = dummy
        final_z = dummy

    d_splits, nw_override, ns_override, acc_override = _pick_phase_b_d_splits(
        BLOCK_D, dot_precision=dot_precision
    )
    if d_splits > 1:
        D_TILE = BLOCK_D // d_splits

        nw_use = nw_override if nw_override is not None else num_warps
        ns_use = ns_override if ns_override is not None else num_stages
        acc_use = acc_override if acc_override is not None else use_acc_fusion
        _phase_b_dtile_kernel[(BH, d_splits)](
            I_P_kv,
            A,
            I_P_z,
            B,
            decay_flat,
            M_fwd,
            z_fwd,
            M_rev,
            z_rev,
            init_kv,
            init_z,
            final_kv,
            final_z,
            BH=BH,
            F=F,
            BLOCK_D=BLOCK_D,
            D_TILE=D_TILE,
            DOT_PRECISION=dot_precision,
            USE_ACC_FUSION=acc_use,
            LOAD_INIT_STATE=1 if load_init else 0,
            SAVE_FINAL_STATE=1 if return_final_state else 0,
            DIRECTION=direction,
            COMBINED_HISTORY=1 if combined_history else 0,
            SKIP_Z=1 if skip_z else 0,
            num_warps=nw_use,
            num_stages=ns_use,
        )
    else:
        _phase_b_kernel[(BH,)](
            I_P_kv,
            A,
            I_P_z,
            B,
            decay_flat,
            M_fwd,
            z_fwd,
            M_rev,
            z_rev,
            init_kv,
            init_z,
            final_kv,
            final_z,
            BH=BH,
            F=F,
            BLOCK_D=BLOCK_D,
            DOT_PRECISION=dot_precision,
            USE_ACC_FUSION=use_acc_fusion,
            LOAD_INIT_STATE=1 if load_init else 0,
            SAVE_FINAL_STATE=1 if return_final_state else 0,
            DIRECTION=direction,
            COMBINED_HISTORY=1 if combined_history else 0,
            SKIP_Z=1 if skip_z else 0,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    if return_final_state:
        return M_fwd, z_fwd, M_rev, z_rev, final_kv, final_z
    return M_fwd, z_fwd, M_rev, z_rev


@triton.jit
def _phase_b_dtile_kernel(
    I_P_kv_ptr,
    A_ptr,
    I_P_z_ptr,
    B_ptr,
    decay_ptr,
    M_fwd_ptr,
    z_fwd_ptr,
    M_rev_ptr,
    z_rev_ptr,
    init_state_kv_ptr,
    init_state_z_ptr,
    final_state_kv_ptr,
    final_state_z_ptr,
    BH: tl.constexpr,
    F: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D_TILE: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    USE_ACC_FUSION: tl.constexpr,
    LOAD_INIT_STATE: tl.constexpr,
    SAVE_FINAL_STATE: tl.constexpr,
    DIRECTION: tl.constexpr,
    COMBINED_HISTORY: tl.constexpr,
    SKIP_Z: tl.constexpr,
):
    if DOT_PRECISION >= 1:
        dot_dtype = tl.float32
    else:
        dot_dtype = tl.bfloat16
    dot_ip: tl.constexpr = "ieee" if DOT_PRECISION == 2 else "tf32"

    pid_bh = tl.program_id(0)
    pid_d = tl.program_id(1)
    bh = pid_bh

    offs_d_full = tl.arange(0, BLOCK_D)
    offs_d_tile = pid_d * D_TILE + tl.arange(0, D_TILE)
    offs_dd_full = offs_d_full[:, None] * BLOCK_D + offs_d_full[None, :]
    offs_dd_tile = offs_d_full[:, None] * BLOCK_D + offs_d_tile[None, :]

    is_lead = pid_d == 0

    if DIRECTION != 2:
        if LOAD_INIT_STATE:
            M = tl.load(init_state_kv_ptr + bh * BLOCK_D * BLOCK_D + offs_dd_tile).to(
                tl.float32
            )
        else:
            M = tl.zeros([BLOCK_D, D_TILE], dtype=tl.float32)
        if not SKIP_Z:
            z = tl.zeros([BLOCK_D], dtype=tl.float32)
            if is_lead and LOAD_INIT_STATE:
                z = tl.load(init_state_z_ptr + bh * BLOCK_D + offs_d_full).to(
                    tl.float32
                )

        for f in range(F):
            I_P_kv_f = tl.load(
                I_P_kv_ptr
                + bh * F * BLOCK_D * BLOCK_D
                + f * BLOCK_D * BLOCK_D
                + offs_dd_full
            )
            A_f = tl.load(
                A_ptr
                + bh * F * BLOCK_D * BLOCK_D
                + f * BLOCK_D * BLOCK_D
                + offs_dd_tile
            )
            g_f = tl.load(decay_ptr + bh * F + f).to(tl.float32)

            if USE_ACC_FUSION:
                I_P_scaled = I_P_kv_f.to(tl.float32) * g_f
                M = tl.dot(
                    I_P_scaled.to(dot_dtype),
                    M.to(dot_dtype),
                    acc=A_f.to(tl.float32),
                    out_dtype=tl.float32,
                    input_precision=dot_ip,
                )
            else:
                M_temp = tl.dot(
                    I_P_kv_f.to(dot_dtype),
                    M.to(dot_dtype),
                    out_dtype=tl.float32,
                    input_precision=dot_ip,
                )
                M = g_f * M_temp + A_f

            tl.store(
                M_fwd_ptr
                + bh * F * BLOCK_D * BLOCK_D
                + f * BLOCK_D * BLOCK_D
                + offs_dd_tile,
                M,
            )

            if is_lead and not SKIP_Z:
                I_P_z_f = tl.load(
                    I_P_z_ptr
                    + bh * F * BLOCK_D * BLOCK_D
                    + f * BLOCK_D * BLOCK_D
                    + offs_dd_full
                )
                B_f = tl.load(B_ptr + bh * F * BLOCK_D + f * BLOCK_D + offs_d_full)
                z_temp = tl.sum(I_P_z_f * z[None, :], axis=1)
                z = g_f * z_temp + B_f
                tl.store(z_fwd_ptr + bh * F * BLOCK_D + f * BLOCK_D + offs_d_full, z)

        if SAVE_FINAL_STATE:
            tl.store(final_state_kv_ptr + bh * BLOCK_D * BLOCK_D + offs_dd_tile, M)
            if is_lead and not SKIP_Z:
                tl.store(final_state_z_ptr + bh * BLOCK_D + offs_d_full, z)

    if DIRECTION != 1:
        M = tl.zeros([BLOCK_D, D_TILE], dtype=tl.float32)
        if not SKIP_Z:
            z = tl.zeros([BLOCK_D], dtype=tl.float32)

        if not COMBINED_HISTORY:
            tl.store(
                M_rev_ptr
                + bh * F * BLOCK_D * BLOCK_D
                + (F - 1) * BLOCK_D * BLOCK_D
                + offs_dd_tile,
                M,
            )
            if is_lead and not SKIP_Z:
                tl.store(
                    z_rev_ptr + bh * F * BLOCK_D + (F - 1) * BLOCK_D + offs_d_full, z
                )

        for f_iter in range(F - 1):
            f_src = F - 1 - f_iter
            f_dst = f_src - 1
            I_P_kv_f = tl.load(
                I_P_kv_ptr
                + bh * F * BLOCK_D * BLOCK_D
                + f_src * BLOCK_D * BLOCK_D
                + offs_dd_full
            )
            A_f = tl.load(
                A_ptr
                + bh * F * BLOCK_D * BLOCK_D
                + f_src * BLOCK_D * BLOCK_D
                + offs_dd_tile
            )
            g_f = tl.load(decay_ptr + bh * F + f_src).to(tl.float32)

            if USE_ACC_FUSION:
                I_P_scaled = I_P_kv_f.to(tl.float32) * g_f
                M = tl.dot(
                    I_P_scaled.to(dot_dtype),
                    M.to(dot_dtype),
                    acc=A_f.to(tl.float32),
                    out_dtype=tl.float32,
                    input_precision=dot_ip,
                )
            else:
                M_temp = tl.dot(
                    I_P_kv_f.to(dot_dtype),
                    M.to(dot_dtype),
                    out_dtype=tl.float32,
                    input_precision=dot_ip,
                )
                M = g_f * M_temp + A_f

            if is_lead and not SKIP_Z:
                I_P_z_f = tl.load(
                    I_P_z_ptr
                    + bh * F * BLOCK_D * BLOCK_D
                    + f_src * BLOCK_D * BLOCK_D
                    + offs_dd_full
                )
                B_f = tl.load(B_ptr + bh * F * BLOCK_D + f_src * BLOCK_D + offs_d_full)
                z_temp = tl.sum(I_P_z_f * z[None, :], axis=1)
                z = g_f * z_temp + B_f

            if COMBINED_HISTORY:
                M_addr = (
                    M_fwd_ptr
                    + bh * F * BLOCK_D * BLOCK_D
                    + f_dst * BLOCK_D * BLOCK_D
                    + offs_dd_tile
                )
                tl.store(M_addr, tl.load(M_addr) + M)
                if is_lead and not SKIP_Z:
                    z_addr = (
                        z_fwd_ptr + bh * F * BLOCK_D + f_dst * BLOCK_D + offs_d_full
                    )
                    tl.store(z_addr, tl.load(z_addr) + z)
            else:
                tl.store(
                    M_rev_ptr
                    + bh * F * BLOCK_D * BLOCK_D
                    + f_dst * BLOCK_D * BLOCK_D
                    + offs_dd_tile,
                    M,
                )
                if is_lead and not SKIP_Z:
                    tl.store(
                        z_rev_ptr + bh * F * BLOCK_D + f_dst * BLOCK_D + offs_d_full, z
                    )


_PHASE_B_DTILE_ARCH_CACHE: dict = {}


def _pick_phase_b_d_splits(BLOCK_D: int, dot_precision: int = 0):

    import os

    env_d = os.environ.get("PHASE_B_D_SPLITS", None)
    if env_d is not None:
        d = int(env_d)
        if d < 1 or BLOCK_D % d != 0:
            return (1, None, None, None)
        nw = int(os.environ.get("PHASE_B_DTILE_NW", "0")) or None
        ns = int(os.environ.get("PHASE_B_DTILE_NS", "0")) or None
        acc_env = os.environ.get("PHASE_B_DTILE_ACC", None)
        acc = bool(int(acc_env)) if acc_env is not None else None
        return (d, nw, ns, acc)
    try:
        import torch

        if not torch.cuda.is_available():
            return (1, None, None, None)
        dev = torch.cuda.current_device()
        cache_key = (dev, dot_precision)
        if cache_key not in _PHASE_B_DTILE_ARCH_CACHE:
            cap = torch.cuda.get_device_capability(dev)
            major, minor = cap[0], cap[1]
            if dot_precision == 2:

                if major == 8 and minor == 0:
                    cfg = (4, 32, 1, True)
                elif major == 9:
                    cfg = (4, 32, 1, True)
                elif major == 8 and minor == 9:
                    cfg = (8, 4, 1, False)
                elif major >= 10:
                    cfg = (8, 4, 1, False)
                else:
                    cfg = (1, None, None, None)
            else:


                if major == 8 and minor == 0:
                    cfg = (4, 8, 2, False)
                elif major == 9:
                    cfg = (4, 8, 2, False)
                elif major == 10:
                    cfg = (4, 8, 2, False)
                elif major == 12 and minor == 0:
                    cfg = (8, 8, 1, False)
                elif major == 12 and minor == 1:
                    cfg = (1, None, None, None)
                else:
                    cfg = (1, None, None, None)
            _PHASE_B_DTILE_ARCH_CACHE[cache_key] = cfg
        return _PHASE_B_DTILE_ARCH_CACHE[cache_key]
    except Exception:
        return (1, None, None, None)


@triton.jit
def _phase_c_kernel(
    qkv_ptr,
    stride_b: tl.constexpr,
    stride_n: tl.constexpr,
    stride_3: tl.constexpr,
    stride_h: tl.constexpr,
    stride_d: tl.constexpr,
    q_inv_rms_ptr,
    q_norm_w_ptr,
    rope_cos_ptr,
    rope_sin_ptr,
    M_ptr,
    z_ptr,
    num_ptr,
    den_ptr,
    H: tl.constexpr,
    F: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    NORM_EPS: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    ACCUMULATE: tl.constexpr = False,
    SKIP_LAST_F: tl.constexpr = False,
    SKIP_RELU: tl.constexpr = False,
    NUM_ONLY: tl.constexpr = False,
):
    if DOT_PRECISION >= 1:
        dot_dtype = tl.float32
    else:
        dot_dtype = tl.bfloat16
    dot_ip: tl.constexpr = "ieee" if DOT_PRECISION == 2 else "tf32"

    pid = tl.program_id(0)
    pid_b = pid // (H * F)
    pid_hf = pid % (H * F)
    pid_h = pid_hf // F
    pid_f = pid_hf % F
    bh = pid_b * H + pid_h
    N: tl.constexpr = F * S


    if SKIP_LAST_F and pid_f == F - 1:
        return

    qkv_bh = qkv_ptr + pid_b * stride_b + pid_h * stride_h
    num_bh = num_ptr + pid_b * (N * H * D) + pid_h * D
    den_bh = den_ptr + bh * N
    M_bhf = M_ptr + bh * F * BLOCK_D * BLOCK_D + pid_f * BLOCK_D * BLOCK_D
    z_bhf = z_ptr + bh * F * BLOCK_D + pid_f * BLOCK_D

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    offs_d_pair = offs_d ^ 1
    mask_d_pair = offs_d_pair < D
    offs_dd = offs_d[:, None] * BLOCK_D + offs_d[None, :]
    mask_dd = mask_d[:, None] & mask_d[None, :]

    nw_offset = pid_h * D
    q_nw = tl.load(q_norm_w_ptr + nw_offset + offs_d, mask=mask_d, other=0.0).to(
        tl.float32
    )
    q_nw_pair = tl.load(
        q_norm_w_ptr + nw_offset + offs_d_pair, mask=mask_d_pair, other=0.0
    ).to(tl.float32)

    M_f = tl.load(M_bhf + offs_dd, mask=mask_dd, other=0.0)
    z_f = tl.load(z_bhf + offs_d, mask=mask_d, other=0.0)

    n_base = pid_f * S
    for s0 in range(0, S, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        mask_s = offs_s < S
        mask_sd = mask_s[:, None] & mask_d[None, :]
        n_idx = n_base + offs_s

        q_ptrs = (
            qkv_bh
            + n_idx[:, None] * stride_n
            + 0 * stride_3
            + offs_d[None, :] * stride_d
        )
        Q_raw = tl.load(q_ptrs, mask=mask_sd, other=0.0).to(tl.float32)
        Q_pair_raw = tl.reshape(
            tl.flip(tl.reshape(Q_raw, (BLOCK_S, BLOCK_D // 2, 2)), dim=2),
            (BLOCK_S, BLOCK_D),
        )

        q_inv_rms = tl.load(
            q_inv_rms_ptr + pid_b * N + n_idx, mask=mask_s, other=1.0
        ).to(tl.float32)
        Q_normed = Q_raw * q_inv_rms[:, None] * q_nw[None, :]
        Q_pair_normed = Q_pair_raw * q_inv_rms[:, None] * q_nw_pair[None, :]
        if SKIP_RELU:
            Q = Q_normed
            Q_pair = Q_pair_normed
        else:
            Q = tl.where(Q_normed > 0, Q_normed, 0.0)
            Q_pair = tl.where(Q_pair_normed > 0, Q_pair_normed, 0.0)

        rope_ptrs = n_idx[:, None] * D + offs_d[None, :]
        Cos = tl.load(rope_cos_ptr + rope_ptrs, mask=mask_sd, other=1.0).to(tl.float32)
        Sin = tl.load(rope_sin_ptr + rope_ptrs, mask=mask_sd, other=0.0).to(tl.float32)
        Q_rot = Q * Cos + Q_pair * Sin

        num = tl.dot(
            Q_rot.to(dot_dtype),
            M_f.to(dot_dtype),
            out_dtype=tl.float32,
            input_precision=dot_ip,
        )
        if not NUM_ONLY:
            den = tl.sum(Q * z_f[None, :], axis=1)

        num_ptrs = num_bh + n_idx[:, None] * (H * D) + offs_d[None, :]
        if not NUM_ONLY:
            den_ptrs = den_bh + n_idx
        if ACCUMULATE:


            prev_num = tl.load(num_ptrs, mask=mask_sd, other=0.0).to(tl.float32)
            num = num + prev_num
            if not NUM_ONLY:
                prev_den = tl.load(den_ptrs, mask=mask_s, other=0.0).to(tl.float32)
                den = den + prev_den
        if DOT_PRECISION >= 1:
            tl.store(num_ptrs, num, mask=mask_sd)
            if not NUM_ONLY:
                tl.store(den_ptrs, den, mask=mask_s)
        else:
            tl.store(num_ptrs, num.to(tl.bfloat16), mask=mask_sd)
            if not NUM_ONLY:
                tl.store(den_ptrs, den.to(tl.bfloat16), mask=mask_s)


def phase_c(
    qkv,
    q_inv_rms,
    q_norm_w,
    rope_cos,
    rope_sin,
    M,
    z,
    F,
    S,
    num_warps=None,
    num_stages=None,
    BLOCK_S=None,
    dot_precision=0,
    num_out=None,
    den_out=None,
    accumulate=False,
    skip_last_frame=False,
    skip_relu: bool = False,
    num_only: bool = False,
):

    if num_warps is None or num_stages is None or BLOCK_S is None:
        *_, c_w, c_bs, c_s = _get_arch_config(dot_precision, device=qkv.device)
        if num_warps is None:
            num_warps = c_w
        if num_stages is None:
            num_stages = c_s
        if BLOCK_S is None:
            BLOCK_S = c_bs
    B, N, three, H, D = qkv.shape
    BLOCK_D = triton.next_power_of_2(D)
    if num_out is None:
        num_out = torch.empty(
            B,
            N,
            H,
            D,
            device=qkv.device,
            dtype=(torch.float32 if dot_precision >= 1 else torch.bfloat16),
        )
    if den_out is None and not num_only:
        den_out = torch.empty(
            B,
            H,
            N,
            device=qkv.device,
            dtype=(torch.float32 if dot_precision >= 1 else torch.bfloat16),
        )
    elif num_only and den_out is None:

        den_out = torch.empty(
            1,
            device=qkv.device,
            dtype=(torch.float32 if dot_precision >= 1 else torch.bfloat16),
        )

    _phase_c_kernel[(B * H * F,)](
        qkv,
        qkv.stride(0),
        qkv.stride(1),
        qkv.stride(2),
        qkv.stride(3),
        qkv.stride(4),
        q_inv_rms,
        q_norm_w,
        rope_cos,
        rope_sin,
        M,
        z,
        num_out,
        den_out,
        H=H,
        F=F,
        S=S,
        D=D,
        NORM_EPS=1e-5,
        DOT_PRECISION=dot_precision,
        BLOCK_D=BLOCK_D,
        BLOCK_S=BLOCK_S,
        ACCUMULATE=1 if accumulate else 0,
        SKIP_LAST_F=skip_last_frame,
        SKIP_RELU=skip_relu,
        NUM_ONLY=num_only,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return num_out, den_out


def fused_bigdn_bidi_chunkwise(
    qkv,
    q_inv_rms,
    k_inv_rms,
    q_norm_w,
    k_norm_w,
    rope_cos,
    rope_sin,
    beta,
    decay,
    F,
    S,
    k_scale=1.0,
    eps=1e-6,
    norm_eps=1e-5,
    dot_precision=0,
    init_state_kv=None,
    init_state_z=None,
    return_final_state=False,
):

    I_P_kv, A, I_P_z, B_z = phase_a(
        qkv,
        beta,
        q_inv_rms,
        k_inv_rms,
        q_norm_w,
        k_norm_w,
        rope_cos,
        rope_sin,
        F=F,
        S=S,
        k_scale=k_scale,
        norm_eps=norm_eps,
        dot_precision=dot_precision,
    )

    if return_final_state:
        M_hist, z_hist, _, _, final_kv, final_z = phase_b_triton(
            I_P_kv,
            A,
            I_P_z,
            B_z,
            decay,
            F=F,
            dot_precision=dot_precision,
            direction=0,
            init_state_kv=init_state_kv,
            init_state_z=init_state_z,
            return_final_state=True,
            combined_history=True,
        )
    else:
        M_hist, z_hist, _, _ = phase_b_triton(
            I_P_kv,
            A,
            I_P_z,
            B_z,
            decay,
            F=F,
            dot_precision=dot_precision,
            direction=0,
            init_state_kv=init_state_kv,
            init_state_z=init_state_z,
            combined_history=True,
        )
    num_out, den_out = phase_c(
        qkv,
        q_inv_rms,
        q_norm_w,
        rope_cos,
        rope_sin,
        M_hist,
        z_hist,
        F=F,
        S=S,
        dot_precision=dot_precision,
        accumulate=False,
    )
    del M_hist, z_hist, I_P_kv, A, I_P_z, B_z

    total_den = den_out.float().permute(0, 2, 1).unsqueeze(-1)
    out = (num_out.float() / (total_den + eps)).to(qkv.dtype)
    del num_out, den_out, total_den
    if return_final_state:
        B = qkv.shape[0]
        H = qkv.shape[3]
        D = qkv.shape[4]
        BLOCK_D = final_kv.shape[1]
        state_kv = (
            final_kv.view(B, H, BLOCK_D, BLOCK_D)[:, :, :D, :D]
            .transpose(-1, -2)
            .contiguous()
        )
        state_z = final_z.view(B, H, BLOCK_D)[:, :, :D].unsqueeze(-1).contiguous()
        return out, state_kv, state_z
    return out


def _default_dot_prec() -> int:
    try:
        from sglang.kernels.ops.diffusion.attention.sana_wm_gdn_triton import (
            _resolve_launch_config,
        )

        _, dot_prec, _, _ = _resolve_launch_config()
        return dot_prec
    except Exception:
        return 0


def _cam_identity_tables(
    *,
    B: int,
    N: int,
    H: int,
    D: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device_index = device.index if device.type == "cuda" else None
    key = (device.type, device_index, B, N, H * D, D)
    cached = _CAM_IDENTITY_CACHE.get(key)
    if cached is not None:
        return cached

    ones_inv_rms = torch.ones(B, N, device=device, dtype=torch.float32)
    ones_nw = torch.ones(H * D, device=device, dtype=torch.float32)
    ones_cos = torch.ones(N, D, device=device, dtype=torch.float32)
    zeros_sin = torch.zeros(N, D, device=device, dtype=torch.float32)
    cached = (ones_inv_rms, ones_nw, ones_cos, zeros_sin)
    _CAM_IDENTITY_CACHE[key] = cached
    return cached


def cam_scan_bidi_chunkwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    *,
    dot_precision: int | None = None,
) -> torch.Tensor:

    assert (
        q.shape == k.shape == v.shape
    ), f"q/k/v shape mismatch: {q.shape} {k.shape} {v.shape}"
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert beta.is_contiguous() and decay.is_contiguous()
    assert (
        q.dtype == torch.float32
    ), f"cam_scan_bidi_chunkwise requires fp32 q/k/v, got {q.dtype}"

    B, H, D, N = q.shape
    F = beta.shape[2]
    assert N % F == 0
    S = N // F
    assert beta.shape == (B, H, F, S)
    assert decay.shape == (B, H, F)

    if dot_precision is None:
        dot_precision = _default_dot_prec()

    qkv = torch.empty(B, N, 3, H, D, device=q.device, dtype=q.dtype)
    qkv[:, :, 0].copy_(q.permute(0, 3, 1, 2))
    qkv[:, :, 1].copy_(k.permute(0, 3, 1, 2))
    qkv[:, :, 2].copy_(v.permute(0, 3, 1, 2))

    ones_inv_rms, ones_nw, ones_cos, zeros_sin = _cam_identity_tables(
        B=B, N=N, H=H, D=D, device=q.device
    )
    I_P_kv, A, I_P_z, B_z = phase_a(
        qkv,
        beta,
        ones_inv_rms,
        ones_inv_rms,
        ones_nw,
        ones_nw,
        ones_cos,
        zeros_sin,
        F=F,
        S=S,
        k_scale=1.0,
        norm_eps=1e-5,
        dot_precision=dot_precision,
        skip_relu=True,
        skip_z=True,
    )
    M_hist, z_hist, _, _ = phase_b_triton(
        I_P_kv,
        A,
        I_P_z,
        B_z,
        decay,
        F=F,
        dot_precision=dot_precision,
        direction=0,
        combined_history=True,
        skip_z=True,
    )
    num_out, _ = phase_c(
        qkv,
        ones_inv_rms,
        ones_nw,
        ones_cos,
        zeros_sin,
        M_hist,
        z_hist,
        F=F,
        S=S,
        dot_precision=dot_precision,
        skip_relu=True,
        num_only=True,
    )
    return num_out.permute(0, 2, 3, 1).contiguous().to(torch.float32)
