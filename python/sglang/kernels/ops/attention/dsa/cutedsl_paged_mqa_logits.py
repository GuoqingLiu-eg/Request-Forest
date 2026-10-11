# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import logging

import cutlass
import cutlass.cute as cute
import torch
from cutlass.utils import HardwareInfo

from sglang.kernels.ops.attention.cutedsl_fp8_paged_mqa_logits import FP8MQALogitsKernel
from sglang.srt.utils import is_sm100_supported

logger = logging.getLogger(__name__)


def pick_dsl_expand(
    next_n: int,
    batch_size: int = 0,
    max_ctx: int = 0,
    num_sms: int = 148,
    kernel_atoms: tuple[int, ...] = (1, 2, 3, 4),
    num_heads: int = 0,
) -> tuple[int, int]:

    if batch_size <= 0 or max_ctx <= 0:
        for eff in sorted(kernel_atoms, reverse=True):
            if next_n % eff == 0:
                return next_n // eff, eff
        return next_n, 1


    if next_n == 6:


        if 0 < num_heads * 6 <= 256:
            native_wins = (
                batch_size >= 16
                or (batch_size >= 4 and max_ctx >= 32768)
                or (batch_size >= 2 and max_ctx >= 131072)
            )
            if native_wins:
                return 1, 6
        use_atom2 = (
            (batch_size >= 45 and max_ctx <= (batch_size - 44) * 32768)
            or (batch_size == 16 and max_ctx >= 49152)
            or (batch_size == 10 and max_ctx >= 90000)
            or (batch_size == 17 and 49152 <= max_ctx <= 110000)
            or (batch_size == 7 and max_ctx >= 120000)
        )
        if use_atom2 and 2 in kernel_atoms:
            return 3, 2

    SPLIT_KV_TOKENS = 256
    cands = []
    for eff in kernel_atoms:
        if next_n % eff == 0:
            factor = next_n // eff
            ntask = (
                batch_size
                * factor
                * ((max_ctx + SPLIT_KV_TOKENS - 1) // SPLIT_KV_TOKENS)
            )
            waves = (ntask + num_sms - 1) // num_sms
            cands.append((waves, factor, eff))
    if not cands:
        return next_n, 1
    cands.sort(key=lambda x: (x[0], -x[1]))
    _, factor, eff = cands[0]
    return factor, eff


_TORCH_TO_CUTLASS_DTYPE = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


_EPI_KV_UMMA_SUB = dict(
    max_kv_pipeline=True, max_umma_pipeline=True, smem_subpartition_opt=True
)
_EPI_NOWAIT = dict(_EPI_KV_UMMA_SUB, remove_kv_wait_in_epilogue=True)


def _propose_epi_config(
    num_heads: int,
    next_n_k: int,
    batch_split: int,
    max_ctx: int,
    num_sms: int,
) -> tuple[int, dict]:


    if num_heads > 32 or num_heads % 8 != 0:
        return 1, {}


    if next_n_k == 6 and num_heads * 6 <= 256:
        return 1, {"max_w_in_reg": 8}


    nst = 2 if next_n_k != 2 else 1
    if (num_heads // nst) % 4 != 0:
        nst = 1
    waves = (batch_split * ((max_ctx + 255) // 256) + num_sms - 1) // num_sms


    if next_n_k == 3:
        resonant = batch_split % 24 == 0
    elif next_n_k == 2:
        resonant = batch_split % 144 == 0
    else:
        resonant = False
    if waves >= 130 and not resonant:
        return 1, {}


    flags = _EPI_NOWAIT if (next_n_k >= 3 or waves >= 50) else {}
    return nst, flags


class CuteDSLPagedMQALogitsRunner:


    kernel_cache: dict[tuple, object] = dict()

    @classmethod
    def _compile(
        cls,
        compute_block_kv,
        phys_block_kv,
        num_heads,
        head_dim,
        next_n,
        num_sms,
        num_epi_subtiles,
        epi_dtype,
        acc_dtype,
        output_dtype,
        pipeline_flags=None,
    ):

        pipeline_flags = pipeline_flags or {}
        key = (
            compute_block_kv,
            phys_block_kv,
            num_heads,
            head_dim,
            next_n,
            num_sms,
            num_epi_subtiles,
            epi_dtype,
            acc_dtype,
            output_dtype,
            tuple(sorted(pipeline_flags.items())),
        )
        if key in cls.kernel_cache:
            return

        to_cutlass = _TORCH_TO_CUTLASS_DTYPE
        N = next_n * num_heads
        block_bytes = phys_block_kv * (head_dim + 4)

        sym_num_phys_blocks = cute.sym_int()
        sym_B = cute.sym_int()
        max_ctx = cute.sym_int()
        max_blocks_per_seq = cute.sym_int()
        num_ctas = cute.sym_int()

        kv_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Uint8,
            (sym_num_phys_blocks, block_bytes),
            stride_order=(1, 0),
        )

        q_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Uint8, (N, head_dim, sym_B), stride_order=(1, 0, 2)
        )

        w_dtype = (
            cutlass.Float16 if epi_dtype == torch.float16 else to_cutlass[epi_dtype]
        )
        w_fake = cute.runtime.make_fake_compact_tensor(
            w_dtype, (N, sym_B), stride_order=(0, 1)
        )

        logits_fake = cute.runtime.make_fake_tensor(
            to_cutlass[output_dtype],
            (cute.sym_int(), max_ctx),
            stride=(cute.sym_int64(), 1),
        )

        bt_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32, (sym_B, max_blocks_per_seq), stride_order=(1, 0)
        )

        cl_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32, (sym_B,), stride_order=(0,)
        )

        sm_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32, (num_ctas, 2), stride_order=(1, 0)
        )

        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

        kernel = FP8MQALogitsKernel(
            block_kv=compute_block_kv,
            phys_block_kv=phys_block_kv,
            num_heads=num_heads,
            head_dim=head_dim,
            next_n=next_n,
            num_sms=num_sms,
            num_epi_subtiles=num_epi_subtiles,
            epi_dtype=to_cutlass[epi_dtype],
            acc_dtype=to_cutlass[acc_dtype],
            output_dtype=to_cutlass[output_dtype],
            **pipeline_flags,
        )

        compiled = cute.compile(
            kernel,
            kv_fake,
            q_fake,
            w_fake,
            logits_fake,
            bt_fake,
            cl_fake,
            sm_fake,
            cutlass.Int32(1),
            cutlass.Int32(1),
            fake_stream,
            options="--enable-tvm-ffi",
        )
        cls.kernel_cache[key] = compiled
        logger.debug(
            f"[compile cute_dsl fp8_paged_mqa_logits] {key}"
            f" kv_stages={kernel.num_kv_stages}"
            f" umma_stages={kernel.num_umma_stages}"
        )

    @classmethod
    def forward(
        cls,
        q: torch.Tensor,
        kv_fused: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor,
        max_context_len: int,
        num_epi_subtiles: int | None = None,
        epi_dtype: torch.dtype = torch.float32,
        acc_dtype: torch.dtype = torch.float32,
        output_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:

        B, next_n, H, D = q.shape
        N = next_n * H
        phys_block_kv = kv_fused.shape[1]
        compute_block_kv = 128
        num_phys_blocks = kv_fused.shape[0]
        num_sms = HardwareInfo().get_device_multiprocessor_count()


        auto_nst, pipeline_flags = _propose_epi_config(
            H, next_n, B, max_context_len, num_sms
        )
        if num_epi_subtiles is None:
            num_epi_subtiles = auto_nst
        else:
            pipeline_flags = {}


        q_3d = q.reshape(B, N, D).permute(1, 2, 0)


        if epi_dtype == torch.float16:

            w_2d = weights.reshape(B, N).half().t()
        else:
            w_2d = weights.reshape(B, N).t()


        kv_flat = kv_fused.reshape(num_phys_blocks, -1)


        SPLIT_KV = compute_block_kv * 2
        aligned_max_ctx = ((max_context_len + SPLIT_KV - 1) // SPLIT_KV) * SPLIT_KV
        logits = torch.empty(
            (B * next_n, aligned_max_ctx),
            device=q.device,
            dtype=output_dtype,
        )
        logits = logits[:, :max_context_len]

        key = (
            compute_block_kv,
            phys_block_kv,
            H,
            D,
            next_n,
            num_sms,
            num_epi_subtiles,
            epi_dtype,
            acc_dtype,
            output_dtype,
            tuple(sorted(pipeline_flags.items())),
        )
        if key not in cls.kernel_cache:
            cls._compile(
                compute_block_kv,
                phys_block_kv,
                H,
                D,
                next_n,
                num_sms,
                num_epi_subtiles,
                epi_dtype,
                acc_dtype,
                output_dtype,
                pipeline_flags,
            )
        compiled = cls.kernel_cache[key]


        q_for_ffi = (
            q_3d.view(torch.uint8)
            if q_3d.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            else q_3d
        )

        compiled(
            kv_flat,
            q_for_ffi,
            w_2d,
            logits,
            block_table,
            context_lens,
            schedule_meta,
            num_phys_blocks,
            B,
        )
        return logits


@torch.library.custom_op(
    "sglang::cute_dsl_fp8_paged_mqa_logits",
    mutates_args=(),
    device_types="cuda",
)
def cute_dsl_fp8_paged_mqa_logits(
    q: torch.Tensor,
    kv_fused: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_meta: torch.Tensor,
    max_context_len: int,
    num_epi_subtiles: int | None = None,
    epi_dtype: torch.dtype = torch.float32,
    acc_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if not is_sm100_supported():
        raise ValueError("CuteDSL FP8 Paged MQA Logits only supports SM 100 family.")
    return CuteDSLPagedMQALogitsRunner.forward(
        q,
        kv_fused,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_context_len,
        num_epi_subtiles=num_epi_subtiles,
        epi_dtype=epi_dtype,
        acc_dtype=acc_dtype,
        output_dtype=output_dtype,
    )


@torch.library.register_fake("sglang::cute_dsl_fp8_paged_mqa_logits")
def _(
    q: torch.Tensor,
    kv_fused: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_meta: torch.Tensor,
    max_context_len: int,
    num_epi_subtiles: int | None = None,
    epi_dtype: torch.dtype = torch.float32,
    acc_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    B = q.shape[0]
    next_n = q.shape[1]
    return torch.empty(
        B * next_n,
        max_context_len,
        dtype=output_dtype,
        device=q.device,
    )
