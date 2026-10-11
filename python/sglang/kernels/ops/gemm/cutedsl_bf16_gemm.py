# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026 by FlashInfer team.
# Copyright 2026 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0


from __future__ import annotations

from typing import List, NamedTuple, Optional, Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import from_dlpack, make_fake_stream

from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.srt.utils import is_sm100_supported
from sglang.srt.utils.common import direct_register_custom_op


_TGV_CUTE_EXT_CTA_K: int = 128
_TGV_CUTE_EXT_DEFAULT_TACTIC: int = 1
_TGV_CUTE_EXT_TACTIC_CONFIGS: List[Tuple[int, int, int, bool]] = [

    (64, 8, 6, False),
    (64, 8, 8, False),
    (64, 8, 10, False),
    (64, 8, 12, False),
    (64, 16, 6, False),
    (64, 16, 8, False),
    (64, 16, 11, False),
    (64, 32, 6, False),
    (64, 32, 9, False),
    (64, 64, 7, False),
    (64, 128, 4, False),
    (128, 8, 6, False),
    (128, 16, 6, False),
    (128, 32, 5, False),
    (128, 64, 4, False),
    (128, 128, 3, False),

    (64, 16, 6, True),
    (64, 16, 8, True),
    (64, 16, 12, True),
    (64, 32, 6, True),
    (64, 32, 8, True),
    (64, 32, 11, True),
    (64, 64, 6, True),
    (64, 64, 9, True),
    (64, 128, 7, True),
    (128, 16, 6, True),
    (128, 32, 6, True),
    (128, 64, 5, True),
    (128, 128, 4, True),
]


def get_tgv_cute_ext_tactic_num() -> int:
    return len(_TGV_CUTE_EXT_TACTIC_CONFIGS)


def get_tgv_cute_ext_default_tactic() -> int:
    return _TGV_CUTE_EXT_DEFAULT_TACTIC


class WorkTileInfo(NamedTuple):


    M_idx: cutlass.Int32
    N_idx: cutlass.Int32
    L_idx: cutlass.Int32
    K_idx_start: cutlass.Int32
    K_idx_end: cutlass.Int32


class TgvGemmCuteExtKernel:


    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        cta_m: int = 64,
        cta_n: int = 8,
        cta_k: int = _TGV_CUTE_EXT_CTA_K,
        num_ab_stage: int = 8,
        use_2cta: bool = False,
        use_pdl: bool = False,
        pdl_launch: Optional[bool] = None,
        pdl_count: int = -1,
        has_bias: bool = False,
        out_dtype: Type[cutlass.Numeric] = cutlass.BFloat16,
    ):
        self.acc_dtype = acc_dtype
        self.out_dtype = out_dtype
        self.cta_m = cta_m
        self.cta_n = cta_n
        self.cta_k = cta_k
        self.num_ab_stage = num_ab_stage
        self.use_2cta = use_2cta
        self.use_pdl = use_pdl
        self.pdl_launch = pdl_launch if pdl_launch is not None else use_pdl
        self.pdl_count = pdl_count


        self.has_bias = has_bias


        min_n, step_n = (16, 16) if use_2cta else (8, 8)
        if cta_n < min_n or cta_n > 256 or cta_n % step_n != 0:
            raise ValueError(
                f"cta_n={cta_n} invalid for use_2cta={use_2cta}: "
                f"bf16 K-major mma requires N ∈ [{min_n}, 256] step {step_n}"
            )


        self.threads_per_cta = 256
        if use_2cta:

            self.cluster_shape = (2, 1, 1)
            self.mma_tiler_mn = (cta_m * 2, cta_n)
            self.cta_group = tcgen05.CtaGroup.TWO
            self.tma_op = cute_ext.OperationTypeEnum.SM100_TMA_LOAD_2SM
        else:

            self.cluster_shape = (1, 1, 1)
            self.mma_tiler_mn = (cta_m, cta_n)
            self.cta_group = tcgen05.CtaGroup.ONE
            self.tma_op = cute_ext.OperationTypeEnum.SM90_TMA_LOAD

    def __repr__(self) -> str:
        return (
            f"TgvGemmCuteExtKernel_cta{self.cta_m}x{self.cta_n}x{self.cta_k}"
            f"_2cta{int(self.use_2cta)}_pdl{int(self.use_pdl)}"
            f"_bias{int(self.has_bias)}"
            f"_out{self.out_dtype.__name__.lower()}"
        )

    @cute.experimental.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        bias: cute.Tensor,
        stream: cuda.CUstream,
    ):


        grid = cute.round_up(
            (
                cute.ceil_div(c.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n),
                c.layout.shape[2],
            ),
            self.cluster_shape,
        )
        self.kernel(a, b, c, bias).launch(
            grid=grid,
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(utils.get_smem_capacity_in_bytes("sm_100")),
            stream=stream,
            use_pdl=self.pdl_launch,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        mBias: cute.Tensor,
    ):

        DMA_Stage = self.num_ab_stage


        a_major = utils.LayoutEnum.from_tensor(mA).mma_major_mode()
        b_major = utils.LayoutEnum.from_tensor(mB).mma_major_mode()
        ab_dtype = mA.element_type
        c_dtype = mC.element_type
        d_layout = utils.LayoutEnum.from_tensor(mC)

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype,
            ab_dtype,
            a_major,
            b_major,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_mn,
        )
        num_mma_ctas = cute.size(tiled_mma.thr_id.shape)


        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = self.cta_k // mma_inst_shape_k

        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        a_tiler_mk = (self.cta_m, self.cta_k)

        b_tiler_nk = (self.cta_n // num_mma_ctas, self.cta_k)
        c_tiler_mn = (self.cta_m, self.cta_n)


        bidx, bidy, bidz = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()


        leader_rank = cutlass.Int32(0)
        if cutlass.const_expr(self.use_2cta):
            cta_rank_in_cluster = cute.arch.make_warp_uniform(
                cute.arch.block_idx_in_cluster()
            )
            is_leader = cta_rank_in_cluster == 0
        else:
            cta_rank_in_cluster = leader_rank
            is_leader = cutlass.Boolean(True)


        a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mnk_tiler,
            ab_dtype,
            DMA_Stage,
        )
        b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mnk_tiler,
            ab_dtype,
            DMA_Stage,
        )


        sA = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            a_smem_layout_staged,
            alignment=1024,
        )
        sB = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            b_smem_layout_staged,
            alignment=1024,
        )


        acc_layout = cute_ext.make_tmem_layout_acc(
            tiled_mma,
            self.mma_tiler_mn,
            acc_stage=1,
        )


        bar_full_arr = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(DMA_Stage),
            alignment=8,
        )
        bar_empty_arr = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(DMA_Stage),
            alignment=8,
        )
        bar_tma_epilog_arr = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=8
        )
        bar_mma_epilog_arr = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=8
        )
        bar_tmem_alloc_arr = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=8
        )


        tmem_base_arr = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        )

        bar_full = bar_full_arr.iterator
        bar_empty = bar_empty_arr.iterator
        bar_tma_epilog = bar_tma_epilog_arr.iterator
        bar_mma_epilog = bar_mma_epilog_arr.iterator
        bar_tmem_alloc = bar_tmem_alloc_arr.iterator
        tmem_base_ptr = tmem_base_arr.iterator


        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(DMA_Stage):
                    cute.arch.mbarrier_init(bar_full + i, 2 * num_mma_ctas)
                for i in range(DMA_Stage):
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                cute.arch.mbarrier_init(bar_tma_epilog, 32)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(
                    bar_tmem_alloc, 32 + 128
                )

        cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(self.use_2cta):


            cute.arch.cluster_arrive_relaxed()
        else:
            cute.arch.barrier()

        work_tile_info = WorkTileInfo(
            M_idx=bidx,
            N_idx=bidy,
            L_idx=bidz,
            K_idx_start=cutlass.Int32(0),
            K_idx_end=cute.ceil_div(cute.size(mA, mode=[1]), self.cta_k),
        )
        k_tile_count = work_tile_info.K_idx_end - work_tile_info.K_idx_start


        a_cta_v_map = cute_ext.get_cta_v_map_ab(mA, mnk_tiler, tiled_mma, "A")
        b_cta_v_map = cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B")


        gA_tile = cute.local_tile(
            mA,
            a_tiler_mk,
            (work_tile_info.M_idx, None, work_tile_info.L_idx),
        )


        if cutlass.const_expr(self.use_2cta):
            gB_n_idx = work_tile_info.N_idx * num_mma_ctas + cta_rank_in_cluster
        else:
            gB_n_idx = work_tile_info.N_idx
        gB_tile = cute.local_tile(
            mB,
            b_tiler_nk,
            (gB_n_idx, None, work_tile_info.L_idx),
        )
        gD_tile = cute.local_tile(
            mC,
            c_tiler_mn,
            (work_tile_info.M_idx, work_tile_info.N_idx, work_tile_info.L_idx),
        )


        gBias_tile = cute.local_tile(
            mBias,
            c_tiler_mn,
            (work_tile_info.M_idx, work_tile_info.N_idx, work_tile_info.L_idx),
        )

        if cutlass.const_expr(self.use_2cta):


            cute.arch.cluster_wait()


        if warp_idx == 0:
            self.dma_a_warp(
                bar_full,
                bar_empty,
                leader_rank,
                gA_tile,
                sA,
                a_cta_v_map,
                k_tile_count,
            )
        elif warp_idx == 1:
            self.dma_b_warp(
                bar_full,
                bar_empty,
                bar_tma_epilog,
                leader_rank,
                gB_tile,
                sB,
                b_cta_v_map,
                k_tile_count,
            )
        elif warp_idx == 2:
            self.mma_warp(
                is_leader,
                bar_full,
                bar_empty,
                bar_mma_epilog,
                bar_tmem_alloc,
                tiled_mma,
                sA,
                sB,
                tmem_base_ptr,
                acc_layout,
                mma_inst_tile_k,
                k_tile_count,
            )
        elif warp_idx >= 4:

            epi_tid = tidx - 128
            self.epilog_warp(
                bar_tma_epilog,
                bar_mma_epilog,
                bar_tmem_alloc,
                tmem_base_ptr,
                acc_layout,
                gD_tile,
                gBias_tile,
                epi_tid,
                c_dtype,
                d_layout,
            )

        if cutlass.const_expr(self.use_2cta):


            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()


    @cute.experimental.jit
    def dma_a_warp(
        self,
        bar_full,
        bar_empty,
        leader_rank: cutlass.Int32,
        gA_tile: cute.Tensor,
        sA: cute.Tensor,
        a_cta_v_map: cute.Layout,
        k_tile_count: cutlass.Int32,
    ):
        DMA_Stage = self.num_ab_stage


        empty_phase = cutlass.Int32(1)
        pdl_count = self.pdl_count

        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % DMA_Stage
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)


            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    bar_full + stage,

                    cute.size_in_bytes(
                        sA.element_type,
                        cute.slice_(sA.layout, (None, None, None, 0)),
                    ),
                    peer_cta_rank_in_cluster=leader_rank if self.use_2cta else None,
                )
            cute_ext.tma_load(
                gA_tile[None, None, k_tile],
                sA[None, None, None, stage],
                (bar_full + stage).value,
                cta_v_map=a_cta_v_map,
                tma_operation_type=self.tma_op,
                update_expect_tx=False,
            )

            if stage == (DMA_Stage - 1):
                empty_phase = empty_phase ^ 1


            if cutlass.const_expr(self.use_pdl):
                if k_tile == pdl_count:
                    cute.arch.griddepcontrol_launch_dependents()

        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()


        for k_tile in cutlass.range(DMA_Stage, unroll=1):
            stage = (k_tile + k_tile_count) % DMA_Stage
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            if stage == (DMA_Stage - 1):
                empty_phase = empty_phase ^ 1


    @cute.experimental.jit
    def dma_b_warp(
        self,
        bar_full,
        bar_empty,
        bar_tma_epilog,
        leader_rank: cutlass.Int32,
        gB_tile: cute.Tensor,
        sB: cute.Tensor,
        b_cta_v_map: cute.Layout,
        k_tile_count: cutlass.Int32,
    ):
        DMA_Stage = self.num_ab_stage


        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()


        empty_phase = cutlass.Int32(1)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % DMA_Stage
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)

            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    bar_full + stage,

                    cute.size_in_bytes(
                        sB.element_type,
                        cute.slice_(sB.layout, (None, None, None, 0)),
                    ),
                    peer_cta_rank_in_cluster=leader_rank if self.use_2cta else None,
                )
            cute_ext.tma_load(
                gB_tile[None, None, k_tile],
                sB[None, None, None, stage],
                (bar_full + stage).value,
                cta_v_map=b_cta_v_map,
                tma_operation_type=self.tma_op,
                update_expect_tx=False,
            )

            if stage == (DMA_Stage - 1):
                empty_phase = empty_phase ^ 1


        if cutlass.const_expr(self.has_bias):
            cute.arch.mbarrier_arrive(bar_tma_epilog)


        for k_tile in cutlass.range(DMA_Stage, unroll=1):
            stage = (k_tile + k_tile_count) % DMA_Stage
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            if stage == (DMA_Stage - 1):
                empty_phase = empty_phase ^ 1


    @cute.experimental.jit
    def mma_warp(
        self,
        is_leader: cutlass.Boolean,
        bar_full,
        bar_empty,
        bar_mma_epilog,
        bar_tmem_alloc,
        tiled_mma: cute.TiledMma,
        sA: cute.Tensor,
        sB: cute.Tensor,
        tmem_base_ptr,
        acc_layout: cutlass.Constexpr,
        mma_inst_tile_k: cutlass.Constexpr,
        k_tile_count: cutlass.Int32,
    ):
        DMA_Stage = self.num_ab_stage


        num_tmem_cols = 256
        cute.arch.alloc_tmem(num_tmem_cols, tmem_base_ptr, is_two_cta=self.use_2cta)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.relinquish_tmem_alloc_permit(is_two_cta=self.use_2cta)


        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        tAcc = cute.make_tensor(
            tmem_ptr, acc_layout
        )
        acc_view = tAcc[
            None, None, None, 0
        ]

        if is_leader:


            mma_atom = cute.make_mma_atom(tiled_mma.op)

            commit_mask = 0b11 if self.use_2cta else None

            full_phase = cutlass.Int32(0)
            for k_tile in cutlass.range(k_tile_count, unroll=1):
                stage = k_tile % DMA_Stage
                cute.arch.mbarrier_wait(bar_full + stage, full_phase)


                for k_block in range(mma_inst_tile_k):
                    if k_block == 0:

                        mma_atom.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                    else:
                        mma_atom.set(tcgen05.Field.ACCUMULATE, True)
                    a_frag = cute.append_ones(
                        sA[None, None, k_block, stage],
                        up_to_rank=3,
                    )
                    b_frag = cute.append_ones(
                        sB[None, None, k_block, stage],
                        up_to_rank=3,
                    )
                    cute_ext.dot(mma_atom, a_frag, b_frag, acc_view)


                with cute.arch.elect_one():
                    tcgen05.commit(bar_empty + stage, commit_mask, self.cta_group)

                if (k_tile % DMA_Stage) == (DMA_Stage - 1):
                    full_phase = full_phase ^ 1


            with cute.arch.elect_one():
                tcgen05.commit(bar_mma_epilog, commit_mask, self.cta_group)


        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 1)
        cute.arch.dealloc_tmem(tmem_ptr, num_tmem_cols, is_two_cta=self.use_2cta)


    @cute.experimental.jit
    def epilog_warp(
        self,
        bar_tma_epilog,
        bar_mma_epilog,
        bar_tmem_alloc,
        tmem_base_ptr,
        acc_layout: cutlass.Constexpr,
        gD_tile: cute.Tensor,
        gBias_tile: cute.Tensor,
        epi_tid: cutlass.Int32,
        c_dtype: cutlass.Constexpr,
        d_layout: cutlass.Constexpr,
    ):


        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)


        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        tCtAcc = cute.make_tensor(
            tmem_ptr, acc_layout
        )
        acc_view = tCtAcc[((None, None), 0, 0, 0)]


        epi_tile = (self.cta_m, self.cta_n)
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            (self.cta_m, self.cta_n, self.cta_k),
            d_layout,
            c_dtype,
            self.acc_dtype,
            epi_tile,
            self.use_2cta,
        )
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(copy_atom_t2r, acc_view)


        gD_epi = cute.flat_divide(gD_tile, epi_tile)


        rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, gD_epi, epi_tid)
        rAcc = cute_ext.allocate(
            self.acc_dtype,
            cute.AddressSpace.rmem,
            rmem_layout,
            alignment=32,
        )
        rD = cute_ext.allocate(
            c_dtype,
            cute.AddressSpace.rmem,
            rmem_layout,
            alignment=32,
        )
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)


        if cutlass.const_expr(self.has_bias):
            bias_dtype = gBias_tile.element_type
            gBias_epi = cute.flat_divide(gBias_tile, epi_tile)
            tiled_copy_g2r = cute.make_tiled_copy_D(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), bias_dtype),
                tiled_copy_t2r,
            )
            thr_g2r = tiled_copy_g2r.get_slice(epi_tid)
            rBias = cute_ext.allocate(
                bias_dtype,
                cute.AddressSpace.rmem,
                rmem_layout,
                alignment=32,
            )
            rBiasAcc = cute_ext.allocate(
                self.acc_dtype,
                cute.AddressSpace.rmem,
                rmem_layout,
                alignment=32,
            )


        if cutlass.const_expr(self.has_bias):
            cute.arch.mbarrier_wait(bar_tma_epilog, 0)


        if cutlass.const_expr(self.has_bias):
            cute_ext.partition_and_copy(thr_g2r, gBias_epi[None, None, 0, 0], rBias)
            rBiasAcc.store(rBias.load().to(self.acc_dtype))

        cute.arch.mbarrier_wait(bar_mma_epilog, 0)


        cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)


        cute.arch.fence_view_async_tmem_load()


        cute.arch.mbarrier_arrive(bar_tmem_alloc)


        if cutlass.const_expr(self.has_bias):
            rAcc.store(rAcc.load() + rBiasAcc.load())


        rD.store(rAcc.load().to(c_dtype))


        cute_ext.partition_and_copy(thr_t2r, rD, gD_epi[None, None, 0, 0])


@cute.experimental.jit
def _bmm_no_bias(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    stream: cuda.CUstream,
):
    a = cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0]))
    b = cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0]))
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))

    bias = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[0, 1, 2]))
    gemm_op(a, b, c, bias, stream)


@cute.experimental.jit
def _bmm_bias(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    bias: cute.Tensor,
    stream: cuda.CUstream,
):
    a = cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0]))
    b = cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0]))
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))

    bias = cute.make_tensor(bias.iterator, cute.select(bias.layout, mode=[1, 2, 0]))
    gemm_op(a, b, c, bias, stream)


_TORCH_TO_CUTLASS_DTYPE = {
    torch.bfloat16: cutlass.BFloat16,
}

_TORCH_TO_CUTLASS_OUT_DTYPE = {
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


_TGV_CUTE_EXT_COMPILE_CACHE: dict = {}


def _detect_leading_dim(t: torch.Tensor) -> int:

    for i, s in enumerate(t.stride()):
        if s == 1:
            return i
    return t.dim() - 1


def _make_layout_tensor(
    shape: Tuple[int, ...], dtype: torch.dtype, leading_dim: int
) -> torch.Tensor:

    ndim = len(shape)

    perm = [i for i in range(ndim) if i != leading_dim] + [leading_dim]
    permuted_shape = tuple(shape[p] for p in perm)
    t = torch.empty(permuted_shape, dtype=dtype, device="cuda")
    inv_perm = [perm.index(i) for i in range(ndim)]
    return t.permute(inv_perm)


def _make_compile_repr_tensors(
    dtype: torch.dtype,
    c_dtype: torch.dtype,
    has_bias: bool,
    a_leading: int,
    b_leading: int,
    c_leading: int,
):


    M, N, K, L = 64, 8, 128, 1
    A_t = _make_layout_tensor(
        (L, N, K), dtype, a_leading
    )
    B_t = _make_layout_tensor(
        (L, K, M), dtype, b_leading
    )
    C_t = _make_layout_tensor(
        (L, N, M), c_dtype, c_leading
    )

    a_ = from_dlpack(A_t, assumed_align=32).mark_layout_dynamic(leading_dim=a_leading)
    b_ = from_dlpack(B_t, assumed_align=32).mark_layout_dynamic(leading_dim=b_leading)
    c_ = from_dlpack(C_t, assumed_align=32).mark_layout_dynamic(leading_dim=c_leading)

    if not has_bias:
        return a_, b_, c_, None


    bias_t = torch.empty((N,), dtype=dtype, device="cuda")
    bias_3d = bias_t.as_strided(size=(L, N, M), stride=(0, 1, 0))
    bias_ = from_dlpack(bias_3d, assumed_align=2).mark_layout_dynamic(leading_dim=1)
    return a_, b_, c_, bias_


def _get_compiled_cute_ext_kernel(
    dtype: torch.dtype,
    c_dtype: torch.dtype,
    cta_m: int,
    cta_n: int,
    cta_k: int,
    num_ab_stage: int,
    use_2cta: bool,
    use_pdl: bool,
    has_bias: bool,
    a_leading: int,
    b_leading: int,
    c_leading: int,
):

    key = (
        dtype,
        c_dtype,
        cta_m,
        cta_n,
        cta_k,
        num_ab_stage,
        bool(use_2cta),
        bool(use_pdl),
        bool(has_bias),
        a_leading,
        b_leading,
        c_leading,
    )
    cached = _TGV_CUTE_EXT_COMPILE_CACHE.get(key)
    if cached is not None:
        return cached

    if dtype not in _TORCH_TO_CUTLASS_DTYPE:
        raise ValueError(
            f"TGV cute_ext backend supports {list(_TORCH_TO_CUTLASS_DTYPE)}; got {dtype}."
        )
    if c_dtype not in _TORCH_TO_CUTLASS_OUT_DTYPE:
        raise ValueError(
            f"TGV cute_ext output supports {list(_TORCH_TO_CUTLASS_OUT_DTYPE)}; "
            f"got {c_dtype}."
        )

    gemm = TgvGemmCuteExtKernel(
        acc_dtype=cutlass.Float32,
        cta_m=cta_m,
        cta_n=cta_n,
        cta_k=cta_k,
        num_ab_stage=num_ab_stage,
        use_2cta=use_2cta,
        use_pdl=use_pdl,
        has_bias=has_bias,
        out_dtype=_TORCH_TO_CUTLASS_OUT_DTYPE[c_dtype],
    )

    a_, b_, c_, bias_ = _make_compile_repr_tensors(
        dtype,
        c_dtype,
        has_bias,
        a_leading,
        b_leading,
        c_leading,
    )
    fake_stream = make_fake_stream()

    if has_bias:
        compiled = cute_ext.compile(_bmm_bias, gemm, a_, b_, c_, bias_, fake_stream)
    else:
        compiled = cute_ext.compile(_bmm_no_bias, gemm, a_, b_, c_, fake_stream)

    _TGV_CUTE_EXT_COMPILE_CACHE[key] = compiled
    return compiled


def _to_cute_swap(
    a_pt: torch.Tensor,
    b_pt: torch.Tensor,
    out_pt: torch.Tensor,
    bias_pt: Optional[torch.Tensor],
):


    if a_pt.dim() == 2:
        a_pt = a_pt.unsqueeze(0)
        b_pt = b_pt.unsqueeze(0)
        out_pt = out_pt.unsqueeze(0)


    a_swap = b_pt.transpose(-2, -1)
    b_swap = a_pt.transpose(-2, -1)
    c_swap = out_pt.transpose(-2, -1)

    a_leading = _detect_leading_dim(a_swap)
    b_leading = _detect_leading_dim(b_swap)
    c_leading = _detect_leading_dim(c_swap)

    a_ = from_dlpack(a_swap, assumed_align=32).mark_layout_dynamic(
        leading_dim=a_leading
    )
    b_ = from_dlpack(b_swap, assumed_align=32).mark_layout_dynamic(
        leading_dim=b_leading
    )
    c_ = from_dlpack(c_swap, assumed_align=32).mark_layout_dynamic(
        leading_dim=c_leading
    )

    layout = (a_leading, b_leading, c_leading)

    if bias_pt is None:
        return a_, b_, c_, None, layout


    L = c_swap.shape[0]
    M_ce = c_swap.shape[1]
    N_ce = c_swap.shape[2]
    bias_3d = bias_pt.as_strided(size=(L, M_ce, N_ce), stride=(0, 1, 0))
    bias_ = from_dlpack(bias_3d.detach(), assumed_align=2).mark_layout_dynamic(
        leading_dim=1
    )
    return a_, b_, c_, bias_, layout


def _resolve_tactic(tactic: int) -> Tuple[int, int, int, bool]:

    if tactic < 0:
        tactic = _TGV_CUTE_EXT_DEFAULT_TACTIC
    if tactic >= len(_TGV_CUTE_EXT_TACTIC_CONFIGS):
        raise ValueError(
            f"TGV cute_ext tactic {tactic} out of range [0, {len(_TGV_CUTE_EXT_TACTIC_CONFIGS)})."
        )
    return _TGV_CUTE_EXT_TACTIC_CONFIGS[tactic]


def _run_tgv(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor],
    out: torch.Tensor,
    pdl: bool,
    tactic: int,
) -> torch.Tensor:

    cta_m, cta_n, num_ab_stage, use_2cta = _resolve_tactic(tactic)
    has_bias = bias is not None

    a_, b_, c_, bias_, (a_leading, b_leading, c_leading) = _to_cute_swap(
        a,
        b,
        out,
        bias,
    )

    compiled = _get_compiled_cute_ext_kernel(
        dtype=a.dtype,
        c_dtype=out.dtype,
        cta_m=cta_m,
        cta_n=cta_n,
        cta_k=_TGV_CUTE_EXT_CTA_K,
        num_ab_stage=num_ab_stage,
        use_2cta=use_2cta,
        use_pdl=bool(pdl),
        has_bias=has_bias,
        a_leading=a_leading,
        b_leading=b_leading,
        c_leading=c_leading,
    )

    stream = cuda.CUstream(torch.cuda.current_stream(a.device).cuda_stream)

    if has_bias:
        compiled(a_, b_, c_, bias_, stream)
    else:
        compiled(a_, b_, c_, stream)
    return out


_TGV_NUM_SMS = 148


_TGV_TACTIC_LADDER = (
    (18, 64, 16, 2),
    (21, 64, 32, 2),
    (23, 64, 64, 2),
    (24, 64, 128, 2),
    (12, 128, 16, 1),
    (26, 128, 32, 2),
    (27, 128, 64, 2),
)


def _grid_ctas(m: int, n: int, cta_m: int, cta_n: int, num_mma_ctas: int) -> int:

    cluster_m = cta_m * num_mma_ctas
    return num_mma_ctas * -(n // -cluster_m) * -(m // -cta_n)


def _pick_tactic(m: int, n: int, k: int) -> int:

    best, best_ctas = None, None
    for tactic, cta_m, cta_n, num_mma_ctas in _TGV_TACTIC_LADDER:
        ctas = _grid_ctas(m, n, cta_m, cta_n, num_mma_ctas)
        if ctas <= _TGV_NUM_SMS:
            return tactic
        if best_ctas is None or ctas < best_ctas:
            best, best_ctas = tactic, ctas
    return best


_K3_TGV_WIN_SHAPES = frozenset(
    {
        (6144, 7168),
        (6016, 7168),
        (7168, 1536),
        (1536, 7168),
        (2304, 1536),
        (3584, 7168),
        (7168, 3584),
        (7168, 768),
    }
)


def use_cutedsl_bf16_gemm(m: int, n: int, k: int) -> bool:

    if m <= 0:


        return False
    if k % 8 != 0:
        return False
    if m <= 8 and (n, k) in _K3_TGV_WIN_SHAPES:
        return True
    if n < 1024 or k < 2048 or k > 6144:
        return False
    ragged = m % 16 != 0
    if n <= 1024:
        return m <= 512 and (m >= 48 or m % 16 >= 9)
    if n <= 2624:
        return m <= 128 and k >= 4096
    if n <= 4096:
        return m <= 64
    if k < 4096:
        return k >= 3072 and 25 <= m <= 48 and ragged
    if n <= 6144:
        if m <= 64:
            return m <= 32 or k >= 6144 or ragged
        return m <= 72 and k >= 6144 and ragged
    if n <= 8192:
        return m <= 48 or (m <= 63 and ragged)
    return n <= 12288 and k >= 6144 and (m <= 32 or (m <= 48 and ragged))


def _tgv_bf16_gemm_run(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    if not is_sm100_supported():
        raise RuntimeError("cutedsl_bf16_gemm requires an SM10x GPU")
    assert x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
    assert x.stride(-1) == 1, "x must be K-major [M, K]"
    assert weight.stride(-1) == 1, "weight must be K-major [N, K]"
    out = torch.empty(
        (x.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=x.device
    )
    if x.shape[0] == 0:


        return out
    return _run_tgv(
        x,
        weight.t(),
        bias,
        out,
        pdl=True,
        tactic=_pick_tactic(x.shape[0], weight.shape[0], weight.shape[1]),
    )


def _tgv_bf16_gemm_out_run(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> None:
    if not is_sm100_supported():
        raise RuntimeError("cutedsl_bf16_gemm requires an SM10x GPU")
    assert x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
    assert out.dtype in (torch.bfloat16, torch.float32) and out.device == x.device
    assert x.ndim == 2 and weight.ndim == 2 and out.ndim == 2
    assert x.stride(-1) == 1, "x must be K-major [M, K]"
    assert weight.stride(-1) == 1, "weight must be K-major [N, K]"
    assert out.is_contiguous() and out.shape == (x.shape[0], weight.shape[0])
    if x.shape[0] == 0:
        return None
    _run_tgv(
        x,
        weight.t(),
        bias,
        out,
        pdl=True,
        tactic=_pick_tactic(x.shape[0], weight.shape[0], weight.shape[1]),
    )
    return None


def _tgv_bf16_gemm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]))


def _tgv_bf16_gemm_out_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> None:
    return None


direct_register_custom_op(
    op_name="cutedsl_tgv_bf16_gemm",
    op_func=_tgv_bf16_gemm_run,
    mutates_args=[],
    fake_impl=_tgv_bf16_gemm_fake,
)

direct_register_custom_op(
    op_name="cutedsl_tgv_bf16_gemm_out",
    op_func=_tgv_bf16_gemm_out_run,
    mutates_args=["out"],
    fake_impl=_tgv_bf16_gemm_out_fake,
)


@debug_kernel_api
def cutedsl_bf16_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:

    return torch.ops.sglang.cutedsl_tgv_bf16_gemm(x, weight, bias)


@debug_kernel_api
def cutedsl_bf16_gemm_out(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:

    torch.ops.sglang.cutedsl_tgv_bf16_gemm_out(x, weight, out, bias)
    return out
