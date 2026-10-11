import cuda.bindings.driver as cuda_drv
import cutlass
import cutlass.cute as cute
from cutlass import for_generate, yield_out
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op

BT = 64
BC = 16
K_DIM = 128
K_PAD = 8
K_STRIDE = K_DIM + K_PAD
CHUNKS_PER_BLOCK = 4
NUM_SMS = 148

NUM_K1_TMA_WARPS = 16
NUM_MMA_WARPS = (
    11
)
NUM_MMA_ACTIVE = 10
TMA_WARP_ID = NUM_K1_TMA_WARPS + NUM_MMA_ACTIVE
NUM_STORE_WARPS = 4
NUM_WARPS = NUM_K1_TMA_WARPS + NUM_MMA_WARPS + NUM_STORE_WARPS
THREADS = NUM_WARPS * 32

NUM_SUB_CHUNKS = BT // BC
NUM_TILES = NUM_SUB_CHUNKS * (NUM_SUB_CHUNKS + 1) // 2
MMA_K_TILE = 16
NUM_MMA_K_TILES = K_DIM // MMA_K_TILE
AQK_TILE_PAD = 8
AQK_TILE_STRIDE = (
    BT + AQK_TILE_PAD
)

AKK_PAD = 8
AKK_STRIDE = BT + AKK_PAD


K1_ROW_GROUPS = 8
K1_COL_GROUPS = 2
ROWS_PER_K1_WARP = BT // K1_ROW_GROUPS
K1_COLS_PER_WARP = K_DIM // K1_COL_GROUPS

VEC = K1_COLS_PER_WARP // 32
K_VEC = K_DIM // VEC
NUM_STAGES = 2
PARTIAL_COLS = K_DIM + 4
PARTIAL_COLS_PER_WARP = K_DIM // NUM_K1_TMA_WARPS

_TILE_IQ = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]
_TILE_IK = [0, 0, 1, 0, 1, 2, 0, 1, 2, 3]

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
RCP_LN2 = LOG2E


@dsl_user_op
def k1_internal_barrier(*, loc=None, ip=None):

    llvm.inline_asm(
        T.i32(),
        [],
        "membar.cta; bar.sync 2, 512; mov.u32 $0, 0;",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def pack_bf16x2_f32(hi_f32, lo_f32, *, loc=None, ip=None):

    result = llvm.inline_asm(
        T.i32(),
        [hi_f32.ir_value(loc=loc, ip=ip), lo_f32.ir_value(loc=loc, ip=ip)],
        "cvt.rn.bf16x2.f32 $0, $1, $2;",
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(result)


@dsl_user_op
def mma_bf16_m16n8k16(
    a0,
    a1,
    a2,
    a3,
    b0,
    b1,
    c0,
    c1,
    c2,
    c3,
    *,
    loc=None,
    ip=None,
):


    result = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(f32, f32, f32, f32)>"),
        [
            a0.ir_value(loc=loc, ip=ip),
            a1.ir_value(loc=loc, ip=ip),
            a2.ir_value(loc=loc, ip=ip),
            a3.ir_value(loc=loc, ip=ip),
            b0.ir_value(loc=loc, ip=ip),
            b1.ir_value(loc=loc, ip=ip),
            c0.ir_value(loc=loc, ip=ip),
            c1.ir_value(loc=loc, ip=ip),
            c2.ir_value(loc=loc, ip=ip),
            c3.ir_value(loc=loc, ip=ip),
        ],
        """{
            mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
                {$0, $1, $2, $3},
                {$4, $5, $6, $7},
                {$8, $9},
                {$10, $11, $12, $13};
        }""",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

    d0 = cutlass.Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip))
    d1 = cutlass.Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip))
    d2 = cutlass.Float32(llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip))
    d3 = cutlass.Float32(llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip))
    return d0, d1, d2, d3


SHFL_W8_CLAMP = 0x1800


@dsl_user_op
def fast_rcp(x, *, loc=None, ip=None):

    result = llvm.inline_asm(
        T.f32(),
        [x.ir_value(loc=loc, ip=ip)],
        "rcp.approx.ftz.f32 $0, $1;",
        "=f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Float32(result)


@dsl_user_op
def opaque_zero_from_work_id(*, loc=None, ip=None):

    result = llvm.inline_asm(
        T.i32(),
        [],
        "mov.b32 $0, 0;",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(result)


@cute.kernel
def fused_kernel123(
    tma_atom_Q: cute.CopyAtom,
    tma_tensor_Q: cute.Tensor,
    tma_atom_K: cute.CopyAtom,
    tma_tensor_K: cute.Tensor,
    tma_atom_G: cute.CopyAtom,
    tma_tensor_G: cute.Tensor,
    mA_log: cute.Tensor,
    mBeta: cute.Tensor,
    scale: cutlass.Float32,
    mKscaled: cute.Tensor,
    mKg: cute.Tensor,
    mQscaled: cute.Tensor,
    mGkLast: cute.Tensor,
    mAqk: cute.Tensor,
    mAkk: cute.Tensor,
    mAqk_v2: cute.Tensor,
    mAkk_v2: cute.Tensor,
    tiled_copy_qk_k1,
    tiled_mma_k2,
    tiled_copy_mma_A,
    tiled_copy_mma_B,
    tiled_copy_Gcum_norm,
    tiled_copy_Gcum_gate,
    qk_smem_layout,
    g_smem_layout,
    g_cumsum_layout,
    num_chunks: int,
    num_heads: int,
    batch_size: int,
    mCuSeqlens: cute.Tensor,
    mChunkIndices: cute.Tensor,
    IS_VARLEN: cutlass.Constexpr[int],
    mDtBias: cute.Tensor,
    lower_bound: cutlass.Float32,
    HAS_BIAS: cutlass.Constexpr[int],
    USE_SAFE_GATE: cutlass.Constexpr[int],
    VARLEN_PURE: cutlass.Constexpr[int] = 0,
):
    block_id, _, _ = cute.arch.block_idx()
    tidx = cute.arch.thread_idx()[0]
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    lane_id = tidx % 32

    total_cgs_per_head = cutlass.Int32(0)
    cgs_per_head = cutlass.Int32(0)
    total_cgs = cutlass.Int32(0)
    if IS_VARLEN:
        total_cgs_per_head = (num_chunks + CHUNKS_PER_BLOCK - 1) // CHUNKS_PER_BLOCK
        total_cgs = total_cgs_per_head * num_heads
    else:
        cgs_per_head = num_chunks // CHUNKS_PER_BLOCK
        total_cgs = cgs_per_head * num_heads * batch_size

    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(
        cutlass.BFloat16, qk_smem_layout.outer, 128, swizzle=qk_smem_layout.inner
    )
    sK = smem.allocate_tensor(
        cutlass.BFloat16, qk_smem_layout.outer, 128, swizzle=qk_smem_layout.inner
    )
    sG = smem.allocate_tensor(cutlass.BFloat16, g_smem_layout, 128)
    sGcum = smem.allocate_tensor(cutlass.Float32, g_cumsum_layout, 128)
    partial_last_layout = cute.make_layout(
        (K1_ROW_GROUPS, PARTIAL_COLS), stride=(PARTIAL_COLS, 1)
    )
    sPartialLast = smem.allocate_tensor(cutlass.Float32, partial_last_layout, 128)


    aqk_tile_layout = cute.make_layout(
        (BT, AQK_TILE_STRIDE, NUM_STAGES),
        stride=(AQK_TILE_STRIDE, 1, BT * AQK_TILE_STRIDE),
    )
    sAqk = smem.allocate_tensor(cutlass.BFloat16, aqk_tile_layout, 128)

    akk_tile_layout = cute.make_layout(
        (BT, AKK_STRIDE, NUM_STAGES), stride=(AKK_STRIDE, 1, BT * AKK_STRIDE)
    )
    sAkk = smem.allocate_tensor(cutlass.BFloat16, akk_tile_layout, 128)


    beta_smem_layout = cute.make_layout((BT, NUM_STAGES), stride=(1, BT))
    sBeta = smem.allocate_tensor(cutlass.BFloat16, beta_smem_layout, 128)


    tma_mbars = smem.allocate_array(cutlass.Int64, NUM_STAGES)
    stage_reuse_mbars = smem.allocate_array(cutlass.Int64, NUM_STAGES)
    k1_done_mbars = smem.allocate_array(cutlass.Int64, NUM_STAGES)
    mma_done_mbars = smem.allocate_array(cutlass.Int64, NUM_STAGES)
    store_done_mbars = smem.allocate_array(cutlass.Int64, NUM_STAGES)

    bytes_per_stage = BT * K_DIM * 2 * 3

    if tidx == 0:
        for s in range(NUM_STAGES):
            cute.arch.mbarrier_init(tma_mbars + s, 1)

            cute.arch.mbarrier_init(stage_reuse_mbars + s, (NUM_MMA_WARPS - 1) * 32)
            cute.arch.mbarrier_init(k1_done_mbars + s, NUM_K1_TMA_WARPS * 32)
            cute.arch.mbarrier_init(mma_done_mbars + s, (NUM_MMA_WARPS - 1) * 32)
            cute.arch.mbarrier_init(store_done_mbars + s, NUM_STORE_WARPS * 32)
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()


    _warp_id_in_cta = tidx >> 5
    _lane_id_warp = tidx & 31
    _row_base = _warp_id_in_cta * 2
    _col_lo = _lane_id_warp * 2
    _col_hi = _col_lo + 1
    for _s in cutlass.range_constexpr(NUM_STAGES):
        for _ri in cutlass.range_constexpr(2):
            _row = _row_base + _ri
            sAqk[_row, _col_lo, _s] = cutlass.BFloat16(0.0)
            sAqk[_row, _col_hi, _s] = cutlass.BFloat16(0.0)
            sAkk[_row, _col_lo, _s] = cutlass.BFloat16(0.0)
            sAkk[_row, _col_hi, _s] = cutlass.BFloat16(0.0)
    cute.arch.barrier()


    if (
        warp_idx >= NUM_K1_TMA_WARPS
        and warp_idx < NUM_K1_TMA_WARPS + NUM_MMA_WARPS
        and warp_idx != TMA_WARP_ID
    ):
        mma_warp_tmp = warp_idx - NUM_K1_TMA_WARPS
        for s in range(NUM_STAGES):
            cute.arch.mbarrier_arrive(stage_reuse_mbars + s)
            if mma_warp_tmp < NUM_STORE_WARPS:
                cute.arch.mbarrier_arrive(store_done_mbars + s)


    for work_id in for_generate(block_id, total_cgs, NUM_SMS):
        i_cg = cutlass.Int32(0)
        i_h = cutlass.Int32(0)
        i_b = cutlass.Int32(0)
        chunk_base = cutlass.Int32(0)
        if IS_VARLEN:
            i_cg = work_id % total_cgs_per_head
            i_h = work_id // total_cgs_per_head
            i_b = cutlass.Int32(0)
            chunk_base = i_cg * CHUNKS_PER_BLOCK
        else:
            i_cg = work_id % cgs_per_head
            i_h = (work_id // cgs_per_head) % num_heads
            i_b = work_id // (cgs_per_head * num_heads)
            chunk_base = i_cg * CHUNKS_PER_BLOCK


        _oz = opaque_zero_from_work_id()
        _lane = lane_id + _oz
        _warp = warp_idx + _oz


        if warp_idx < NUM_K1_TMA_WARPS:

            k1_warp = _warp
            warp_row_group = k1_warp % K1_ROW_GROUPS
            warp_col_group = k1_warp // K1_ROW_GROUPS
            k1_row_start = warp_row_group * ROWS_PER_K1_WARP
            col_base = warp_col_group * K1_COLS_PER_WARP + _lane * VEC
            col_vec_idx = warp_col_group * (K1_COLS_PER_WARP // VEC) + _lane
            cumsum_scale = cutlass.Float32(RCP_LN2)
            thr_copy_k1 = tiled_copy_qk_k1.get_slice(_lane)

            rAcc = cute.make_rmem_tensor(cute.make_layout((VEC,)), cutlass.Float32)
            rPrefix = cute.make_rmem_tensor(cute.make_layout((VEC,)), cutlass.Float32)
            rGkLast = cute.make_rmem_tensor(cute.make_layout((VEC,)), cutlass.Float32)
            rKsOut = cute.make_rmem_tensor(cute.make_layout((VEC,)), cutlass.BFloat16)
            rQsOut = cute.make_rmem_tensor(cute.make_layout((VEC,)), cutlass.BFloat16)
            rKgOut = cute.make_rmem_tensor(cute.make_layout((VEC,)), cutlass.BFloat16)
            rGkOut = cute.make_rmem_tensor(cute.make_layout((VEC,)), cutlass.Float32)


            exp_A = cute.exp(mA_log[i_h], fastmath=True)


            rBias = cute.make_rmem_tensor(cute.make_layout((VEC,)), cutlass.Float32)
            if HAS_BIAS:
                for vi in cutlass.range_constexpr(VEC):
                    rBias[vi] = mDtBias[i_h, col_base + vi]
            else:
                for vi in cutlass.range_constexpr(VEC):
                    rBias[vi] = cutlass.Float32(0.0)


            gQ_head = tma_tensor_Q[(None, None, i_h)]
            gK_head = tma_tensor_K[(None, None, i_h)]
            gG_head = tma_tensor_G[(None, None, i_h)]

            for chunk_iter in cutlass.range_constexpr(CHUNKS_PER_BLOCK):
                cur_stage = chunk_iter % NUM_STAGES
                cur_phase = chunk_iter // NUM_STAGES % 2
                chunk_idx = chunk_base + chunk_iter
                chunk_start = cutlass.Int32(0)
                ci_eos = cutlass.Int32(0)
                if IS_VARLEN:
                    if chunk_idx < num_chunks:
                        _sid = cutlass.Int32(mChunkIndices[chunk_idx, 0])
                        chunk_start = (
                            cutlass.Int32(mCuSeqlens[_sid])
                            + cutlass.Int32(mChunkIndices[chunk_idx, 1]) * BT
                        )
                        ci_eos = cutlass.Int32(mCuSeqlens[_sid + 1])
                else:
                    chunk_start = chunk_idx * BT

                cute.arch.mbarrier_wait(tma_mbars + cur_stage, cur_phase)

                csG = sG[(None, None, cur_stage)]
                csGcum = sGcum[(None, None, cur_stage)]
                csQ = sQ[(None, None, cur_stage)]
                csK = sK[(None, None, cur_stage)]
                csBeta = sBeta[(None, cur_stage)]


                if k1_warp == 0:
                    for _bi in cutlass.range_constexpr(BT // 32):
                        _idx = _bi * 32 + _lane
                        csBeta[_idx] = mBeta[i_b, chunk_start + _idx, i_h]

                rGact = cute.make_rmem_tensor(
                    cute.make_layout((ROWS_PER_K1_WARP, VEC)), cutlass.Float32
                )
                for vi in cutlass.range_constexpr(VEC):
                    rAcc[vi] = cutlass.Float32(0.0)

                for ri in cutlass.range_constexpr(ROWS_PER_K1_WARP):
                    row = k1_row_start + ri
                    for vi in cutlass.range_constexpr(VEC):
                        c = col_base + vi
                        g_val = csG[row, c].to(cutlass.Float32)
                        if HAS_BIAS:
                            g_val = g_val + rBias[vi]
                        g_activated = cutlass.Float32(0.0)
                        if USE_SAFE_GATE:
                            sigmoid_g = fast_rcp(
                                cutlass.Float32(1.0)
                                + cute.exp2(-exp_A * g_val * LOG2E, fastmath=True)
                            )
                            g_activated = lower_bound * sigmoid_g
                        else:
                            softplus_g = (
                                cute.log2(
                                    cutlass.Float32(1.0)
                                    + cute.exp2(g_val * LOG2E, fastmath=True),
                                    fastmath=True,
                                )
                                * LN2
                            )
                            g_activated = -exp_A * softplus_g


                        if IS_VARLEN and not VARLEN_PURE:
                            if chunk_start + row >= ci_eos:
                                g_activated = cutlass.Float32(0.0)
                        rGact[ri, vi] = g_activated
                        rAcc[vi] = rAcc[vi] + g_activated

                for vi in cutlass.range_constexpr(VEC):
                    sPartialLast[warp_row_group, col_base + vi] = rAcc[vi]

                k1_internal_barrier()

                prefix_col_start = k1_warp * PARTIAL_COLS_PER_WARP
                row_in_prefix = lane_id % K1_ROW_GROUPS
                col_in_group = lane_id // K1_ROW_GROUPS

                for j in cutlass.range_constexpr(PARTIAL_COLS_PER_WARP // 4):
                    col = prefix_col_start + j * 4 + col_in_group
                    val = cutlass.Float32(sPartialLast[row_in_prefix, col])
                    tmp = cute.arch.shuffle_sync_up(
                        val, 1, mask=-1, mask_and_clamp=SHFL_W8_CLAMP
                    )
                    if row_in_prefix >= 1:
                        val = val + tmp
                    tmp = cute.arch.shuffle_sync_up(
                        val, 2, mask=-1, mask_and_clamp=SHFL_W8_CLAMP
                    )
                    if row_in_prefix >= 2:
                        val = val + tmp
                    tmp = cute.arch.shuffle_sync_up(
                        val, 4, mask=-1, mask_and_clamp=SHFL_W8_CLAMP
                    )
                    if row_in_prefix >= 4:
                        val = val + tmp
                    sPartialLast[row_in_prefix, col] = val

                k1_internal_barrier()

                for vi in cutlass.range_constexpr(VEC):
                    rGkLast[vi] = sPartialLast[K1_ROW_GROUPS - 1, col_base + vi]

                for vi in cutlass.range_constexpr(VEC):
                    rPrefix[vi] = cutlass.Float32(0.0)
                if warp_row_group > 0:
                    for vi in cutlass.range_constexpr(VEC):
                        rPrefix[vi] = sPartialLast[warp_row_group - 1, col_base + vi]


                for vi in cutlass.range_constexpr(VEC):
                    rAcc[vi] = rPrefix[vi]

                for ri in cutlass.range_constexpr(ROWS_PER_K1_WARP):
                    row = k1_row_start + ri
                    for vi in cutlass.range_constexpr(VEC):
                        rAcc[vi] = rAcc[vi] + rGact[ri, vi]
                        csGcum[row, col_base + vi] = rAcc[vi] * cumsum_scale


                cute.arch.mbarrier_arrive(k1_done_mbars + cur_stage)


                for vi in cutlass.range_constexpr(VEC):
                    rAcc[vi] = rPrefix[vi]

                for ri in cutlass.range_constexpr(ROWS_PER_K1_WARP):
                    row = k1_row_start + ri
                    t = chunk_start + row

                    sK_tile = cute.local_tile(
                        csK, tiler=(1, K1_COLS_PER_WARP), coord=(row, warp_col_group)
                    )
                    tCsK = thr_copy_k1.partition_S(sK_tile)
                    tCrK = cute.make_fragment_like(tCsK)
                    cute.copy(tiled_copy_qk_k1, tCsK, thr_copy_k1.retile(tCrK))

                    sQ_tile = cute.local_tile(
                        csQ, tiler=(1, K1_COLS_PER_WARP), coord=(row, warp_col_group)
                    )
                    tCsQ = thr_copy_k1.partition_S(sQ_tile)
                    tCrQ = cute.make_fragment_like(tCsQ)
                    cute.copy(tiled_copy_qk_k1, tCsQ, thr_copy_k1.retile(tCrQ))


                    beta_val = cutlass.Float32(csBeta[row])

                    for vi in cutlass.range_constexpr(VEC):
                        rAcc[vi] = rAcc[vi] + rGact[ri, vi]
                        cs = rAcc[vi] * cumsum_scale

                        k_val = tCrK[vi].to(cutlass.Float32)
                        q_val = tCrQ[vi].to(cutlass.Float32)

                        exp2_cs = cute.exp2(cs, fastmath=True)
                        gk_last_cs = rGkLast[vi] * cumsum_scale
                        exp2_kg = cute.exp2(gk_last_cs - cs, fastmath=True)

                        rKsOut[vi] = (k_val * exp2_cs).to(cutlass.BFloat16)
                        rQsOut[vi] = (q_val * exp2_cs * scale).to(cutlass.BFloat16)
                        rKgOut[vi] = (k_val * exp2_kg).to(cutlass.BFloat16)

                    if IS_VARLEN and not VARLEN_PURE:
                        if t < ci_eos:
                            cute.autovec_copy(
                                rKsOut, mKscaled[i_b, t, i_h, col_vec_idx, None]
                            )
                            cute.autovec_copy(
                                rQsOut, mQscaled[i_b, t, i_h, col_vec_idx, None]
                            )
                            cute.autovec_copy(
                                rKgOut, mKg[i_b, t, i_h, col_vec_idx, None]
                            )
                    else:
                        cute.autovec_copy(
                            rKsOut, mKscaled[i_b, t, i_h, col_vec_idx, None]
                        )
                        cute.autovec_copy(
                            rQsOut, mQscaled[i_b, t, i_h, col_vec_idx, None]
                        )
                        cute.autovec_copy(rKgOut, mKg[i_b, t, i_h, col_vec_idx, None])

                if warp_row_group == 0:
                    for vi in cutlass.range_constexpr(VEC):
                        rGkOut[vi] = cute.exp2(
                            rGkLast[vi] * cumsum_scale, fastmath=True
                        )
                    if IS_VARLEN:
                        if ci_eos > cutlass.Int32(0):
                            cute.autovec_copy(
                                rGkOut, mGkLast[i_b, chunk_idx, i_h, col_vec_idx, None]
                            )
                    else:
                        cute.autovec_copy(
                            rGkOut, mGkLast[i_b, chunk_idx, i_h, col_vec_idx, None]
                        )


        if warp_idx == TMA_WARP_ID:
            gQ_head = tma_tensor_Q[(None, None, i_h)]
            gK_head = tma_tensor_K[(None, None, i_h)]
            gG_head = tma_tensor_G[(None, None, i_h)]


            pf_cs = cutlass.Int32(0)
            if IS_VARLEN:
                pf_seq_id_0 = cutlass.Int32(mChunkIndices[chunk_base, 0])
                pf_local_0 = cutlass.Int32(mChunkIndices[chunk_base, 1])
                pf_bos_0 = cutlass.Int32(mCuSeqlens[pf_seq_id_0])
                pf_cs = pf_bos_0 + pf_local_0 * BT
            else:
                pf_cs = i_b * num_chunks * BT + chunk_base * BT
            cute.arch.mbarrier_wait(stage_reuse_mbars, 0)
            if lane_id == 0:
                cute.arch.mbarrier_expect_tx(tma_mbars, bytes_per_stage)
            sQ_pf = sQ[(None, None, 0)]
            gQ_pf = cute.local_tile(
                cute.domain_offset((pf_cs, 0), gQ_head), (BT, K_DIM), (0, 0)
            )
            ts_pf, tg_pf = cpasync.tma_partition(
                tma_atom_Q,
                0,
                cute.make_layout(1),
                cute.group_modes(sQ_pf, 0, 2),
                cute.group_modes(gQ_pf, 0, 2),
            )
            cute.copy(tma_atom_Q, tg_pf, ts_pf, tma_bar_ptr=tma_mbars)
            sK_pf = sK[(None, None, 0)]
            gK_pf = cute.local_tile(
                cute.domain_offset((pf_cs, 0), gK_head), (BT, K_DIM), (0, 0)
            )
            ts_pf, tg_pf = cpasync.tma_partition(
                tma_atom_K,
                0,
                cute.make_layout(1),
                cute.group_modes(sK_pf, 0, 2),
                cute.group_modes(gK_pf, 0, 2),
            )
            cute.copy(tma_atom_K, tg_pf, ts_pf, tma_bar_ptr=tma_mbars)
            sG_pf = sG[(None, None, 0)]
            gG_pf = cute.local_tile(
                cute.domain_offset((pf_cs, 0), gG_head), (BT, K_DIM), (0, 0)
            )
            ts_pf, tg_pf = cpasync.tma_partition(
                tma_atom_G,
                0,
                cute.make_layout(1),
                cute.group_modes(sG_pf, 0, 2),
                cute.group_modes(gG_pf, 0, 2),
            )
            cute.copy(tma_atom_G, tg_pf, ts_pf, tma_bar_ptr=tma_mbars)
            if lane_id == 0:
                cute.arch.mbarrier_arrive(tma_mbars)


            for next_i in cutlass.range_constexpr(1, CHUNKS_PER_BLOCK):
                next_stage = next_i % NUM_STAGES
                next_phase = next_i // NUM_STAGES % 2
                next_cs = cutlass.Int32(0)
                if IS_VARLEN:
                    next_chunk_idx = chunk_base + next_i
                    if next_chunk_idx < num_chunks:
                        _nsid = cutlass.Int32(mChunkIndices[next_chunk_idx, 0])
                        next_cs = (
                            cutlass.Int32(mCuSeqlens[_nsid])
                            + cutlass.Int32(mChunkIndices[next_chunk_idx, 1]) * BT
                        )
                else:
                    next_cs = i_b * num_chunks * BT + (chunk_base + next_i) * BT
                cute.arch.mbarrier_wait(stage_reuse_mbars + next_stage, next_phase)
                if lane_id == 0:
                    cute.arch.mbarrier_expect_tx(
                        tma_mbars + next_stage, bytes_per_stage
                    )
                sQ_ns = sQ[(None, None, next_stage)]
                gQ_ns = cute.local_tile(
                    cute.domain_offset((next_cs, 0), gQ_head), (BT, K_DIM), (0, 0)
                )
                ts_ns, tg_ns = cpasync.tma_partition(
                    tma_atom_Q,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sQ_ns, 0, 2),
                    cute.group_modes(gQ_ns, 0, 2),
                )
                cute.copy(tma_atom_Q, tg_ns, ts_ns, tma_bar_ptr=tma_mbars + next_stage)
                sK_ns = sK[(None, None, next_stage)]
                gK_ns = cute.local_tile(
                    cute.domain_offset((next_cs, 0), gK_head), (BT, K_DIM), (0, 0)
                )
                ts_ns, tg_ns = cpasync.tma_partition(
                    tma_atom_K,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sK_ns, 0, 2),
                    cute.group_modes(gK_ns, 0, 2),
                )
                cute.copy(tma_atom_K, tg_ns, ts_ns, tma_bar_ptr=tma_mbars + next_stage)
                sG_ns = sG[(None, None, next_stage)]
                gG_ns = cute.local_tile(
                    cute.domain_offset((next_cs, 0), gG_head), (BT, K_DIM), (0, 0)
                )
                ts_ns, tg_ns = cpasync.tma_partition(
                    tma_atom_G,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sG_ns, 0, 2),
                    cute.group_modes(gG_ns, 0, 2),
                )
                cute.copy(tma_atom_G, tg_ns, ts_ns, tma_bar_ptr=tma_mbars + next_stage)
                if lane_id == 0:
                    cute.arch.mbarrier_arrive(tma_mbars + next_stage)


        if (
            warp_idx >= NUM_K1_TMA_WARPS
            and warp_idx < NUM_K1_TMA_WARPS + NUM_MMA_WARPS
            and warp_idx != TMA_WARP_ID
        ):

            _tid_in_group = _lane % 4
            _group_id = _lane // 4
            mma_warp = _warp - NUM_K1_TMA_WARPS
            my_i_q = cutlass.Int32(0)
            my_i_k = cutlass.Int32(0)
            if mma_warp < 1:
                my_i_q = cutlass.Int32(0)
                my_i_k = mma_warp
            elif mma_warp < 3:
                my_i_q = cutlass.Int32(1)
                my_i_k = mma_warp - 1
            elif mma_warp < 6:
                my_i_q = cutlass.Int32(2)
                my_i_k = mma_warp - 3
            elif mma_warp < NUM_MMA_ACTIVE:
                my_i_q = cutlass.Int32(3)
                my_i_k = mma_warp - 6
            q_row_base = my_i_q * BC
            k_row_base = my_i_k * BC
            akk_row_base = k_row_base
            akk_col_base = q_row_base
            norm_row = q_row_base
            if my_i_q == my_i_k:
                norm_row = q_row_base + cutlass.Int32(BC // 2)
            row0 = _group_id
            row1 = _group_id + 8
            col0 = _tid_in_group * 2
            col1 = _tid_in_group * 2 + 1
            col2 = 8 + _tid_in_group * 2
            col3 = 8 + _tid_in_group * 2 + 1
            thr_mma = tiled_mma_k2.get_slice(_lane)
            thr_copy_A = tiled_copy_mma_A.get_slice(_lane)
            thr_copy_B = tiled_copy_mma_B.get_slice(_lane)
            thr_copy_Gn = tiled_copy_Gcum_norm.get_slice(_tid_in_group)
            thr_copy_Ggate = tiled_copy_Gcum_gate.get_slice(_lane)

            for chunk_iter in cutlass.range_constexpr(CHUNKS_PER_BLOCK):
                s = chunk_iter % NUM_STAGES
                phase = chunk_iter // NUM_STAGES % 2
                chunk_idx = chunk_base + chunk_iter
                chunk_start = cutlass.Int32(0)
                if IS_VARLEN:
                    if chunk_idx < num_chunks:
                        _sid = cutlass.Int32(mChunkIndices[chunk_idx, 0])
                        chunk_start = (
                            cutlass.Int32(mCuSeqlens[_sid])
                            + cutlass.Int32(mChunkIndices[chunk_idx, 1]) * BT
                        )
                else:
                    chunk_start = chunk_idx * BT

                cute.arch.mbarrier_wait(k1_done_mbars + s, phase)
                cute.arch.mbarrier_wait(store_done_mbars + s, phase)

                if mma_warp < NUM_MMA_ACTIVE:
                    csQ = sQ[(None, None, s)]
                    csK = sK[(None, None, s)]
                    csGcum = sGcum[(None, None, s)]
                    csAqk = sAqk[(None, None, s)]
                    csAkk = sAkk[(None, None, s)]

                    _z = cutlass.Float32(0.0)

                    beta_row0 = mBeta[i_b, chunk_start + q_row_base + row0, i_h].to(
                        cutlass.Float32
                    )
                    beta_row1 = mBeta[i_b, chunk_start + q_row_base + row1, i_h].to(
                        cutlass.Float32
                    )

                    acc_aqk_n0_0, acc_aqk_n0_1, acc_aqk_n0_2, acc_aqk_n0_3 = (
                        _z,
                        _z,
                        _z,
                        _z,
                    )
                    acc_aqk_n1_0, acc_aqk_n1_1, acc_aqk_n1_2, acc_aqk_n1_3 = (
                        _z,
                        _z,
                        _z,
                        _z,
                    )
                    acc_akk_n0_0, acc_akk_n0_1, acc_akk_n0_2, acc_akk_n0_3 = (
                        _z,
                        _z,
                        _z,
                        _z,
                    )
                    acc_akk_n1_0, acc_akk_n1_1, acc_akk_n1_2, acc_akk_n1_3 = (
                        _z,
                        _z,
                        _z,
                        _z,
                    )


                    for k_block in cutlass.range_constexpr(NUM_MMA_K_TILES):

                        sQ_tile = cute.local_tile(
                            csQ, tiler=(16, 16), coord=(my_i_q, k_block)
                        )
                        tCrQ = tiled_mma_k2.make_fragment_A(
                            thr_mma.partition_A(sQ_tile)
                        )
                        cute.copy(
                            tiled_copy_mma_A,
                            thr_copy_A.partition_S(sQ_tile),
                            thr_copy_A.retile(tCrQ),
                        )

                        sKq_tile = cute.local_tile(
                            csK, tiler=(16, 16), coord=(my_i_q, k_block)
                        )
                        tCrKq = tiled_mma_k2.make_fragment_A(
                            thr_mma.partition_A(sKq_tile)
                        )
                        cute.copy(
                            tiled_copy_mma_A,
                            thr_copy_A.partition_S(sKq_tile),
                            thr_copy_A.retile(tCrKq),
                        )


                        sK_tile_n0 = cute.local_tile(
                            csK, tiler=(8, 16), coord=(my_i_k * 2, k_block)
                        )
                        tCrK_n0 = tiled_mma_k2.make_fragment_B(
                            thr_mma.partition_B(sK_tile_n0)
                        )
                        cute.copy(
                            tiled_copy_mma_B,
                            thr_copy_B.partition_S(sK_tile_n0),
                            thr_copy_B.retile(tCrK_n0),
                        )

                        sK_tile_n1 = cute.local_tile(
                            csK, tiler=(8, 16), coord=(my_i_k * 2 + 1, k_block)
                        )
                        tCrK_n1 = tiled_mma_k2.make_fragment_B(
                            thr_mma.partition_B(sK_tile_n1)
                        )
                        cute.copy(
                            tiled_copy_mma_B,
                            thr_copy_B.partition_S(sK_tile_n1),
                            thr_copy_B.retile(tCrK_n1),
                        )


                        sGn_a = cute.local_tile(
                            csGcum, tiler=(1, 8), coord=(norm_row, k_block * 2)
                        )
                        tCsGn_a = thr_copy_Gn.partition_S(sGn_a)
                        tCrGn_a = cute.make_fragment_like(tCsGn_a, cutlass.Float32)
                        cute.copy(
                            tiled_copy_Gcum_norm, tCsGn_a, thr_copy_Gn.retile(tCrGn_a)
                        )
                        gn_a0 = tCrGn_a[0]
                        gn_a1 = tCrGn_a[1]

                        sGn_b = cute.local_tile(
                            csGcum, tiler=(1, 8), coord=(norm_row, k_block * 2 + 1)
                        )
                        tCsGn_b = thr_copy_Gn.partition_S(sGn_b)
                        tCrGn_b = cute.make_fragment_like(tCsGn_b, cutlass.Float32)
                        cute.copy(
                            tiled_copy_Gcum_norm, tCsGn_b, thr_copy_Gn.retile(tCrGn_b)
                        )
                        gn_b0 = tCrGn_b[0]
                        gn_b1 = tCrGn_b[1]


                        sGq_a = cute.local_tile(
                            csGcum, tiler=(16, 8), coord=(my_i_q, k_block * 2)
                        )
                        tCrGq_a = tiled_mma_k2.make_fragment_C(
                            thr_mma.partition_C(sGq_a)
                        )
                        cute.copy(
                            tiled_copy_Gcum_gate,
                            thr_copy_Ggate.partition_S(sGq_a),
                            thr_copy_Ggate.retile(tCrGq_a),
                        )

                        sGq_b = cute.local_tile(
                            csGcum, tiler=(16, 8), coord=(my_i_q, k_block * 2 + 1)
                        )
                        tCrGq_b = tiled_mma_k2.make_fragment_C(
                            thr_mma.partition_C(sGq_b)
                        )
                        cute.copy(
                            tiled_copy_Gcum_gate,
                            thr_copy_Ggate.partition_S(sGq_b),
                            thr_copy_Ggate.retile(tCrGq_b),
                        )


                        gate_q_0 = cute.exp2(tCrGq_a[0] - gn_a0, fastmath=True)
                        gate_q_1 = cute.exp2(tCrGq_a[1] - gn_a1, fastmath=True)
                        gate_q_2 = cute.exp2(tCrGq_a[2] - gn_a0, fastmath=True)
                        gate_q_3 = cute.exp2(tCrGq_a[3] - gn_a1, fastmath=True)
                        gate_q_4 = cute.exp2(tCrGq_b[0] - gn_b0, fastmath=True)
                        gate_q_5 = cute.exp2(tCrGq_b[1] - gn_b1, fastmath=True)
                        gate_q_6 = cute.exp2(tCrGq_b[2] - gn_b0, fastmath=True)
                        gate_q_7 = cute.exp2(tCrGq_b[3] - gn_b1, fastmath=True)


                        qa0 = tCrQ[0].to(cutlass.Float32) * gate_q_0
                        qa1 = tCrQ[1].to(cutlass.Float32) * gate_q_1
                        qa2 = tCrQ[2].to(cutlass.Float32) * gate_q_2
                        qa3 = tCrQ[3].to(cutlass.Float32) * gate_q_3
                        qa4 = tCrQ[4].to(cutlass.Float32) * gate_q_4
                        qa5 = tCrQ[5].to(cutlass.Float32) * gate_q_5
                        qa6 = tCrQ[6].to(cutlass.Float32) * gate_q_6
                        qa7 = tCrQ[7].to(cutlass.Float32) * gate_q_7


                        qa_u32_0 = pack_bf16x2_f32(qa1, qa0)
                        qa_u32_1 = pack_bf16x2_f32(qa3, qa2)
                        qa_u32_2 = pack_bf16x2_f32(qa5, qa4)
                        qa_u32_3 = pack_bf16x2_f32(qa7, qa6)

                        ka0 = tCrKq[0].to(cutlass.Float32) * gate_q_0
                        ka1 = tCrKq[1].to(cutlass.Float32) * gate_q_1
                        ka2 = tCrKq[2].to(cutlass.Float32) * gate_q_2
                        ka3 = tCrKq[3].to(cutlass.Float32) * gate_q_3
                        ka4 = tCrKq[4].to(cutlass.Float32) * gate_q_4
                        ka5 = tCrKq[5].to(cutlass.Float32) * gate_q_5
                        ka6 = tCrKq[6].to(cutlass.Float32) * gate_q_6
                        ka7 = tCrKq[7].to(cutlass.Float32) * gate_q_7

                        ka_u32_0 = pack_bf16x2_f32(ka1, ka0)
                        ka_u32_1 = pack_bf16x2_f32(ka3, ka2)
                        ka_u32_2 = pack_bf16x2_f32(ka5, ka4)
                        ka_u32_3 = pack_bf16x2_f32(ka7, ka6)


                        sGk_a = cute.local_tile(
                            csGcum, tiler=(16, 8), coord=(my_i_k, k_block * 2)
                        )
                        tCrGk_a = tiled_mma_k2.make_fragment_C(
                            thr_mma.partition_C(sGk_a)
                        )
                        cute.copy(
                            tiled_copy_Gcum_gate,
                            thr_copy_Ggate.partition_S(sGk_a),
                            thr_copy_Ggate.retile(tCrGk_a),
                        )

                        sGk_b = cute.local_tile(
                            csGcum, tiler=(16, 8), coord=(my_i_k, k_block * 2 + 1)
                        )
                        tCrGk_b = tiled_mma_k2.make_fragment_C(
                            thr_mma.partition_C(sGk_b)
                        )
                        cute.copy(
                            tiled_copy_Gcum_gate,
                            thr_copy_Ggate.partition_S(sGk_b),
                            thr_copy_Ggate.retile(tCrGk_b),
                        )


                        gk_n0_0 = cute.exp2(gn_a0 - tCrGk_a[0], fastmath=True)
                        gk_n0_1 = cute.exp2(gn_a1 - tCrGk_a[1], fastmath=True)
                        gk_n0_2 = cute.exp2(gn_b0 - tCrGk_b[0], fastmath=True)
                        gk_n0_3 = cute.exp2(gn_b1 - tCrGk_b[1], fastmath=True)

                        gk_n1_0 = cute.exp2(gn_a0 - tCrGk_a[2], fastmath=True)
                        gk_n1_1 = cute.exp2(gn_a1 - tCrGk_a[3], fastmath=True)
                        gk_n1_2 = cute.exp2(gn_b0 - tCrGk_b[2], fastmath=True)
                        gk_n1_3 = cute.exp2(gn_b1 - tCrGk_b[3], fastmath=True)


                        k_n0_b0 = tCrK_n0[0].to(cutlass.Float32) * gk_n0_0
                        k_n0_b1 = tCrK_n0[1].to(cutlass.Float32) * gk_n0_1
                        k_n0_b2 = tCrK_n0[2].to(cutlass.Float32) * gk_n0_2
                        k_n0_b3 = tCrK_n0[3].to(cutlass.Float32) * gk_n0_3

                        k_n1_b0 = tCrK_n1[0].to(cutlass.Float32) * gk_n1_0
                        k_n1_b1 = tCrK_n1[1].to(cutlass.Float32) * gk_n1_1
                        k_n1_b2 = tCrK_n1[2].to(cutlass.Float32) * gk_n1_2
                        k_n1_b3 = tCrK_n1[3].to(cutlass.Float32) * gk_n1_3


                        k_n0_u32_0 = pack_bf16x2_f32(k_n0_b1, k_n0_b0)
                        k_n0_u32_1 = pack_bf16x2_f32(k_n0_b3, k_n0_b2)
                        k_n1_u32_0 = pack_bf16x2_f32(k_n1_b1, k_n1_b0)
                        k_n1_u32_1 = pack_bf16x2_f32(k_n1_b3, k_n1_b2)


                        acc_aqk_n0_0, acc_aqk_n0_1, acc_aqk_n0_2, acc_aqk_n0_3 = (
                            mma_bf16_m16n8k16(
                                qa_u32_0,
                                qa_u32_1,
                                qa_u32_2,
                                qa_u32_3,
                                k_n0_u32_0,
                                k_n0_u32_1,
                                acc_aqk_n0_0,
                                acc_aqk_n0_1,
                                acc_aqk_n0_2,
                                acc_aqk_n0_3,
                            )
                        )
                        acc_aqk_n1_0, acc_aqk_n1_1, acc_aqk_n1_2, acc_aqk_n1_3 = (
                            mma_bf16_m16n8k16(
                                qa_u32_0,
                                qa_u32_1,
                                qa_u32_2,
                                qa_u32_3,
                                k_n1_u32_0,
                                k_n1_u32_1,
                                acc_aqk_n1_0,
                                acc_aqk_n1_1,
                                acc_aqk_n1_2,
                                acc_aqk_n1_3,
                            )
                        )
                        acc_akk_n0_0, acc_akk_n0_1, acc_akk_n0_2, acc_akk_n0_3 = (
                            mma_bf16_m16n8k16(
                                ka_u32_0,
                                ka_u32_1,
                                ka_u32_2,
                                ka_u32_3,
                                k_n0_u32_0,
                                k_n0_u32_1,
                                acc_akk_n0_0,
                                acc_akk_n0_1,
                                acc_akk_n0_2,
                                acc_akk_n0_3,
                            )
                        )
                        acc_akk_n1_0, acc_akk_n1_1, acc_akk_n1_2, acc_akk_n1_3 = (
                            mma_bf16_m16n8k16(
                                ka_u32_0,
                                ka_u32_1,
                                ka_u32_2,
                                ka_u32_3,
                                k_n1_u32_0,
                                k_n1_u32_1,
                                acc_akk_n1_0,
                                acc_akk_n1_1,
                                acc_akk_n1_2,
                                acc_akk_n1_3,
                            )
                        )


                    cute.arch.mbarrier_arrive(stage_reuse_mbars + s)


                    _z16 = cutlass.BFloat16(0.0)
                    _one16 = cutlass.BFloat16(1.0)
                    if IS_VARLEN and not VARLEN_PURE:
                        if my_i_q == my_i_k:


                            _v_q00 = (acc_aqk_n0_0 * scale).to(cutlass.BFloat16)
                            if row0 < col0:
                                _v_q00 = _z16
                            csAqk[q_row_base + row0, k_row_base + col0] = _v_q00
                            _v_q01 = (acc_aqk_n0_1 * scale).to(cutlass.BFloat16)
                            if row0 < col1:
                                _v_q01 = _z16
                            csAqk[q_row_base + row0, k_row_base + col1] = _v_q01
                            _v_q02 = (acc_aqk_n0_2 * scale).to(cutlass.BFloat16)
                            if row1 < col0:
                                _v_q02 = _z16
                            csAqk[q_row_base + row1, k_row_base + col0] = _v_q02
                            _v_q03 = (acc_aqk_n0_3 * scale).to(cutlass.BFloat16)
                            if row1 < col1:
                                _v_q03 = _z16
                            csAqk[q_row_base + row1, k_row_base + col1] = _v_q03
                            _v_q04 = (acc_aqk_n1_0 * scale).to(cutlass.BFloat16)
                            if row0 < col2:
                                _v_q04 = _z16
                            csAqk[q_row_base + row0, k_row_base + col2] = _v_q04
                            _v_q05 = (acc_aqk_n1_1 * scale).to(cutlass.BFloat16)
                            if row0 < col3:
                                _v_q05 = _z16
                            csAqk[q_row_base + row0, k_row_base + col3] = _v_q05
                            _v_q06 = (acc_aqk_n1_2 * scale).to(cutlass.BFloat16)
                            if row1 < col2:
                                _v_q06 = _z16
                            csAqk[q_row_base + row1, k_row_base + col2] = _v_q06
                            _v_q07 = (acc_aqk_n1_3 * scale).to(cutlass.BFloat16)
                            if row1 < col3:
                                _v_q07 = _z16
                            csAqk[q_row_base + row1, k_row_base + col3] = _v_q07

                            _v_k00 = (acc_akk_n0_0 * beta_row0).to(cutlass.BFloat16)
                            if row0 == col0:
                                _v_k00 = _one16
                            if row0 < col0:
                                _v_k00 = _z16
                            csAkk[akk_row_base + row0, akk_col_base + col0] = _v_k00
                            _v_k01 = (acc_akk_n0_1 * beta_row0).to(cutlass.BFloat16)
                            if row0 == col1:
                                _v_k01 = _one16
                            if row0 < col1:
                                _v_k01 = _z16
                            csAkk[akk_row_base + row0, akk_col_base + col1] = _v_k01
                            _v_k02 = (acc_akk_n0_2 * beta_row1).to(cutlass.BFloat16)
                            if row1 == col0:
                                _v_k02 = _one16
                            if row1 < col0:
                                _v_k02 = _z16
                            csAkk[akk_row_base + row1, akk_col_base + col0] = _v_k02
                            _v_k03 = (acc_akk_n0_3 * beta_row1).to(cutlass.BFloat16)
                            if row1 == col1:
                                _v_k03 = _one16
                            if row1 < col1:
                                _v_k03 = _z16
                            csAkk[akk_row_base + row1, akk_col_base + col1] = _v_k03
                            _v_k04 = (acc_akk_n1_0 * beta_row0).to(cutlass.BFloat16)
                            if row0 == col2:
                                _v_k04 = _one16
                            if row0 < col2:
                                _v_k04 = _z16
                            csAkk[akk_row_base + row0, akk_col_base + col2] = _v_k04
                            _v_k05 = (acc_akk_n1_1 * beta_row0).to(cutlass.BFloat16)
                            if row0 == col3:
                                _v_k05 = _one16
                            if row0 < col3:
                                _v_k05 = _z16
                            csAkk[akk_row_base + row0, akk_col_base + col3] = _v_k05
                            _v_k06 = (acc_akk_n1_2 * beta_row1).to(cutlass.BFloat16)
                            if row1 == col2:
                                _v_k06 = _one16
                            if row1 < col2:
                                _v_k06 = _z16
                            csAkk[akk_row_base + row1, akk_col_base + col2] = _v_k06
                            _v_k07 = (acc_akk_n1_3 * beta_row1).to(cutlass.BFloat16)
                            if row1 == col3:
                                _v_k07 = _one16
                            if row1 < col3:
                                _v_k07 = _z16
                            csAkk[akk_row_base + row1, akk_col_base + col3] = _v_k07
                        else:

                            csAqk[q_row_base + row0, k_row_base + col0] = (
                                acc_aqk_n0_0 * scale
                            ).to(cutlass.BFloat16)
                            csAqk[q_row_base + row0, k_row_base + col1] = (
                                acc_aqk_n0_1 * scale
                            ).to(cutlass.BFloat16)
                            csAqk[q_row_base + row1, k_row_base + col0] = (
                                acc_aqk_n0_2 * scale
                            ).to(cutlass.BFloat16)
                            csAqk[q_row_base + row1, k_row_base + col1] = (
                                acc_aqk_n0_3 * scale
                            ).to(cutlass.BFloat16)
                            csAqk[q_row_base + row0, k_row_base + col2] = (
                                acc_aqk_n1_0 * scale
                            ).to(cutlass.BFloat16)
                            csAqk[q_row_base + row0, k_row_base + col3] = (
                                acc_aqk_n1_1 * scale
                            ).to(cutlass.BFloat16)
                            csAqk[q_row_base + row1, k_row_base + col2] = (
                                acc_aqk_n1_2 * scale
                            ).to(cutlass.BFloat16)
                            csAqk[q_row_base + row1, k_row_base + col3] = (
                                acc_aqk_n1_3 * scale
                            ).to(cutlass.BFloat16)
                            csAkk[akk_row_base + row0, akk_col_base + col0] = (
                                acc_akk_n0_0 * beta_row0
                            ).to(cutlass.BFloat16)
                            csAkk[akk_row_base + row0, akk_col_base + col1] = (
                                acc_akk_n0_1 * beta_row0
                            ).to(cutlass.BFloat16)
                            csAkk[akk_row_base + row1, akk_col_base + col0] = (
                                acc_akk_n0_2 * beta_row1
                            ).to(cutlass.BFloat16)
                            csAkk[akk_row_base + row1, akk_col_base + col1] = (
                                acc_akk_n0_3 * beta_row1
                            ).to(cutlass.BFloat16)
                            csAkk[akk_row_base + row0, akk_col_base + col2] = (
                                acc_akk_n1_0 * beta_row0
                            ).to(cutlass.BFloat16)
                            csAkk[akk_row_base + row0, akk_col_base + col3] = (
                                acc_akk_n1_1 * beta_row0
                            ).to(cutlass.BFloat16)
                            csAkk[akk_row_base + row1, akk_col_base + col2] = (
                                acc_akk_n1_2 * beta_row1
                            ).to(cutlass.BFloat16)
                            csAkk[akk_row_base + row1, akk_col_base + col3] = (
                                acc_akk_n1_3 * beta_row1
                            ).to(cutlass.BFloat16)
                    else:


                        csAqk[q_row_base + row0, k_row_base + col0] = (
                            acc_aqk_n0_0 * scale
                        ).to(cutlass.BFloat16)
                        csAqk[q_row_base + row0, k_row_base + col1] = (
                            acc_aqk_n0_1 * scale
                        ).to(cutlass.BFloat16)
                        csAqk[q_row_base + row1, k_row_base + col0] = (
                            acc_aqk_n0_2 * scale
                        ).to(cutlass.BFloat16)
                        csAqk[q_row_base + row1, k_row_base + col1] = (
                            acc_aqk_n0_3 * scale
                        ).to(cutlass.BFloat16)
                        csAqk[q_row_base + row0, k_row_base + col2] = (
                            acc_aqk_n1_0 * scale
                        ).to(cutlass.BFloat16)
                        csAqk[q_row_base + row0, k_row_base + col3] = (
                            acc_aqk_n1_1 * scale
                        ).to(cutlass.BFloat16)
                        csAqk[q_row_base + row1, k_row_base + col2] = (
                            acc_aqk_n1_2 * scale
                        ).to(cutlass.BFloat16)
                        csAqk[q_row_base + row1, k_row_base + col3] = (
                            acc_aqk_n1_3 * scale
                        ).to(cutlass.BFloat16)
                        csAkk[akk_row_base + row0, akk_col_base + col0] = (
                            acc_akk_n0_0 * beta_row0
                        ).to(cutlass.BFloat16)
                        csAkk[akk_row_base + row0, akk_col_base + col1] = (
                            acc_akk_n0_1 * beta_row0
                        ).to(cutlass.BFloat16)
                        csAkk[akk_row_base + row1, akk_col_base + col0] = (
                            acc_akk_n0_2 * beta_row1
                        ).to(cutlass.BFloat16)
                        csAkk[akk_row_base + row1, akk_col_base + col1] = (
                            acc_akk_n0_3 * beta_row1
                        ).to(cutlass.BFloat16)
                        csAkk[akk_row_base + row0, akk_col_base + col2] = (
                            acc_akk_n1_0 * beta_row0
                        ).to(cutlass.BFloat16)
                        csAkk[akk_row_base + row0, akk_col_base + col3] = (
                            acc_akk_n1_1 * beta_row0
                        ).to(cutlass.BFloat16)
                        csAkk[akk_row_base + row1, akk_col_base + col2] = (
                            acc_akk_n1_2 * beta_row1
                        ).to(cutlass.BFloat16)
                        csAkk[akk_row_base + row1, akk_col_base + col3] = (
                            acc_akk_n1_3 * beta_row1
                        ).to(cutlass.BFloat16)
                else:
                    cute.arch.mbarrier_arrive(stage_reuse_mbars + s)

                cute.arch.mbarrier_arrive(mma_done_mbars + s)


        if warp_idx >= NUM_K1_TMA_WARPS + NUM_MMA_WARPS:
            store_warp = warp_idx - (NUM_K1_TMA_WARPS + NUM_MMA_WARPS)
            for chunk_iter in cutlass.range_constexpr(CHUNKS_PER_BLOCK):
                s = chunk_iter % NUM_STAGES
                phase = chunk_iter // NUM_STAGES % 2
                chunk_idx = chunk_base + chunk_iter
                chunk_start = cutlass.Int32(0)
                st_eos = cutlass.Int32(0)
                if IS_VARLEN:
                    if chunk_idx < num_chunks:
                        _sid = cutlass.Int32(mChunkIndices[chunk_idx, 0])
                        chunk_start = (
                            cutlass.Int32(mCuSeqlens[_sid])
                            + cutlass.Int32(mChunkIndices[chunk_idx, 1]) * BT
                        )
                        st_eos = cutlass.Int32(mCuSeqlens[_sid + 1])
                else:
                    chunk_start = chunk_idx * BT

                cute.arch.mbarrier_wait(mma_done_mbars + s, phase)

                csAqk = sAqk[(None, None, s)]
                csAkk = sAkk[(None, None, s)]


                col_lo = lane_id * 2
                col_hi = col_lo + 1
                col_vec_idx = lane_id
                rAqkOut = cute.make_rmem_tensor(
                    cute.make_layout((VEC,)), cutlass.BFloat16
                )
                rAkkOut = cute.make_rmem_tensor(
                    cute.make_layout((VEC,)), cutlass.BFloat16
                )

                if IS_VARLEN and not VARLEN_PURE:


                    row_base_warp = store_warp * (BT // NUM_STORE_WARPS)
                    for ri in cutlass.range_constexpr(BT // NUM_STORE_WARPS):
                        local_row = row_base_warp + ri
                        abs_row = chunk_start + local_row
                        rAqkOut[0] = csAqk[local_row, col_lo]
                        rAqkOut[1] = csAqk[local_row, col_hi]
                        rAkkOut[0] = csAkk[local_row, col_lo]
                        rAkkOut[1] = csAkk[local_row, col_hi]
                        if abs_row < st_eos:
                            cute.autovec_copy(
                                rAqkOut, mAqk_v2[i_b, abs_row, i_h, col_vec_idx, None]
                            )
                            cute.autovec_copy(
                                rAkkOut, mAkk_v2[i_b, abs_row, i_h, col_vec_idx, None]
                            )
                else:


                    for tile_idx in cutlass.range_constexpr(NUM_TILES):
                        i_q = _TILE_IQ[tile_idx]
                        i_k = _TILE_IK[tile_idx]
                        is_diag = _TILE_IQ[tile_idx] == _TILE_IK[tile_idx]
                        gmem_aqk_row_base = chunk_start + i_q * BC
                        gmem_aqk_col_base = i_k * BC
                        gmem_akk_row_base = chunk_start + i_k * BC
                        gmem_akk_col_base = i_q * BC
                        smem_aqk_row_base = i_q * BC
                        smem_aqk_col_base = i_k * BC
                        smem_akk_row_base = i_k * BC
                        smem_akk_col_base = i_q * BC
                        for ri in cutlass.range_constexpr(BC // NUM_STORE_WARPS):
                            local_row = store_warp * (BC // NUM_STORE_WARPS) + ri
                            if lane_id < BC:
                                local_col = lane_id
                                aqk_val = csAqk[
                                    smem_aqk_row_base + local_row,
                                    smem_aqk_col_base + local_col,
                                ]
                                akk_val = csAkk[
                                    smem_akk_row_base + local_row,
                                    smem_akk_col_base + local_col,
                                ]
                                if is_diag and local_row < local_col:
                                    aqk_val = cutlass.BFloat16(0.0)
                                if is_diag and local_row < local_col:
                                    akk_val = cutlass.BFloat16(0.0)
                                if is_diag and local_row == local_col:
                                    akk_val = cutlass.BFloat16(1.0)
                                mAqk[
                                    i_b,
                                    gmem_aqk_row_base + local_row,
                                    i_h,
                                    gmem_aqk_col_base + local_col,
                                ] = aqk_val
                                mAkk[
                                    i_b,
                                    gmem_akk_row_base + local_row,
                                    i_h,
                                    gmem_akk_col_base + local_col,
                                ] = akk_val

                cute.arch.mbarrier_arrive(store_done_mbars + s)

        yield_out()


def make_host_function(
    B,
    NT,
    H,
    is_varlen=False,
    T_padded=None,
    has_bias=False,
    use_safe_gate=False,
    varlen_pure=False,
):

    _B, _NT, _H = B, NT, H
    _IS_VARLEN = 1 if is_varlen else 0
    _HAS_BIAS = 1 if has_bias else 0
    _USE_SAFE_GATE = 1 if use_safe_gate else 0
    _VARLEN_PURE = 1 if (is_varlen and varlen_pure) else 0
    if is_varlen:
        assert _B == 1, "Varlen requires B=1"
        assert T_padded is not None, "T_padded required for varlen"
        _T = T_padded
    else:
        _T = _NT * BT

    if is_varlen:
        _total_cgs_val = ((_NT + CHUNKS_PER_BLOCK - 1) // CHUNKS_PER_BLOCK) * _H
    else:
        _total_cgs_val = (_NT // CHUNKS_PER_BLOCK) * _H * _B


    _T_total = _B * _T
    s_row = _H * K_DIM
    s_col = 1
    s_h = K_DIM

    @cute.jit
    def host_fn(
        mQ,
        mK,
        mG,
        mA_log,
        mBeta,
        scale,
        mKscaled,
        mKg,
        mQscaled,
        mGkLast,
        mAqk,
        mAkk,
        mCuSeqlens,
        mChunkIndices,
        mDtBias,
        lower_bound_val,
        stream: cuda_drv.CUstream,
    ):


        view_layout_3d = cute.make_layout(
            (_T_total, K_DIM, _H),
            stride=(s_row, s_col, s_h),
        )
        mQ_view = cute.make_tensor(mQ.iterator, view_layout_3d)
        mK_view = cute.make_tensor(mK.iterator, view_layout_3d)
        mG_view = cute.make_tensor(mG.iterator, view_layout_3d)

        smem_atom_qk = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.BFloat16
        )
        qk_smem_2d = cute.tile_to_shape(smem_atom_qk, (BT, K_DIM), order=(0, 1))
        qk_smem_3d = cute.tile_to_shape(
            smem_atom_qk, (BT, K_DIM, NUM_STAGES), order=(0, 1, 2)
        )

        g_smem_2d = cute.make_layout((BT, K_DIM), stride=(K_DIM, 1))
        g_smem_3d = cute.make_layout(
            (BT, K_DIM, NUM_STAGES), stride=(K_DIM, 1, BT * K_DIM)
        )

        tma_op = cpasync.CopyBulkTensorTileG2SOp(cpasync.CtaGroup.ONE)
        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            tma_op,
            mQ_view,
            qk_smem_2d,
            cute.product_each(qk_smem_2d.shape),
            num_multicast=1,
        )
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            tma_op,
            mK_view,
            qk_smem_2d,
            cute.product_each(qk_smem_2d.shape),
            num_multicast=1,
        )
        tma_atom_G, tma_tensor_G = cpasync.make_tiled_tma_atom(
            tma_op,
            mG_view,
            g_smem_2d,
            cute.product_each(g_smem_2d.shape),
            num_multicast=1,
        )

        g_cumsum_layout = cute.make_layout(
            (BT, K_DIM, NUM_STAGES), stride=(K_STRIDE, 1, BT * K_STRIDE)
        )

        copy_atom_qk_k1 = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=32
        )
        tiled_copy_qk_k1 = cute.make_tiled_copy_tv(
            copy_atom_qk_k1,
            thr_layout=cute.make_layout((1, 32)),
            val_layout=cute.make_layout((1, 2)),
        )

        out_v2_layout = cute.make_layout(
            (_B, _T, _H, K_VEC, VEC),
            stride=(_T * _H * K_DIM, _H * K_DIM, K_DIM, VEC, 1),
        )
        mKscaled_v2 = cute.make_tensor(mKscaled.iterator, out_v2_layout)
        mQscaled_v2 = cute.make_tensor(mQscaled.iterator, out_v2_layout)
        mKg_v2 = cute.make_tensor(mKg.iterator, out_v2_layout)


        BT_VEC = BT // VEC
        akk_v2_layout = cute.make_layout(
            (_B, _T, _H, BT_VEC, VEC),
            stride=(_T * _H * BT, _H * BT, BT, VEC, 1),
        )
        mAqk_v2 = cute.make_tensor(mAqk.iterator, akk_v2_layout)
        mAkk_v2 = cute.make_tensor(mAkk.iterator, akk_v2_layout)

        gklast_v2_layout = cute.make_layout(
            (_B, _NT, _H, K_VEC, VEC),
            stride=(_NT * _H * K_DIM, _H * K_DIM, K_DIM, VEC, 1),
        )
        mGkLast_v2 = cute.make_tensor(mGkLast.iterator, gklast_v2_layout)

        mma_op = cute.nvgpu.warp.MmaF16BF16Op(
            cutlass.BFloat16, cutlass.Float32, (16, 8, 16)
        )
        tiled_mma_k2 = cute.make_tiled_mma(
            mma_op, cute.make_layout((1, 1, 1)), permutation_mnk=(16, 8, 16)
        )

        tiled_copy_mma_A = cute.make_tiled_copy_A(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), cutlass.BFloat16
            ),
            tiled_mma_k2,
        )
        tiled_copy_mma_B = cute.make_tiled_copy_B(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 2), cutlass.BFloat16
            ),
            tiled_mma_k2,
        )

        copy_atom_Gcum = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=64
        )
        tiled_copy_Gcum_norm = cute.make_tiled_copy_tv(
            copy_atom_Gcum,
            thr_layout=cute.make_layout((1, 4)),
            val_layout=cute.make_layout((1, 2)),
        )
        tiled_copy_Gcum_gate = cute.make_tiled_copy_C(copy_atom_Gcum, tiled_mma_k2)

        smem_size = (
            BT * K_DIM * 2 * 2 * NUM_STAGES
            + BT * K_DIM * 2 * NUM_STAGES
            + BT * K_STRIDE * 4 * NUM_STAGES
            + K1_ROW_GROUPS * PARTIAL_COLS * 4
            + BT * AQK_TILE_STRIDE * 2 * NUM_STAGES
            + BT * AKK_STRIDE * 2 * NUM_STAGES
            + 512
        )

        _grid_x = min(NUM_SMS, _total_cgs_val)

        fused_kernel123(
            tma_atom_Q,
            tma_tensor_Q,
            tma_atom_K,
            tma_tensor_K,
            tma_atom_G,
            tma_tensor_G,
            mA_log,
            mBeta,
            scale,
            mKscaled_v2,
            mKg_v2,
            mQscaled_v2,
            mGkLast_v2,
            mAqk,
            mAkk,
            mAqk_v2,
            mAkk_v2,
            tiled_copy_qk_k1,
            tiled_mma_k2,
            tiled_copy_mma_A,
            tiled_copy_mma_B,
            tiled_copy_Gcum_norm,
            tiled_copy_Gcum_gate,
            qk_smem_3d,
            g_smem_3d,
            g_cumsum_layout,
            _NT,
            _H,
            _B,
            mCuSeqlens,
            mChunkIndices,
            _IS_VARLEN,
            mDtBias,
            lower_bound_val,
            _HAS_BIAS,
            _USE_SAFE_GATE,
            _VARLEN_PURE,
        ).launch(
            grid=(_grid_x, 1, 1),
            block=(THREADS, 1, 1),
            smem=smem_size,
            stream=stream,
        )

    return host_fn
