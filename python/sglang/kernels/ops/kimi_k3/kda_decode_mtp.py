import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import nvvm
from cutlass.cute.nvgpu import cpasync

WARP_SIZE = 32
TILE_K = 128
KERNEL_WIDTH = 4


P1_NUM_JOBS = 3


P1_V_WARP_DIVISOR = 4


NUM_STATE_STAGES = 2


BLOCK_THREADS_NARROW = 256
BLOCK_THREADS_WIDE = 512


P2_LANES_RESIDENT = 8
P2_LANES_STREAM = 16

HEAD_DIM = TILE_K
VEC_SIZE = HEAD_DIM // WARP_SIZE

V_WEIGHT_BASE = KERNEL_WIDTH * HEAD_DIM
CONV_WEIGHT_ELEMS = 2 * V_WEIGHT_BASE


def _stream_state(*, block_threads: int, num_spec: int) -> bool:

    return num_spec == 0 and block_threads == BLOCK_THREADS_NARROW


def _p2_lanes(*, block_threads: int, num_spec: int) -> int:
    return (
        P2_LANES_STREAM
        if _stream_state(block_threads=block_threads, num_spec=num_spec)
        else P2_LANES_RESIDENT
    )


def _qk_smem(*, block_threads: int, num_spec: int) -> tuple[tuple[int, ...], int]:

    p2_lanes = _p2_lanes(block_threads=block_threads, num_spec=num_spec)
    p2_vec = TILE_K // p2_lanes
    p2_qk_row = p2_vec + 4
    if _stream_state(block_threads=block_threads, num_spec=num_spec):
        return (p2_lanes * p2_qk_row, p2_qk_row, 4, 1), p2_lanes * p2_qk_row
    return (TILE_K, 1, 4 * p2_lanes, p2_lanes), TILE_K


def _state_tile_v(*, block_threads: int, num_spec: int) -> int:

    p2_rows_lane = WARP_SIZE // _p2_lanes(
        block_threads=block_threads, num_spec=num_spec
    )
    min_rows = (block_threads // WARP_SIZE) * p2_rows_lane
    if _stream_state(block_threads=block_threads, num_spec=num_spec):


        return max(min_rows, 32)
    return max(min_rows, HEAD_DIM // NUM_STATE_STAGES)


def _issue_state_tile(
    state_g2s_copy: cute.TiledCopy,
    thr_state_copy,
    gStateTiles: cute.Tensor,
    sState: cute.Tensor,
    i_v: int,
    num_stages: int,
) -> None:

    cute.copy(
        state_g2s_copy,
        thr_state_copy.partition_S(gStateTiles[(None, None, i_v)]),
        thr_state_copy.partition_D(sState[(None, None, i_v % num_stages)]),
    )
    cute.arch.cp_async_commit_group()


@cute.kernel
def kda_decode_mtp_kernel(
    state_g2s_copy: cute.TiledCopy,
    h0: cute.Tensor,
    x_q: cute.Tensor,
    x_k: cute.Tensor,
    x_v: cute.Tensor,
    w_q: cute.Tensor,
    w_k: cute.Tensor,
    w_v: cute.Tensor,
    cs_q: cute.Tensor,
    cs_k: cute.Tensor,
    cs_v: cute.Tensor,
    A_log: cute.Tensor,
    g: cute.Tensor,
    dt_bias: cute.Tensor,
    beta: cute.Tensor,
    o: cute.Tensor,
    ht: cute.Tensor,
    intermediate_state_indices: cute.Tensor,
    intermediate_conv_q: cute.Tensor,
    intermediate_conv_k: cute.Tensor,
    intermediate_conv_v: cute.Tensor,
    ring_rawv: cute.Tensor,
    ring_rawk: cute.Tensor,
    ring_g: cute.Tensor,
    ring_beta: cute.Tensor,
    onorm_g: cute.Tensor,
    onorm_weight: cute.Tensor,
    smem_qk_layout: cute.Layout,
    smem_state_layout: cute.Layout,
    ssm_state_indices: cute.Tensor,
    cu_seqlens: cute.Tensor,
    scale: cutlass.Constexpr[float],
    NUM_SPEC: cutlass.Constexpr[int],
    BLOCK_THREADS: cutlass.Constexpr[int],
    P2_LANES_K: cutlass.Constexpr[int],
    lower_bound: cutlass.Constexpr[float],
    CACHE_RING: cutlass.Constexpr[bool],
    APPLY_ONORM: cutlass.Constexpr[bool],
    onorm_eps: cutlass.Constexpr[float],
):

    tidx, _, _ = cute.arch.thread_idx()
    in_warp_tid = tidx % WARP_SIZE
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    i_hv, i_n, _ = cute.arch.block_idx()
    head_off = i_hv * HEAD_DIM
    T_LOOP = 1 + NUM_SPEC


    NUM_WARPS = BLOCK_THREADS // WARP_SIZE
    P1_V_WARPS = max(1, NUM_WARPS // P1_V_WARP_DIVISOR)
    P1_JOB_WARPS = (NUM_WARPS - P1_V_WARPS) // P1_NUM_JOBS
    P1_QKG_WARPS = P1_NUM_JOBS * P1_JOB_WARPS
    V_CH_PER_THREAD = TILE_K // (P1_V_WARPS * WARP_SIZE)
    P2_ROWS_LANE = WARP_SIZE // P2_LANES_K
    P2_VEC = TILE_K // P2_LANES_K
    TILE_V = _state_tile_v(block_threads=BLOCK_THREADS, num_spec=NUM_SPEC)
    NUM_V_TILES = HEAD_DIM // TILE_V
    STATE_STAGES = min(NUM_STATE_STAGES, NUM_V_TILES)
    NUM_V_ROWS = TILE_V // NUM_WARPS
    P2_BATCHES = NUM_V_ROWS // P2_ROWS_LANE


    STREAM_STATE = _stream_state(block_threads=BLOCK_THREADS, num_spec=NUM_SPEC)
    TILES_PER_PASS = 1 if STREAM_STATE else NUM_V_TILES
    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
    sK = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
    sG = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T_LOOP,)), 16)
    sVall = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((T_LOOP * HEAD_DIM,)), 16
    )
    sConvW = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((CONV_WEIGHT_ELEMS,)),
        16,
    )
    sState = smem.allocate_tensor(cutlass.Float32, smem_state_layout, 16)
    if cutlass.const_expr(APPLY_ONORM):
        sOall = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((T_LOOP * HEAD_DIM,)), 16
        )
    else:


        sOall = sVall
    r_q = cute.make_rmem_tensor(
        cute.make_layout((P2_VEC,), stride=(1,)), cutlass.Float32
    )
    r_k = cute.make_rmem_tensor(
        cute.make_layout((P2_VEC,), stride=(1,)), cutlass.Float32
    )
    r_decay = cute.make_rmem_tensor(
        cute.make_layout((P2_VEC,), stride=(1,)), cutlass.Float32
    )


    R_STATE_ELEMS = max(
        TILES_PER_PASS * P2_BATCHES * P2_VEC, 2 * (KERNEL_WIDTH - 1) * VEC_SIZE
    )
    r_state = cute.make_rmem_tensor(
        cute.make_layout((R_STATE_ELEMS,), stride=(1,)), cutlass.Float32
    )
    r_wq = cute.make_rmem_tensor(
        cute.make_layout((KERNEL_WIDTH * VEC_SIZE,), stride=(1,)), cutlass.Float32
    )


    r_w4 = cute.make_rmem_tensor(
        cute.make_layout((KERNEL_WIDTH,), stride=(1,)), cutlass.Float32
    )
    if cutlass.const_expr(STREAM_STATE):
        r_v4q = cute.make_rmem_tensor(
            cute.make_layout((4,), stride=(1,)), cutlass.Float32
        )
        r_v4k = cute.make_rmem_tensor(
            cute.make_layout((4,), stride=(1,)), cutlass.Float32
        )
        r_v4g = cute.make_rmem_tensor(
            cute.make_layout((4,), stride=(1,)), cutlass.Float32
        )

    slot = ssm_state_indices[i_n]

    if slot < 0:
        cute.arch.griddepcontrol_wait()
        pad_bos = cu_seqlens[i_n]
        for i_t in cutlass.range_constexpr(T_LOOP):
            if tidx < HEAD_DIM:
                o[0, pad_bos + i_t, i_hv, tidx] = cutlass.BFloat16(0.0)
        cute.arch.griddepcontrol_launch_dependents()


        nvvm.exit()


    slot = cutlass.Int64(slot)


    p1_job = warp_idx % P1_NUM_JOBS
    p1_par = warp_idx // P1_NUM_JOBS


    if warp_idx < P1_QKG_WARPS:
        if p1_job == 0:
            for i in range(VEC_SIZE):
                k_idx = i * 32 + in_warp_tid
                for w in range(KERNEL_WIDTH - 1):
                    r_state[w * VEC_SIZE + i] = cutlass.Float32(
                        cs_q[slot, head_off + k_idx, w]
                    )
        elif p1_job == 1:
            for i in range(VEC_SIZE):
                k_idx = i * 32 + in_warp_tid
                for w in range(KERNEL_WIDTH - 1):
                    r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + w * VEC_SIZE + i] = (
                        cutlass.Float32(cs_k[slot, head_off + k_idx, w])
                    )

    if tidx < HEAD_DIM:
        cute.autovec_copy(w_k[(head_off + tidx, None)], r_w4)
        for w in range(KERNEL_WIDTH):
            sConvW[w * HEAD_DIM + tidx] = r_w4[w]
    if tidx < HEAD_DIM:
        cute.autovec_copy(w_v[(head_off + tidx, None)], r_w4)
        for w in range(KERNEL_WIDTH):
            sConvW[V_WEIGHT_BASE + w * HEAD_DIM + tidx] = r_w4[w]
    if warp_idx < P1_QKG_WARPS and warp_idx % P1_NUM_JOBS == 0:
        for i in range(VEC_SIZE):
            cute.autovec_copy(w_q[(head_off + i * 32 + in_warp_tid, None)], r_w4)
            for w in range(KERNEL_WIDTH):
                r_wq[w * VEC_SIZE + i] = r_w4[w]


    gState = h0[(slot, i_hv, None, None)]
    gStateTiles = cute.local_tile(gState, (TILE_V, TILE_K), (None, 0))
    thr_state_copy = state_g2s_copy.get_slice(tidx)
    for i_v in cutlass.range_constexpr(STATE_STAGES):
        _issue_state_tile(
            state_g2s_copy, thr_state_copy, gStateTiles, sState, i_v, STATE_STAGES
        )

    cute.arch.griddepcontrol_wait()

    bos = cu_seqlens[i_n]
    eos = cu_seqlens[i_n + 1]
    n_tok = eos - bos
    scratch_row = intermediate_state_indices[i_n]
    r_exp_A = cutlass.Float32(0.0)

    cute.arch.barrier()

    if warp_idx < P1_QKG_WARPS:
        if p1_job == 2:
            r_exp_A = cute.math.exp(cutlass.Float32(A_log[i_hv]), fastmath=True)


        if p1_job == 0:
            for _pi in cutlass.range(cutlass.min(p1_par, n_tok)):
                for i in range(VEC_SIZE):
                    _xn = cutlass.Float32(x_q[0, bos + _pi, i_hv, i * 32 + in_warp_tid])
                    r_state[0 * VEC_SIZE + i] = r_state[1 * VEC_SIZE + i]
                    r_state[1 * VEC_SIZE + i] = r_state[2 * VEC_SIZE + i]
                    r_state[2 * VEC_SIZE + i] = _xn
        elif p1_job == 1:
            for _pi in cutlass.range(cutlass.min(p1_par, n_tok)):
                for i in range(VEC_SIZE):
                    _xn = cutlass.Float32(x_k[0, bos + _pi, i_hv, i * 32 + in_warp_tid])
                    r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 0 * VEC_SIZE + i] = r_state[
                        (KERNEL_WIDTH - 1) * VEC_SIZE + 1 * VEC_SIZE + i
                    ]
                    r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 1 * VEC_SIZE + i] = r_state[
                        (KERNEL_WIDTH - 1) * VEC_SIZE + 2 * VEC_SIZE + i
                    ]
                    r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 2 * VEC_SIZE + i] = _xn
        for i_t in cutlass.range(p1_par, T_LOOP, P1_JOB_WARPS):
            token = bos + cutlass.min(i_t, n_tok - 1)
            if p1_job == 0:
                for i_pair in range(VEC_SIZE // 2):
                    i0 = i_pair * 2
                    i1 = i_pair * 2 + 1
                    k_idx0 = i0 * 32 + in_warp_tid
                    k_idx1 = i1 * 32 + in_warp_tid
                    r_conv_0 = 0.0
                    r_conv_1 = 0.0
                    for w in range(KERNEL_WIDTH - 1):
                        r_conv_0 += r_state[w * VEC_SIZE + i0] * r_wq[w * VEC_SIZE + i0]
                        r_conv_1 += r_state[w * VEC_SIZE + i1] * r_wq[w * VEC_SIZE + i1]
                    r_xq_0 = cutlass.Float32(x_q[0, token, i_hv, k_idx0])
                    r_xq_1 = cutlass.Float32(x_q[0, token, i_hv, k_idx1])
                    _cwq_last_0 = r_wq[(KERNEL_WIDTH - 1) * VEC_SIZE + i0]
                    _cwq_last_1 = r_wq[(KERNEL_WIDTH - 1) * VEC_SIZE + i1]
                    r_conv_0 += r_xq_0 * _cwq_last_0
                    r_conv_1 += r_xq_1 * _cwq_last_1
                    e0 = cute.math.exp(-r_conv_0, fastmath=True)
                    e1 = cute.math.exp(-r_conv_1, fastmath=True)
                    sig_0 = cute.arch.rcp_approx(cutlass.Float32(1.0) + e0)
                    sig_1 = cute.arch.rcp_approx(cutlass.Float32(1.0) + e1)
                    r_q[i0] = r_conv_0 * sig_0
                    r_q[i1] = r_conv_1 * sig_1
                    r_state[0 * VEC_SIZE + i0] = r_state[1 * VEC_SIZE + i0]
                    r_state[0 * VEC_SIZE + i1] = r_state[1 * VEC_SIZE + i1]
                    r_state[1 * VEC_SIZE + i0] = r_state[2 * VEC_SIZE + i0]
                    r_state[1 * VEC_SIZE + i1] = r_state[2 * VEC_SIZE + i1]
                    r_state[2 * VEC_SIZE + i0] = r_xq_0
                    r_state[2 * VEC_SIZE + i1] = r_xq_1
                sum_q = 0.0
                for i in range(VEC_SIZE):
                    sum_q += r_q[i] * r_q[i]
                for offset in [16, 8, 4, 2, 1]:
                    sum_q += cute.arch.shuffle_sync_bfly(
                        sum_q, offset=offset, mask=-1, mask_and_clamp=31
                    )
                rnorm_q_scaled = cute.math.rsqrt(sum_q + 1e-06, fastmath=True) * scale
                for i in range(VEC_SIZE):
                    r_q[i] = r_q[i] * rnorm_q_scaled
                for i in range(VEC_SIZE):
                    k_idx = i * WARP_SIZE + in_warp_tid
                    qk_grp = k_idx % P2_LANES_K
                    qk_j = k_idx // P2_LANES_K
                    sQ[
                        i_t,
                        qk_grp,
                        qk_j // 4,
                        qk_j % 4,
                    ] = r_q[i]
            elif p1_job == 1:
                r_b_raw = cutlass.Float32(0.0)
                if in_warp_tid == 0:
                    r_b_raw = cutlass.Float32(beta[0, token, i_hv])
                for i in range(VEC_SIZE):
                    k_idx = i * 32 + in_warp_tid
                    r_conv = (
                        r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 0 * VEC_SIZE + i]
                        * sConvW[0 * HEAD_DIM + i * 32 + in_warp_tid]
                    )
                    r_conv += (
                        r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 1 * VEC_SIZE + i]
                        * sConvW[1 * HEAD_DIM + i * 32 + in_warp_tid]
                    )
                    r_conv += (
                        r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 2 * VEC_SIZE + i]
                        * sConvW[2 * HEAD_DIM + i * 32 + in_warp_tid]
                    )
                    r_xk = cutlass.Float32(x_k[0, token, i_hv, k_idx])
                    r_conv += (
                        r_xk
                        * sConvW[(KERNEL_WIDTH - 1) * HEAD_DIM + i * 32 + in_warp_tid]
                    )
                    r_conv = r_conv * cute.arch.rcp_approx(
                        cutlass.Float32(1.0) + cute.math.exp(-r_conv, fastmath=True)
                    )
                    r_k[i] = r_conv
                    if cutlass.const_expr(CACHE_RING):
                        ring_rawk[slot, i_hv, i_t, k_idx] = cutlass.BFloat16(r_conv)
                    r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 0 * VEC_SIZE + i] = r_state[
                        (KERNEL_WIDTH - 1) * VEC_SIZE + 1 * VEC_SIZE + i
                    ]
                    r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 1 * VEC_SIZE + i] = r_state[
                        (KERNEL_WIDTH - 1) * VEC_SIZE + 2 * VEC_SIZE + i
                    ]
                    r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 2 * VEC_SIZE + i] = r_xk
                sum_k = 0.0
                for i in range(VEC_SIZE):
                    sum_k += r_k[i] * r_k[i]
                for offset in [16, 8, 4, 2, 1]:
                    sum_k += cute.arch.shuffle_sync_bfly(
                        sum_k, offset=offset, mask=-1, mask_and_clamp=31
                    )
                rnorm_k = cute.math.rsqrt(sum_k + 1e-06, fastmath=True)
                for i in range(VEC_SIZE):
                    r_k[i] = r_k[i] * rnorm_k
                for i in range(VEC_SIZE):
                    k_idx = i * WARP_SIZE + in_warp_tid
                    qk_grp = k_idx % P2_LANES_K
                    qk_j = k_idx // P2_LANES_K
                    sK[
                        i_t,
                        qk_grp,
                        qk_j // 4,
                        qk_j % 4,
                    ] = r_k[i]
                if in_warp_tid == 0:
                    sBeta[i_t] = cute.arch.rcp_approx(
                        cutlass.Float32(1.0) + cute.math.exp(-r_b_raw, fastmath=True)
                    )
                    if cutlass.const_expr(CACHE_RING):
                        ring_beta[slot, i_hv, i_t] = sBeta[i_t]
            else:
                for i in range(VEC_SIZE):
                    k_idx = i * 32 + in_warp_tid
                    r_g_raw = cutlass.Float32(g[0, token, i_hv, k_idx])
                    r_g_raw = r_g_raw + cutlass.Float32(
                        dt_bias[i_hv * HEAD_DIM + k_idx]
                    )
                    exp_A_x = r_exp_A * r_g_raw
                    sigmoid_val = cute.arch.rcp_approx(
                        cutlass.Float32(1.0) + cute.math.exp(-exp_A_x, fastmath=True)
                    )
                    r_gk = lower_bound * sigmoid_val
                    qk_grp = k_idx % P2_LANES_K
                    qk_j = k_idx // P2_LANES_K
                    sG[
                        i_t,
                        qk_grp,
                        qk_j // 4,
                        qk_j % 4,
                    ] = cute.math.exp(r_gk, fastmath=True)
                    if cutlass.const_expr(CACHE_RING):
                        ring_g[slot, i_hv, i_t, k_idx] = r_gk
            if p1_job == 0:
                for i in range(VEC_SIZE):
                    k_idx = i * 32 + in_warp_tid
                    for w in range(KERNEL_WIDTH - 1):
                        intermediate_conv_q[scratch_row, i_t, head_off + k_idx, w] = (
                            cutlass.BFloat16(r_state[w * VEC_SIZE + i])
                        )
            elif p1_job == 1:
                for i in range(VEC_SIZE):
                    k_idx = i * 32 + in_warp_tid
                    for w in range(KERNEL_WIDTH - 1):
                        intermediate_conv_k[scratch_row, i_t, head_off + k_idx, w] = (
                            cutlass.BFloat16(
                                r_state[
                                    (KERNEL_WIDTH - 1) * VEC_SIZE + w * VEC_SIZE + i
                                ]
                            )
                        )


            if p1_job == 0:
                for _a in range(P1_JOB_WARPS - 1):
                    _nx = bos + cutlass.min(i_t + 1 + _a, T_LOOP - 1)
                    for i in range(VEC_SIZE):
                        _xn = cutlass.Float32(x_q[0, _nx, i_hv, i * 32 + in_warp_tid])
                        r_state[0 * VEC_SIZE + i] = r_state[1 * VEC_SIZE + i]
                        r_state[1 * VEC_SIZE + i] = r_state[2 * VEC_SIZE + i]
                        r_state[2 * VEC_SIZE + i] = _xn
            elif p1_job == 1:
                for _a in range(P1_JOB_WARPS - 1):
                    _nx = bos + cutlass.min(i_t + 1 + _a, T_LOOP - 1)
                    for i in range(VEC_SIZE):
                        _xn = cutlass.Float32(x_k[0, _nx, i_hv, i * 32 + in_warp_tid])
                        r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 0 * VEC_SIZE + i] = (
                            r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 1 * VEC_SIZE + i]
                        )
                        r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 1 * VEC_SIZE + i] = (
                            r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 2 * VEC_SIZE + i]
                        )
                        r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 2 * VEC_SIZE + i] = _xn
    else:
        for _c in range(V_CH_PER_THREAD):
            _v_idx = (tidx - P1_QKG_WARPS * 32) + _c * (HEAD_DIM // V_CH_PER_THREAD)
            _csv0 = cutlass.Float32(cs_v[slot, head_off + _v_idx, 0])
            _csv1 = cutlass.Float32(cs_v[slot, head_off + _v_idx, 1])
            _csv2 = cutlass.Float32(cs_v[slot, head_off + _v_idx, 2])
            _wv = [
                sConvW[V_WEIGHT_BASE + w * HEAD_DIM + _v_idx]
                for w in range(KERNEL_WIDTH)
            ]

            _win = [_csv0, _csv1, _csv2]
            for _t in cutlass.range_constexpr(T_LOOP):
                _win.append(
                    cutlass.Float32(
                        x_v[0, bos + cutlass.min(_t, n_tok - 1), i_hv, _v_idx]
                    )
                )
            for _t in cutlass.range_constexpr(T_LOOP):
                _vconv = _win[_t] * _wv[0]
                for _w in cutlass.range_constexpr(1, KERNEL_WIDTH):
                    _vconv += _win[_t + _w] * _wv[_w]
                _vconv = _vconv * cute.arch.rcp_approx(
                    cutlass.Float32(1.0) + cute.math.exp(-_vconv, fastmath=True)
                )
                sVall[_t * HEAD_DIM + _v_idx] = _vconv
                if cutlass.const_expr(CACHE_RING):
                    ring_rawv[slot, i_hv, _t, _v_idx] = cutlass.BFloat16(_vconv)
                for _w in cutlass.range_constexpr(KERNEL_WIDTH - 1):
                    intermediate_conv_v[scratch_row, _t, head_off + _v_idx, _w] = (
                        cutlass.BFloat16(_win[_t + 1 + _w])
                    )

    k_grp = in_warp_tid % P2_LANES_K
    row_grp = in_warp_tid // P2_LANES_K

    staged = cutlass.const_expr(STATE_STAGES < NUM_V_TILES)
    if cutlass.const_expr(not staged):
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
    r_beta_val = cutlass.Float32(0.0)
    for i_p in cutlass.range_constexpr(NUM_V_TILES // TILES_PER_PASS):
        for i_l in cutlass.range_constexpr(TILES_PER_PASS):
            i_v = i_p * TILES_PER_PASS + i_l
            if cutlass.const_expr(staged):
                cute.arch.cp_async_wait_group(
                    min(STATE_STAGES + i_v, NUM_V_TILES) - 1 - i_v
                )
                cute.arch.barrier()
            for b in range(P2_BATCHES):
                _st = (i_l * P2_BATCHES + b) * P2_VEC
                _row = warp_idx * NUM_V_ROWS + b * P2_ROWS_LANE + row_grp
                for j in range(P2_VEC):
                    r_state[_st + j] = cutlass.Float32(
                        sState[
                            _row,
                            j * P2_LANES_K + k_grp,
                            i_v % STATE_STAGES,
                        ]
                    )
            if cutlass.const_expr(staged) and i_v + STATE_STAGES < NUM_V_TILES:
                cute.arch.barrier()
                _issue_state_tile(
                    state_g2s_copy,
                    thr_state_copy,
                    gStateTiles,
                    sState,
                    i_v + STATE_STAGES,
                    STATE_STAGES,
                )
        if cutlass.const_expr(i_p == NUM_V_TILES // TILES_PER_PASS - 1):

            cute.arch.griddepcontrol_launch_dependents()

        for i_t in cutlass.range(T_LOOP):
            if cutlass.const_expr(STREAM_STATE):


                if cutlass.const_expr(i_p == 0):
                    r_beta_val = sBeta[i_t]


                    for jv in range(P2_VEC // 4):
                        cute.autovec_copy(sQ[(i_t, k_grp, jv, None)], r_v4q)
                        cute.autovec_copy(sK[(i_t, k_grp, jv, None)], r_v4k)
                        cute.autovec_copy(sG[(i_t, k_grp, jv, None)], r_v4g)
                        for c in range(4):
                            r_q[jv * 4 + c] = r_v4q[c]
                            r_k[jv * 4 + c] = r_v4k[c]
                            r_decay[jv * 4 + c] = r_v4g[c]
            else:


                r_beta_val = sBeta[i_t]
                for j in range(P2_VEC):
                    r_q[j] = sQ[i_t, k_grp, j // 4, j % 4]
                    r_k[j] = sK[i_t, k_grp, j // 4, j % 4]
                    r_decay[j] = sG[i_t, k_grp, j // 4, j % 4]
            for i_l in range(TILES_PER_PASS):
                v_base = (i_p * TILES_PER_PASS + i_l) * TILE_V
                for b in range(P2_BATCHES):
                    _st = (i_l * P2_BATCHES + b) * P2_VEC
                    v_row = warp_idx * NUM_V_ROWS + b * P2_ROWS_LANE + row_grp

                    r_v = sVall[i_t * HEAD_DIM + v_base + v_row]
                    shk_1 = 0.0
                    shk_2 = 0.0
                    for jp in range(P2_VEC // 2):
                        _p = jp * 2
                        r_state[_st + _p], r_state[_st + _p + 1] = (
                            cute.arch.mul_packed_f32x2(
                                (r_decay[_p], r_decay[_p + 1]),
                                (r_state[_st + _p], r_state[_st + _p + 1]),
                            )
                        )
                        shk_1, shk_2 = cute.arch.fma_packed_f32x2(
                            src_a=(r_state[_st + _p], r_state[_st + _p + 1]),
                            src_b=(r_k[_p], r_k[_p + 1]),
                            src_c=(shk_1, shk_2),
                        )
                    shk = shk_1 + shk_2
                    for offset in [16, 8, 4, 2, 1]:
                        if cutlass.const_expr(offset < P2_LANES_K):
                            shk += cute.arch.shuffle_sync_bfly(
                                shk, offset=offset, mask=-1, mask_and_clamp=31
                            )
                    vnb = (r_v - shk) * r_beta_val
                    shq_1 = 0.0
                    shq_2 = 0.0
                    for jp in range(P2_VEC // 2):
                        _p = jp * 2
                        r_state[_st + _p], r_state[_st + _p + 1] = (
                            cute.arch.fma_packed_f32x2(
                                src_a=(vnb, vnb),
                                src_b=(r_k[_p], r_k[_p + 1]),
                                src_c=(r_state[_st + _p], r_state[_st + _p + 1]),
                            )
                        )
                        shq_1, shq_2 = cute.arch.fma_packed_f32x2(
                            src_a=(r_state[_st + _p], r_state[_st + _p + 1]),
                            src_b=(r_q[_p], r_q[_p + 1]),
                            src_c=(shq_1, shq_2),
                        )
                    shq = shq_1 + shq_2
                    for offset in [16, 8, 4, 2, 1]:
                        if cutlass.const_expr(offset < P2_LANES_K):
                            shq += cute.arch.shuffle_sync_bfly(
                                shq, offset=offset, mask=-1, mask_and_clamp=31
                            )
                    if k_grp == 0 and i_t < n_tok:
                        if cutlass.const_expr(APPLY_ONORM):
                            sOall[i_t * HEAD_DIM + v_base + v_row] = shq
                        else:
                            o[0, bos + i_t, i_hv, v_base + v_row] = cutlass.BFloat16(
                                shq
                            )
                if cutlass.const_expr(not CACHE_RING):
                    for b in range(P2_BATCHES):
                        _st = (i_l * P2_BATCHES + b) * P2_VEC
                        v_row = warp_idx * NUM_V_ROWS + b * P2_ROWS_LANE + row_grp
                        for j in range(P2_VEC):
                            ht[
                                scratch_row,
                                i_t,
                                i_hv,
                                v_base + v_row,
                                j * P2_LANES_K + k_grp,
                            ] = r_state[_st + j]

    if cutlass.const_expr(APPLY_ONORM):
        cute.arch.barrier()
        for i_t in cutlass.range(warp_idx, T_LOOP, NUM_WARPS):
            sumsq = cutlass.Float32(0.0)
            for i in range(VEC_SIZE):
                _o = sOall[i_t * HEAD_DIM + i * 32 + in_warp_tid]
                sumsq += _o * _o
            for offset in [16, 8, 4, 2, 1]:
                sumsq += cute.arch.shuffle_sync_bfly(
                    sumsq, offset=offset, mask=-1, mask_and_clamp=31
                )
            rms = cute.math.rsqrt(
                sumsq / cutlass.Float32(HEAD_DIM) + onorm_eps, fastmath=True
            )
            for i in range(VEC_SIZE):
                v_idx = i * 32 + in_warp_tid
                raw_o = sOall[i_t * HEAD_DIM + v_idx]
                _tok = bos + cutlass.min(i_t, n_tok - 1)
                gate_raw = cutlass.Float32(onorm_g[0, _tok, i_hv, v_idx])
                gate = cute.arch.rcp_approx(
                    cutlass.Float32(1.0) + cute.math.exp(-gate_raw, fastmath=True)
                )
                if i_t < n_tok:
                    o[0, bos + i_t, i_hv, v_idx] = cutlass.BFloat16(
                        raw_o * rms * cutlass.Float32(onorm_weight[v_idx]) * gate
                    )


@cute.jit
def _run_kda_decode_mtp_dspark(
    h0: cute.Tensor,
    x_q: cute.Tensor,
    x_k: cute.Tensor,
    x_v: cute.Tensor,
    w_q: cute.Tensor,
    w_k: cute.Tensor,
    w_v: cute.Tensor,
    cs_q: cute.Tensor,
    cs_k: cute.Tensor,
    cs_v: cute.Tensor,
    A_log: cute.Tensor,
    g: cute.Tensor,
    dt_bias: cute.Tensor,
    beta: cute.Tensor,
    out: cute.Tensor,
    intermediate_ssm: cute.Tensor,
    intermediate_state_indices: cute.Tensor,
    intermediate_conv_q: cute.Tensor,
    intermediate_conv_k: cute.Tensor,
    intermediate_conv_v: cute.Tensor,
    ssm_state_indices: cute.Tensor,
    cu_seqlens: cute.Tensor,
    ring_rawv: cute.Tensor,
    ring_rawk: cute.Tensor,
    ring_g: cute.Tensor,
    ring_beta: cute.Tensor,
    onorm_g: cute.Tensor,
    onorm_weight: cute.Tensor,
    scale: cutlass.Constexpr[float],
    H: cutlass.Constexpr[int],
    N: cutlass.Constexpr[int],
    NUM_SPEC: cutlass.Constexpr[int],
    BLOCK_THREADS: cutlass.Constexpr[int],
    lower_bound: cutlass.Constexpr[float],
    CACHE_RING: cutlass.Constexpr[bool],
    APPLY_ONORM: cutlass.Constexpr[bool],
    onorm_eps: cutlass.Constexpr[float],
    stream: cuda.CUstream,
):

    p2_lanes = _p2_lanes(block_threads=BLOCK_THREADS, num_spec=NUM_SPEC)
    p2_vec = TILE_K // p2_lanes

    qk_stride, qk_elems = _qk_smem(block_threads=BLOCK_THREADS, num_spec=NUM_SPEC)
    smem_qk_layout = cute.make_layout(
        (1 + NUM_SPEC, p2_lanes, p2_vec // 4, 4), stride=qk_stride
    )


    smem_state_stride = TILE_K + p2_lanes
    TILE_V = _state_tile_v(block_threads=BLOCK_THREADS, num_spec=NUM_SPEC)
    state_stages = min(NUM_STATE_STAGES, TILE_K // TILE_V)
    state_stage_elems = TILE_V * smem_state_stride
    state_smem_elems = state_stages * state_stage_elems
    smem_state_layout = cute.make_layout(
        (TILE_V, TILE_K, state_stages),
        stride=(smem_state_stride, 1, state_stage_elems),
    )
    state_cache_mode = (
        cpasync.LoadCacheMode.ALWAYS
        if NUM_SPEC == 0 and BLOCK_THREADS == BLOCK_THREADS_WIDE
        else cpasync.LoadCacheMode.GLOBAL
    )
    state_copy_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=state_cache_mode),
        cutlass.Float32,
        num_bits_per_copy=128,
    )
    state_g2s_copy = cute.make_tiled_copy_tv(
        state_copy_atom,
        thr_layout=cute.make_layout(
            (TILE_V, BLOCK_THREADS // TILE_V),
            stride=(BLOCK_THREADS // TILE_V, 1),
        ),
        val_layout=cute.make_layout((1, 4)),
    )
    t_loop = 1 + NUM_SPEC
    smem_bytes = (

        (3 * t_loop * qk_elems + t_loop + t_loop * TILE_K + 8 * TILE_K) * 4

        + state_smem_elems * 4
        + (t_loop * TILE_K * 4 if cutlass.const_expr(APPLY_ONORM) else 0)
        + 256
    )
    kda_decode_mtp_kernel(
        state_g2s_copy,
        h0,
        x_q,
        x_k,
        x_v,
        w_q,
        w_k,
        w_v,
        cs_q,
        cs_k,
        cs_v,
        A_log,
        g,
        dt_bias,
        beta,
        out,
        intermediate_ssm,
        intermediate_state_indices,
        intermediate_conv_q,
        intermediate_conv_k,
        intermediate_conv_v,
        ring_rawv,
        ring_rawk,
        ring_g,
        ring_beta,
        onorm_g,
        onorm_weight,
        smem_qk_layout,
        smem_state_layout,
        ssm_state_indices,
        cu_seqlens,
        scale,
        NUM_SPEC,
        BLOCK_THREADS,
        p2_lanes,
        lower_bound,
        CACHE_RING,
        APPLY_ONORM,
        onorm_eps,
    ).launch(
        grid=(H, N, 1),
        block=[BLOCK_THREADS, 1, 1],
        smem=smem_bytes,
        stream=stream,
        use_pdl=True,
    )


def _block_threads(*, H: int, N: int) -> int:

    import torch

    num_sms = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    return BLOCK_THREADS_WIDE if H * N <= num_sms else BLOCK_THREADS_NARROW


_DSPARK_COMPILED = {}


def _tensor_layout_key(tensor):
    return (tensor.device, tensor.dtype, tuple(tensor.shape), tuple(tensor.stride()))


def _fits_32bit_stride(tensor):
    max_offset = int(tensor.storage_offset())
    for size, stride in zip(tensor.shape, tensor.stride()):
        if abs(int(stride)) > 2**31 - 1:
            return False
        if size:
            max_offset += (int(size) - 1) * abs(int(stride))
            if max_offset > 2**31 - 1:
                return False
    return True


def _cute_tensor(tensor, *, dynamic=True):
    from cutlass.cute.runtime import from_dlpack

    if tensor.requires_grad:
        tensor = tensor.detach()
    value = from_dlpack(
        tensor,
        assumed_align=16,
        use_32bit_stride=_fits_32bit_stride(tensor),
    )
    leading_dim = next(
        (dim for dim, stride in enumerate(tensor.stride()) if stride == 1), None
    )
    if not dynamic or leading_dim is None:
        return value
    return value.mark_layout_dynamic(leading_dim)


def fused_kda_decode_mtp_dspark(
    *,
    x_q,
    x_k,
    x_v,
    w_q,
    w_k,
    w_v,
    cs_q,
    cs_k,
    cs_v,
    g,
    beta,
    A_log,
    dt_bias,
    recurrent_state,
    intermediate_ssm,
    intermediate_state_indices,
    intermediate_conv_q,
    intermediate_conv_k,
    intermediate_conv_v,
    ssm_state_indices,
    cu_seqlens,
    lower_bound,
    scale=None,
    replayssm_rawv=None,
    replayssm_rawk=None,
    replayssm_g=None,
    replayssm_beta=None,
    onorm_gate=None,
    onorm_weight=None,
    onorm_eps=None,
):

    import torch

    H = x_q.shape[2]
    N = cu_seqlens.numel() - 1
    T = x_q.shape[1]
    expected_shape = (1, T, H, TILE_K)
    if tuple(x_q.shape) != expected_shape or tuple(x_k.shape) != expected_shape:
        raise ValueError(f"expected q/k shape {expected_shape}")
    if tuple(x_v.shape) != expected_shape or tuple(g.shape) != expected_shape:
        raise ValueError(f"expected v/g shape {expected_shape}")
    if tuple(beta.shape) != (1, T, H):
        raise ValueError(f"expected beta shape {(1, T, H)}")


    if N <= 0 or T % N != 0 or T // N < 1:
        raise ValueError(
            f"DSpARK KDA MTP requires a fixed 1 + num_spec dense tokens per "
            f"request; got T={T}, N={N}"
        )
    num_spec = T // N - 1
    if recurrent_state.shape[1:] != (H, TILE_K, TILE_K):
        raise ValueError("expected recurrent state layout [pool, H, V=128, K=128]")
    if (
        recurrent_state.dtype != torch.float32
        or tuple(recurrent_state.stride()[-3:]) != (TILE_K * TILE_K, TILE_K, 1)
        or recurrent_state.stride(0) % 4 != 0
        or recurrent_state.storage_offset() % 4 != 0
    ):
        raise ValueError(
            "cp.async recurrent state requires fp32 contiguous [H, V, K] "
            "inner layout and 16-byte-aligned slot offsets"
        )
    rings = (replayssm_rawv, replayssm_rawk, replayssm_g, replayssm_beta)
    cache_ring = all(ring is not None for ring in rings)
    if any(ring is not None for ring in rings) and not cache_ring:
        raise ValueError("ReplaySSM requires all four replayssm_* rings")
    if cache_ring:
        ring_len = replayssm_rawv.shape[2]
        if (
            ring_len < 1 + num_spec
            or replayssm_rawv.shape[1:] != (H, ring_len, TILE_K)
            or replayssm_rawk.shape[1:] != (H, ring_len, TILE_K)
            or replayssm_g.shape[1:] != (H, ring_len, TILE_K)
            or replayssm_beta.shape[1:] != (H, ring_len)
        ):
            raise ValueError(
                f"expected ReplaySSM ring layouts [slots, H={H}, L>={1 + num_spec}, "
                f"{TILE_K}] / [slots, H, L]"
            )
        if (
            replayssm_rawv.dtype != torch.bfloat16
            or replayssm_rawk.dtype != torch.bfloat16
            or replayssm_g.dtype != torch.float32
            or replayssm_beta.dtype != torch.float32
        ):
            raise ValueError(
                "expected ReplaySSM ring dtypes rawv/rawk=bf16, g/beta=fp32"
            )
        if intermediate_ssm is None:


            intermediate_ssm = recurrent_state.unsqueeze(1)
    if not cache_ring and (
        intermediate_ssm.shape[1] < 1 + num_spec
        or intermediate_ssm.shape[2:5] != (H, TILE_K, TILE_K)
    ):
        raise ValueError(
            f"expected intermediate SSM layout [scratch, >={1 + num_spec}, H, "
            f"V=128, K=128]"
        )
    if scale is None:
        scale = TILE_K**-0.5
    onorm_args = (onorm_gate, onorm_weight, onorm_eps)
    apply_onorm = all(value is not None for value in onorm_args)
    if any(value is not None for value in onorm_args) and not apply_onorm:
        raise ValueError(
            "fused output norm requires onorm_gate, onorm_weight, and onorm_eps"
        )
    if apply_onorm:
        if tuple(onorm_gate.shape) != expected_shape:
            raise ValueError(f"expected output-norm gate shape {expected_shape}")
        if tuple(onorm_weight.shape) != (TILE_K,):
            raise ValueError(f"expected output-norm weight shape {(TILE_K,)}")
        if onorm_gate.dtype != torch.bfloat16 or onorm_weight.dtype != torch.float32:
            raise ValueError("expected output-norm gate=bf16 and weight=fp32")
    block_threads = _block_threads(H=H, N=N)
    out = torch.empty_like(x_v)
    args = (
        recurrent_state,
        x_q,
        x_k,
        x_v,
        w_q,
        w_k,
        w_v,
        cs_q,
        cs_k,
        cs_v,
        A_log,
        g,
        dt_bias,
        beta,
        out,
        intermediate_ssm,
        intermediate_state_indices,
        intermediate_conv_q,
        intermediate_conv_k,
        intermediate_conv_v,
        ssm_state_indices,
        cu_seqlens,

        replayssm_rawv if cache_ring else intermediate_conv_q,
        replayssm_rawk if cache_ring else intermediate_conv_q,
        replayssm_g if cache_ring else intermediate_conv_q,
        replayssm_beta if cache_ring else intermediate_conv_q[..., 0],
        onorm_gate if apply_onorm else x_v,
        onorm_weight if apply_onorm else dt_bias,
    )
    key = (
        H,
        N,
        num_spec,
        block_threads,
        cache_ring,
        apply_onorm,
        float(onorm_eps) if apply_onorm else 0.0,
        float(scale),
        float(lower_bound),
        *(_tensor_layout_key(tensor) for tensor in args),
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = _DSPARK_COMPILED.get(key)
    if compiled is None:
        cute_args = tuple(_cute_tensor(tensor, dynamic=False) for tensor in args)
        compiled = cute.compile(
            _run_kda_decode_mtp_dspark,
            *cute_args,
            scale=float(scale),
            H=H,
            N=N,
            NUM_SPEC=num_spec,
            BLOCK_THREADS=block_threads,
            lower_bound=float(lower_bound),
            CACHE_RING=cache_ring,
            APPLY_ONORM=apply_onorm,
            onorm_eps=float(onorm_eps) if apply_onorm else 0.0,
            stream=stream,
        )
        _DSPARK_COMPILED[key] = compiled
    compiled(
        *(_cute_tensor(tensor) for tensor in args),
        stream,
    )
    return out
