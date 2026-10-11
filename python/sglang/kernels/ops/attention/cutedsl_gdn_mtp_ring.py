# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack


def _mark_batch_dynamic(torch_t: torch.Tensor, *, assumed_align: int = 32):

    return from_dlpack(
        torch_t, assumed_align=assumed_align, enable_tvm_ffi=True
    ).mark_layout_dynamic()


def _mark_slot_dynamic(torch_t: torch.Tensor, *, assumed_align: int = 32):


    stride_order = tuple(range(torch_t.dim()))
    return from_dlpack(
        torch_t, assumed_align=assumed_align, enable_tvm_ffi=True
    ).mark_compact_shape_dynamic(mode=0, stride_order=stride_order, divisibility=1)


def _mark_index_dynamic(torch_t: torch.Tensor, *, assumed_align: int = 32):


    stride_order = tuple(sorted(range(torch_t.dim()), key=lambda d: -torch_t.stride(d)))
    return from_dlpack(
        torch_t, assumed_align=assumed_align, enable_tvm_ffi=True
    ).mark_compact_shape_dynamic(mode=0, stride_order=stride_order, divisibility=1)


@cute.jit
def fma_pair_mul(a1, a2, b1, b2):

    result1 = a1 * b1
    result2 = a2 * b2
    return result1, result2


@cute.jit
def fma_pair(a1, a2, b1, b2, c1, c2):

    result1 = a1 * b1 + c1
    result2 = a2 * b2 + c2
    return result1, result2


MTP_NUM_THREADS = 128
MTP_VEC_SIZE = 4


LANES_PER_ROW = 16
ELEMS_PER_LANE = 8
NUM_WARPS = 4
NUM_THREADS = NUM_WARPS * 32
NUM_GROUPS = NUM_THREADS // LANES_PER_ROW
ILP_ROWS = 4


MTP_ILP4_ROWS = 4


@cute.kernel
def gdn_decode_bf16state_mtp_ilp4_kernel(
    h0_source: cute.Tensor,
    intermediate_states: cute.Tensor,
    vec_size: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    tile_v: cutlass.Constexpr[int],
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    h0_out_indices: cute.Tensor,
    accepted_steps: cute.Tensor,
    ssm_state_indices: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
    disable_state_update: cutlass.Constexpr[bool],
    cache_intermediate_states: cutlass.Constexpr[bool],
    use_packed_fma: cutlass.Constexpr[bool],
    same_pool: cutlass.Constexpr[bool],
    disable_output: cutlass.Constexpr[bool],
    per_request_accepted_steps: cutlass.Constexpr[bool],
    per_token_pool_scatter: cutlass.Constexpr[bool],
    per_token_pool_scatter_flat: cutlass.Constexpr[bool],
    replayssm_rawv: cute.Tensor,
    replayssm_rawk: cute.Tensor,
    replayssm_g: cute.Tensor,
    replayssm_beta: cute.Tensor,
    CACHE_RING: cutlass.Constexpr[bool],
):

    tidx, _, _ = cute.arch.thread_idx()
    lane_id = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)

    threads_per_group: cutlass.Constexpr[int] = 32
    num_groups: cutlass.Constexpr[int] = 4
    group_idx = warp_idx
    lane_in_group = lane_id

    batch_idx, _, _ = cute.arch.block_idx()

    i_v = batch_idx % num_v_tiles
    tmp = batch_idx // num_v_tiles
    i_hv = tmp % HV
    i_n = tmp // HV
    i_h = i_hv // (HV // H)


    if cutlass.const_expr(per_request_accepted_steps):
        my_accepted_step = cutlass.Int32(accepted_steps[i_n])
    else:
        my_accepted_step = cutlass.Int32(T - 1)


    cache_idx = h0_indices[i_n]
    ring_slot = cache_idx
    if cutlass.const_expr(same_pool):


        write_cache_idx = cache_idx
    else:
        write_cache_idx = h0_out_indices[i_n]
        if write_cache_idx < 0:
            write_cache_idx = cutlass.Int32(0)

    r_A_log = cutlass.Float32(A_log[i_hv])
    r_dt_bias = cutlass.Float32(dt_bias[i_hv])

    if cutlass.const_expr(T > 1):
        smem = cutlass.utils.SmemAllocator()
        sQ = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((T, K), stride=(K + 8, 1)), 16
        )
        sK = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((T, K), stride=(K + 8, 1)), 16
        )
        sGB = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((T, 2), stride=(2, 1)), 16
        )

    ILP4: cutlass.Constexpr[int] = 4
    r_q = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_k = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_h = cute.make_rmem_tensor(
        cute.make_layout((ILP4, vec_size), stride=(vec_size, 1)),
        cutlass.Float32,
    )
    r_q_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_k_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_hb4_0 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_hb4_1 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_hb4_2 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_hb4_3 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_o4_bf16 = cute.make_rmem_tensor(
        cute.make_layout((ILP4,), stride=(1,)), cutlass.BFloat16
    )
    r_v4_bf16 = cute.make_rmem_tensor(
        cute.make_layout((ILP4,), stride=(1,)), cutlass.BFloat16
    )

    if cache_idx < 0:
        cache_idx = cutlass.Int32(0)

    if cache_idx >= 0:
        k_start = lane_in_group * vec_size

        if cutlass.const_expr(T > 1):
            num_precompute_passes: cutlass.Constexpr[int] = (
                T + num_groups - 1
            ) // num_groups
            for pass_idx in cutlass.range_constexpr(num_precompute_passes):
                i_t_pre = pass_idx * num_groups + group_idx
                if i_t_pre < T:


                    if cutlass.const_expr(not disable_output):
                        q_tile_pre = cute.local_tile(
                            q, (1, 1, 1, vec_size), (i_n, i_t_pre, i_h, lane_in_group)
                        )
                        cute.autovec_copy(q_tile_pre, r_q_bf16)
                    k_tile_pre = cute.local_tile(
                        k, (1, 1, 1, vec_size), (i_n, i_t_pre, i_h, lane_in_group)
                    )
                    cute.autovec_copy(k_tile_pre, r_k_bf16)

                    if cutlass.const_expr(CACHE_RING):
                        if ring_slot >= 0 and i_v == 0:
                            if i_hv % (HV // H) == 0:
                                for i in cutlass.range_constexpr(vec_size):
                                    replayssm_rawk[
                                        (ring_slot, i_h, i_t_pre, k_start + i)
                                    ] = r_k_bf16[i]
                            for i in cutlass.range_constexpr(vec_size):
                                replayssm_rawv[
                                    (ring_slot, i_hv, i_t_pre, k_start + i)
                                ] = v[(i_n, i_t_pre, i_hv, k_start + i)]

                    if cutlass.const_expr(not disable_output):
                        for i in cutlass.range_constexpr(vec_size):
                            r_q[i] = cutlass.Float32(r_q_bf16[i])
                    for i in cutlass.range_constexpr(vec_size):
                        r_k[i] = cutlass.Float32(r_k_bf16[i])

                    if cutlass.const_expr(use_qk_l2norm):
                        sum_k = cutlass.Float32(0.0)
                        for i in cutlass.range_constexpr(vec_size):
                            sum_k += r_k[i] * r_k[i]
                        for offset in [16, 8, 4, 2, 1]:
                            sum_k += cute.arch.shuffle_sync_bfly(
                                sum_k, offset=offset, mask=-1, mask_and_clamp=31
                            )
                        inv_norm_k = cute.rsqrt(sum_k + 1e-6, fastmath=True)
                        for i in cutlass.range_constexpr(vec_size):
                            r_k[i] = r_k[i] * inv_norm_k

                        if cutlass.const_expr(not disable_output):
                            sum_q = cutlass.Float32(0.0)
                            for i in cutlass.range_constexpr(vec_size):
                                sum_q += r_q[i] * r_q[i]
                            for offset in [16, 8, 4, 2, 1]:
                                sum_q += cute.arch.shuffle_sync_bfly(
                                    sum_q, offset=offset, mask=-1, mask_and_clamp=31
                                )
                            inv_norm_q_scaled = (
                                cute.rsqrt(sum_q + 1e-6, fastmath=True) * scale
                            )
                            for i in cutlass.range_constexpr(vec_size):
                                r_q[i] = r_q[i] * inv_norm_q_scaled
                    elif cutlass.const_expr(not disable_output):

                        for i in cutlass.range_constexpr(vec_size):
                            r_q[i] = r_q[i] * scale

                    if cutlass.const_expr(not disable_output):
                        for i in cutlass.range_constexpr(vec_size):
                            sQ[(i_t_pre, k_start + i)] = r_q[i]
                    for i in cutlass.range_constexpr(vec_size):
                        sK[(i_t_pre, k_start + i)] = r_k[i]

                    if cutlass.const_expr(T > 2):
                        r_a_pre = cutlass.Float32(a[i_n, i_t_pre, i_hv])
                        r_b_pre = cutlass.Float32(b[i_n, i_t_pre, i_hv])
                        x_pre = r_a_pre + r_dt_bias
                        beta_x_pre = softplus_beta * x_pre
                        exp_beta_x_pre = cute.exp(beta_x_pre, fastmath=True)
                        softplus_val_pre = (
                            cutlass.Float32(1.0) / softplus_beta
                        ) * cute.log(
                            cutlass.Float32(1.0) + exp_beta_x_pre, fastmath=True
                        )
                        use_softplus_pre = (
                            cutlass.Float32(1.0)
                            if beta_x_pre <= softplus_threshold
                            else cutlass.Float32(0.0)
                        )
                        softplus_x_pre = (
                            use_softplus_pre * softplus_val_pre
                            + (cutlass.Float32(1.0) - use_softplus_pre) * x_pre
                        )
                        r_g_value_pre = (
                            -cute.exp(r_A_log, fastmath=True) * softplus_x_pre
                        )
                        r_beta_pre = cutlass.Float32(1.0) / (
                            cutlass.Float32(1.0) + cute.exp(-r_b_pre, fastmath=True)
                        )
                        r_g_pre = cute.exp(r_g_value_pre, fastmath=True)
                        if lane_in_group == 0:
                            sGB[(i_t_pre, 0)] = r_g_pre
                            sGB[(i_t_pre, 1)] = r_beta_pre
                        if cutlass.const_expr(CACHE_RING):
                            if ring_slot >= 0 and i_v == 0 and lane_in_group == 0:
                                replayssm_g[(ring_slot, i_hv, i_t_pre)] = r_g_value_pre
                                replayssm_beta[(ring_slot, i_hv, i_t_pre)] = r_beta_pre

                cute.arch.barrier()


        h0_slot_r = h0_source[(cutlass.Int64(cache_idx), i_hv, None, None)]
        if cutlass.const_expr(same_pool):
            h0_slot_w = h0_slot_r
        else:
            h0_slot_w = h0_source[(cutlass.Int64(write_cache_idx), i_hv, None, None)]
        rows_per_group: cutlass.Constexpr[int] = tile_v // num_groups

        sum_q = cutlass.Float32(0.0)
        sum_k = cutlass.Float32(0.0)
        inv_norm_q_scaled = cutlass.Float32(1.0)
        inv_norm_k = cutlass.Float32(1.0)

        quarter_rows: cutlass.Constexpr[int] = rows_per_group // ILP4

        for row_quad in cutlass.range(quarter_rows, unroll=1, unroll_full=(T <= 1)):
            vb4 = i_v * tile_v + group_idx * rows_per_group + row_quad * ILP4
            va = vb4
            vb = vb4 + 1
            vc = vb4 + 2
            vd = vb4 + 3


            hta = cute.local_tile(h0_slot_r, (1, vec_size), (va, lane_in_group))
            htb = cute.local_tile(h0_slot_r, (1, vec_size), (vb, lane_in_group))
            htc = cute.local_tile(h0_slot_r, (1, vec_size), (vc, lane_in_group))
            htd = cute.local_tile(h0_slot_r, (1, vec_size), (vd, lane_in_group))


            if cutlass.const_expr(same_pool):
                hta_w = hta
                htb_w = htb
                htc_w = htc
                htd_w = htd
            else:
                hta_w = cute.local_tile(h0_slot_w, (1, vec_size), (va, lane_in_group))
                htb_w = cute.local_tile(h0_slot_w, (1, vec_size), (vb, lane_in_group))
                htc_w = cute.local_tile(h0_slot_w, (1, vec_size), (vc, lane_in_group))
                htd_w = cute.local_tile(h0_slot_w, (1, vec_size), (vd, lane_in_group))
            cute.autovec_copy(hta, r_hb4_0)
            cute.autovec_copy(htb, r_hb4_1)
            cute.autovec_copy(htc, r_hb4_2)
            cute.autovec_copy(htd, r_hb4_3)

            for i in cutlass.range_constexpr(vec_size):
                r_h[0, i] = cutlass.Float32(r_hb4_0[i])
                r_h[1, i] = cutlass.Float32(r_hb4_1[i])
                r_h[2, i] = cutlass.Float32(r_hb4_2[i])
                r_h[3, i] = cutlass.Float32(r_hb4_3[i])


            if cutlass.const_expr(per_request_accepted_steps):
                _loop_limit = my_accepted_step + cutlass.Int32(1)
            else:
                _loop_limit = (
                    T
                )
            for i_t in cutlass.range(
                _loop_limit,
                unroll=1,
                unroll_full=(T <= 1) and not per_request_accepted_steps,
            ):
                if cutlass.const_expr(T > 1):
                    sQ_tile = cute.local_tile(sQ, (1, vec_size), (i_t, lane_in_group))
                    sK_tile = cute.local_tile(sK, (1, vec_size), (i_t, lane_in_group))
                    cute.autovec_copy(sQ_tile, r_q)
                    cute.autovec_copy(sK_tile, r_k)
                    if cutlass.const_expr(T > 2):
                        r_g = sGB[(i_t, 0)]
                        r_beta = sGB[(i_t, 1)]
                    else:
                        r_a_val = cutlass.Float32(a[i_n, i_t, i_hv])
                        r_b_val = cutlass.Float32(b[i_n, i_t, i_hv])
                        x_val = r_a_val + r_dt_bias
                        beta_x_val = softplus_beta * x_val
                        exp_beta_x_val = cute.exp(beta_x_val, fastmath=True)
                        softplus_val_v = (
                            cutlass.Float32(1.0) / softplus_beta
                        ) * cute.log(
                            cutlass.Float32(1.0) + exp_beta_x_val, fastmath=True
                        )
                        use_softplus_v = (
                            cutlass.Float32(1.0)
                            if beta_x_val <= softplus_threshold
                            else cutlass.Float32(0.0)
                        )
                        softplus_x_v = (
                            use_softplus_v * softplus_val_v
                            + (cutlass.Float32(1.0) - use_softplus_v) * x_val
                        )
                        r_g_value_v = -cute.exp(r_A_log, fastmath=True) * softplus_x_v
                        r_beta = cutlass.Float32(1.0) / (
                            cutlass.Float32(1.0) + cute.exp(-r_b_val, fastmath=True)
                        )
                        r_g = cute.exp(r_g_value_v, fastmath=True)
                else:
                    q_tile_t = cute.local_tile(
                        q, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane_in_group)
                    )
                    k_tile_t = cute.local_tile(
                        k, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane_in_group)
                    )
                    cute.autovec_copy(q_tile_t, r_q_bf16)
                    cute.autovec_copy(k_tile_t, r_k_bf16)
                    for i in cutlass.range_constexpr(vec_size):
                        r_q[i] = cutlass.Float32(r_q_bf16[i])
                        r_k[i] = cutlass.Float32(r_k_bf16[i])
                    if cutlass.const_expr(use_qk_l2norm):
                        sum_q = cutlass.Float32(0.0)
                        sum_k = cutlass.Float32(0.0)
                        for i in cutlass.range_constexpr(vec_size):
                            sum_q += r_q[i] * r_q[i]
                            sum_k += r_k[i] * r_k[i]
                        for offset in [16, 8, 4, 2, 1]:
                            sum_q += cute.arch.shuffle_sync_bfly(
                                sum_q, offset=offset, mask=-1, mask_and_clamp=31
                            )
                            sum_k += cute.arch.shuffle_sync_bfly(
                                sum_k, offset=offset, mask=-1, mask_and_clamp=31
                            )
                        inv_norm_q_scaled = (
                            cute.rsqrt(sum_q + 1e-6, fastmath=True) * scale
                        )
                        inv_norm_k = cute.rsqrt(sum_k + 1e-6, fastmath=True)
                        for i in cutlass.range_constexpr(vec_size):
                            r_q[i] = r_q[i] * inv_norm_q_scaled
                            r_k[i] = r_k[i] * inv_norm_k
                    else:
                        for i in cutlass.range_constexpr(vec_size):
                            r_q[i] = r_q[i] * scale
                    r_a_val = cutlass.Float32(a[i_n, i_t, i_hv])
                    r_b_val = cutlass.Float32(b[i_n, i_t, i_hv])
                    x_val = r_a_val + r_dt_bias
                    beta_x_val = softplus_beta * x_val
                    exp_beta_x_val = cute.exp(beta_x_val, fastmath=True)
                    softplus_val_v = (cutlass.Float32(1.0) / softplus_beta) * cute.log(
                        cutlass.Float32(1.0) + exp_beta_x_val, fastmath=True
                    )
                    use_softplus_v = (
                        cutlass.Float32(1.0)
                        if beta_x_val <= softplus_threshold
                        else cutlass.Float32(0.0)
                    )
                    softplus_x_v = (
                        use_softplus_v * softplus_val_v
                        + (cutlass.Float32(1.0) - use_softplus_v) * x_val
                    )
                    r_g_value_v = -cute.exp(r_A_log, fastmath=True) * softplus_x_v
                    r_beta = cutlass.Float32(1.0) / (
                        cutlass.Float32(1.0) + cute.exp(-r_b_val, fastmath=True)
                    )
                    r_g = cute.exp(r_g_value_v, fastmath=True)

                sa = 0.0
                sb = 0.0
                sc = 0.0
                sd = 0.0
                sa2 = 0.0
                sb2 = 0.0
                sc2 = 0.0
                sd2 = 0.0
                for i in cutlass.range_constexpr(0, vec_size, 2):
                    r_h[0, i] = r_h[0, i] * r_g
                    r_h[0, i + 1] = r_h[0, i + 1] * r_g
                    r_h[1, i] = r_h[1, i] * r_g
                    r_h[1, i + 1] = r_h[1, i + 1] * r_g
                    r_h[2, i] = r_h[2, i] * r_g
                    r_h[2, i + 1] = r_h[2, i + 1] * r_g
                    r_h[3, i] = r_h[3, i] * r_g
                    r_h[3, i + 1] = r_h[3, i + 1] * r_g
                    if cutlass.const_expr(use_packed_fma):
                        sa, sa2 = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[0, i], r_h[0, i + 1]),
                            src_b=(r_k[i], r_k[i + 1]),
                            src_c=(sa, sa2),
                        )
                        sb, sb2 = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[1, i], r_h[1, i + 1]),
                            src_b=(r_k[i], r_k[i + 1]),
                            src_c=(sb, sb2),
                        )
                        sc, sc2 = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[2, i], r_h[2, i + 1]),
                            src_b=(r_k[i], r_k[i + 1]),
                            src_c=(sc, sc2),
                        )
                        sd, sd2 = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[3, i], r_h[3, i + 1]),
                            src_b=(r_k[i], r_k[i + 1]),
                            src_c=(sd, sd2),
                        )
                    else:
                        sa, sa2 = fma_pair(
                            r_h[0, i], r_h[0, i + 1], r_k[i], r_k[i + 1], sa, sa2
                        )
                        sb, sb2 = fma_pair(
                            r_h[1, i], r_h[1, i + 1], r_k[i], r_k[i + 1], sb, sb2
                        )
                        sc, sc2 = fma_pair(
                            r_h[2, i], r_h[2, i + 1], r_k[i], r_k[i + 1], sc, sc2
                        )
                        sd, sd2 = fma_pair(
                            r_h[3, i], r_h[3, i + 1], r_k[i], r_k[i + 1], sd, sd2
                        )
                sa = sa + sa2
                sb = sb + sb2
                sc = sc + sc2
                sd = sd + sd2

                for offset in [16, 8, 4, 2, 1]:
                    sa += cute.arch.shuffle_sync_bfly(
                        sa, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    sb += cute.arch.shuffle_sync_bfly(
                        sb, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    sc += cute.arch.shuffle_sync_bfly(
                        sc, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    sd += cute.arch.shuffle_sync_bfly(
                        sd, offset=offset, mask=-1, mask_and_clamp=31
                    )

                vt4_slice = cute.local_tile(
                    v, (1, 1, 1, ILP4), (i_n, i_t, i_hv, vb4 // ILP4)
                )
                cute.autovec_copy(vt4_slice, r_v4_bf16)
                vna = (cutlass.Float32(r_v4_bf16[0]) - sa) * r_beta
                vnb = (cutlass.Float32(r_v4_bf16[1]) - sb) * r_beta
                vnc = (cutlass.Float32(r_v4_bf16[2]) - sc) * r_beta
                vnd = (cutlass.Float32(r_v4_bf16[3]) - sd) * r_beta

                oa = 0.0
                ob = 0.0
                oc = 0.0
                od = 0.0
                oa2 = 0.0
                ob2 = 0.0
                oc2 = 0.0
                od2 = 0.0
                for i in cutlass.range_constexpr(0, vec_size, 2):
                    if cutlass.const_expr(use_packed_fma):
                        r_h[0, i], r_h[0, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(r_k[i], r_k[i + 1]),
                            src_b=(vna, vna),
                            src_c=(r_h[0, i], r_h[0, i + 1]),
                        )
                        r_h[1, i], r_h[1, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(r_k[i], r_k[i + 1]),
                            src_b=(vnb, vnb),
                            src_c=(r_h[1, i], r_h[1, i + 1]),
                        )
                        r_h[2, i], r_h[2, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(r_k[i], r_k[i + 1]),
                            src_b=(vnc, vnc),
                            src_c=(r_h[2, i], r_h[2, i + 1]),
                        )
                        r_h[3, i], r_h[3, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(r_k[i], r_k[i + 1]),
                            src_b=(vnd, vnd),
                            src_c=(r_h[3, i], r_h[3, i + 1]),
                        )
                    else:
                        r_h[0, i], r_h[0, i + 1] = fma_pair(
                            r_k[i], r_k[i + 1], vna, vna, r_h[0, i], r_h[0, i + 1]
                        )
                        r_h[1, i], r_h[1, i + 1] = fma_pair(
                            r_k[i], r_k[i + 1], vnb, vnb, r_h[1, i], r_h[1, i + 1]
                        )
                        r_h[2, i], r_h[2, i + 1] = fma_pair(
                            r_k[i], r_k[i + 1], vnc, vnc, r_h[2, i], r_h[2, i + 1]
                        )
                        r_h[3, i], r_h[3, i + 1] = fma_pair(
                            r_k[i], r_k[i + 1], vnd, vnd, r_h[3, i], r_h[3, i + 1]
                        )
                    if cutlass.const_expr(not disable_output):
                        if cutlass.const_expr(use_packed_fma):
                            oa, oa2 = cute.arch.fma_packed_f32x2(
                                src_a=(r_h[0, i], r_h[0, i + 1]),
                                src_b=(r_q[i], r_q[i + 1]),
                                src_c=(oa, oa2),
                            )
                            ob, ob2 = cute.arch.fma_packed_f32x2(
                                src_a=(r_h[1, i], r_h[1, i + 1]),
                                src_b=(r_q[i], r_q[i + 1]),
                                src_c=(ob, ob2),
                            )
                            oc, oc2 = cute.arch.fma_packed_f32x2(
                                src_a=(r_h[2, i], r_h[2, i + 1]),
                                src_b=(r_q[i], r_q[i + 1]),
                                src_c=(oc, oc2),
                            )
                            od, od2 = cute.arch.fma_packed_f32x2(
                                src_a=(r_h[3, i], r_h[3, i + 1]),
                                src_b=(r_q[i], r_q[i + 1]),
                                src_c=(od, od2),
                            )
                        else:
                            oa, oa2 = fma_pair(
                                r_h[0, i], r_h[0, i + 1], r_q[i], r_q[i + 1], oa, oa2
                            )
                            ob, ob2 = fma_pair(
                                r_h[1, i], r_h[1, i + 1], r_q[i], r_q[i + 1], ob, ob2
                            )
                            oc, oc2 = fma_pair(
                                r_h[2, i], r_h[2, i + 1], r_q[i], r_q[i + 1], oc, oc2
                            )
                            od, od2 = fma_pair(
                                r_h[3, i], r_h[3, i + 1], r_q[i], r_q[i + 1], od, od2
                            )
                if cutlass.const_expr(not disable_output):
                    oa = oa + oa2
                    ob = ob + ob2
                    oc = oc + oc2
                    od = od + od2


                if cutlass.const_expr(
                    cache_intermediate_states or per_token_pool_scatter
                ):
                    for i in cutlass.range_constexpr(vec_size):
                        r_hb4_0[i] = cutlass.BFloat16(r_h[0, i])
                        r_hb4_1[i] = cutlass.BFloat16(r_h[1, i])
                        r_hb4_2[i] = cutlass.BFloat16(r_h[2, i])
                        r_hb4_3[i] = cutlass.BFloat16(r_h[3, i])

                if cutlass.const_expr(per_token_pool_scatter):


                    pool_slot_t = cutlass.Int32(ssm_state_indices[i_n, i_t])
                    if cutlass.const_expr(per_token_pool_scatter_flat):


                        fla_idx = cutlass.Int64(pool_slot_t) * HV + i_hv
                        ita = cute.local_tile(
                            intermediate_states,
                            (1, 1, vec_size),
                            (fla_idx, va, lane_in_group),
                        )
                        itb = cute.local_tile(
                            intermediate_states,
                            (1, 1, vec_size),
                            (fla_idx, vb, lane_in_group),
                        )
                        itc = cute.local_tile(
                            intermediate_states,
                            (1, 1, vec_size),
                            (fla_idx, vc, lane_in_group),
                        )
                        itd = cute.local_tile(
                            intermediate_states,
                            (1, 1, vec_size),
                            (fla_idx, vd, lane_in_group),
                        )
                    else:

                        h0_slot_t = h0_source[
                            (cutlass.Int64(pool_slot_t), i_hv, None, None)
                        ]
                        ita = cute.local_tile(
                            h0_slot_t, (1, vec_size), (va, lane_in_group)
                        )
                        itb = cute.local_tile(
                            h0_slot_t, (1, vec_size), (vb, lane_in_group)
                        )
                        itc = cute.local_tile(
                            h0_slot_t, (1, vec_size), (vc, lane_in_group)
                        )
                        itd = cute.local_tile(
                            h0_slot_t, (1, vec_size), (vd, lane_in_group)
                        )
                    cute.autovec_copy(r_hb4_0, ita)
                    cute.autovec_copy(r_hb4_1, itb)
                    cute.autovec_copy(r_hb4_2, itc)
                    cute.autovec_copy(r_hb4_3, itd)
                elif cutlass.const_expr(cache_intermediate_states):


                    flat_idx = i_n * T * HV + i_t * HV + i_hv
                    ita = cute.local_tile(
                        intermediate_states,
                        (1, 1, vec_size),
                        (flat_idx, va, lane_in_group),
                    )
                    itb = cute.local_tile(
                        intermediate_states,
                        (1, 1, vec_size),
                        (flat_idx, vb, lane_in_group),
                    )
                    itc = cute.local_tile(
                        intermediate_states,
                        (1, 1, vec_size),
                        (flat_idx, vc, lane_in_group),
                    )
                    itd = cute.local_tile(
                        intermediate_states,
                        (1, 1, vec_size),
                        (flat_idx, vd, lane_in_group),
                    )
                    cute.autovec_copy(r_hb4_0, ita)
                    cute.autovec_copy(r_hb4_1, itb)
                    cute.autovec_copy(r_hb4_2, itc)
                    cute.autovec_copy(r_hb4_3, itd)

                if cutlass.const_expr(not disable_output):
                    for offset in [16, 8, 4, 2, 1]:
                        oa += cute.arch.shuffle_sync_bfly(
                            oa, offset=offset, mask=-1, mask_and_clamp=31
                        )
                        ob += cute.arch.shuffle_sync_bfly(
                            ob, offset=offset, mask=-1, mask_and_clamp=31
                        )
                        oc += cute.arch.shuffle_sync_bfly(
                            oc, offset=offset, mask=-1, mask_and_clamp=31
                        )
                        od += cute.arch.shuffle_sync_bfly(
                            od, offset=offset, mask=-1, mask_and_clamp=31
                        )

                    if lane_in_group == 0:
                        r_o4_bf16[0] = cutlass.BFloat16(oa)
                        r_o4_bf16[1] = cutlass.BFloat16(ob)
                        r_o4_bf16[2] = cutlass.BFloat16(oc)
                        r_o4_bf16[3] = cutlass.BFloat16(od)
                        ot4_slice = cute.local_tile(
                            o,
                            (1, 1, 1, ILP4),
                            (i_n, i_t, i_hv, vb4 // ILP4),
                        )
                        cute.autovec_copy(r_o4_bf16, ot4_slice)

            if cutlass.const_expr(not disable_state_update):


                if cutlass.const_expr(not (per_token_pool_scatter and same_pool)):
                    if cutlass.const_expr(
                        not cache_intermediate_states and not per_token_pool_scatter
                    ):
                        for i in cutlass.range_constexpr(vec_size):
                            r_hb4_0[i] = cutlass.BFloat16(r_h[0, i])
                            r_hb4_1[i] = cutlass.BFloat16(r_h[1, i])
                            r_hb4_2[i] = cutlass.BFloat16(r_h[2, i])
                            r_hb4_3[i] = cutlass.BFloat16(r_h[3, i])
                    cute.autovec_copy(r_hb4_0, hta_w)
                    cute.autovec_copy(r_hb4_1, htb_w)
                    cute.autovec_copy(r_hb4_2, htc_w)
                    cute.autovec_copy(r_hb4_3, htd_w)


@cute.kernel
def gdn_wide_vec_kernel(
    h0_source: cute.Tensor,
    intermediate_states: cute.Tensor,
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b_gate: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    h0_out_indices: cute.Tensor,
    accepted_steps: cute.Tensor,
    ssm_state_indices: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    tile_v: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
    disable_state_update: cutlass.Constexpr[bool],
    cache_intermediate_states: cutlass.Constexpr[bool],
    use_packed_fma: cutlass.Constexpr[bool],
    same_pool: cutlass.Constexpr[bool],
    disable_output: cutlass.Constexpr[bool],
    recovery_steps: cutlass.Constexpr[int],
    per_request_accepted_steps: cutlass.Constexpr[bool],
    per_token_pool_scatter: cutlass.Constexpr[bool],
    per_token_pool_scatter_flat: cutlass.Constexpr[bool],
    replayssm_rawv: cute.Tensor,
    replayssm_rawk: cute.Tensor,
    replayssm_g: cute.Tensor,
    replayssm_beta: cute.Tensor,
    CACHE_RING: cutlass.Constexpr[bool],
):
    tidx, _, _ = cute.arch.thread_idx()
    lane_in_warp = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)


    group_idx = tidx // LANES_PER_ROW
    lane_in_group = tidx % LANES_PER_ROW
    k_start = lane_in_group * ELEMS_PER_LANE

    ROWS_PER_GROUP: cutlass.Constexpr[int] = tile_v // NUM_GROUPS
    ITERS_PER_GROUP: cutlass.Constexpr[int] = ROWS_PER_GROUP // ILP_ROWS

    batch_idx, _, _ = cute.arch.block_idx()


    i_v = batch_idx % num_v_tiles
    tmp = batch_idx // num_v_tiles
    i_hv = tmp % HV
    i_n = tmp // HV
    i_h = i_hv // (HV // H)


    if cutlass.const_expr(per_request_accepted_steps):
        my_accepted_step = cutlass.Int32(accepted_steps[i_n])
    else:
        my_accepted_step = cutlass.Int32(T - 1)

    per_request_fused: cutlass.Constexpr[bool] = (
        per_request_accepted_steps
        and not disable_output
        and not disable_state_update


        and not per_token_pool_scatter
    )


    cache_idx = h0_indices[i_n]
    ring_slot = cache_idx

    r_A_log = cutlass.Float32(A_log[i_hv])
    r_dt_bias = cutlass.Float32(dt_bias[i_hv])


    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((T, K), stride=(K + 8, 1)), 16
    )
    sK = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((T, K), stride=(K + 8, 1)), 16
    )
    sGB = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((T, 2), stride=(2, 1)), 16
    )


    vec: cutlass.Constexpr[int] = ELEMS_PER_LANE
    r_h = cute.make_rmem_tensor(
        cute.make_layout((ILP_ROWS, vec), stride=(vec, 1)), cutlass.Float32
    )
    r_q = cute.make_rmem_tensor(cute.make_layout((vec,), stride=(1,)), cutlass.Float32)
    r_k = cute.make_rmem_tensor(cute.make_layout((vec,), stride=(1,)), cutlass.Float32)
    r_q_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec,), stride=(1,)), cutlass.BFloat16
    )
    r_k_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec,), stride=(1,)), cutlass.BFloat16
    )
    r_hb0 = cute.make_rmem_tensor(
        cute.make_layout((vec,), stride=(1,)), cutlass.BFloat16
    )
    r_hb1 = cute.make_rmem_tensor(
        cute.make_layout((vec,), stride=(1,)), cutlass.BFloat16
    )
    r_hb2 = cute.make_rmem_tensor(
        cute.make_layout((vec,), stride=(1,)), cutlass.BFloat16
    )
    r_hb3 = cute.make_rmem_tensor(
        cute.make_layout((vec,), stride=(1,)), cutlass.BFloat16
    )


    if cutlass.const_expr(not disable_output):
        r_o4_bf16 = cute.make_rmem_tensor(
            cute.make_layout((ILP_ROWS,), stride=(1,)), cutlass.BFloat16
        )

    if cache_idx < 0:
        cache_idx = cutlass.Int32(0)


    if cutlass.const_expr(same_pool):
        write_cache_idx = cache_idx
    else:
        write_cache_idx = h0_out_indices[i_n]
        if write_cache_idx < 0:
            write_cache_idx = cutlass.Int32(0)

    if cache_idx >= 0:


        h0_slot_r = h0_source[(cutlass.Int64(cache_idx), i_hv, None, None)]
        if cutlass.const_expr(same_pool):
            h0_slot_w = h0_slot_r
        else:
            h0_slot_w = h0_source[(cutlass.Int64(write_cache_idx), i_hv, None, None)]


        if cutlass.const_expr(tile_v == 32):
            v_base_pre = i_v * tile_v + group_idx * ROWS_PER_GROUP
            ht0_pre = cute.local_tile(
                h0_slot_r, (1, vec), (v_base_pre + 0, lane_in_group)
            )
            ht1_pre = cute.local_tile(
                h0_slot_r, (1, vec), (v_base_pre + 1, lane_in_group)
            )
            ht2_pre = cute.local_tile(
                h0_slot_r, (1, vec), (v_base_pre + 2, lane_in_group)
            )
            ht3_pre = cute.local_tile(
                h0_slot_r, (1, vec), (v_base_pre + 3, lane_in_group)
            )
            cute.autovec_copy(
                ht0_pre,
                r_hb0,
                l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.EVICT_FIRST,
            )
            cute.autovec_copy(
                ht1_pre,
                r_hb1,
                l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.EVICT_FIRST,
            )
            cute.autovec_copy(
                ht2_pre,
                r_hb2,
                l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.EVICT_FIRST,
            )
            cute.autovec_copy(
                ht3_pre,
                r_hb3,
                l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.EVICT_FIRST,
            )


        num_precompute_passes: cutlass.Constexpr[int] = (T + NUM_WARPS - 1) // NUM_WARPS
        member_pre = lane_in_warp % LANES_PER_ROW
        k_start_pre = member_pre * ELEMS_PER_LANE
        for pass_idx in cutlass.range_constexpr(num_precompute_passes):
            i_t_pre = pass_idx * NUM_WARPS + warp_idx


            all_recovery_pass: cutlass.Constexpr[bool] = (
                pass_idx + 1
            ) * NUM_WARPS <= recovery_steps
            do_q_pass: cutlass.Constexpr[bool] = (not disable_output) and (
                not all_recovery_pass
            )
            if i_t_pre < T:


                if cutlass.const_expr(do_q_pass):
                    q_tile_pre = cute.local_tile(
                        q, (1, 1, 1, vec), (i_n, i_t_pre, i_h, member_pre)
                    )
                    cute.autovec_copy(q_tile_pre, r_q_bf16)
                k_tile_pre = cute.local_tile(
                    k, (1, 1, 1, vec), (i_n, i_t_pre, i_h, member_pre)
                )
                cute.autovec_copy(k_tile_pre, r_k_bf16)

                if cutlass.const_expr(CACHE_RING):
                    if ring_slot >= 0 and i_v == 0:
                        if i_hv % (HV // H) == 0:
                            for i in cutlass.range_constexpr(vec):
                                replayssm_rawk[
                                    (ring_slot, i_h, i_t_pre, k_start_pre + i)
                                ] = r_k_bf16[i]
                        for i in cutlass.range_constexpr(vec):
                            replayssm_rawv[
                                (ring_slot, i_hv, i_t_pre, k_start_pre + i)
                            ] = v[(i_n, i_t_pre, i_hv, k_start_pre + i)]
                if cutlass.const_expr(do_q_pass):
                    for i in cutlass.range_constexpr(vec):
                        r_q[i] = cutlass.Float32(r_q_bf16[i])
                for i in cutlass.range_constexpr(vec):
                    r_k[i] = cutlass.Float32(r_k_bf16[i])

                if cutlass.const_expr(use_qk_l2norm):
                    sum_k = cutlass.Float32(0.0)
                    for i in cutlass.range_constexpr(vec):
                        sum_k += r_k[i] * r_k[i]

                    for offset in [8, 4, 2, 1]:
                        sum_k += cute.arch.shuffle_sync_bfly(
                            sum_k, offset=offset, mask=-1, mask_and_clamp=31
                        )
                    inv_norm_k = cute.rsqrt(sum_k + 1e-6, fastmath=True)
                    for i in cutlass.range_constexpr(vec):
                        r_k[i] = r_k[i] * inv_norm_k

                    if cutlass.const_expr(do_q_pass):
                        sum_q = cutlass.Float32(0.0)
                        for i in cutlass.range_constexpr(vec):
                            sum_q += r_q[i] * r_q[i]
                        for offset in [8, 4, 2, 1]:
                            sum_q += cute.arch.shuffle_sync_bfly(
                                sum_q, offset=offset, mask=-1, mask_and_clamp=31
                            )
                        inv_norm_q_scaled = (
                            cute.rsqrt(sum_q + 1e-6, fastmath=True) * scale
                        )
                        for i in cutlass.range_constexpr(vec):
                            r_q[i] = r_q[i] * inv_norm_q_scaled
                elif cutlass.const_expr(do_q_pass):
                    for i in cutlass.range_constexpr(vec):
                        r_q[i] = r_q[i] * scale


                if cutlass.const_expr(do_q_pass):
                    for i in cutlass.range_constexpr(vec):
                        sQ[(i_t_pre, k_start_pre + i)] = r_q[i]
                for i in cutlass.range_constexpr(vec):
                    sK[(i_t_pre, k_start_pre + i)] = r_k[i]


                r_a_pre = cutlass.Float32(a[i_n, i_t_pre, i_hv])
                r_b_pre = cutlass.Float32(b_gate[i_n, i_t_pre, i_hv])
                x_pre = r_a_pre + r_dt_bias
                beta_x_pre = softplus_beta * x_pre
                exp_beta_x_pre = cute.exp(beta_x_pre, fastmath=True)
                softplus_val_pre = (cutlass.Float32(1.0) / softplus_beta) * cute.log(
                    cutlass.Float32(1.0) + exp_beta_x_pre, fastmath=True
                )
                use_softplus_pre = (
                    cutlass.Float32(1.0)
                    if beta_x_pre <= softplus_threshold
                    else cutlass.Float32(0.0)
                )
                softplus_x_pre = (
                    use_softplus_pre * softplus_val_pre
                    + (cutlass.Float32(1.0) - use_softplus_pre) * x_pre
                )
                r_g_value_pre = -cute.exp(r_A_log, fastmath=True) * softplus_x_pre
                r_g_pre = cute.exp(r_g_value_pre, fastmath=True)
                r_beta_pre = cutlass.Float32(1.0) / (
                    cutlass.Float32(1.0) + cute.exp(-r_b_pre, fastmath=True)
                )

                if lane_in_warp == 0:
                    sGB[(i_t_pre, 0)] = r_g_pre
                    sGB[(i_t_pre, 1)] = r_beta_pre
                if cutlass.const_expr(CACHE_RING):
                    if ring_slot >= 0 and i_v == 0 and lane_in_warp == 0:
                        replayssm_g[(ring_slot, i_hv, i_t_pre)] = r_g_value_pre
                        replayssm_beta[(ring_slot, i_hv, i_t_pre)] = r_beta_pre

            cute.arch.barrier()


        for iter_idx in cutlass.range_constexpr(ITERS_PER_GROUP):
            v_base = i_v * tile_v + group_idx * ROWS_PER_GROUP + iter_idx * ILP_ROWS
            v0 = v_base + 0
            v1 = v_base + 1
            v2 = v_base + 2
            v3 = v_base + 3


            ht0 = cute.local_tile(h0_slot_r, (1, vec), (v0, lane_in_group))
            ht1 = cute.local_tile(h0_slot_r, (1, vec), (v1, lane_in_group))
            ht2 = cute.local_tile(h0_slot_r, (1, vec), (v2, lane_in_group))
            ht3 = cute.local_tile(h0_slot_r, (1, vec), (v3, lane_in_group))


            if cutlass.const_expr(same_pool):
                ht_w0 = ht0
                ht_w1 = ht1
                ht_w2 = ht2
                ht_w3 = ht3
            else:
                ht_w0 = cute.local_tile(h0_slot_w, (1, vec), (v0, lane_in_group))
                ht_w1 = cute.local_tile(h0_slot_w, (1, vec), (v1, lane_in_group))
                ht_w2 = cute.local_tile(h0_slot_w, (1, vec), (v2, lane_in_group))
                ht_w3 = cute.local_tile(h0_slot_w, (1, vec), (v3, lane_in_group))


            if cutlass.const_expr(iter_idx > 0 or tile_v != 32):
                cute.autovec_copy(
                    ht0,
                    r_hb0,
                    l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.EVICT_FIRST,
                )
                cute.autovec_copy(
                    ht1,
                    r_hb1,
                    l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.EVICT_FIRST,
                )
                cute.autovec_copy(
                    ht2,
                    r_hb2,
                    l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.EVICT_FIRST,
                )
                cute.autovec_copy(
                    ht3,
                    r_hb3,
                    l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.EVICT_FIRST,
                )
            for i in cutlass.range_constexpr(vec):
                r_h[0, i] = cutlass.Float32(r_hb0[i])
                r_h[1, i] = cutlass.Float32(r_hb1[i])
                r_h[2, i] = cutlass.Float32(r_hb2[i])
                r_h[3, i] = cutlass.Float32(r_hb3[i])


            r_g = cutlass.Float32(0.0)
            r_beta = cutlass.Float32(0.0)
            s0 = cutlass.Float32(0.0)
            s1 = cutlass.Float32(0.0)
            s2 = cutlass.Float32(0.0)
            s3 = cutlass.Float32(0.0)
            kv0 = cutlass.Float32(0.0)
            kv1 = cutlass.Float32(0.0)
            qv0 = cutlass.Float32(0.0)
            qv1 = cutlass.Float32(0.0)
            v_val0 = cutlass.Float32(0.0)
            v_val1 = cutlass.Float32(0.0)
            v_val2 = cutlass.Float32(0.0)
            v_val3 = cutlass.Float32(0.0)
            vn0 = cutlass.Float32(0.0)
            vn1 = cutlass.Float32(0.0)
            vn2 = cutlass.Float32(0.0)
            vn3 = cutlass.Float32(0.0)
            o0 = cutlass.Float32(0.0)
            o1 = cutlass.Float32(0.0)
            o2 = cutlass.Float32(0.0)
            o3 = cutlass.Float32(0.0)
            flat_idx = cutlass.Int32(0)


            flat_idx_fla = cutlass.Int64(0)
            if cutlass.const_expr(cache_intermediate_states):

                it0 = cute.local_tile(
                    intermediate_states, (1, 1, vec), (flat_idx, v0, lane_in_group)
                )
                it1 = cute.local_tile(
                    intermediate_states, (1, 1, vec), (flat_idx, v1, lane_in_group)
                )
                it2 = cute.local_tile(
                    intermediate_states, (1, 1, vec), (flat_idx, v2, lane_in_group)
                )
                it3 = cute.local_tile(
                    intermediate_states, (1, 1, vec), (flat_idx, v3, lane_in_group)
                )
            elif cutlass.const_expr(
                per_token_pool_scatter and per_token_pool_scatter_flat
            ):


                it0 = cute.local_tile(
                    intermediate_states, (1, 1, vec), (flat_idx_fla, v0, lane_in_group)
                )
                it1 = cute.local_tile(
                    intermediate_states, (1, 1, vec), (flat_idx_fla, v1, lane_in_group)
                )
                it2 = cute.local_tile(
                    intermediate_states, (1, 1, vec), (flat_idx_fla, v2, lane_in_group)
                )
                it3 = cute.local_tile(
                    intermediate_states, (1, 1, vec), (flat_idx_fla, v3, lane_in_group)
                )
            elif cutlass.const_expr(per_token_pool_scatter):


                _ph_slot = cutlass.Int32(0)
                _ph_slot_view = h0_source[(cutlass.Int64(_ph_slot), i_hv, None, None)]
                it0 = cute.local_tile(_ph_slot_view, (1, vec), (v0, lane_in_group))
                it1 = cute.local_tile(_ph_slot_view, (1, vec), (v1, lane_in_group))
                it2 = cute.local_tile(_ph_slot_view, (1, vec), (v2, lane_in_group))
                it3 = cute.local_tile(_ph_slot_view, (1, vec), (v3, lane_in_group))


            if cutlass.const_expr(per_request_fused):
                _phase_a_bound = my_accepted_step + cutlass.Int32(1)
            else:
                _phase_a_bound = recovery_steps
            for i_t in cutlass.range(
                _phase_a_bound,
                unroll=1,
                unroll_full=(not per_request_fused) and (recovery_steps <= 1),
            ):
                r_g = sGB[(i_t, 0)]
                r_beta = sGB[(i_t, 1)]


                s0 = cutlass.Float32(0.0)
                s1 = cutlass.Float32(0.0)
                s2 = cutlass.Float32(0.0)
                s3 = cutlass.Float32(0.0)
                for i in cutlass.range_constexpr(0, vec, 2):
                    kv0 = sK[(i_t, k_start + i)]
                    kv1 = sK[(i_t, k_start + i + 1)]
                    if cutlass.const_expr(use_packed_fma):
                        r_h[0, i], r_h[0, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[0, i], r_h[0, i + 1]),
                            src_b=(r_g, r_g),
                            src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
                        )
                        r_h[1, i], r_h[1, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[1, i], r_h[1, i + 1]),
                            src_b=(r_g, r_g),
                            src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
                        )
                        r_h[2, i], r_h[2, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[2, i], r_h[2, i + 1]),
                            src_b=(r_g, r_g),
                            src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
                        )
                        r_h[3, i], r_h[3, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[3, i], r_h[3, i + 1]),
                            src_b=(r_g, r_g),
                            src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
                        )
                    else:
                        r_h[0, i], r_h[0, i + 1] = fma_pair_mul(
                            r_h[0, i], r_h[0, i + 1], r_g, r_g
                        )
                        r_h[1, i], r_h[1, i + 1] = fma_pair_mul(
                            r_h[1, i], r_h[1, i + 1], r_g, r_g
                        )
                        r_h[2, i], r_h[2, i + 1] = fma_pair_mul(
                            r_h[2, i], r_h[2, i + 1], r_g, r_g
                        )
                        r_h[3, i], r_h[3, i + 1] = fma_pair_mul(
                            r_h[3, i], r_h[3, i + 1], r_g, r_g
                        )
                    s0 = s0 + r_h[0, i] * kv0 + r_h[0, i + 1] * kv1
                    s1 = s1 + r_h[1, i] * kv0 + r_h[1, i + 1] * kv1
                    s2 = s2 + r_h[2, i] * kv0 + r_h[2, i + 1] * kv1
                    s3 = s3 + r_h[3, i] * kv0 + r_h[3, i + 1] * kv1


                for offset in [8, 4, 2, 1]:
                    s0 += cute.arch.shuffle_sync_bfly(
                        s0, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    s1 += cute.arch.shuffle_sync_bfly(
                        s1, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    s2 += cute.arch.shuffle_sync_bfly(
                        s2, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    s3 += cute.arch.shuffle_sync_bfly(
                        s3, offset=offset, mask=-1, mask_and_clamp=31
                    )


                v_val0 = cutlass.Float32(v[(i_n, i_t, i_hv, v0)])
                v_val1 = cutlass.Float32(v[(i_n, i_t, i_hv, v1)])
                v_val2 = cutlass.Float32(v[(i_n, i_t, i_hv, v2)])
                v_val3 = cutlass.Float32(v[(i_n, i_t, i_hv, v3)])
                vn0 = (v_val0 - s0) * r_beta
                vn1 = (v_val1 - s1) * r_beta
                vn2 = (v_val2 - s2) * r_beta
                vn3 = (v_val3 - s3) * r_beta


                o0 = cutlass.Float32(0.0)
                o1 = cutlass.Float32(0.0)
                o2 = cutlass.Float32(0.0)
                o3 = cutlass.Float32(0.0)


                emit_output: cutlass.Constexpr[bool] = False
                for i in cutlass.range_constexpr(0, vec, 2):
                    if cutlass.const_expr(emit_output):
                        qv0 = sQ[(i_t, k_start + i)]
                        qv1 = sQ[(i_t, k_start + i + 1)]
                    kv0 = sK[(i_t, k_start + i)]
                    kv1 = sK[(i_t, k_start + i + 1)]
                    if cutlass.const_expr(use_packed_fma):
                        r_h[0, i], r_h[0, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(kv0, kv1),
                            src_b=(vn0, vn0),
                            src_c=(r_h[0, i], r_h[0, i + 1]),
                        )
                        r_h[1, i], r_h[1, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(kv0, kv1),
                            src_b=(vn1, vn1),
                            src_c=(r_h[1, i], r_h[1, i + 1]),
                        )
                        r_h[2, i], r_h[2, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(kv0, kv1),
                            src_b=(vn2, vn2),
                            src_c=(r_h[2, i], r_h[2, i + 1]),
                        )
                        r_h[3, i], r_h[3, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(kv0, kv1),
                            src_b=(vn3, vn3),
                            src_c=(r_h[3, i], r_h[3, i + 1]),
                        )
                    else:
                        r_h[0, i], r_h[0, i + 1] = fma_pair(
                            kv0, kv1, vn0, vn0, r_h[0, i], r_h[0, i + 1]
                        )
                        r_h[1, i], r_h[1, i + 1] = fma_pair(
                            kv0, kv1, vn1, vn1, r_h[1, i], r_h[1, i + 1]
                        )
                        r_h[2, i], r_h[2, i + 1] = fma_pair(
                            kv0, kv1, vn2, vn2, r_h[2, i], r_h[2, i + 1]
                        )
                        r_h[3, i], r_h[3, i + 1] = fma_pair(
                            kv0, kv1, vn3, vn3, r_h[3, i], r_h[3, i + 1]
                        )
                    if cutlass.const_expr(emit_output):

                        o0 = o0 + r_h[0, i] * qv0 + r_h[0, i + 1] * qv1
                        o1 = o1 + r_h[1, i] * qv0 + r_h[1, i + 1] * qv1
                        o2 = o2 + r_h[2, i] * qv0 + r_h[2, i + 1] * qv1
                        o3 = o3 + r_h[3, i] * qv0 + r_h[3, i + 1] * qv1

                if cutlass.const_expr(emit_output):

                    for offset in [8, 4, 2, 1]:
                        o0 += cute.arch.shuffle_sync_bfly(
                            o0, offset=offset, mask=-1, mask_and_clamp=31
                        )
                        o1 += cute.arch.shuffle_sync_bfly(
                            o1, offset=offset, mask=-1, mask_and_clamp=31
                        )
                        o2 += cute.arch.shuffle_sync_bfly(
                            o2, offset=offset, mask=-1, mask_and_clamp=31
                        )
                        o3 += cute.arch.shuffle_sync_bfly(
                            o3, offset=offset, mask=-1, mask_and_clamp=31
                        )


                    if lane_in_group == 0:
                        r_o4_bf16[0] = cutlass.BFloat16(o0)
                        r_o4_bf16[1] = cutlass.BFloat16(o1)
                        r_o4_bf16[2] = cutlass.BFloat16(o2)
                        r_o4_bf16[3] = cutlass.BFloat16(o3)
                        o_slice = cute.local_tile(
                            o,
                            (1, 1, 1, ILP_ROWS),
                            (i_n, i_t, i_hv, v_base // ILP_ROWS),
                        )
                        cute.autovec_copy(r_o4_bf16, o_slice)


                if cutlass.const_expr(cache_intermediate_states):
                    for i in cutlass.range_constexpr(vec):
                        r_hb0[i] = cutlass.BFloat16(r_h[0, i])
                        r_hb1[i] = cutlass.BFloat16(r_h[1, i])
                        r_hb2[i] = cutlass.BFloat16(r_h[2, i])
                        r_hb3[i] = cutlass.BFloat16(r_h[3, i])


                    flat_idx = i_n * T * HV + i_t * HV + i_hv
                    it0 = cute.local_tile(
                        intermediate_states,
                        (1, 1, vec),
                        (flat_idx, v0, lane_in_group),
                    )
                    it1 = cute.local_tile(
                        intermediate_states,
                        (1, 1, vec),
                        (flat_idx, v1, lane_in_group),
                    )
                    it2 = cute.local_tile(
                        intermediate_states,
                        (1, 1, vec),
                        (flat_idx, v2, lane_in_group),
                    )
                    it3 = cute.local_tile(
                        intermediate_states,
                        (1, 1, vec),
                        (flat_idx, v3, lane_in_group),
                    )
                    cute.autovec_copy(r_hb0, it0)
                    cute.autovec_copy(r_hb1, it1)
                    cute.autovec_copy(r_hb2, it2)
                    cute.autovec_copy(r_hb3, it3)


            if cutlass.const_expr(
                (recovery_steps > 0 or per_request_fused)
                and not disable_state_update
                and not cache_intermediate_states
            ):
                for i in cutlass.range_constexpr(vec):
                    r_hb0[i] = cutlass.BFloat16(r_h[0, i])
                    r_hb1[i] = cutlass.BFloat16(r_h[1, i])
                    r_hb2[i] = cutlass.BFloat16(r_h[2, i])
                    r_hb3[i] = cutlass.BFloat16(r_h[3, i])
                cute.autovec_copy(r_hb0, ht_w0)
                cute.autovec_copy(r_hb1, ht_w1)
                cute.autovec_copy(r_hb2, ht_w2)
                cute.autovec_copy(r_hb3, ht_w3)


            T_decode_const: cutlass.Constexpr[int] = T - recovery_steps
            if cutlass.const_expr(per_request_fused):
                _loop_limit = cutlass.Int32(T) - my_accepted_step - cutlass.Int32(1)
            elif cutlass.const_expr(per_request_accepted_steps):
                _loop_limit = my_accepted_step + cutlass.Int32(1 - recovery_steps)
            else:
                _loop_limit = T_decode_const
            for i_t_offset in cutlass.range(
                _loop_limit,
                unroll=1,
                unroll_full=(T_decode_const <= 1) and not per_request_accepted_steps,
            ):
                if cutlass.const_expr(per_request_fused):
                    i_t = my_accepted_step + cutlass.Int32(1) + i_t_offset
                else:
                    i_t = recovery_steps + i_t_offset
                r_g = sGB[(i_t, 0)]
                r_beta = sGB[(i_t, 1)]


                s0 = cutlass.Float32(0.0)
                s1 = cutlass.Float32(0.0)
                s2 = cutlass.Float32(0.0)
                s3 = cutlass.Float32(0.0)
                for i in cutlass.range_constexpr(0, vec, 2):
                    kv0 = sK[(i_t, k_start + i)]
                    kv1 = sK[(i_t, k_start + i + 1)]
                    if cutlass.const_expr(use_packed_fma):
                        r_h[0, i], r_h[0, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[0, i], r_h[0, i + 1]),
                            src_b=(r_g, r_g),
                            src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
                        )
                        r_h[1, i], r_h[1, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[1, i], r_h[1, i + 1]),
                            src_b=(r_g, r_g),
                            src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
                        )
                        r_h[2, i], r_h[2, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[2, i], r_h[2, i + 1]),
                            src_b=(r_g, r_g),
                            src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
                        )
                        r_h[3, i], r_h[3, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[3, i], r_h[3, i + 1]),
                            src_b=(r_g, r_g),
                            src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
                        )
                    else:
                        r_h[0, i], r_h[0, i + 1] = fma_pair_mul(
                            r_h[0, i], r_h[0, i + 1], r_g, r_g
                        )
                        r_h[1, i], r_h[1, i + 1] = fma_pair_mul(
                            r_h[1, i], r_h[1, i + 1], r_g, r_g
                        )
                        r_h[2, i], r_h[2, i + 1] = fma_pair_mul(
                            r_h[2, i], r_h[2, i + 1], r_g, r_g
                        )
                        r_h[3, i], r_h[3, i + 1] = fma_pair_mul(
                            r_h[3, i], r_h[3, i + 1], r_g, r_g
                        )
                    s0 = s0 + r_h[0, i] * kv0 + r_h[0, i + 1] * kv1
                    s1 = s1 + r_h[1, i] * kv0 + r_h[1, i + 1] * kv1
                    s2 = s2 + r_h[2, i] * kv0 + r_h[2, i + 1] * kv1
                    s3 = s3 + r_h[3, i] * kv0 + r_h[3, i + 1] * kv1

                for offset in [8, 4, 2, 1]:
                    s0 += cute.arch.shuffle_sync_bfly(
                        s0, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    s1 += cute.arch.shuffle_sync_bfly(
                        s1, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    s2 += cute.arch.shuffle_sync_bfly(
                        s2, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    s3 += cute.arch.shuffle_sync_bfly(
                        s3, offset=offset, mask=-1, mask_and_clamp=31
                    )

                v_val0 = cutlass.Float32(v[(i_n, i_t, i_hv, v0)])
                v_val1 = cutlass.Float32(v[(i_n, i_t, i_hv, v1)])
                v_val2 = cutlass.Float32(v[(i_n, i_t, i_hv, v2)])
                v_val3 = cutlass.Float32(v[(i_n, i_t, i_hv, v3)])
                vn0 = (v_val0 - s0) * r_beta
                vn1 = (v_val1 - s1) * r_beta
                vn2 = (v_val2 - s2) * r_beta
                vn3 = (v_val3 - s3) * r_beta


                o0 = cutlass.Float32(0.0)
                o1 = cutlass.Float32(0.0)
                o2 = cutlass.Float32(0.0)
                o3 = cutlass.Float32(0.0)
                for i in cutlass.range_constexpr(0, vec, 2):
                    if cutlass.const_expr(not disable_output):
                        qv0 = sQ[(i_t, k_start + i)]
                        qv1 = sQ[(i_t, k_start + i + 1)]
                    kv0 = sK[(i_t, k_start + i)]
                    kv1 = sK[(i_t, k_start + i + 1)]
                    if cutlass.const_expr(use_packed_fma):
                        r_h[0, i], r_h[0, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(kv0, kv1),
                            src_b=(vn0, vn0),
                            src_c=(r_h[0, i], r_h[0, i + 1]),
                        )
                        r_h[1, i], r_h[1, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(kv0, kv1),
                            src_b=(vn1, vn1),
                            src_c=(r_h[1, i], r_h[1, i + 1]),
                        )
                        r_h[2, i], r_h[2, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(kv0, kv1),
                            src_b=(vn2, vn2),
                            src_c=(r_h[2, i], r_h[2, i + 1]),
                        )
                        r_h[3, i], r_h[3, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(kv0, kv1),
                            src_b=(vn3, vn3),
                            src_c=(r_h[3, i], r_h[3, i + 1]),
                        )
                    else:
                        r_h[0, i], r_h[0, i + 1] = fma_pair(
                            kv0, kv1, vn0, vn0, r_h[0, i], r_h[0, i + 1]
                        )
                        r_h[1, i], r_h[1, i + 1] = fma_pair(
                            kv0, kv1, vn1, vn1, r_h[1, i], r_h[1, i + 1]
                        )
                        r_h[2, i], r_h[2, i + 1] = fma_pair(
                            kv0, kv1, vn2, vn2, r_h[2, i], r_h[2, i + 1]
                        )
                        r_h[3, i], r_h[3, i + 1] = fma_pair(
                            kv0, kv1, vn3, vn3, r_h[3, i], r_h[3, i + 1]
                        )
                    if cutlass.const_expr(not disable_output):
                        o0 = o0 + r_h[0, i] * qv0 + r_h[0, i + 1] * qv1
                        o1 = o1 + r_h[1, i] * qv0 + r_h[1, i + 1] * qv1
                        o2 = o2 + r_h[2, i] * qv0 + r_h[2, i + 1] * qv1
                        o3 = o3 + r_h[3, i] * qv0 + r_h[3, i + 1] * qv1

                if cutlass.const_expr(not disable_output):
                    for offset in [8, 4, 2, 1]:
                        o0 += cute.arch.shuffle_sync_bfly(
                            o0, offset=offset, mask=-1, mask_and_clamp=31
                        )
                        o1 += cute.arch.shuffle_sync_bfly(
                            o1, offset=offset, mask=-1, mask_and_clamp=31
                        )
                        o2 += cute.arch.shuffle_sync_bfly(
                            o2, offset=offset, mask=-1, mask_and_clamp=31
                        )
                        o3 += cute.arch.shuffle_sync_bfly(
                            o3, offset=offset, mask=-1, mask_and_clamp=31
                        )

                    if lane_in_group == 0:
                        r_o4_bf16[0] = cutlass.BFloat16(o0)
                        r_o4_bf16[1] = cutlass.BFloat16(o1)
                        r_o4_bf16[2] = cutlass.BFloat16(o2)
                        r_o4_bf16[3] = cutlass.BFloat16(o3)
                        o_slice = cute.local_tile(
                            o,
                            (1, 1, 1, ILP_ROWS),
                            (i_n, i_t, i_hv, v_base // ILP_ROWS),
                        )
                        cute.autovec_copy(r_o4_bf16, o_slice)


                if cutlass.const_expr(
                    cache_intermediate_states or per_token_pool_scatter
                ):
                    for i in cutlass.range_constexpr(vec):
                        r_hb0[i] = cutlass.BFloat16(r_h[0, i])
                        r_hb1[i] = cutlass.BFloat16(r_h[1, i])
                        r_hb2[i] = cutlass.BFloat16(r_h[2, i])
                        r_hb3[i] = cutlass.BFloat16(r_h[3, i])

                if cutlass.const_expr(per_token_pool_scatter):


                    if cutlass.const_expr(per_token_pool_scatter_flat):


                        pool_slot_t = cutlass.Int32(ssm_state_indices[i_n, i_t])
                        flat_idx_fla = cutlass.Int64(pool_slot_t) * HV + i_hv
                        it0 = cute.local_tile(
                            intermediate_states,
                            (1, 1, vec),
                            (flat_idx_fla, v0, lane_in_group),
                        )
                        it1 = cute.local_tile(
                            intermediate_states,
                            (1, 1, vec),
                            (flat_idx_fla, v1, lane_in_group),
                        )
                        it2 = cute.local_tile(
                            intermediate_states,
                            (1, 1, vec),
                            (flat_idx_fla, v2, lane_in_group),
                        )
                        it3 = cute.local_tile(
                            intermediate_states,
                            (1, 1, vec),
                            (flat_idx_fla, v3, lane_in_group),
                        )
                    else:


                        pool_slot_t = cutlass.Int32(ssm_state_indices[i_n, i_t])
                        h0_slot_t = h0_source[
                            (cutlass.Int64(pool_slot_t), i_hv, None, None)
                        ]
                        it0 = cute.local_tile(h0_slot_t, (1, vec), (v0, lane_in_group))
                        it1 = cute.local_tile(h0_slot_t, (1, vec), (v1, lane_in_group))
                        it2 = cute.local_tile(h0_slot_t, (1, vec), (v2, lane_in_group))
                        it3 = cute.local_tile(h0_slot_t, (1, vec), (v3, lane_in_group))
                    cute.autovec_copy(r_hb0, it0)
                    cute.autovec_copy(r_hb1, it1)
                    cute.autovec_copy(r_hb2, it2)
                    cute.autovec_copy(r_hb3, it3)
                elif cutlass.const_expr(cache_intermediate_states):


                    flat_idx = i_n * T * HV + i_t * HV + i_hv
                    it0 = cute.local_tile(
                        intermediate_states,
                        (1, 1, vec),
                        (flat_idx, v0, lane_in_group),
                    )
                    it1 = cute.local_tile(
                        intermediate_states,
                        (1, 1, vec),
                        (flat_idx, v1, lane_in_group),
                    )
                    it2 = cute.local_tile(
                        intermediate_states,
                        (1, 1, vec),
                        (flat_idx, v2, lane_in_group),
                    )
                    it3 = cute.local_tile(
                        intermediate_states,
                        (1, 1, vec),
                        (flat_idx, v3, lane_in_group),
                    )
                    cute.autovec_copy(r_hb0, it0)
                    cute.autovec_copy(r_hb1, it1)
                    cute.autovec_copy(r_hb2, it2)
                    cute.autovec_copy(r_hb3, it3)


            if cutlass.const_expr(
                not disable_state_update
                and not cache_intermediate_states
                and recovery_steps == 0
                and not per_request_fused
                and not (per_token_pool_scatter and same_pool)
            ):
                for i in cutlass.range_constexpr(vec):
                    r_hb0[i] = cutlass.BFloat16(r_h[0, i])
                    r_hb1[i] = cutlass.BFloat16(r_h[1, i])
                    r_hb2[i] = cutlass.BFloat16(r_h[2, i])
                    r_hb3[i] = cutlass.BFloat16(r_h[3, i])
                cute.autovec_copy(r_hb0, ht_w0)
                cute.autovec_copy(r_hb1, ht_w1)
                cute.autovec_copy(r_hb2, ht_w2)
                cute.autovec_copy(r_hb3, ht_w3)


@cute.kernel
def gdn_wide_vec_kernel_t1(
    h0_source: cute.Tensor,
    intermediate_states: cute.Tensor,
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b_gate: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    h0_out_indices: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    tile_v: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
    disable_state_update: cutlass.Constexpr[bool],
    cache_intermediate_states: cutlass.Constexpr[bool],
    use_packed_fma: cutlass.Constexpr[bool],
    same_pool: cutlass.Constexpr[bool],
):
    tidx, _, _ = cute.arch.thread_idx()
    lane_in_warp = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)


    group_idx = tidx // LANES_PER_ROW
    lane_in_group = tidx % LANES_PER_ROW
    k_start = lane_in_group * ELEMS_PER_LANE

    ROWS_PER_GROUP: cutlass.Constexpr[int] = tile_v // NUM_GROUPS
    ITERS_PER_GROUP: cutlass.Constexpr[int] = ROWS_PER_GROUP // ILP_ROWS

    batch_idx, _, _ = cute.arch.block_idx()


    i_v = batch_idx % num_v_tiles
    tmp = batch_idx // num_v_tiles
    i_hv = tmp % HV
    i_n = tmp // HV
    i_h = i_hv // (HV // H)


    cache_idx = h0_indices[i_n]

    r_A_log = cutlass.Float32(A_log[i_hv])
    r_dt_bias = cutlass.Float32(dt_bias[i_hv])


    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((T, K), stride=(K + 8, 1)), 16
    )
    sK = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((T, K), stride=(K + 8, 1)), 16
    )
    sGB = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((T, 3), stride=(3, 1)), 16
    )


    vec: cutlass.Constexpr[int] = ELEMS_PER_LANE
    r_h = cute.make_rmem_tensor(
        cute.make_layout((ILP_ROWS, vec), stride=(vec, 1)), cutlass.Float32
    )
    r_q = cute.make_rmem_tensor(cute.make_layout((vec,), stride=(1,)), cutlass.Float32)
    r_k = cute.make_rmem_tensor(cute.make_layout((vec,), stride=(1,)), cutlass.Float32)
    r_q_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec,), stride=(1,)), cutlass.BFloat16
    )
    r_k_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec,), stride=(1,)), cutlass.BFloat16
    )
    r_hb0 = cute.make_rmem_tensor(
        cute.make_layout((vec,), stride=(1,)), cutlass.BFloat16
    )
    r_hb1 = cute.make_rmem_tensor(
        cute.make_layout((vec,), stride=(1,)), cutlass.BFloat16
    )
    r_hb2 = cute.make_rmem_tensor(
        cute.make_layout((vec,), stride=(1,)), cutlass.BFloat16
    )
    r_hb3 = cute.make_rmem_tensor(
        cute.make_layout((vec,), stride=(1,)), cutlass.BFloat16
    )

    if cache_idx < 0:
        cache_idx = cutlass.Int32(0)


    if cutlass.const_expr(same_pool):
        write_cache_idx = cache_idx
    else:
        write_cache_idx = h0_out_indices[i_n]
        if write_cache_idx < 0:
            write_cache_idx = cutlass.Int32(0)

    if cache_idx >= 0:


        h0_slot_r = h0_source[(cutlass.Int64(cache_idx), i_hv, None, None)]
        if cutlass.const_expr(same_pool):
            h0_slot_w = h0_slot_r
        else:
            h0_slot_w = h0_source[(cutlass.Int64(write_cache_idx), i_hv, None, None)]


        num_precompute_passes: cutlass.Constexpr[int] = (T + NUM_WARPS - 1) // NUM_WARPS
        member_pre = lane_in_warp % LANES_PER_ROW
        k_start_pre = member_pre * ELEMS_PER_LANE
        for pass_idx in cutlass.range_constexpr(num_precompute_passes):
            i_t_pre = pass_idx * NUM_WARPS + warp_idx
            if i_t_pre < T:
                q_tile_pre = cute.local_tile(
                    q, (1, 1, 1, vec), (i_n, i_t_pre, i_h, member_pre)
                )
                k_tile_pre = cute.local_tile(
                    k, (1, 1, 1, vec), (i_n, i_t_pre, i_h, member_pre)
                )
                cute.autovec_copy(q_tile_pre, r_q_bf16)
                cute.autovec_copy(k_tile_pre, r_k_bf16)
                for i in cutlass.range_constexpr(vec):
                    r_q[i] = cutlass.Float32(r_q_bf16[i])
                    r_k[i] = cutlass.Float32(r_k_bf16[i])

                if cutlass.const_expr(use_qk_l2norm):
                    sum_q = cutlass.Float32(0.0)
                    sum_k = cutlass.Float32(0.0)
                    for i in cutlass.range_constexpr(vec):
                        sum_q += r_q[i] * r_q[i]
                        sum_k += r_k[i] * r_k[i]

                    for offset in [8, 4, 2, 1]:
                        sum_q += cute.arch.shuffle_sync_bfly(
                            sum_q, offset=offset, mask=-1, mask_and_clamp=31
                        )
                        sum_k += cute.arch.shuffle_sync_bfly(
                            sum_k, offset=offset, mask=-1, mask_and_clamp=31
                        )
                    inv_norm_q_scaled = cute.rsqrt(sum_q + 1e-6, fastmath=True) * scale
                    inv_norm_k = cute.rsqrt(sum_k + 1e-6, fastmath=True)
                    for i in cutlass.range_constexpr(vec):
                        r_q[i] = r_q[i] * inv_norm_q_scaled
                        r_k[i] = r_k[i] * inv_norm_k
                else:
                    for i in cutlass.range_constexpr(vec):
                        r_q[i] = r_q[i] * scale


                for i in cutlass.range_constexpr(vec):
                    sQ[(i_t_pre, k_start_pre + i)] = r_q[i]
                    sK[(i_t_pre, k_start_pre + i)] = r_k[i]


                kq_partial = cutlass.Float32(0.0)
                for i in cutlass.range_constexpr(vec):
                    kq_partial += r_k[i] * r_q[i]
                for offset in [8, 4, 2, 1]:
                    kq_partial += cute.arch.shuffle_sync_bfly(
                        kq_partial, offset=offset, mask=-1, mask_and_clamp=31
                    )


                r_a_pre = cutlass.Float32(a[i_n, i_t_pre, i_hv])
                r_b_pre = cutlass.Float32(b_gate[i_n, i_t_pre, i_hv])
                x_pre = r_a_pre + r_dt_bias
                beta_x_pre = softplus_beta * x_pre
                exp_beta_x_pre = cute.exp(beta_x_pre, fastmath=True)
                softplus_val_pre = (cutlass.Float32(1.0) / softplus_beta) * cute.log(
                    cutlass.Float32(1.0) + exp_beta_x_pre, fastmath=True
                )
                use_softplus_pre = (
                    cutlass.Float32(1.0)
                    if beta_x_pre <= softplus_threshold
                    else cutlass.Float32(0.0)
                )
                softplus_x_pre = (
                    use_softplus_pre * softplus_val_pre
                    + (cutlass.Float32(1.0) - use_softplus_pre) * x_pre
                )
                r_g_pre = cute.exp(
                    -cute.exp(r_A_log, fastmath=True) * softplus_x_pre, fastmath=True
                )
                r_beta_pre = cutlass.Float32(1.0) / (
                    cutlass.Float32(1.0) + cute.exp(-r_b_pre, fastmath=True)
                )

                if lane_in_warp == 0:
                    sGB[(i_t_pre, 0)] = r_g_pre
                    sGB[(i_t_pre, 1)] = r_beta_pre
                    sGB[(i_t_pre, 2)] = kq_partial

            cute.arch.barrier()


        for iter_idx in cutlass.range_constexpr(ITERS_PER_GROUP):
            v_base = i_v * tile_v + group_idx * ROWS_PER_GROUP + iter_idx * ILP_ROWS
            v0 = v_base + 0
            v1 = v_base + 1
            v2 = v_base + 2
            v3 = v_base + 3


            ht0 = cute.local_tile(h0_slot_r, (1, vec), (v0, lane_in_group))
            ht1 = cute.local_tile(h0_slot_r, (1, vec), (v1, lane_in_group))
            ht2 = cute.local_tile(h0_slot_r, (1, vec), (v2, lane_in_group))
            ht3 = cute.local_tile(h0_slot_r, (1, vec), (v3, lane_in_group))


            if cutlass.const_expr(same_pool):
                ht_w0 = ht0
                ht_w1 = ht1
                ht_w2 = ht2
                ht_w3 = ht3
            else:
                ht_w0 = cute.local_tile(h0_slot_w, (1, vec), (v0, lane_in_group))
                ht_w1 = cute.local_tile(h0_slot_w, (1, vec), (v1, lane_in_group))
                ht_w2 = cute.local_tile(h0_slot_w, (1, vec), (v2, lane_in_group))
                ht_w3 = cute.local_tile(h0_slot_w, (1, vec), (v3, lane_in_group))
            cute.autovec_copy(ht0, r_hb0)
            cute.autovec_copy(ht1, r_hb1)
            cute.autovec_copy(ht2, r_hb2)
            cute.autovec_copy(ht3, r_hb3)
            for i in cutlass.range_constexpr(vec):
                r_h[0, i] = cutlass.Float32(r_hb0[i])
                r_h[1, i] = cutlass.Float32(r_hb1[i])
                r_h[2, i] = cutlass.Float32(r_hb2[i])
                r_h[3, i] = cutlass.Float32(r_hb3[i])


            for i_t in cutlass.range(T, unroll=1, unroll_full=(T <= 1)):
                r_g = sGB[(i_t, 0)]
                r_beta = sGB[(i_t, 1)]


                s0 = cutlass.Float32(0.0)
                s1 = cutlass.Float32(0.0)
                s2 = cutlass.Float32(0.0)
                s3 = cutlass.Float32(0.0)
                for i in cutlass.range_constexpr(0, vec, 2):
                    kv0 = sK[(i_t, k_start + i)]
                    kv1 = sK[(i_t, k_start + i + 1)]
                    if cutlass.const_expr(use_packed_fma):
                        r_h[0, i], r_h[0, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[0, i], r_h[0, i + 1]),
                            src_b=(r_g, r_g),
                            src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
                        )
                        r_h[1, i], r_h[1, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[1, i], r_h[1, i + 1]),
                            src_b=(r_g, r_g),
                            src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
                        )
                        r_h[2, i], r_h[2, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[2, i], r_h[2, i + 1]),
                            src_b=(r_g, r_g),
                            src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
                        )
                        r_h[3, i], r_h[3, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[3, i], r_h[3, i + 1]),
                            src_b=(r_g, r_g),
                            src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
                        )
                    else:
                        r_h[0, i], r_h[0, i + 1] = fma_pair_mul(
                            r_h[0, i], r_h[0, i + 1], r_g, r_g
                        )
                        r_h[1, i], r_h[1, i + 1] = fma_pair_mul(
                            r_h[1, i], r_h[1, i + 1], r_g, r_g
                        )
                        r_h[2, i], r_h[2, i + 1] = fma_pair_mul(
                            r_h[2, i], r_h[2, i + 1], r_g, r_g
                        )
                        r_h[3, i], r_h[3, i + 1] = fma_pair_mul(
                            r_h[3, i], r_h[3, i + 1], r_g, r_g
                        )
                    s0 = s0 + r_h[0, i] * kv0 + r_h[0, i + 1] * kv1
                    s1 = s1 + r_h[1, i] * kv0 + r_h[1, i + 1] * kv1
                    s2 = s2 + r_h[2, i] * kv0 + r_h[2, i + 1] * kv1
                    s3 = s3 + r_h[3, i] * kv0 + r_h[3, i + 1] * kv1


                for offset in [8, 4, 2, 1]:
                    s0 += cute.arch.shuffle_sync_bfly(
                        s0, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    s1 += cute.arch.shuffle_sync_bfly(
                        s1, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    s2 += cute.arch.shuffle_sync_bfly(
                        s2, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    s3 += cute.arch.shuffle_sync_bfly(
                        s3, offset=offset, mask=-1, mask_and_clamp=31
                    )


                v_val0 = cutlass.Float32(v[(i_n, i_t, i_hv, v0)])
                v_val1 = cutlass.Float32(v[(i_n, i_t, i_hv, v1)])
                v_val2 = cutlass.Float32(v[(i_n, i_t, i_hv, v2)])
                v_val3 = cutlass.Float32(v[(i_n, i_t, i_hv, v3)])
                vn0 = (v_val0 - s0) * r_beta
                vn1 = (v_val1 - s1) * r_beta
                vn2 = (v_val2 - s2) * r_beta
                vn3 = (v_val3 - s3) * r_beta


                o0 = cutlass.Float32(0.0)
                o1 = cutlass.Float32(0.0)
                o2 = cutlass.Float32(0.0)
                o3 = cutlass.Float32(0.0)
                for i in cutlass.range_constexpr(0, vec, 2):
                    qv0 = sQ[(i_t, k_start + i)]
                    qv1 = sQ[(i_t, k_start + i + 1)]
                    kv0 = sK[(i_t, k_start + i)]
                    kv1 = sK[(i_t, k_start + i + 1)]
                    if cutlass.const_expr(use_packed_fma):
                        r_h[0, i], r_h[0, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(kv0, kv1),
                            src_b=(vn0, vn0),
                            src_c=(r_h[0, i], r_h[0, i + 1]),
                        )
                        r_h[1, i], r_h[1, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(kv0, kv1),
                            src_b=(vn1, vn1),
                            src_c=(r_h[1, i], r_h[1, i + 1]),
                        )
                        r_h[2, i], r_h[2, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(kv0, kv1),
                            src_b=(vn2, vn2),
                            src_c=(r_h[2, i], r_h[2, i + 1]),
                        )
                        r_h[3, i], r_h[3, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(kv0, kv1),
                            src_b=(vn3, vn3),
                            src_c=(r_h[3, i], r_h[3, i + 1]),
                        )
                    else:
                        r_h[0, i], r_h[0, i + 1] = fma_pair(
                            kv0, kv1, vn0, vn0, r_h[0, i], r_h[0, i + 1]
                        )
                        r_h[1, i], r_h[1, i + 1] = fma_pair(
                            kv0, kv1, vn1, vn1, r_h[1, i], r_h[1, i + 1]
                        )
                        r_h[2, i], r_h[2, i + 1] = fma_pair(
                            kv0, kv1, vn2, vn2, r_h[2, i], r_h[2, i + 1]
                        )
                        r_h[3, i], r_h[3, i + 1] = fma_pair(
                            kv0, kv1, vn3, vn3, r_h[3, i], r_h[3, i + 1]
                        )

                    o0 = o0 + r_h[0, i] * qv0 + r_h[0, i + 1] * qv1
                    o1 = o1 + r_h[1, i] * qv0 + r_h[1, i + 1] * qv1
                    o2 = o2 + r_h[2, i] * qv0 + r_h[2, i + 1] * qv1
                    o3 = o3 + r_h[3, i] * qv0 + r_h[3, i + 1] * qv1


                for offset in [8, 4, 2, 1]:
                    o0 += cute.arch.shuffle_sync_bfly(
                        o0, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    o1 += cute.arch.shuffle_sync_bfly(
                        o1, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    o2 += cute.arch.shuffle_sync_bfly(
                        o2, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    o3 += cute.arch.shuffle_sync_bfly(
                        o3, offset=offset, mask=-1, mask_and_clamp=31
                    )


                if lane_in_group == 0:
                    o[(i_n, i_t, i_hv, v0)] = cutlass.BFloat16(o0)
                    o[(i_n, i_t, i_hv, v1)] = cutlass.BFloat16(o1)
                    o[(i_n, i_t, i_hv, v2)] = cutlass.BFloat16(o2)
                    o[(i_n, i_t, i_hv, v3)] = cutlass.BFloat16(o3)


                if cutlass.const_expr(cache_intermediate_states):
                    for i in cutlass.range_constexpr(vec):
                        r_hb0[i] = cutlass.BFloat16(r_h[0, i])
                        r_hb1[i] = cutlass.BFloat16(r_h[1, i])
                        r_hb2[i] = cutlass.BFloat16(r_h[2, i])
                        r_hb3[i] = cutlass.BFloat16(r_h[3, i])


                    flat_idx = i_n * T * HV + i_t * HV + i_hv
                    it0 = cute.local_tile(
                        intermediate_states,
                        (1, 1, vec),
                        (flat_idx, v0, lane_in_group),
                    )
                    it1 = cute.local_tile(
                        intermediate_states,
                        (1, 1, vec),
                        (flat_idx, v1, lane_in_group),
                    )
                    it2 = cute.local_tile(
                        intermediate_states,
                        (1, 1, vec),
                        (flat_idx, v2, lane_in_group),
                    )
                    it3 = cute.local_tile(
                        intermediate_states,
                        (1, 1, vec),
                        (flat_idx, v3, lane_in_group),
                    )
                    cute.autovec_copy(r_hb0, it0)
                    cute.autovec_copy(r_hb1, it1)
                    cute.autovec_copy(r_hb2, it2)
                    cute.autovec_copy(r_hb3, it3)


            if cutlass.const_expr(
                not disable_state_update and not cache_intermediate_states
            ):
                for i in cutlass.range_constexpr(vec):
                    r_hb0[i] = cutlass.BFloat16(r_h[0, i])
                    r_hb1[i] = cutlass.BFloat16(r_h[1, i])
                    r_hb2[i] = cutlass.BFloat16(r_h[2, i])
                    r_hb3[i] = cutlass.BFloat16(r_h[3, i])
                cute.autovec_copy(r_hb0, ht_w0)
                cute.autovec_copy(r_hb1, ht_w1)
                cute.autovec_copy(r_hb2, ht_w2)
                cute.autovec_copy(r_hb3, ht_w3)


@cute.jit
def run_gdn_decode_bf16state_mtp_ilp4(
    h0_source: cute.Tensor,
    intermediate_states: cute.Tensor,
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    h0_out_indices: cute.Tensor,
    accepted_steps: cute.Tensor,
    ssm_state_indices: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    tile_v_param: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
    disable_state_update: cutlass.Constexpr[bool],
    cache_intermediate_states: cutlass.Constexpr[bool],
    use_packed_fma: cutlass.Constexpr[bool],
    same_pool: cutlass.Constexpr[bool],
    disable_output: cutlass.Constexpr[bool],
    per_request_accepted_steps: cutlass.Constexpr[bool],
    per_token_pool_scatter: cutlass.Constexpr[bool],
    per_token_pool_scatter_flat: cutlass.Constexpr[bool],
    replayssm_rawv: cute.Tensor,
    replayssm_rawk: cute.Tensor,
    replayssm_g: cute.Tensor,
    replayssm_beta: cute.Tensor,
    CACHE_RING: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):

    tile_v = tile_v_param
    vec_size = MTP_VEC_SIZE

    _, _, v_dim, _k_dim = (
        h0_source.layout.shape[0],
        h0_source.layout.shape[1],
        h0_source.layout.shape[2],
        h0_source.layout.shape[3],
    )

    num_v_tiles = cute.ceil_div(v_dim, tile_v)
    B = cute.size(q.shape[0])
    grid_size = B * HV * num_v_tiles

    smem_bytes = 128
    if T > 1:
        smem_bytes = 4 * T * (K + 8) + 4 * T * (K + 8) + 4 * T * 2 + 128

    gdn_decode_bf16state_mtp_ilp4_kernel(
        h0_source,
        intermediate_states,
        vec_size,
        num_v_tiles,
        tile_v,
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        o,
        h0_indices,
        h0_out_indices,
        accepted_steps,
        ssm_state_indices,
        softplus_beta,
        softplus_threshold,
        scale,
        HV,
        T,
        H,
        K,
        V,
        use_qk_l2norm,
        disable_state_update,
        cache_intermediate_states,
        use_packed_fma,
        same_pool,
        disable_output,
        per_request_accepted_steps,
        per_token_pool_scatter,
        per_token_pool_scatter_flat,
        replayssm_rawv,
        replayssm_rawk,
        replayssm_g,
        replayssm_beta,
        CACHE_RING,
    ).launch(
        grid=(grid_size, 1, 1),
        block=[MTP_NUM_THREADS, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


@cute.jit
def _run_wide_vec(
    h0_source: cute.Tensor,
    intermediate_states: cute.Tensor,
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b_gate: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    h0_out_indices: cute.Tensor,
    accepted_steps: cute.Tensor,
    ssm_state_indices: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    tile_v: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
    disable_state_update: cutlass.Constexpr[bool],
    cache_intermediate_states: cutlass.Constexpr[bool],
    use_packed_fma: cutlass.Constexpr[bool],
    same_pool: cutlass.Constexpr[bool],
    disable_output: cutlass.Constexpr[bool],
    recovery_steps: cutlass.Constexpr[int],
    per_request_accepted_steps: cutlass.Constexpr[bool],
    per_token_pool_scatter: cutlass.Constexpr[bool],
    per_token_pool_scatter_flat: cutlass.Constexpr[bool],
    replayssm_rawv: cute.Tensor,
    replayssm_rawk: cute.Tensor,
    replayssm_g: cute.Tensor,
    replayssm_beta: cute.Tensor,
    CACHE_RING: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):


    B = cute.size(q.shape[0])
    num_v_tiles: cutlass.Constexpr[int] = V // tile_v
    grid_size = B * HV * num_v_tiles
    smem_bytes = (
        4 * T * (K + 8)
        + 4 * T * (K + 8)
        + 4 * T * 2
        + 256
    )
    gdn_wide_vec_kernel(
        h0_source,
        intermediate_states,
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b_gate,
        o,
        h0_indices,
        h0_out_indices,
        accepted_steps,
        ssm_state_indices,
        softplus_beta,
        softplus_threshold,
        scale,
        HV,
        T,
        H,
        K,
        V,
        tile_v,
        num_v_tiles,
        use_qk_l2norm,
        disable_state_update,
        cache_intermediate_states,
        use_packed_fma,
        same_pool,
        disable_output,
        recovery_steps,
        per_request_accepted_steps,
        per_token_pool_scatter,
        per_token_pool_scatter_flat,
        replayssm_rawv,
        replayssm_rawk,
        replayssm_g,
        replayssm_beta,
        CACHE_RING,
    ).launch(
        grid=(grid_size, 1, 1),
        block=[NUM_THREADS, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


@cute.jit
def _run_wide_vec_t1(
    h0_source: cute.Tensor,
    intermediate_states: cute.Tensor,
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b_gate: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    h0_out_indices: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    tile_v: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
    disable_state_update: cutlass.Constexpr[bool],
    cache_intermediate_states: cutlass.Constexpr[bool],
    use_packed_fma: cutlass.Constexpr[bool],
    same_pool: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    num_v_tiles: cutlass.Constexpr[int] = V // tile_v
    B = cute.size(q.shape[0])
    grid_size = B * HV * num_v_tiles
    smem_bytes = (
        4 * T * (K + 8) + 4 * T * (K + 8) + 4 * T * 3 + 256
    )
    gdn_wide_vec_kernel_t1(
        h0_source,
        intermediate_states,
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b_gate,
        o,
        h0_indices,
        h0_out_indices,
        softplus_beta,
        softplus_threshold,
        scale,
        HV,
        T,
        H,
        K,
        V,
        tile_v,
        num_v_tiles,
        use_qk_l2norm,
        disable_state_update,
        cache_intermediate_states,
        use_packed_fma,
        same_pool,
    ).launch(
        grid=(grid_size, 1, 1),
        block=[NUM_THREADS, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count


_GPU_MAJOR, _ = torch.cuda.get_device_capability(0)
_USE_PACKED_FMA = _GPU_MAJOR >= 10


def gated_delta_rule(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    q: Optional[torch.Tensor] = None,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    initial_state_source: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    output_state_indices: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = True,
    scale: Optional[float] = None,
) -> torch.Tensor:

    assert q is not None and k is not None and v is not None
    assert b is not None and initial_state_source is not None

    B, T, H, K = q.shape
    assert T == 1, f"This kernel only supports T=1, got T={T}"
    HV = v.shape[2]
    V = v.shape[3]
    assert K == 128 and V == 128, f"K and V must be 128, got K={K}, V={V}"
    assert initial_state_source.dtype == torch.bfloat16
    assert initial_state_indices is not None, (
        "Pool mode is required: pass initial_state_indices. "
        "Non-pool mode is no longer supported by the BF16 GDN kernels."
    )

    if scale is None:
        scale = 1.0 / math.sqrt(K)

    if output_state_indices is not None and output_state_indices.dtype != torch.int32:
        output_state_indices = output_state_indices.to(torch.int32)


    wv_tile_v = _select_wide_vec_tile_v(B, HV)
    if wv_tile_v is not None and wv_tile_v < 64:
        wv_tile_v = None
    if wv_tile_v is not None:


        return gated_delta_rule_t1_wide_vec(
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=initial_state_source,
            initial_state_indices=initial_state_indices,
            output_state_indices=output_state_indices,
            intermediate_states_buffer=None,
            disable_state_update=False,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            scale=scale,
            output=output,
            tile_v=wv_tile_v,
        )


    return gated_delta_rule_mtp(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=initial_state_source,
        initial_state_indices=initial_state_indices,
        output_state_indices=output_state_indices,
        output=output,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        scale=scale,
    )


_compiled_kernels_mtp: dict = {}
_compiled_kernels_wide_vec: dict = {}


def _select_tile_v_for_mtp(B: int, HV: int, V: int, T: int = 1) -> int:

    for tv in [128, 64, 32]:
        num_v_tiles = V // tv
        grid_size = B * HV * num_v_tiles

        if grid_size >= 4 * NUM_SMS:
            return tv
    return 32


def _get_bf16_mtp_config(
    batch_size: int, seq_len: int, num_v_heads: int, v_dim: int
) -> tuple:

    work_units = batch_size * num_v_heads
    if work_units <= 128:

        return min(16, v_dim), 4
    return _select_tile_v_for_mtp(batch_size, num_v_heads, v_dim, seq_len), 4


_WIDE_VEC_WORK_UNITS_THRESHOLD = 128


def _select_wide_vec_tile_v(B: int, HV: int) -> Optional[int]:

    work_units = B * HV
    if work_units >= 1024:
        return 128
    if work_units >= 512:
        return 64
    if work_units >= _WIDE_VEC_WORK_UNITS_THRESHOLD:
        return 32
    return None


def gated_delta_rule_mtp_wide_vec(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    q: Optional[torch.Tensor] = None,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    initial_state_source: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    output_state_indices: Optional[torch.Tensor] = None,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    accepted_steps: Optional[torch.Tensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    disable_state_update: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
    tile_v: int = 128,
    disable_output: bool = False,
    recovery_steps: int = 0,
    cache_ring: bool = False,
    replayssm_rawv: Optional[torch.Tensor] = None,
    replayssm_rawk: Optional[torch.Tensor] = None,
    replayssm_g: Optional[torch.Tensor] = None,
    replayssm_beta: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    global _compiled_kernels_wide_vec

    assert q is not None and k is not None and v is not None
    assert b is not None and initial_state_source is not None

    B_val, T_val, H_val, K_val = q.shape
    HV_val = v.shape[2]
    V_val = v.shape[3]
    pool_size = initial_state_source.shape[0]
    assert K_val == 128 and V_val == 128
    assert initial_state_source.dtype == torch.bfloat16
    assert tile_v in (32, 64, 128), f"tile_v must be 32/64/128, got {tile_v}"
    assert (
        V_val % tile_v == 0 and (tile_v // NUM_GROUPS) % ILP_ROWS == 0
    ), f"tile_v={tile_v} incompatible with 8 groups × ILP=4 layout"

    if cache_ring:
        assert replayssm_rawv is not None and replayssm_rawk is not None
        assert replayssm_g is not None and replayssm_beta is not None
        assert T_val >= 3, f"cache_ring requires T >= 3, got T={T_val}"
        assert (
            replayssm_rawv.dim() == 4
            and replayssm_rawv.shape[1] == HV_val
            and replayssm_rawv.shape[2] >= T_val
            and replayssm_rawv.dtype == torch.bfloat16
        ), "cache_ring: rawv must be a per-layer [slots, HV, >=T, V] bf16 view"
        assert (
            replayssm_rawk.dim() == 4
            and replayssm_rawk.shape[1] == H_val
            and replayssm_rawk.shape[2] >= T_val
            and replayssm_rawk.dtype == torch.bfloat16
        ), "cache_ring: rawk must be a per-layer [slots, H, >=T, K] bf16 view"
        assert (
            replayssm_g.dim() == 3
            and replayssm_g.dtype == torch.float32
            and replayssm_beta.dim() == 3
            and replayssm_beta.dtype == torch.float32
        ), "cache_ring: g/beta must be per-layer [slots, HV, >=T] fp32 views"
    else:
        replayssm_rawv = torch.zeros(
            1, HV_val, T_val, V_val, device=q.device, dtype=torch.bfloat16
        )
        replayssm_rawk = torch.zeros(
            1, H_val, T_val, K_val, device=q.device, dtype=torch.bfloat16
        )
        replayssm_g = torch.zeros(
            1, HV_val, T_val, device=q.device, dtype=torch.float32
        )
        replayssm_beta = torch.zeros(
            1, HV_val, T_val, device=q.device, dtype=torch.float32
        )

    if scale is None:
        scale = 1.0 / math.sqrt(K_val)


    h0_source = initial_state_source


    canonical_pool_stride = (
        HV_val * V_val * K_val,
        V_val * K_val,
        K_val,
        1,
    )
    per_token_pool_scatter_flat = (
        ssm_state_indices is not None
        and intermediate_states_buffer is None
        and tuple(int(s) for s in h0_source.stride()) == canonical_pool_stride
    )

    cache_intermediate_states = intermediate_states_buffer is not None
    if cache_intermediate_states:


        buffer_size = intermediate_states_buffer.shape[0]
        cache_steps = intermediate_states_buffer.shape[1]
        assert buffer_size == B_val, (
            f"intermediate_states_buffer dim 0 ({buffer_size}) must equal "
            f"batch size B={B_val}; the buffer is batch-scoped, not pool-scoped"
        )
        assert cache_steps >= T_val
        assert intermediate_states_buffer.dtype == torch.bfloat16
        intermediate_states = intermediate_states_buffer.reshape(
            B_val * cache_steps * HV_val, V_val, K_val
        )
        if not intermediate_states.is_contiguous():
            intermediate_states = intermediate_states.contiguous()

        effective_disable_final = True
    elif per_token_pool_scatter_flat:


        intermediate_states = h0_source.reshape(-1, V_val, K_val)
        effective_disable_final = disable_state_update
    else:
        intermediate_states = h0_source[:1, :1, :1]
        effective_disable_final = disable_state_update

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    use_packed_fma = _USE_PACKED_FMA


    same_pool = (
        output_state_indices is None or output_state_indices is initial_state_indices
    )


    assert (
        0 <= recovery_steps <= T_val
    ), f"recovery_steps must be in [0, T={T_val}], got {recovery_steps}"
    if recovery_steps > 0:
        assert (
            not cache_intermediate_states
        ), "recovery_steps > 0 is incompatible with intermediate state caching"
        assert not disable_state_update, (
            "recovery_steps > 0 requires state writeback "
            "(disable_state_update=False); the boundary writeback at i_t=K-1 "
            "writes h_K and is the whole point of fused mode"
        )


    per_request_accepted_steps = accepted_steps is not None
    if per_request_accepted_steps:
        assert accepted_steps.shape == (
            B_val,
        ), f"accepted_steps must have shape [B={B_val}], got {accepted_steps.shape}"
        assert (
            accepted_steps.dtype == torch.int32
        ), f"accepted_steps must be int32, got {accepted_steps.dtype}"
        assert accepted_steps.device == q.device


    per_token_pool_scatter = ssm_state_indices is not None
    if per_token_pool_scatter:
        assert (
            intermediate_states_buffer is None
        ), "ssm_state_indices and intermediate_states_buffer are mutually exclusive"
        assert (
            not disable_state_update
        ), "ssm_state_indices requires state writes; disable_state_update must be False"
        assert (
            recovery_steps == 0
        ), "ssm_state_indices + recovery_steps>0 not yet supported (MVP exclusion)"
        assert T_val >= 2, (
            f"ssm_state_indices requires T >= 2 (got T={T_val}); "
            f"for T=1 use output_state_indices"
        )
        assert ssm_state_indices.shape == (B_val, T_val), (
            f"ssm_state_indices must have shape [B={B_val}, T={T_val}], "
            f"got {tuple(ssm_state_indices.shape)}"
        )
        assert (
            ssm_state_indices.dtype == torch.int32
        ), f"ssm_state_indices must be int32, got {ssm_state_indices.dtype}"
        assert ssm_state_indices.device == q.device


    contiguous_pool = initial_state_source.is_contiguous()
    if contiguous_pool:
        pool_size_key = -1
        pool_slot_stride: tuple[int, ...] = (-1,)
    else:
        pool_size_key = pool_size
        pool_slot_stride = tuple(int(s) for s in initial_state_source.stride())
    cache_key = (
        "v3_mtp_bf16_tiled_dynB",
        T_val,
        H_val,
        HV_val,
        K_val,
        V_val,
        pool_size_key,
        pool_slot_stride,
        tile_v,
        effective_disable_final,
        cache_intermediate_states,
        use_qk_l2norm_in_kernel,
        scale,
        softplus_beta,
        softplus_threshold,
        use_packed_fma,
        same_pool,
        disable_output,
        recovery_steps,
        per_request_accepted_steps,
        per_token_pool_scatter,
        per_token_pool_scatter_flat,
        cache_ring,
    )

    if cache_key not in _compiled_kernels_wide_vec:


        _placeholder_indices = torch.arange(1, dtype=torch.int32, device=q.device)
        _placeholder_output = torch.empty(
            1, T_val, HV_val, V_val, device=q.device, dtype=q.dtype
        )
        _placeholder_accepted_steps = torch.zeros(1, dtype=torch.int32, device=q.device)
        _placeholder_ssm_state_indices = torch.zeros(
            1, T_val, dtype=torch.int32, device=q.device
        )

        if contiguous_pool:
            h_ = _mark_slot_dynamic(h0_source)
        else:
            h_ = from_dlpack(h0_source, assumed_align=32, enable_tvm_ffi=True)
        inter_ = from_dlpack(intermediate_states, assumed_align=32, enable_tvm_ffi=True)


        if cache_intermediate_states or per_token_pool_scatter_flat:
            inter_ = _mark_slot_dynamic(intermediate_states)
        q_ = _mark_batch_dynamic(q)
        k_ = _mark_batch_dynamic(k)
        v_ = _mark_batch_dynamic(v)
        a_ = _mark_batch_dynamic(a)
        b_ = _mark_batch_dynamic(b)
        A_log_ = from_dlpack(A_log, assumed_align=32, enable_tvm_ffi=True)
        dt_bias_ = from_dlpack(dt_bias, assumed_align=32, enable_tvm_ffi=True)
        o_ = _mark_batch_dynamic(output if output is not None else _placeholder_output)
        h0_idx_ = _mark_index_dynamic(
            initial_state_indices
            if initial_state_indices is not None
            else _placeholder_indices
        )
        h0_out_idx_ = h0_idx_
        acc_steps_ = _mark_index_dynamic(
            accepted_steps
            if accepted_steps is not None
            else _placeholder_accepted_steps
        )
        ssm_idx_ = _mark_index_dynamic(
            ssm_state_indices
            if ssm_state_indices is not None
            else _placeholder_ssm_state_indices
        )

        if cache_ring:
            rawv_ = _mark_slot_dynamic(replayssm_rawv)
            rawk_ = _mark_slot_dynamic(replayssm_rawk)
            g_ = _mark_slot_dynamic(replayssm_g)
            beta_ = _mark_slot_dynamic(replayssm_beta)
        else:
            rawv_ = from_dlpack(replayssm_rawv, assumed_align=32, enable_tvm_ffi=True)
            rawk_ = from_dlpack(replayssm_rawk, assumed_align=32, enable_tvm_ffi=True)
            g_ = from_dlpack(replayssm_g, assumed_align=32, enable_tvm_ffi=True)
            beta_ = from_dlpack(replayssm_beta, assumed_align=32, enable_tvm_ffi=True)

        _compiled_kernels_wide_vec[cache_key] = {
            "compiled": cute.compile(
                _run_wide_vec,
                h_,
                inter_,
                A_log_,
                a_,
                dt_bias_,
                q_,
                k_,
                v_,
                b_,
                o_,
                h0_idx_,
                h0_out_idx_,
                acc_steps_,
                ssm_idx_,
                softplus_beta,
                softplus_threshold,
                scale,
                HV_val,
                T_val,
                H_val,
                K_val,
                V_val,
                tile_v,
                use_qk_l2norm_in_kernel,
                effective_disable_final,
                cache_intermediate_states,
                use_packed_fma,
                same_pool,
                disable_output,
                recovery_steps,
                per_request_accepted_steps,
                per_token_pool_scatter,
                per_token_pool_scatter_flat,
                rawv_,
                rawk_,
                g_,
                beta_,
                cache_ring,
                stream,
                options="--enable-tvm-ffi --generate-line-info --opt-level 3",
            ),


            "defaults_by_B": {},
        }

    cache = _compiled_kernels_wide_vec[cache_key]
    defaults_by_B = cache["defaults_by_B"]
    if B_val not in defaults_by_B:
        defaults_by_B[B_val] = {
            "indices": torch.arange(B_val, dtype=torch.int32, device=q.device),
            "output": torch.empty(
                B_val, T_val, HV_val, V_val, device=q.device, dtype=q.dtype
            ),
            "accepted_steps": torch.zeros(B_val, dtype=torch.int32, device=q.device),
            "ssm_state_indices": torch.zeros(
                B_val, T_val, dtype=torch.int32, device=q.device
            ),
        }
    defs = defaults_by_B[B_val]
    if initial_state_indices is None:
        initial_state_indices = defs["indices"]
    if output_state_indices is None:


        output_state_indices = initial_state_indices
    if output is None:
        output = defs["output"]
    accepted_steps_arg = (
        accepted_steps if accepted_steps is not None else defs["accepted_steps"]
    )
    ssm_state_indices_arg = (
        ssm_state_indices
        if ssm_state_indices is not None
        else defs["ssm_state_indices"]
    )

    cache["compiled"](
        h0_source,
        intermediate_states,
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        output,
        initial_state_indices,
        output_state_indices,
        accepted_steps_arg,
        ssm_state_indices_arg,
        replayssm_rawv,
        replayssm_rawk,
        replayssm_g,
        replayssm_beta,
        stream,
    )
    return output


def gated_delta_rule_t1_wide_vec(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    q: Optional[torch.Tensor] = None,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    initial_state_source: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    output_state_indices: Optional[torch.Tensor] = None,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    disable_state_update: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
    tile_v: int = 128,
) -> torch.Tensor:

    global _compiled_kernels_wide_vec

    assert q is not None and k is not None and v is not None
    assert b is not None and initial_state_source is not None

    B_val, T_val, H_val, K_val = q.shape
    HV_val = v.shape[2]
    V_val = v.shape[3]
    pool_size = initial_state_source.shape[0]
    assert K_val == 128 and V_val == 128
    assert initial_state_source.dtype == torch.bfloat16
    assert tile_v in (32, 64, 128), f"tile_v must be 32/64/128, got {tile_v}"
    assert (
        V_val % tile_v == 0 and (tile_v // NUM_GROUPS) % ILP_ROWS == 0
    ), f"tile_v={tile_v} incompatible with 8 groups × ILP=4 layout"

    if scale is None:
        scale = 1.0 / math.sqrt(K_val)


    h0_source = initial_state_source

    cache_intermediate_states = intermediate_states_buffer is not None
    if cache_intermediate_states:


        buffer_size = intermediate_states_buffer.shape[0]
        cache_steps = intermediate_states_buffer.shape[1]
        assert buffer_size == B_val, (
            f"intermediate_states_buffer dim 0 ({buffer_size}) must equal "
            f"batch size B={B_val}; the buffer is batch-scoped, not pool-scoped"
        )
        assert cache_steps >= T_val
        assert intermediate_states_buffer.dtype == torch.bfloat16
        intermediate_states = intermediate_states_buffer.reshape(
            B_val * cache_steps * HV_val, V_val, K_val
        )
        if not intermediate_states.is_contiguous():
            intermediate_states = intermediate_states.contiguous()

        effective_disable_final = True
    else:
        intermediate_states = h0_source[:1, :1, :1]
        effective_disable_final = disable_state_update

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    use_packed_fma = _USE_PACKED_FMA


    same_pool = (
        output_state_indices is None or output_state_indices is initial_state_indices
    )


    contiguous_pool = initial_state_source.is_contiguous()
    if contiguous_pool:
        pool_size_key = -1
        pool_slot_stride: tuple[int, ...] = (-1,)
    else:
        pool_size_key = pool_size
        pool_slot_stride = tuple(int(s) for s in initial_state_source.stride())
    cache_key = (
        "v3_mtp_bf16_tiled_dynB",
        T_val,
        H_val,
        HV_val,
        K_val,
        V_val,
        pool_size_key,
        pool_slot_stride,
        tile_v,
        effective_disable_final,
        cache_intermediate_states,
        use_qk_l2norm_in_kernel,
        scale,
        softplus_beta,
        softplus_threshold,
        use_packed_fma,
        same_pool,
    )

    if cache_key not in _compiled_kernels_wide_vec:


        _placeholder_indices = torch.arange(1, dtype=torch.int32, device=q.device)
        _placeholder_output = torch.empty(
            1, T_val, HV_val, V_val, device=q.device, dtype=q.dtype
        )

        if contiguous_pool:
            h_ = _mark_slot_dynamic(h0_source)
        else:
            h_ = from_dlpack(h0_source, assumed_align=32, enable_tvm_ffi=True)
        if cache_intermediate_states:
            inter_ = _mark_slot_dynamic(intermediate_states)
        else:

            inter_ = from_dlpack(
                intermediate_states, assumed_align=32, enable_tvm_ffi=True
            )
        q_ = _mark_batch_dynamic(q)
        k_ = _mark_batch_dynamic(k)
        v_ = _mark_batch_dynamic(v)
        a_ = _mark_batch_dynamic(a)
        b_ = _mark_batch_dynamic(b)
        A_log_ = from_dlpack(A_log, assumed_align=32, enable_tvm_ffi=True)
        dt_bias_ = from_dlpack(dt_bias, assumed_align=32, enable_tvm_ffi=True)
        o_ = _mark_batch_dynamic(output if output is not None else _placeholder_output)
        h0_idx_ = _mark_index_dynamic(
            initial_state_indices
            if initial_state_indices is not None
            else _placeholder_indices
        )
        h0_out_idx_ = h0_idx_

        _compiled_kernels_wide_vec[cache_key] = {
            "compiled": cute.compile(
                _run_wide_vec_t1,
                h_,
                inter_,
                A_log_,
                a_,
                dt_bias_,
                q_,
                k_,
                v_,
                b_,
                o_,
                h0_idx_,
                h0_out_idx_,
                softplus_beta,
                softplus_threshold,
                scale,
                HV_val,
                T_val,
                H_val,
                K_val,
                V_val,
                tile_v,
                use_qk_l2norm_in_kernel,
                effective_disable_final,
                cache_intermediate_states,
                use_packed_fma,
                same_pool,
                stream,
                options="--enable-tvm-ffi --generate-line-info --opt-level 3",
            ),


            "defaults_by_B": {},
        }

    cache = _compiled_kernels_wide_vec[cache_key]
    defaults_by_B = cache["defaults_by_B"]
    if B_val not in defaults_by_B:
        defaults_by_B[B_val] = {
            "indices": torch.arange(B_val, dtype=torch.int32, device=q.device),
            "output": torch.empty(
                B_val, T_val, HV_val, V_val, device=q.device, dtype=q.dtype
            ),
        }
    defs = defaults_by_B[B_val]
    if initial_state_indices is None:
        initial_state_indices = defs["indices"]
    if output_state_indices is None:


        output_state_indices = initial_state_indices
    if output is None:
        output = defs["output"]

    cache["compiled"](
        h0_source,
        intermediate_states,
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        output,
        initial_state_indices,
        output_state_indices,
        stream,
    )
    return output


def gated_delta_rule_mtp(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    q: Optional[torch.Tensor] = None,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    initial_state_source: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    output_state_indices: Optional[torch.Tensor] = None,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    accepted_steps: Optional[torch.Tensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    disable_state_update: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
    disable_output: bool = False,
    recovery_steps: int = 0,
    cache_ring: bool = False,
    replayssm_rawv: Optional[torch.Tensor] = None,
    replayssm_rawk: Optional[torch.Tensor] = None,
    replayssm_g: Optional[torch.Tensor] = None,
    replayssm_beta: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    global _compiled_kernels_mtp

    assert q is not None and k is not None and v is not None
    assert b is not None and initial_state_source is not None

    B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]
    pool_size = initial_state_source.shape[0]
    assert K == 128 and V == 128, f"K and V must be 128, got K={K}, V={V}"
    assert initial_state_source.dtype == torch.bfloat16
    assert initial_state_indices is not None, (
        "Pool mode is required: pass initial_state_indices. "
        "Non-pool mode is no longer supported by the BF16 GDN MTP kernels."
    )

    if cache_ring:
        assert replayssm_rawv is not None and replayssm_rawk is not None
        assert replayssm_g is not None and replayssm_beta is not None
        assert T >= 3, f"cache_ring requires T >= 3, got T={T}"
        assert (
            replayssm_rawv.dim() == 4
            and replayssm_rawv.shape[1] == HV
            and replayssm_rawv.shape[2] >= T
            and replayssm_rawv.dtype == torch.bfloat16
        ), "cache_ring: rawv must be a per-layer [slots, HV, >=T, V] bf16 view"
        assert (
            replayssm_rawk.dim() == 4
            and replayssm_rawk.shape[1] == H
            and replayssm_rawk.shape[2] >= T
            and replayssm_rawk.dtype == torch.bfloat16
        ), "cache_ring: rawk must be a per-layer [slots, H, >=T, K] bf16 view"
        assert (
            replayssm_g.dim() == 3
            and replayssm_g.dtype == torch.float32
            and replayssm_beta.dim() == 3
            and replayssm_beta.dtype == torch.float32
        ), "cache_ring: g/beta must be per-layer [slots, HV, >=T] fp32 views"
    else:
        replayssm_rawv = torch.zeros(1, HV, T, V, device=q.device, dtype=torch.bfloat16)
        replayssm_rawk = torch.zeros(1, H, T, K, device=q.device, dtype=torch.bfloat16)
        replayssm_g = torch.zeros(1, HV, T, device=q.device, dtype=torch.float32)
        replayssm_beta = torch.zeros(1, HV, T, device=q.device, dtype=torch.float32)

    if scale is None:
        scale = 1.0 / math.sqrt(K)

    if output_state_indices is not None and output_state_indices.dtype != torch.int32:
        output_state_indices = output_state_indices.to(torch.int32)


    h0_source = initial_state_source


    cache_intermediate_states = intermediate_states_buffer is not None
    if cache_intermediate_states:
        buffer_size = intermediate_states_buffer.shape[0]
        cache_steps = intermediate_states_buffer.shape[1]
        assert buffer_size == B, (
            f"intermediate_states_buffer dim 0 ({buffer_size}) must equal "
            f"batch size B={B}; the buffer is batch-scoped, not pool-scoped"
        )
        assert (
            cache_steps >= T
        ), f"intermediate_states_buffer dim 1 ({cache_steps}) must be >= T={T}"
        assert intermediate_states_buffer.dtype == torch.bfloat16
        intermediate_states = intermediate_states_buffer.reshape(
            B * cache_steps * HV, V, K
        )
        if not intermediate_states.is_contiguous():
            intermediate_states = intermediate_states.contiguous()
        per_token_pool_scatter_flat = False
    elif ssm_state_indices is not None and tuple(
        int(s) for s in h0_source.stride()
    ) == (HV * V * K, V * K, K, 1):


        intermediate_states = h0_source.reshape(-1, V, K)
        per_token_pool_scatter_flat = True
    else:
        intermediate_states = h0_source[
            :1, :1, :1
        ]
        per_token_pool_scatter_flat = False


    per_token_pool_scatter = ssm_state_indices is not None
    if per_token_pool_scatter:
        assert (
            intermediate_states_buffer is None
        ), "ssm_state_indices and intermediate_states_buffer are mutually exclusive"
        assert (
            not disable_state_update
        ), "ssm_state_indices requires state writes; disable_state_update must be False"
        assert (
            recovery_steps == 0
        ), "ssm_state_indices + recovery_steps>0 not yet supported (MVP exclusion)"
        assert (
            T >= 2
        ), f"ssm_state_indices requires T >= 2 (got T={T}); for T=1 use output_state_indices"
        assert ssm_state_indices.shape == (B, T), (
            f"ssm_state_indices must have shape [B={B}, T={T}], "
            f"got {tuple(ssm_state_indices.shape)}"
        )
        assert (
            ssm_state_indices.dtype == torch.int32
        ), f"ssm_state_indices must be int32, got {ssm_state_indices.dtype}"
        assert (
            ssm_state_indices.device == q.device
        ), f"ssm_state_indices device {ssm_state_indices.device} != q device {q.device}"


    wv_tile_v = _select_wide_vec_tile_v(B, HV) if T >= 2 else None
    if recovery_steps > 0:
        assert wv_tile_v is not None, (
            f"recovery_steps>0 requires the wide_vec path (B*HV >= 128); got "
            f"B={B}, HV={HV} which routes to MTP ILP4 fallback. "
            f"recovery_steps support not implemented in MTP ILP4 yet."
        )


        if recovery_steps == 1:
            if wv_tile_v == 128 and B <= 32:
                wv_tile_v = 64
            elif wv_tile_v == 64 and B <= 8:
                wv_tile_v = 32
    if wv_tile_v is not None:
        return gated_delta_rule_mtp_wide_vec(
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=initial_state_source,
            initial_state_indices=initial_state_indices,
            output_state_indices=output_state_indices,
            intermediate_states_buffer=intermediate_states_buffer,
            accepted_steps=accepted_steps,
            ssm_state_indices=ssm_state_indices,
            disable_state_update=disable_state_update,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            scale=scale,
            output=output,
            tile_v=wv_tile_v,
            disable_output=disable_output,
            recovery_steps=recovery_steps,
            cache_ring=cache_ring,
            replayssm_rawv=replayssm_rawv,
            replayssm_rawk=replayssm_rawk,
            replayssm_g=replayssm_g,
            replayssm_beta=replayssm_beta,
        )


    tile_v, ilp_rows = _get_bf16_mtp_config(B, T, HV, V)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    use_packed_fma = _USE_PACKED_FMA


    same_pool = (
        output_state_indices is None or output_state_indices is initial_state_indices
    )


    per_request_accepted_steps = accepted_steps is not None
    if per_request_accepted_steps:
        assert accepted_steps.shape == (
            B,
        ), f"accepted_steps must have shape [B={B}], got {accepted_steps.shape}"
        assert (
            accepted_steps.dtype == torch.int32
        ), f"accepted_steps must be int32, got {accepted_steps.dtype}"
        assert accepted_steps.device == q.device


    contiguous_pool = initial_state_source.is_contiguous()
    if contiguous_pool:
        pool_size_key = -1
        pool_slot_stride: tuple[int, ...] = (-1,)
    else:
        pool_size_key = pool_size
        pool_slot_stride = tuple(int(s) for s in initial_state_source.stride())
    cache_key = (
        "mtp_bf16_dynB",
        T,
        H,
        HV,
        K,
        V,
        pool_size_key,
        pool_slot_stride,
        tile_v,
        ilp_rows,
        disable_state_update,
        cache_intermediate_states,
        use_qk_l2norm_in_kernel,
        scale,
        softplus_beta,
        softplus_threshold,
        use_packed_fma,
        same_pool,
        disable_output,
        per_request_accepted_steps,
        per_token_pool_scatter,
        per_token_pool_scatter_flat,
        cache_ring,
    )

    if cache_key not in _compiled_kernels_mtp:


        _placeholder_indices = torch.arange(1, dtype=torch.int32, device=q.device)
        _placeholder_output = torch.empty(1, T, HV, V, device=q.device, dtype=q.dtype)
        _placeholder_accepted_steps = torch.zeros(1, dtype=torch.int32, device=q.device)
        _placeholder_ssm_state_indices = torch.zeros(
            1, T, dtype=torch.int32, device=q.device
        )

        if contiguous_pool:
            h_ = _mark_slot_dynamic(h0_source)
        else:
            h_ = from_dlpack(h0_source, assumed_align=32, enable_tvm_ffi=True)


        if cache_intermediate_states or per_token_pool_scatter_flat:
            inter_ = _mark_slot_dynamic(intermediate_states)
        else:
            inter_ = from_dlpack(
                intermediate_states, assumed_align=32, enable_tvm_ffi=True
            )
        q_ = _mark_batch_dynamic(q)
        k_ = _mark_batch_dynamic(k)
        v_ = _mark_batch_dynamic(v)
        a_ = _mark_batch_dynamic(a)
        b_ = _mark_batch_dynamic(b)
        A_log_ = from_dlpack(A_log, assumed_align=32, enable_tvm_ffi=True)
        dt_bias_ = from_dlpack(dt_bias, assumed_align=32, enable_tvm_ffi=True)
        o_ = _mark_batch_dynamic(output if output is not None else _placeholder_output)
        h0_idx_ = _mark_index_dynamic(
            initial_state_indices
            if initial_state_indices is not None
            else _placeholder_indices
        )
        h0_out_idx_ = h0_idx_
        acc_steps_ = _mark_index_dynamic(
            accepted_steps
            if accepted_steps is not None
            else _placeholder_accepted_steps
        )
        ssm_idx_ = _mark_index_dynamic(
            ssm_state_indices
            if ssm_state_indices is not None
            else _placeholder_ssm_state_indices
        )

        if cache_ring:
            rawv_ = _mark_slot_dynamic(replayssm_rawv)
            rawk_ = _mark_slot_dynamic(replayssm_rawk)
            g_ = _mark_slot_dynamic(replayssm_g)
            beta_ = _mark_slot_dynamic(replayssm_beta)
        else:
            rawv_ = from_dlpack(replayssm_rawv, assumed_align=32, enable_tvm_ffi=True)
            rawk_ = from_dlpack(replayssm_rawk, assumed_align=32, enable_tvm_ffi=True)
            g_ = from_dlpack(replayssm_g, assumed_align=32, enable_tvm_ffi=True)
            beta_ = from_dlpack(replayssm_beta, assumed_align=32, enable_tvm_ffi=True)

        _compiled_kernels_mtp[cache_key] = {
            "compiled": cute.compile(
                run_gdn_decode_bf16state_mtp_ilp4,
                h_,
                inter_,
                A_log_,
                a_,
                dt_bias_,
                q_,
                k_,
                v_,
                b_,
                o_,
                h0_idx_,
                h0_out_idx_,
                acc_steps_,
                ssm_idx_,
                softplus_beta,
                softplus_threshold,
                scale,
                HV,
                T,
                H,
                K,
                V,
                tile_v,
                use_qk_l2norm_in_kernel,
                disable_state_update,
                cache_intermediate_states,
                use_packed_fma,
                same_pool,
                disable_output,
                per_request_accepted_steps,
                per_token_pool_scatter,
                per_token_pool_scatter_flat,
                rawv_,
                rawk_,
                g_,
                beta_,
                cache_ring,
                stream,
                options="--enable-tvm-ffi --generate-line-info --opt-level 3",
            ),


            "defaults_by_B": {},
        }

    cache = _compiled_kernels_mtp[cache_key]
    defaults_by_B = cache["defaults_by_B"]
    if B not in defaults_by_B:
        defaults_by_B[B] = {
            "indices": torch.arange(B, dtype=torch.int32, device=q.device),
            "output": torch.empty(B, T, HV, V, device=q.device, dtype=q.dtype),
            "accepted_steps": torch.zeros(B, dtype=torch.int32, device=q.device),
            "ssm_state_indices": torch.zeros(B, T, dtype=torch.int32, device=q.device),
        }
    defs = defaults_by_B[B]
    if initial_state_indices is None:
        initial_state_indices = defs["indices"]
    if output is None:
        output = defs["output"]
    if output_state_indices is None:
        output_state_indices = initial_state_indices
    accepted_steps_arg = (
        accepted_steps if accepted_steps is not None else defs["accepted_steps"]
    )
    ssm_state_indices_arg = (
        ssm_state_indices
        if ssm_state_indices is not None
        else defs["ssm_state_indices"]
    )

    cache["compiled"](
        h0_source,
        intermediate_states,
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        output,
        initial_state_indices,
        output_state_indices,
        accepted_steps_arg,
        ssm_state_indices_arg,
        replayssm_rawv,
        replayssm_rawk,
        replayssm_g,
        replayssm_beta,
        stream,
    )

    return output


gated_delta_rule_bf16state_cooprow = gated_delta_rule
gated_delta_rule_bf16state_cooprow_mtp = gated_delta_rule_mtp
