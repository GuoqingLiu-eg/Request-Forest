/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/













#pragma once

#include "config.h"
#include "helpers.h"
#include <cuda_fp8.h>





using namespace cute;


#include "dense_fp8_transpose_v.h"

#include "dense_fp8_utils.h"

namespace sm90 {
namespace fwd {

template <typename Kernel, typename TMAParamsT>
__global__ void sparse_mla_q8kv8_prefill_kernel(
    __grid_constant__ const SparseMlaQ8Kv8PrefillParams params, __grid_constant__ const TMAParamsT tma_params);

template <int D_QK, bool HAVE_TOPK_LENGTH, bool HAVE_ATTN_SINK>
struct SparseMlaQ8Kv8PrefillKernel {
  static constexpr int D_Q = D_QK;
  static constexpr int D_K = D_QK;
  static constexpr int D_V = 512;

  static constexpr int B_H = 64;
  static constexpr int B_TOPK = 64;
  static constexpr int NUM_THREADS = 128 * 3;
  static constexpr float MAX_INIT_VAL = -1e30f;

  using fp8_t = cutlass::float_e4m3_t;

  enum NamedBarriers : uint32_t {
    wg0_bunch_0_ready = 0,
    wg1_bunch_0_ready = 1,
    vt0_left_ready = 2,
    vt0_right_ready = 3,
    sL_ready = 4,
    warpgroup0_sync = 5,
    warpgroup1_sync = 6,
    epilogue_sync = 7,


  };

  static constexpr uint32_t q_load_done = epilogue_sync;
  static constexpr uint32_t vt1_for_wg0 = warpgroup0_sync;
  static constexpr uint32_t vt1_for_wg1 = warpgroup1_sync;
  static constexpr uint32_t s_consumed_ready = sL_ready;






  template <int NUM_TILES>
  using SmemLayoutQTiles_FP8 = decltype(coalesce(
      tile_to_shape(GMMA::Layout_K_SW64_Atom<fp8_t>{}, Shape<Int<B_H>, Int<64 * NUM_TILES>>{}, Step<_1, _2>{}),
      Shape<_1, _1>{}));


  template <int NUM_TILES>
  using SmemLayoutKTiles_FP8 = decltype(coalesce(
      tile_to_shape(GMMA::Layout_K_SW64_Atom<fp8_t>{}, Shape<Int<B_TOPK>, Int<64 * NUM_TILES>>{}, Step<_1, _2>{}),
      Shape<_1, _1>{}));



  template <int NUM_TILES>
  using SmemLayoutVtTiles_FP8 = decltype(coalesce(
      tile_to_shape(GMMA::Layout_K_SW64_Atom<fp8_t>{}, Shape<Int<64 * NUM_TILES>, Int<B_TOPK>>{}, Step<_1, _2>{}),
      Shape<_1, _1>{}));


  template <int NUM_TILES>
  using SmemLayoutOTiles = decltype(coalesce(
      tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<B_H>, Int<64 * NUM_TILES>>{}, Step<_1, _2>{}),
      Shape<_1, _1>{}));

  using SmemLayoutQ = SmemLayoutQTiles_FP8<D_Q / 64>;
  using SmemLayoutK = SmemLayoutKTiles_FP8<D_Q / 64>;
  using SmemLayoutVt = SmemLayoutVtTiles_FP8<D_V / 64>;
  using SmemLayoutHalfVt = SmemLayoutVtTiles_FP8<D_V / 64 / 2>;
  using SmemLayoutO = SmemLayoutOTiles<D_V / 64>;

  using SmemTransposeV = SmemTransposeFp8_64x64<B_TOPK, D_V>;





  using TiledMMA_QK = decltype(make_tiled_mma(GMMA::MMA_64x64x32_F32E4M3E4M3_SS_TN<>{}, Layout<Shape<_1, _1, _1>>{}));


  using TiledMMA_PV_LocalP =
      decltype(make_tiled_mma(GMMA::MMA_64x256x32_F32E4M3E4M3_RS_TN<>{}, Layout<Shape<_1, _1, _1>>{}));


  using TiledMMA_PV_RemoteP =
      decltype(make_tiled_mma(GMMA::MMA_64x256x32_F32E4M3E4M3_SS_TN<>{}, Layout<Shape<_1, _1, _1>>{}));




  struct SharedMemoryPlan {
    union {
      array_aligned<fp8_t, cosize_v<SmemLayoutQ>> q;
      array_aligned<bf16, cosize_v<SmemLayoutO>> o;
    } q_o;
    array_aligned<fp8_t, cosize_v<SmemLayoutK>> k[2];




    array_aligned<fp8_t, cosize_v<SmemLayoutHalfVt>> vt_loc[2];
    array_aligned<fp8_t, cosize_v<SmemLayoutHalfVt>> vt_rem[2];
    array_aligned<fp8_t, 128 * 36> s[2];










    bool is_kv_valid[2][2][B_TOPK];






    float2 sM[2][32];
    float2 sL[64];
    float final_max_logits[64], final_lse[64];
    transac_bar_t bar_q, bar_k0_ready[2], bar_k1_ready[2],
        bar_is_kv_valid_ready[2];
    transac_bar_t bar_k0_free, bar_k1_free;


    transac_bar_t bar_vt_free[2];
  };

  struct TmaParams_t {
    CUtensorMap tensor_map_O;
  };




  template <typename TMAParamType>
  static __device__ __forceinline__ void
  devfunc(const SparseMlaQ8Kv8PrefillParams& params, const TMAParamType& tma_params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)) || (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
    const int q_h_idx = blockIdx.x % (params.h_q / B_H);
    const int s_q_idx = blockIdx.x / (params.h_q / B_H);
    const int warpgroup_idx = cutlass::canonical_warp_group_idx();
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int idx_in_warpgroup = threadIdx.x % 128;

    extern __shared__ char wksp_buf[];
    SharedMemoryPlan& plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);

    const float q_scale = __ldg(params.q_scale_ptr);
    const float kv_scale = __ldg(params.kv_scale_ptr);
    const float qk_combined_scale_div_log2 = q_scale * kv_scale * params.sm_scale_div_log2;

    if (warp_idx == 0 && elect_one_sync()) {
      cute::prefetch_tma_descriptor(&tma_params.tensor_map_O);

      plan.bar_q.init(1);
      plan.bar_k0_free.init(128);
      plan.bar_k1_free.init(128);
      CUTE_UNROLL
      for (int i = 0; i < 2; ++i) {
        plan.bar_k0_ready[i].init(128);
        plan.bar_k1_ready[i].init(128);
      }
      CUTE_UNROLL
      for (int i = 0; i < 2; ++i) {
        plan.bar_is_kv_valid_ready[i].init(16);
      }
      CUTE_UNROLL
      for (int i = 0; i < 2; ++i) {

        plan.bar_vt_free[i].init(256);
      }
      fence_barrier_init();
    }

    __syncthreads();
    const int topk_length = HAVE_TOPK_LENGTH ? __ldg(params.topk_length + s_q_idx) : params.topk;
    const int num_topk_blocks = HAVE_TOPK_LENGTH ? ku::ceil_div(topk_length, (int)B_TOPK)
                                                 : (int)((unsigned int)params.topk / (unsigned int)B_TOPK);




    if (warpgroup_idx == 0 || warpgroup_idx == 1) {
      cutlass::arch::warpgroup_reg_alloc<216>();





      if (warpgroup_idx == 0) {
        const fp8_t* gQ = reinterpret_cast<const fp8_t*>(params.q) + s_q_idx * (int64_t)params.stride_q_s_q +
                          q_h_idx * B_H * (int64_t)params.stride_q_h_q;





        constexpr int Q_GROUP_SIZE = 4;
        constexpr int Q_NUM_GROUPS = 128 / Q_GROUP_SIZE;
        constexpr int Q_ROWS_PER_GROUP = B_H / Q_NUM_GROUPS;
        int q_ig = idx_in_warpgroup % Q_GROUP_SIZE;
        int q_gg = idx_in_warpgroup / Q_GROUP_SIZE;
        fp8_t* sQ_base = &(make_tensor(make_smem_ptr(plan.q_o.q.data()), SmemLayoutQTiles_FP8<1>{})(q_gg, q_ig * 16));
        constexpr int NUM_Q_TILES = D_Q / 64;
        int64_t q_cache_policy = createpolicy_evict_first();
        CUTE_UNROLL
        for (int lr = 0; lr < Q_ROWS_PER_GROUP; ++lr) {
          CUTE_UNROLL
          for (int ti = 0; ti < NUM_Q_TILES; ++ti) {

            bool q_pred = (ti * 64 + q_ig * 16 + 16) <= D_Q;
            cp_async_cacheglobal_l2_prefetch_256B(
                gQ + (q_gg + lr * Q_NUM_GROUPS) * (int64_t)params.stride_q_h_q + ti * 64 + q_ig * 16,
                sQ_base + ti * (B_H * 64) + lr * Q_NUM_GROUPS * 64,
                q_pred,
                q_cache_policy);
          }
        }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 0;\n" ::);
      }
      fence_view_async_shared();
      NamedBarrier::arrive_and_wait(256, q_load_done);




      float rM[2] = {MAX_INIT_VAL, MAX_INIT_VAL};
      float rL[2] = {0.0f, 0.0f};





      float peer_scale[2] = {1.0f, 1.0f};
      Tensor rO = partition_fragment_C(TiledMMA_PV_LocalP{}, Shape<Int<B_H>, Int<D_V / 2>>{});
      Tensor rP = partition_fragment_C(TiledMMA_QK{}, Shape<Int<B_H>, Int<B_TOPK>>{});
      cute::fill(rO, 0.0f);



      using rP_fp8_layout_t = decltype(flash::convert_layout_acc_Aregs<TiledMMA_PV_LocalP>(
          partition_fragment_C(TiledMMA_QK{}, Shape<Int<B_H>, Int<B_TOPK>>{}).layout()));
      Tensor rP_fp8_local = make_tensor<fp8_t>(rP_fp8_layout_t{});

      bool cur_bar_wait_phase = 0;
      bool kv_valid_phase[2] = {false, false};
      struct Warpgroup0 {};
      struct Warpgroup1 {};


      auto qkt_gemm_one_tile = [&](auto wg_tag, int tile_idx, bool clear_accum) {
        constexpr bool IS_WG1 = std::is_same_v<decltype(wg_tag), Warpgroup1>;
        TiledMMA_QK tiled_mma_QK;
        Tensor sQ_tile = make_tensor(make_smem_ptr(plan.q_o.q.data() + tile_idx * B_H * 64), SmemLayoutQTiles_FP8<1>{});
        Tensor sK_tile =
            make_tensor(make_smem_ptr(plan.k[(int)IS_WG1].data() + tile_idx * B_TOPK * 64), SmemLayoutKTiles_FP8<1>{});
        gemm_ss(clear_accum, tiled_mma_QK, sQ_tile, sK_tile, rP, idx_in_warpgroup);
      };

      auto mask_rP = [&](auto wg_tag, int block_idx) {
        constexpr bool IS_WG1 = std::is_same_v<decltype(wg_tag), Warpgroup1>;







        int _kv_buf = (block_idx >> 1) & 1;
        plan.bar_is_kv_valid_ready[_kv_buf].wait(kv_valid_phase[_kv_buf]);
        kv_valid_phase[_kv_buf] ^= 1;
        CUTE_UNROLL
        for (int row_idx = 0; row_idx < 2; ++row_idx) {
          CUTE_UNROLL
          for (int i = row_idx * 2; i < size(rP); i += 4) {
            int col = 8 * (i / 4) + (idx_in_warpgroup % 4) * 2;
            if (!plan.is_kv_valid[IS_WG1][_kv_buf][col]) rP(i) = -INFINITY;
            if (!plan.is_kv_valid[IS_WG1][_kv_buf][col + 1]) rP(i + 1) = -INFINITY;
          }
        }
      };




      auto online_softmax_and_rescale_o = [&](auto wg_tag, int _par) {

        constexpr bool IS_WG1 = std::is_same_v<decltype(wg_tag), Warpgroup1>;
        const float scale = qk_combined_scale_div_log2;
        float r_sM[2];
        if constexpr (IS_WG1) {
          *(float2*)r_sM = plan.sM[_par][idx_in_warpgroup / 4];
        }
        float new_maxs[2];
        CUTE_UNROLL
        for (int row_idx = 0; row_idx < 2; ++row_idx) {
          float cur_max = -INFINITY;
          CUTE_UNROLL
          for (int i = row_idx * 2; i < size(rP); i += 4) {
            cur_max = max(cur_max, max(rP(i), rP(i + 1)));
          }
          cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
          cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));
          cur_max *= scale;
          new_maxs[row_idx] = max(IS_WG1 ? r_sM[row_idx] : rM[row_idx], cur_max);



          if constexpr (IS_WG1) {
            peer_scale[row_idx] = fmaxf(exp2f(r_sM[row_idx] - new_maxs[row_idx]), exp2f(-30.0f));
          }
          float scale_for_o = exp2f(rM[row_idx] - new_maxs[row_idx]);
          CUTE_UNROLL
          for (int i = row_idx * 2; i < size(rO); i += 4) {
            rO(i) *= scale_for_o;
            rO(i + 1) *= scale_for_o;
          }
          float cur_sum = 0;
          CUTE_UNROLL
          for (int i = row_idx * 2; i < size(rP); i += 4) {
            float p0 = exp2f(rP(i) * scale - new_maxs[row_idx]);
            float p1 = exp2f(rP(i + 1) * scale - new_maxs[row_idx]);
            rP(i) = p0;
            rP(i + 1) = p1;
            cur_sum += p0 + p1;
          }
          rL[row_idx] = rL[row_idx] * scale_for_o + cur_sum;
        }
        __syncwarp();
        if (idx_in_warpgroup % 4 == 0) {
          plan.sM[_par][idx_in_warpgroup / 4] = *(float2*)new_maxs;
        }
        rM[0] = new_maxs[0];
        rM[1] = new_maxs[1];



        flash::permute_Cregs_fp8(rP);

        Tensor rP_acc = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<TiledMMA_PV_LocalP>(rP.layout()));

        flash::convert_type_out(rP_acc, rP_fp8_local);
      };

      auto reduce_L = [&]() {
        rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 1);
        rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 2);
        rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 1);
        rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 2);
        if (idx_in_warpgroup % 4 == 0) plan.sL[threadIdx.x / 4] = *(float2*)(rL);
        NamedBarrier::arrive_and_wait(256, NamedBarriers::sL_ready);
        float2 peer_L = plan.sL[(threadIdx.x / 4) ^ 32];
        rL[0] += peer_L.x;
        rL[1] += peer_L.y;
      };

      auto store_O = [&]() {
        float scale_factors[2];
        CUTE_UNROLL
        for (int i = 0; i < 2; ++i) {
          if constexpr (HAVE_ATTN_SINK) {
            int attn_sink_idx = q_h_idx * B_H + get_AorC_row_idx(i, idx_in_warpgroup);
            float attn_sink = __ldg(params.attn_sink + attn_sink_idx) * CUDART_L2E_F;
            scale_factors[i] = kv_scale / (rL[i] + exp2f(attn_sink - rM[i]));
          } else {
            scale_factors[i] = kv_scale / rL[i];
          }
          if (rL[i] == 0.0f) scale_factors[i] = 0.0f;
        }

        Tensor sO_tile =
            make_tensor(make_smem_ptr(plan.q_o.o.data() + warpgroup_idx * B_H * (D_V / 2)), SmemLayoutOTiles<4>{});
        bf16* stsm_addrs[4];
        int stsm_row = (idx_in_warpgroup / 32) * 16 + (idx_in_warpgroup % 16);
        CUTE_UNROLL
        for (int i = 0; i < 64 / 16; ++i) {
          stsm_addrs[i] = &sO_tile(stsm_row, (idx_in_warpgroup % 32 / 16 * 8) + 16 * i);
        }
        bool s2g_pred = idx_in_warpgroup == 0;

        warpgroup_wait<0>();
        warpgroup_fence_operand(rO);
        CUTE_UNROLL
        for (int tile_idx = 0; tile_idx < (D_V / 2) / 64; tile_idx += 1) {
          constexpr int NUM_ELEMS_EACH_TILE = B_H * 64 / 128;
          bf16 cur_rOb[NUM_ELEMS_EACH_TILE];
          CUTE_UNROLL
          for (int i = 0; i < NUM_ELEMS_EACH_TILE; ++i) {
            float out_value = rO(tile_idx * NUM_ELEMS_EACH_TILE + i) * scale_factors[i % 4 >= 2];
            cur_rOb[i] = (bf16)out_value;
          }
          CUTE_UNROLL
          for (int i = 0; i < 64 / 16; ++i) {
            SM90_U32x4_STSM_N::copy(
                *reinterpret_cast<uint32_t*>(cur_rOb + i * 8 + 0),
                *reinterpret_cast<uint32_t*>(cur_rOb + i * 8 + 2),
                *reinterpret_cast<uint32_t*>(cur_rOb + i * 8 + 4),
                *reinterpret_cast<uint32_t*>(cur_rOb + i * 8 + 6),
                *reinterpret_cast<uint128_t*>(stsm_addrs[i] + tile_idx * (B_H * 64)));
          }

          cute::tma_store_fence();
          NamedBarrier::arrive_and_wait(
              128, warpgroup_idx ? NamedBarriers::warpgroup1_sync : NamedBarriers::warpgroup0_sync);
          if (s2g_pred) {
            int g_tile_idx = warpgroup_idx * 4 + tile_idx;
            SM90_TMA_STORE_3D::copy(
                &tma_params.tensor_map_O,
                plan.q_o.o.data() + g_tile_idx * (B_H * 64),
                g_tile_idx * 64,
                q_h_idx * B_H,
                s_q_idx);
          }
        }
        cute::tma_store_arrive();
      };





      constexpr int kP_per_thread = 32;



      constexpr int kP_stride = 36;

      auto save_rP_fp8_to_sS = [&](fp8_t* sS_data) {
        uint32_t* dst = reinterpret_cast<uint32_t*>(sS_data + idx_in_warpgroup * kP_stride);
        uint32_t* src = reinterpret_cast<uint32_t*>(&rP_fp8_local(0));
        CUTE_UNROLL
        for (int i = 0; i < kP_per_thread / 4; i++) {
          dst[i] = src[i];
        }
      };

      auto load_sS_to_rP = [&](fp8_t* sS_data) {
        uint32_t* src = reinterpret_cast<uint32_t*>(sS_data + idx_in_warpgroup * kP_stride);
        uint32_t* dst = reinterpret_cast<uint32_t*>(&rP_fp8_local(0));
        CUTE_UNROLL
        for (int i = 0; i < kP_per_thread / 4; i++) {
          dst[i] = src[i];
        }
      };









      if (warpgroup_idx == 0) {
        auto pipelined_wait_and_qkt_gemm_l = [&]() __attribute__((always_inline)) {
          plan.bar_k0_ready[0].wait(cur_bar_wait_phase);
          qkt_gemm_one_tile(Warpgroup0{}, 0, true);
          qkt_gemm_one_tile(Warpgroup0{}, 1, false);
          qkt_gemm_one_tile(Warpgroup0{}, 2, false);
          qkt_gemm_one_tile(Warpgroup0{}, 3, false);
          warpgroup_commit_batch();
        };

        auto pipelined_wait_and_qkt_gemm_r = [&]() __attribute__((always_inline)) {
          plan.bar_k0_ready[1].wait(cur_bar_wait_phase);
          qkt_gemm_one_tile(Warpgroup0{}, 4, false);
          qkt_gemm_one_tile(Warpgroup0{}, 5, false);
          qkt_gemm_one_tile(Warpgroup0{}, 6, false);
          qkt_gemm_one_tile(Warpgroup0{}, 7, false);
          if constexpr (D_QK == 576) {
            qkt_gemm_one_tile(Warpgroup0{}, 8, false);
          }
          warpgroup_commit_batch();
        };

        auto rescale_rO = [&](float scales[2]) {
          CUTE_UNROLL
          for (int row = 0; row < 2; ++row) {
            CUTE_UNROLL
            for (int i = row * 2; i < size(rO); i += 4) {
              rO(i) *= scales[row];
              rO(i + 1) *= scales[row];
            }
            rL[row] *= scales[row];
          }
        };

        CUTE_NO_UNROLL
        for (int block_idx = 0; block_idx < num_topk_blocks; block_idx += 2) {

          Tensor sVt0l = make_tensor(make_smem_ptr(plan.vt_loc[0].data()), SmemLayoutHalfVt{});

          if (block_idx == 0) {
            pipelined_wait_and_qkt_gemm_l();
            pipelined_wait_and_qkt_gemm_r();
            warpgroup_wait<0>();
            warpgroup_fence_operand(rP);
            plan.bar_k0_free.arrive();
          }

          mask_rP(Warpgroup0{}, block_idx);
          online_softmax_and_rescale_o(Warpgroup0{}, (block_idx >> 1) & 1);

          save_rP_fp8_to_sS(plan.s[0].data());





          NamedBarrier::arrive_and_wait(256, NamedBarriers::wg0_bunch_0_ready);



          NamedBarrier::arrive_and_wait(256, vt0_left_ready);


          gemm_rs(false, TiledMMA_PV_LocalP{}, rP_fp8_local, sVt0l, rO, idx_in_warpgroup);
          warpgroup_commit_batch();


          NamedBarrier::arrive_and_wait(256, NamedBarriers::wg1_bunch_0_ready);
          float new_rM[2], scale_factors_arr[2];
          *(float2*)new_rM = plan.sM[(block_idx >> 1) & 1][idx_in_warpgroup / 4];
          CUTE_UNROLL
          for (int i = 0; i < 2; ++i) {
            scale_factors_arr[i] = exp2f(rM[i] - new_rM[i]);
            rM[i] = new_rM[i];
          }

          warpgroup_wait<0>();
          warpgroup_fence_operand(rO);
          warpgroup_fence_operand(rP_fp8_local);
          plan.bar_vt_free[0].arrive();

          load_sS_to_rP(plan.s[1].data());
          NamedBarrier::arrive_and_wait(256, s_consumed_ready);


          NamedBarrier::arrive_and_wait(256, vt1_for_wg0);


          rescale_rO(scale_factors_arr);

          Tensor sVt1l = make_tensor(make_smem_ptr(plan.vt_rem[0].data()), SmemLayoutHalfVt{});
          gemm_rs(false, TiledMMA_PV_LocalP{}, rP_fp8_local, sVt1l, rO, idx_in_warpgroup);
          warpgroup_commit_batch();

          cur_bar_wait_phase ^= 1;

          if (block_idx + 2 < num_topk_blocks) {
            pipelined_wait_and_qkt_gemm_l();
            warpgroup_wait<1>();
            warpgroup_fence_operand(rO);
            warpgroup_fence_operand(rP_fp8_local);
            plan.bar_vt_free[1].arrive();
            pipelined_wait_and_qkt_gemm_r();
            warpgroup_wait<0>();
            warpgroup_fence_operand(rP);
            plan.bar_k0_free.arrive();
          } else {
            warpgroup_wait<0>();
            warpgroup_fence_operand(rO);
            plan.bar_vt_free[1].arrive();
          }
        }

        reduce_L();
        store_O();

      } else {




        auto pipelined_wait_and_qkt_gemm_r_wg1 = [&]() __attribute__((always_inline)) {

          plan.bar_k1_ready[1].wait(cur_bar_wait_phase);
          qkt_gemm_one_tile(Warpgroup1{}, 4, true);
          qkt_gemm_one_tile(Warpgroup1{}, 5, false);
          qkt_gemm_one_tile(Warpgroup1{}, 6, false);
          qkt_gemm_one_tile(Warpgroup1{}, 7, false);
          if constexpr (D_QK == 576) {
            qkt_gemm_one_tile(Warpgroup1{}, 8, false);
          }
          warpgroup_commit_batch();
        };

        auto pipelined_wait_and_qkt_gemm_l_wg1 = [&]() __attribute__((always_inline)) {
          plan.bar_k1_ready[0].wait(cur_bar_wait_phase);
          qkt_gemm_one_tile(Warpgroup1{}, 0, false);
          qkt_gemm_one_tile(Warpgroup1{}, 1, false);
          qkt_gemm_one_tile(Warpgroup1{}, 2, false);
          qkt_gemm_one_tile(Warpgroup1{}, 3, false);
          warpgroup_commit_batch();
        };

        CUTE_NO_UNROLL
        for (int block_idx = 0; block_idx < num_topk_blocks; block_idx += 2) {

          Tensor sVt1r = make_tensor(make_smem_ptr(plan.vt_loc[1].data()), SmemLayoutHalfVt{});

          if (block_idx == 0) {
            pipelined_wait_and_qkt_gemm_r_wg1();
            pipelined_wait_and_qkt_gemm_l_wg1();
            warpgroup_wait<0>();
            warpgroup_fence_operand(rP);
            plan.bar_k1_free.arrive();
          }

          mask_rP(Warpgroup1{}, block_idx);

          NamedBarrier::arrive_and_wait(256, NamedBarriers::wg0_bunch_0_ready);
          online_softmax_and_rescale_o(Warpgroup1{}, (block_idx >> 1) & 1);

          save_rP_fp8_to_sS(plan.s[1].data());







          NamedBarrier::arrive_and_wait(256, NamedBarriers::wg1_bunch_0_ready);


          NamedBarrier::arrive_and_wait(256, vt1_for_wg1);


          gemm_rs(false, TiledMMA_PV_LocalP{}, rP_fp8_local, sVt1r, rO, idx_in_warpgroup);
          warpgroup_commit_batch();

          warpgroup_wait<0>();
          warpgroup_fence_operand(rO);
          warpgroup_fence_operand(rP_fp8_local);
          plan.bar_vt_free[1].arrive();



          CUTE_UNROLL
          for (int row = 0; row < 2; ++row) {
            float inv = 1.0f / peer_scale[row];
            CUTE_UNROLL
            for (int i = row * 2; i < size(rO); i += 4) {
              rO(i) *= inv;
              rO(i + 1) *= inv;
            }
          }
          load_sS_to_rP(plan.s[0].data());
          NamedBarrier::arrive_and_wait(256, s_consumed_ready);



          NamedBarrier::arrive_and_wait(256, vt0_right_ready);

          Tensor sVt0r = make_tensor(make_smem_ptr(plan.vt_rem[1].data()), SmemLayoutHalfVt{});
          gemm_rs(false, TiledMMA_PV_LocalP{}, rP_fp8_local, sVt0r, rO, idx_in_warpgroup);
          warpgroup_commit_batch();

          if (block_idx + 2 < num_topk_blocks) {
            cur_bar_wait_phase ^= 1;

            pipelined_wait_and_qkt_gemm_r_wg1();
            warpgroup_wait<1>();
            warpgroup_fence_operand(rO);

            CUTE_UNROLL
            for (int row = 0; row < 2; ++row) {
              CUTE_UNROLL
              for (int i = row * 2; i < size(rO); i += 4) {
                rO(i) *= peer_scale[row];
                rO(i + 1) *= peer_scale[row];
              }
            }
            warpgroup_fence_operand(rP_fp8_local);
            plan.bar_vt_free[0].arrive();
            pipelined_wait_and_qkt_gemm_l_wg1();
            warpgroup_wait<0>();
            warpgroup_fence_operand(rP);
            plan.bar_k1_free.arrive();
          } else {
            warpgroup_wait<0>();
            warpgroup_fence_operand(rO);

            CUTE_UNROLL
            for (int row = 0; row < 2; ++row) {
              CUTE_UNROLL
              for (int i = row * 2; i < size(rO); i += 4) {
                rO(i) *= peer_scale[row];
                rO(i + 1) *= peer_scale[row];
              }
            }
            plan.bar_vt_free[0].arrive();
          }
        }

        reduce_L();
        store_O();

        if (idx_in_warpgroup % 4 == 0) {
          for (int row = 0; row < 2; ++row) {
            int real_row = get_AorC_row_idx(row, idx_in_warpgroup);
            bool is_no_valid_tokens = rL[row] == 0.0f;
            plan.final_max_logits[real_row] = is_no_valid_tokens ? -INFINITY : rM[row] * CUDART_LN2_F;
            plan.final_lse[real_row] = is_no_valid_tokens ? +INFINITY : logf(rL[row]) + rM[row] * CUDART_LN2_F;
          }

          asm volatile("" ::: "memory");
        }

        NamedBarrier::arrive_and_wait(128, NamedBarriers::warpgroup1_sync);
        if (idx_in_warpgroup == 0) {
          int g_offset = s_q_idx * params.h_q + q_h_idx * B_H;
          SM90_BULK_COPY_S2G::copy(plan.final_max_logits, params.max_logits + g_offset, B_H * sizeof(float));
          SM90_BULK_COPY_S2G::copy(plan.final_lse, params.lse + g_offset, B_H * sizeof(float));
          cute::tma_store_arrive();
        }
      }

    } else {



      cutlass::arch::warpgroup_reg_dealloc<72>();

      constexpr int GROUP_SIZE = 8, NUM_GROUPS = 128 / GROUP_SIZE;
      constexpr int NUM_ROWS_PER_GROUP = B_TOPK / NUM_GROUPS;
      int idx_in_group = idx_in_warpgroup % GROUP_SIZE;
      int group_idx = idx_in_warpgroup / GROUP_SIZE;
      int* gIndices = params.indices + s_q_idx * params.stride_indices_s_q;

      int tile_shift = idx_in_group / 4;
      int col_in_tile = (idx_in_group % 4) * 16;
      fp8_t* my_sK_base =
          &(make_tensor(make_smem_ptr(plan.k[0].data()), SmemLayoutKTiles_FP8<1>{})(group_idx, col_in_tile)) +
          tile_shift * (B_TOPK * 64);
      const fp8_t* my_gKV_base = reinterpret_cast<const fp8_t*>(params.kv) + idx_in_group * 16;

      int64_t token_indices[2][NUM_ROWS_PER_GROUP];
      bool is_token_valid[2][NUM_ROWS_PER_GROUP];



      const int pad_base = params.s_kv - params.topk;
      auto load_token_indices = [&](int block_idx) {
        CUTE_UNROLL
        for (int buf_idx = 0; buf_idx < 2; ++buf_idx) {
          CUTE_UNROLL
          for (int local_row = 0; local_row < NUM_ROWS_PER_GROUP; ++local_row) {
            int offs = (block_idx + buf_idx) * B_TOPK + local_row * NUM_GROUPS + group_idx;
            int t = __ldg(gIndices + offs);
            bool is_cur_token_valid;
            if constexpr (!HAVE_TOPK_LENGTH) {


















              const bool t_is_pad = (t < 0);
              t = t_is_pad ? (pad_base + offs) : t;
              is_cur_token_valid = !t_is_pad;
            } else {
              is_cur_token_valid = (t >= 0 && t < params.s_kv) && (offs < topk_length);
            }
            token_indices[buf_idx][local_row] = (int64_t)t * (int64_t)params.stride_kv_s_kv;
            is_token_valid[buf_idx][local_row] = is_cur_token_valid;
          }
        }
      };

      int64_t cache_policy = createpolicy_evict_last();

      auto copy_tiles = [&](int buf_idx, int smem_buf, int tile_start, int tile_end) {
        CUTE_UNROLL
        for (int local_row = 0; local_row < NUM_ROWS_PER_GROUP; ++local_row) {
          int64_t token_index = token_indices[buf_idx][local_row];
          CUTE_UNROLL
          for (int tile_idx = tile_start; tile_idx < tile_end; tile_idx += 2) {
            int phys_tile = tile_idx + tile_shift;
            if constexpr ((D_K % 128) != 0) {
              if (phys_tile >= (D_K / 64)) continue;
            }
            bool kv_pred = is_token_valid[buf_idx][local_row] && phys_tile < (D_K / 64);
            cp_async_cacheglobal_l2_prefetch_256B(
                my_gKV_base + token_index + tile_idx * 64,
                my_sK_base +
                    (smem_buf * cosize_v<SmemLayoutK> + tile_idx * (B_TOPK * 64) + local_row * NUM_GROUPS * 64),
                kv_pred,
                cache_policy);
          }
        }
      };

      auto commit_to_mbar = [&](transac_bar_t& bar) { cutlass::arch::cpasync_barrier_arrive_noinc((uint64_t*)(&bar)); };


      SmemTransposeV smem_transpose_v;
      using SmemLayoutTransposeV_t = typename SmemTransposeV::SmemLayoutTransposeV;
      using SmemLayoutTransposeVt_t = typename SmemTransposeV::SmemLayoutTransposeVt;




      using SmemTransposeV_half_t = SmemTransposeFp8_64x64<B_TOPK, D_V / 2>;
      using SmemLayoutTransposeVt_half_t = typename SmemTransposeV_half_t::SmemLayoutTransposeVt;


      auto transpose_v_to_half = [&](int smem_k_buf, fp8_t* vt_dst, int tile_start, int tile_end) {
        Tensor sV_src = as_position_independent_swizzle_tensor(
            make_tensor(make_smem_ptr(plan.k[smem_k_buf].data()), SmemLayoutTransposeV_t{}));
        Tensor sVt_dst =
            as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(vt_dst), SmemLayoutTransposeVt_half_t{}));

        static_assert((D_V / 64 / 2) % 2 == 0, "half tile count must be even for pair transpose");
        CUTE_UNROLL
        for (int j = tile_start; j < tile_end; j += 2) {
          smem_transpose_v.transpose_pair(
              flatten(sV_src(_, 0, j)),
              flatten(sVt_dst(_, 0, j - tile_start)),
              flatten(sV_src(_, 0, j + 1)),
              flatten(sVt_dst(_, 0, j + 1 - tile_start)));
        }
        asm volatile("" ::: "memory");
      };

      int cur_bar_wait_phase_prod = 1;




      load_token_indices(0);

      CUTE_NO_UNROLL
      for (int block_idx = 0; block_idx < num_topk_blocks; block_idx += 2) {

        plan.bar_k0_free.wait(cur_bar_wait_phase_prod);
        plan.bar_k1_free.wait(cur_bar_wait_phase_prod);





        if (idx_in_group == 0) {
          CUTE_UNROLL
          for (int buf_idx = 0; buf_idx < 2; ++buf_idx)
            CUTE_UNROLL
          for (int local_row = 0; local_row < NUM_ROWS_PER_GROUP; ++local_row)
            plan.is_kv_valid[buf_idx][(block_idx >> 1) & 1][local_row * NUM_GROUPS + group_idx] =
                is_token_valid[buf_idx][local_row];
          plan.bar_is_kv_valid_ready[(block_idx >> 1) & 1].arrive();
        }

        copy_tiles(0, 0, 0, 4);
        commit_to_mbar(plan.bar_k0_ready[0]);
        asm volatile("cp.async.commit_group;\n" ::);

        constexpr int kv_tile_end = D_K / 64;

        copy_tiles(1, 1, 4, kv_tile_end);
        commit_to_mbar(plan.bar_k1_ready[1]);

        copy_tiles(0, 0, 4, kv_tile_end);
        commit_to_mbar(plan.bar_k0_ready[1]);

        copy_tiles(1, 1, 0, 4);
        commit_to_mbar(plan.bar_k1_ready[0]);
        asm volatile("cp.async.commit_group;\n" ::);




        asm volatile("cp.async.wait_group 1;\n" ::);


        fence_view_async_shared();
        asm volatile("bar.sync 7, 128;\n" ::: "memory");

        if (block_idx > 0) {
          plan.bar_vt_free[0].wait(cur_bar_wait_phase_prod);
        }



        if (block_idx + 2 < num_topk_blocks) {
          load_token_indices(block_idx + 2);
        }

        transpose_v_to_half(0, plan.vt_loc[0].data(), 0, 4);




















        asm volatile("cp.async.wait_group 0;\n" ::);


        fence_view_async_shared();
        asm volatile("bar.sync 7, 128;\n" ::: "memory");
        NamedBarrier::arrive(256, vt0_left_ready);


        if (block_idx > 0) {
          plan.bar_vt_free[1].wait(cur_bar_wait_phase_prod);
        }
        transpose_v_to_half(1, plan.vt_rem[0].data(), 0, 4);


        fence_view_async_shared();
        NamedBarrier::arrive(256, vt1_for_wg0);


        transpose_v_to_half(0, plan.vt_rem[1].data(), 4, 8);

        transpose_v_to_half(1, plan.vt_loc[1].data(), 4, 8);

        fence_view_async_shared();
        NamedBarrier::arrive(256, vt0_right_ready);
        NamedBarrier::arrive(256, vt1_for_wg1);

        asm volatile("bar.sync 7, 128;\n" ::: "memory");

        cur_bar_wait_phase_prod ^= 1;
      }
    }

    cute::tma_store_wait<0>();
#else
    if (cute::thread0()) {
      CUTE_INVALID_CONTROL_PATH("This kernel only supports sm90");
    }
#endif
  }




  static void run(const SparseMlaQ8Kv8PrefillParams& params) {
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.topk % (2 * B_TOPK) == 0);
    KU_ASSERT(params.topk > 0);
    KU_ASSERT(params.h_q % B_H == 0);

    CUtensorMap tensor_map_O;
    {
      uint64_t size[3] = {(uint64_t)D_V, (uint64_t)params.h_q, (uint64_t)params.s_q};
      uint64_t stride[2] = {D_V * sizeof(bf16), D_V * params.h_q * sizeof(bf16)};
      uint32_t box_size[3] = {64, B_H, 1};
      uint32_t elem_stride[3] = {1, 1, 1};
      CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
          &tensor_map_O,
          CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
          3,
          params.out,
          size,
          stride,
          box_size,
          elem_stride,
          CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
          CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
          CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
          CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      KU_ASSERT(res == CUresult::CUDA_SUCCESS);
    }

    TmaParams_t tma_p = {tensor_map_O};

    auto kernel = &sparse_mla_q8kv8_prefill_kernel<
        SparseMlaQ8Kv8PrefillKernel<D_QK, HAVE_TOPK_LENGTH, HAVE_ATTN_SINK>,
        TmaParams_t>;

    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    cutlass::ClusterLaunchParams launch_params = {
        dim3((params.h_q / B_H) * params.s_q, 1, 1), dim3(NUM_THREADS, 1, 1), dim3(1, 1, 1), smem_size, params.stream};
    cutlass::launch_kernel_on_cluster(launch_params, (void*)kernel, params, tma_p);
    KU_CHECK_KERNEL_LAUNCH();
  }
};




template <typename Kernel, typename TMAParamsT>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, 1) sparse_mla_q8kv8_prefill_kernel(
    __grid_constant__ const SparseMlaQ8Kv8PrefillParams params, __grid_constant__ const TMAParamsT tma_params) {
  Kernel::devfunc(params, tma_params);
}




template <int D_QK, bool HAVE_TOPK_LENGTH, bool HAVE_ATTN_SINK>
void run_sparse_mla_q8kv8_prefill_kernel(const SparseMlaQ8Kv8PrefillParams& params) {
  SparseMlaQ8Kv8PrefillKernel<D_QK, HAVE_TOPK_LENGTH, HAVE_ATTN_SINK>::run(params);
}

}
}
