






























#pragma once

#ifndef USE_ROCM
#error "route_radix4_hip.cuh targets CDNA; it uses amdgcn DPP and wave64 ballots"
#endif

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

inline constexpr uint32_t kRadix4NumExperts = 896;
inline constexpr uint32_t kRadix4TopK = 16;
inline constexpr uint32_t kRadix4Block = 256;
inline constexpr uint32_t kRadix4Wave = 64;



inline constexpr float kAiterSigmoidLog2E = 1.44269504088896340736f;

struct RouteRadix4Params {
  const void* __restrict__ scores;
  const void* __restrict__ bias;
  fp32_t* __restrict__ out_w;
  int32_t* __restrict__ out_i;
  uint32_t stride_scores;
  uint32_t stride_out;
  fp32_t routed_scaling_factor;
  bool renormalize;
};

namespace radix4 {







static __device__ __constant__ uint8_t kAiterTieLaneRank[64] = {
    56, 57, 58, 59, 63, 62, 61, 60, 52, 53, 54, 55, 51, 50, 49, 48, 40, 41, 42, 43, 47, 46,
    45, 44, 36, 37, 38, 39, 35, 34, 33, 32, 24, 25, 26, 27, 31, 30, 29, 28, 20, 21, 22, 23,
    19, 18, 17, 16, 8,  9,  10, 11, 15, 14, 13, 12, 4,  5,  6,  7,  3,  2,  1,  0,
};









SGL_DEVICE uint32_t tie_priority(int expert) {
  const int group = expert >> 2;
  const int lane = group & 63;
  const int bank = group >> 6;
  const int rank = static_cast<int>(kAiterTieLaneRank[lane]);
  assert((rank < 32) == (lane >= 32));
  const int group_rank = (rank < 32) ? (rank * 3 + bank) : (96 + (rank - 32) * 4 + bank);
  return static_cast<uint32_t>((group_rank << 2) + (expert & 3));
}




template <int NWORDS>
SGL_DEVICE int tie_rank(const uint64_t* bits, uint32_t p) {
  const uint32_t w = p >> 6;
  int n = 0;
#pragma unroll
  for (uint32_t j = 0; j < NWORDS; ++j) {
    const uint64_t below = (j < w) ? ~0ull : ((j == w) ? ((1ull << (p & 63)) - 1ull) : 0ull);
    n += __popcll(bits[j] & below);
  }
  return n;
}

SGL_DEVICE float load_score(const bf16_t* p, int i) {
  return __uint_as_float(static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(p)[i]) << 16);
}

SGL_DEVICE float load_score(const fp32_t* p, int i) {
  return p[i];
}





SGL_DEVICE uint32_t sortable(float f) {
  uint32_t u = __float_as_uint(f);

  if (u == 0x80000000u) u = 0u;
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}



SGL_DEVICE uint32_t rank_key(float x) {
  return (x == x) ? sortable(x) : 0u;
}

template <int CTRL, int RM, int BM, int N>
SGL_DEVICE void dpp_add_stage(uint32_t (&x)[N]) {
#pragma unroll
  for (int j = 0; j < N; ++j)
    x[j] += static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(x[j]), CTRL, RM, BM, false));
}






template <int N>
SGL_DEVICE void wave_sum_dpp(uint32_t (&x)[N]) {
  dpp_add_stage<0x111, 0xf, 0xf>(x);
  dpp_add_stage<0x112, 0xf, 0xf>(x);
  dpp_add_stage<0x114, 0xf, 0xe>(x);
  dpp_add_stage<0x118, 0xf, 0xc>(x);
  dpp_add_stage<0x142, 0xa, 0xf>(x);
  dpp_add_stage<0x143, 0xc, 0xf>(x);
}

template <int CTRL, int RM, int BM>
SGL_DEVICE float dpp_fadd_stage(float x) {
  const int moved = __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, x), CTRL, RM, BM, false);
  return x + __builtin_bit_cast(float, moved);
}





SGL_DEVICE void stage_wave_sum(float v, int lane, int wid, float* out) {
  v = dpp_fadd_stage<0x111, 0xf, 0xf>(v);
  v = dpp_fadd_stage<0x112, 0xf, 0xf>(v);
  v = dpp_fadd_stage<0x114, 0xf, 0xe>(v);
  v = dpp_fadd_stage<0x118, 0xf, 0xc>(v);
  v = dpp_fadd_stage<0x142, 0xa, 0xf>(v);
  v = dpp_fadd_stage<0x143, 0xc, 0xf>(v);
  if (lane == static_cast<int>(kRadix4Wave) - 1) out[wid] = v;
}

}

template <typename T, int EXPERTS, int TOPK, int BLOCK>
__global__ __launch_bounds__(BLOCK) void route_radix4_kernel(__grid_constant__ const RouteRadix4Params params) {
  constexpr int WAVE = static_cast<int>(kRadix4Wave);
  constexpr int NWAVE = BLOCK / WAVE;
  constexpr int VPT = (EXPERTS + BLOCK - 1) / BLOCK;




  constexpr int CHUNK = 15;
  constexpr int NACC = (VPT + CHUNK - 1) / CHUNK;
  static_assert(BLOCK % WAVE == 0, "block must be whole waves");
  static_assert(TOPK <= WAVE, "topk must fit in one lane-indexed row");
  static_assert(VPT <= 32, "alive mask is 32 bits");

  const int token = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid % WAVE;
  const int wid = tid / WAVE;
  const auto* srow = static_cast<const T*>(params.scores) + static_cast<size_t>(token) * params.stride_scores;
  const auto* sbias = static_cast<const T*>(params.bias);

  float sig[VPT];
  uint32_t key[VPT];

  uint32_t valid = (VPT >= 32) ? 0xffffffffu : ((1u << VPT) - 1u);





  uint32_t or_all = 0u, and_all = 0xffffffffu;
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    const int e = tid + i * BLOCK;
    if (e < EXPERTS) {
      const float x = radix4::load_score(srow, e);
      const float g = __builtin_amdgcn_rcpf(1.0f + exp2f(-kAiterSigmoidLog2E * x));
      sig[i] = g;
      key[i] = radix4::rank_key(g + radix4::load_score(sbias, e));
      or_all |= key[i];
      and_all &= key[i];
    } else {
      sig[i] = 0.0f;
      key[i] = 0u;
      valid &= ~(1u << i);
    }
  }
  uint32_t alive = valid;


  constexpr int TIE_WORDS = (EXPERTS + 63) / 64;

  __shared__ uint32_t s_hist[2][NWAVE][8];
  __shared__ uint32_t s_pre[2][NWAVE];
  __shared__ uint64_t s_tie[TIE_WORDS];
  __shared__ float s_w[TOPK];
  __shared__ float s_wsum[NWAVE];
  __shared__ int s_id[TOPK];
  __shared__ uint32_t s_key[TOPK];
  __shared__ int s_cnt;

#pragma unroll
  for (int s = 32; s > 0; s >>= 1) {
    or_all |= __shfl_xor(or_all, s, WAVE);
    and_all &= __shfl_xor(and_all, s, WAVE);
  }
  if (lane == 0) {
    s_pre[0][wid] = or_all;
    s_pre[1][wid] = and_all;
  }
  __syncthreads();
#pragma unroll
  for (int w = 0; w < NWAVE; ++w) {
    or_all |= s_pre[0][w];
    and_all &= s_pre[1][w];
  }





  const uint32_t diff = or_all ^ and_all;
  const int start = diff ? (31 - __clz(diff)) : -1;
  uint32_t pivot = (start >= 31) ? 0u : (and_all & ~((1u << (start + 1)) - 1u));

  int need = TOPK;

  int bend = 0;
  bool capped = (start < 0);







#pragma unroll 1
  for (int b = (start < 0) ? -4 : (start >> 2) << 2; b >= 0; b -= 4) {


    const int buf = (b >> 2) & 1;

    uint64_t h[NACC];
#pragma unroll
    for (int a = 0; a < NACC; ++a)
      h[a] = 0ull;
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
      const uint32_t bin = (key[i] >> b) & 15u;
      h[i / CHUNK] += static_cast<uint64_t>((alive >> i) & 1u) << (4 * bin);
    }


    uint32_t p[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      uint32_t lo = 0u, hi = 0u;
#pragma unroll
      for (int a = 0; a < NACC; ++a) {
        lo += static_cast<uint32_t>((h[a] >> (8 * j)) & 0xfull);
        hi += static_cast<uint32_t>((h[a] >> (8 * j + 4)) & 0xfull);
      }
      p[j] = lo | (hi << 16);
    }
    radix4::wave_sum_dpp(p);
    if (lane == WAVE - 1) {
#pragma unroll
      for (int j = 0; j < 8; ++j)
        s_hist[buf][wid][j] = p[j];
    }
    __syncthreads();





    int c = 0;
    if (lane < 16) {
      const int d = 15 - lane;
#pragma unroll
      for (int w = 0; w < NWAVE; ++w)
        c += static_cast<int>((s_hist[buf][w][d >> 1] >> ((d & 1) * 16)) & 0xffffu);
    }



    int cum = c;
    cum += __builtin_amdgcn_update_dpp(0, cum, 0x111, 0xf, 0xf, false);
    cum += __builtin_amdgcn_update_dpp(0, cum, 0x112, 0xf, 0xf, false);
    cum += __builtin_amdgcn_update_dpp(0, cum, 0x114, 0xf, 0xf, false);
    cum += __builtin_amdgcn_update_dpp(0, cum, 0x118, 0xf, 0xf, false);



    const unsigned long long mk = __ballot(lane < 16 && cum >= need);
    const int L0 = __ffsll(mk) - 1;
    const int sel = 15 - L0;
    const int nsel = __builtin_amdgcn_readlane(c, L0);
    const int above = __builtin_amdgcn_readlane(cum, L0) - nsel;

    need -= above;
    pivot |= static_cast<uint32_t>(sel) << b;
    bend = b;
#pragma unroll
    for (int i = 0; i < VPT; ++i)
      if (((key[i] >> b) & 15u) != static_cast<uint32_t>(sel)) alive &= ~(1u << i);




    if (nsel == need) break;
    if (b == 0) capped = true;
  }

  const uint32_t pmask = ~((1u << bend) - 1u);




  float wsum = 0.0f;
  if (!capped) {



    if (tid == 0) s_cnt = 0;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
      if (((valid >> i) & 1u) && (key[i] & pmask) >= pivot) {
        const int pos = atomicAdd(&s_cnt, 1);
        if (pos < TOPK) {
          s_w[pos] = sig[i];
          s_id[pos] = tid + i * BLOCK;
          s_key[pos] = key[i];
          wsum += sig[i];
        }
      }
    }
    radix4::stage_wave_sum(wsum, lane, wid, s_wsum);
    __syncthreads();
  } else {






    for (int j = tid; j < TIE_WORDS; j += BLOCK)
      s_tie[j] = 0ull;
    if (tid == 0) s_cnt = 0;
    __syncthreads();

    uint32_t eqm = 0u;
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
      if (((valid >> i) & 1u) && (key[i] & pmask) == pivot) {
        eqm |= 1u << i;
        const uint32_t p = radix4::tie_priority(tid + i * BLOCK);
        atomicOr(&s_tie[p >> 6], 1ull << (p & 63));
      }
    }
    __syncthreads();



#pragma unroll
    for (int i = 0; i < VPT; ++i) {
      if (!((valid >> i) & 1u)) continue;
      const int e = tid + i * BLOCK;
      const bool tied = ((eqm >> i) & 1u) != 0u;
      const bool taken =
          (key[i] & pmask) > pivot || (tied && radix4::tie_rank<TIE_WORDS>(s_tie, radix4::tie_priority(e)) < need);
      if (taken) {
        const int pos = atomicAdd(&s_cnt, 1);
        if (pos < TOPK) {
          s_w[pos] = sig[i];
          s_id[pos] = e;
          s_key[pos] = key[i];
          wsum += sig[i];
        }
      }
    }
    radix4::stage_wave_sum(wsum, lane, wid, s_wsum);
    __syncthreads();
  }

  if (tid < TOPK) {







    const uint32_t k = s_key[tid];
    const int id = s_id[tid];
    const auto p = static_cast<int>(radix4::tie_priority(id));
    int rank = 0;
#pragma unroll
    for (int q = 0; q < TOPK; ++q) {
      const auto kq = static_cast<uint32_t>(__builtin_amdgcn_readlane(static_cast<int>(k), q));
      const int pq = __builtin_amdgcn_readlane(p, q);
      rank += (kq > k || (kq == k && pq < p)) ? 1 : 0;
    }

    float scale = params.routed_scaling_factor;
    if (params.renormalize) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < NWAVE; ++w)
        sum += s_wsum[w];


      scale /= (sum > 0.0f) ? sum : 1.0f;
    }
    const size_t o = static_cast<size_t>(token) * params.stride_out + rank;
    params.out_w[o] = s_w[tid] * scale;
    params.out_i[o] = id;
  }
}

struct RouteRadix4Kernel {
  static void
  run(const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView bias,
      const tvm::ffi::TensorView out_w,
      const tvm::ffi::TensorView out_i,
      int64_t topk,
      double routed_scaling_factor,
      bool renormalize) {
    using namespace host;

    auto M_ = SymbolicSize{"num_tokens"};
    auto N_ = SymbolicSize{"num_experts"};
    auto K_ = SymbolicSize{"topk"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    auto score_dtype = SymbolicDType{};
    TensorMatcher({M_, N_})
        .with_dtype<bf16_t, fp32_t>(score_dtype)
        .with_device(device_)
        .with_strides({-1, 1})
        .verify(scores);

    TensorMatcher({N_}).with_dtype<bf16_t, fp32_t>(score_dtype).with_device(device_).verify(bias);
    TensorMatcher({M_, K_}).with_dtype<fp32_t>().with_device(device_).verify(out_w);
    TensorMatcher({M_, K_}).with_dtype<int32_t>().with_device(device_).verify(out_i);

    RuntimeCheck(
        N_.unwrap() == kRadix4NumExperts && K_.unwrap() == kRadix4TopK && topk == kRadix4TopK,
        "route_radix4 is specialized for N=896, K=16");

    const auto M = static_cast<uint32_t>(M_.unwrap());
    if (M == 0) return;

    const auto params = RouteRadix4Params{
        .scores = scores.data_ptr(),
        .bias = bias.data_ptr(),
        .out_w = static_cast<fp32_t*>(out_w.data_ptr()),
        .out_i = static_cast<int32_t*>(out_i.data_ptr()),
        .stride_scores = static_cast<uint32_t>(scores.stride(0)),
        .stride_out = static_cast<uint32_t>(out_w.stride(0)),
        .routed_scaling_factor = static_cast<fp32_t>(routed_scaling_factor),
        .renormalize = renormalize,
    };

    constexpr auto kExperts = static_cast<int>(kRadix4NumExperts);
    constexpr auto kTopK = static_cast<int>(kRadix4TopK);
    constexpr auto kBlock = static_cast<int>(kRadix4Block);
    const auto device = device_.unwrap();
    if (score_dtype.is_type<bf16_t>()) {
      LaunchKernel(M, kBlock, device)(route_radix4_kernel<bf16_t, kExperts, kTopK, kBlock>, params);
    } else {
      LaunchKernel(M, kBlock, device)(route_radix4_kernel<fp32_t, kExperts, kTopK, kBlock>, params);
    }
  }
};

}
