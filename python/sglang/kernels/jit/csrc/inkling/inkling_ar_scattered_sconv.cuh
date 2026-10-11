







































#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include "inkling_ar_barrier.cuh"
#include <bit>
#include <cstdint>
#include <cuda_bf16.h>
#include <mutex>
#include <type_traits>
#include <unordered_map>

namespace sglang {

constexpr uint32_t kSsVecElems = 8;
constexpr int kSsPadSlot = -1;
constexpr uint32_t kSsMaxHcW = 6144;
constexpr uint32_t kSsTileElems = 17920;

struct ArScatteredSconvParams {
  const void* __restrict__ mc_in;
  void* __restrict__ mc_out;
  void* __restrict__ x_scratch;
  void* __restrict__ cache;
  const void* __restrict__ safe_idx;
  const void* __restrict__ cache_mask;
  const void* __restrict__ ci;
  const void* __restrict__ has_init;
  const void* __restrict__ cu;
  const void* __restrict__ si;
  const void* __restrict__ weight;
  const void* __restrict__ track_rows;
  const void* __restrict__ track_mask;
  const void* __restrict__ track_dst;


  const void* __restrict__ out_local;
  const void* __restrict__ norm_gamma;
  void* __restrict__ norm_residual;
  void* __restrict__ norm_out;
  void* const* __restrict__ flag_ptrs;
  uint32_t* __restrict__ state;
  int64_t cache_stride_slot;
  int64_t cache_stride_w;
  int64_t track_dst_stride;
  float norm_eps;
  uint32_t rank;
  uint32_t T;
  uint32_t H;
  uint32_t Hc;
  uint32_t B;
  uint32_t chunk_rows;
  uint32_t cvec_chunks;
  uint32_t track_from_cache;
  uint32_t use_tile;
  uint32_t need_scratch;





  uint32_t cache_col0;
  uint32_t full_update;
};












__device__ __forceinline__ void ss_fused_norm_tail(const ArScatteredSconvParams& p) {
  const auto* outp = static_cast<const __nv_bfloat16*>(p.out_local);
  const auto* gamma = static_cast<const __nv_bfloat16*>(p.norm_gamma);
  auto* resid = static_cast<__nv_bfloat16*>(p.norm_residual);
  auto* nout = static_cast<__nv_bfloat16*>(p.norm_out);
  const uint32_t hvecs = p.H / kSsVecElems;
  __shared__ float red[32];
  for (uint32_t r = blockIdx.x; r < p.T; r += gridDim.x) {
    const int64_t base = static_cast<int64_t>(r) * p.H;
    float ssq = 0.0f;
    for (uint32_t i = threadIdx.x; i < hvecs; i += blockDim.x) {
      const uint32_t c = i * kSsVecElems;
      const uint4 ov = *reinterpret_cast<const uint4*>(outp + base + c);
      const uint4 rv = *reinterpret_cast<const uint4*>(resid + base + c);
      const auto* oh = reinterpret_cast<const __nv_bfloat16*>(&ov);
      const auto* rh = reinterpret_cast<const __nv_bfloat16*>(&rv);
      __nv_bfloat16 sb[kSsVecElems];
#pragma unroll
      for (int j = 0; j < static_cast<int>(kSsVecElems); ++j) {
        const float v = __bfloat162float(oh[j]) + __bfloat162float(rh[j]);
        ssq += v * v;
        sb[j] = __float2bfloat16(v);
      }
      *reinterpret_cast<uint4*>(resid + base + c) = *reinterpret_cast<const uint4*>(sb);
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      ssq += __shfl_down_sync(0xffffffffu, ssq, off);
    }
    if ((threadIdx.x & 31u) == 0) red[threadIdx.x >> 5] = ssq;
    __syncthreads();
    if (threadIdx.x < 32) {
      float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.0f;
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
      }
      if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    const float rms = rsqrtf(red[0] / static_cast<float>(p.H) + p.norm_eps);
    for (uint32_t i = threadIdx.x; i < hvecs; i += blockDim.x) {
      const uint32_t c = i * kSsVecElems;
      const uint4 sv = *reinterpret_cast<const uint4*>(resid + base + c);
      const uint4 gv = *reinterpret_cast<const uint4*>(gamma + c);
      const auto* sh = reinterpret_cast<const __nv_bfloat16*>(&sv);
      const auto* gh = reinterpret_cast<const __nv_bfloat16*>(&gv);
      __nv_bfloat16 ob[kSsVecElems];
#pragma unroll
      for (int j = 0; j < static_cast<int>(kSsVecElems); ++j) {
        ob[j] = __float2bfloat16(__bfloat162float(sh[j]) * rms * __bfloat162float(gh[j]));
      }
      *reinterpret_cast<uint4*>(nout + base + c) = *reinterpret_cast<const uint4*>(ob);
    }
    __syncthreads();
  }
}

template <typename DType, uint32_t kNumGPU, int W, bool USE_SILU, bool USE_RESIDUAL, bool kPerBlockBarrier>
__global__
__launch_bounds__(1024, 1) void inkling_ar_scattered_sconv_kernel(const __grid_constant__ ArScatteredSconvParams p) {
  static_assert(std::is_same_v<DType, __nv_bfloat16>, "multimem path is bf16-only");
  constexpr int W1 = W - 1;




  __shared__ alignas(16) __nv_bfloat16 smem_w[kSsMaxHcW];




  __shared__ alignas(16) __nv_bfloat16 smem_x[kSsTileElems];


  const bool use_smw = p.T >= 8192;
  if (use_smw) {
    const auto* wg = static_cast<const __nv_bfloat16*>(p.weight);
    const uint32_t nw = p.Hc * W;
    for (uint32_t i = threadIdx.x; i < nw; i += blockDim.x)
      smem_w[i] = wg[i];
    __syncthreads();
  }


  if constexpr (kPerBlockBarrier) {
    inkling_ar::block_system_barrier<kNumGPU>(p.state, p.flag_ptrs, p.rank);
  } else {
    inkling_ar::grid_system_barrier<kNumGPU>(p.state, p.flag_ptrs, p.rank, 0,                    false);
  }

  const uint32_t stride = gridDim.x * blockDim.x;
  const auto* si = static_cast<const int32_t*>(p.si);
  const auto* cu = static_cast<const int64_t*>(p.cu);
  const auto* safe_idx = static_cast<const int64_t*>(p.safe_idx);
  const auto* cmask = static_cast<const bool*>(p.cache_mask);
  const auto* cache = static_cast<const __nv_bfloat16*>(p.cache);
  const auto* mc_in = static_cast<const __nv_bfloat16*>(p.mc_in);
  auto* mc_out = static_cast<__nv_bfloat16*>(p.mc_out);
  auto* scratch = static_cast<__nv_bfloat16*>(p.x_scratch);











  const uint32_t cvecs = p.Hc / kSsVecElems;
  const uint32_t token_chunks = (p.T + p.chunk_rows - 1) / p.chunk_rows;
  const uint32_t total_chunks = token_chunks * p.cvec_chunks;
  const uint32_t cvec_per = (cvecs + p.cvec_chunks - 1) / p.cvec_chunks;
  const uint32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;

  for (uint32_t chunk = blockIdx.x; chunk < total_chunks; chunk += gridDim.x) {
    const uint32_t tc = chunk / p.cvec_chunks;
    const uint32_t cc = chunk % p.cvec_chunks;
    const uint32_t t0 = tc * p.chunk_rows;
    const uint32_t t1 = min(t0 + p.chunk_rows, p.T);
    const uint32_t c0 = min(cc * cvec_per, cvecs);
    const uint32_t c1 = min(c0 + cvec_per, cvecs);
    const uint32_t ncv = c1 - c0;
    if (ncv == 0) continue;
    const uint32_t h0 = t0 > W1 ? t0 - W1 : 0;



    const uint32_t ncv8 = ncv * kSsVecElems;
    {
      const uint32_t rows = t1 - h0;
      const uint32_t items = rows * ncv;
      for (uint32_t i = threadIdx.x; i < items; i += blockDim.x) {
        const uint32_t t = h0 + i / ncv;
        const uint32_t cv = i % ncv;
        const uint32_t lc = (c0 + cv) * kSsVecElems;
        const __nv_bfloat16* addr = mc_in + static_cast<int64_t>(t) * p.H + p.rank * p.Hc + lc;
        uint4 v;
        asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                     : "l"(addr));
        if (p.use_tile) {
          *reinterpret_cast<uint4*>(smem_x + (t - h0) * ncv8 + cv * kSsVecElems) = v;
          if (p.need_scratch && t >= t0) {
            *reinterpret_cast<uint4*>(scratch + static_cast<int64_t>(t) * p.Hc + lc) = v;
          }
        } else {
          *reinterpret_cast<uint4*>(scratch + static_cast<int64_t>(t) * p.Hc + lc) = v;
        }
      }
    }
    __syncthreads();


    {
      const uint32_t rows = t1 - t0;
      const uint32_t items = rows * ncv;
      for (uint32_t i = threadIdx.x; i < items; i += blockDim.x) {
        const uint32_t t = t0 + i / ncv;
        const uint32_t lc = (c0 + i % ncv) * kSsVecElems;
        const uint32_t col = p.rank * p.Hc + lc;
        const int s = si[t];
        const int64_t bos = cu[s];
        const bool cm = cmask[s];
        const int64_t cache_base = safe_idx[s] * p.cache_stride_slot + p.cache_col0 + lc;
        const uint32_t cv8 = lc - c0 * kSsVecElems;
        const uint4 xt = p.use_tile ? *reinterpret_cast<const uint4*>(smem_x + (t - h0) * ncv8 + cv8)
                                    : *reinterpret_cast<const uint4*>(scratch + static_cast<int64_t>(t) * p.Hc + lc);

        uint4 taps[W1];
#pragma unroll
        for (int k = 0; k < W1; ++k) {
          const int64_t pos = static_cast<int64_t>(t) - (W1 - k);
          if (pos >= bos) {
            taps[k] = p.use_tile ? *reinterpret_cast<const uint4*>(smem_x + (pos - h0) * ncv8 + cv8)
                                 : *reinterpret_cast<const uint4*>(scratch + pos * p.Hc + lc);
          } else {
            const int64_t prow = pos - bos + W1;
            taps[k] = cm ? *reinterpret_cast<const uint4*>(&cache[cache_base + prow * p.cache_stride_w])
                         : make_uint4(0, 0, 0, 0);
          }
        }



        const auto* wsrc = use_smw ? smem_w : static_cast<const __nv_bfloat16*>(p.weight);
        uint2 wraw[kSsVecElems];
#pragma unroll
        for (int j = 0; j < static_cast<int>(kSsVecElems); ++j) {
          if constexpr (W == 4) {
            wraw[j] = *reinterpret_cast<const uint2*>(&wsrc[(lc + j) * W]);
          }
        }
        const auto wt = [&](int j, int w) -> float {
          if constexpr (W == 4) {
            return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&wraw[j])[w]);
          } else {
            return __bfloat162float(wsrc[(lc + j) * W + w]);
          }
        };
        const auto* xh = reinterpret_cast<const __nv_bfloat16*>(&xt);
        __nv_bfloat162 yb[4];
#pragma unroll
        for (int j2 = 0; j2 < 4; ++j2) {
          float yj[2];
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int j = 2 * j2 + h;
            const float xj = __bfloat162float(xh[j]);
            float acc = xj * wt(j, W1);
#pragma unroll
            for (int k = 0; k < W1; ++k) {
              const float tap = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&taps[k])[j]);
              acc += tap * wt(j, k);
            }
            if constexpr (USE_SILU) acc = __fdividef(acc, 1.0f + __expf(-acc));
            if constexpr (USE_RESIDUAL) acc += xj;
            yj[h] = acc;
          }
          yb[j2] = __floats2bfloat162_rn(yj[0], yj[1]);
        }

        __nv_bfloat16* addr = mc_out + static_cast<int64_t>(t) * p.H + col;
        asm volatile("multimem.st.relaxed.sys.global.v4.bf16x2 [%0], {%1,%2,%3,%4};" ::"l"(addr),
                     "r"(reinterpret_cast<const uint32_t*>(yb)[0]),
                     "r"(reinterpret_cast<const uint32_t*>(yb)[1]),
                     "r"(reinterpret_cast<const uint32_t*>(yb)[2]),
                     "r"(reinterpret_cast<const uint32_t*>(yb)[3])
                     : "memory");
      }
    }
    __syncthreads();
  }




  if constexpr (kPerBlockBarrier) {
    inkling_ar::block_system_barrier<kNumGPU>(p.state, p.flag_ptrs, p.rank);
  } else {
    inkling_ar::grid_system_barrier<kNumGPU>(p.state, p.flag_ptrs, p.rank, 1,                    true);
  }




  if constexpr (kPerBlockBarrier) {
    inkling_ar::grid_local_sync(p.state);
  }






  {
    auto* wcache = static_cast<__nv_bfloat16*>(p.cache);
    const auto* cip = static_cast<const int32_t*>(p.ci);
    const auto* hinit = static_cast<const bool*>(p.has_init);


    const uint32_t ucv = p.full_update ? p.H / kSsVecElems : cvecs;
    const uint32_t items3 = p.B * ucv;
    for (uint32_t it = gtid; it < items3; it += stride) {
      const uint32_t b = it / ucv;
      const uint32_t lc = (it % ucv) * kSsVecElems;
      const uint32_t ccol = p.full_update ? lc : p.cache_col0 + lc;
      const uint32_t mcol = p.full_update ? lc : p.rank * p.Hc + lc;
      const int slot = cip[b];
      const int64_t qlen = cu[b + 1] - cu[b];
      const bool updated = slot != kSsPadSlot && qlen > 0;
      uint4 nv_reg[W1];
      if (updated) {
        const bool hs = hinit[b];
        const int64_t cb = static_cast<int64_t>(slot) * p.cache_stride_slot + ccol;
        uint4 old_reg[W1];
#pragma unroll
        for (int w = 0; w < W1; ++w) {
          old_reg[w] = *reinterpret_cast<const uint4*>(&wcache[cb + w * p.cache_stride_w]);
        }
        const uint4 zero = make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int w = 0; w < W1; ++w) {
          uint4 nv;
          if (qlen >= W1 - w) {
            const int64_t row = cu[b + 1] - W1 + w;
            if (p.full_update || (p.use_tile && !p.need_scratch)) {




              const __nv_bfloat16* a3 = mc_in + row * p.H + mcol;
              asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
                           : "=r"(nv.x), "=r"(nv.y), "=r"(nv.z), "=r"(nv.w)
                           : "l"(a3));
            } else {
              nv = *reinterpret_cast<const uint4*>(scratch + row * p.Hc + lc);
            }
          } else {
            uint4 shift = zero;
#pragma unroll
            for (int src = 0; src < W1; ++src) {
              if (src == w + qlen) shift = old_reg[src];
            }
            nv = hs ? shift : zero;
          }
          nv_reg[w] = nv;
          *reinterpret_cast<uint4*>(&wcache[cb + w * p.cache_stride_w]) = nv;
        }
      }
      if (p.track_mask != nullptr) {
        const auto* tmask = static_cast<const bool*>(p.track_mask);
        if (tmask[b]) {
          const int64_t dst = static_cast<const int64_t*>(p.track_dst)[static_cast<int64_t>(b) * p.track_dst_stride];
          const int64_t db = dst * p.cache_stride_slot + ccol;
          if (p.track_from_cache) {


            if (updated) {
#pragma unroll
              for (int w = 0; w < W1; ++w) {
                *reinterpret_cast<uint4*>(&wcache[db + w * p.cache_stride_w]) = nv_reg[w];
              }
            }
          } else {
            const auto* trows = static_cast<const int64_t*>(p.track_rows);
#pragma unroll
            for (int w = 0; w < W1; ++w) {
              const int64_t row = trows[static_cast<int64_t>(b) * W1 + w];
              uint4 nv;
              if (p.full_update || (p.use_tile && !p.need_scratch)) {
                const __nv_bfloat16* a4 = mc_in + row * p.H + mcol;
                asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
                             : "=r"(nv.x), "=r"(nv.y), "=r"(nv.z), "=r"(nv.w)
                             : "l"(a4));
              } else {
                nv = *reinterpret_cast<const uint4*>(scratch + row * p.Hc + lc);
              }
              *reinterpret_cast<uint4*>(&wcache[db + w * p.cache_stride_w]) = nv;
            }
          }
        }
      }
    }
  }


  if (p.norm_gamma != nullptr) {
    ss_fused_norm_tail(p);
  }
}



template <typename Kernel>
uint32_t ss_max_resident_blocks(Kernel kernel, uint32_t block_size, DLDevice device) {
  using namespace host;
  static std::mutex mu;
  static std::unordered_map<uint64_t, uint32_t> cache;
  const uint64_t key = (std::bit_cast<uint64_t>(reinterpret_cast<void*>(kernel)) << 12) ^
                       (static_cast<uint64_t>(block_size) << 8) ^ static_cast<uint64_t>(device.device_id);
  {
    std::lock_guard<std::mutex> lk(mu);
    if (auto it = cache.find(key); it != cache.end()) return it->second;
  }
  int sm_count = 0;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device.device_id);
  RuntimeCheck(sm_count > 0, "failed to query multiProcessorCount");
  const uint32_t bps = runtime::get_blocks_per_sm(kernel, block_size);
  RuntimeCheck(bps > 0, "kernel has zero occupancy at block_size ", block_size);
  const uint32_t result = static_cast<uint32_t>(sm_count) * bps;
  std::lock_guard<std::mutex> lk(mu);
  cache.emplace(key, result);
  return result;
}












template <typename DType, uint32_t kNumGPU, int W, bool USE_SILU, bool USE_RESIDUAL, bool kPerBlockBarrier>
__global__
__launch_bounds__(1024, 1) void inkling_ar_stream_sconv_kernel(const __grid_constant__ ArScatteredSconvParams p) {
  static_assert(std::is_same_v<DType, __nv_bfloat16>, "multimem path is bf16-only");
  constexpr int W1 = W - 1;

  if constexpr (kPerBlockBarrier) {
    inkling_ar::block_system_barrier<kNumGPU>(p.state, p.flag_ptrs, p.rank);
  } else {
    inkling_ar::grid_system_barrier<kNumGPU>(p.state, p.flag_ptrs, p.rank, 0, false);
  }

  const auto* si = static_cast<const int32_t*>(p.si);
  const auto* cu = static_cast<const int64_t*>(p.cu);
  const auto* safe_idx = static_cast<const int64_t*>(p.safe_idx);
  const auto* cmask = static_cast<const bool*>(p.cache_mask);
  const auto* cache = static_cast<const __nv_bfloat16*>(p.cache);
  const auto* wp = static_cast<const __nv_bfloat16*>(p.weight);
  const auto* mc_in = static_cast<const __nv_bfloat16*>(p.mc_in);
  auto* mc_out = static_cast<__nv_bfloat16*>(p.mc_out);
  auto* scratch = static_cast<__nv_bfloat16*>(p.x_scratch);

  const uint32_t cvecs = p.Hc / kSsVecElems;
  const uint32_t L = p.chunk_rows;
  const uint32_t walks_t = (p.T + L - 1) / L;
  const uint32_t total = walks_t * cvecs;
  const uint32_t gstride = gridDim.x * blockDim.x;
  const uint32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;

  for (uint32_t wk = gtid; wk < total; wk += gstride) {
    const uint32_t tw = wk / cvecs;
    const uint32_t cv = wk % cvecs;
    const uint32_t lc = cv * kSsVecElems;
    const uint32_t col = p.rank * p.Hc + lc;
    const uint32_t tw0 = tw * L;
    const uint32_t tw1 = min(tw0 + L, p.T);



    uint4 taps[W1];
#pragma unroll
    for (int k = 0; k < W1; ++k) {
      const int64_t pos = static_cast<int64_t>(tw0) - (W1 - k);
      if (pos >= 0) {
        const __nv_bfloat16* a = mc_in + pos * p.H + col;
        asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
                     : "=r"(taps[k].x), "=r"(taps[k].y), "=r"(taps[k].z), "=r"(taps[k].w)
                     : "l"(a));
      } else {
        taps[k] = make_uint4(0, 0, 0, 0);
      }
    }


    uint2 wraw[kSsVecElems];
#pragma unroll
    for (int j = 0; j < static_cast<int>(kSsVecElems); ++j) {
      if constexpr (W == 4) {
        wraw[j] = *reinterpret_cast<const uint2*>(&wp[(lc + j) * W]);
      }
    }
    const auto wt = [&](int j, int w) -> float {
      if constexpr (W == 4) {
        return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&wraw[j])[w]);
      } else {
        return __bfloat162float(wp[(lc + j) * W + w]);
      }
    };

    for (uint32_t t = tw0; t < tw1; ++t) {
      const int sq = si[t];
      const int64_t bos = cu[sq];
      uint4 v;
      const __nv_bfloat16* a = mc_in + static_cast<int64_t>(t) * p.H + col;
      asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
                   : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                   : "l"(a));
      if (p.need_scratch) {
        *reinterpret_cast<uint4*>(scratch + static_cast<int64_t>(t) * p.Hc + lc) = v;
      }


      const bool cm = cmask[sq];
      const int64_t cb = safe_idx[sq] * p.cache_stride_slot + p.cache_col0 + lc;
      uint4 tp[W1];
#pragma unroll
      for (int k = 0; k < W1; ++k) {
        const int64_t pos = static_cast<int64_t>(t) - (W1 - k);
        if (pos >= bos) {
          tp[k] = taps[k];
        } else {
          const int64_t prow = pos - bos + W1;
          tp[k] = cm ? *reinterpret_cast<const uint4*>(&cache[cb + prow * p.cache_stride_w]) : make_uint4(0, 0, 0, 0);
        }
      }

      const auto* xh = reinterpret_cast<const __nv_bfloat16*>(&v);
      __nv_bfloat162 yb[4];
#pragma unroll
      for (int j2 = 0; j2 < 4; ++j2) {
        float yj[2];
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int j = 2 * j2 + hh;
          const float xj = __bfloat162float(xh[j]);
          float acc = xj * wt(j, W1);
#pragma unroll
          for (int k = 0; k < W1; ++k) {
            const float tap = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&tp[k])[j]);
            acc += tap * wt(j, k);
          }
          if constexpr (USE_SILU) acc = __fdividef(acc, 1.0f + __expf(-acc));
          if constexpr (USE_RESIDUAL) acc += xj;
          yj[hh] = acc;
        }
        yb[j2] = __floats2bfloat162_rn(yj[0], yj[1]);
      }
      __nv_bfloat16* ao = mc_out + static_cast<int64_t>(t) * p.H + col;
      asm volatile("multimem.st.relaxed.sys.global.v4.bf16x2 [%0], {%1,%2,%3,%4};" ::"l"(ao),
                   "r"(reinterpret_cast<const uint32_t*>(yb)[0]),
                   "r"(reinterpret_cast<const uint32_t*>(yb)[1]),
                   "r"(reinterpret_cast<const uint32_t*>(yb)[2]),
                   "r"(reinterpret_cast<const uint32_t*>(yb)[3])
                   : "memory");


#pragma unroll
      for (int k = 0; k < W1 - 1; ++k)
        taps[k] = taps[k + 1];
      taps[W1 - 1] = v;
    }
  }


  if constexpr (kPerBlockBarrier) {
    inkling_ar::block_system_barrier<kNumGPU>(p.state, p.flag_ptrs, p.rank);
  } else {
    inkling_ar::grid_system_barrier<kNumGPU>(p.state, p.flag_ptrs, p.rank, 1, true);
  }
  if constexpr (kPerBlockBarrier) {
    inkling_ar::grid_local_sync(p.state);
  }
  {
    auto* wcache = static_cast<__nv_bfloat16*>(p.cache);
    const auto* cip = static_cast<const int32_t*>(p.ci);
    const auto* hinit = static_cast<const bool*>(p.has_init);


    const uint32_t ucv = p.full_update ? p.H / kSsVecElems : cvecs;
    const uint32_t items3 = p.B * ucv;
    for (uint32_t it = gtid; it < items3; it += gstride) {
      const uint32_t b = it / ucv;
      const uint32_t lc = (it % ucv) * kSsVecElems;
      const uint32_t ccol = p.full_update ? lc : p.cache_col0 + lc;
      const uint32_t mcol = p.full_update ? lc : p.rank * p.Hc + lc;
      const int slot = cip[b];
      const int64_t qlen = cu[b + 1] - cu[b];
      const bool updated = slot != kSsPadSlot && qlen > 0;
      uint4 nv_reg[W1];
      if (updated) {
        const bool hs = hinit[b];
        const int64_t cb = static_cast<int64_t>(slot) * p.cache_stride_slot + ccol;
        uint4 old_reg[W1];
#pragma unroll
        for (int w = 0; w < W1; ++w) {
          old_reg[w] = *reinterpret_cast<const uint4*>(&wcache[cb + w * p.cache_stride_w]);
        }
        const uint4 zero = make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int w = 0; w < W1; ++w) {
          uint4 nv;
          if (qlen >= W1 - w) {
            const int64_t row = cu[b + 1] - W1 + w;
            const __nv_bfloat16* a3 = mc_in + row * p.H + mcol;
            asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
                         : "=r"(nv.x), "=r"(nv.y), "=r"(nv.z), "=r"(nv.w)
                         : "l"(a3));
          } else {
            uint4 shift = zero;
#pragma unroll
            for (int src = 0; src < W1; ++src) {
              if (src == w + qlen) shift = old_reg[src];
            }
            nv = hs ? shift : zero;
          }
          nv_reg[w] = nv;
          *reinterpret_cast<uint4*>(&wcache[cb + w * p.cache_stride_w]) = nv;
        }
      }
      if (p.track_mask != nullptr) {
        const auto* tmask = static_cast<const bool*>(p.track_mask);
        if (tmask[b]) {
          const int64_t dst = static_cast<const int64_t*>(p.track_dst)[static_cast<int64_t>(b) * p.track_dst_stride];
          const int64_t db = dst * p.cache_stride_slot + ccol;
          if (p.track_from_cache) {
            if (updated) {
#pragma unroll
              for (int w = 0; w < W1; ++w) {
                *reinterpret_cast<uint4*>(&wcache[db + w * p.cache_stride_w]) = nv_reg[w];
              }
            }
          } else {
            const auto* trows = static_cast<const int64_t*>(p.track_rows);
#pragma unroll
            for (int w = 0; w < W1; ++w) {
              const int64_t row = trows[static_cast<int64_t>(b) * W1 + w];
              uint4 nv;
              const __nv_bfloat16* a4 = mc_in + row * p.H + mcol;
              asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
                           : "=r"(nv.x), "=r"(nv.y), "=r"(nv.z), "=r"(nv.w)
                           : "l"(a4));
              *reinterpret_cast<uint4*>(&wcache[db + w * p.cache_stride_w]) = nv;
            }
          }
        }
      }
    }
  }

  if (p.norm_gamma != nullptr) {
    ss_fused_norm_tail(p);
  }
}

template <typename DType, uint32_t kNumGPU, int W, bool USE_SILU, bool USE_RESIDUAL>
struct ArScatteredSconvKernel {
  static void
  run(tvm::ffi::TensorView in_buffer,
      tvm::ffi::TensorView x_scratch,
      tvm::ffi::TensorView cache,
      tvm::ffi::TensorView safe_idx,
      tvm::ffi::TensorView cache_mask,
      tvm::ffi::TensorView ci,
      tvm::ffi::TensorView has_init,
      tvm::ffi::TensorView cu,
      tvm::ffi::TensorView si,
      tvm::ffi::TensorView weight,
      tvm::ffi::TensorView track_rows,
      tvm::ffi::TensorView track_mask,
      tvm::ffi::TensorView track_dst,
      tvm::ffi::TensorView out_local,
      tvm::ffi::TensorView norm_gamma,
      tvm::ffi::TensorView norm_residual,
      tvm::ffi::TensorView norm_out,
      int64_t mc_in,
      int64_t mc_out,
      int64_t flag_ptrs_dev,
      int64_t state_ptr,
      int64_t rank,
      int64_t nb_override,
      int64_t bs_override,
      bool per_block_barrier,
      bool track_from_cache,
      double norm_eps,
      bool need_scratch,
      bool use_stream,
      int64_t stream_walk,
      bool full_update,
      int64_t cache_col0) {
    using namespace host;
    RuntimeCheck(in_buffer.ndim() == 2, "in must be [T, H]");
    const uint32_t T = static_cast<uint32_t>(in_buffer.size(0));
    const uint32_t H = static_cast<uint32_t>(in_buffer.size(1));
    const uint32_t Hc = static_cast<uint32_t>(x_scratch.size(1));
    RuntimeCheck(Hc * kNumGPU == H, "Hc * world must equal H");
    RuntimeCheck(Hc % kSsVecElems == 0, "Hc must be a multiple of 8");
    RuntimeCheck(x_scratch.size(0) == T, "x_scratch rows must equal T");
    RuntimeCheck(weight.size(0) == Hc && weight.size(1) == W, "weight must be [Hc, W]");
    RuntimeCheck(weight.stride(1) == 1 && weight.stride(0) == W, "weight must be contiguous [Hc, W] (smem stage)");
    RuntimeCheck(Hc * W <= kSsMaxHcW, "Hc * W exceeds the smem weight stage");
    if (full_update) {


      RuntimeCheck(cache.size(1) == W - 1 && cache.size(2) == H, "full-width cache must be [slots, W-1, H]");
      RuntimeCheck(
          cache_col0 % kSsVecElems == 0 && cache_col0 + Hc <= H,
          "cache_col0 must be an aligned in-range column offset");
      RuntimeCheck(!need_scratch, "verify (need_scratch) unsupported full-width");
    } else {
      RuntimeCheck(cache.size(1) == W - 1 && cache.size(2) == Hc, "cache must be [slots, W-1, Hc]");
      RuntimeCheck(cache_col0 == 0, "cache_col0 is full-width-mode only");
    }
    RuntimeCheck(cache.stride(2) == 1, "cache must be channel-contiguous");
    RuntimeCheck(si.size(0) >= T, "si must cover T tokens");
    RuntimeCheck(mc_in % 16 == 0 && mc_out % 16 == 0, "multicast ptrs must be 16B aligned");
    RuntimeCheck(flag_ptrs_dev != 0 && state_ptr != 0, "barrier resources are null");
    RuntimeCheck(rank >= 0 && rank < kNumGPU, "rank out of range");
    if (T == 0) return;



    const bool do_track = track_mask.numel() > 0;
    if (do_track && !track_from_cache) {
      RuntimeCheck(track_rows.numel() > 0, "extend track needs gather rows");
    }
    const bool do_norm = norm_gamma.numel() > 0;
    if (do_norm) {
      RuntimeCheck(norm_gamma.numel() == H, "norm_gamma must be [H]");
      RuntimeCheck(
          norm_residual.numel() == static_cast<int64_t>(T) * H && norm_out.numel() == static_cast<int64_t>(T) * H,
          "norm residual/out must be [T, H]");
      RuntimeCheck(out_local.numel() == static_cast<int64_t>(T) * H, "out_local must be [T, H]");
    }
    const auto params = ArScatteredSconvParams{
        .mc_in = reinterpret_cast<const void*>(mc_in),
        .mc_out = reinterpret_cast<void*>(mc_out),
        .x_scratch = x_scratch.data_ptr(),
        .cache = cache.data_ptr(),
        .safe_idx = safe_idx.data_ptr(),
        .cache_mask = cache_mask.data_ptr(),
        .ci = ci.data_ptr(),
        .has_init = has_init.data_ptr(),
        .cu = cu.data_ptr(),
        .si = si.data_ptr(),
        .weight = weight.data_ptr(),
        .track_rows = do_track && !track_from_cache ? track_rows.data_ptr() : nullptr,
        .track_mask = do_track ? track_mask.data_ptr() : nullptr,
        .track_dst = do_track ? track_dst.data_ptr() : nullptr,
        .out_local = do_norm ? out_local.data_ptr() : nullptr,
        .norm_gamma = do_norm ? norm_gamma.data_ptr() : nullptr,
        .norm_residual = do_norm ? norm_residual.data_ptr() : nullptr,
        .norm_out = do_norm ? norm_out.data_ptr() : nullptr,
        .flag_ptrs = reinterpret_cast<void* const*>(flag_ptrs_dev),
        .state = reinterpret_cast<uint32_t*>(state_ptr),
        .cache_stride_slot = cache.stride(0),
        .cache_stride_w = cache.stride(1),
        .track_dst_stride = do_track ? track_dst.stride(0) : 0,
        .norm_eps = static_cast<float>(norm_eps),
        .rank = static_cast<uint32_t>(rank),
        .T = T,
        .H = H,
        .Hc = Hc,
        .B = static_cast<uint32_t>(safe_idx.size(0)),
        .chunk_rows = 1,
        .cvec_chunks = 1,
        .track_from_cache = track_from_cache ? 1u : 0u,
        .use_tile = 0,
        .need_scratch = 1,
        .cache_col0 = static_cast<uint32_t>(cache_col0),
        .full_update = full_update ? 1u : 0u,
    };

    const auto device = in_buffer.device();
    const uint32_t block_size = bs_override > 0 ? static_cast<uint32_t>(bs_override) : 256u;
    const uint32_t cvecs = Hc / kSsVecElems;
    const auto launch = [&](auto kernel) {
      uint32_t cap = ss_max_resident_blocks(kernel, block_size, device);
      if (per_block_barrier) cap = min(cap, inkling_ar::kMaxBarrierBlocks);
      if (nb_override > 0) cap = min(static_cast<uint32_t>(nb_override), cap);






      const uint32_t min_rows = 16u * (W - 1);
      uint32_t chunk_rows;
      if (T <= 2u * min_rows * max(1u, cap / cvecs)) {
        chunk_rows = T;
      } else {
        chunk_rows = max(min_rows, host::div_ceil(T, 3u * cap));
      }
      const uint32_t token_chunks = host::div_ceil(T, chunk_rows);
      const uint32_t cvec_chunks =
          max(1u,
              min(cvecs,
                  max(cap / max(1u, token_chunks),
                      host::div_ceil(cvecs * min(chunk_rows, T) * 2u, block_size * max(1u, token_chunks)))));



      uint32_t cvec_chunks2 = cvec_chunks;


      bool use_tile = chunk_rows < T && T >= 8192;
      if (use_tile) {
        const uint32_t max_cvec8 = kSsTileElems / (chunk_rows + W - 1);
        const uint32_t max_cv = max_cvec8 / kSsVecElems;
        if (max_cv == 0) {
          use_tile = false;
        } else {
          cvec_chunks2 = max(cvec_chunks, host::div_ceil(cvecs, max_cv));
        }
      }
      auto pp = params;
      pp.chunk_rows = chunk_rows;
      pp.cvec_chunks = cvec_chunks2;
      pp.use_tile = use_tile ? 1u : 0u;
      pp.need_scratch = need_scratch ? 1u : 0u;
      const uint32_t num_blocks = min(cap, token_chunks * cvec_chunks2);
      const auto stream = LaunchKernel::resolve_device(device);
      LaunchKernel(num_blocks, block_size, stream)(kernel, pp);
    };
    if (use_stream) {


      const auto launch_stream = [&](auto kernel) {
        uint32_t cap = ss_max_resident_blocks(kernel, block_size, device);
        if (per_block_barrier) cap = min(cap, inkling_ar::kMaxBarrierBlocks);
        if (nb_override > 0) cap = min(static_cast<uint32_t>(nb_override), cap);
        const uint32_t threads = cap * block_size;
        uint32_t L =
            stream_walk > 0 ? static_cast<uint32_t>(stream_walk) : max(48u, host::div_ceil(T * cvecs, threads));
        const uint32_t walks = host::div_ceil(T, L) * cvecs;
        const uint32_t nblk = min(cap, host::div_ceil(walks, block_size));
        auto pp = params;
        pp.chunk_rows = L;
        pp.use_tile = 0;
        pp.need_scratch = need_scratch ? 1u : 0u;
        const auto stream = LaunchKernel::resolve_device(device);
        LaunchKernel(nblk, block_size, stream)(kernel, pp);
      };
      if (per_block_barrier) {
        launch_stream(inkling_ar_stream_sconv_kernel<DType, kNumGPU, W, USE_SILU, USE_RESIDUAL, true>);
      } else {
        launch_stream(inkling_ar_stream_sconv_kernel<DType, kNumGPU, W, USE_SILU, USE_RESIDUAL, false>);
      }
      return;
    }
    if (per_block_barrier) {
      launch(inkling_ar_scattered_sconv_kernel<DType, kNumGPU, W, USE_SILU, USE_RESIDUAL, true>);
    } else {
      launch(inkling_ar_scattered_sconv_kernel<DType, kNumGPU, W, USE_SILU, USE_RESIDUAL, false>);
    }
  }
};













struct ColDecodeParams {
  const void* __restrict__ mc_in;
  void* __restrict__ mc_out;
  const void* __restrict__ out_local;
  void* const* __restrict__ flag_ptrs;
  uint32_t* __restrict__ state;
  void* __restrict__ cache;
  const void* __restrict__ cache_indices;
  const void* __restrict__ cache_mask;
  const void* __restrict__ weight;
  const void* __restrict__ track_mask;
  const void* __restrict__ track_dst;
  const void* __restrict__ residual_in;
  void* __restrict__ residual_out;
  void* __restrict__ hs_out;
  const void* __restrict__ norm_weight;
  float eps;
  int64_t cache_stride_slot;
  int64_t cache_stride_w;
  int64_t track_dst_stride;
  uint32_t rank;
  uint32_t T;
  uint32_t H;
  uint32_t Hc;
};

template <typename DType, uint32_t kNumGPU, int W, bool USE_SILU, bool USE_RESIDUAL, int VPT>
__global__ __launch_bounds__(1024, 1) void inkling_ar_col_decode_kernel(const __grid_constant__ ColDecodeParams p) {
  static_assert(std::is_same_v<DType, __nv_bfloat16>, "multimem path is bf16-only");
  constexpr int W1 = W - 1;
  const uint32_t t = blockIdx.x;
  const uint32_t vecs = p.H / kSsVecElems;
  const uint32_t ovecs = p.Hc / kSsVecElems;


  uint32_t c0[VPT];
  bool act[VPT];
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    const uint32_t v = threadIdx.x + i * blockDim.x;
    act[i] = v < vecs;
    c0[i] = (act[i] ? v : 0) * kSsVecElems;
  }
  const uint32_t ov = threadIdx.x;
  const bool own = ov < ovecs;
  const uint32_t olc = (own ? ov : 0) * kSsVecElems;


  const int ci = static_cast<const int32_t*>(p.cache_indices)[t];
  const bool valid = ci != kSsPadSlot;
  const int slot = valid ? ci : 0;
  const float cm = static_cast<const bool*>(p.cache_mask)[t] ? 1.0f : 0.0f;
  auto* cp = static_cast<__nv_bfloat16*>(p.cache);
  const auto* wp = static_cast<const __nv_bfloat16*>(p.weight);
  uint4 hist[W1];
  if (own) {
    const int64_t cb = static_cast<int64_t>(slot) * p.cache_stride_slot + olc;
#pragma unroll
    for (int w = 0; w < W1; ++w) {
      hist[w] = *reinterpret_cast<const uint4*>(&cp[cb + w * p.cache_stride_w]);
    }
  }

  uint2 wr8[kSsVecElems];
#pragma unroll
  for (int j = 0; j < static_cast<int>(kSsVecElems); ++j) {
    if (own) {
      if constexpr (W == 4) {
        wr8[j] = *reinterpret_cast<const uint2*>(&wp[(olc + j) * W]);
      }
    }
  }
  uint4 res_raw[VPT];
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    if (act[i]) {
      res_raw[i] = *reinterpret_cast<const uint4*>(
          static_cast<const __nv_bfloat16*>(p.residual_in) + static_cast<int64_t>(t) * p.H + c0[i]);
    }
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");


  inkling_ar::block_system_barrier<kNumGPU>(p.state, p.flag_ptrs, p.rank);


  const auto* mc_in = static_cast<const __nv_bfloat16*>(p.mc_in);
  auto* mc_out = static_cast<__nv_bfloat16*>(p.mc_out);
  uint4 xb;
  if (own) {
    const __nv_bfloat16* a = mc_in + static_cast<int64_t>(t) * p.H + p.rank * p.Hc + olc;
    asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
                 : "=r"(xb.x), "=r"(xb.y), "=r"(xb.z), "=r"(xb.w)
                 : "l"(a));
    const auto* xh = reinterpret_cast<const __nv_bfloat16*>(&xb);
    __nv_bfloat162 yb[4];
#pragma unroll
    for (int j2 = 0; j2 < 4; ++j2) {
      float yj[2];
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int j = 2 * j2 + hh;
        const float xj = __bfloat162float(xh[j]);
        const auto wt = [&](int w) -> float {
          if constexpr (W == 4) {
            return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&wr8[j])[w]);
          } else {
            return __bfloat162float(wp[(olc + j) * W + w]);
          }
        };
        float acc = xj * wt(W1);
#pragma unroll
        for (int k = 0; k < W1; ++k) {
          const float tap = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&hist[k])[j]);
          acc += tap * cm * wt(k);
        }
        if constexpr (USE_SILU) acc = __fdividef(acc, 1.0f + __expf(-acc));
        if constexpr (USE_RESIDUAL) acc += xj;
        yj[hh] = acc;
      }
      yb[j2] = __floats2bfloat162_rn(yj[0], yj[1]);
    }
    __nv_bfloat16* ao = mc_out + static_cast<int64_t>(t) * p.H + p.rank * p.Hc + olc;
    asm volatile("multimem.st.relaxed.sys.global.v4.bf16x2 [%0], {%1,%2,%3,%4};" ::"l"(ao),
                 "r"(reinterpret_cast<const uint32_t*>(yb)[0]),
                 "r"(reinterpret_cast<const uint32_t*>(yb)[1]),
                 "r"(reinterpret_cast<const uint32_t*>(yb)[2]),
                 "r"(reinterpret_cast<const uint32_t*>(yb)[3])
                 : "memory");


    if (valid) {
      const int64_t cb = static_cast<int64_t>(slot) * p.cache_stride_slot + olc;
      const uint4 zero = make_uint4(0, 0, 0, 0);
      uint4 nv[W1];
#pragma unroll
      for (int w = 0; w < W1; ++w) {
        nv[w] = (w < W1 - 1) ? ((cm != 0.0f) ? hist[w + 1] : zero) : xb;
        *reinterpret_cast<uint4*>(&cp[cb + w * p.cache_stride_w]) = nv[w];
      }
      if (p.track_mask != nullptr && static_cast<const bool*>(p.track_mask)[t]) {
        const int64_t dst = static_cast<const int64_t*>(p.track_dst)[static_cast<int64_t>(t) * p.track_dst_stride];
        const int64_t db = dst * p.cache_stride_slot + olc;
#pragma unroll
        for (int w = 0; w < W1; ++w) {
          *reinterpret_cast<uint4*>(&cp[db + w * p.cache_stride_w]) = nv[w];
        }
      }
    }
  }


  inkling_ar::block_system_barrier<kNumGPU>(p.state, p.flag_ptrs, p.rank);


  const auto* outp = static_cast<const __nv_bfloat16*>(p.out_local);
  float r[VPT][kSsVecElems];
  float sumsq = 0.0f;
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    if (!act[i]) continue;
    const uint4 ov4 = *reinterpret_cast<const uint4*>(outp + static_cast<int64_t>(t) * p.H + c0[i]);
    const auto* oh = reinterpret_cast<const __nv_bfloat16*>(&ov4);
    const auto* rh = reinterpret_cast<const __nv_bfloat16*>(&res_raw[i]);
#pragma unroll
    for (int j = 0; j < static_cast<int>(kSsVecElems); ++j) {
      const float v = __bfloat162float(oh[j]) + __bfloat162float(rh[j]);
      r[i][j] = v;
      sumsq += v * v;
    }
  }
  __shared__ float s_warp[32];
  __shared__ float s_inv;
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t warp = threadIdx.x >> 5;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    sumsq += __shfl_down_sync(~0u, sumsq, off);
  if (lane == 0) s_warp[warp] = sumsq;
  __syncthreads();
  if (warp == 0) {
    const uint32_t nwarps = (blockDim.x + 31u) >> 5;
    float total = (lane < nwarps && lane < 32u) ? s_warp[lane] : 0.0f;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      total += __shfl_down_sync(~0u, total, off);
    if (lane == 0) s_inv = rsqrtf(total / static_cast<float>(p.H) + p.eps);
  }
  __syncthreads();
  const float inv = s_inv;
  const auto* gw = static_cast<const __nv_bfloat16*>(p.norm_weight);
  auto* ro = static_cast<__nv_bfloat16*>(p.residual_out) + static_cast<int64_t>(t) * p.H;
  auto* ho = static_cast<__nv_bfloat16*>(p.hs_out) + static_cast<int64_t>(t) * p.H;
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    if (!act[i]) continue;
    __nv_bfloat162 rr[4], hh2[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float g0 = __bfloat162float(gw[c0[i] + 2 * j]);
      const float g1 = __bfloat162float(gw[c0[i] + 2 * j + 1]);
      rr[j] = __floats2bfloat162_rn(r[i][2 * j], r[i][2 * j + 1]);
      hh2[j] = __floats2bfloat162_rn(r[i][2 * j] * inv * g0, r[i][2 * j + 1] * inv * g1);
    }
    *reinterpret_cast<uint4*>(ro + c0[i]) = *reinterpret_cast<const uint4*>(rr);
    *reinterpret_cast<uint4*>(ho + c0[i]) = *reinterpret_cast<const uint4*>(hh2);
  }
}

template <typename DType, uint32_t kNumGPU, int W, bool USE_SILU, bool USE_RESIDUAL>
struct ColDecodeKernel {
  template <int VPT>
  static void launch(const ColDecodeParams& params, uint32_t t_num, uint32_t vecs, DLDevice dev) {
    using namespace host;
    const uint32_t block = min(1024u, host::div_ceil(host::div_ceil(vecs, VPT), 32u) * 32u);
    constexpr auto kernel = inkling_ar_col_decode_kernel<DType, kNumGPU, W, USE_SILU, USE_RESIDUAL, VPT>;
    const auto stream = LaunchKernel::resolve_device(dev);
    LaunchKernel(dim3{t_num}, dim3{block}, stream)(kernel, params);
  }

  static void
  run(tvm::ffi::TensorView in_buffer,
      tvm::ffi::TensorView out_local,
      tvm::ffi::TensorView residual_in,
      tvm::ffi::TensorView residual_out,
      tvm::ffi::TensorView hs_out,
      tvm::ffi::TensorView norm_weight,
      double eps,
      tvm::ffi::TensorView cache,
      tvm::ffi::TensorView cache_indices,
      tvm::ffi::TensorView cache_mask,
      tvm::ffi::TensorView weight,
      tvm::ffi::TensorView track_mask,
      tvm::ffi::TensorView track_dst,
      int64_t mc_in,
      int64_t mc_out,
      int64_t flag_ptrs_dev,
      int64_t state_ptr,
      int64_t rank,
      int64_t vecs_per_thread) {
    using namespace host;
    const uint32_t T = static_cast<uint32_t>(in_buffer.size(0));
    const uint32_t H = static_cast<uint32_t>(in_buffer.size(1));
    const uint32_t Hc = static_cast<uint32_t>(cache.size(2));
    const uint32_t vecs = H / kSsVecElems;
    RuntimeCheck(Hc * kNumGPU == H, "Hc * world must equal H");
    RuntimeCheck(T >= 1 && T <= inkling_ar::kMaxBarrierBlocks, "T must fit one barrier slot per token");
    RuntimeCheck(
        weight.size(0) == Hc && weight.size(1) == W && weight.stride(1) == 1 && weight.stride(0) == W,
        "weight must be contiguous [Hc, W]");
    RuntimeCheck(cache.stride(2) == 1, "cache must be channel-contiguous");
    RuntimeCheck(norm_weight.numel() == H, "norm_weight must be [H]");
    RuntimeCheck(mc_in % 16 == 0 && mc_out % 16 == 0, "mc ptrs must be 16B aligned");
    RuntimeCheck(flag_ptrs_dev != 0 && state_ptr != 0, "null barrier resources");
    const bool do_track = track_mask.numel() > 0;
    const auto params = ColDecodeParams{
        .mc_in = reinterpret_cast<const void*>(mc_in),
        .mc_out = reinterpret_cast<void*>(mc_out),
        .out_local = out_local.data_ptr(),
        .flag_ptrs = reinterpret_cast<void* const*>(flag_ptrs_dev),
        .state = reinterpret_cast<uint32_t*>(state_ptr),
        .cache = cache.data_ptr(),
        .cache_indices = cache_indices.data_ptr(),
        .cache_mask = cache_mask.data_ptr(),
        .weight = weight.data_ptr(),
        .track_mask = do_track ? track_mask.data_ptr() : nullptr,
        .track_dst = do_track ? track_dst.data_ptr() : nullptr,
        .residual_in = residual_in.data_ptr(),
        .residual_out = residual_out.data_ptr(),
        .hs_out = hs_out.data_ptr(),
        .norm_weight = norm_weight.data_ptr(),
        .eps = static_cast<float>(eps),
        .cache_stride_slot = cache.stride(0),
        .cache_stride_w = cache.stride(1),
        .track_dst_stride = do_track ? track_dst.stride(0) : 0,
        .rank = static_cast<uint32_t>(rank),
        .T = T,
        .H = H,
        .Hc = Hc,
    };
    const int vpt = vecs_per_thread > 0 ? static_cast<int>(vecs_per_thread) : 1;
    const auto dev = in_buffer.device();

    RuntimeCheck(
        host::div_ceil(vecs, static_cast<uint32_t>(vpt)) >= Hc / kSsVecElems,
        "block too small for the own-shard lanes");
    switch (vpt) {
      case 1:
        launch<1>(params, T, vecs, dev);
        break;
      case 2:
        launch<2>(params, T, vecs, dev);
        break;
      case 3:
        launch<3>(params, T, vecs, dev);
        break;
      case 4:
        launch<4>(params, T, vecs, dev);
        break;
      default:
        RuntimeCheck(false, "unsupported vecs_per_thread (1/2/3/4)");
    }
  }
};


















struct ArBandedSconvParams {
  const void* __restrict__ mc_in;
  void* __restrict__ mc_out;
  void* __restrict__ scratch;
  void* __restrict__ cache;




  void* __restrict__ mc_wstage;
  const void* __restrict__ wstage;
  const void* __restrict__ safe_idx;
  const void* __restrict__ cache_mask;
  const void* __restrict__ ci;
  const void* __restrict__ has_init;
  const void* __restrict__ cu;
  const void* __restrict__ si;
  const void* __restrict__ weight;
  const void* __restrict__ track_rows;
  const void* __restrict__ track_mask;
  const void* __restrict__ track_dst;
  void* const* __restrict__ flag_ptrs;
  uint32_t* __restrict__ state;
  int64_t cache_stride_slot;
  int64_t cache_stride_w;
  int64_t track_dst_stride;
  uint32_t rank;
  uint32_t T;
  uint32_t H;
  uint32_t Hc;
  uint32_t B;
  uint32_t debug_phase;
};

template <typename DType, uint32_t kNumGPU, int W, bool USE_SILU, bool USE_RESIDUAL, bool kPerBlockBarrier>
__global__
__launch_bounds__(1024, 1) void inkling_ar_banded_sconv_kernel(const __grid_constant__ ArBandedSconvParams p) {
  static_assert(std::is_same_v<DType, __nv_bfloat16>, "multimem path is bf16-only");
  constexpr int W1 = W - 1;
  const bool scattered = p.Hc < p.H;




  if (scattered) {
    const uint32_t pstride = gridDim.x * blockDim.x;
    const uint32_t ptid = blockIdx.x * blockDim.x + threadIdx.x;
    const auto* sidx = static_cast<const int64_t*>(p.safe_idx);
    const auto* csh = static_cast<const __nv_bfloat16*>(p.cache);
    auto* wst = static_cast<__nv_bfloat16*>(p.mc_wstage);
    const uint32_t cvecs = p.Hc / kSsVecElems;
    const uint32_t items = p.B * W1 * cvecs;
    for (uint32_t it = ptid; it < items; it += pstride) {
      const uint32_t b = it / (W1 * cvecs);
      const uint32_t w = (it / cvecs) % W1;
      const uint32_t lc = (it % cvecs) * kSsVecElems;
      const uint4 v = *reinterpret_cast<const uint4*>(&csh[sidx[b] * p.cache_stride_slot + w * p.cache_stride_w + lc]);
      __nv_bfloat16* addr = wst + (static_cast<int64_t>(b) * W1 + w) * p.H + p.rank * p.Hc + lc;
      asm volatile("multimem.st.relaxed.sys.global.v4.bf16x2 [%0], {%1,%2,%3,%4};" ::"l"(addr),
                   "r"(v.x),
                   "r"(v.y),
                   "r"(v.z),
                   "r"(v.w)
                   : "memory");
    }
  }

  if constexpr (kPerBlockBarrier) {
    inkling_ar::block_system_barrier<kNumGPU>(p.state, p.flag_ptrs, p.rank);
  } else {

    inkling_ar::grid_system_barrier<kNumGPU>(p.state, p.flag_ptrs, p.rank, 0, scattered);
  }

  const uint32_t stride = gridDim.x * blockDim.x;
  const uint32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t vrow = p.H / kSsVecElems;
  const uint32_t tpr = (p.T + kNumGPU - 1) / kNumGPU;
  const uint32_t t_lo = min(p.rank * tpr, p.T);
  const uint32_t t_hi = min(t_lo + tpr, p.T);
  const int64_t base = static_cast<int64_t>(t_lo) - W1;
  const uint32_t halo_lo = static_cast<uint32_t>(base > 0 ? base : 0);
  const auto* si = static_cast<const int32_t*>(p.si);
  const auto* cu = static_cast<const int64_t*>(p.cu);
  const auto* safe_idx = static_cast<const int64_t*>(p.safe_idx);
  const auto* cmask = static_cast<const bool*>(p.cache_mask);
  auto* cache = static_cast<__nv_bfloat16*>(p.cache);
  const auto* wp = static_cast<const __nv_bfloat16*>(p.weight);
  const auto* mc_in = static_cast<const __nv_bfloat16*>(p.mc_in);
  auto* mc_out = static_cast<__nv_bfloat16*>(p.mc_out);
  auto* scratch = static_cast<__nv_bfloat16*>(p.scratch);


  {
    const uint32_t v0 = halo_lo * vrow;
    const uint32_t v1 = t_hi * vrow;
    for (uint32_t i = v0 + gtid; i < v1; i += stride) {
      const __nv_bfloat16* addr = mc_in + static_cast<int64_t>(i) * kSsVecElems;
      uint4 v;
      asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
                   : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                   : "l"(addr));
      *reinterpret_cast<uint4*>(scratch + (static_cast<int64_t>(i) - base * vrow) * kSsVecElems) = v;
    }
  }


  inkling_ar::grid_local_sync(p.state);


  if (p.debug_phase != 1) {
    const uint32_t v0 = t_lo * vrow;
    const uint32_t v1 = t_hi * vrow;
    for (uint32_t i = v0 + gtid; i < v1; i += stride) {
      const uint32_t t = i / vrow;
      const uint32_t c = (i % vrow) * kSsVecElems;
      if (p.debug_phase == 2) {
        const int64_t srow2 = static_cast<int64_t>(t) - base;
        const uint4 xr = *reinterpret_cast<const uint4*>(scratch + (srow2 * vrow + i % vrow) * kSsVecElems);
        __nv_bfloat16* addr2 = mc_out + static_cast<int64_t>(t) * p.H + c;
        asm volatile("multimem.st.relaxed.sys.global.v4.bf16x2 [%0], {%1,%2,%3,%4};" ::"l"(addr2),
                     "r"(xr.x),
                     "r"(xr.y),
                     "r"(xr.z),
                     "r"(xr.w)
                     : "memory");
        continue;
      }
      const int s = si[t];
      const int64_t bos = cu[s];
      const bool cm = cmask[s];
      const int64_t cache_base = safe_idx[s] * p.cache_stride_slot + c;
      const int64_t srow = static_cast<int64_t>(t) - base;
      const uint4 xt = *reinterpret_cast<const uint4*>(scratch + (srow * vrow + i % vrow) * kSsVecElems);



      const auto* wstage = static_cast<const __nv_bfloat16*>(p.wstage);
      uint4 taps[W1];
#pragma unroll
      for (int k = 0; k < W1; ++k) {
        const int64_t pos = static_cast<int64_t>(t) - (W1 - k);
        if (pos >= bos) {
          taps[k] = *reinterpret_cast<const uint4*>(scratch + ((pos - base) * vrow + i % vrow) * kSsVecElems);
        } else {
          const int64_t prow = pos - bos + W1;
          if (!cm) {
            taps[k] = make_uint4(0, 0, 0, 0);
          } else if (scattered) {
            taps[k] = *reinterpret_cast<const uint4*>(wstage + (static_cast<int64_t>(s) * W1 + prow) * p.H + c);
          } else {
            taps[k] = *reinterpret_cast<const uint4*>(&cache[cache_base + prow * p.cache_stride_w]);
          }
        }
      }

      float wt[kSsVecElems][W];
#pragma unroll
      for (int j = 0; j < static_cast<int>(kSsVecElems); ++j) {
#pragma unroll
        for (int w = 0; w < W; ++w) {
          wt[j][w] = __bfloat162float(wp[static_cast<int64_t>(c + j) * W + w]);
        }
      }
      const auto* xh = reinterpret_cast<const __nv_bfloat16*>(&xt);
      __nv_bfloat162 yb[4];
#pragma unroll
      for (int j2 = 0; j2 < 4; ++j2) {
        float yj[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int j = 2 * j2 + h;
          const float xj = __bfloat162float(xh[j]);
          float acc = xj * wt[j][W1];
#pragma unroll
          for (int k = 0; k < W1; ++k) {
            const float tap = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&taps[k])[j]);
            acc += tap * wt[j][k];
          }
          if constexpr (USE_SILU) acc = __fdividef(acc, 1.0f + __expf(-acc));
          if constexpr (USE_RESIDUAL) acc += xj;
          yj[h] = acc;
        }
        yb[j2] = __floats2bfloat162_rn(yj[0], yj[1]);
      }

      __nv_bfloat16* addr = mc_out + static_cast<int64_t>(t) * p.H + c;
      asm volatile("multimem.st.relaxed.sys.global.v4.bf16x2 [%0], {%1,%2,%3,%4};" ::"l"(addr),
                   "r"(reinterpret_cast<const uint32_t*>(yb)[0]),
                   "r"(reinterpret_cast<const uint32_t*>(yb)[1]),
                   "r"(reinterpret_cast<const uint32_t*>(yb)[2]),
                   "r"(reinterpret_cast<const uint32_t*>(yb)[3])
                   : "memory");
    }
  }


  inkling_ar::grid_local_sync(p.state);
  if (p.debug_phase == 0)







  {
    const auto* ci = static_cast<const int32_t*>(p.ci);
    const auto* hinit = static_cast<const bool*>(p.has_init);



    const uint32_t uvecs = scattered ? p.Hc / kSsVecElems : vrow;
    const uint32_t items3 = p.B * uvecs;
    for (uint32_t it = gtid; it < items3; it += stride) {
      const uint32_t b = it / uvecs;
      const uint32_t co = (it % uvecs) * kSsVecElems;
      const uint32_t c = scattered ? p.rank * p.Hc + co : co;
      const int slot = ci[b];
      const int64_t qlen = cu[b + 1] - cu[b];
      if (slot != kSsPadSlot && qlen > 0) {
        const bool hs = hinit[b];
        const int64_t cb = static_cast<int64_t>(slot) * p.cache_stride_slot + co;
        uint4 old_reg[W1];
#pragma unroll
        for (int w = 0; w < W1; ++w) {
          old_reg[w] = *reinterpret_cast<const uint4*>(&cache[cb + w * p.cache_stride_w]);
        }
        const uint4 zero = make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int w = 0; w < W1; ++w) {
          uint4 nv;
          if (qlen >= W1 - w) {
            const int64_t row = cu[b + 1] - W1 + w;
            const __nv_bfloat16* addr = mc_in + row * p.H + c;
            asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
                         : "=r"(nv.x), "=r"(nv.y), "=r"(nv.z), "=r"(nv.w)
                         : "l"(addr));
          } else {
            uint4 shift = zero;
#pragma unroll
            for (int src = 0; src < W1; ++src) {
              if (src == w + qlen) shift = old_reg[src];
            }
            nv = hs ? shift : zero;
          }
          *reinterpret_cast<uint4*>(&cache[cb + w * p.cache_stride_w]) = nv;
        }
      }

      if (p.track_rows != nullptr) {
        const auto* tmask = static_cast<const bool*>(p.track_mask);
        if (tmask[b]) {
          const auto* trows = static_cast<const int64_t*>(p.track_rows);
          const int64_t dst = static_cast<const int64_t*>(p.track_dst)[static_cast<int64_t>(b) * p.track_dst_stride];
          const int64_t db = dst * p.cache_stride_slot + co;
#pragma unroll
          for (int w = 0; w < W1; ++w) {
            const int64_t row = trows[static_cast<int64_t>(b) * W1 + w];
            uint4 nv;
            const __nv_bfloat16* addr = mc_in + row * p.H + c;
            asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
                         : "=r"(nv.x), "=r"(nv.y), "=r"(nv.z), "=r"(nv.w)
                         : "l"(addr));
            *reinterpret_cast<uint4*>(&cache[db + w * p.cache_stride_w]) = nv;
          }
        }
      }
    }
  }

  if constexpr (kPerBlockBarrier) {
    inkling_ar::block_system_barrier<kNumGPU>(p.state, p.flag_ptrs, p.rank);
  } else {
    inkling_ar::grid_system_barrier<kNumGPU>(p.state, p.flag_ptrs, p.rank, 1, true);
  }
}

template <typename DType, uint32_t kNumGPU, int W, bool USE_SILU, bool USE_RESIDUAL>
struct ArBandedSconvKernel {
  static void
  run(tvm::ffi::TensorView in_buffer,
      tvm::ffi::TensorView scratch,
      tvm::ffi::TensorView cache,
      tvm::ffi::TensorView safe_idx,
      tvm::ffi::TensorView cache_mask,
      tvm::ffi::TensorView ci,
      tvm::ffi::TensorView has_init,
      tvm::ffi::TensorView cu,
      tvm::ffi::TensorView si,
      tvm::ffi::TensorView weight,
      tvm::ffi::TensorView track_rows,
      tvm::ffi::TensorView track_mask,
      tvm::ffi::TensorView track_dst,
      int64_t mc_in,
      int64_t mc_out,
      int64_t flag_ptrs_dev,
      int64_t state_ptr,
      int64_t rank,
      int64_t nb_override,
      int64_t bs_override,
      bool per_block_barrier,
      int64_t debug_phase,
      int64_t mc_wstage,
      int64_t local_wstage) {
    using namespace host;
    RuntimeCheck(in_buffer.ndim() == 2, "in must be [T, H]");
    const uint32_t T = static_cast<uint32_t>(in_buffer.size(0));
    const uint32_t H = static_cast<uint32_t>(in_buffer.size(1));
    const uint32_t B = static_cast<uint32_t>(safe_idx.size(0));
    const uint32_t Hc = static_cast<uint32_t>(cache.size(2));
    const bool scattered = Hc < H;
    RuntimeCheck(H % kSsVecElems == 0, "H must be a multiple of 8");
    RuntimeCheck(weight.size(0) == H && weight.size(1) == W, "weight must be [H, W]");
    RuntimeCheck(cache.size(1) == W - 1, "cache must be [slots, W-1, Hc|H]");
    if (scattered) {
      RuntimeCheck(Hc * kNumGPU == H, "sharded cache: Hc * world must equal H");
      RuntimeCheck(Hc % kSsVecElems == 0, "Hc must be a multiple of 8");
      RuntimeCheck(
          mc_wstage != 0 && local_wstage != 0 && mc_wstage % 16 == 0, "scattered mode needs the window staging region");
    } else {
      RuntimeCheck(Hc == H, "full-width cache must be [slots, W-1, H]");
    }
    RuntimeCheck(cache.stride(2) == 1, "cache must be channel-contiguous");
    RuntimeCheck(si.size(0) >= T, "si must cover T tokens");
    const uint32_t tpr = host::div_ceil(T, kNumGPU);
    RuntimeCheck(scratch.size(0) >= tpr + W - 1 && scratch.size(1) == H, "scratch must be [tpr + W-1, H]");
    RuntimeCheck(mc_in % 16 == 0 && mc_out % 16 == 0, "multicast ptrs must be 16B aligned");
    RuntimeCheck(flag_ptrs_dev != 0 && state_ptr != 0, "barrier resources are null");
    RuntimeCheck(rank >= 0 && rank < kNumGPU, "rank out of range");
    const bool do_track = track_rows.numel() > 0;
    if (T == 0) return;

    const auto params = ArBandedSconvParams{
        .mc_in = reinterpret_cast<const void*>(mc_in),
        .mc_out = reinterpret_cast<void*>(mc_out),
        .scratch = scratch.data_ptr(),
        .cache = cache.data_ptr(),
        .mc_wstage = reinterpret_cast<void*>(mc_wstage),
        .wstage = reinterpret_cast<const void*>(local_wstage),
        .safe_idx = safe_idx.data_ptr(),
        .cache_mask = cache_mask.data_ptr(),
        .ci = ci.data_ptr(),
        .has_init = has_init.data_ptr(),
        .cu = cu.data_ptr(),
        .si = si.data_ptr(),
        .weight = weight.data_ptr(),
        .track_rows = do_track ? track_rows.data_ptr() : nullptr,
        .track_mask = do_track ? track_mask.data_ptr() : nullptr,
        .track_dst = do_track ? track_dst.data_ptr() : nullptr,
        .flag_ptrs = reinterpret_cast<void* const*>(flag_ptrs_dev),
        .state = reinterpret_cast<uint32_t*>(state_ptr),
        .cache_stride_slot = cache.stride(0),
        .cache_stride_w = cache.stride(1),
        .track_dst_stride = do_track ? track_dst.stride(0) : 0,
        .rank = static_cast<uint32_t>(rank),
        .T = T,
        .H = H,
        .Hc = Hc,
        .B = B,
        .debug_phase = static_cast<uint32_t>(debug_phase),
    };

    const auto device = in_buffer.device();
    const uint32_t block_size = bs_override > 0 ? static_cast<uint32_t>(bs_override) : 512u;
    const auto launch = [&](auto kernel) {
      const uint32_t items = tpr * (H / kSsVecElems);
      uint32_t cap = ss_max_resident_blocks(kernel, block_size, device);
      if (per_block_barrier) cap = min(cap, inkling_ar::kMaxBarrierBlocks);
      const uint32_t want = max(1u, host::div_ceil(items, block_size));
      const uint32_t num_blocks = nb_override > 0 ? min(static_cast<uint32_t>(nb_override), cap) : min(want, cap);
      const auto stream = LaunchKernel::resolve_device(device);
      LaunchKernel(num_blocks, block_size, stream)(kernel, params);
    };
    if (per_block_barrier) {
      launch(inkling_ar_banded_sconv_kernel<DType, kNumGPU, W, USE_SILU, USE_RESIDUAL, true>);
    } else {
      launch(inkling_ar_banded_sconv_kernel<DType, kNumGPU, W, USE_SILU, USE_RESIDUAL, false>);
    }
  }
};

























struct SsconvNormDecodeParams {
  const void* __restrict__ in;
  void* __restrict__ mc_stage;
  const void* __restrict__ stage;
  void* __restrict__ mc_wstage;
  const void* __restrict__ wstage;
  void* const* __restrict__ flag_ptrs;
  uint32_t* __restrict__ state;
  void* __restrict__ cache;
  const void* __restrict__ cache_indices;
  const void* __restrict__ cache_mask;
  const void* __restrict__ conv_weight;
  const void* __restrict__ track_mask;
  const void* __restrict__ track_indices;
  const void* __restrict__ residual_in;
  void* __restrict__ residual_out;
  void* __restrict__ hs_out;
  const void* __restrict__ norm_weight;
  float eps;
  int64_t in_stride_t;
  int64_t res_in_stride_t;
  int64_t res_out_stride_t;
  int64_t hs_stride_t;
  int64_t cache_stride_slot;
  int64_t cache_stride_w;
  int64_t conv_weight_stride_d;
  int64_t track_idx_stride;
  uint32_t rank;
  uint32_t T;
  uint32_t D;
  uint32_t Hc;
};

template <typename DType, uint32_t kNumGPU, int W, bool USE_SILU, bool USE_RESIDUAL, int VPT>
__global__
__launch_bounds__(1024, 1) void inkling_ar_ssconv_norm_decode_kernel(const __grid_constant__ SsconvNormDecodeParams p) {
  static_assert(std::is_same_v<DType, __nv_bfloat16>, "multimem push path is bf16-only");
  constexpr int W1 = W - 1;
  const uint32_t t = blockIdx.x;
  const uint32_t vecs = p.D / kSsVecElems;
  const uint32_t shard_lo = p.rank * p.Hc;

  uint32_t c0[VPT];
  bool act[VPT];
  bool mine[VPT];
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    const uint32_t v = threadIdx.x + i * blockDim.x;
    act[i] = v < vecs;
    c0[i] = (act[i] ? v : 0) * kSsVecElems;
    mine[i] = act[i] && c0[i] >= shard_lo && c0[i] < shard_lo + p.Hc;
  }


  const int ci = static_cast<const int32_t*>(p.cache_indices)[t];
  const bool valid = ci != kSsPadSlot;
  const int slot_id = valid ? ci : 0;
  const float cm = static_cast<const bool*>(p.cache_mask)[t] ? 1.0f : 0.0f;
  auto* cp = static_cast<__nv_bfloat16*>(p.cache);
  const auto* wp = static_cast<const __nv_bfloat16*>(p.conv_weight);
  uint4 hist_raw[VPT][W1];
  __nv_bfloat16 wtaps[VPT][kSsVecElems][W];
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    if (!act[i]) continue;
    if (mine[i]) {
      const int64_t cache_base = static_cast<int64_t>(slot_id) * p.cache_stride_slot + (c0[i] - shard_lo);
#pragma unroll
      for (int w = 0; w < W1; ++w) {
        hist_raw[i][w] = *reinterpret_cast<const uint4*>(&cp[cache_base + w * p.cache_stride_w]);
      }
    }
#pragma unroll
    for (int j = 0; j < static_cast<int>(kSsVecElems); ++j) {
      const int64_t wrow = static_cast<int64_t>(c0[i] + j) * p.conv_weight_stride_d;
      if constexpr (W == 4) {
        if (p.conv_weight_stride_d == W) {
          *reinterpret_cast<uint2*>(wtaps[i][j]) = *reinterpret_cast<const uint2*>(wp + wrow);
          continue;
        }
      }
#pragma unroll
      for (int w = 0; w < W; ++w)
        wtaps[i][j][w] = wp[wrow + w];
    }
  }


  asm volatile("griddepcontrol.wait;" ::: "memory");
  const auto* in_row = static_cast<const __nv_bfloat16*>(p.in) + t * p.in_stride_t;
  auto* slot = static_cast<__nv_bfloat16*>(p.mc_stage) + (static_cast<uint64_t>(p.rank) * p.T + t) * p.D;
  auto* wslot = static_cast<__nv_bfloat16*>(p.mc_wstage) + static_cast<uint64_t>(t) * W1 * p.D;
  uint4 res_raw[VPT];
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    if (!act[i]) continue;
    const uint4 d = *reinterpret_cast<const uint4*>(in_row + c0[i]);
    asm volatile("multimem.st.relaxed.sys.global.v4.bf16x2 [%0], {%1,%2,%3,%4};" ::"l"(slot + c0[i]),
                 "r"(d.x),
                 "r"(d.y),
                 "r"(d.z),
                 "r"(d.w)
                 : "memory");
    if (mine[i]) {
#pragma unroll
      for (int w = 0; w < W1; ++w) {
        const uint4 h = hist_raw[i][w];
        asm volatile("multimem.st.relaxed.sys.global.v4.bf16x2 [%0], {%1,%2,%3,%4};" ::"l"(
                         wslot + static_cast<uint64_t>(w) * p.D + c0[i]),
                     "r"(h.x),
                     "r"(h.y),
                     "r"(h.z),
                     "r"(h.w)
                     : "memory");
      }
    }
    res_raw[i] = *reinterpret_cast<const uint4*>(
        static_cast<const __nv_bfloat16*>(p.residual_in) + t * p.res_in_stride_t + c0[i]);
  }


  inkling_ar::block_system_barrier<kNumGPU>(p.state, p.flag_ptrs, p.rank);


  float r[VPT][kSsVecElems];
  float sumsq = 0.0f;
  const auto* stage = static_cast<const __nv_bfloat16*>(p.stage);
  const auto* wstage = static_cast<const __nv_bfloat16*>(p.wstage) + static_cast<uint64_t>(t) * W1 * p.D;
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    if (!act[i]) continue;

    float xf[kSsVecElems];
#pragma unroll
    for (int j = 0; j < static_cast<int>(kSsVecElems); ++j)
      xf[j] = 0.0f;
#pragma unroll
    for (uint32_t rr = 0; rr < kNumGPU; ++rr) {
      const uint4 d = *reinterpret_cast<const uint4*>(stage + (static_cast<uint64_t>(rr) * p.T + t) * p.D + c0[i]);
      const auto* h2 = reinterpret_cast<const __nv_bfloat162*>(&d);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(h2[j]);
        xf[2 * j] += f.x;
        xf[2 * j + 1] += f.y;
      }
    }
    __nv_bfloat162 xb2[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
      xb2[j] = __floats2bfloat162_rn(xf[2 * j], xf[2 * j + 1]);


    uint4 taps[W1];
#pragma unroll
    for (int w = 0; w < W1; ++w) {
      taps[w] = *reinterpret_cast<const uint4*>(wstage + static_cast<uint64_t>(w) * p.D + c0[i]);
    }
    float y[kSsVecElems];
#pragma unroll
    for (int j = 0; j < static_cast<int>(kSsVecElems); ++j) {
      const float xj = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(xb2)[j]);
      float acc = 0.0f;
#pragma unroll
      for (int w = 0; w < W1; ++w) {
        const float h = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&taps[w])[j]);
        acc += h * cm * __bfloat162float(wtaps[i][j][w]);
      }
      acc += xj * __bfloat162float(wtaps[i][j][W1]);
      if constexpr (USE_SILU) acc = __fdividef(acc, 1.0f + __expf(-acc));
      if constexpr (USE_RESIDUAL) acc += xj;
      y[j] = acc;
    }


    if (valid && mine[i]) {
      const int64_t cache_base = static_cast<int64_t>(slot_id) * p.cache_stride_slot + (c0[i] - shard_lo);
      int64_t track_base = 0;
      bool do_tr = false;
      if (p.track_mask != nullptr) {
        do_tr = static_cast<const bool*>(p.track_mask)[t];
        if (do_tr) {
          const int64_t tslot =
              static_cast<const int64_t*>(p.track_indices)[static_cast<int64_t>(t) * p.track_idx_stride];
          track_base = tslot * p.cache_stride_slot + (c0[i] - shard_lo);
        }
      }
      const uint4 zero = make_uint4(0, 0, 0, 0);
#pragma unroll
      for (int w = 0; w < W1; ++w) {
        const uint4 nv =
            (w < W1 - 1) ? ((cm != 0.0f) ? hist_raw[i][w + 1] : zero) : *reinterpret_cast<const uint4*>(xb2);
        *reinterpret_cast<uint4*>(&cp[cache_base + w * p.cache_stride_w]) = nv;
        if (do_tr) {
          *reinterpret_cast<uint4*>(&cp[track_base + w * p.cache_stride_w]) = nv;
        }
      }
    }


#pragma unroll
    for (int j = 0; j < static_cast<int>(kSsVecElems); ++j) {
      const float yb = __bfloat162float(__float2bfloat16_rn(y[j]));
      r[i][j] = yb + __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&res_raw[i])[j]);
      sumsq += r[i][j] * r[i][j];
    }
  }


  __shared__ float s_warp[32];
  __shared__ float s_inv;
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t warp = threadIdx.x >> 5;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    sumsq += __shfl_down_sync(~0u, sumsq, off);
  if (lane == 0) s_warp[warp] = sumsq;
  __syncthreads();
  if (warp == 0) {
    const uint32_t nwarps = (blockDim.x + 31u) >> 5;
    float total = (lane < nwarps && lane < 32u) ? s_warp[lane] : 0.0f;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      total += __shfl_down_sync(~0u, total, off);
    if (lane == 0) s_inv = rsqrtf(total / static_cast<float>(p.D) + p.eps);
  }
  __syncthreads();
  const float inv = s_inv;

  const auto* gw = static_cast<const __nv_bfloat16*>(p.norm_weight);
  auto* res_out = static_cast<__nv_bfloat16*>(p.residual_out) + t * p.res_out_stride_t;
  auto* hs_out = static_cast<__nv_bfloat16*>(p.hs_out) + t * p.hs_stride_t;
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    if (!act[i]) continue;
    __nv_bfloat162 ro[4], ho[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float g0 = __bfloat162float(gw[c0[i] + 2 * j]);
      const float g1 = __bfloat162float(gw[c0[i] + 2 * j + 1]);
      ro[j] = __floats2bfloat162_rn(r[i][2 * j], r[i][2 * j + 1]);
      ho[j] = __floats2bfloat162_rn(r[i][2 * j] * inv * g0, r[i][2 * j + 1] * inv * g1);
    }
    *reinterpret_cast<uint4*>(res_out + c0[i]) = *reinterpret_cast<const uint4*>(ro);
    *reinterpret_cast<uint4*>(hs_out + c0[i]) = *reinterpret_cast<const uint4*>(ho);
  }
}

template <typename DType, uint32_t kNumGPU, int W, bool USE_SILU, bool USE_RESIDUAL>
struct SsconvNormDecodeKernel {
  template <int VPT>
  static void launch(const SsconvNormDecodeParams& params, uint32_t t_num, uint32_t vecs, DLDevice dev) {
    using namespace host;
    const uint32_t block = min(1024u, host::div_ceil(host::div_ceil(vecs, VPT), 32u) * 32u);
    constexpr auto kernel = inkling_ar_ssconv_norm_decode_kernel<DType, kNumGPU, W, USE_SILU, USE_RESIDUAL, VPT>;
    const auto stream = LaunchKernel::resolve_device(dev);
    LaunchKernel(dim3{t_num}, dim3{block}, stream)(kernel, params);
  }

  static void
  run(tvm::ffi::TensorView in,
      tvm::ffi::TensorView residual_in,
      tvm::ffi::TensorView residual_out,
      tvm::ffi::TensorView hs_out,
      tvm::ffi::TensorView norm_weight,
      double eps,
      tvm::ffi::TensorView cache,
      tvm::ffi::TensorView cache_indices,
      tvm::ffi::TensorView cache_mask,
      tvm::ffi::TensorView conv_weight,
      tvm::ffi::TensorView track_mask,
      tvm::ffi::TensorView track_indices,
      int64_t mc_stage_ptr,
      int64_t local_stage_ptr,
      int64_t mc_wstage_ptr,
      int64_t local_wstage_ptr,
      int64_t flag_ptrs_dev,
      int64_t state_ptr,
      int64_t rank,
      int64_t vecs_per_thread) {
    using namespace host;
    RuntimeCheck(in.ndim() == 2, "in must be [T, D]");
    const uint32_t t_num = static_cast<uint32_t>(in.size(0));
    const uint32_t d_num = static_cast<uint32_t>(in.size(1));
    const uint32_t hc = static_cast<uint32_t>(cache.size(2));
    const uint32_t vecs = d_num / kSsVecElems;
    RuntimeCheck(hc * kNumGPU == d_num, "Hc * world must equal D");
    RuntimeCheck(hc % kSsVecElems == 0, "Hc must be a multiple of 8");
    RuntimeCheck(t_num >= 1 && t_num <= inkling_ar::kMaxBarrierBlocks, "T must be in [1, kMaxBarrierBlocks]");
    RuntimeCheck(d_num % kSsVecElems == 0, "D must be a multiple of 8");
    RuntimeCheck(cache.size(1) == W - 1, "cache must be [pool, W-1, Hc]");
    RuntimeCheck(cache.stride(2) == 1, "cache must be channel-contiguous");
    RuntimeCheck(conv_weight.size(0) == d_num && conv_weight.size(1) == W, "conv_weight must be FULL [D, W]");
    RuntimeCheck(norm_weight.numel() == d_num, "norm_weight must be [D]");
    RuntimeCheck(mc_stage_ptr % 16 == 0 && mc_wstage_ptr % 16 == 0, "staging ptrs must be 16B aligned");
    RuntimeCheck(local_stage_ptr != 0 && local_wstage_ptr != 0, "null staging");
    RuntimeCheck(flag_ptrs_dev != 0 && state_ptr != 0, "null barrier resources");
    RuntimeCheck(rank >= 0 && rank < kNumGPU, "rank out of range");
    const bool do_track = track_mask.numel() > 0;

    const auto params = SsconvNormDecodeParams{
        .in = in.data_ptr(),
        .mc_stage = reinterpret_cast<void*>(mc_stage_ptr),
        .stage = reinterpret_cast<const void*>(local_stage_ptr),
        .mc_wstage = reinterpret_cast<void*>(mc_wstage_ptr),
        .wstage = reinterpret_cast<const void*>(local_wstage_ptr),
        .flag_ptrs = reinterpret_cast<void* const*>(flag_ptrs_dev),
        .state = reinterpret_cast<uint32_t*>(state_ptr),
        .cache = cache.data_ptr(),
        .cache_indices = cache_indices.data_ptr(),
        .cache_mask = cache_mask.data_ptr(),
        .conv_weight = conv_weight.data_ptr(),
        .track_mask = do_track ? track_mask.data_ptr() : nullptr,
        .track_indices = do_track ? track_indices.data_ptr() : nullptr,
        .residual_in = residual_in.data_ptr(),
        .residual_out = residual_out.data_ptr(),
        .hs_out = hs_out.data_ptr(),
        .norm_weight = norm_weight.data_ptr(),
        .eps = static_cast<float>(eps),
        .in_stride_t = in.stride(0),
        .res_in_stride_t = residual_in.stride(0),
        .res_out_stride_t = residual_out.stride(0),
        .hs_stride_t = hs_out.stride(0),
        .cache_stride_slot = cache.stride(0),
        .cache_stride_w = cache.stride(1),
        .conv_weight_stride_d = conv_weight.stride(0),
        .track_idx_stride = do_track ? track_indices.stride(0) : 0,
        .rank = static_cast<uint32_t>(rank),
        .T = t_num,
        .D = d_num,
        .Hc = hc,
    };

    const int vpt = vecs_per_thread > 0 ? static_cast<int>(vecs_per_thread) : 1;
    const auto dev = in.device();
    switch (vpt) {
      case 1:
        RuntimeCheck(vecs <= 1024, "D/8 must fit one block at VPT=1");
        launch<1>(params, t_num, vecs, dev);
        break;
      case 2:
        launch<2>(params, t_num, vecs, dev);
        break;
      case 3:
        launch<3>(params, t_num, vecs, dev);
        break;
      case 4:
        launch<4>(params, t_num, vecs, dev);
        break;
      default:
        RuntimeCheck(false, "unsupported vecs_per_thread (use 1/2/3/4)");
    }
  }
};

}
