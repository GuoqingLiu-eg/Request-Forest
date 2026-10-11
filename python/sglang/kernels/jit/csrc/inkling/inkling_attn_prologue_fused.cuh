




















#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <bit>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <type_traits>

namespace sglang {

constexpr int kPadSlot = -1;
constexpr uint32_t kVecElems = 8;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kHeadLanes = kHeadDim / kVecElems;
constexpr uint32_t kMXFP8Block = 32;
constexpr uint32_t kMXFP8BlockLanes = kMXFP8Block / kVecElems;
constexpr float kE4M3Max = 448.0f;










__device__ __forceinline__ float head_rmsnorm_inv(float ss, float eps) {
  const unsigned hmask = 0xFFFFu << (threadIdx.x & 16u);
#pragma unroll
  for (int off = 8; off > 0; off >>= 1)
    ss += __shfl_xor_sync(hmask, ss, off, 16);
  return rsqrtf(ss / static_cast<float>(kHeadDim) + eps);
}

__device__ __forceinline__ uint8_t mxfp8_scale_byte(float local_amax, float* descale) {
  const unsigned qmask = 0xFu << (threadIdx.x & 28u);
#pragma unroll
  for (int off = 1; off < static_cast<int>(kMXFP8BlockLanes); off <<= 1) {
    local_amax = fmaxf(local_amax, __shfl_xor_sync(qmask, local_amax, off, kMXFP8BlockLanes));
  }
  const float amax = fmaxf(local_amax, 1.0e-30f);
  float scale_biased = ceilf(log2f(amax / kE4M3Max)) + 127.0f;
  scale_biased = fminf(fmaxf(scale_biased, 0.0f), 254.0f);
  *descale = exp2f(scale_biased - 127.0f);
  return static_cast<uint8_t>(scale_biased);
}

__device__ __forceinline__ void store_mxfp8_vec(const float (&x)[kVecElems], void* dst, uint8_t* sf, uint32_t c) {
  float local_amax = 0.0f;
#pragma unroll
  for (int j = 0; j < static_cast<int>(kVecElems); ++j) {
    local_amax = fmaxf(local_amax, fabsf(x[j]));
  }
  float descale;
  const uint8_t sf_byte = mxfp8_scale_byte(local_amax, &descale);
  if ((c & (kMXFP8Block - 1)) == 0) *sf = sf_byte;

  union {
    __nv_fp8x2_e4m3 fp8x2[4];
    uint2 raw;
  } u;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float x0 = fminf(fmaxf(x[2 * j] / descale, -kE4M3Max), kE4M3Max);
    const float x1 = fminf(fmaxf(x[2 * j + 1] / descale, -kE4M3Max), kE4M3Max);
    u.fp8x2[j] = __nv_fp8x2_e4m3(make_float2(x0, x1));
  }
  *reinterpret_cast<uint2*>(static_cast<uint8_t*>(dst) + c) = u.raw;
}

struct AttnPrologueParams {
  const void* __restrict__ qkvr;

  const void* __restrict__ k_cache;
  const void* __restrict__ v_cache;
  const void* __restrict__ cache_indices;
  const void* __restrict__ cache_mask;
  const void* __restrict__ k_weight;
  const void* __restrict__ v_weight;
  void* __restrict__ k_inter;
  void* __restrict__ v_inter;

  const void* __restrict__ q_gamma;
  const void* __restrict__ k_gamma;
  const void* __restrict__ log_tau;
  float eps;

  void* __restrict__ q_out;
  void* __restrict__ k_out;
  void* __restrict__ v_out;
  const void* __restrict__ loc;
  void* __restrict__ k_buf;
  void* __restrict__ v_buf;
  void* __restrict__ sfq;
  void* __restrict__ sfk;
  void* __restrict__ sfv;
  int64_t qkvr_stride_t;
  int64_t q_off;
  int64_t k_off;
  int64_t v_off;
  int64_t cache_stride_slot;
  int64_t cache_stride_w;
  int64_t weight_stride_d;
  int64_t inter_stride_b;
  int64_t inter_stride_t;
  int64_t inter_stride_w;
  int64_t kv_buf_stride;
  uint32_t T;
  uint32_t q;
  uint32_t dq;
  uint32_t dkv;
  uint32_t page_size;
};

template <typename DType, int W, bool USE_SILU, bool USE_RESIDUAL, bool DO_STORE, bool USE_MXFP8, bool USE_PDL>
__global__ __launch_bounds__(1024, 1) void inkling_attn_prologue_kernel(const __grid_constant__ AttnPrologueParams p) {
  static_assert(std::is_same_v<DType, __nv_bfloat16>);
  static_assert(!USE_MXFP8 || DO_STORE, "MXFP8 prologue quantization owns the KV store");
  constexpr int W1 = W - 1;
  constexpr uint32_t SF = kHeadDim / kMXFP8Block;
  const uint32_t t = blockIdx.x;
  const uint32_t seq = t / p.q;
  const int bos = static_cast<int>(seq * p.q);
  const uint32_t tq = t - seq * p.q;
  const uint32_t nq = p.dq / kVecElems;
  const uint32_t nkv = p.dkv / kVecElems;
  const uint32_t vi = threadIdx.x;
  const auto* base = static_cast<const __nv_bfloat16*>(p.qkvr);
  const int64_t row = static_cast<int64_t>(t) * p.qkvr_stride_t;

  const int ci = static_cast<const int32_t*>(p.cache_indices)[seq];
  const bool valid = ci != kPadSlot;
  const int slot_id = valid ? ci : 0;
  const float cm = (valid && static_cast<const bool*>(p.cache_mask)[seq]) ? 1.0f : 0.0f;




  if (vi < nq) {

    const uint32_t c = vi * kVecElems;
    const uint4 gqraw = *reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(p.q_gamma) + (c % kHeadDim));
    const auto* gq = reinterpret_cast<const __nv_bfloat16*>(&gqraw);
    const bool do_tau = p.log_tau != nullptr;
    float tau = 0.0f;
    if (do_tau) tau = static_cast<const float*>(p.log_tau)[t];
    device::PDLWaitPrimary<USE_PDL>();
    const uint4 raw = *reinterpret_cast<const uint4*>(base + row + p.q_off + c);
    float x[kVecElems];
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      x[j] = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&raw)[j]);
      ss += x[j] * x[j];
    }
    const float inv = head_rmsnorm_inv(ss, p.eps);
    __nv_bfloat162 o[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      o[j] = __floats2bfloat162_rn(
          x[2 * j] * inv * __bfloat162float(gq[2 * j]), x[2 * j + 1] * inv * __bfloat162float(gq[2 * j + 1]));
    }
    if (do_tau) {



#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(o[j]);
        o[j] = __floats2bfloat162_rn(f.x * tau, f.y * tau);
      }
    }
    if constexpr (USE_MXFP8) {
      float q_quant[kVecElems];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(o[j]);
        q_quant[2 * j] = f.x;
        q_quant[2 * j + 1] = f.y;
      }
      const uint32_t sf_idx = static_cast<uint32_t>(t) * (p.dq / kMXFP8Block) + c / kMXFP8Block;
      store_mxfp8_vec(
          q_quant,
          static_cast<uint8_t*>(p.q_out) + static_cast<int64_t>(t) * p.dq,
          static_cast<uint8_t*>(p.sfq) + sf_idx,
          c);
    } else {
      *reinterpret_cast<uint4*>(static_cast<__nv_bfloat16*>(p.q_out) + static_cast<int64_t>(t) * p.dq + c) =
          *reinterpret_cast<const uint4*>(o);
    }
    device::PDLTriggerSecondary<USE_PDL>();
    return;
  }
  if (vi >= nq + 2 * nkv) return;


  const bool is_k = vi < nq + nkv;
  const uint32_t ch = (is_k ? vi - nq : vi - nq - nkv) * kVecElems;
  const int64_t x_off = is_k ? p.k_off : p.v_off;
  const auto* cp = static_cast<const __nv_bfloat16*>(is_k ? p.k_cache : p.v_cache);
  const auto* wp = static_cast<const __nv_bfloat16*>(is_k ? p.k_weight : p.v_weight);
  auto* ip = static_cast<__nv_bfloat16*>(is_k ? p.k_inter : p.v_inter);
  const int64_t cache_base = static_cast<int64_t>(slot_id) * p.cache_stride_slot + ch;

  uint4 pref[W1];
  __nv_bfloat16 wt[kVecElems][W];
#pragma unroll
  for (int w = 0; w < W1; ++w) {
    pref[w] = *reinterpret_cast<const uint4*>(&cp[cache_base + w * p.cache_stride_w]);
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int64_t wrow = static_cast<int64_t>(ch + j) * p.weight_stride_d;
    if constexpr (W == 4) {
      if (p.weight_stride_d == W) {
        *reinterpret_cast<uint2*>(wt[j]) = *reinterpret_cast<const uint2*>(wp + wrow);
        continue;
      }
    }
#pragma unroll
    for (int w = 0; w < W; ++w)
      wt[j][w] = wp[wrow + w];
  }

  uint4 gkraw = make_uint4(0, 0, 0, 0);
  if (is_k) {
    gkraw = *reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(p.k_gamma) + (ch % kHeadDim));
  }
  device::PDLWaitPrimary<USE_PDL>();

  const uint4 xcur = *reinterpret_cast<const uint4*>(base + row + x_off + ch);
  uint4 xn[W1];
#pragma unroll
  for (int j = 1; j <= W1; ++j) {
    const int n = static_cast<int>(t) - j;
    if (n >= bos) {
      xn[j - 1] = *reinterpret_cast<const uint4*>(base + static_cast<int64_t>(n) * p.qkvr_stride_t + x_off + ch);
    }
  }

  float y[kVecElems];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float xj = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&xcur)[j]);
    float acc = 0.0f;
#pragma unroll
    for (int iw = 0; iw < W1; ++iw) {
      const int shifted = static_cast<int>(t) - W1 + iw;
      float tap = 0.0f;
      if (shifted >= bos) {
        tap = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&xn[W1 - 1 - iw])[j]);
      } else {
        const int prefix_pos = shifted - bos + W1;
        if (prefix_pos >= 0) {
          tap = cm * __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&pref[prefix_pos])[j]);
        }
      }
      acc += tap * __bfloat162float(wt[j][iw]);
    }
    acc += xj * __bfloat162float(wt[j][W1]);
    if constexpr (USE_SILU) acc = __fdividef(acc, 1.0f + __expf(-acc));
    if constexpr (USE_RESIDUAL) acc += xj;
    y[j] = acc;
  }

  if (valid) {
    auto* op = ip + static_cast<int64_t>(seq) * p.inter_stride_b + static_cast<int64_t>(tq) * p.inter_stride_t + ch;
#pragma unroll
    for (int w = 0; w < W1; ++w) {
      const int position = static_cast<int>(tq) + 1 + w;
      uint4 val;
      if (position < W1) {
        val = pref[position];
      } else {
        const int g = bos + position - W1;
        val = (g == static_cast<int>(t)) ? xcur : xn[t - g - 1];
      }
      *reinterpret_cast<uint4*>(op + w * p.inter_stride_w) = val;
    }
  }

  __nv_bfloat162 o[4];
  if (is_k) {



    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      y[j] = __bfloat162float(__float2bfloat16_rn(y[j]));
      ss += y[j] * y[j];
    }
    const float inv = head_rmsnorm_inv(ss, p.eps);
    const auto* gk = reinterpret_cast<const __nv_bfloat16*>(&gkraw);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      o[j] = __floats2bfloat162_rn(
          y[2 * j] * inv * __bfloat162float(gk[2 * j]), y[2 * j + 1] * inv * __bfloat162float(gk[2 * j + 1]));
    }
  } else {
#pragma unroll
    for (int j = 0; j < 4; ++j)
      o[j] = __floats2bfloat162_rn(y[2 * j], y[2 * j + 1]);
  }
  const uint4 ov = *reinterpret_cast<const uint4*>(o);
  auto* out = static_cast<__nv_bfloat16*>(is_k ? p.k_out : p.v_out);
  *reinterpret_cast<uint4*>(out + static_cast<int64_t>(t) * p.dkv + ch) = ov;




  if (DO_STORE) {
    const int64_t kv_slot = static_cast<const int64_t*>(p.loc)[t];
    if (kv_slot >= 0) {
      if constexpr (USE_MXFP8) {
        float xo[kVecElems];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 f = __bfloat1622float2(o[j]);
          xo[2 * j] = f.x;
          xo[2 * j + 1] = f.y;
        }
        auto* buf = static_cast<uint8_t*>(is_k ? p.k_buf : p.v_buf);
        auto* sfb = static_cast<uint8_t*>(is_k ? p.sfk : p.sfv);
        const int64_t po = kv_slot % static_cast<int64_t>(p.page_size);
        const int64_t sf_base = ((kv_slot / static_cast<int64_t>(p.page_size)) * (p.dkv / kHeadDim) + ch / kHeadDim) *
                                    (kMXFP8Block * (p.page_size / kMXFP8Block) * SF) +
                                (po % kMXFP8Block) * ((p.page_size / kMXFP8Block) * SF) + (po / kMXFP8Block) * SF +
                                (ch % kHeadDim) / kMXFP8Block;
        store_mxfp8_vec(xo, buf + kv_slot * p.kv_buf_stride, sfb + sf_base, ch);
      } else {
        auto* buf = static_cast<__nv_bfloat16*>(is_k ? p.k_buf : p.v_buf);
        *reinterpret_cast<uint4*>(buf + kv_slot * p.kv_buf_stride + ch) = ov;
      }
    }
  }
  device::PDLTriggerSecondary<USE_PDL>();
}

template <typename DType, int W, bool USE_SILU, bool USE_RESIDUAL, bool USE_MXFP8, bool USE_PDL>
struct AttnPrologueKernel {
  static void
  run(tvm::ffi::TensorView qkvr,
      tvm::ffi::TensorView k_cache,
      tvm::ffi::TensorView v_cache,
      tvm::ffi::TensorView cache_indices,
      tvm::ffi::TensorView cache_mask,
      tvm::ffi::TensorView k_weight,
      tvm::ffi::TensorView v_weight,
      tvm::ffi::TensorView k_inter,
      tvm::ffi::TensorView v_inter,
      tvm::ffi::TensorView q_gamma,
      tvm::ffi::TensorView k_gamma,
      double eps,
      tvm::ffi::TensorView q_out,
      tvm::ffi::TensorView k_out,
      tvm::ffi::TensorView v_out,
      tvm::ffi::TensorView loc,
      tvm::ffi::TensorView k_buf,
      tvm::ffi::TensorView v_buf,
      tvm::ffi::TensorView sfq,
      tvm::ffi::TensorView sfk,
      tvm::ffi::TensorView sfv,
      int64_t q_off,
      int64_t k_off,
      int64_t v_off,
      int64_t q_num,
      int64_t do_store,
      int64_t page_size,
      tvm::ffi::TensorView log_tau) {
    using namespace host;
    const uint32_t T = static_cast<uint32_t>(qkvr.size(0));
    const uint32_t B = static_cast<uint32_t>(cache_indices.size(0));
    const uint32_t dq = static_cast<uint32_t>(q_out.size(1));
    const uint32_t dkv = static_cast<uint32_t>(k_out.size(1));
    RuntimeCheck(q_num > 0 && T == B * static_cast<uint32_t>(q_num), "T != B*q");
    RuntimeCheck(dq % kHeadDim == 0 && dkv % kHeadDim == 0, "dims % head_dim");
    RuntimeCheck((dq / kVecElems) % kHeadLanes == 0, "q lanes must tile heads");
    RuntimeCheck(
        qkvr.stride(1) == 1 && qkvr.stride(0) % kVecElems == 0, "qkvr must be row-major with 16B-aligned rows");
    RuntimeCheck(
        q_off % kVecElems == 0 && k_off % kVecElems == 0 && v_off % kVecElems == 0,
        "slice offsets must be 16B aligned");
    RuntimeCheck(k_buf.stride(0) == v_buf.stride(0), "kv buf stride mismatch");
    RuntimeCheck(k_cache.stride(2) == 1 && v_cache.stride(2) == 1, "conv caches must be channel-contiguous");
    RuntimeCheck(
        k_inter.stride(3) == 1 && v_inter.stride(3) == 1 && k_inter.stride(0) == v_inter.stride(0) &&
            k_inter.stride(1) == v_inter.stride(1) && k_inter.stride(2) == v_inter.stride(2),
        "inter buffers must be channel-contiguous with equal strides");
    const uint32_t lanes = dq / kVecElems + 2 * (dkv / kVecElems);
    RuntimeCheck(lanes <= 1024, "token lanes must fit one block");
    if constexpr (USE_MXFP8) {
      RuntimeCheck(do_store, "MXFP8 fused prologue requires do_store=True");
      RuntimeCheck(dq % kMXFP8Block == 0 && dkv % kMXFP8Block == 0, "MXFP8 dims must tile 32-element scale blocks");
      RuntimeCheck(page_size > 0 && page_size % kMXFP8Block == 0, "MXFP8 page size must tile 32-token scale blocks");
      RuntimeCheck(is_type<fp8_e4m3_t>(q_out.dtype()), "MXFP8 q_out must be fp8_e4m3");
      RuntimeCheck(
          is_type<fp8_e4m3_t>(k_buf.dtype()) && is_type<fp8_e4m3_t>(v_buf.dtype()),
          "MXFP8 KV buffers must be fp8_e4m3");
      RuntimeCheck(
          is_type<uint8_t>(sfq.dtype()) && is_type<uint8_t>(sfk.dtype()) && is_type<uint8_t>(sfv.dtype()),
          "MXFP8 scale buffers must be passed as uint8 views");
      RuntimeCheck(q_out.stride(1) == 1 && q_out.stride(0) == dq, "MXFP8 q_out must be contiguous");
      RuntimeCheck(
          sfq.stride(2) == 1 && sfq.stride(1) == kHeadDim / kMXFP8Block, "MXFP8 sfq must be contiguous [T, Hq, 4]");
      const int64_t hkv = dkv / kHeadDim;
      const int64_t sf_dim = kHeadDim / kMXFP8Block;
      const int64_t page_chunks = page_size / kMXFP8Block;
      RuntimeCheck(sfk.ndim() == 5 && sfv.ndim() == 5, "MXFP8 SFK/SFV must be 5D interleaved");
      RuntimeCheck(
          sfk.size(1) == hkv && sfv.size(1) == hkv && sfk.size(2) == kMXFP8Block && sfv.size(2) == kMXFP8Block &&
              sfk.size(3) == page_chunks && sfv.size(3) == page_chunks && sfk.size(4) == sf_dim &&
              sfv.size(4) == sf_dim,
          "MXFP8 SFK/SFV must use [pages, Hkv, 32, page/32, 4] layout");
      RuntimeCheck(
          sfk.stride(4) == 1 && sfv.stride(4) == 1 && sfk.stride(3) == sf_dim && sfv.stride(3) == sf_dim &&
              sfk.stride(2) == page_chunks * sf_dim && sfv.stride(2) == page_chunks * sf_dim &&
              sfk.stride(1) == kMXFP8Block * page_chunks * sf_dim &&
              sfv.stride(1) == kMXFP8Block * page_chunks * sf_dim,
          "MXFP8 SFK/SFV must be contiguous BlockScaledBasicChunk layout");
      RuntimeCheck(k_buf.stride(0) % kMXFP8Block == 0, "MXFP8 kv buf row alignment");
    } else {
      RuntimeCheck(k_buf.stride(0) % kVecElems == 0, "kv buf rows must be 16B aligned");
      RuntimeCheck(is_type<DType>(q_out.dtype()), "q_out dtype mismatch");
    }

    const bool do_tau = log_tau.numel() > 0;
    if (do_tau) {
      RuntimeCheck(is_type<fp32_t>(log_tau.dtype()), "log_tau must be fp32");
      RuntimeCheck(log_tau.IsContiguous(), "log_tau must be contiguous");
      RuntimeCheck(log_tau.numel() >= qkvr.size(0), "log_tau smaller than T");
    }
    const auto params = AttnPrologueParams{
        .qkvr = qkvr.data_ptr(),
        .k_cache = k_cache.data_ptr(),
        .v_cache = v_cache.data_ptr(),
        .cache_indices = cache_indices.data_ptr(),
        .cache_mask = cache_mask.data_ptr(),
        .k_weight = k_weight.data_ptr(),
        .v_weight = v_weight.data_ptr(),
        .k_inter = k_inter.data_ptr(),
        .v_inter = v_inter.data_ptr(),
        .q_gamma = q_gamma.data_ptr(),
        .k_gamma = k_gamma.data_ptr(),
        .log_tau = do_tau ? log_tau.data_ptr() : nullptr,
        .eps = static_cast<float>(eps),
        .q_out = q_out.data_ptr(),
        .k_out = k_out.data_ptr(),
        .v_out = v_out.data_ptr(),
        .loc = loc.data_ptr(),
        .k_buf = k_buf.data_ptr(),
        .v_buf = v_buf.data_ptr(),
        .sfq = USE_MXFP8 ? sfq.data_ptr() : nullptr,
        .sfk = USE_MXFP8 ? sfk.data_ptr() : nullptr,
        .sfv = USE_MXFP8 ? sfv.data_ptr() : nullptr,
        .qkvr_stride_t = qkvr.stride(0),
        .q_off = q_off,
        .k_off = k_off,
        .v_off = v_off,
        .cache_stride_slot = k_cache.stride(0),
        .cache_stride_w = k_cache.stride(1),
        .weight_stride_d = k_weight.stride(0),
        .inter_stride_b = k_inter.stride(0),
        .inter_stride_t = k_inter.stride(1),
        .inter_stride_w = k_inter.stride(2),
        .kv_buf_stride = k_buf.stride(0),
        .T = T,
        .q = static_cast<uint32_t>(q_num),
        .dq = dq,
        .dkv = dkv,
        .page_size = static_cast<uint32_t>(page_size),
    };
    RuntimeCheck(
        k_cache.stride(0) == v_cache.stride(0) && k_cache.stride(1) == v_cache.stride(1) &&
            k_weight.stride(0) == v_weight.stride(0),
        "k/v cache+weight strides must match");
    const uint32_t block = div_ceil(lanes, 32u) * 32u;
    const auto kernel = do_store
                            ? inkling_attn_prologue_kernel<DType, W, USE_SILU, USE_RESIDUAL, true, USE_MXFP8, USE_PDL>
                            : inkling_attn_prologue_kernel<DType, W, USE_SILU, USE_RESIDUAL, false, false, USE_PDL>;
    LaunchKernel(dim3{T}, dim3{block}, qkvr.device()).enable_pdl(USE_PDL)(kernel, params);
  }
};









struct AttnPrologueDecodeParams {
  const void* __restrict__ qkvr;
  void* __restrict__ k_cache;
  void* __restrict__ v_cache;
  const void* __restrict__ cache_indices;
  const void* __restrict__ cache_mask;
  const void* __restrict__ k_weight;
  const void* __restrict__ v_weight;
  const void* __restrict__ track_mask;
  const void* __restrict__ track_indices;
  const void* __restrict__ q_gamma;
  const void* __restrict__ k_gamma;
  const void* __restrict__ log_tau;
  float eps;
  void* __restrict__ q_out;
  void* __restrict__ k_out;
  void* __restrict__ v_out;
  const void* __restrict__ loc;
  void* __restrict__ k_buf;
  void* __restrict__ v_buf;
  void* __restrict__ sfq;
  void* __restrict__ sfk;
  void* __restrict__ sfv;
  int64_t qkvr_stride_t;
  int64_t q_off;
  int64_t k_off;
  int64_t v_off;
  int64_t cache_stride_slot;
  int64_t cache_stride_w;
  int64_t weight_stride_d;
  int64_t track_idx_stride;
  int64_t kv_buf_stride;
  uint32_t T;
  uint32_t dq;
  uint32_t dkv;
  uint32_t page_size;
};

template <
    typename DType,
    int W,
    bool USE_SILU,
    bool USE_RESIDUAL,
    bool DO_TRACK,
    bool DO_STORE,
    bool USE_MXFP8,
    bool USE_PDL>
__global__ __launch_bounds__(1024, 1) void inkling_attn_prologue_decode_kernel(
    const __grid_constant__ AttnPrologueDecodeParams p) {
  static_assert(std::is_same_v<DType, __nv_bfloat16>);
  static_assert(!USE_MXFP8 || DO_STORE, "MXFP8 decode prologue quantization owns the KV store");
  constexpr int W1 = W - 1;
  const uint32_t t = blockIdx.x;
  const uint32_t nq = p.dq / kVecElems;
  const uint32_t nkv = p.dkv / kVecElems;
  const uint32_t vi = threadIdx.x;
  const auto* base = static_cast<const __nv_bfloat16*>(p.qkvr);
  const int64_t row = static_cast<int64_t>(t) * p.qkvr_stride_t;






  if (vi < nq) {

    const uint32_t c = vi * kVecElems;
    const uint4 gqraw = *reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(p.q_gamma) + (c % kHeadDim));
    const auto* gq = reinterpret_cast<const __nv_bfloat16*>(&gqraw);
    const bool do_tau = p.log_tau != nullptr;
    float tau = 0.0f;
    if (do_tau) tau = static_cast<const float*>(p.log_tau)[t];
    device::PDLWaitPrimary<USE_PDL>();
    const uint4 raw = *reinterpret_cast<const uint4*>(base + row + p.q_off + c);
    float x[kVecElems];
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      x[j] = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&raw)[j]);
      ss += x[j] * x[j];
    }
    const float inv = head_rmsnorm_inv(ss, p.eps);
    __nv_bfloat162 o[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      o[j] = __floats2bfloat162_rn(
          x[2 * j] * inv * __bfloat162float(gq[2 * j]), x[2 * j + 1] * inv * __bfloat162float(gq[2 * j + 1]));
    }
    if (do_tau) {



#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(o[j]);
        o[j] = __floats2bfloat162_rn(f.x * tau, f.y * tau);
      }
    }
    if constexpr (USE_MXFP8) {
      float q_quant[kVecElems];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(o[j]);
        q_quant[2 * j] = f.x;
        q_quant[2 * j + 1] = f.y;
      }
      const uint32_t sf_idx = static_cast<uint32_t>(t) * (p.dq / kMXFP8Block) + c / kMXFP8Block;
      store_mxfp8_vec(
          q_quant,
          static_cast<uint8_t*>(p.q_out) + static_cast<int64_t>(t) * p.dq,
          static_cast<uint8_t*>(p.sfq) + sf_idx,
          c);
    } else {
      *reinterpret_cast<uint4*>(static_cast<__nv_bfloat16*>(p.q_out) + static_cast<int64_t>(t) * p.dq + c) =
          *reinterpret_cast<const uint4*>(o);
    }
    device::PDLTriggerSecondary<USE_PDL>();
    return;
  }
  if (vi >= nq + 2 * nkv) return;


  const bool is_k = vi < nq + nkv;
  const uint32_t ch = (is_k ? vi - nq : vi - nq - nkv) * kVecElems;
  const int64_t x_off = is_k ? p.k_off : p.v_off;
  auto* cp = static_cast<__nv_bfloat16*>(is_k ? p.k_cache : p.v_cache);
  const auto* wp = static_cast<const __nv_bfloat16*>(is_k ? p.k_weight : p.v_weight);

  const int ci = static_cast<const int32_t*>(p.cache_indices)[t];
  const bool valid = ci != kPadSlot;
  const int slot_id = valid ? ci : 0;
  const float cm = static_cast<const bool*>(p.cache_mask)[t] ? 1.0f : 0.0f;
  const int64_t cache_base = static_cast<int64_t>(slot_id) * p.cache_stride_slot + ch;


  uint4 hist[W1];
#pragma unroll
  for (int w = 0; w < W1; ++w) {
    hist[w] = *reinterpret_cast<const uint4*>(&cp[cache_base + w * p.cache_stride_w]);
  }
  __nv_bfloat16 wt[kVecElems][W];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int64_t wrow = static_cast<int64_t>(ch + j) * p.weight_stride_d;
    if constexpr (W == 4) {
      if (p.weight_stride_d == W) {
        *reinterpret_cast<uint2*>(wt[j]) = *reinterpret_cast<const uint2*>(wp + wrow);
        continue;
      }
    }
#pragma unroll
    for (int w = 0; w < W; ++w)
      wt[j][w] = wp[wrow + w];
  }
  uint4 gkraw = make_uint4(0, 0, 0, 0);
  if (is_k) {
    gkraw = *reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(p.k_gamma) + (ch % kHeadDim));
  }
  device::PDLWaitPrimary<USE_PDL>();
  const uint4 xv = *reinterpret_cast<const uint4*>(base + row + x_off + ch);


  float y[kVecElems];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float xj = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&xv)[j]);
    float acc = 0.0f;
#pragma unroll
    for (int w = 0; w < W1; ++w) {
      const float h = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&hist[w])[j]);
      acc += h * cm * __bfloat162float(wt[j][w]);
    }
    acc += xj * __bfloat162float(wt[j][W1]);
    if constexpr (USE_SILU) acc = __fdividef(acc, 1.0f + __expf(-acc));
    if constexpr (USE_RESIDUAL) acc += xj;
    y[j] = acc;
  }


  if (valid) {
    int64_t track_base = 0;
    bool do_tr = false;
    if constexpr (DO_TRACK) {
      do_tr = static_cast<const bool*>(p.track_mask)[t];
      if (do_tr) {
        const int64_t tslot =
            static_cast<const int64_t*>(p.track_indices)[static_cast<int64_t>(t) * p.track_idx_stride];
        track_base = tslot * p.cache_stride_slot + ch;
      }
    }
    const uint4 zero = make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int w = 0; w < W1; ++w) {
      const uint4 nv = (w < W1 - 1) ? ((cm != 0.0f) ? hist[w + 1] : zero) : xv;
      *reinterpret_cast<uint4*>(&cp[cache_base + w * p.cache_stride_w]) = nv;
      if constexpr (DO_TRACK) {
        if (do_tr) *reinterpret_cast<uint4*>(&cp[track_base + w * p.cache_stride_w]) = nv;
      }
    }
  }


  __nv_bfloat162 o[4];
  if (is_k) {
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      y[j] = __bfloat162float(__float2bfloat16_rn(y[j]));
      ss += y[j] * y[j];
    }
    const float inv = head_rmsnorm_inv(ss, p.eps);
    const auto* gk = reinterpret_cast<const __nv_bfloat16*>(&gkraw);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      o[j] = __floats2bfloat162_rn(
          y[2 * j] * inv * __bfloat162float(gk[2 * j]), y[2 * j + 1] * inv * __bfloat162float(gk[2 * j + 1]));
    }
  } else {
#pragma unroll
    for (int j = 0; j < 4; ++j)
      o[j] = __floats2bfloat162_rn(y[2 * j], y[2 * j + 1]);
  }
  const uint4 ov = *reinterpret_cast<const uint4*>(o);
  auto* out = static_cast<__nv_bfloat16*>(is_k ? p.k_out : p.v_out);
  *reinterpret_cast<uint4*>(out + static_cast<int64_t>(t) * p.dkv + ch) = ov;
  if (DO_STORE && valid) {
    const int64_t kv_slot = static_cast<const int64_t*>(p.loc)[t];
    if (kv_slot >= 0) {
      if constexpr (USE_MXFP8) {
        float xo[kVecElems];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 f = __bfloat1622float2(o[j]);
          xo[2 * j] = f.x;
          xo[2 * j + 1] = f.y;
        }
        auto* buf = static_cast<uint8_t*>(is_k ? p.k_buf : p.v_buf);
        auto* sfb = static_cast<uint8_t*>(is_k ? p.sfk : p.sfv);
        const int64_t po = kv_slot % static_cast<int64_t>(p.page_size);
        const int64_t sf_base = ((kv_slot / static_cast<int64_t>(p.page_size)) * (p.dkv / kHeadDim) + ch / kHeadDim) *
                                    (kMXFP8Block * (p.page_size / kMXFP8Block) * (kHeadDim / kMXFP8Block)) +
                                (po % kMXFP8Block) * ((p.page_size / kMXFP8Block) * (kHeadDim / kMXFP8Block)) +
                                (po / kMXFP8Block) * (kHeadDim / kMXFP8Block) + (ch % kHeadDim) / kMXFP8Block;
        store_mxfp8_vec(xo, buf + kv_slot * p.kv_buf_stride, sfb + sf_base, ch);
      } else {
        auto* buf = static_cast<__nv_bfloat16*>(is_k ? p.k_buf : p.v_buf);
        *reinterpret_cast<uint4*>(buf + kv_slot * p.kv_buf_stride + ch) = ov;
      }
    }
  }
  device::PDLTriggerSecondary<USE_PDL>();
}

template <typename DType, int W, bool USE_SILU, bool USE_RESIDUAL, bool USE_MXFP8, bool USE_PDL>
struct AttnPrologueDecodeKernel {
  static void
  run(tvm::ffi::TensorView qkvr,
      tvm::ffi::TensorView k_cache,
      tvm::ffi::TensorView v_cache,
      tvm::ffi::TensorView cache_indices,
      tvm::ffi::TensorView cache_mask,
      tvm::ffi::TensorView k_weight,
      tvm::ffi::TensorView v_weight,
      tvm::ffi::TensorView track_mask,
      tvm::ffi::TensorView track_indices,
      tvm::ffi::TensorView q_gamma,
      tvm::ffi::TensorView k_gamma,
      double eps,
      tvm::ffi::TensorView q_out,
      tvm::ffi::TensorView k_out,
      tvm::ffi::TensorView v_out,
      tvm::ffi::TensorView loc,
      tvm::ffi::TensorView k_buf,
      tvm::ffi::TensorView v_buf,
      tvm::ffi::TensorView sfq,
      tvm::ffi::TensorView sfk,
      tvm::ffi::TensorView sfv,
      int64_t q_off,
      int64_t k_off,
      int64_t v_off,
      int64_t do_track,
      int64_t do_store,
      int64_t page_size,
      tvm::ffi::TensorView log_tau) {
    using namespace host;
    const uint32_t T = static_cast<uint32_t>(qkvr.size(0));
    const uint32_t dq = static_cast<uint32_t>(q_out.size(1));
    const uint32_t dkv = static_cast<uint32_t>(k_out.size(1));
    RuntimeCheck(dq % kHeadDim == 0 && dkv % kHeadDim == 0, "dims % head_dim");
    RuntimeCheck((dq / kVecElems) % kHeadLanes == 0, "q lanes must tile heads");
    RuntimeCheck(
        qkvr.stride(1) == 1 && qkvr.stride(0) % kVecElems == 0, "qkvr must be row-major with 16B-aligned rows");
    RuntimeCheck(
        q_off % kVecElems == 0 && k_off % kVecElems == 0 && v_off % kVecElems == 0,
        "slice offsets must be 16B aligned");
    RuntimeCheck(k_cache.stride(2) == 1 && v_cache.stride(2) == 1, "conv caches must be channel-contiguous");
    RuntimeCheck(
        k_cache.stride(0) == v_cache.stride(0) && k_cache.stride(1) == v_cache.stride(1) &&
            k_weight.stride(0) == v_weight.stride(0),
        "k/v cache+weight strides must match");
    const uint32_t lanes = dq / kVecElems + 2 * (dkv / kVecElems);
    RuntimeCheck(lanes <= 1024, "token lanes must fit one block");
    RuntimeCheck(k_buf.stride(0) == v_buf.stride(0), "kv buf stride mismatch");
    if constexpr (USE_MXFP8) {
      RuntimeCheck(do_store, "MXFP8 fused decode prologue requires do_store=True");
      RuntimeCheck(dq % kMXFP8Block == 0 && dkv % kMXFP8Block == 0, "MXFP8 dims must tile 32-element scale blocks");
      RuntimeCheck(page_size > 0 && page_size % kMXFP8Block == 0, "MXFP8 page size must tile 32-token scale blocks");
      RuntimeCheck(is_type<fp8_e4m3_t>(q_out.dtype()), "MXFP8 q_out must be fp8_e4m3");
      RuntimeCheck(
          is_type<fp8_e4m3_t>(k_buf.dtype()) && is_type<fp8_e4m3_t>(v_buf.dtype()),
          "MXFP8 KV buffers must be fp8_e4m3");
      RuntimeCheck(
          is_type<uint8_t>(sfq.dtype()) && is_type<uint8_t>(sfk.dtype()) && is_type<uint8_t>(sfv.dtype()),
          "MXFP8 scale buffers must be passed as uint8 views");
      RuntimeCheck(q_out.stride(1) == 1 && q_out.stride(0) == dq, "MXFP8 q_out must be contiguous");
      RuntimeCheck(
          sfq.stride(2) == 1 && sfq.stride(1) == kHeadDim / kMXFP8Block, "MXFP8 sfq must be contiguous [T, Hq, 4]");
      const int64_t hkv = dkv / kHeadDim;
      const int64_t sf_dim = kHeadDim / kMXFP8Block;
      const int64_t page_chunks = page_size / kMXFP8Block;
      RuntimeCheck(sfk.ndim() == 5 && sfv.ndim() == 5, "MXFP8 SFK/SFV must be 5D interleaved");
      RuntimeCheck(
          sfk.size(1) == hkv && sfv.size(1) == hkv && sfk.size(2) == kMXFP8Block && sfv.size(2) == kMXFP8Block &&
              sfk.size(3) == page_chunks && sfv.size(3) == page_chunks && sfk.size(4) == sf_dim &&
              sfv.size(4) == sf_dim,
          "MXFP8 SFK/SFV must use [pages, Hkv, 32, page/32, 4] layout");
      RuntimeCheck(
          sfk.stride(4) == 1 && sfv.stride(4) == 1 && sfk.stride(3) == sf_dim && sfv.stride(3) == sf_dim &&
              sfk.stride(2) == page_chunks * sf_dim && sfv.stride(2) == page_chunks * sf_dim &&
              sfk.stride(1) == kMXFP8Block * page_chunks * sf_dim &&
              sfv.stride(1) == kMXFP8Block * page_chunks * sf_dim,
          "MXFP8 SFK/SFV must be contiguous BlockScaledBasicChunk layout");
      RuntimeCheck(k_buf.stride(0) % kMXFP8Block == 0, "MXFP8 kv buf row alignment");
    } else {
      RuntimeCheck(k_buf.stride(0) % kVecElems == 0, "kv buf rows must be 16B aligned");
      RuntimeCheck(is_type<DType>(q_out.dtype()), "q_out dtype mismatch");
    }

    const bool do_tau = log_tau.numel() > 0;
    if (do_tau) {
      RuntimeCheck(is_type<fp32_t>(log_tau.dtype()), "log_tau must be fp32");
      RuntimeCheck(log_tau.IsContiguous(), "log_tau must be contiguous");
      RuntimeCheck(log_tau.numel() >= qkvr.size(0), "log_tau smaller than T");
    }
    const auto params = AttnPrologueDecodeParams{
        .qkvr = qkvr.data_ptr(),
        .k_cache = k_cache.data_ptr(),
        .v_cache = v_cache.data_ptr(),
        .cache_indices = cache_indices.data_ptr(),
        .cache_mask = cache_mask.data_ptr(),
        .k_weight = k_weight.data_ptr(),
        .v_weight = v_weight.data_ptr(),
        .track_mask = do_track ? track_mask.data_ptr() : nullptr,
        .track_indices = do_track ? track_indices.data_ptr() : nullptr,
        .q_gamma = q_gamma.data_ptr(),
        .k_gamma = k_gamma.data_ptr(),
        .log_tau = do_tau ? log_tau.data_ptr() : nullptr,
        .eps = static_cast<float>(eps),
        .q_out = q_out.data_ptr(),
        .k_out = k_out.data_ptr(),
        .v_out = v_out.data_ptr(),
        .loc = loc.data_ptr(),
        .k_buf = k_buf.data_ptr(),
        .v_buf = v_buf.data_ptr(),
        .sfq = USE_MXFP8 ? sfq.data_ptr() : nullptr,
        .sfk = USE_MXFP8 ? sfk.data_ptr() : nullptr,
        .sfv = USE_MXFP8 ? sfv.data_ptr() : nullptr,
        .qkvr_stride_t = qkvr.stride(0),
        .q_off = q_off,
        .k_off = k_off,
        .v_off = v_off,
        .cache_stride_slot = k_cache.stride(0),
        .cache_stride_w = k_cache.stride(1),
        .weight_stride_d = k_weight.stride(0),
        .track_idx_stride = do_track ? track_indices.stride(0) : 0,
        .kv_buf_stride = k_buf.stride(0),
        .T = T,
        .dq = dq,
        .dkv = dkv,
        .page_size = static_cast<uint32_t>(page_size),
    };
    const uint32_t block = div_ceil(lanes, 32u) * 32u;
    auto pick = [&](auto tr, auto st, auto mx) {
      return inkling_attn_prologue_decode_kernel<
          DType,
          W,
          USE_SILU,
          USE_RESIDUAL,
          decltype(tr)::value,
          decltype(st)::value,
          decltype(mx)::value,
          USE_PDL>;
    };
    const bool tr = do_track != 0, st = do_store != 0;
    const auto kernel = USE_MXFP8 ? (tr ? pick(std::true_type{}, std::true_type{}, std::true_type{})
                                        : pick(std::false_type{}, std::true_type{}, std::true_type{}))
                                  : (tr ? (st ? pick(std::true_type{}, std::true_type{}, std::false_type{})
                                              : pick(std::true_type{}, std::false_type{}, std::false_type{}))
                                        : (st ? pick(std::false_type{}, std::true_type{}, std::false_type{})
                                              : pick(std::false_type{}, std::false_type{}, std::false_type{})));
    LaunchKernel(dim3{T}, dim3{block}, qkvr.device()).enable_pdl(USE_PDL)(kernel, params);
  }
};













struct AttnPrologueExtendParams {
  const void* __restrict__ qkvr;
  const void* __restrict__ k_cache;
  const void* __restrict__ v_cache;
  const void* __restrict__ cache_indices;
  const void* __restrict__ cache_mask;
  const void* __restrict__ cu;
  const void* __restrict__ si;
  const void* __restrict__ k_weight;
  const void* __restrict__ v_weight;
  const void* __restrict__ q_gamma;
  const void* __restrict__ k_gamma;
  const void* __restrict__ log_tau;
  float eps;
  void* __restrict__ q_out;
  void* __restrict__ k_out;
  void* __restrict__ v_out;
  const void* __restrict__ loc;
  void* __restrict__ k_buf;
  void* __restrict__ v_buf;
  void* __restrict__ sfq;
  void* __restrict__ sfk;
  void* __restrict__ sfv;
  int64_t qkvr_stride_t;
  int64_t q_off;
  int64_t k_off;
  int64_t v_off;
  int64_t cache_stride_slot;
  int64_t cache_stride_w;
  int64_t weight_stride_d;
  int64_t kv_buf_stride;
  uint32_t T;
  uint32_t dq;
  uint32_t dkv;
  uint32_t page_size;
};

template <typename DType, int W, bool USE_SILU, bool USE_RESIDUAL, bool DO_STORE, bool USE_MXFP8, bool USE_PDL>
__global__ __launch_bounds__(1024, 1) void inkling_attn_prologue_extend_kernel(
    const __grid_constant__ AttnPrologueExtendParams p) {
  static_assert(std::is_same_v<DType, __nv_bfloat16>);
  static_assert(!USE_MXFP8 || DO_STORE, "MXFP8 prologue quantization owns the KV store");
  constexpr int W1 = W - 1;
  const uint32_t t = blockIdx.x;
  const uint32_t seq = static_cast<uint32_t>(static_cast<const int32_t*>(p.si)[t]);
  const int bos = static_cast<int>(static_cast<const int64_t*>(p.cu)[seq]);
  const uint32_t nq = p.dq / kVecElems;
  const uint32_t nkv = p.dkv / kVecElems;
  const uint32_t vi = threadIdx.x;
  const auto* base = static_cast<const __nv_bfloat16*>(p.qkvr);
  const int64_t row = static_cast<int64_t>(t) * p.qkvr_stride_t;

  const int ci = static_cast<const int32_t*>(p.cache_indices)[seq];
  const bool valid = ci != kPadSlot;
  const int slot_id = valid ? ci : 0;
  const float cm = static_cast<const bool*>(p.cache_mask)[seq] ? 1.0f : 0.0f;







  if (vi < nq) {

    const uint32_t c = vi * kVecElems;
    const uint4 gqraw = *reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(p.q_gamma) + (c % kHeadDim));
    const auto* gq = reinterpret_cast<const __nv_bfloat16*>(&gqraw);
    const bool do_tau = p.log_tau != nullptr;
    float tau = 0.0f;
    if (do_tau) tau = static_cast<const float*>(p.log_tau)[t];
    device::PDLWaitPrimary<USE_PDL>();
    const uint4 raw = *reinterpret_cast<const uint4*>(base + row + p.q_off + c);
    float x[kVecElems];
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      x[j] = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&raw)[j]);
      ss += x[j] * x[j];
    }
    const float inv = head_rmsnorm_inv(ss, p.eps);
    __nv_bfloat162 o[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      o[j] = __floats2bfloat162_rn(
          x[2 * j] * inv * __bfloat162float(gq[2 * j]), x[2 * j + 1] * inv * __bfloat162float(gq[2 * j + 1]));
    }
    if (do_tau) {



#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(o[j]);
        o[j] = __floats2bfloat162_rn(f.x * tau, f.y * tau);
      }
    }
    if constexpr (USE_MXFP8) {
      float q_quant[kVecElems];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(o[j]);
        q_quant[2 * j] = f.x;
        q_quant[2 * j + 1] = f.y;
      }
      const uint32_t sf_idx = static_cast<uint32_t>(t) * (p.dq / kMXFP8Block) + c / kMXFP8Block;
      store_mxfp8_vec(
          q_quant,
          static_cast<uint8_t*>(p.q_out) + static_cast<int64_t>(t) * p.dq,
          static_cast<uint8_t*>(p.sfq) + sf_idx,
          c);
    } else {
      *reinterpret_cast<uint4*>(static_cast<__nv_bfloat16*>(p.q_out) + static_cast<int64_t>(t) * p.dq + c) =
          *reinterpret_cast<const uint4*>(o);
    }
    device::PDLTriggerSecondary<USE_PDL>();
    return;
  }
  if (vi >= nq + 2 * nkv) return;


  const bool is_k = vi < nq + nkv;
  const uint32_t ch = (is_k ? vi - nq : vi - nq - nkv) * kVecElems;
  const int64_t x_off = is_k ? p.k_off : p.v_off;
  const auto* cp = static_cast<const __nv_bfloat16*>(is_k ? p.k_cache : p.v_cache);
  const auto* wp = static_cast<const __nv_bfloat16*>(is_k ? p.k_weight : p.v_weight);
  const int64_t cache_base = static_cast<int64_t>(slot_id) * p.cache_stride_slot + ch;

  uint4 pref[W1];
  __nv_bfloat16 wt[kVecElems][W];
#pragma unroll
  for (int w = 0; w < W1; ++w) {
    pref[w] = *reinterpret_cast<const uint4*>(&cp[cache_base + w * p.cache_stride_w]);
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int64_t wrow = static_cast<int64_t>(ch + j) * p.weight_stride_d;
    if constexpr (W == 4) {
      if (p.weight_stride_d == W) {
        *reinterpret_cast<uint2*>(wt[j]) = *reinterpret_cast<const uint2*>(wp + wrow);
        continue;
      }
    }
#pragma unroll
    for (int w = 0; w < W; ++w)
      wt[j][w] = wp[wrow + w];
  }

  uint4 gkraw = make_uint4(0, 0, 0, 0);
  if (is_k) {
    gkraw = *reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(p.k_gamma) + (ch % kHeadDim));
  }
  device::PDLWaitPrimary<USE_PDL>();

  const uint4 xcur = *reinterpret_cast<const uint4*>(base + row + x_off + ch);
  uint4 xn[W1];
#pragma unroll
  for (int j = 1; j <= W1; ++j) {
    const int n = static_cast<int>(t) - j;
    if (n >= bos) {
      xn[j - 1] = *reinterpret_cast<const uint4*>(base + static_cast<int64_t>(n) * p.qkvr_stride_t + x_off + ch);
    }
  }

  float y[kVecElems];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float xj = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&xcur)[j]);
    float acc = 0.0f;
#pragma unroll
    for (int iw = 0; iw < W1; ++iw) {
      const int shifted = static_cast<int>(t) - W1 + iw;
      float tap = 0.0f;
      if (shifted >= bos) {
        tap = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&xn[W1 - 1 - iw])[j]);
      } else {
        const int prefix_pos = shifted - bos + W1;
        if (prefix_pos >= 0) {
          tap = cm * __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&pref[prefix_pos])[j]);
        }
      }
      acc += tap * __bfloat162float(wt[j][iw]);
    }
    acc += xj * __bfloat162float(wt[j][W1]);
    if constexpr (USE_SILU) acc = __fdividef(acc, 1.0f + __expf(-acc));
    if constexpr (USE_RESIDUAL) acc += xj;
    y[j] = acc;
  }

  __nv_bfloat162 o[4];
  if (is_k) {



    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      y[j] = __bfloat162float(__float2bfloat16_rn(y[j]));
      ss += y[j] * y[j];
    }
    const float inv = head_rmsnorm_inv(ss, p.eps);
    const auto* gk = reinterpret_cast<const __nv_bfloat16*>(&gkraw);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      o[j] = __floats2bfloat162_rn(
          y[2 * j] * inv * __bfloat162float(gk[2 * j]), y[2 * j + 1] * inv * __bfloat162float(gk[2 * j + 1]));
    }
  } else {
#pragma unroll
    for (int j = 0; j < 4; ++j)
      o[j] = __floats2bfloat162_rn(y[2 * j], y[2 * j + 1]);
  }
  const uint4 ov = *reinterpret_cast<const uint4*>(o);
  auto* out = static_cast<__nv_bfloat16*>(is_k ? p.k_out : p.v_out);
  *reinterpret_cast<uint4*>(out + static_cast<int64_t>(t) * p.dkv + ch) = ov;
  if (DO_STORE) {
    const int64_t kv_slot = static_cast<const int64_t*>(p.loc)[t];
    if (kv_slot >= 0) {
      if constexpr (USE_MXFP8) {
        float xo[kVecElems];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 f = __bfloat1622float2(o[j]);
          xo[2 * j] = f.x;
          xo[2 * j + 1] = f.y;
        }
        auto* buf = static_cast<uint8_t*>(is_k ? p.k_buf : p.v_buf);
        auto* sfb = static_cast<uint8_t*>(is_k ? p.sfk : p.sfv);
        const int64_t po = kv_slot % static_cast<int64_t>(p.page_size);
        constexpr uint32_t SF = kHeadDim / kMXFP8Block;
        const int64_t sf_base = ((kv_slot / static_cast<int64_t>(p.page_size)) * (p.dkv / kHeadDim) + ch / kHeadDim) *
                                    (kMXFP8Block * (p.page_size / kMXFP8Block) * SF) +
                                (po % kMXFP8Block) * ((p.page_size / kMXFP8Block) * SF) + (po / kMXFP8Block) * SF +
                                (ch % kHeadDim) / kMXFP8Block;
        store_mxfp8_vec(xo, buf + kv_slot * p.kv_buf_stride, sfb + sf_base, ch);
      } else {
        auto* buf = static_cast<__nv_bfloat16*>(is_k ? p.k_buf : p.v_buf);
        *reinterpret_cast<uint4*>(buf + kv_slot * p.kv_buf_stride + ch) = ov;
      }
    }
  }
  device::PDLTriggerSecondary<USE_PDL>();
}





struct KvConvUpdateParams {
  const void* __restrict__ qkvr;
  void* __restrict__ k_cache;
  void* __restrict__ v_cache;
  const void* __restrict__ cache_indices;
  const void* __restrict__ has_init;
  const void* __restrict__ cu;
  const void* __restrict__ track_rows;
  const void* __restrict__ track_mask;
  const void* __restrict__ track_dst;
  int64_t qkvr_stride_t;
  int64_t k_off;
  int64_t v_off;
  int64_t cache_stride_slot;
  int64_t cache_stride_w;
  int64_t track_dst_stride;
  uint32_t B;
  uint32_t dkv;
};

template <int W, bool DO_TRACK>
__global__ void inkling_kv_conv_update_kernel(const __grid_constant__ KvConvUpdateParams p) {
  constexpr int W1 = W - 1;
  const uint32_t nkv = p.dkv / kVecElems;
  const uint32_t items = p.B * 2u * nkv;
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= items) return;
  const uint32_t b = idx / (2u * nkv);
  const uint32_t r = idx % (2u * nkv);
  const bool is_k = r < nkv;
  const uint32_t ch = (is_k ? r : r - nkv) * kVecElems;
  const auto* base = static_cast<const __nv_bfloat16*>(p.qkvr);
  const int64_t x_off = is_k ? p.k_off : p.v_off;
  auto* cp = static_cast<__nv_bfloat16*>(is_k ? p.k_cache : p.v_cache);
  const auto* cu = static_cast<const int64_t*>(p.cu);
  const int slot = static_cast<const int32_t*>(p.cache_indices)[b];
  const int64_t qlen = cu[b + 1] - cu[b];
  if (slot != kPadSlot && qlen > 0) {
    const bool hs = static_cast<const bool*>(p.has_init)[b];
    const int64_t cb = static_cast<int64_t>(slot) * p.cache_stride_slot + ch;
    uint4 old_reg[W1];
#pragma unroll
    for (int w = 0; w < W1; ++w) {
      old_reg[w] = *reinterpret_cast<const uint4*>(&cp[cb + w * p.cache_stride_w]);
    }
    const uint4 zero = make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int w = 0; w < W1; ++w) {
      uint4 nv;
      if (qlen >= W1 - w) {
        const int64_t row = cu[b + 1] - W1 + w;
        nv = *reinterpret_cast<const uint4*>(base + row * p.qkvr_stride_t + x_off + ch);
      } else {
        uint4 shift = zero;
#pragma unroll
        for (int src = 0; src < W1; ++src) {
          if (src == w + qlen) shift = old_reg[src];
        }
        nv = hs ? shift : zero;
      }
      *reinterpret_cast<uint4*>(&cp[cb + w * p.cache_stride_w]) = nv;
    }
  }
  if constexpr (DO_TRACK) {
    if (static_cast<const bool*>(p.track_mask)[b]) {
      const int64_t dst = static_cast<const int64_t*>(p.track_dst)[static_cast<int64_t>(b) * p.track_dst_stride];
      const int64_t db = dst * p.cache_stride_slot + ch;
      const auto* trows = static_cast<const int64_t*>(p.track_rows);
#pragma unroll
      for (int w = 0; w < W1; ++w) {
        const int64_t row = trows[static_cast<int64_t>(b) * W1 + w];
        *reinterpret_cast<uint4*>(&cp[db + w * p.cache_stride_w]) =
            *reinterpret_cast<const uint4*>(base + row * p.qkvr_stride_t + x_off + ch);
      }
    }
  }
}

template <typename DType, int W, bool USE_SILU, bool USE_RESIDUAL, bool USE_MXFP8, bool USE_PDL>
struct AttnPrologueExtendKernel {
  static void
  run(tvm::ffi::TensorView qkvr,
      tvm::ffi::TensorView k_cache,
      tvm::ffi::TensorView v_cache,
      tvm::ffi::TensorView cache_indices,
      tvm::ffi::TensorView cache_mask,
      tvm::ffi::TensorView has_init,
      tvm::ffi::TensorView cu,
      tvm::ffi::TensorView si,
      tvm::ffi::TensorView k_weight,
      tvm::ffi::TensorView v_weight,
      tvm::ffi::TensorView track_rows,
      tvm::ffi::TensorView track_mask,
      tvm::ffi::TensorView track_dst,
      tvm::ffi::TensorView q_gamma,
      tvm::ffi::TensorView k_gamma,
      double eps,
      tvm::ffi::TensorView q_out,
      tvm::ffi::TensorView k_out,
      tvm::ffi::TensorView v_out,
      tvm::ffi::TensorView loc,
      tvm::ffi::TensorView k_buf,
      tvm::ffi::TensorView v_buf,
      tvm::ffi::TensorView sfq,
      tvm::ffi::TensorView sfk,
      tvm::ffi::TensorView sfv,
      int64_t q_off,
      int64_t k_off,
      int64_t v_off,
      int64_t do_store,
      int64_t page_size,
      int64_t do_cache_update,
      tvm::ffi::TensorView log_tau) {
    using namespace host;
    const uint32_t T = static_cast<uint32_t>(qkvr.size(0));
    const uint32_t B = static_cast<uint32_t>(cache_indices.size(0));
    const uint32_t dq = static_cast<uint32_t>(q_out.size(1));
    const uint32_t dkv = static_cast<uint32_t>(k_out.size(1));
    RuntimeCheck(dq % kHeadDim == 0 && dkv % kHeadDim == 0, "dims % head_dim");
    RuntimeCheck((dq / kVecElems) % kHeadLanes == 0, "q lanes must tile heads");
    RuntimeCheck(
        qkvr.stride(1) == 1 && qkvr.stride(0) % kVecElems == 0, "qkvr must be row-major with 16B-aligned rows");
    RuntimeCheck(
        q_off % kVecElems == 0 && k_off % kVecElems == 0 && v_off % kVecElems == 0,
        "slice offsets must be 16B aligned");
    RuntimeCheck(si.size(0) >= T, "si must cover T tokens");
    RuntimeCheck(cu.size(0) == B + 1, "cu must be [B+1]");
    RuntimeCheck(cache_mask.size(0) == B && has_init.size(0) == B, "per-seq arrays must be [B]");
    RuntimeCheck(k_cache.stride(2) == 1 && v_cache.stride(2) == 1, "conv caches must be channel-contiguous");
    RuntimeCheck(
        k_cache.stride(0) == v_cache.stride(0) && k_cache.stride(1) == v_cache.stride(1) &&
            k_weight.stride(0) == v_weight.stride(0),
        "k/v cache+weight strides must match");
    const uint32_t lanes = dq / kVecElems + 2 * (dkv / kVecElems);
    RuntimeCheck(lanes <= 1024, "token lanes must fit one block");
    RuntimeCheck(k_buf.stride(0) == v_buf.stride(0), "kv buf stride mismatch");
    const bool do_track = track_mask.numel() > 0;
    if (do_track) {
      RuntimeCheck(track_rows.numel() > 0, "extend track needs gather rows");
    }
    if constexpr (USE_MXFP8) {
      RuntimeCheck(do_store, "MXFP8 fused extend prologue requires do_store=True");
      RuntimeCheck(dq % kMXFP8Block == 0 && dkv % kMXFP8Block == 0, "MXFP8 dims must tile 32-element scale blocks");
      RuntimeCheck(page_size > 0 && page_size % kMXFP8Block == 0, "MXFP8 page size must tile 32-token scale blocks");
      RuntimeCheck(is_type<fp8_e4m3_t>(q_out.dtype()), "MXFP8 q_out must be fp8_e4m3");
      RuntimeCheck(
          is_type<fp8_e4m3_t>(k_buf.dtype()) && is_type<fp8_e4m3_t>(v_buf.dtype()),
          "MXFP8 KV buffers must be fp8_e4m3");
      RuntimeCheck(
          is_type<uint8_t>(sfq.dtype()) && is_type<uint8_t>(sfk.dtype()) && is_type<uint8_t>(sfv.dtype()),
          "MXFP8 scale buffers must be passed as uint8 views");
      RuntimeCheck(q_out.stride(1) == 1 && q_out.stride(0) == dq, "MXFP8 q_out must be contiguous");
      RuntimeCheck(
          sfq.stride(2) == 1 && sfq.stride(1) == kHeadDim / kMXFP8Block, "MXFP8 sfq must be contiguous [T, Hq, 4]");
      const int64_t hkv = dkv / kHeadDim;
      const int64_t sf_dim = kHeadDim / kMXFP8Block;
      const int64_t page_chunks = page_size / kMXFP8Block;
      RuntimeCheck(sfk.ndim() == 5 && sfv.ndim() == 5, "MXFP8 SFK/SFV must be 5D interleaved");
      RuntimeCheck(
          sfk.size(1) == hkv && sfv.size(1) == hkv && sfk.size(2) == kMXFP8Block && sfv.size(2) == kMXFP8Block &&
              sfk.size(3) == page_chunks && sfv.size(3) == page_chunks && sfk.size(4) == sf_dim &&
              sfv.size(4) == sf_dim,
          "MXFP8 SFK/SFV must use [pages, Hkv, 32, page/32, 4] layout");
      RuntimeCheck(
          sfk.stride(4) == 1 && sfv.stride(4) == 1 && sfk.stride(3) == sf_dim && sfv.stride(3) == sf_dim &&
              sfk.stride(2) == page_chunks * sf_dim && sfv.stride(2) == page_chunks * sf_dim &&
              sfk.stride(1) == kMXFP8Block * page_chunks * sf_dim &&
              sfv.stride(1) == kMXFP8Block * page_chunks * sf_dim,
          "MXFP8 SFK/SFV must be contiguous BlockScaledBasicChunk layout");
      RuntimeCheck(k_buf.stride(0) % kMXFP8Block == 0, "MXFP8 kv buf row alignment");
    } else {
      RuntimeCheck(k_buf.stride(0) % kVecElems == 0, "kv buf rows must be 16B aligned");
      RuntimeCheck(is_type<DType>(q_out.dtype()), "q_out dtype mismatch");
    }
    if (T == 0) return;

    const bool do_tau = log_tau.numel() > 0;
    if (do_tau) {
      RuntimeCheck(is_type<fp32_t>(log_tau.dtype()), "log_tau must be fp32");
      RuntimeCheck(log_tau.IsContiguous(), "log_tau must be contiguous");
      RuntimeCheck(log_tau.numel() >= qkvr.size(0), "log_tau smaller than T");
    }
    const auto params = AttnPrologueExtendParams{
        .qkvr = qkvr.data_ptr(),
        .k_cache = k_cache.data_ptr(),
        .v_cache = v_cache.data_ptr(),
        .cache_indices = cache_indices.data_ptr(),
        .cache_mask = cache_mask.data_ptr(),
        .cu = cu.data_ptr(),
        .si = si.data_ptr(),
        .k_weight = k_weight.data_ptr(),
        .v_weight = v_weight.data_ptr(),
        .q_gamma = q_gamma.data_ptr(),
        .k_gamma = k_gamma.data_ptr(),
        .log_tau = do_tau ? log_tau.data_ptr() : nullptr,
        .eps = static_cast<float>(eps),
        .q_out = q_out.data_ptr(),
        .k_out = k_out.data_ptr(),
        .v_out = v_out.data_ptr(),
        .loc = loc.data_ptr(),
        .k_buf = k_buf.data_ptr(),
        .v_buf = v_buf.data_ptr(),
        .sfq = USE_MXFP8 ? sfq.data_ptr() : nullptr,
        .sfk = USE_MXFP8 ? sfk.data_ptr() : nullptr,
        .sfv = USE_MXFP8 ? sfv.data_ptr() : nullptr,
        .qkvr_stride_t = qkvr.stride(0),
        .q_off = q_off,
        .k_off = k_off,
        .v_off = v_off,
        .cache_stride_slot = k_cache.stride(0),
        .cache_stride_w = k_cache.stride(1),
        .weight_stride_d = k_weight.stride(0),
        .kv_buf_stride = k_buf.stride(0),
        .T = T,
        .dq = dq,
        .dkv = dkv,
        .page_size = static_cast<uint32_t>(page_size),
    };
    const uint32_t block = div_ceil(lanes, 32u) * 32u;
    const auto kernel =
        do_store ? inkling_attn_prologue_extend_kernel<DType, W, USE_SILU, USE_RESIDUAL, true, USE_MXFP8, USE_PDL>
                 : inkling_attn_prologue_extend_kernel<DType, W, USE_SILU, USE_RESIDUAL, false, false, USE_PDL>;
    LaunchKernel(dim3{T}, dim3{block}, qkvr.device()).enable_pdl(USE_PDL)(kernel, params);





    if (!do_cache_update) return;
    const auto uparams = KvConvUpdateParams{
        .qkvr = qkvr.data_ptr(),
        .k_cache = k_cache.data_ptr(),
        .v_cache = v_cache.data_ptr(),
        .cache_indices = cache_indices.data_ptr(),
        .has_init = has_init.data_ptr(),
        .cu = cu.data_ptr(),
        .track_rows = do_track ? track_rows.data_ptr() : nullptr,
        .track_mask = do_track ? track_mask.data_ptr() : nullptr,
        .track_dst = do_track ? track_dst.data_ptr() : nullptr,
        .qkvr_stride_t = qkvr.stride(0),
        .k_off = k_off,
        .v_off = v_off,
        .cache_stride_slot = k_cache.stride(0),
        .cache_stride_w = k_cache.stride(1),
        .track_dst_stride = do_track ? track_dst.stride(0) : 0,
        .B = B,
        .dkv = dkv,
    };
    const uint32_t uitems = B * 2u * (dkv / kVecElems);
    const uint32_t ublock = 256;
    const uint32_t ugrid = div_ceil(uitems, ublock);
    const auto ukernel = do_track ? inkling_kv_conv_update_kernel<W, true> : inkling_kv_conv_update_kernel<W, false>;
    LaunchKernel(dim3{ugrid}, dim3{ublock}, qkvr.device())(ukernel, uparams);
  }
};

}
