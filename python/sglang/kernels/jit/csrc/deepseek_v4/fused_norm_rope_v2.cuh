#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/compress_v2.cuh>
#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

using PlanC = device::compress::CompressPlan;
using PlanD = device::compress::DecodePlan;
using deepseek_v4::fp8::cast_to_ue8m0;
using deepseek_v4::fp8::inv_scale_ue8m0;
using deepseek_v4::fp8::pack_fp8;

SGL_DEVICE uint8_t quant_fp4_e2m1(float x) {
  const float ax = fminf(fabsf(x), 6.0f);
  uint8_t idx = 0;
  idx += ax > 0.25f;
  idx += ax > 0.75f;
  idx += ax > 1.25f;
  idx += ax > 1.75f;
  idx += ax > 2.5f;
  idx += ax > 3.5f;
  idx += ax > 5.0f;
  if (x < 0.0f && idx != 0) idx |= 0x8;
  return idx;
}

constexpr uint32_t kBlockSize = 256;
constexpr uint32_t kNumWarps = kBlockSize / device::kWarpThreads;

struct FusedNormRopeStoreParams {
  void* __restrict__ input;
  const void* __restrict__ handle;
  const void* __restrict__ weight;
  const float* __restrict__ freqs_cis;
  const int64_t* __restrict__ out_loc;
  uint8_t* __restrict__ kvcache;
  float eps;
  uint32_t compress_ratio;
  uint32_t num_tokens;
};

enum class ForwardMode : bool {
  CompressExtend = 0,
  CompressDecode = 1,
};

#define INDEXER_KERNEL __global__ __launch_bounds__(kBlockSize, 8)
#define FLASHMLA_KERNEL __global__ __launch_bounds__(kBlockSize, 8)






template <typename DType, ForwardMode kMode, int32_t kPageBits, bool kUsePDL, int32_t kPreshuffleSize = 0>
INDEXER_KERNEL void fused_norm_rope_indexer(const __grid_constant__ FusedNormRopeStoreParams params) {
  using namespace device;
  using enum ForwardMode;

  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kRopeDim = 64;
  constexpr int64_t kVecSize = 4;
  constexpr uint32_t kRopeSize = kRopeDim / kVecSize;
  constexpr int64_t kPageBytes = 132ll << kPageBits;
  static_assert(kHeadDim == kWarpThreads * kVecSize);
  static_assert(kRopeDim == kWarpThreads * 2);
  static_assert(kRopeSize <= kWarpThreads);
  using Storage = AlignedVector<DType, kVecSize>;
  using Float4 = AlignedVector<float, kVecSize>;

  const auto warp_id = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto work_id = blockIdx.x * kNumWarps + warp_id;

  const bool is_rope_lane = lane_id >= kWarpThreads - kRopeSize;

  if (work_id >= params.num_tokens) return;

  const auto input = static_cast<DType*>(params.input) + work_id * kHeadDim;
  int32_t position;
  int64_t out_loc;
  if constexpr (kMode == CompressExtend) {
    const auto plan = static_cast<const PlanC*>(params.handle)[work_id];
    if (plan.is_invalid()) return;
    position = plan.seq_len - params.compress_ratio;
    out_loc = params.out_loc[plan.ragged_id];
  } else if constexpr (kMode == CompressDecode) {
    const auto plan = static_cast<const PlanD*>(params.handle)[work_id];
    if (plan.seq_len % params.compress_ratio != 0) return;
    position = plan.seq_len - params.compress_ratio;
    out_loc = params.out_loc[work_id];
  } else {
    static_assert(host::dependent_false_v<DType>, "Unsupported Mode");
  }
  const auto freqs_cis = params.freqs_cis + position * kRopeDim;

  PDLWaitPrimary<kUsePDL>();
  Float4 data, freq;


  {
    Storage input_vec, weight_vec;
    input_vec.load(input, lane_id);
    weight_vec.load(params.weight, lane_id);
    if (is_rope_lane) freq.load(freqs_cis, lane_id - (kWarpThreads - kRopeSize));

    float sum_of_squares = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const auto fp32_input = cast<float>(input_vec[i]);
      sum_of_squares += fp32_input * fp32_input;
    }

    sum_of_squares = warp::reduce_sum(sum_of_squares);
    const auto norm_factor = math::rsqrt(sum_of_squares / kHeadDim + params.eps);

#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const auto fp32_input = cast<float>(input_vec[i]);
      const auto fp32_weight = cast<float>(weight_vec[i]);
      data[i] = fp32_input * norm_factor * fp32_weight;
    }
  }


  if (is_rope_lane) {
    const auto x_real = data[0];
    const auto x_imag = data[1];
    const auto y_real = data[2];
    const auto y_imag = data[3];
    const auto freq_x_real = freq[0];
    const auto freq_x_imag = freq[1];
    const auto freq_y_real = freq[2];
    const auto freq_y_imag = freq[3];
    data[0] = x_real * freq_x_real - x_imag * freq_x_imag;
    data[1] = x_real * freq_x_imag + x_imag * freq_x_real;
    data[2] = y_real * freq_y_real - y_imag * freq_y_imag;
    data[3] = y_real * freq_y_imag + y_imag * freq_y_real;
  }


  {

    {
      const float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
      data[0] = a0 + a1;
      data[1] = a0 - a1;
      data[2] = a2 + a3;
      data[3] = a2 - a3;
    }

    {
      const float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
      data[0] = a0 + a2;
      data[1] = a1 + a3;
      data[2] = a0 - a2;
      data[3] = a1 - a3;
    }





#pragma unroll
    for (uint32_t mask = 1; mask < kWarpThreads; mask <<= 1) {
#pragma unroll
      for (int i = 0; i < kVecSize; ++i) {
#ifndef USE_ROCM
        const float other = __shfl_xor_sync(kFullMask, data[i], mask, kWarpThreads);
#else
        const float other = __shfl_xor(data[i], mask, kWarpThreads);
#endif
        data[i] = (lane_id & mask) ? (other - data[i]) : (data[i] + other);
      }
    }
    const float kHadamardScale = math::rsqrt(static_cast<float>(kHeadDim));
#pragma unroll
    for (int i = 0; i < kVecSize; ++i)
      data[i] *= kHadamardScale;
  }




  {
    using OutStorage = AlignedVector<fp8x2_e4m3_t, 2>;
    float local_max = math::abs(data[0]);
#pragma unroll
    for (int i = 1; i < kVecSize; ++i) {
      local_max = math::max(local_max, math::abs(data[i]));
    }
    const auto abs_max = warp::reduce_max(local_max);
    const auto scale = fmaxf(1e-4f, abs_max) / kFP8E4M3Max;
    const auto inv_scale = 1.0f / scale;
    const int64_t page = out_loc >> kPageBits;
    const int64_t offset = out_loc & ((1 << kPageBits) - 1);
    const auto page_ptr = params.kvcache + page * kPageBytes;
    const auto value_ptr = page_ptr + offset * 128;
    const auto scale_ptr = page_ptr + (128 << kPageBits) + offset * 4;
    OutStorage result;
    result[0] = pack_fp8(data[0] * inv_scale, data[1] * inv_scale);
    result[1] = pack_fp8(data[2] * inv_scale, data[3] * inv_scale);
    PDLTriggerSecondary<kUsePDL>();
    if constexpr (kPreshuffleSize != 0) {
      constexpr int32_t kTile = kPreshuffleSize;
      const int32_t dim_base = lane_id * kVecSize;
      const int32_t token_tile_id = offset / kTile;
      const int32_t token_in_tile = offset % kTile;
      const int32_t col_tile_id = dim_base / kTile;
      const int32_t col_in_tile = dim_base % kTile;
      const int32_t value_offset = token_tile_id * (kTile * static_cast<int32_t>(kHeadDim)) +
                                   col_tile_id * (kTile * kTile) + token_in_tile * kTile + col_in_tile;
      result.store(page_ptr + value_offset, 0);
    } else {
      result.store(value_ptr, lane_id);
    }

    if (lane_id == 0) reinterpret_cast<float*>(scale_ptr)[0] = scale;
  }
}

template <typename DType, ForwardMode kMode, int32_t kPageBits, bool kUsePDL>
INDEXER_KERNEL void fused_norm_rope_indexer_fp4(const __grid_constant__ FusedNormRopeStoreParams params) {
  using namespace device;
  using enum ForwardMode;

  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kRopeDim = 64;
  constexpr int64_t kVecSize = 4;
  constexpr uint32_t kRopeSize = kRopeDim / kVecSize;
  constexpr int64_t kPageBytes = 68ll << kPageBits;
  static_assert(kHeadDim == kWarpThreads * kVecSize);
  static_assert(kRopeDim == kWarpThreads * 2);
  static_assert(kRopeSize <= kWarpThreads);
  using Storage = AlignedVector<DType, kVecSize>;
  using Float4 = AlignedVector<float, kVecSize>;

  const auto warp_id = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto work_id = blockIdx.x * kNumWarps + warp_id;
  const bool is_rope_lane = lane_id >= kWarpThreads - kRopeSize;

  if (work_id >= params.num_tokens) return;

  const auto input = static_cast<DType*>(params.input) + work_id * kHeadDim;
  int32_t position;
  int64_t out_loc;
  if constexpr (kMode == CompressExtend) {
    const auto plan = static_cast<const PlanC*>(params.handle)[work_id];
    if (plan.is_invalid()) return;
    position = plan.seq_len - params.compress_ratio;
    out_loc = params.out_loc[plan.ragged_id];
  } else if constexpr (kMode == CompressDecode) {
    const auto plan = static_cast<const PlanD*>(params.handle)[work_id];
    if (plan.seq_len % params.compress_ratio != 0) return;
    position = plan.seq_len - params.compress_ratio;
    out_loc = params.out_loc[work_id];
  } else {
    static_assert(host::dependent_false_v<DType>, "Unsupported Mode");
  }
  const auto freqs_cis = params.freqs_cis + position * kRopeDim;

  PDLWaitPrimary<kUsePDL>();
  Float4 data, freq;

  {
    Storage input_vec, weight_vec;
    input_vec.load(input, lane_id);
    weight_vec.load(params.weight, lane_id);
    if (is_rope_lane) freq.load(freqs_cis, lane_id - (kWarpThreads - kRopeSize));

    float sum_of_squares = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const auto fp32_input = cast<float>(input_vec[i]);
      sum_of_squares += fp32_input * fp32_input;
    }

    sum_of_squares = warp::reduce_sum(sum_of_squares);
    const auto norm_factor = math::rsqrt(sum_of_squares / kHeadDim + params.eps);

#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const auto fp32_input = cast<float>(input_vec[i]);
      const auto fp32_weight = cast<float>(weight_vec[i]);
      data[i] = fp32_input * norm_factor * fp32_weight;
    }
  }

  if (is_rope_lane) {
    const auto x_real = data[0];
    const auto x_imag = data[1];
    const auto y_real = data[2];
    const auto y_imag = data[3];
    const auto freq_x_real = freq[0];
    const auto freq_x_imag = freq[1];
    const auto freq_y_real = freq[2];
    const auto freq_y_imag = freq[3];
    data[0] = x_real * freq_x_real - x_imag * freq_x_imag;
    data[1] = x_real * freq_x_imag + x_imag * freq_x_real;
    data[2] = y_real * freq_y_real - y_imag * freq_y_imag;
    data[3] = y_real * freq_y_imag + y_imag * freq_y_real;
  }

  {
    {
      const float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
      data[0] = a0 + a1;
      data[1] = a0 - a1;
      data[2] = a2 + a3;
      data[3] = a2 - a3;
    }
    {
      const float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
      data[0] = a0 + a2;
      data[1] = a1 + a3;
      data[2] = a0 - a2;
      data[3] = a1 - a3;
    }
#pragma unroll
    for (uint32_t mask = 1; mask < kWarpThreads; mask <<= 1) {
#pragma unroll
      for (int i = 0; i < kVecSize; ++i) {
#ifndef USE_ROCM
        const float other = __shfl_xor_sync(kFullMask, data[i], mask, kWarpThreads);
#else
        const float other = __shfl_xor(data[i], mask, kWarpThreads);
#endif
        data[i] = (lane_id & mask) ? (other - data[i]) : (data[i] + other);
      }
    }
    const float kHadamardScale = math::rsqrt(static_cast<float>(kHeadDim));
#pragma unroll
    for (int i = 0; i < kVecSize; ++i)
      data[i] *= kHadamardScale;
  }

  {
    float local_max = math::abs(data[0]);
#pragma unroll
    for (int i = 1; i < kVecSize; ++i) {
      local_max = math::max(local_max, math::abs(data[i]));
    }
    local_max = warp::reduce_max<8>(local_max);

    const auto scale_raw = fmaxf(1e-4f, local_max) / 6.0f;
    const auto scale_ue8m0 = static_cast<uint8_t>(cast_to_ue8m0(scale_raw));
    const auto inv_scale = inv_scale_ue8m0(scale_ue8m0);

    const uint8_t packed0 = quant_fp4_e2m1(data[0] * inv_scale) | (quant_fp4_e2m1(data[1] * inv_scale) << 4);
    const uint8_t packed1 = quant_fp4_e2m1(data[2] * inv_scale) | (quant_fp4_e2m1(data[3] * inv_scale) << 4);
    const uint16_t packed = static_cast<uint16_t>(packed0) | (static_cast<uint16_t>(packed1) << 8);

    const int64_t page = out_loc >> kPageBits;
    const int64_t offset = out_loc & ((1 << kPageBits) - 1);
    const auto page_ptr = params.kvcache + page * kPageBytes;
    const auto value_ptr = page_ptr + offset * 64;
    const auto scale_ptr = page_ptr + (64 << kPageBits) + offset * 4;

    PDLTriggerSecondary<kUsePDL>();
    reinterpret_cast<uint16_t*>(value_ptr)[lane_id] = packed;
    if ((lane_id & 7) == 0) static_cast<uint8_t*>(scale_ptr)[lane_id >> 3] = scale_ue8m0;
  }
}






template <typename DType, ForwardMode kMode, int32_t kPageBits, bool kUsePDL, bool kBf16Store = false>
FLASHMLA_KERNEL void fused_norm_rope_flashmla(const __grid_constant__ FusedNormRopeStoreParams params) {
  using namespace device;
  using enum ForwardMode;

  constexpr int64_t kHeadDim = 512;
  constexpr int64_t kRopeDim = 64;
  constexpr int64_t kVecSize = 2;


  constexpr uint32_t kRopeWarp = kNumWarps - 1;


  constexpr int64_t kPageBytes =
      kBf16Store ? ((kHeadDim * 2ll) << kPageBits) : host::div_ceil(584ll << kPageBits, 576) * 576;
  static_assert(kHeadDim == kBlockSize * kVecSize);
  static_assert(kRopeDim == kWarpThreads * kVecSize);
  static_assert(kHeadDim - kRopeDim == kRopeWarp * kWarpThreads * kVecSize);
  using Storage = AlignedVector<DType, kVecSize>;
  using Float2 = AlignedVector<float, kVecSize>;

  const auto tx = threadIdx.x;
  const auto warp_id = tx / kWarpThreads;
  const auto lane_id = tx % kWarpThreads;
  const auto work_id = blockIdx.x;

  if (work_id >= params.num_tokens) return;

  const auto input = static_cast<DType*>(params.input) + work_id * kHeadDim;
  int32_t position;
  int64_t out_loc;
  if constexpr (kMode == CompressExtend) {
    const auto plan = static_cast<const PlanC*>(params.handle)[work_id];
    if (plan.is_invalid()) return;
    position = plan.seq_len - params.compress_ratio;
    out_loc = params.out_loc[plan.ragged_id];
  } else if constexpr (kMode == CompressDecode) {
    const auto plan = static_cast<const PlanD*>(params.handle)[work_id];
    if (plan.seq_len % params.compress_ratio != 0) return;
    position = plan.seq_len - params.compress_ratio;
    out_loc = params.out_loc[work_id];
  } else {
    static_assert(host::dependent_false_v<DType>, "Unsupported Mode");
  }
  const auto freqs_cis = params.freqs_cis + position * kRopeDim;

  PDLWaitPrimary<kUsePDL>();
  Float2 data, freq;



  {
    __shared__ float partial_sums[kNumWarps];

    Storage input_vec, weight_vec;
    input_vec.load(input, tx);
    weight_vec.load(params.weight, tx);
    if (warp_id == kRopeWarp) freq.load(freqs_cis, lane_id);

    float sum_of_squares = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const auto fp32_input = cast<float>(input_vec[i]);
      sum_of_squares += fp32_input * fp32_input;
    }

    const auto warp_sum = warp::reduce_sum(sum_of_squares);
    if (lane_id == 0) partial_sums[warp_id] = warp_sum;
    __syncthreads();


    sum_of_squares = warp::reduce_sum<kNumWarps>(partial_sums[lane_id % kNumWarps]);
    const auto norm_factor = math::rsqrt(sum_of_squares / kHeadDim + params.eps);

#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const auto fp32_input = cast<float>(input_vec[i]);
      const auto fp32_weight = cast<float>(weight_vec[i]);
      data[i] = fp32_input * norm_factor * fp32_weight;
    }
  }

  const int64_t page = out_loc >> kPageBits;
  const int64_t offset = out_loc & ((1 << kPageBits) - 1);
  const auto page_ptr = params.kvcache + page * kPageBytes;
  const auto value_ptr = page_ptr + offset * (kBf16Store ? (kHeadDim * 2) : 576);

  PDLTriggerSecondary<kUsePDL>();


  if constexpr (kBf16Store) {
    Float2 d = data;
    if (warp_id == kRopeWarp) {
      const auto x_real = data[0];
      const auto x_imag = data[1];
      const auto freq_real = freq[0];
      const auto freq_imag = freq[1];
      d[0] = x_real * freq_real - x_imag * freq_imag;
      d[1] = x_real * freq_imag + x_imag * freq_real;
    }
    reinterpret_cast<bf16x2_t*>(value_ptr)[tx] = cast<bf16x2_t>(fp32x2_t{d[0], d[1]});
  } else if (warp_id == kRopeWarp) {


    const auto x_real = data[0];
    const auto x_imag = data[1];
    const auto freq_real = freq[0];
    const auto freq_imag = freq[1];
    data[0] = x_real * freq_real - x_imag * freq_imag;
    data[1] = x_real * freq_imag + x_imag * freq_real;
    const auto result = cast<bf16x2_t>(fp32x2_t{data[0], data[1]});
    const auto rope_ptr = value_ptr + 448;
    reinterpret_cast<bf16x2_t*>(rope_ptr)[lane_id] = result;
  } else {



    const auto x = cast<float>(cast<bf16_t>(data[0]));
    const auto y = cast<float>(cast<bf16_t>(data[1]));
    const auto abs_max = warp::reduce_max(fmaxf(fabs(x), fabs(y)));
    const auto scale_raw = fmaxf(1e-4f, abs_max) / kFP8E4M3Max;
    const auto scale_ue8m0 = cast_to_ue8m0(scale_raw);
    const auto inv_scale = inv_scale_ue8m0(scale_ue8m0);
    const auto result = pack_fp8(x * inv_scale, y * inv_scale);
    const auto scale_ptr = page_ptr + (576 << kPageBits) + offset * 8;
    reinterpret_cast<fp8x2_e4m3_t*>(value_ptr)[tx] = result;

    if (lane_id == 0) static_cast<uint8_t*>(scale_ptr)[warp_id] = scale_ue8m0;
  }
}

template <
    typename DType,
    int64_t kHeadDim,
    int64_t kRopeDim,
    uint32_t kPageSize,
    bool kUsePDL,
    int32_t kPreshuffleSize = 0,
    bool kBf16Store = false>
struct FusedNormRopeKernel {
  static constexpr int32_t kLogPageSize = std::countr_zero(kPageSize);
  static constexpr bool kIsIndexer = (kHeadDim == 128);
  static_assert(!(kIsIndexer && kBf16Store), "bf16 store only for flashmla head_dim=512");
  static constexpr int64_t kIndexerBytes = 132 * kPageSize;
  static constexpr int64_t kFlashMLABytes = host::div_ceil(584 * kPageSize, 576) * 576;
  static constexpr int64_t kBf16Bytes = kHeadDim * 2 * kPageSize;
  static constexpr int64_t kPageBytes = kBf16Store ? kBf16Bytes : (kIsIndexer ? kIndexerBytes : kFlashMLABytes);


  static_assert(kRopeDim == 64 && (kHeadDim == 128 || kHeadDim == 512));
  static_assert(std::has_single_bit(kPageSize), "kPageSize must be a power of 2");

  template <ForwardMode kMode>
  static constexpr auto select_kernel() {
    if constexpr (kIsIndexer) {
      return fused_norm_rope_indexer<DType, kMode, kLogPageSize, kUsePDL, kPreshuffleSize>;
    } else {
      return fused_norm_rope_flashmla<DType, kMode, kLogPageSize, kUsePDL, kBf16Store>;
    }
  }

  template <ForwardMode kMode>
  static constexpr auto select_fp4_kernel() {
    static_assert(kIsIndexer, "FP4 fused store is only defined for the indexer");
    return fused_norm_rope_indexer_fp4<DType, kMode, kLogPageSize, kUsePDL>;
  }

  static void forward(
      const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView plan,
      const tvm::ffi::TensorView weight,
      const float eps,
      const tvm::ffi::TensorView freqs_cis,
      const tvm::ffi::TensorView out_loc,
      const tvm::ffi::TensorView kvcache,
      const bool is_decode,
      const uint32_t compress_ratio) {
    using namespace host;
    using enum ForwardMode;

    const auto mode = static_cast<ForwardMode>(is_decode);

    auto N = SymbolicSize{"num_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLGPU>();

    TensorMatcher({N, kHeadDim})
        .with_dtype<DType>()
        .with_device(device_)
        .verify(input);
    TensorMatcher({kHeadDim})
        .with_dtype<DType>()
        .with_device(device_)
        .verify(weight);
    TensorMatcher({-1, kRopeDim})
        .with_dtype<float>()
        .with_device(device_)
        .verify(freqs_cis);
    TensorMatcher({-1})
        .with_dtype<int64_t>()
        .with_device(device_)
        .verify(out_loc);
    TensorMatcher({-1, -1})
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(kvcache);

    switch (mode) {
      case CompressExtend:
        compress::verify_plan_c(plan, N, device_);
        RuntimeCheck(out_loc.size(0) >= N.unwrap());
        break;
      case CompressDecode:
        compress::verify_plan_d(plan, N, device_);
        RuntimeCheck(out_loc.size(0) == N.unwrap());
        break;
    }

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    if (num_tokens == 0) return;
    const auto params = FusedNormRopeStoreParams{
        .input = input.data_ptr(),
        .handle = plan.data_ptr(),
        .weight = weight.data_ptr(),
        .freqs_cis = static_cast<const float*>(freqs_cis.data_ptr()),
        .out_loc = static_cast<const int64_t*>(out_loc.data_ptr()),
        .kvcache = static_cast<uint8_t*>(kvcache.data_ptr()),
        .eps = eps,
        .compress_ratio = compress_ratio,
        .num_tokens = num_tokens,
    };


    const uint32_t num_blocks = kIsIndexer ? div_ceil(num_tokens, kNumWarps) : num_tokens;
    const auto device = device_.unwrap();
    const auto kernel = mode == CompressExtend ? select_kernel<CompressExtend>() : select_kernel<CompressDecode>();
    LaunchKernel(num_blocks, kBlockSize, device).enable_pdl(kUsePDL)(kernel, params);
  }

  static void forward_fp4(
      const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView plan,
      const tvm::ffi::TensorView weight,
      const float eps,
      const tvm::ffi::TensorView freqs_cis,
      const tvm::ffi::TensorView out_loc,
      const tvm::ffi::TensorView kvcache,
      const bool is_decode,
      const uint32_t compress_ratio) {
    using namespace host;
    using enum ForwardMode;

    static_assert(kIsIndexer, "FP4 fused store is only defined for the indexer");
    constexpr int64_t kFp4PageBytes = 68 * kPageSize;
    const auto mode = static_cast<ForwardMode>(is_decode);

    auto N = SymbolicSize{"num_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({N, kHeadDim}).with_dtype<DType>().with_device(device_).verify(input);
    TensorMatcher({kHeadDim}).with_dtype<DType>().with_device(device_).verify(weight);
    TensorMatcher({-1, kRopeDim}).with_dtype<float>().with_device(device_).verify(freqs_cis);
    TensorMatcher({-1}).with_dtype<int64_t>().with_device(device_).verify(out_loc);
    TensorMatcher({-1, -1}).with_strides({kFp4PageBytes, 1}).with_dtype<uint8_t>().with_device(device_).verify(kvcache);

    switch (mode) {
      case CompressExtend:
        compress::verify_plan_c(plan, N, device_);
        RuntimeCheck(out_loc.size(0) >= N.unwrap());
        break;
      case CompressDecode:
        compress::verify_plan_d(plan, N, device_);
        RuntimeCheck(out_loc.size(0) == N.unwrap());
        break;
    }

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    if (num_tokens == 0) return;
    const auto params = FusedNormRopeStoreParams{
        .input = input.data_ptr(),
        .handle = plan.data_ptr(),
        .weight = weight.data_ptr(),
        .freqs_cis = static_cast<const float*>(freqs_cis.data_ptr()),
        .out_loc = static_cast<const int64_t*>(out_loc.data_ptr()),
        .kvcache = static_cast<uint8_t*>(kvcache.data_ptr()),
        .eps = eps,
        .compress_ratio = compress_ratio,
        .num_tokens = num_tokens,
    };
    const uint32_t num_blocks = div_ceil(num_tokens, kNumWarps);
    const auto device = device_.unwrap();
    const auto kernel =
        mode == CompressExtend ? select_fp4_kernel<CompressExtend>() : select_fp4_kernel<CompressDecode>();
    LaunchKernel(num_blocks, kBlockSize, device).enable_pdl(kUsePDL)(kernel, params);
  }
};

}
