#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/compress_v2.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/container/tuple.h>

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace sglang {

using PlanD = device::compress::DecodePlan;
using PlanC = device::compress::CompressPlan;






struct Compress128OnlineDecodeParams {
  void* __restrict__ kv_score_buffer;
  const void* __restrict__ kv_score_input;
  void* __restrict__ kv_compressed_output;
  const void* __restrict__ score_bias;
  const PlanD* __restrict__ plan_d;
  uint32_t batch_size;
};

template <int64_t kHeadDim, typename BufferFloat, bool kUsePDL>
__global__ void flash_c128_online_decode_v2(const __grid_constant__ Compress128OnlineDecodeParams params) {
  using namespace device;
  constexpr uint32_t kVecSize = 4;
  constexpr uint32_t kBlockSize = kHeadDim / kVecSize;
  using Vec = AlignedVector<float, kVecSize>;
  const auto gmem = tile::Memory<Vec>::cta(kBlockSize);
  const auto batch_id = blockIdx.x;
  if (batch_id >= params.batch_size) return;





  PDLWaitPrimary<kUsePDL>();

  const auto plan = params.plan_d[batch_id];
  const auto pos_in_chunk = (plan.seq_len - 1) % 128;

  const auto kv_score_buffer = static_cast<BufferFloat*>(params.kv_score_buffer);
  const auto kv_score_input = static_cast<const float*>(params.kv_score_input);
  const auto kv_load_buf = kv_score_buffer + plan.read_page_0 * (kHeadDim * 3);
  const auto kv_store_buf = kv_score_buffer + plan.write_loc * (kHeadDim * 3);
  const auto kv_src = kv_score_input + batch_id * (kHeadDim * 2);


  const auto new_kv_vec = gmem.load(kv_src, 0);
  const auto new_score_raw_vec = gmem.load(kv_src, 1);
  const auto bias_vec = gmem.load(params.score_bias, pos_in_chunk);

  Vec out_kv_vec;
  Vec out_max_vec;
  Vec out_sum_vec;
  if (pos_in_chunk != 0) {

    Vec max_score_vec, sum_score_vec, old_kv_vec;
    if constexpr (std::is_same_v<BufferFloat, float>) {
      max_score_vec = gmem.load(kv_load_buf, 0);
      sum_score_vec = gmem.load(kv_load_buf, 1);
      old_kv_vec = gmem.load(kv_load_buf, 2);
    } else {
      using BufferVec = AlignedVector<BufferFloat, kVecSize>;
      const auto gmem_buffer = tile::Memory<BufferVec>::cta(kBlockSize);
      const auto max_score_tmp = gmem_buffer.load(kv_load_buf, 0);
      const auto sum_score_tmp = gmem_buffer.load(kv_load_buf, 1);
      const auto old_kv_tmp = gmem_buffer.load(kv_load_buf, 2);
#pragma unroll
      for (uint32_t i = 0; i < kVecSize; ++i) {
        max_score_vec[i] = cast<float>(max_score_tmp[i]);
        sum_score_vec[i] = cast<float>(sum_score_tmp[i]);
        old_kv_vec[i] = cast<float>(old_kv_tmp[i]);
      }
    }
#pragma unroll
    for (uint32_t i = 0; i < kVecSize; ++i) {
      const auto old_max = max_score_vec[i];
      const auto old_kv = old_kv_vec[i];
      const auto new_score = new_score_raw_vec[i] + bias_vec[i];
      const auto new_kv = new_kv_vec[i];
      const auto new_max = fmaxf(old_max, new_score);
      const auto old_sum = sum_score_vec[i] * expf(old_max - new_max);
      const auto new_exp = expf(new_score - new_max);
      const auto new_sum = old_sum + new_exp;
      out_kv_vec[i] = (old_kv * old_sum + new_kv * new_exp) / new_sum;
      out_max_vec[i] = new_max;
      out_sum_vec[i] = new_sum;
    }
  } else {

#pragma unroll
    for (uint32_t i = 0; i < kVecSize; ++i) {
      out_kv_vec[i] = new_kv_vec[i];
      out_max_vec[i] = new_score_raw_vec[i] + bias_vec[i];
      out_sum_vec[i] = 1.0f;
    }
  }

  if (pos_in_chunk == 127) {

    const auto kv_out = static_cast<float*>(params.kv_compressed_output) + batch_id * kHeadDim;
    gmem.store(kv_out, out_kv_vec);
  } else {
    if constexpr (std::is_same_v<BufferFloat, float>) {
      gmem.store(kv_store_buf, out_max_vec, 0);
      gmem.store(kv_store_buf, out_sum_vec, 1);
      gmem.store(kv_store_buf, out_kv_vec, 2);
    } else {
      using BufferVec = AlignedVector<BufferFloat, kVecSize>;
      const auto gmem_buffer = tile::Memory<BufferVec>::cta(kBlockSize);
      BufferVec out_max_tmp, out_sum_tmp, out_kv_tmp;
#pragma unroll
      for (uint32_t i = 0; i < kVecSize; ++i) {
        out_max_tmp[i] = cast<BufferFloat>(out_max_vec[i]);
        out_sum_tmp[i] = cast<BufferFloat>(out_sum_vec[i]);
        out_kv_tmp[i] = cast<BufferFloat>(out_kv_vec[i]);
      }
      gmem_buffer.store(kv_store_buf, out_max_tmp, 0);
      gmem_buffer.store(kv_store_buf, out_sum_tmp, 1);
      gmem_buffer.store(kv_store_buf, out_kv_tmp, 2);
    }
  }
}







constexpr int32_t kTileElements = 2;
constexpr int32_t kElementsPerWarp = 8;
constexpr uint32_t kNumWarps = 128 / kElementsPerWarp;
constexpr uint32_t kPrefillBlockSize = device::kWarpThreads * kNumWarps;
using PrefillStorage = device::AlignedVector<float, kTileElements>;

struct Compress128OnlinePrefillParams {
  void* __restrict__ kv_score_buffer;
  const void* __restrict__ kv_score_input;
  void* __restrict__ kv_compressed_output;
  const void* __restrict__ score_bias;
  const PlanC* __restrict__ plan_c;
  const PlanC* __restrict__ plan_w;
  uint32_t num_compress;
  uint32_t num_write;
};

struct Compress128SharedBuffer {
  using Storage = device::AlignedVector<float, 4>;
  Storage data[kNumWarps][device::kWarpThreads + 1];
  SGL_DEVICE Storage& operator()(uint32_t warp_id, uint32_t lane_id) {
    return data[warp_id][lane_id];
  }
  SGL_DEVICE float& operator()(uint32_t warp_id, uint32_t lane_id, uint32_t tile_id) {
    return data[warp_id][lane_id][tile_id];
  }
};


constexpr float kPadScore = -FLT_MAX;

[[maybe_unused]] SGL_DEVICE void c128_prefill_segment_softmax(
    const PrefillStorage (&kv)[kElementsPerWarp],
    const PrefillStorage (&score)[kElementsPerWarp],
    float* seg_kv,
    float* seg_max,
    float* seg_sum,
    const uint32_t warp_id,
    const uint32_t lane_id) {
  using namespace device;


  using TmpStorage = typename Compress128SharedBuffer::Storage;
  __shared__ Compress128SharedBuffer s_local_val_max;
  __shared__ Compress128SharedBuffer s_local_exp_sum;
  __shared__ Compress128SharedBuffer s_local_product;

  TmpStorage tmp_val_max;
  TmpStorage tmp_exp_sum;
  TmpStorage tmp_product;

#pragma unroll
  for (int32_t i = 0; i < kTileElements; ++i) {
    float score_fp32[kElementsPerWarp];
#pragma unroll
    for (int32_t j = 0; j < kElementsPerWarp; ++j) {
      score_fp32[j] = score[j][i];
    }
    float max_value = score_fp32[0];
#pragma unroll
    for (int32_t j = 1; j < kElementsPerWarp; ++j) {
      max_value = fmaxf(max_value, score_fp32[j]);
    }
    float sum_exp_value = 0.0f;
    float sum_product = 0.0f;
#pragma unroll
    for (int32_t j = 0; j < kElementsPerWarp; ++j) {
      const auto exp_score = expf(score_fp32[j] - max_value);
      sum_product += kv[j][i] * exp_score;
      sum_exp_value += exp_score;
    }
    tmp_val_max[i] = max_value;
    tmp_exp_sum[i] = sum_exp_value;
    tmp_product[i] = sum_product;
  }


  s_local_val_max(warp_id, lane_id) = tmp_val_max;
  s_local_exp_sum(warp_id, lane_id) = tmp_exp_sum;
  s_local_product(warp_id, lane_id) = tmp_product;

  __syncthreads();



  constexpr uint32_t kReductionCount = kTileElements * kWarpThreads * kNumWarps;
  constexpr uint32_t kIteration = kReductionCount / kPrefillBlockSize;
  static_assert(kTileElements * kNumWarps == kWarpThreads, "TODO: support other configs");

#pragma unroll
  for (uint32_t i = 0; i < kIteration; ++i) {
    const uint32_t j = i * kPrefillBlockSize + warp_id * kWarpThreads + lane_id;
    const uint32_t local_warp_id = j % kNumWarps;
    const uint32_t local_elem_id = j / kNumWarps;
    const uint32_t local_tile_id = local_elem_id % kTileElements;
    const uint32_t local_lane_id = local_elem_id / kTileElements;
    const auto local_val_max = s_local_val_max(local_warp_id, local_lane_id, local_tile_id);
    const auto local_exp_sum = s_local_exp_sum(local_warp_id, local_lane_id, local_tile_id);
    const auto local_product = s_local_product(local_warp_id, local_lane_id, local_tile_id);
    const auto global_val_max = warp::reduce_max<kNumWarps>(local_val_max);
    const auto rescale = expf(local_val_max - global_val_max);
    const auto global_exp_sum = warp::reduce_sum<kNumWarps>(local_exp_sum * rescale);
    const auto final_scale = rescale / global_exp_sum;
    const auto global_product = warp::reduce_sum<kNumWarps>(local_product * final_scale);
    seg_kv[local_elem_id] = global_product;
    seg_max[local_elem_id] = global_val_max;
    seg_sum[local_elem_id] = global_exp_sum;
  }
  __syncthreads();
}









template <int64_t kHeadDim, typename BufferFloat, bool kWrite, bool kUsePDL>
__global__ __launch_bounds__(kPrefillBlockSize, 2)
    void flash_c128_online_prefill_v2(const __grid_constant__ Compress128OnlinePrefillParams params) {
  using namespace device;

  constexpr int64_t kTileDim = kTileElements * kWarpThreads;
  constexpr uint32_t kNumSplit = kHeadDim / kTileDim;
  static_assert(kHeadDim % kTileDim == 0);


  const auto num_plans = kWrite ? params.num_write : params.num_compress;
  const auto plan_ptr = kWrite ? params.plan_w : params.plan_c;
  const uint32_t global_id = blockIdx.x;
  const uint32_t global_pid = global_id / kNumSplit;
  const uint32_t global_sid = global_id % kNumSplit;
  if (global_pid >= num_plans) return;

  const uint32_t warp_id = threadIdx.x / kWarpThreads;
  const uint32_t lane_id = threadIdx.x % kWarpThreads;
  const int32_t split_offset = global_sid * kTileDim;







  PDLWaitPrimary<kUsePDL>();

  const auto plan = plan_ptr[global_pid];
  if (plan.is_invalid()) [[unlikely]]
    return;

  const auto kv_score_buffer = static_cast<BufferFloat*>(params.kv_score_buffer);
  const auto kv_score_input = static_cast<const float*>(params.kv_score_input);
  const auto kv_compressed_output = static_cast<float*>(params.kv_compressed_output);
  const auto score_bias_base = static_cast<const float*>(params.score_bias);

  constexpr int64_t kElementSize = kHeadDim * 2;



  const uint32_t window_len = plan.buffer_len;
  const uint32_t position = plan.seq_len - 1;
  const uint32_t pos_in_chunk_end = (position % 128u) + 1u;
  const uint32_t chunk_offset = pos_in_chunk_end - window_len;
  const int32_t chunk_start_ragged = static_cast<int32_t>(plan.ragged_id) - static_cast<int32_t>(pos_in_chunk_end) + 1;


  PrefillStorage kv[kElementsPerWarp];
  PrefillStorage score[kElementsPerWarp];
  PrefillStorage bias[kElementsPerWarp];
  const uint32_t warp_offset = warp_id * kElementsPerWarp;

#pragma unroll
  for (uint32_t i = 0; i < kElementsPerWarp; ++i) {
    const uint32_t j = i + warp_offset;
    if (j >= chunk_offset && j < pos_in_chunk_end) {
      const int32_t ragged_id = chunk_start_ragged + static_cast<int32_t>(j);
      const auto kv_src_ptr = kv_score_input + ragged_id * kElementSize + split_offset;
      const auto score_src_ptr = kv_src_ptr + kHeadDim;
      const auto bias_src_ptr = score_bias_base + j * kHeadDim + split_offset;
      kv[i].load(kv_src_ptr, lane_id);
      score[i].load(score_src_ptr, lane_id);
      bias[i].load(bias_src_ptr, lane_id);
    }
  }



#pragma unroll
  for (uint32_t i = 0; i < kElementsPerWarp; ++i) {
    const uint32_t j = i + warp_offset;
    const bool is_valid = (j >= chunk_offset && j < pos_in_chunk_end);
#pragma unroll
    for (uint32_t ii = 0; ii < kTileElements; ++ii) {
      score[i][ii] = is_valid ? score[i][ii] + bias[i][ii] : kPadScore;
      kv[i][ii] = is_valid ? kv[i][ii] : 0.0f;
    }
  }


  __shared__ alignas(16) float seg_kv[kTileDim];
  __shared__ alignas(16) float seg_max[kTileDim];
  __shared__ alignas(16) float seg_sum[kTileDim];
  c128_prefill_segment_softmax(kv, score, seg_kv, seg_max, seg_sum, warp_id, lane_id);

  PDLTriggerSecondary<kUsePDL>();


  if (warp_id == 0) {
    PrefillStorage out_kv_vec, out_max_vec, out_sum_vec;
    out_kv_vec.load(seg_kv, lane_id);
    out_max_vec.load(seg_max, lane_id);
    out_sum_vec.load(seg_sum, lane_id);

    const int32_t read_page = plan.read_page_1 >= 0 ? plan.read_page_1 : plan.read_page_0;
    if (chunk_offset != 0 && read_page >= 0) {

      const auto buf_load = kv_score_buffer + read_page * (kHeadDim * 3) + split_offset;
      PrefillStorage buf_max_vec, buf_sum_vec, buf_kv_vec;
      if constexpr (std::is_same_v<BufferFloat, float>) {
        buf_max_vec.load(buf_load + 0 * kHeadDim, lane_id);
        buf_sum_vec.load(buf_load + 1 * kHeadDim, lane_id);
        buf_kv_vec.load(buf_load + 2 * kHeadDim, lane_id);
      } else {
        using BufferPrefillStorage = AlignedVector<BufferFloat, kTileElements>;
        BufferPrefillStorage buf_max_tmp, buf_sum_tmp, buf_kv_tmp;
        buf_max_tmp.load(buf_load + 0 * kHeadDim, lane_id);
        buf_sum_tmp.load(buf_load + 1 * kHeadDim, lane_id);
        buf_kv_tmp.load(buf_load + 2 * kHeadDim, lane_id);
#pragma unroll
        for (uint32_t ii = 0; ii < kTileElements; ++ii) {
          buf_max_vec[ii] = cast<float>(buf_max_tmp[ii]);
          buf_sum_vec[ii] = cast<float>(buf_sum_tmp[ii]);
          buf_kv_vec[ii] = cast<float>(buf_kv_tmp[ii]);
        }
      }
#pragma unroll
      for (uint32_t ii = 0; ii < kTileElements; ++ii) {
        const float m1 = buf_max_vec[ii];
        const float s1 = buf_sum_vec[ii];
        const float k1 = buf_kv_vec[ii];
        const float m2 = out_max_vec[ii];
        const float s2 = out_sum_vec[ii];
        const float k2 = out_kv_vec[ii];
        const float new_max = fmaxf(m1, m2);
        const float new_s1 = s1 * expf(m1 - new_max);
        const float new_s2 = s2 * expf(m2 - new_max);
        const float new_sum = new_s1 + new_s2;
        const float new_kv = (k1 * new_s1 + k2 * new_s2) / new_sum;
        out_max_vec[ii] = new_max;
        out_sum_vec[ii] = new_sum;
        out_kv_vec[ii] = new_kv;
      }
    }

    if constexpr (kWrite) {



      const auto buf_store = kv_score_buffer + plan.read_page_0 * (kHeadDim * 3) + split_offset;
      if constexpr (std::is_same_v<BufferFloat, float>) {
        reinterpret_cast<PrefillStorage*>(buf_store + 0 * kHeadDim)[lane_id] = out_max_vec;
        reinterpret_cast<PrefillStorage*>(buf_store + 1 * kHeadDim)[lane_id] = out_sum_vec;
        reinterpret_cast<PrefillStorage*>(buf_store + 2 * kHeadDim)[lane_id] = out_kv_vec;
      } else {
        using BufferPrefillStorage = AlignedVector<BufferFloat, kTileElements>;
        BufferPrefillStorage out_max_tmp, out_sum_tmp, out_kv_tmp;
#pragma unroll
        for (uint32_t ii = 0; ii < kTileElements; ++ii) {
          out_max_tmp[ii] = cast<BufferFloat>(out_max_vec[ii]);
          out_sum_tmp[ii] = cast<BufferFloat>(out_sum_vec[ii]);
          out_kv_tmp[ii] = cast<BufferFloat>(out_kv_vec[ii]);
        }
        reinterpret_cast<BufferPrefillStorage*>(buf_store + 0 * kHeadDim)[lane_id] = out_max_tmp;
        reinterpret_cast<BufferPrefillStorage*>(buf_store + 1 * kHeadDim)[lane_id] = out_sum_tmp;
        reinterpret_cast<BufferPrefillStorage*>(buf_store + 2 * kHeadDim)[lane_id] = out_kv_tmp;
      }
    } else {

      const auto out_ptr = kv_compressed_output + global_pid * kHeadDim + split_offset;
      reinterpret_cast<PrefillStorage*>(out_ptr)[lane_id] = out_kv_vec;
    }
  }
}








template <int64_t kHeadDim, typename BufferFloat, bool kUsePDL>
struct FlashCompress128OnlineKernel {
  static constexpr auto decode_kernel = flash_c128_online_decode_v2<kHeadDim, BufferFloat, kUsePDL>;
  template <bool kWrite>
  static constexpr auto prefill_kernel = flash_c128_online_prefill_v2<kHeadDim, BufferFloat, kWrite, kUsePDL>;
  static constexpr int64_t kTileDim = kTileElements * device::kWarpThreads;
  static constexpr uint32_t kNumSplit = kHeadDim / kTileDim;
  static constexpr uint32_t kDecodeBlockSize = kHeadDim / 4;

  static void run_decode(
      const tvm::ffi::TensorView kv_score_buffer,
      const tvm::ffi::TensorView kv_score_input,
      const tvm::ffi::TensorView kv_compressed_output,
      const tvm::ffi::TensorView ape,
      const tvm::ffi::TensorView plan_d_) {
    using namespace host;

    auto B = SymbolicSize{"batch_size"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({-1, 1, kHeadDim * 3})
        .with_dtype<BufferFloat>()
        .with_device(device_)
        .verify(kv_score_buffer);
    TensorMatcher({B, kHeadDim * 2})
        .with_dtype<float>()
        .with_device(device_)
        .verify(kv_score_input);
    TensorMatcher({B, kHeadDim})
        .with_dtype<float>()
        .with_device(device_)
        .verify(kv_compressed_output);
    TensorMatcher({128, kHeadDim})
        .with_dtype<float>()
        .with_device(device_)
        .verify(ape);

    const auto plan_d = compress::verify_plan_d(plan_d_, B, device_);
    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    if (batch_size == 0) return;
    const auto params = Compress128OnlineDecodeParams{
        .kv_score_buffer = kv_score_buffer.data_ptr(),
        .kv_score_input = kv_score_input.data_ptr(),
        .kv_compressed_output = kv_compressed_output.data_ptr(),
        .score_bias = ape.data_ptr(),
        .plan_d = plan_d,
        .batch_size = batch_size,
    };
    LaunchKernel(batch_size, kDecodeBlockSize, device_.unwrap())
        .enable_pdl(kUsePDL)(decode_kernel, params);
  }

  static void run_prefill(
      const tvm::ffi::TensorView kv_score_buffer,
      const tvm::ffi::TensorView kv_score_input,
      const tvm::ffi::TensorView kv_compressed_output,
      const tvm::ffi::TensorView ape,
      const tvm::ffi::TensorView plan_c_,
      const tvm::ffi::TensorView plan_w_) {
    using namespace host;

    auto N = SymbolicSize{"num_q_tokens"};
    auto C = SymbolicSize{"num_c_plans"};
    auto W = SymbolicSize{"num_w_plans"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({-1, 1, kHeadDim * 3})
        .with_dtype<BufferFloat>()
        .with_device(device_)
        .verify(kv_score_buffer);
    TensorMatcher({N, kHeadDim * 2})
        .with_dtype<float>()
        .with_device(device_)
        .verify(kv_score_input);
    TensorMatcher({C, kHeadDim})
        .with_dtype<float>()
        .with_device(device_)
        .verify(kv_compressed_output);
    TensorMatcher({128, kHeadDim})
        .with_dtype<float>()
        .with_device(device_)
        .verify(ape);



    const auto plan_c = compress::verify_plan_c(plan_c_, C, device_);
    const auto plan_w = compress::verify_plan_c(plan_w_, W, device_);
    const auto device = device_.unwrap();
    const auto num_q_tokens = static_cast<uint32_t>(N.unwrap());
    const auto num_c = static_cast<uint32_t>(C.unwrap());
    const auto num_w = static_cast<uint32_t>(W.unwrap());
    RuntimeCheck(num_q_tokens >= num_w, "invalid prefill plan: num_q < num_w");
    const auto params = Compress128OnlinePrefillParams{
        .kv_score_buffer = kv_score_buffer.data_ptr(),
        .kv_score_input = kv_score_input.data_ptr(),
        .kv_compressed_output = kv_compressed_output.data_ptr(),
        .score_bias = ape.data_ptr(),
        .plan_c = plan_c,
        .plan_w = plan_w,
        .num_compress = num_c,
        .num_write = num_w,
    };



    if (const auto num_c_blocks = num_c * kNumSplit) {
      LaunchKernel(num_c_blocks, kPrefillBlockSize, device)
          .enable_pdl(kUsePDL)(prefill_kernel<           false>, params);
    }
    if (const auto num_w_blocks = num_w * kNumSplit) {
      LaunchKernel(num_w_blocks, kPrefillBlockSize, device)
          .enable_pdl(kUsePDL)(prefill_kernel<           true>, params);
    }
  }
};
















namespace host::compress {

using device::compress::CompressPlan;
using device::compress::DecodePlan;





struct OnlineDecodePlanParams {
  DecodePlan* __restrict__ plan_d;
  const int64_t* __restrict__ seq_lens;
  const int64_t* __restrict__ req_pool_indices;
  const int32_t* __restrict__ req_to_token;
  int64_t stride_r2t;
  int32_t state_slot_offset;
  uint32_t batch_size;
};

__global__ void plan_c128_online_decode_kernel(const OnlineDecodePlanParams params) {
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= params.batch_size) return;
  const auto seq_len = static_cast<uint32_t>(params.seq_lens[idx]);
  const auto rid = params.req_pool_indices[idx];
  const int32_t slot = static_cast<int32_t>(rid) + params.state_slot_offset;
  params.plan_d[idx] = DecodePlan{
      .seq_len = seq_len,
      .write_loc = slot,
      .read_page_0 = slot,
      .read_page_1 = -1,
  };
}






inline void plan_online_decode(
    const tvm::ffi::TensorView seq_lens,
    const tvm::ffi::TensorView req_pool_indices,
    const tvm::ffi::TensorView req_to_token,
    const tvm::ffi::TensorView plan_d_dev_,
    const int32_t state_slot_offset) {
  auto B = SymbolicSize{"batch_size"};
  auto device_ = SymbolicDevice{};
  device_.set_options<kDLCUDA>();

  auto seq_dtype = SymbolicDType{};
  TensorMatcher({B})
      .with_dtype<int64_t>(seq_dtype)
      .with_device(device_)
      .verify(seq_lens);
  TensorMatcher({B})
      .with_dtype<int64_t>()
      .with_device(device_)
      .verify(req_pool_indices);
  TensorMatcher({-1, -1})
      .with_dtype<int32_t>()
      .with_device(device_)
      .verify(req_to_token);
  TensorMatcher({B, sizeof(DecodePlan)})
      .with_dtype<uint8_t>()
      .with_device(device_)
      .verify(plan_d_dev_);
  RuntimeCheck(state_slot_offset >= 0);

  const auto batch_size = static_cast<uint32_t>(B.unwrap());
  if (batch_size == 0) return;

  const auto device = device_.unwrap();
  constexpr uint32_t kBlockSize = 256;
  const uint32_t num_blocks = host::div_ceil(batch_size, kBlockSize);
  const auto stride_r2t = req_to_token.stride(0);
  const auto params = OnlineDecodePlanParams{
      .plan_d = static_cast<DecodePlan*>(plan_d_dev_.data_ptr()),
      .seq_lens = static_cast<const int64_t*>(seq_lens.data_ptr()),
      .req_pool_indices = static_cast<const int64_t*>(req_pool_indices.data_ptr()),
      .req_to_token = static_cast<const int32_t*>(req_to_token.data_ptr()),
      .stride_r2t = stride_r2t,
      .state_slot_offset = state_slot_offset,
      .batch_size = batch_size,
  };
  LaunchKernel(num_blocks, kBlockSize, device)(plan_c128_online_decode_kernel, params);
}





struct OnlinePrefillStage0Params {
  CompressPlan* __restrict__ plan_c;
  CompressPlan* __restrict__ plan_w;
  const int64_t* __restrict__ seq_lens;
  const int64_t* __restrict__ extend_lens;
  uint32_t batch_size;
  uint32_t num_q_tokens;
};

inline std::tuple<uint32_t, uint32_t> _plan_prefill_partial(const OnlinePrefillStage0Params& p) {
  uint32_t counter = 0;
  uint32_t compress_count = 0;
  uint32_t write_count = 0;
  for (const auto i : irange(p.batch_size)) {
    const uint32_t seq_len = static_cast<uint32_t>(p.seq_lens[i]);
    const uint32_t extend_len = static_cast<uint32_t>(p.extend_lens[i]);
    RuntimeCheck(0 < extend_len && extend_len <= seq_len);
    const uint32_t prefix_len = seq_len - extend_len;
    const uint32_t end_pos = prefix_len + extend_len;

    uint32_t pos = prefix_len;
    while (pos < end_pos) {
      const uint32_t chunk_start = (pos / 128u) * 128u;
      const uint32_t seg_end = std::min(end_pos, chunk_start + 128u);
      const uint32_t seg_len = seg_end - pos;
      const uint32_t chunk_off = pos - chunk_start;
      const uint32_t last_pos = seg_end - 1;
      const uint32_t last_ragged = counter + (last_pos - prefix_len);
      RuntimeCheck(last_ragged < (1u << 16), "PlanC.ragged_id is uint16; ragged ", last_ragged, " overflows");
      RuntimeCheck(seg_len <= 128u);



      const auto plan = CompressPlan{
          .seq_len = last_pos + 1u,
          .ragged_id = static_cast<uint16_t>(last_ragged),
          .buffer_len = static_cast<uint16_t>(seg_len),
          .read_page_0 = static_cast<int32_t>(i),
          .read_page_1 = -1,
      };
      if (chunk_off + seg_len == 128u) {

        RuntimeCheck(compress_count < p.num_q_tokens);
        p.plan_c[compress_count++] = plan;
      } else {

        RuntimeCheck(write_count < p.num_q_tokens);
        p.plan_w[write_count++] = plan;
      }
      pos = seg_end;
    }
    counter += extend_len;
  }
  RuntimeCheck(counter == p.num_q_tokens, "input size ", counter, " != num_q_tokens ", p.num_q_tokens);
  return std::tuple<uint32_t, uint32_t>{compress_count, write_count};
}

struct OnlinePrefillStage1Params {
  CompressPlan* __restrict__ plan_c;
  CompressPlan* __restrict__ plan_w;
  const int64_t* __restrict__ req_pool_indices;
  const int32_t* __restrict__ req_to_token;
  int64_t stride_r2t;
  int32_t state_slot_offset;
  uint32_t num_c;
  uint32_t num_w;
};

__global__ void plan_c128_online_prefill_kernel(const OnlinePrefillStage1Params params) {
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t total = params.num_c + params.num_w;
  if (idx >= total) return;

  const bool is_compress = idx < params.num_c;
  CompressPlan* const plan_ptr = is_compress ? &params.plan_c[idx] : &params.plan_w[idx - params.num_c];
  auto plan = *plan_ptr;
  if (plan.is_invalid()) return;
  const auto batch_id = plan.read_page_0;
  const auto rid = params.req_pool_indices[batch_id];
  const int32_t main_slot = static_cast<int32_t>(rid);
  plan.read_page_0 = main_slot + params.state_slot_offset;
  plan.read_page_1 = main_slot;
  *plan_ptr = plan;
}

using OnlinePrefillPlan = tvm::ffi::Tuple<uint32_t, uint32_t>;

inline OnlinePrefillPlan plan_online_prefill(
    const tvm::ffi::TensorView seq_lens,
    const tvm::ffi::TensorView extend_lens,
    const tvm::ffi::TensorView req_pool_indices,
    const tvm::ffi::TensorView req_to_token,
    const tvm::ffi::TensorView plan_c_pin,
    const tvm::ffi::TensorView plan_w_pin,
    const tvm::ffi::TensorView plan_c_dev_,
    const tvm::ffi::TensorView plan_w_dev_,
    const int32_t state_slot_offset,
    const bool use_cuda_graph) {
  auto B = SymbolicSize{"batch_size"};
  auto N = SymbolicSize{"num_q_tokens"};
  auto cpu = SymbolicDevice{};
  auto device_ = SymbolicDevice{};
  cpu.set_options<kDLCPU, kDLCUDAHost>();
  device_.set_options<kDLCUDA>();

  TensorMatcher({B})
      .with_dtype<int64_t>()
      .with_device(cpu)
      .verify(seq_lens)
      .verify(extend_lens);
  TensorMatcher({B})
      .with_dtype<int64_t>()
      .with_device(device_)
      .verify(req_pool_indices);
  TensorMatcher({-1, -1})
      .with_dtype<int32_t>()
      .with_device(device_)
      .verify(req_to_token);
  TensorMatcher({N, sizeof(CompressPlan)})
      .with_dtype<uint8_t>()
      .with_device(cpu)
      .verify(plan_c_pin)
      .verify(plan_w_pin);
  TensorMatcher({N, sizeof(CompressPlan)})
      .with_dtype<uint8_t>()
      .with_device(device_)
      .verify(plan_c_dev_)
      .verify(plan_w_dev_);
  RuntimeCheck(state_slot_offset >= 0);

  const auto stage0_params = OnlinePrefillStage0Params{
      .plan_c = static_cast<CompressPlan*>(plan_c_pin.data_ptr()),
      .plan_w = static_cast<CompressPlan*>(plan_w_pin.data_ptr()),
      .seq_lens = static_cast<const int64_t*>(seq_lens.data_ptr()),
      .extend_lens = static_cast<const int64_t*>(extend_lens.data_ptr()),
      .batch_size = static_cast<uint32_t>(B.unwrap()),
      .num_q_tokens = static_cast<uint32_t>(N.unwrap()),
  };






  static const bool kGuard = []() {
    const char* v = std::getenv("SGLANG_DEBUG_C128_ONLINE_GUARD");
    return v != nullptr && v[0] == '1';
  }();
  if (kGuard) {
    RuntimeCheck(stage0_params.batch_size <= 65536u, "batch_size out of bound: ", stage0_params.batch_size);
    RuntimeCheck(stage0_params.num_q_tokens <= 65536u, "num_q_tokens out of bound: ", stage0_params.num_q_tokens);




    auto* pc = static_cast<uint8_t*>(plan_c_pin.data_ptr());
    auto* pw = static_cast<uint8_t*>(plan_w_pin.data_ptr());
    const auto bytes = static_cast<size_t>(N.unwrap()) * sizeof(CompressPlan);
    std::memset(pc, 0xAB, bytes);
    std::memset(pw, 0xAB, bytes);
  }

  const auto [num_c, num_w] = _plan_prefill_partial(stage0_params);
  const auto num_c_padded = use_cuda_graph ? static_cast<uint32_t>(N.unwrap()) : num_c;
  const auto num_w_padded = use_cuda_graph ? static_cast<uint32_t>(N.unwrap()) : num_w;

  if (kGuard) {

    auto* pc = static_cast<const uint8_t*>(plan_c_pin.data_ptr());
    auto* pw = static_cast<const uint8_t*>(plan_w_pin.data_ptr());
    const auto end_c = static_cast<size_t>(num_c) * sizeof(CompressPlan);
    const auto end_w = static_cast<size_t>(num_w) * sizeof(CompressPlan);
    const auto pin_bytes = static_cast<size_t>(N.unwrap()) * sizeof(CompressPlan);
    for (size_t k = end_c; k < pin_bytes; ++k) {
      RuntimeCheck(
          pc[k] == 0xAB,
          "GUARD: plan_c_pin OOB write at byte ",
          k,
          " (num_c=",
          num_c,
          ", num_q_tokens=",
          N.unwrap(),
          ")");
    }
    for (size_t k = end_w; k < pin_bytes; ++k) {
      RuntimeCheck(
          pw[k] == 0xAB,
          "GUARD: plan_w_pin OOB write at byte ",
          k,
          " (num_w=",
          num_w,
          ", num_q_tokens=",
          N.unwrap(),
          ")");
    }
  }

  const auto device = device_.unwrap();

  auto* const plan_c_dev_ptr = static_cast<CompressPlan*>(plan_c_dev_.data_ptr());
  auto* const plan_w_dev_ptr = static_cast<CompressPlan*>(plan_w_dev_.data_ptr());

  if (use_cuda_graph) {
    const auto kInvalidPlan = CompressPlan::invalid();
    auto* const plan_c_pin_ptr = static_cast<CompressPlan*>(plan_c_pin.data_ptr());
    auto* const plan_w_pin_ptr = static_cast<CompressPlan*>(plan_w_pin.data_ptr());
    for (const auto i : irange(num_c, num_c_padded)) {
      plan_c_pin_ptr[i] = kInvalidPlan;
    }
    for (const auto i : irange(num_w, num_w_padded)) {
      plan_w_pin_ptr[i] = kInvalidPlan;
    }
  }

  if (const auto total = num_c_padded + num_w_padded) {
    const auto stream = LaunchKernel::resolve_device(device);

    static const bool kSyncH2D = []() {
      const char* v = std::getenv("SGLANG_DEBUG_C128_ONLINE_SYNC_H2D");
      return v != nullptr && v[0] == '1';
    }();

    static const bool kNoH2D = []() {
      const char* v = std::getenv("SGLANG_DEBUG_C128_ONLINE_NO_H2D");
      return v != nullptr && v[0] == '1';
    }();
    const auto copy_to_device = [stream](void* dst, void* src, int64_t count) {
      if (kNoH2D) return;
      const auto bytes = count * sizeof(CompressPlan);
      if (kSyncH2D) {
        RuntimeDeviceCheck(::cudaMemcpy(dst, src, bytes, ::cudaMemcpyHostToDevice));
      } else {
        RuntimeDeviceCheck(::cudaMemcpyAsync(dst, src, bytes, ::cudaMemcpyHostToDevice, stream));
      }
    };
    if (num_c_padded) copy_to_device(plan_c_dev_ptr, plan_c_pin.data_ptr(), num_c_padded);
    if (num_w_padded) copy_to_device(plan_w_dev_ptr, plan_w_pin.data_ptr(), num_w_padded);

    const auto stage1_params = OnlinePrefillStage1Params{
        .plan_c = plan_c_dev_ptr,
        .plan_w = plan_w_dev_ptr,
        .req_pool_indices = static_cast<const int64_t*>(req_pool_indices.data_ptr()),
        .req_to_token = static_cast<const int32_t*>(req_to_token.data_ptr()),
        .stride_r2t = req_to_token.stride(0),
        .state_slot_offset = state_slot_offset,
        .num_c = num_c_padded,
        .num_w = num_w_padded,
    };
    constexpr uint32_t kBlockSize = 128;
    const auto num_blocks = host::div_ceil(total, kBlockSize);
    LaunchKernel(num_blocks, kBlockSize, device)(plan_c128_online_prefill_kernel, stage1_params);
  }
  return OnlinePrefillPlan{num_c_padded, num_w_padded};
}

}

[[maybe_unused]] constexpr auto& plan_compress_128_online_decode = host::compress::plan_online_decode;
[[maybe_unused]] constexpr auto& plan_compress_128_online_prefill = host::compress::plan_online_prefill;

}
