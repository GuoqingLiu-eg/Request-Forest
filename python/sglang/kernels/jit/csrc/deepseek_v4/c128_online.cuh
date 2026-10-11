#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/compress.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/container/tuple.h>
#include <tvm/ffi/object.h>

#include <algorithm>
#include <cfloat>
#include <cstdint>

namespace sglang {

namespace device::compress {
















struct alignas(16) OnlinePrefillPlan {


  uint32_t ragged_id;

  uint32_t batch_id;



  uint32_t position;



  uint32_t window_len;
};

static_assert(alignof(OnlinePrefillPlan) == alignof(PrefillPlan));
static_assert(sizeof(OnlinePrefillPlan) == sizeof(PrefillPlan));

}

namespace host::compress {

using device::compress::OnlinePrefillPlan;
using OnlinePrefillPlanTensorDtype = uint8_t;
inline constexpr int64_t kOnlinePrefillPlanDim = 16;

static_assert(alignof(OnlinePrefillPlan) == sizeof(OnlinePrefillPlan));
static_assert(sizeof(OnlinePrefillPlan) == kOnlinePrefillPlanDim * sizeof(OnlinePrefillPlanTensorDtype));

}

using OnlinePlan = device::compress::OnlinePrefillPlan;
using IndiceT = int32_t;


struct Compress128OnlineDecodeParams {

  void* __restrict__ kv_score_buffer;

  const void* __restrict__ kv_score_input;

  void* __restrict__ kv_compressed_output;

  const void* __restrict__ score_bias;

  const IndiceT* __restrict__ indices;

  const IndiceT* __restrict__ seq_lens;

  uint32_t batch_size;
};


struct Compress128OnlinePrefillParams {

  void* __restrict__ kv_score_buffer;

  const void* __restrict__ kv_score_input;

  void* __restrict__ kv_compressed_output;

  const void* __restrict__ score_bias;

  const IndiceT* __restrict__ indices;

  const IndiceT* __restrict__ load_indices;


  const OnlinePlan* __restrict__ compress_plan;


  const OnlinePlan* __restrict__ write_plan;
  uint32_t num_compress;
  uint32_t num_write;
};


template <int64_t kHeadDim, bool kUsePDL>
__global__ void flash_c128_online_decode(const __grid_constant__ Compress128OnlineDecodeParams params) {
  using namespace device;
  constexpr uint32_t kVecSize = 4;
  constexpr uint32_t kBlockSize = kHeadDim / kVecSize;
  using Vec = AlignedVector<float, kVecSize>;
  const auto gmem = tile::Memory<Vec>::cta(kBlockSize);
  const auto batch_id = blockIdx.x;
  const auto index = params.indices[batch_id];
  const auto seq_len = params.seq_lens[batch_id];

  const auto kv_score_buffer = static_cast<float*>(params.kv_score_buffer);
  const auto kv_buf = kv_score_buffer + index * (kHeadDim * 3);
  const auto kv_score_input = static_cast<const float*>(params.kv_score_input);
  const auto kv_src = kv_score_input + batch_id * (kHeadDim * 2);



  const auto max_score_vec = gmem.load(kv_buf, 0);
  const auto sum_score_vec = gmem.load(kv_buf, 1);
  const auto old_kv_vec = gmem.load(kv_buf, 2);



  const auto new_kv_vec = gmem.load(kv_src, 0);
  const auto new_score_raw_vec = gmem.load(kv_src, 1);




  const auto pos_in_chunk = (seq_len - 1) % 128;
  const auto bias_vec = gmem.load(params.score_bias, pos_in_chunk);

  Vec out_kv_vec;
  Vec out_max_vec;
  Vec out_sum_vec;
  if (pos_in_chunk != 0) {

#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      const auto old_max = max_score_vec[i];
      const auto old_kv = old_kv_vec[i];
      const auto new_score = new_score_raw_vec[i] + bias_vec[i];
      const auto new_kv = new_kv_vec[i];
      const auto new_max = fmax(old_max, new_score);
      const auto old_sum = sum_score_vec[i] * expf(old_max - new_max);
      const auto new_exp = expf(new_score - new_max);
      const auto new_sum = old_sum + new_exp;
      out_kv_vec[i] = (old_kv * old_sum + new_kv * new_exp) / new_sum;
      out_max_vec[i] = new_max;
      out_sum_vec[i] = new_sum;
    }
  } else {

#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      out_kv_vec[i] = new_kv_vec[i];
      out_max_vec[i] = new_score_raw_vec[i] + bias_vec[i];
      out_sum_vec[i] = 1.0f;
    }
  }

  if (pos_in_chunk == 127) {


    const auto kv_out = static_cast<float*>(params.kv_compressed_output) + batch_id * kHeadDim;
    gmem.store(kv_out, out_kv_vec);
  } else {

    gmem.store(kv_buf, out_max_vec, 0);
    gmem.store(kv_buf, out_sum_vec, 1);
    gmem.store(kv_buf, out_kv_vec, 2);
  }
}

constexpr int32_t kTileElements = 2;

constexpr int32_t kElementsPerWarp = 8;
constexpr uint32_t kNumWarps = 128 / kElementsPerWarp;
constexpr uint32_t kPrefillBlockSize = device::kWarpThreads * kNumWarps;
using PrefillStorage = device::AlignedVector<float, kTileElements>;

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

template <bool kNeedData>
SGL_DEVICE void c128_prefill_forward(
    const PrefillStorage (&kv)[kElementsPerWarp],
    const PrefillStorage (&score)[kElementsPerWarp],
    float* kv_out,
    float* max_out,
    float* sum_out,
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
    float sum_exp_value = 0.0f;

#pragma unroll
    for (int32_t j = 1; j < kElementsPerWarp; ++j) {
      const auto fp32_score = score_fp32[j];
      max_value = fmaxf(max_value, fp32_score);
    }

    float sum_product = 0.0f;
#pragma unroll
    for (int32_t j = 0; j < 8; ++j) {
      const auto fp32_score = score_fp32[j];
      const auto exp_score = expf(fp32_score - max_value);
      sum_product += cast<float>(kv[j][i]) * exp_score;
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

#pragma unroll
  for (uint32_t i = 0; i < kIteration; ++i) {

    const uint32_t j = i * kPrefillBlockSize + warp_id * kWarpThreads + lane_id;

    const uint32_t local_warp_id = j % kNumWarps;

    const uint32_t local_elem_id = j / kNumWarps;

    const uint32_t local_tile_id = local_elem_id % kTileElements;

    const uint32_t local_lane_id = local_elem_id / kTileElements;



    static_assert(kTileElements * kNumWarps == kWarpThreads, "TODO: support other configs");
    const auto local_val_max = s_local_val_max(local_warp_id, local_lane_id, local_tile_id);
    const auto local_exp_sum = s_local_exp_sum(local_warp_id, local_lane_id, local_tile_id);
    const auto local_product = s_local_product(local_warp_id, local_lane_id, local_tile_id);
    const auto global_val_max = warp::reduce_max<kNumWarps>(local_val_max);
    const auto rescale = expf(local_val_max - global_val_max);
    const auto global_exp_sum = warp::reduce_sum<kNumWarps>(local_exp_sum * rescale);
    const auto final_scale = rescale / global_exp_sum;
    const auto global_product = warp::reduce_sum<kNumWarps>(local_product * final_scale);
    kv_out[local_elem_id] = global_product;
    if constexpr (kNeedData) {
      max_out[local_elem_id] = global_val_max;
      sum_out[local_elem_id] = global_exp_sum;
    }
  }
  if constexpr (kNeedData) __syncthreads();
}




constexpr float kPadScore = -FLT_MAX;











template <int64_t kHeadDim, bool kWrite, bool kUsePDL>
__global__ __launch_bounds__(kPrefillBlockSize, 2)
    void flash_c128_online_prefill(const __grid_constant__ Compress128OnlinePrefillParams params) {
  using namespace device;

  constexpr int64_t kTileDim = kTileElements * kWarpThreads;
  constexpr uint32_t kNumSplit = kHeadDim / kTileDim;
  static_assert(kHeadDim % kTileDim == 0, "Head dim must be multiple of tile dim");


  const auto num_plans = kWrite ? params.num_write : params.num_compress;
  const auto plan_ptr = kWrite ? params.write_plan : params.compress_plan;
  const uint32_t global_id = blockIdx.x;
  const uint32_t global_pid = global_id / kNumSplit;
  const uint32_t global_sid = global_id % kNumSplit;
  if (global_pid >= num_plans) return;
  const auto [ragged_id, batch_id, position, window_len] = plan_ptr[global_pid];
  if (ragged_id == 0xFFFFFFFFu) [[unlikely]]
    return;

  const uint32_t warp_id = threadIdx.x / kWarpThreads;
  const uint32_t lane_id = threadIdx.x % kWarpThreads;
  const int32_t split_offset = global_sid * kTileDim;

  const auto kv_score_buffer = static_cast<float*>(params.kv_score_buffer);
  const auto kv_score_input = static_cast<const float*>(params.kv_score_input);
  const auto kv_compressed_output = static_cast<float*>(params.kv_compressed_output);
  const auto score_bias_base = static_cast<const float*>(params.score_bias);

  constexpr int64_t kElementSize = kHeadDim * 2;
  const uint32_t chunk_offset = (position % 128u) + 1u - window_len;
  const uint32_t window_end = chunk_offset + window_len;
  const int32_t segment_start = ragged_id - (position % 128u);
  const int32_t load_index = chunk_offset != 0 ? params.load_indices[batch_id] : -1;
  const int32_t store_index = kWrite ? params.indices[batch_id] : -1;

  PDLWaitPrimary<kUsePDL>();


  PrefillStorage kv[kElementsPerWarp];
  PrefillStorage score[kElementsPerWarp];
  PrefillStorage bias[kElementsPerWarp];
  const auto warp_offset = warp_id * kElementsPerWarp;

#pragma unroll
  for (uint32_t i = 0; i < kElementsPerWarp; ++i) {
    const uint32_t j = i + warp_offset;
    if (j >= chunk_offset && j < window_end) {
      const auto kv_src_ptr = kv_score_input + (segment_start + j) * kElementSize + split_offset;
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
    const bool is_valid = (j >= chunk_offset && j < window_end);
#pragma unroll
    for (uint32_t ii = 0; ii < kTileElements; ++ii) {
      score[i][ii] = is_valid ? score[i][ii] + bias[i][ii] : kPadScore;





      kv[i][ii] = is_valid ? kv[i][ii] : 0.0f;
    }
  }

  __shared__ alignas(16) float seg_kv[kTileDim];
  __shared__ alignas(16) float seg_max[kTileDim];
  __shared__ alignas(16) float seg_sum[kTileDim];

  c128_prefill_forward<true>(kv, score, seg_kv, seg_max, seg_sum, warp_id, lane_id);

  PDLTriggerSecondary<kUsePDL>();

  if (warp_id == 0) {
    PrefillStorage out_kv_vec, out_max_vec, out_sum_vec;
    out_kv_vec.load(seg_kv, lane_id);
    out_max_vec.load(seg_max, lane_id);
    out_sum_vec.load(seg_sum, lane_id);
    if (chunk_offset != 0) {



      const auto buf_load = kv_score_buffer + load_index * (kHeadDim * 3) + split_offset;
      PrefillStorage buf_max_vec, buf_sum_vec, buf_kv_vec;
      buf_max_vec.load(buf_load + 0 * kHeadDim, lane_id);
      buf_sum_vec.load(buf_load + 1 * kHeadDim, lane_id);
      buf_kv_vec.load(buf_load + 2 * kHeadDim, lane_id);
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
      const auto buf_store = kv_score_buffer + store_index * (kHeadDim * 3) + split_offset;
      reinterpret_cast<PrefillStorage*>(buf_store + 0 * kHeadDim)[lane_id] = out_max_vec;
      reinterpret_cast<PrefillStorage*>(buf_store + 1 * kHeadDim)[lane_id] = out_sum_vec;
      reinterpret_cast<PrefillStorage*>(buf_store + 2 * kHeadDim)[lane_id] = out_kv_vec;
    } else {
      const auto out_ptr = kv_compressed_output + ragged_id * kHeadDim + split_offset;
      reinterpret_cast<PrefillStorage*>(out_ptr)[lane_id] = out_kv_vec;
    }
  }
}

template <int64_t kHeadDim, bool kUsePDL>
struct FlashCompress128OnlineKernel {
  static constexpr auto decode_kernel = flash_c128_online_decode<kHeadDim, kUsePDL>;
  template <bool kWrite>
  static constexpr auto prefill_kernel = flash_c128_online_prefill<kHeadDim, kWrite, kUsePDL>;
  static constexpr auto prefill_c_kernel = prefill_kernel<           false>;
  static constexpr auto prefill_w_kernel = prefill_kernel<           true>;
  static constexpr int64_t kTileDim = kTileElements * device::kWarpThreads;
  static constexpr uint32_t kNumSplit = kHeadDim / kTileDim;
  static constexpr uint32_t kDecodeBlockSize = kHeadDim / 4;

  static void run_decode(
      const tvm::ffi::TensorView kv_score_buffer,
      const tvm::ffi::TensorView kv_score_input,
      const tvm::ffi::TensorView kv_compressed_output,
      const tvm::ffi::TensorView ape,
      const tvm::ffi::TensorView indices,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::Optional<tvm::ffi::TensorView>             ) {
    using namespace host;

    auto B = SymbolicSize{"batch_size"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({-1, 1, kHeadDim * 3})
        .with_dtype<float>()
        .with_device(device)
        .verify(kv_score_buffer);
    TensorMatcher({B, kHeadDim * 2})
        .with_dtype<float>()
        .with_device(device)
        .verify(kv_score_input);
    TensorMatcher({B, kHeadDim})
        .with_dtype<float>()
        .with_device(device)
        .verify(kv_compressed_output);
    TensorMatcher({128, kHeadDim})
        .with_dtype<float>()
        .with_device(device)
        .verify(ape);
    TensorMatcher({B}).with_dtype<IndiceT>().with_device(device).verify(indices);
    TensorMatcher({B}).with_dtype<IndiceT>().with_device(device).verify(seq_lens);

    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    const auto params = Compress128OnlineDecodeParams{
        .kv_score_buffer = kv_score_buffer.data_ptr(),
        .kv_score_input = kv_score_input.data_ptr(),
        .kv_compressed_output = kv_compressed_output.data_ptr(),
        .score_bias = ape.data_ptr(),
        .indices = static_cast<const IndiceT*>(indices.data_ptr()),
        .seq_lens = static_cast<const IndiceT*>(seq_lens.data_ptr()),
        .batch_size = batch_size,
    };
    LaunchKernel(batch_size, kDecodeBlockSize, device.unwrap())
        .enable_pdl(kUsePDL)(decode_kernel, params);
  }

  static void run_prefill(
      const tvm::ffi::TensorView kv_score_buffer,
      const tvm::ffi::TensorView kv_score_input,
      const tvm::ffi::TensorView kv_compressed_output,
      const tvm::ffi::TensorView ape,
      const tvm::ffi::TensorView indices,
      const tvm::ffi::TensorView compress_plan,
      const tvm::ffi::TensorView write_plan,
      const tvm::ffi::Optional<tvm::ffi::TensorView> extra) {
    using namespace host;
    using host::compress::kOnlinePrefillPlanDim;
    using host::compress::OnlinePrefillPlanTensorDtype;

    auto B = SymbolicSize{"batch_size"};
    auto N = SymbolicSize{"num_q_tokens"};
    auto X = SymbolicSize{"compress_tokens"};
    auto Y = SymbolicSize{"write_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({-1, 1, kHeadDim * 3})
        .with_dtype<float>()
        .with_device(device_)
        .verify(kv_score_buffer);
    TensorMatcher({N, kHeadDim * 2})
        .with_dtype<float>()
        .with_device(device_)
        .verify(kv_score_input);
    TensorMatcher({N, kHeadDim})
        .with_dtype<float>()
        .with_device(device_)
        .verify(kv_compressed_output);
    TensorMatcher({128, kHeadDim})
        .with_dtype<float>()
        .with_device(device_)
        .verify(ape);
    TensorMatcher({B})
        .with_dtype<IndiceT>()
        .with_device(device_)
        .verify(indices);
    TensorMatcher({X, kOnlinePrefillPlanDim})
        .with_dtype<OnlinePrefillPlanTensorDtype>()
        .with_device(device_)
        .verify(compress_plan);
    TensorMatcher({Y, kOnlinePrefillPlanDim})
        .with_dtype<OnlinePrefillPlanTensorDtype>()
        .with_device(device_)
        .verify(write_plan);




    const auto load_indices = extra.value_or(indices);
    TensorMatcher({B}).with_dtype<IndiceT>().with_device(device_).verify(load_indices);

    const auto device = device_.unwrap();
    const auto num_c = static_cast<uint32_t>(X.unwrap());
    const auto num_w = static_cast<uint32_t>(Y.unwrap());
    const auto params = Compress128OnlinePrefillParams{
        .kv_score_buffer = kv_score_buffer.data_ptr(),
        .kv_score_input = kv_score_input.data_ptr(),
        .kv_compressed_output = kv_compressed_output.data_ptr(),
        .score_bias = ape.data_ptr(),
        .indices = static_cast<const IndiceT*>(indices.data_ptr()),
        .load_indices = static_cast<const IndiceT*>(load_indices.data_ptr()),
        .compress_plan = static_cast<const OnlinePlan*>(compress_plan.data_ptr()),
        .write_plan = static_cast<const OnlinePlan*>(write_plan.data_ptr()),
        .num_compress = num_c,
        .num_write = num_w,
    };






    if (const auto num_c_blocks = num_c * kNumSplit) {
      LaunchKernel(num_c_blocks, kPrefillBlockSize, device)
          .enable_pdl(kUsePDL)(prefill_c_kernel, params);
    }
    if (const auto num_w_blocks = num_w * kNumSplit) {
      LaunchKernel(num_w_blocks, kPrefillBlockSize, device)
          .enable_pdl(kUsePDL)(prefill_w_kernel, params);
    }
  }
};

namespace host::compress {

using OnlinePlanResult = tvm::ffi::Tuple<uint32_t, uint32_t>;

struct OnlinePrefillCompressParams {
  OnlinePrefillPlan* __restrict__ compress_plan;
  OnlinePrefillPlan* __restrict__ write_plan;
  const int64_t* __restrict__ seq_lens;
  const int64_t* __restrict__ extend_lens;
  uint32_t batch_size;
  uint32_t num_tokens;
};















inline OnlinePlanResult plan_online_prefill_host(const OnlinePrefillCompressParams& params, const bool use_cuda_graph) {
  const auto& [compress_plan, write_plan, seq_lens, extend_lens, batch_size, num_tokens] = params;

  uint32_t counter = 0;
  uint32_t compress_count = 0;
  uint32_t write_count = 0;
  for (const auto i : irange(batch_size)) {
    const uint32_t seq_len = static_cast<uint32_t>(seq_lens[i]);
    const uint32_t extend_len = static_cast<uint32_t>(extend_lens[i]);
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
      const auto plan = OnlinePrefillPlan{
          .ragged_id = last_ragged,
          .batch_id = i,
          .position = last_pos,
          .window_len = seg_len,
      };
      if (chunk_off + seg_len == 128u) {

        RuntimeCheck(compress_count < num_tokens);
        compress_plan[compress_count++] = plan;
      } else {

        RuntimeCheck(write_count < num_tokens);
        write_plan[write_count++] = plan;
      }
      pos = seg_end;
    }
    counter += extend_len;
  }
  RuntimeCheck(counter == num_tokens, "input size ", counter, " != num_q_tokens ", num_tokens);
  if (!use_cuda_graph) return OnlinePlanResult{compress_count, write_count};


  constexpr auto kInvalid = static_cast<uint32_t>(-1);
  constexpr auto kInvalidPlan = OnlinePrefillPlan{kInvalid, kInvalid, kInvalid, kInvalid};
  for (const auto i : irange(compress_count, num_tokens)) {
    compress_plan[i] = kInvalidPlan;
  }
  for (const auto i : irange(write_count, num_tokens)) {
    write_plan[i] = kInvalidPlan;
  }
  return OnlinePlanResult{num_tokens, num_tokens};
}

inline OnlinePlanResult plan_online_prefill(
    const tvm::ffi::TensorView extend_lens,
    const tvm::ffi::TensorView seq_lens,
    const tvm::ffi::TensorView compress_plan,
    const tvm::ffi::TensorView write_plan,
    const bool use_cuda_graph) {
  auto N = SymbolicSize{"batch_size"};
  auto M = SymbolicSize{"num_tokens"};
  auto device = SymbolicDevice{};

  device.set_options<kDLCPU, kDLCUDAHost>();
  TensorMatcher({N})
      .with_dtype<int64_t>()
      .with_device(device)
      .verify(extend_lens)
      .verify(seq_lens);
  TensorMatcher({M, kOnlinePrefillPlanDim})
      .with_dtype<OnlinePrefillPlanTensorDtype>()
      .with_device(device)
      .verify(compress_plan)
      .verify(write_plan);
  const auto params = OnlinePrefillCompressParams{
      .compress_plan = static_cast<OnlinePrefillPlan*>(compress_plan.data_ptr()),
      .write_plan = static_cast<OnlinePrefillPlan*>(write_plan.data_ptr()),
      .seq_lens = static_cast<const int64_t*>(seq_lens.data_ptr()),
      .extend_lens = static_cast<const int64_t*>(extend_lens.data_ptr()),
      .batch_size = static_cast<uint32_t>(N.unwrap()),
      .num_tokens = static_cast<uint32_t>(M.unwrap()),
  };
  return plan_online_prefill_host(params, use_cuda_graph);
}

}

[[maybe_unused]]
constexpr auto& plan_compress_online_prefill = host::compress::plan_online_prefill;

}
