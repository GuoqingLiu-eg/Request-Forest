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
#include <tvm/ffi/object.h>

#include <cstdint>

namespace sglang {

using Plan128 = device::compress::PrefillPlan;
using IndiceT = int32_t;


constexpr int32_t kTileElements = 2;

constexpr int32_t kElementsPerWarp = 8;
constexpr uint32_t kNumWarps = 128 / kElementsPerWarp;
constexpr uint32_t kBlockSize = device::kWarpThreads * kNumWarps;


#define C128_KERNEL __global__ __launch_bounds__(kBlockSize, 2)

struct Compress128DecodeParams {





  void* __restrict__ kv_score_buffer;

  const void* __restrict__ kv_score_input;

  void* __restrict__ kv_compressed_output;

  const void* __restrict__ score_bias;

  const IndiceT* __restrict__ indices;

  const IndiceT* __restrict__ seq_lens;

  uint32_t batch_size;
};

struct Compress128PrefillParams {





  void* __restrict__ kv_score_buffer;

  const void* __restrict__ kv_score_input;

  void* __restrict__ kv_compressed_output;

  const void* __restrict__ score_bias;

  const IndiceT* __restrict__ indices;

  const int32_t* __restrict__ load_indices;

  const Plan128* __restrict__ compress_plan;
  const Plan128* __restrict__ write_plan;
  uint32_t num_compress;
  uint32_t num_write;
};

struct Compress128SharedBuffer {
  using Storage = device::AlignedVector<float, kTileElements>;
  Storage data[kNumWarps][device::kWarpThreads + 1];
  SGL_DEVICE Storage& operator()(uint32_t warp_id, uint32_t lane_id) {
    return data[warp_id][lane_id];
  }
  SGL_DEVICE float& operator()(uint32_t warp_id, uint32_t lane_id, uint32_t tile_id) {
    return data[warp_id][lane_id][tile_id];
  }
};

template <typename T>
SGL_DEVICE void c128_write(
    T* kv_score_buf,
    const T* kv_score_src,
    const int64_t head_dim,
    const int32_t write_pos,
    const uint32_t lane_id) {
  using namespace device;

  using Storage = AlignedVector<T, kTileElements>;
  const auto element_size = head_dim * 2;
  const auto gmem = tile::Memory<Storage>{lane_id, kWarpThreads};
  kv_score_buf += write_pos * element_size;


  Storage kv_score[2];
#pragma unroll
  for (int32_t i = 0; i < 2; ++i) {
    kv_score[i] = gmem.load(kv_score_src + head_dim * i);
  }
#pragma unroll
  for (int32_t i = 0; i < 2; ++i) {
    gmem.store(kv_score_buf + head_dim * i, kv_score[i]);
  }
}

template <typename InFloat, typename OutFloat>
SGL_DEVICE void c128_forward(
    const InFloat* kv_score_buf,
    const InFloat* kv_score_src,
    OutFloat* kv_out,
    const InFloat* score_bias,
    const int64_t head_dim,
    const int32_t window_len,
    const uint32_t warp_id,
    const uint32_t lane_id) {
  using namespace device;

  const auto element_size = head_dim * 2;
  const auto score_offset = head_dim;


  using StorageIn = AlignedVector<InFloat, kTileElements>;
  const auto gmem_in = tile::Memory<StorageIn>{lane_id, kWarpThreads};
  StorageIn kv[kElementsPerWarp];
  StorageIn score[kElementsPerWarp];
  StorageIn bias[kElementsPerWarp];
  const int32_t warp_offset = warp_id * kElementsPerWarp;

#pragma unroll
  for (int32_t i = 0; i < 8; ++i) {
    const int32_t j = i + warp_offset;
    bias[i] = gmem_in.load(score_bias + j * head_dim);
  }

#pragma unroll
  for (int32_t i = 0; i < kElementsPerWarp; ++i) {
    const int32_t j = i + warp_offset;
    const InFloat* src;
    __builtin_assume(j < 128);
    if (j < window_len) {
      src = kv_score_buf + j * element_size;
    } else {

      const int32_t k = j - 127;
      src = kv_score_src + k * element_size;
    }
    kv[i] = gmem_in.load(src);
    score[i] = gmem_in.load(src + score_offset);
  }


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
      score_fp32[j] = cast<float>(score[j][i]) + cast<float>(bias[j][i]);
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
  constexpr uint32_t kIteration = kReductionCount / kBlockSize;

#pragma unroll
  for (uint32_t i = 0; i < kIteration; ++i) {

    const uint32_t j = i * kBlockSize + warp_id * kWarpThreads + lane_id;

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
    kv_out[local_elem_id] = cast<OutFloat>(global_product);
  }
}

template <int64_t kHeadDim, typename InFloat, typename OutFloat, bool kUsePDL>
C128_KERNEL void flash_c128_decode(const __grid_constant__ Compress128DecodeParams params) {
  using namespace device;

  constexpr int64_t kTileDim = kTileElements * kWarpThreads;
  constexpr uint32_t kNumSplit = kHeadDim / kTileDim;
  constexpr int64_t kElementSize = kHeadDim * 2;
  static_assert(kHeadDim % kTileDim == 0, "Head dim must be multiple of tile dim");

  const auto& [
    _kv_score_buffer, _kv_score_input, _kv_compressed_output, _score_bias,
    indices, seq_lens, batch_size
  ] = params;
  const uint32_t warp_id = threadIdx.x / kWarpThreads;
  const uint32_t lane_id = threadIdx.x % kWarpThreads;

  const uint32_t global_bid = blockIdx.x / kNumSplit;
  const uint32_t global_sid = blockIdx.x % kNumSplit;
  if (global_bid >= batch_size) return;

  const int32_t index = indices[global_bid];
  const int32_t seq_len = seq_lens[global_bid];
  const int64_t split_offset = global_sid * kTileDim;


  const auto kv_score_buffer = static_cast<InFloat*>(_kv_score_buffer);
  const auto kv_buf = kv_score_buffer + index * (kElementSize * 128) + split_offset;


  const auto kv_score_input = static_cast<const InFloat*>(_kv_score_input);
  const auto kv_src = kv_score_input + global_bid * kElementSize + split_offset;


  const auto kv_compressed_output = static_cast<OutFloat*>(_kv_compressed_output);
  const auto kv_out = kv_compressed_output + global_bid * kHeadDim + split_offset;


  const auto score_bias = static_cast<const InFloat*>(_score_bias) + split_offset;

  PDLWaitPrimary<kUsePDL>();




  if (warp_id == kNumWarps - 1) {
    c128_write(kv_buf, kv_src, kHeadDim,               (seq_len + 127) % 128, lane_id);
  }
  if (seq_len % 128 == 0) {
    c128_forward(kv_buf, kv_src, kv_out, score_bias, kHeadDim,                128, warp_id, lane_id);
  }

  PDLTriggerSecondary<kUsePDL>();
}


template <int64_t kHeadDim, typename InFloat, typename OutFloat, bool kWrite, bool kUsePDL>
C128_KERNEL void flash_c128_prefill(const __grid_constant__ Compress128PrefillParams params) {
  using namespace device;

  constexpr int64_t kTileDim = kTileElements * kWarpThreads;
  constexpr uint32_t kNumSplit = kHeadDim / kTileDim;
  constexpr int64_t kElementSize = kHeadDim * 2;
  static_assert(kHeadDim % kTileDim == 0, "Head dim must be multiple of tile dim");

  const auto& [
    _kv_score_buffer, _kv_score_input, _kv_compressed_output, _score_bias,
    indices, load_indices, compress_plan, write_plan, num_compress, num_write
  ] = params;
  const uint32_t warp_id = threadIdx.x / kWarpThreads;
  const uint32_t lane_id = threadIdx.x % kWarpThreads;

  uint32_t global_id;
  if constexpr (kWrite) {

    global_id = (blockIdx.x * blockDim.x + threadIdx.x) / kWarpThreads;
  } else {

    global_id = blockIdx.x;
  }
  const uint32_t global_pid = global_id / kNumSplit;
  const uint32_t global_sid = global_id % kNumSplit;


  const auto num_plans = kWrite ? num_write : num_compress;
  const auto plan_ptr = kWrite ? write_plan : compress_plan;
  if (global_pid >= num_plans) return;

  const auto& [ragged_id, global_bid, position, window_len] = plan_ptr[global_pid];
  const auto indices_ptr = kWrite ? indices : load_indices;

  const int64_t split_offset = global_sid * kTileDim;


  const auto kv_score_input = static_cast<const InFloat*>(_kv_score_input);
  const auto kv_src = kv_score_input + ragged_id * kElementSize + split_offset;


  const auto kv_compressed_output = static_cast<OutFloat*>(_kv_compressed_output);
  const auto kv_out = kv_compressed_output + ragged_id * kHeadDim + split_offset;


  const auto score_bias = static_cast<const InFloat*>(_score_bias) + split_offset;

  if (ragged_id == 0xFFFFFFFF) [[unlikely]]
    return;

  const int32_t index = indices_ptr[global_bid];

  const auto kv_score_buffer = static_cast<InFloat*>(_kv_score_buffer);
  const auto kv_buf = kv_score_buffer + index * (kElementSize * 128) + split_offset;

  PDLWaitPrimary<kUsePDL>();


  if constexpr (kWrite) {
    c128_write(kv_buf, kv_src, kHeadDim,               position % 128, lane_id);
  } else {
    c128_forward(kv_buf, kv_src, kv_out, score_bias, kHeadDim, window_len, warp_id, lane_id);
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kHeadDim, typename InFloat, typename OutFloat, bool kUsePDL>
struct FlashCompress128Kernel {
  static constexpr auto decode_kernel = flash_c128_decode<kHeadDim, InFloat, OutFloat, kUsePDL>;
  template <bool kWrite>
  static constexpr auto prefill_kernel = flash_c128_prefill<kHeadDim, InFloat, OutFloat, kWrite, kUsePDL>;
  static constexpr auto prefill_c_kernel = prefill_kernel<           false>;
  static constexpr auto prefill_w_kernel = prefill_kernel<           true>;
  static constexpr int64_t kTileDim = kTileElements * device::kWarpThreads;
  static constexpr uint32_t kNumSplit = kHeadDim / kTileDim;
  static constexpr uint32_t kWriteBlockSize = 128;
  static constexpr uint32_t kWarpsPerWriteBlock = kWriteBlockSize / device::kWarpThreads;

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

    TensorMatcher({-1, 128, kHeadDim * 2})
        .with_dtype<InFloat>()
        .with_device(device)
        .verify(kv_score_buffer);
    TensorMatcher({B, kHeadDim * 2})
        .with_dtype<InFloat>()
        .with_device(device)
        .verify(kv_score_input);
    TensorMatcher({B, kHeadDim})
        .with_dtype<OutFloat>()
        .with_device(device)
        .verify(kv_compressed_output);
    TensorMatcher({128, kHeadDim})
        .with_dtype<InFloat>()
        .with_device(device)
        .verify(ape);
    TensorMatcher({B})
        .with_dtype<IndiceT>()
        .with_device(device)
        .verify(indices);
    TensorMatcher({B})
        .with_dtype<IndiceT>()
        .with_device(device)
        .verify(seq_lens);

    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    const auto params = Compress128DecodeParams{
        .kv_score_buffer = kv_score_buffer.data_ptr(),
        .kv_score_input = kv_score_input.data_ptr(),
        .kv_compressed_output = kv_compressed_output.data_ptr(),
        .score_bias = ape.data_ptr(),
        .indices = static_cast<const IndiceT*>(indices.data_ptr()),
        .seq_lens = static_cast<const IndiceT*>(seq_lens.data_ptr()),
        .batch_size = batch_size,
    };

    const uint32_t num_blocks = batch_size * kNumSplit;
    LaunchKernel(num_blocks, kBlockSize, device.unwrap())
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

    auto B = SymbolicSize{"batch_size"};
    auto N = SymbolicSize{"num_q_tokens"};
    auto X = SymbolicSize{"compress_tokens"};
    auto Y = SymbolicSize{"write_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({-1, 128, kHeadDim * 2})
        .with_dtype<InFloat>()
        .with_device(device_)
        .verify(kv_score_buffer);
    TensorMatcher({N, kHeadDim * 2})
        .with_dtype<InFloat>()
        .with_device(device_)
        .verify(kv_score_input);
    TensorMatcher({N, kHeadDim})
        .with_dtype<OutFloat>()
        .with_device(device_)
        .verify(kv_compressed_output);
    TensorMatcher({128, kHeadDim})
        .with_dtype<InFloat>()
        .with_device(device_)
        .verify(ape);
    TensorMatcher({B})
        .with_dtype<IndiceT>()
        .with_device(device_)
        .verify(indices);
    TensorMatcher({X, compress::kPrefillPlanDim})
        .with_dtype<compress::PrefillPlanTensorDtype>()
        .with_device(device_)
        .verify(compress_plan);
    TensorMatcher({Y, compress::kPrefillPlanDim})
        .with_dtype<compress::PrefillPlanTensorDtype>()
        .with_device(device_)
        .verify(write_plan);


    const auto load_indices = extra.value_or(indices);
    TensorMatcher({B})
        .with_dtype<IndiceT>()
        .with_device(device_)
        .verify(load_indices);

    const auto device = device_.unwrap();
    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    const auto num_q_tokens = static_cast<uint32_t>(N.unwrap());
    const auto num_c = static_cast<uint32_t>(X.unwrap());
    const auto num_w = static_cast<uint32_t>(Y.unwrap());
    const auto params = Compress128PrefillParams{
        .kv_score_buffer = kv_score_buffer.data_ptr(),
        .kv_score_input = kv_score_input.data_ptr(),
        .kv_compressed_output = kv_compressed_output.data_ptr(),
        .score_bias = ape.data_ptr(),
        .indices = static_cast<const IndiceT*>(indices.data_ptr()),
        .load_indices = static_cast<const IndiceT*>(load_indices.data_ptr()),
        .compress_plan = static_cast<const Plan128*>(compress_plan.data_ptr()),
        .write_plan = static_cast<const Plan128*>(write_plan.data_ptr()),
        .num_compress = num_c,
        .num_write = num_w,
    };
    RuntimeCheck(num_q_tokens >= batch_size, "num_q_tokens must be >= batch_size");
    RuntimeCheck(num_q_tokens >= std::max(num_c, num_w), "invalid prefill plan");

    constexpr auto kBlockSize_C = kBlockSize;
    constexpr auto kBlockSize_W = kWriteBlockSize;
    if (const auto num_c_blocks = num_c * kNumSplit) {
      LaunchKernel(num_c_blocks, kBlockSize_C, device)
          .enable_pdl(kUsePDL)(prefill_c_kernel, params);
    }
    if (const auto num_w_blocks = div_ceil(num_w * kNumSplit, kWarpsPerWriteBlock)) {
      LaunchKernel(num_w_blocks, kBlockSize_W, device)
          .enable_pdl(kUsePDL)(prefill_w_kernel, params);
    }
  }
};

}
