#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cassert>
#include <cstdint>

namespace sglang {

struct StoreKVCacheParams {
  const void* __restrict__ k;
  const void* __restrict__ v;
  void* __restrict__ k_cache;
  void* __restrict__ v_cache;
  const void* __restrict__ indices;
  int64_t stride_k_bytes;
  int64_t stride_v_bytes;

  int64_t stride_k_cache_bytes;
  int64_t stride_v_cache_bytes;
  int64_t stride_indices;
  uint32_t batch_size;
  int64_t size_limit;
  int64_t reserved_skip_index;
};

constexpr uint32_t kNumWarps = 4;
constexpr uint32_t kThreadsPerBlock = kNumWarps * device::kWarpThreads;








template <int64_t kElementBytes>
struct RowVecPlan {
  static constexpr int64_t kAlignment = (kElementBytes % (16 * device::kWarpThreads) == 0) ? 16
                                        : kElementBytes % (8 * device::kWarpThreads) == 0  ? 8
                                        : kElementBytes % (4 * device::kWarpThreads) == 0  ? 4
                                        : kElementBytes % 4 == 0                           ? 4
                                                                                           : 0;

  static_assert(kAlignment > 0, "Element size must be multiple of 4 bytes");

  using vec_t = device::AlignedStorage<uint32_t, kAlignment / 4>;
  static constexpr int64_t kLoopBytes = sizeof(vec_t) * device::kWarpThreads;
  static constexpr int64_t kLoopCount = kElementBytes / kLoopBytes;
  static constexpr int64_t kElementCount = kElementBytes / sizeof(vec_t);
  static constexpr bool kHasEpilogue = kLoopCount * kLoopBytes < kElementBytes;
};














template <int64_t kElementBytes>
SGL_DEVICE void copy_kv_warp(
    const void* __restrict__ k_src,
    const void* __restrict__ v_src,
    void* __restrict__ k_dst,
    void* __restrict__ v_dst) {
  using namespace device;
  using plan_t = RowVecPlan<kElementBytes>;
  using vec_t = typename plan_t::vec_t;
  constexpr auto kLoopCount = plan_t::kLoopCount;

  const auto gmem = tile::Memory<vec_t>::warp();

#pragma unroll kLoopCount
  for (int64_t i = 0; i < kLoopCount; ++i) {
    const auto k = gmem.load(k_src, i);
    const auto v = gmem.load(v_src, i);
    gmem.store(k_dst, k, i);
    gmem.store(v_dst, v, i);
  }


  if constexpr (plan_t::kHasEpilogue) {
    if (gmem.in_bound(plan_t::kElementCount, kLoopCount)) {
      const auto k = gmem.load(k_src, kLoopCount);
      const auto v = gmem.load(v_src, kLoopCount);
      gmem.store(k_dst, k, kLoopCount);
      gmem.store(v_dst, v, kLoopCount);
    }
  }
}









template <int64_t kElementBytes>
SGL_DEVICE void copy_row_warp(const void* __restrict__ src, void* __restrict__ dst) {
  using namespace device;
  using plan_t = RowVecPlan<kElementBytes>;
  using vec_t = typename plan_t::vec_t;
  constexpr auto kLoopCount = plan_t::kLoopCount;

  const auto gmem = tile::Memory<vec_t>::warp();

#pragma unroll kLoopCount
  for (int64_t i = 0; i < kLoopCount; ++i) {
    gmem.store(dst, gmem.load(src, i), i);
  }


  if constexpr (plan_t::kHasEpilogue) {
    if (gmem.in_bound(plan_t::kElementCount, kLoopCount)) {
      gmem.store(dst, gmem.load(src, kLoopCount), kLoopCount);
    }
  }
}







template <int64_t kKBytes, int64_t kVBytes>
SGL_DEVICE void copy_kv_rows_warp(
    const void* __restrict__ k_src,
    const void* __restrict__ v_src,
    void* __restrict__ k_dst,
    void* __restrict__ v_dst) {
  using namespace device;
  constexpr auto kCommon = kKBytes < kVBytes ? kKBytes : kVBytes;
  constexpr auto kTail = (kKBytes < kVBytes ? kVBytes : kKBytes) - kCommon;







  constexpr auto kTailOrCommon = kTail == 0 ? kCommon : kTail;
  constexpr auto kCommonAlign = RowVecPlan<kCommon>::kAlignment;
  constexpr auto kTailAlign = RowVecPlan<kTailOrCommon>::kAlignment;
  constexpr bool kCanInterleave =
      kKBytes % kCommonAlign == 0 && kVBytes % kCommonAlign == 0 && kCommon % kTailAlign == 0;

  if constexpr (kCanInterleave) {
    copy_kv_warp<kCommon>(k_src, v_src, k_dst, v_dst);
    if constexpr (kTail > 0) {
      if constexpr (kKBytes > kVBytes) {
        copy_row_warp<kTail>(pointer::offset(k_src, kCommon), pointer::offset(k_dst, kCommon));
      } else {
        copy_row_warp<kTail>(pointer::offset(v_src, kCommon), pointer::offset(v_dst, kCommon));
      }
    }
  } else {
    copy_row_warp<kKBytes>(k_src, k_dst);
    copy_row_warp<kVBytes>(v_src, v_dst);
  }
}











template <int64_t kKElementBytes, int64_t kVElementBytes, int kSplit, bool kUsePDL, typename T>
__global__ void store_kvcache(const __grid_constant__ StoreKVCacheParams params) {
  using namespace device;
  constexpr auto kKSplitSize = kKElementBytes / kSplit;
  constexpr auto kVSplitSize = kVElementBytes / kSplit;
  const uint32_t warp_id = blockIdx.x * kNumWarps + threadIdx.x / kWarpThreads;
  const uint32_t item_id = warp_id / kSplit;
  const uint32_t split_id = warp_id % kSplit;
  const auto& [
    k_input, v_input, k_cache, v_cache, indices,
    stride_k, stride_v, stride_k_cache, stride_v_cache, stride_indices, batch_size,
    size_limit, reserved_skip_index
  ] = params;
  if (item_id >= batch_size) return;

  const auto index_ptr = static_cast<const T*>(indices) + item_id * stride_indices;
  PDLWaitPrimary<kUsePDL>();

  const auto index = *index_ptr;


  assert(index >= 0 && index < size_limit);
  const auto k_src = pointer::offset(k_input, item_id * stride_k, split_id * kKSplitSize);
  const auto v_src = pointer::offset(v_input, item_id * stride_v, split_id * kVSplitSize);
  const auto k_dst = pointer::offset(k_cache, index * stride_k_cache, split_id * kKSplitSize);
  const auto v_dst = pointer::offset(v_cache, index * stride_v_cache, split_id * kVSplitSize);

  if (index != reserved_skip_index) {
    copy_kv_rows_warp<kKSplitSize, kVSplitSize>(k_src, v_src, k_dst, v_dst);
  }
  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kKElementBytes, int64_t kVElementBytes, bool kUsePDL>
struct StoreKVCacheKernel {
  static_assert(kKElementBytes > 0 && kKElementBytes % 4 == 0);
  static_assert(kVElementBytes > 0 && kVElementBytes % 4 == 0);

  template <int kSplit, typename T>
  static constexpr auto store_kernel = store_kvcache<kKElementBytes, kVElementBytes, kSplit, kUsePDL, T>;

  template <typename T>
  static auto get_kernel(const int num_split) {
    using namespace host;

    if constexpr (kKElementBytes % (4 * 128) == 0 && kVElementBytes % (4 * 128) == 0) {
      if (num_split == 4) return store_kernel<4, T>;
    }
    if constexpr (kKElementBytes % (2 * 128) == 0 && kVElementBytes % (2 * 128) == 0) {
      if (num_split == 2) return store_kernel<2, T>;
    }
    if (num_split == 1) return store_kernel<1, T>;
    Panic("Unsupported num_split {} for element sizes k={} v={}", num_split, kKElementBytes, kVElementBytes);
  }

  static void
  run(const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView v,
      const tvm::ffi::TensorView k_cache,
      const tvm::ffi::TensorView v_cache,
      const tvm::ffi::TensorView indices,
      const int num_split,
      const int64_t size_limit,
      const int64_t reserved_skip_index) {
    using namespace host;
    auto B = SymbolicSize{"batch_size"};
    auto DK = SymbolicSize{"k_element_size"};
    auto DV = SymbolicSize{"v_element_size"};
    auto KS = SymbolicSize{"k_stride"};
    auto VS = SymbolicSize{"v_stride"};
    auto SK = SymbolicSize{"k_cache_stride"};
    auto SV = SymbolicSize{"v_cache_stride"};
    auto I = SymbolicSize{"indices_stride"};
    auto dtype = SymbolicDType{};
    auto device = SymbolicDevice{};
    auto indice_dtype = SymbolicDType{};
    device.set_options<kDLCUDA, kDLROCM>();

    TensorMatcher({B, DK})
        .with_strides({KS, 1})
        .with_dtype(dtype)
        .with_device(device)
        .verify(k);
    TensorMatcher({B, DV})
        .with_strides({VS, 1})
        .with_dtype(dtype)
        .with_device(device)
        .verify(v);
    TensorMatcher({-1, DK})
        .with_strides({SK, 1})
        .with_dtype(dtype)
        .with_device(device)
        .verify(k_cache);
    TensorMatcher({-1, DV})
        .with_strides({SV, 1})
        .with_dtype(dtype)
        .with_device(device)
        .verify(v_cache);
    TensorMatcher({B})
        .with_strides({I})
        .with_dtype<int32_t, int64_t>(indice_dtype)
        .with_device(device)
        .verify(indices);

    const int64_t dtype_size = dtype_bytes(dtype.unwrap());
    const uint32_t num_elements = static_cast<uint32_t>(B.unwrap());
    RuntimeCheck(kKElementBytes == dtype_size * DK.unwrap());
    RuntimeCheck(kVElementBytes == dtype_size * DV.unwrap());

    const auto params = StoreKVCacheParams{
        .k = k.data_ptr(),
        .v = v.data_ptr(),
        .k_cache = k_cache.data_ptr(),
        .v_cache = v_cache.data_ptr(),
        .indices = indices.data_ptr(),
        .stride_k_bytes = KS.unwrap() * dtype_size,
        .stride_v_bytes = VS.unwrap() * dtype_size,
        .stride_k_cache_bytes = SK.unwrap() * dtype_size,
        .stride_v_cache_bytes = SV.unwrap() * dtype_size,
        .stride_indices = I.unwrap(),
        .batch_size = static_cast<uint32_t>(B.unwrap()),
        .size_limit = size_limit,
        .reserved_skip_index = reserved_skip_index,
    };

    const auto use_int32 = indice_dtype.is_type<int32_t>();
    const auto kernel = use_int32 ? get_kernel<int32_t>(num_split) : get_kernel<int64_t>(num_split);
    const auto num_blocks = div_ceil(num_elements * num_split, kNumWarps);
    LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap())
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}
