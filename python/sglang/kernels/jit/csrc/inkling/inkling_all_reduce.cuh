




























#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <dlpack/dlpack.h>

#include "inkling_ar_barrier.cuh"
#include <bit>
#include <cstdint>
#include <mutex>
#include <type_traits>
#include <unordered_map>

namespace sglang {

template <typename DType, uint32_t kNumGPU>
struct InklingAllReduceTrait {
  static constexpr uint32_t kVecSize = 16 / (sizeof(DType) * 2);
  static constexpr uint32_t kElemsPerVec = kVecSize * 2;
  using DType2 = packed_t<DType>;
  using Storage = device::AlignedVector<DType2, kVecSize>;
  static_assert(sizeof(Storage) == 16 && alignof(Storage) == 16, "Storage must be 16B");
  static_assert(std::has_single_bit(kNumGPU), "kNumGPU must be a power of two");
};




template <typename DType>
__device__ __forceinline__ typename InklingAllReduceTrait<DType, 2>::Storage add_vec_rn(
    const typename InklingAllReduceTrait<DType, 2>::Storage& a,
    const typename InklingAllReduceTrait<DType, 2>::Storage& b) {
  using namespace device;
  using Trait = InklingAllReduceTrait<DType, 2>;
  using DType2 = typename Trait::DType2;
  typename Trait::Storage out;
#pragma unroll
  for (uint32_t j = 0; j < Trait::kVecSize; ++j) {
    const fp32x2_t x = cast<fp32x2_t>(a[j]);
    const fp32x2_t y = cast<fp32x2_t>(b[j]);
    fp32x2_t s;
    s.x = x.x + y.x;
    s.y = x.y + y.y;
    out[j] = cast<DType2>(s);
  }
  return out;
}











template <typename DType, uint32_t kNumGPU>
__device__ __forceinline__ void
fold_shared_local(DType* __restrict__ buf, const DType* __restrict__ shared, uint32_t num_items) {
  using Trait = InklingAllReduceTrait<DType, kNumGPU>;
  using Storage = typename Trait::Storage;
  const uint32_t total_vec = num_items / Trait::kElemsPerVec;
  const uint32_t stride = gridDim.x * blockDim.x;
  for (uint32_t v = blockIdx.x * blockDim.x + threadIdx.x; v < total_vec; v += stride) {
    Storage a, b;
    a.load(buf, v);
    b.load(shared, v);
    add_vec_rn<DType>(a, b).store(buf, v);
  }
}



template <typename DType, uint32_t kNumGPU>
__device__ __forceinline__ uint2 rank_vec_slice(uint32_t rank, uint32_t num_items) {
  using namespace device;
  using Trait = InklingAllReduceTrait<DType, kNumGPU>;
  const uint32_t total_vec = num_items / Trait::kElemsPerVec;
  const uint32_t vec_per_rank = div_ceil(div_ceil(total_vec, kNumGPU), kWarpThreads) * kWarpThreads;
  const uint32_t start = min(rank * vec_per_rank, total_vec);
  const uint32_t finish = min(start + vec_per_rank, total_vec);
  return {start, finish - start};
}



template <typename DType, uint32_t kNumGPU>
__device__ __forceinline__ uint32_t
slice_setup(DType* (&input)[kNumGPU], void* const* peer_ptrs, uint32_t rank, uint32_t num_items) {
  using Trait = InklingAllReduceTrait<DType, kNumGPU>;
  const uint2 slice = rank_vec_slice<DType, kNumGPU>(rank, num_items);
  const uint32_t base = slice.x * Trait::kElemsPerVec;
#pragma unroll
  for (uint32_t i = 0; i < kNumGPU; ++i)
    input[i] = static_cast<DType*>(peer_ptrs[i]) + base;
  return slice.y;
}

template <typename DType, uint32_t kNumGPU>
__device__ __forceinline__ void two_shot_reduce_local(DType* (&input)[kNumGPU], uint32_t local_vecs) {
  using namespace device;
  using Trait = InklingAllReduceTrait<DType, kNumGPU>;
  using Storage = typename Trait::Storage;
  using DType2 = typename Trait::DType2;
  constexpr uint32_t kVecSize = Trait::kVecSize;
  const uint32_t stride = gridDim.x * blockDim.x;
  for (uint32_t v = blockIdx.x * blockDim.x + threadIdx.x; v < local_vecs; v += stride) {
    Storage s[kNumGPU];
#pragma unroll
    for (uint32_t i = 0; i < kNumGPU; ++i)
      s[i].load(input[i], v);
    Storage result;
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      fp32x2_t acc = cast<fp32x2_t>(s[0][j]);
#pragma unroll
      for (uint32_t i = 1; i < kNumGPU; ++i) {
        const fp32x2_t x = cast<fp32x2_t>(s[i][j]);
        acc.x += x.x;
        acc.y += x.y;
      }
      result[j] = cast<DType2>(acc);
    }
#pragma unroll
    for (uint32_t i = 0; i < kNumGPU; ++i)
      result.store(input[i], v);
  }
}


template <typename DType, uint32_t kNumGPU>
__global__ __launch_bounds__(1024, 1) void inkling_two_shot_all_reduce_kernel(
    void* const* __restrict__ peer_ptrs, const uint32_t rank, const uint32_t num_items) {
  DType* input[kNumGPU];
  const uint32_t local_vecs = slice_setup<DType, kNumGPU>(input, peer_ptrs, rank, num_items);
  two_shot_reduce_local<DType, kNumGPU>(input, local_vecs);
}




template <typename DType, uint32_t kNumGPU>
__global__ __launch_bounds__(1024, 1) void inkling_two_shot_all_reduce_fused_kernel(
    void* const* __restrict__ peer_ptrs,
    void* const* __restrict__ flag_ptrs,
    uint32_t* __restrict__ state,
    const DType* __restrict__ shared,
    const uint32_t rank,
    const uint32_t num_items) {
  DType* input[kNumGPU];
  const uint32_t local_vecs = slice_setup<DType, kNumGPU>(input, peer_ptrs, rank, num_items);
  if (shared != nullptr) {
    fold_shared_local<DType, kNumGPU>(static_cast<DType*>(peer_ptrs[rank]), shared, num_items);
  }

  inkling_ar::grid_system_barrier<kNumGPU>(state, flag_ptrs, rank, 0,                    shared != nullptr);
  two_shot_reduce_local<DType, kNumGPU>(input, local_vecs);
  inkling_ar::grid_system_barrier<kNumGPU>(
      state, flag_ptrs, rank, 1,                    true);
}














template <typename DType, uint32_t kNumGPU, bool kPerBlockBarrier>
__global__ __launch_bounds__(1024, 1) void inkling_multimem_one_shot_fused_kernel(
    DType* __restrict__ mc_ptr,
    DType* __restrict__ local_ptr,
    void* const* __restrict__ flag_ptrs,
    uint32_t* __restrict__ state,
    const DType* __restrict__ shared,
    const uint32_t rank,
    const uint32_t num_items) {
  using namespace device;
  using Trait = InklingAllReduceTrait<DType, kNumGPU>;
  static_assert(std::is_same_v<DType, bf16_t>, "multimem.add path is bf16-only");
  constexpr uint32_t kElemsPerVec = Trait::kElemsPerVec;

  const uint2 slice = rank_vec_slice<DType, kNumGPU>(rank, num_items);
  const uint32_t local_vecs = slice.y;
  DType* mc = mc_ptr + slice.x * kElemsPerVec;

  if (shared != nullptr) {



    fold_shared_local<DType, kNumGPU>(local_ptr, shared, num_items);
    inkling_ar::grid_system_barrier<kNumGPU>(state, flag_ptrs, rank, 0,                    true);
  } else if constexpr (kPerBlockBarrier) {
    inkling_ar::block_system_barrier<kNumGPU>(state, flag_ptrs, rank);
  } else {
    inkling_ar::grid_system_barrier<kNumGPU>(
        state, flag_ptrs, rank, 0,                    false);
  }
  const uint32_t stride = gridDim.x * blockDim.x;
  for (uint32_t v = blockIdx.x * blockDim.x + threadIdx.x; v < local_vecs; v += stride) {
    DType* addr = mc + v * kElemsPerVec;
    uint32_t r0, r1, r2, r3;

    asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "l"(addr));


    asm volatile(
        "multimem.st.relaxed.sys.global.v4.bf16x2 [%0], {%1,%2,%3,%4};" ::"l"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3)
        : "memory");
  }
  if constexpr (kPerBlockBarrier) {
    inkling_ar::block_system_barrier<kNumGPU>(state, flag_ptrs, rank);
  } else {
    inkling_ar::grid_system_barrier<kNumGPU>(
        state, flag_ptrs, rank, 1,                    true);
  }
}






















template <typename DType, uint32_t kNumGPU, bool kPerBlockBarrier>
__global__ __launch_bounds__(1024, 1) void inkling_multimem_push_oneshot_kernel(
    const DType* __restrict__ in_ptr,
    DType* __restrict__ mc_stage_ptr,
    const DType* __restrict__ stage_ptr,
    DType* __restrict__ out_ptr,
    void* const* __restrict__ flag_ptrs,
    uint32_t* __restrict__ state,
    const DType* __restrict__ shared,
    const uint32_t rank,
    const uint32_t num_items) {
  using namespace device;
  using Trait = InklingAllReduceTrait<DType, kNumGPU>;
  static_assert(std::is_same_v<DType, bf16_t>, "multimem path is bf16-only");
  constexpr uint32_t kElemsPerVec = Trait::kElemsPerVec;
  const uint32_t total_vec = num_items / kElemsPerVec;
  const uint32_t stride = gridDim.x * blockDim.x;






  DType* slot = mc_stage_ptr + rank * num_items;
  if (shared != nullptr) {
    using Storage = typename Trait::Storage;
    for (uint32_t v = blockIdx.x * blockDim.x + threadIdx.x; v < total_vec; v += stride) {
      Storage a, b;
      a.load(in_ptr, v);
      b.load(shared, v);
      const Storage s = add_vec_rn<DType>(a, b);
      const uint4 d = *reinterpret_cast<const uint4*>(&s);
      asm volatile("multimem.st.relaxed.sys.global.v4.bf16x2 [%0], {%1,%2,%3,%4};" ::"l"(slot + v * kElemsPerVec),
                   "r"(d.x),
                   "r"(d.y),
                   "r"(d.z),
                   "r"(d.w)
                   : "memory");
    }
  } else {
    for (uint32_t v = blockIdx.x * blockDim.x + threadIdx.x; v < total_vec; v += stride) {
      const uint4 d = *reinterpret_cast<const uint4*>(in_ptr + v * kElemsPerVec);
      asm volatile("multimem.st.relaxed.sys.global.v4.bf16x2 [%0], {%1,%2,%3,%4};" ::"l"(slot + v * kElemsPerVec),
                   "r"(d.x),
                   "r"(d.y),
                   "r"(d.z),
                   "r"(d.w)
                   : "memory");
    }
  }



  if constexpr (kPerBlockBarrier) {
    inkling_ar::block_system_barrier<kNumGPU>(state, flag_ptrs, rank);
  } else {
    inkling_ar::grid_system_barrier<kNumGPU>(state, flag_ptrs, rank, 0,                    true);
  }



  using Storage = typename Trait::Storage;
  using DType2 = typename Trait::DType2;
  constexpr uint32_t kVecSize = Trait::kVecSize;
  for (uint32_t v = blockIdx.x * blockDim.x + threadIdx.x; v < total_vec; v += stride) {
    Storage s[kNumGPU];
#pragma unroll
    for (uint32_t i = 0; i < kNumGPU; ++i)
      s[i].load(stage_ptr + i * num_items, v);
    Storage result;
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      fp32x2_t acc = cast<fp32x2_t>(s[0][j]);
#pragma unroll
      for (uint32_t i = 1; i < kNumGPU; ++i) {
        const fp32x2_t x = cast<fp32x2_t>(s[i][j]);
        acc.x += x.x;
        acc.y += x.y;
      }
      result[j] = cast<DType2>(acc);
    }
    result.store(out_ptr, v);
  }
}






template <typename DType, uint32_t kNumGPU>
__global__ __launch_bounds__(1024, 1) void inkling_multimem_full_oneshot_kernel(
    DType* __restrict__ mc_ptr,
    DType* __restrict__ local_in_ptr,
    DType* __restrict__ out_ptr,
    void* const* __restrict__ flag_ptrs,
    uint32_t* __restrict__ state,
    const DType* __restrict__ shared,
    const uint32_t rank,
    const uint32_t num_items) {
  using namespace device;
  using Trait = InklingAllReduceTrait<DType, kNumGPU>;
  static_assert(std::is_same_v<DType, bf16_t>, "multimem.add path is bf16-only");
  constexpr uint32_t kElemsPerVec = Trait::kElemsPerVec;
  const uint32_t total_vec = num_items / kElemsPerVec;

  if (shared != nullptr) {




    fold_shared_local<DType, kNumGPU>(local_in_ptr, shared, num_items);
  }
  inkling_ar::grid_system_barrier<kNumGPU>(
      state, flag_ptrs, rank, 0,                    shared != nullptr);
  const uint32_t stride = gridDim.x * blockDim.x;
  for (uint32_t v = blockIdx.x * blockDim.x + threadIdx.x; v < total_vec; v += stride) {
    DType* in = mc_ptr + v * kElemsPerVec;
    uint32_t r0, r1, r2, r3;
    asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "l"(in));

    *reinterpret_cast<uint4*>(out_ptr + v * kElemsPerVec) = make_uint4(r0, r1, r2, r3);
  }

}


template <typename DType, uint32_t kNumGPU>
uint32_t work_num_blocks(uint32_t n, uint32_t block_size) {
  using Trait = InklingAllReduceTrait<DType, kNumGPU>;
  const uint32_t total_vec = n / Trait::kElemsPerVec;
  const uint32_t vec_per_rank =
      host::div_ceil(host::div_ceil(total_vec, kNumGPU), device::kWarpThreads) * device::kWarpThreads;
  return max(1u, host::div_ceil(vec_per_rank, block_size));
}



template <typename DType>
uint32_t full_range_num_blocks(uint32_t n, uint32_t block_size) {
  constexpr uint32_t kElemsPerVec = InklingAllReduceTrait<DType, 2>::kElemsPerVec;
  return max(1u, host::div_ceil(n / kElemsPerVec, block_size));
}







template <typename Kernel>
uint32_t max_resident_blocks(Kernel kernel, uint32_t block_size, DLDevice device) {
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



template <typename DType>
const DType* shared_ptr_or_null(tvm::ffi::TensorView shared, int64_t num_items) {
  using namespace host;
  if (shared.numel() == 0) return nullptr;
  RuntimeCheck(shared.IsContiguous(), "shared must be contiguous");
  RuntimeCheck(is_type<DType>(shared.dtype()), "shared dtype mismatch");
  RuntimeCheck(shared.numel() >= num_items, "shared smaller than num_items");
  RuntimeCheck(std::bit_cast<intptr_t>(shared.data_ptr()) % 16 == 0, "shared not 16B aligned");
  return reinterpret_cast<const DType*>(shared.data_ptr());
}

template <typename DType, uint32_t kNumGPU>
void validate(tvm::ffi::TensorView buf, int64_t peer_ptrs_dev, int64_t rank, int64_t num_items, uint32_t& n) {
  using namespace host;
  using Trait = InklingAllReduceTrait<DType, kNumGPU>;
  n = static_cast<uint32_t>(num_items);
  RuntimeCheck(buf.IsContiguous(), "buffer must be contiguous");
  RuntimeCheck(buf.device().device_type == kDLCUDA, "buffer must be on a CUDA device");
  RuntimeCheck(is_type<DType>(buf.dtype()), "buffer dtype mismatch");
  RuntimeCheck(static_cast<int64_t>(n) == num_items, "num_items exceeds 4G");
  RuntimeCheck(buf.numel() >= num_items, "buffer smaller than num_items");
  RuntimeCheck(n % Trait::kElemsPerVec == 0, "num_items must be a multiple of ", Trait::kElemsPerVec);
  RuntimeCheck(std::bit_cast<intptr_t>(buf.data_ptr()) % 16 == 0, "buffer not 16B aligned");
  RuntimeCheck(peer_ptrs_dev != 0, "peer_ptrs_dev is null");
  RuntimeCheck(rank >= 0 && rank < kNumGPU, "rank out of range");
}

template <typename DType, uint32_t kNumGPU>
void inkling_two_shot_all_reduce(
    tvm::ffi::TensorView local_buffer, int64_t peer_ptrs_dev, int64_t rank, int64_t num_items) {
  using namespace host;
  uint32_t n;
  validate<DType, kNumGPU>(local_buffer, peer_ptrs_dev, rank, num_items, n);
  const auto device = local_buffer.device();
  const uint32_t num_blocks = work_num_blocks<DType, kNumGPU>(n, 1024u);
  const auto stream = LaunchKernel::resolve_device(device);
  LaunchKernel(num_blocks, 1024u, stream)(
      inkling_two_shot_all_reduce_kernel<DType, kNumGPU>,
      reinterpret_cast<void* const*>(peer_ptrs_dev),
      static_cast<uint32_t>(rank),
      n);
}

template <typename DType, uint32_t kNumGPU>
void inkling_two_shot_all_reduce_fused(
    tvm::ffi::TensorView local_buffer,
    int64_t data_ptrs_dev,
    int64_t flag_ptrs_dev,
    int64_t state_ptr,
    int64_t rank,
    int64_t num_items,
    int64_t nb_override,
    int64_t bs_override,
    tvm::ffi::TensorView shared) {
  using namespace host;
  uint32_t n;
  validate<DType, kNumGPU>(local_buffer, data_ptrs_dev, rank, num_items, n);
  RuntimeCheck(flag_ptrs_dev != 0, "flag_ptrs_dev is null");
  RuntimeCheck(state_ptr != 0, "state_ptr is null");
  const DType* shared_ptr = shared_ptr_or_null<DType>(shared, num_items);
  const auto device = local_buffer.device();
  const auto kernel = inkling_two_shot_all_reduce_fused_kernel<DType, kNumGPU>;
  const uint32_t block_size = bs_override > 0 ? static_cast<uint32_t>(bs_override) : 1024u;
  const uint32_t cap = max_resident_blocks(kernel, block_size, device);
  const uint32_t num_blocks = nb_override > 0 ? min(static_cast<uint32_t>(nb_override), cap)
                                              : min(work_num_blocks<DType, kNumGPU>(n, block_size), cap);
  const auto stream = LaunchKernel::resolve_device(device);
  LaunchKernel(num_blocks, block_size, stream)(
      kernel,
      reinterpret_cast<void* const*>(data_ptrs_dev),
      reinterpret_cast<void* const*>(flag_ptrs_dev),
      reinterpret_cast<uint32_t*>(state_ptr),
      shared_ptr,
      static_cast<uint32_t>(rank),
      n);
}

template <typename DType, uint32_t kNumGPU>
void inkling_multimem_one_shot_fused(
    tvm::ffi::TensorView local_buffer,
    int64_t multicast_ptr,
    int64_t flag_ptrs_dev,
    int64_t state_ptr,
    int64_t rank,
    int64_t num_items,
    int64_t nb_override,
    int64_t bs_override,
    int64_t per_block_barrier,
    tvm::ffi::TensorView shared) {
  using namespace host;
  uint32_t n;


  validate<DType, kNumGPU>(local_buffer, multicast_ptr, rank, num_items, n);
  RuntimeCheck(flag_ptrs_dev != 0, "flag_ptrs_dev is null");
  RuntimeCheck(state_ptr != 0, "state_ptr is null");
  RuntimeCheck(multicast_ptr % 16 == 0, "multicast_ptr not 16B aligned");
  const DType* shared_ptr = shared_ptr_or_null<DType>(shared, num_items);
  const auto device = local_buffer.device();
  const auto kernel = per_block_barrier ? inkling_multimem_one_shot_fused_kernel<DType, kNumGPU, true>
                                        : inkling_multimem_one_shot_fused_kernel<DType, kNumGPU, false>;
  const uint32_t block_size = bs_override > 0 ? static_cast<uint32_t>(bs_override) : 1024u;
  uint32_t cap = max_resident_blocks(kernel, block_size, device);
  if (per_block_barrier) cap = min(cap, inkling_ar::kMaxBarrierBlocks);
  const uint32_t num_blocks = nb_override > 0 ? min(static_cast<uint32_t>(nb_override), cap)
                                              : min(work_num_blocks<DType, kNumGPU>(n, block_size), cap);
  const auto stream = LaunchKernel::resolve_device(device);
  LaunchKernel(num_blocks, block_size, stream)(
      kernel,
      reinterpret_cast<DType*>(multicast_ptr),
      reinterpret_cast<DType*>(local_buffer.data_ptr()),
      reinterpret_cast<void* const*>(flag_ptrs_dev),
      reinterpret_cast<uint32_t*>(state_ptr),
      shared_ptr,
      static_cast<uint32_t>(rank),
      n);
}

template <typename DType, uint32_t kNumGPU>
void inkling_multimem_push_oneshot(
    tvm::ffi::TensorView in_buffer,
    tvm::ffi::TensorView out_buffer,
    int64_t mc_stage_ptr,
    int64_t local_stage_ptr,
    int64_t flag_ptrs_dev,
    int64_t state_ptr,
    int64_t rank,
    int64_t num_items,
    int64_t nb_override,
    int64_t bs_override,
    int64_t per_block_barrier,
    tvm::ffi::TensorView shared) {
  using namespace host;
  uint32_t n;



  validate<DType, kNumGPU>(in_buffer, mc_stage_ptr, rank, num_items, n);
  const DType* shared_ptr = shared_ptr_or_null<DType>(shared, num_items);
  RuntimeCheck(out_buffer.IsContiguous(), "out must be contiguous");
  RuntimeCheck(is_type<DType>(out_buffer.dtype()), "out dtype mismatch");
  RuntimeCheck(out_buffer.numel() >= num_items, "out smaller than num_items");
  RuntimeCheck(std::bit_cast<intptr_t>(out_buffer.data_ptr()) % 16 == 0, "out not 16B aligned");
  RuntimeCheck(flag_ptrs_dev != 0, "flag_ptrs_dev is null");
  RuntimeCheck(state_ptr != 0, "state_ptr is null");
  RuntimeCheck(local_stage_ptr != 0, "local_stage_ptr is null");
  RuntimeCheck(mc_stage_ptr % 16 == 0, "mc_stage_ptr not 16B aligned");
  RuntimeCheck(local_stage_ptr % 16 == 0, "local_stage_ptr not 16B aligned");
  const auto device = in_buffer.device();
  const auto kernel = per_block_barrier ? inkling_multimem_push_oneshot_kernel<DType, kNumGPU, true>
                                        : inkling_multimem_push_oneshot_kernel<DType, kNumGPU, false>;
  const uint32_t block_size = bs_override > 0 ? static_cast<uint32_t>(bs_override) : 1024u;
  uint32_t cap = max_resident_blocks(kernel, block_size, device);

  if (per_block_barrier) cap = min(cap, inkling_ar::kMaxBarrierBlocks);
  const uint32_t num_blocks = nb_override > 0 ? min(static_cast<uint32_t>(nb_override), cap)
                                              : min(full_range_num_blocks<DType>(n, block_size), cap);
  const auto stream = LaunchKernel::resolve_device(device);
  LaunchKernel(num_blocks, block_size, stream)(
      kernel,
      reinterpret_cast<const DType*>(in_buffer.data_ptr()),
      reinterpret_cast<DType*>(mc_stage_ptr),
      reinterpret_cast<const DType*>(local_stage_ptr),
      reinterpret_cast<DType*>(out_buffer.data_ptr()),
      reinterpret_cast<void* const*>(flag_ptrs_dev),
      reinterpret_cast<uint32_t*>(state_ptr),
      shared_ptr,
      static_cast<uint32_t>(rank),
      n);
}

template <typename DType, uint32_t kNumGPU>
void inkling_multimem_full_oneshot(
    tvm::ffi::TensorView in_buffer,
    tvm::ffi::TensorView out_buffer,
    int64_t multicast_ptr,
    int64_t flag_ptrs_dev,
    int64_t state_ptr,
    int64_t rank,
    int64_t num_items,
    int64_t nb_override,
    int64_t bs_override,
    tvm::ffi::TensorView shared) {
  using namespace host;
  uint32_t n;
  validate<DType, kNumGPU>(in_buffer, multicast_ptr, rank, num_items, n);
  RuntimeCheck(out_buffer.IsContiguous(), "out must be contiguous");
  RuntimeCheck(is_type<DType>(out_buffer.dtype()), "out dtype mismatch");
  RuntimeCheck(out_buffer.numel() >= num_items, "out smaller than num_items");
  RuntimeCheck(std::bit_cast<intptr_t>(out_buffer.data_ptr()) % 16 == 0, "out not 16B aligned");
  RuntimeCheck(flag_ptrs_dev != 0, "flag_ptrs_dev is null");
  RuntimeCheck(state_ptr != 0, "state_ptr is null");
  RuntimeCheck(multicast_ptr % 16 == 0, "multicast_ptr not 16B aligned");
  const DType* shared_ptr = shared_ptr_or_null<DType>(shared, num_items);
  const auto device = in_buffer.device();
  const auto kernel = inkling_multimem_full_oneshot_kernel<DType, kNumGPU>;
  const uint32_t block_size = bs_override > 0 ? static_cast<uint32_t>(bs_override) : 1024u;
  const uint32_t cap = max_resident_blocks(kernel, block_size, device);
  const uint32_t num_blocks = nb_override > 0 ? min(static_cast<uint32_t>(nb_override), cap)
                                              : min(full_range_num_blocks<DType>(n, block_size), cap);
  const auto stream = LaunchKernel::resolve_device(device);
  LaunchKernel(num_blocks, block_size, stream)(
      kernel,
      reinterpret_cast<DType*>(multicast_ptr),
      reinterpret_cast<DType*>(in_buffer.data_ptr()),
      reinterpret_cast<DType*>(out_buffer.data_ptr()),
      reinterpret_cast<void* const*>(flag_ptrs_dev),
      reinterpret_cast<uint32_t*>(state_ptr),
      shared_ptr,
      static_cast<uint32_t>(rank),
      n);
}

}
