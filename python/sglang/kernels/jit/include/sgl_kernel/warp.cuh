


#pragma once
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <cstdint>
#include <type_traits>

namespace sglang {

namespace device::warp {


#ifndef USE_ROCM
static constexpr uint32_t kFullMask = 0xffffffffu;
using mask_t = uint32_t;
#else
static constexpr uint64_t kFullMask = 0xffffffffffffffffULL;
using mask_t = uint64_t;
#endif




















template <ReductionOp OP, uint32_t kNumThreads = kWarpThreads, bool kInner = true, typename T>
SGL_DEVICE T reduce(T value, mask_t active_mask = kFullMask) {
  static_assert(kNumThreads >= 1 && kNumThreads <= kWarpThreads);
  static_assert(std::has_single_bit(kNumThreads), "must be pow of 2");
  using Trait = ReductionTrait<OP, T>;

#ifdef SGL_CUDA_ARCH

  constexpr bool kFullReduction = (kNumThreads == kWarpThreads && kInner) || (kNumThreads == 1 && !kInner);
  if constexpr (kFullReduction) {
#if SGL_CUDA_ARCH >= 800

    if constexpr (std::is_integral_v<T> && sizeof(T) <= 4) {
      if constexpr (OP == ReductionOp::SUM) {
        return __reduce_add_sync(active_mask, value);
      } else if constexpr (OP == ReductionOp::MAX) {
        return __reduce_max_sync(active_mask, value);
      } else if constexpr (OP == ReductionOp::MIN) {
        return __reduce_min_sync(active_mask, value);
      }
    }
#endif
#if SGL_CUDA_ARCH >= 1000 && SGL_CUDA_ARCH < 1100

    if constexpr (std::is_same_v<T, float>) {
      if constexpr (OP == ReductionOp::MAX) {
        float result;
        asm("redux.sync.max.f32 %0, %1, %2;" : "=f"(result) : "f"(value), "r"(active_mask));
        return result;
      } else if constexpr (OP == ReductionOp::MIN) {
        float result;
        asm("redux.sync.min.f32 %0, %1, %2;" : "=f"(result) : "f"(value), "r"(active_mask));
        return result;
      }
    }
#endif
  }
#endif

  if constexpr (kInner) {
#pragma unroll
    for (uint32_t mask = kNumThreads / 2; mask >= 1; mask >>= 1) {
#ifndef USE_ROCM
      value = Trait::reduce(value, __shfl_xor_sync(active_mask, value, mask, 32));
#else
      value = Trait::reduce(value, __shfl_xor(value, mask, kNumThreads));
#endif
    }
  } else {
#pragma unroll
    for (uint32_t mask = kNumThreads; mask <= kWarpThreads / 2; mask <<= 1) {
#ifndef USE_ROCM
      value = Trait::reduce(value, __shfl_xor_sync(active_mask, value, mask, 32));
#else


      value = Trait::reduce(value, __shfl_xor(value, mask, kWarpThreads));
#endif
    }
  }
  return value;
}


template <uint32_t kNumThreads = kWarpThreads, bool kInner = true, typename T>
SGL_DEVICE T reduce_sum(T value, mask_t active_mask = kFullMask) {
  return reduce<ReductionOp::SUM, kNumThreads, kInner>(value, active_mask);
}


template <uint32_t kNumThreads = kWarpThreads, bool kInner = true, typename T>
SGL_DEVICE T reduce_max(T value, mask_t active_mask = kFullMask) {
  return reduce<ReductionOp::MAX, kNumThreads, kInner>(value, active_mask);
}


template <uint32_t kNumThreads = kWarpThreads, bool kInner = true, typename T>
SGL_DEVICE T reduce_min(T value, mask_t active_mask = kFullMask) {
  return reduce<ReductionOp::MIN, kNumThreads, kInner>(value, active_mask);
}

























template <int64_t kBytes>
SGL_DEVICE void copy_bytes(const void* __restrict__ src, void* __restrict__ dst) {
  constexpr int64_t kAlignment = (kBytes % (16 * kWarpThreads) == 0)  ? 16
                                 : (kBytes % (8 * kWarpThreads) == 0) ? 8
                                 : (kBytes % (4 * kWarpThreads) == 0) ? 4
                                 : (kBytes % 4 == 0)                  ? 4
                                                                      : 0;
  static_assert(kAlignment > 0, "kBytes must be a multiple of 4");

  using vec_t = AlignedStorage<uint32_t, kAlignment / 4>;
  constexpr auto kLoopBytes = sizeof(vec_t) * kWarpThreads;
  constexpr auto kLoopCount = kBytes / kLoopBytes;
  constexpr int64_t kTailVecs = (kBytes - kLoopCount * kLoopBytes) / sizeof(vec_t);

  const auto gmem = tile::Memory<vec_t>::warp();

#pragma unroll
  for (int64_t i = 0; i < kLoopCount; ++i) {
    const auto v = gmem.load(src, i);
    gmem.store(dst, v, i);
  }
  if constexpr (kTailVecs > 0) {
    if (gmem.in_bound(kLoopCount * kWarpThreads + kTailVecs, kLoopCount)) {
      const auto v = gmem.load(src, kLoopCount);
      gmem.store(dst, v, kLoopCount);
    }
  }
}




SGL_DEVICE uint32_t inclusive_sum(uint32_t lane_id, uint32_t val) {
  static_assert(kWarpThreads == 32);
#pragma unroll
  for (uint32_t offset = 1; offset < 32; offset *= 2) {
#ifndef USE_ROCM
    uint32_t n = __shfl_up_sync(0xFFFFFFFF, val, offset);
#else
    uint32_t n = __shfl_up_sync(kFullMask, val, offset, kWarpThreads);
#endif
    if (lane_id >= offset) val += n;
  }
  return val;
}





SGL_DEVICE bool elect_one_lane() {
  uint32_t pred;
  asm volatile(
      "{\n"
      "  .reg .pred p;\n"
      "  .reg .b32  r;\n"
      "  elect.sync r|p, 0xFFFFFFFF;\n"
      "  selp.b32 %0, 1, 0, p;\n"
      "}\n"
      : "=r"(pred));
  return pred != 0;
}

}

}
