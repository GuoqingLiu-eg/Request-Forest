













#pragma once

#include <cstdint>

namespace sglang {

namespace inkling_ar {

constexpr uint32_t kLeaderStateWords = 8;
constexpr uint32_t kMaxBarrierBlocks = 256;
















template <uint32_t kNumGPU>
__device__ __forceinline__ void grid_system_barrier(
    uint32_t* __restrict__ st, void* const* __restrict__ flag_ptrs, uint32_t rank, uint32_t idx, bool publish_writes) {













  uint32_t* xepoch = st + 4;
  __shared__ uint32_t s_e;
  __shared__ uint32_t s_prev;
  __shared__ int s_leader;
  const bool solo = (gridDim.x == 1u);
  __syncthreads();
  if (threadIdx.x == 0) {
    if (solo) {
      s_leader = 1;
    } else {
      if (publish_writes) __threadfence_system();
      s_prev = *static_cast<volatile uint32_t*>(st + 2 + idx);





      uint32_t old;
      asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                   : "=r"(old)
                   : "l"(st + idx), "r"(gridDim.x - 1u)
                   : "memory");
      s_leader = (old == gridDim.x - 1u) ? 1 : 0;
    }
  }
  __syncthreads();
  if (s_leader) {
    if (threadIdx.x == 0) {
      const uint32_t e = *xepoch + 1u;
      *xepoch = e;
      s_e = e;
    }
    __syncthreads();
    const uint32_t e = s_e;






    if (threadIdx.x < kNumGPU) {
      const uint32_t peer = threadIdx.x;
      uint32_t* remote = static_cast<uint32_t*>(flag_ptrs[peer]) + rank;
      asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(remote), "r"(e) : "memory");
      uint32_t* mine = static_cast<uint32_t*>(flag_ptrs[rank]) + peer;
      uint32_t got;
      do {
        asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(got) : "l"(mine) : "memory");
      } while (static_cast<int32_t>(got - e) < 0);
    }
    __syncthreads();
    if (!solo && threadIdx.x == 0) {



      asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(st + 2 + idx), "r"(1u) : "memory");
    }
  } else {
    if (threadIdx.x == 0) {

      uint32_t* rel = st + 2 + idx;
      uint32_t got;
      do {
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(got) : "l"(rel) : "memory");
      } while (static_cast<int32_t>(got - s_prev) <= 0);
    }
    __syncthreads();
  }
}








__device__ __forceinline__ void grid_local_sync(uint32_t* __restrict__ st) {
  __syncthreads();
  if (gridDim.x > 1u) {
    if (threadIdx.x == 0) {
      uint32_t* arrive = st + 5;
      uint32_t* release = st + 6;
      const uint32_t prev = *static_cast<volatile uint32_t*>(release);
      uint32_t old;
      asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                   : "=r"(old)
                   : "l"(arrive), "r"(gridDim.x - 1u)
                   : "memory");
      if (old == gridDim.x - 1u) {
        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(release), "r"(1u) : "memory");
      } else {
        uint32_t got;
        do {
          asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(got) : "l"(release) : "memory");
        } while (static_cast<int32_t>(got - prev) <= 0);
      }
    }
    __syncthreads();
  }
}











template <uint32_t kNumGPU>
__device__ __forceinline__ void
block_system_barrier(uint32_t* __restrict__ st, void* const* __restrict__ flag_ptrs, uint32_t rank) {
  __shared__ uint32_t s_e;
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t* epoch = st + kLeaderStateWords + blockIdx.x;
    const uint32_t e = *epoch + 1u;
    *epoch = e;
    s_e = e;
  }
  __syncthreads();
  const uint32_t e = s_e;
  if (threadIdx.x < kNumGPU) {
    const uint32_t peer = threadIdx.x;
    uint32_t* remote = static_cast<uint32_t*>(flag_ptrs[peer]) + kNumGPU + rank * kMaxBarrierBlocks + blockIdx.x;
    asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(remote), "r"(e) : "memory");
    uint32_t* mine = static_cast<uint32_t*>(flag_ptrs[rank]) + kNumGPU + peer * kMaxBarrierBlocks + blockIdx.x;
    uint32_t got;
    do {
      asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(got) : "l"(mine) : "memory");
    } while (static_cast<int32_t>(got - e) < 0);
  }
  __syncthreads();
}

}

}
