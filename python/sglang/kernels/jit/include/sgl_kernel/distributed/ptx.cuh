#pragma once



















#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <cstdint>
#include <type_traits>

namespace sglang {

namespace device::ptx {









template <typename V>
SGL_DEVICE void ld_global_16B(V& x, const void* addr, int64_t vec_offset) {
  static_assert(alignof(V) == 16 && sizeof(V) == 16);
  addr = static_cast<const uint8_t*>(addr) + vec_offset * sizeof(V);
  uint4 val;
  asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
               : "l"(addr));
  x = *reinterpret_cast<const V*>(&val);
}

template <typename V>
SGL_DEVICE void st_global_16B(const V& x, void* addr, int64_t vec_offset) {
  static_assert(alignof(V) == 16 && sizeof(V) == 16);
  const uint4 val = *reinterpret_cast<const uint4*>(&x);
  addr = static_cast<uint8_t*>(addr) + vec_offset * sizeof(V);
  asm volatile("st.global.v4.b32 [%4], {%0, %1, %2, %3};"
               :
               : "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w), "l"(addr));
}








template <typename V>
SGL_DEVICE void ld_relaxed_16B(V& x, const void* addr, int64_t vec_offset) {
  static_assert(alignof(V) == 16 && sizeof(V) == 16);
  addr = static_cast<const uint8_t*>(addr) + vec_offset * sizeof(V);
  uint4 val;
  asm volatile("ld.relaxed.sys.global.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
               : "l"(addr));
  x = *reinterpret_cast<const V*>(&val);
}

template <typename V>
SGL_DEVICE void st_relaxed_16B(const V& x, void* addr, int64_t vec_offset) {
  static_assert(alignof(V) == 16 && sizeof(V) == 16);
  const uint4 val = *reinterpret_cast<const uint4*>(&x);
  addr = static_cast<uint8_t*>(addr) + vec_offset * sizeof(V);
  asm volatile("st.relaxed.sys.global.v4.b32 [%4], {%0, %1, %2, %3};"
               :
               : "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w), "l"(addr));
}



template <typename V>
SGL_DEVICE void ld_multimem_16B(V& x, const void* mc_addr, int64_t vec_offset) {
#if SGL_ARCH_HOPPER_OR_GREATER
  static_assert(alignof(V) == 16 && sizeof(V) == 16);
  mc_addr = static_cast<const uint8_t*>(mc_addr) + vec_offset * 16;
  if constexpr (std::is_same_v<V, device::AlignedVector<fp32x2_t, 2>>) {
    float4 val;
    asm volatile("multimem.ld_reduce.weak.add.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w)
                 : "l"(mc_addr));
    x = *reinterpret_cast<const V*>(&val);
  } else {



    uint4 val;
    if constexpr (std::is_same_v<V, device::AlignedVector<fp16x2_t, 4>>) {
      asm volatile("multimem.ld_reduce.weak.add.acc::f32.v4.f16x2 {%0, %1, %2, %3}, [%4];"
                   : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
                   : "l"(mc_addr));
    } else {
      static_assert(std::is_same_v<V, device::AlignedVector<bf16x2_t, 4>>);
      asm volatile("multimem.ld_reduce.weak.add.acc::f32.v4.bf16x2 {%0, %1, %2, %3}, [%4];"
                   : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
                   : "l"(mc_addr));
    }
    x = *reinterpret_cast<const V*>(&val);
  }
#else
  assert(false && "multimem load is only supported on Hopper or later architecture");
#endif
}




template <typename V>
SGL_DEVICE void ld_relaxed_8B(V& x, const void* addr, int64_t vec_offset) {
  static_assert(alignof(V) == 8 && sizeof(V) == 8);
  addr = static_cast<const uint8_t*>(addr) + vec_offset * sizeof(V);
  uint2 val;
  asm volatile("ld.relaxed.sys.global.v2.b32 {%0, %1}, [%2];" : "=r"(val.x), "=r"(val.y) : "l"(addr) : "memory");
  x = *reinterpret_cast<const V*>(&val);
}

template <typename V>
SGL_DEVICE void st_relaxed_8B(const V& x, void* addr, int64_t vec_offset) {
  static_assert(alignof(V) == 8 && sizeof(V) == 8);
  const uint2 val = *reinterpret_cast<const uint2*>(&x);
  addr = static_cast<uint8_t*>(addr) + vec_offset * sizeof(V);
  asm volatile("st.relaxed.sys.global.v2.b32 [%2], {%0, %1};" ::"r"(val.x), "r"(val.y), "l"(addr) : "memory");
}


template <typename V>
SGL_DEVICE void st_multimem_16B(const V& x, void* mc_addr, int64_t vec_offset) {
#if SGL_ARCH_HOPPER_OR_GREATER
  static_assert(alignof(V) == 16 && sizeof(V) == 16);
  const auto val = *reinterpret_cast<const float4*>(&x);
  mc_addr = static_cast<uint8_t*>(mc_addr) + vec_offset * 16;
  asm volatile("multimem.st.weak.v4.f32 [%4], {%0, %1, %2, %3};"
               :
               : "f"(val.x), "f"(val.y), "f"(val.z), "f"(val.w), "l"(mc_addr));
#else
  assert(false && "multimem store is only supported on Hopper or later architecture");
#endif
}









SGL_DEVICE uint32_t load_relaxed_sys(const uint32_t* ptr) {
  uint32_t val;
  asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(val) : "l"(ptr) : "memory");
  return val;
}



SGL_DEVICE uint32_t load_acquire_sys(const uint32_t* ptr) {
  uint32_t val;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(val) : "l"(ptr) : "memory");
  return val;
}



SGL_DEVICE void red_add_relaxed_sys(uint32_t* ptr, uint32_t val) {
  asm volatile("red.relaxed.sys.global.add.u32 [%0], %1;" : : "l"(ptr), "r"(val) : "memory");
}


SGL_DEVICE void red_add_release_sys(uint32_t* ptr, uint32_t val) {
  asm volatile("red.release.sys.global.add.u32 [%0], %1;" : : "l"(ptr), "r"(val) : "memory");
}



SGL_DEVICE void fence_release_sys() {
  asm volatile("fence.release.sys;" ::: "memory");
}



SGL_DEVICE uint32_t atomic_add_acq_rel_gpu(uint32_t* ptr, uint32_t val) {
  uint32_t old;
  asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(val) : "memory");
  return old;
}








SGL_DEVICE void multimem_store_relaxed(uint32_t* mc_ptr, uint32_t val) {
  asm volatile("multimem.st.relaxed.sys.global.b32 [%0], %1;" : : "l"(mc_ptr), "r"(val) : "memory");
}

SGL_DEVICE void multimem_red_add_relaxed(uint32_t* mc_ptr, uint32_t val) {
#if SGL_ARCH_HOPPER_OR_GREATER
  asm volatile("multimem.red.relaxed.sys.global.add.u32 [%0], %1;" : : "l"(mc_ptr), "r"(val) : "memory");
#else
  assert(false && "multimem red is only supported on Hopper or later architecture");
#endif
}

SGL_DEVICE void multimem_red_add_release(uint32_t* mc_ptr, uint32_t val) {
#if SGL_ARCH_HOPPER_OR_GREATER
  asm volatile("multimem.red.release.sys.global.add.u32 [%0], %1;" : : "l"(mc_ptr), "r"(val) : "memory");
#else
  assert(false && "multimem red is only supported on Hopper or later architecture");
#endif
}

}

}
