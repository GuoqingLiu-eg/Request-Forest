






#pragma once
#include <sgl_kernel/type.cuh>

namespace sglang {

namespace device::math {


inline constexpr float log2e = 1.44269504088896340736f;

inline constexpr float loge2 = 0.693147180559945309417f;





inline constexpr float FP8_E4M3_MAX = kFP8E4M3Max;
static_assert(log2e * loge2 == 1.0f, "log2e * loge2 must be 1");


template <typename T>
SGL_DEVICE T max(T a, T b) {
  return DTypeTrait<T>::max(a, b);
}


template <typename T>
SGL_DEVICE T min(T a, T b) {
  return DTypeTrait<T>::min(a, b);
}


template <typename T>
SGL_DEVICE T abs(T a) {
  return DTypeTrait<T>::abs(a);
}


template <typename T>
SGL_DEVICE T sqrt(T a) {
  return DTypeTrait<T>::sqrt(a);
}


template <typename T>
SGL_DEVICE T rsqrt(T a) {
  return DTypeTrait<T>::rsqrt(a);
}


template <typename T>
SGL_DEVICE T exp(T a) {
  return DTypeTrait<T>::exp(a);
}


SGL_DEVICE float sigmoid_fast(float x) {
  return 1.0f / (1.0f + __expf(-x));
}


SGL_DEVICE float silu_fast(float x) {
  return x * sigmoid_fast(x);
}





SGL_DEVICE float softplus_fast(float x) {
  return x > 20.0f ? x : log1pf(__expf(x));
}


template <typename T>
SGL_DEVICE T sin(T a) {
  return DTypeTrait<T>::sin(a);
}


template <typename T>
SGL_DEVICE T cos(T a) {
  return DTypeTrait<T>::cos(a);
}





SGL_DEVICE float fma_f32_bf16(bf16_t a, bf16_t b, float acc) {
#if SGL_ARCH_BLACKWELL_OR_GREATER
  const uint16_t a_bits = __bfloat16_as_ushort(a);
  const uint16_t b_bits = __bfloat16_as_ushort(b);
  float result;
  asm("fma.rn.f32.bf16 %0, %1, %2, %3;" : "=f"(result) : "h"(a_bits), "h"(b_bits), "f"(acc));
  return result;
#else
  return fmaf(cast<fp32_t>(a), cast<fp32_t>(b), acc);
#endif
}

}

}
