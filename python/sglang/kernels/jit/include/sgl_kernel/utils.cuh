














#pragma once

#include <sgl_kernel/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <concepts>
#include <cstddef>
#include <type_traits>
#ifndef USE_ROCM
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#else
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#ifndef __grid_constant__
#define __grid_constant__
#endif
using cudaError_t = hipError_t;
using cudaStream_t = hipStream_t;
using cudaLaunchConfig_t = hipLaunchConfig_t;
using cudaLaunchAttribute = hipLaunchAttribute;
inline constexpr auto cudaSuccess = hipSuccess;
#define cudaStreamPerThread hipStreamPerThread
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaLaunchKernel hipLaunchKernel
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMinor hipDeviceAttributeComputeCapabilityMinor
#endif

namespace sglang {

#ifndef USE_ROCM
using fp32_t = float;
using fp16_t = __half;
using bf16_t = __nv_bfloat16;
using fp8_e4m3_t = __nv_fp8_e4m3;
using fp8_e5m2_t = __nv_fp8_e5m2;

using fp32x2_t = float2;
using fp16x2_t = __half2;
using bf16x2_t = __nv_bfloat162;
using fp8x2_e4m3_t = __nv_fp8x2_e4m3;
using fp8x2_e5m2_t = __nv_fp8x2_e5m2;
using fp8x4_e4m3_t = __nv_fp8x4_e4m3;
using fp8x4_e5m2_t = __nv_fp8x4_e5m2;

using fp32x4_t = float4;
#else
using fp32_t = float;
using fp16_t = __half;
using bf16_t = __hip_bfloat16;
using fp8_e4m3_t = uint8_t;
using fp8_e5m2_t = uint8_t;
using fp32x2_t = float2;
using fp16x2_t = half2;
using bf16x2_t = __hip_bfloat162;
using fp8x2_e4m3_t = uint16_t;
using fp8x2_e5m2_t = uint16_t;
using fp8x4_e4m3_t = uint32_t;
using fp8x4_e5m2_t = uint32_t;
using fp32x4_t = float4;
#endif




#ifndef USE_ROCM
#define SGLANG_LDG(arg) __ldg(arg)
#else
#define SGLANG_LDG(arg) *(arg)
#endif


#ifndef USE_ROCM
inline constexpr auto kDLGPU = kDLCUDA;
inline constexpr auto kDLGPUHost = kDLCUDAHost;
#else
inline constexpr auto kDLGPU = kDLROCM;
inline constexpr auto kDLGPUHost = kDLROCMHost;
#endif

namespace device {


#define SGL_DEVICE __forceinline__ __device__




#if !defined(USE_ROCM)
#if !defined(SGL_CUDA_ARCH)
#error "SGL_CUDA_ARCH is not defined. JIT compilation must inject -DSGL_CUDA_ARCH via load_jit()."
#endif
#if defined(__CUDA_ARCH__)
static_assert(
    __CUDA_ARCH__ == SGL_CUDA_ARCH, "SGL_CUDA_ARCH mismatch: injected arch flag does not match device target");
#endif
#define SGL_ARCH_HOPPER_OR_GREATER (SGL_CUDA_ARCH >= 900)
#define SGL_ARCH_BLACKWELL_OR_GREATER ((SGL_CUDA_ARCH >= 1000) && (CUDA_VERSION >= 12090))
#else
#define SGL_ARCH_HOPPER_OR_GREATER 0
#define SGL_ARCH_BLACKWELL_OR_GREATER 0
#endif




inline constexpr std::size_t kMaxVecBytes = SGL_ARCH_BLACKWELL_OR_GREATER ? 32 : 16;


inline constexpr auto kWarpThreads = 32u;

#ifndef USE_ROCM
inline constexpr auto kFullMask = 0xffffffffu;
#else
inline constexpr auto kFullMask = 0xffffffffffffffffULL;
#endif








template <bool kUsePDL>
SGL_DEVICE void PDLWaitPrimary() {
#if SGL_ARCH_HOPPER_OR_GREATER
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
#endif
}







template <bool kUsePDL>
SGL_DEVICE void PDLTriggerSecondary() {
#if SGL_ARCH_HOPPER_OR_GREATER
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.launch_dependents;" :::);
  }
#endif
}

template <std::integral T, std::integral U>
SGL_DEVICE constexpr auto div_ceil(T a, U b) {
  return (a + b - 1) / b;
}







template <typename T>
SGL_DEVICE T load_as(const void* ptr, int64_t offset = 0) {
  return static_cast<const T*>(ptr)[offset];
}










template <typename T>
SGL_DEVICE void store_as(void* ptr, std::type_identity_t<T> val, int64_t offset = 0) {
  static_cast<T*>(ptr)[offset] = val;
}


namespace pointer {



template <typename T = char, std::integral... U>
SGL_DEVICE auto offset(void* ptr, U... offset) -> void* {
  return static_cast<T*>(ptr) + (... + offset);
}

template <typename T = char, std::integral... U>
SGL_DEVICE auto offset(const void* ptr, U... offset) -> const void* {
  return static_cast<const T*>(ptr) + (... + offset);
}

}


SGL_DEVICE void enable_smem_spilling() {
#if defined(__CUDA_ARCH__) && CUDART_VERSION >= 13000
  asm(".pragma \"enable_smem_spilling\";");
#endif
}

}

namespace host {




inline void RuntimeDeviceCheck(::cudaError_t error, DebugInfo location = {}) {
  if (error != ::cudaSuccess) {
    [[unlikely]];
    host::panic(location, "CUDA error: ", ::cudaGetErrorString(error));
  }
}


inline void RuntimeDeviceCheck(DebugInfo location = {}) {
  return RuntimeDeviceCheck(::cudaGetLastError(), location);
}















struct LaunchKernel {
 private:
  struct KernelConfig {
    bool use_pdl = false;
    std::optional<dim3> cluster_dim = std::nullopt;
  };

 public:
  explicit LaunchKernel(
      dim3 grid_dim,
      dim3 block_dim,
      DLDevice device,
      std::size_t dynamic_shared_mem_bytes = 0,
      DebugInfo location = {}) noexcept
      : m_config(s_make_config(grid_dim, block_dim, resolve_device(device), dynamic_shared_mem_bytes)),
        m_location(location) {}

  explicit LaunchKernel(
      dim3 grid_dim,
      dim3 block_dim,
      cudaStream_t stream,
      std::size_t dynamic_shared_mem_bytes = 0,
      DebugInfo location = {}) noexcept
      : m_config(s_make_config(grid_dim, block_dim, stream, dynamic_shared_mem_bytes)), m_location(location) {}

  LaunchKernel(const LaunchKernel&) = delete;
  LaunchKernel& operator=(const LaunchKernel&) = delete;

  static auto resolve_device(DLDevice device) -> cudaStream_t {
    return static_cast<cudaStream_t>(::TVMFFIEnvGetStream(device.device_type, device.device_id));
  }

  auto enable_pdl(bool enabled = true) -> LaunchKernel& {
#ifdef USE_ROCM
    (void)enabled;
    m_config.numAttrs = 0;
#else
    if (enabled) {
      auto& attr = m_attrs[m_config.numAttrs++];
      attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr.val.programmaticStreamSerializationAllowed = true;
      m_config.attrs = m_attrs;
    }
#endif
    return *this;
  }

  auto enable_cluster(dim3 cluster_dim) -> LaunchKernel& {
#ifdef USE_ROCM
    (void)cluster_dim;
#else
    auto& attr = m_attrs[m_config.numAttrs++];
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {cluster_dim.x, cluster_dim.y, cluster_dim.z};
    m_config.attrs = m_attrs;
#endif
    return *this;
  }









  auto config(const KernelConfig& config) -> LaunchKernel& {
    if (config.use_pdl) this->enable_pdl(true);
    if (config.cluster_dim) this->enable_cluster(*config.cluster_dim);
    return *this;
  }

  template <typename T, typename... Args>
  auto operator()(T&& kernel, Args&&... args) const -> void {
#ifdef USE_ROCM
    hipLaunchKernelGGL(
        std::forward<T>(kernel),
        m_config.gridDim,
        m_config.blockDim,
        m_config.dynamicSmemBytes,
        m_config.stream,
        std::forward<Args>(args)...);
    RuntimeDeviceCheck(m_location);
#else
    RuntimeDeviceCheck(::cudaLaunchKernelEx(&m_config, kernel, std::forward<Args>(args)...), m_location);
#endif
  }

  template <typename T, typename... Args>
  auto launch(T&& kernel, Args&&... args) const -> void {
    return (*this)(std::forward<T>(kernel), std::forward<Args>(args)...);
  }

 private:
  static auto s_make_config(
      dim3 grid_dim,
      dim3 block_dim,
      cudaStream_t stream,
      std::size_t smem) -> cudaLaunchConfig_t {
    auto config = ::cudaLaunchConfig_t{};
    config.gridDim = grid_dim;
    config.blockDim = block_dim;
    config.dynamicSmemBytes = smem;
    config.stream = stream;
    config.numAttrs = 0;
    return config;
  }

  cudaLaunchConfig_t m_config;
  const DebugInfo m_location;
  cudaLaunchAttribute m_attrs[2];
};



#define CHECK_CUDA(COND)                                              \
  if (const auto error = (COND); error == ::cudaSuccess) [[likely]] { \
  } else                                                              \
    host::Error() << "CUDA error: " << ::cudaGetErrorString(error) << ". "

}

}
