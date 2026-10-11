



















#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

template <int NC, int NV, int NUM_SINGLE, int NUM_RED_LOG, int NUM_GPUS, int BLOCK_DIM>
__global__ void lp_prep_kernel(
    float* __restrict__ A_full,
    float* __restrict__ b,
    float* __restrict__ t1,
    const float* __restrict__ global_counts,
    const int64_t* __restrict__ log_single,
    const int64_t* __restrict__ log_replicated,
    const float* __restrict__ B1,
    const float* __restrict__ A_base_row_sum) {
  static_assert(NC == NUM_RED_LOG + NUM_GPUS, "NC must equal NUM_RED_LOG + NUM_GPUS");
  constexpr int WARPS_PER_BLOCK = BLOCK_DIM / 32;

  extern __shared__ unsigned char raw_smem[];

  float* shared_t1 = reinterpret_cast<float*>(raw_smem);
  float* shared_b1 = shared_t1 + NUM_SINGLE;
  float* reduce_buf = shared_b1 + NUM_RED_LOG;
  float* shared_total = reduce_buf + WARPS_PER_BLOCK;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;


  float local_sum = 0.f;
  for (int i = tid; i < NUM_SINGLE; i += BLOCK_DIM) {
    float v = global_counts[log_single[i]];
    shared_t1[i] = v;
    local_sum += v;
  }
  for (int i = tid; i < NUM_RED_LOG; i += BLOCK_DIM) {
    float v = global_counts[log_replicated[i]];
    shared_b1[i] = v;
    local_sum += v;
  }


  for (int offset = 16; offset > 0; offset >>= 1) {
    local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
  }
  if (lane == 0) {
    reduce_buf[warp_id] = local_sum;
  }
  __syncthreads();
  if (warp_id == 0) {
    float v = (tid < WARPS_PER_BLOCK) ? reduce_buf[tid] : 0.f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      v += __shfl_xor_sync(0xffffffff, v, offset);
    }
    if (tid == 0) {
      shared_total[0] = fmaxf(v, 1.0f);
    }
  }
  __syncthreads();
  const float inv_total = 1.0f / shared_total[0];



  for (int i = tid; i < NUM_SINGLE; i += BLOCK_DIM) {
    float v = shared_t1[i] * inv_total;
    shared_t1[i] = v;
    t1[i] = v;
  }
  for (int i = tid; i < NUM_RED_LOG; i += BLOCK_DIM) {
    b[i] = shared_b1[i] * inv_total;
  }
  __syncthreads();




  for (int j = 0; j < NUM_GPUS; j++) {
    float dot = 0.f;
    for (int k = tid; k < NUM_SINGLE; k += BLOCK_DIM) {
      dot += B1[j * NUM_SINGLE + k] * shared_t1[k];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      dot += __shfl_xor_sync(0xffffffff, dot, offset);
    }
    if (lane == 0) {
      reduce_buf[warp_id] = dot;
    }
    __syncthreads();
    if (warp_id == 0) {
      float v = (tid < WARPS_PER_BLOCK) ? reduce_buf[tid] : 0.f;
      for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, offset);
      }
      if (tid == 0) {
        b[NUM_RED_LOG + j] = -v;
      }
    }
    __syncthreads();
  }




  for (int i = tid; i < NC; i += BLOCK_DIM) {
    A_full[i * NV + (NV - 1)] = b[i] - A_base_row_sum[i];
  }
}

template <int NC, int NV, int NUM_SINGLE, int NUM_RED_LOG, int NUM_GPUS, int BLOCK_DIM>
void lp_prep(
    tvm::ffi::TensorView A_full,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView t1,
    tvm::ffi::TensorView global_counts,
    tvm::ffi::TensorView log_single,
    tvm::ffi::TensorView log_replicated,
    tvm::ffi::TensorView B1,
    tvm::ffi::TensorView A_base_row_sum) {
  using namespace host;

  SymbolicDevice device_;
  TensorMatcher({NC, NV}).with_dtype<float>().with_device<kDLCUDA>(device_).verify(A_full);
  TensorMatcher({NC}).with_dtype<float>().with_device<kDLCUDA>(device_).verify(b);
  TensorMatcher({NUM_SINGLE}).with_dtype<float>().with_device<kDLCUDA>(device_).verify(t1);
  TensorMatcher({NUM_SINGLE}).with_dtype<int64_t>().with_device<kDLCUDA>(device_).verify(log_single);
  TensorMatcher({NUM_RED_LOG}).with_dtype<int64_t>().with_device<kDLCUDA>(device_).verify(log_replicated);
  TensorMatcher({NUM_GPUS, NUM_SINGLE}).with_dtype<float>().with_device<kDLCUDA>(device_).verify(B1);
  TensorMatcher({NC}).with_dtype<float>().with_device<kDLCUDA>(device_).verify(A_base_row_sum);

  constexpr int WARPS_PER_BLOCK = BLOCK_DIM / 32;
  const size_t smem_bytes = (NUM_SINGLE + NUM_RED_LOG + WARPS_PER_BLOCK + 1) * sizeof(float);

  using KernelT =
      void (*)(float*, float*, float*, const float*, const int64_t*, const int64_t*, const float*, const float*);
  KernelT kernel = lp_prep_kernel<NC, NV, NUM_SINGLE, NUM_RED_LOG, NUM_GPUS, BLOCK_DIM>;

  if (smem_bytes > 48 * 1024) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes));
  }

  const DLDevice device = device_.unwrap();
  LaunchKernel(         1,           BLOCK_DIM, device, smem_bytes)(
      kernel,
      static_cast<float*>(A_full.data_ptr()),
      static_cast<float*>(b.data_ptr()),
      static_cast<float*>(t1.data_ptr()),
      static_cast<const float*>(global_counts.data_ptr()),
      static_cast<const int64_t*>(log_single.data_ptr()),
      static_cast<const int64_t*>(log_replicated.data_ptr()),
      static_cast<const float*>(B1.data_ptr()),
      static_cast<const float*>(A_base_row_sum.data_ptr()));
}

}
