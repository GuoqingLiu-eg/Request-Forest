













#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

namespace sglang {

struct RMSNormHFParams {
  const void* input;
  const void* __restrict__ weight;
  void* output;
  int64_t input_stride;
  int64_t output_stride;
  uint32_t num_tokens;
  float eps;
};






template <int64_t kDim, bool kUsePDL, typename Float>
__global__ __launch_bounds__(32) void rmsnorm_hf_warp_kernel(const RMSNormHFParams __grid_constant__ params) {
  using namespace device;
  constexpr int kElemsPerThread = kDim / kWarpThreads;

  const auto& [input, weight_ptr, output, input_stride, output_stride, num_tokens, eps] = params;
  const auto wr = static_cast<const Float*>(weight_ptr);

  PDLWaitPrimary<kUsePDL>();

  for (uint32_t row = blockIdx.x; row < num_tokens; row += gridDim.x) {
    const auto xr = static_cast<const Float*>(pointer::offset<Float>(input, row * input_stride));
    const auto yr = static_cast<Float*>(pointer::offset<Float>(output, row * output_stride));

    float xi_cache[kElemsPerThread];
    float lsq = 0.f;
#pragma unroll
    for (int k = 0; k < kElemsPerThread; ++k) {
      const int i = threadIdx.x + k * kWarpThreads;
      xi_cache[k] = static_cast<float>(xr[i]);
      lsq += xi_cache[k] * xi_cache[k];
    }
    lsq = warp::reduce_sum(lsq);
    const float rstd = math::rsqrt(lsq / kDim + eps);


#pragma unroll
    for (int k = 0; k < kElemsPerThread; ++k) {
      const int i = threadIdx.x + k * kWarpThreads;
      const Float xn = cast<Float>(xi_cache[k] * rstd);
      yr[i] = cast<Float>(static_cast<float>(xn) * static_cast<float>(wr[i]));
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}











template <int64_t kDim, bool kUsePDL, typename Float>
__global__ __launch_bounds__(512) void rmsnorm_hf_scalar_kernel(const RMSNormHFParams __grid_constant__ params) {
  using namespace device;
  constexpr int kNumThreads = 512;
  constexpr int kNumWarps = kNumThreads / kWarpThreads;

  constexpr int kElemsPerThread = (kDim + kNumThreads - 1) / kNumThreads;

  const auto& [input, weight_ptr, output, input_stride, output_stride, num_tokens, eps] = params;
  const auto xr = static_cast<const Float*>(pointer::offset<Float>(input, blockIdx.x * input_stride));
  const auto yr = static_cast<Float*>(pointer::offset<Float>(output, blockIdx.x * output_stride));
  const auto wr = static_cast<const Float*>(weight_ptr);

  PDLWaitPrimary<kUsePDL>();


  float xi_cache[kElemsPerThread];
  float lsq = 0.f;
#pragma unroll
  for (int k = 0; k < kElemsPerThread; ++k) {
    const int i = threadIdx.x + k * kNumThreads;
    xi_cache[k] = static_cast<float>(xr[i]);
    lsq += xi_cache[k] * xi_cache[k];
  }


  lsq = warp::reduce_sum(lsq);


  __shared__ float smem[32];
  const int warp_id = threadIdx.x / kWarpThreads;
  const int lane_id = threadIdx.x & (kWarpThreads - 1);
  if (lane_id == 0) smem[warp_id] = lsq;
  __syncthreads();

  __shared__ float rstd_s;
  if (threadIdx.x < kWarpThreads) {
    float v = (threadIdx.x < kNumWarps) ? smem[threadIdx.x] : 0.f;
    v = warp::reduce_sum(v);
    if (threadIdx.x == 0) rstd_s = math::rsqrt(v / kDim + eps);
  }
  __syncthreads();
  const float rstd = rstd_s;


#pragma unroll
  for (int k = 0; k < kElemsPerThread; ++k) {
    const int i = threadIdx.x + k * kNumThreads;
    const Float xn = cast<Float>(xi_cache[k] * rstd);
    yr[i] = cast<Float>(static_cast<float>(xn) * static_cast<float>(wr[i]));
  }

  PDLTriggerSecondary<kUsePDL>();
}






template <int64_t kDim, bool kUsePDL, typename DType>
struct HFRMSNormWarpKernel {
  static_assert(sizeof(DType) == 2, "rmsnorm_hf: DType must be fp16_t or bf16_t");
  static_assert(
      kDim >= 32 && kDim < 512 && kDim % 32 == 0, "rmsnorm_hf_warp: kDim must be a multiple of 32, in [32, 512)");
  static constexpr auto kernel = rmsnorm_hf_warp_kernel<kDim, kUsePDL, DType>;
  static constexpr uint32_t kBlockSize = device::kWarpThreads;

  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView output,
      float eps) {
    using namespace host;
    auto N = SymbolicSize{"num_tokens"};
    auto D = SymbolicSize{"hidden_size"};
    auto SI = SymbolicSize{"input_stride"};
    auto SO = SymbolicSize{"output_stride"};
    auto device_ = SymbolicDevice{};
    D.set_value(kDim);
    device_.set_options<kDLCUDA>();

    TensorMatcher({N, D}).with_strides({SI, 1}).with_dtype<DType>().with_device(device_).verify(input);
    TensorMatcher({D}).with_dtype<DType>().with_device(device_).verify(weight);
    TensorMatcher({N, D}).with_strides({SO, 1}).with_dtype<DType>().with_device(device_).verify(output);

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    RuntimeCheck(num_tokens > 0, "rmsnorm_hf: num_tokens must be > 0");

    const auto params = RMSNormHFParams{
        .input = input.data_ptr(),
        .weight = weight.data_ptr(),
        .output = output.data_ptr(),
        .input_stride = SI.unwrap(),
        .output_stride = SO.unwrap(),
        .num_tokens = num_tokens,
        .eps = eps,
    };

    static const uint32_t max_occupancy = runtime::get_blocks_per_sm(kernel, kBlockSize);
    static const uint32_t kNumSM = runtime::get_sm_count(device_.unwrap().device_id);
    const auto num_blocks = std::min<uint32_t>(num_tokens, max_occupancy * kNumSM);
    LaunchKernel(num_blocks, kBlockSize, device_.unwrap())
        .enable_pdl(kUsePDL)(kernel, params);
  }
};




template <int64_t kDim, bool kUsePDL, typename DType>
struct HFRMSNormKernel {
  static_assert(sizeof(DType) == 2, "rmsnorm_hf: DType must be fp16_t or bf16_t");
  static_assert(kDim >= 512 && kDim % 512 == 0, "rmsnorm_hf: kDim must be a multiple of 512");
  static constexpr auto kernel = rmsnorm_hf_scalar_kernel<kDim, kUsePDL, DType>;
  static constexpr uint32_t kBlockSize = 512;

  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView output,
      float eps) {
    using namespace host;
    auto N = SymbolicSize{"num_tokens"};
    auto D = SymbolicSize{"hidden_size"};
    auto SI = SymbolicSize{"input_stride"};
    auto SO = SymbolicSize{"output_stride"};
    auto device_ = SymbolicDevice{};
    D.set_value(kDim);
    device_.set_options<kDLCUDA>();

    TensorMatcher({N, D})
        .with_strides({SI, 1})
        .with_dtype<DType>()
        .with_device(device_)
        .verify(input);
    TensorMatcher({D})
        .with_dtype<DType>()
        .with_device(device_)
        .verify(weight);
    TensorMatcher({N, D})
        .with_strides({SO, 1})
        .with_dtype<DType>()
        .with_device(device_)
        .verify(output);

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    RuntimeCheck(num_tokens > 0, "rmsnorm_hf: num_tokens must be > 0");

    const auto params = RMSNormHFParams{
        .input = input.data_ptr(),
        .weight = weight.data_ptr(),
        .output = output.data_ptr(),
        .input_stride = SI.unwrap(),
        .output_stride = SO.unwrap(),
        .num_tokens = num_tokens,
        .eps = eps,
    };

    LaunchKernel(num_tokens, kBlockSize, device_.unwrap())
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}
