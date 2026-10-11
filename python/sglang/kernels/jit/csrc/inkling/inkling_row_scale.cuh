







#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

constexpr uint32_t kRsVec = 8;
constexpr uint32_t kRsBlock = 256;





template <bool kUsePDL, bool kHasTau>
__global__ __launch_bounds__(kRsBlock, 1) void row_scale_kernel(
    const bf16_t* __restrict__ x,
    const fp32_t* __restrict__ tau,
    bf16_t* __restrict__ out,
    const int64_t x_stride_row,
    const uint32_t inner,
    const uint32_t rows) {
  using namespace device;
  PDLWaitPrimary<kUsePDL>();
  const uint32_t vrow = inner / kRsVec;
  const uint32_t total = rows * vrow;
  for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridDim.x * blockDim.x) {
    const uint32_t row = idx / vrow;
    const uint32_t v = idx % vrow;
    AlignedVector<bf16_t, kRsVec> a;
    a.load(x + static_cast<int64_t>(row) * x_stride_row, v);
    if constexpr (kHasTau) {
      const float tv = tau[row];
#pragma unroll
      for (int k = 0; k < static_cast<int>(kRsVec); ++k) {
        a[k] = static_cast<bf16_t>(static_cast<float>(a[k]) * tv);
      }
    }
    a.store(out, idx);
  }
  PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL, bool kHasTau>
void row_scale_launch(
    tvm::ffi::TensorView x,
    const fp32_t* tau_ptr,
    tvm::ffi::TensorView out,
    host::SymbolicSize& R,
    host::SymbolicSize& N,
    host::SymbolicDevice& dev) {
  using namespace host;
  TensorMatcher({R, N}).with_dtype<bf16_t>().with_device(dev).with_strides({-1, 1}).verify(x);
  TensorMatcher({R, N}).with_dtype<bf16_t>().with_device(dev).verify(out);

  const uint32_t rows = static_cast<uint32_t>(R.unwrap());
  const uint32_t inner = static_cast<uint32_t>(N.unwrap());
  RuntimeCheck(inner % kRsVec == 0, "inner must be a multiple of ", kRsVec);
  RuntimeCheck((x.stride(0) * 2) % 16 == 0, "x row stride must keep 16B alignment");
  RuntimeCheck(std::bit_cast<intptr_t>(x.data_ptr()) % 16 == 0, "x not 16B aligned");

  const auto kernel = row_scale_kernel<kUsePDL, kHasTau>;
  const uint32_t sm = runtime::get_sm_count(dev.unwrap().device_id);
  const uint32_t bps = runtime::get_blocks_per_sm(kernel, kRsBlock);
  const uint32_t want = div_ceil(rows * (inner / kRsVec), kRsBlock);
  const uint32_t grid = std::min(sm * std::max(1u, bps), std::max(1u, want));
  LaunchKernel(grid, kRsBlock, dev.unwrap())
      .enable_pdl(kUsePDL)(
          kernel,
          static_cast<const bf16_t*>(x.data_ptr()),
          tau_ptr,
          static_cast<bf16_t*>(out.data_ptr()),
          x.stride(0),
          inner,
          rows);
}

template <bool kUsePDL>
void row_scale(tvm::ffi::TensorView x, tvm::ffi::TensorView tau, tvm::ffi::TensorView out) {
  using namespace host;
  auto R = SymbolicSize{"rows"};
  auto N = SymbolicSize{"inner"};
  auto dev = SymbolicDevice{};
  dev.set_options<kDLCUDA>();
  TensorMatcher({R}).with_dtype<fp32_t>().with_device(dev).verify(tau);
  row_scale_launch<kUsePDL, true>(x, static_cast<const fp32_t*>(tau.data_ptr()), out, R, N, dev);
}


template <bool kUsePDL>
void row_compact(tvm::ffi::TensorView x, tvm::ffi::TensorView out) {
  using namespace host;
  auto R = SymbolicSize{"rows"};
  auto N = SymbolicSize{"inner"};
  auto dev = SymbolicDevice{};
  dev.set_options<kDLCUDA>();
  row_scale_launch<kUsePDL, false>(x, nullptr, out, R, N, dev);
}

}
