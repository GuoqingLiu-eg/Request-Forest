#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>

namespace sglang {

constexpr size_t kBlockSize = 256;
constexpr size_t kVectorizedMinElements = 1 << 20;
constexpr size_t kVectorBytes = device::kMaxVecBytes;
static_assert(kVectorBytes % sizeof(int32_t) == 0, "Vector byte width must contain whole int32_t elements");
constexpr size_t kElementsPerVector = kVectorBytes / sizeof(int32_t);

template <typename Vector>
bool is_aligned_for_vector(const int32_t* ptr) {
  return reinterpret_cast<uintptr_t>(ptr) % alignof(Vector) == 0;
}

template <int32_t kConstant>
__global__ void add_constant_kernel(int32_t* dst, const int32_t* src, size_t length) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < length) {
    dst[idx] = src[idx] + kConstant;
  }
}

template <int32_t kConstant, size_t kElementsPerVector>
__global__ void add_constant_vectorized_kernel(int32_t* dst, const int32_t* src, size_t length) {
  using Vector = device::AlignedVector<int32_t, kElementsPerVector>;

  const size_t work_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t vector_count = length / kElementsPerVector;
  const size_t tail_start = vector_count * kElementsPerVector;

  if (work_idx < vector_count) {
    auto values = device::load_as<Vector>(src, work_idx);
#pragma unroll
    for (size_t i = 0; i < kElementsPerVector; ++i) {
      values[i] += kConstant;
    }
    device::store_as<Vector>(dst, values, work_idx);
  } else {
    const size_t tail_idx = tail_start + work_idx - vector_count;
    if (tail_idx < length) {
      dst[tail_idx] = src[tail_idx] + kConstant;
    }
  }
}


template <int32_t kConstant>
void add_constant(tvm::ffi::TensorView dst, tvm::ffi::TensorView src) {
  using namespace host;


  SymbolicSize N = {"num_elements"};
  SymbolicDevice device_;
  TensorMatcher({N})
      .with_dtype<int32_t>()
      .with_device<kDLGPU>(device_)
      .verify(dst)
      .verify(src);


  const size_t num_elements = N.unwrap();
  const DLDevice device = device_.unwrap();
  [[maybe_unused]]
  const size_t dynamic_smem = 0;
  [[maybe_unused]]
  const cudaStream_t stream = LaunchKernel::resolve_device(device);

  RuntimeCheck(num_elements > 0, "We only support non-empty tensors, got num_elements = ", num_elements);

  const auto* src_ptr = static_cast<const int32_t*>(src.data_ptr());
  auto* dst_ptr = static_cast<int32_t*>(dst.data_ptr());
  using Vector = device::AlignedVector<int32_t, kElementsPerVector>;
  const bool is_vector_aligned = is_aligned_for_vector<Vector>(src_ptr) && is_aligned_for_vector<Vector>(dst_ptr);


  if (num_elements >= kVectorizedMinElements && is_vector_aligned) {
    const size_t vector_count = num_elements / kElementsPerVector;
    const size_t tail_count = num_elements - vector_count * kElementsPerVector;
    const size_t work_items = vector_count + tail_count;
    const size_t grid_size = div_ceil(work_items, kBlockSize);
    LaunchKernel(grid_size, kBlockSize, device                   )(
        add_constant_vectorized_kernel<kConstant, kElementsPerVector>, dst_ptr, src_ptr, num_elements);
  } else {
    const size_t grid_size = div_ceil(num_elements, kBlockSize);
    LaunchKernel(grid_size, kBlockSize, device                   )(
        add_constant_kernel<kConstant>, dst_ptr, src_ptr, num_elements);
  }
}

}
