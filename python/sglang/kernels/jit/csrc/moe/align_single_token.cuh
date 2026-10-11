








#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

struct AlignSingleTokenParams {
  const int32_t* __restrict__ topk_ids;
  int32_t* __restrict__ sorted_ids;
  int32_t* __restrict__ expert_ids;
  int32_t* __restrict__ num_post;
  uint32_t topk;
  uint32_t block_size;
};

template <bool kUsePDL>
__global__ void align_single_token_kernel(const AlignSingleTokenParams __grid_constant__ params) {
  using namespace device;
  const uint32_t lane = threadIdx.x;
  const uint32_t topk = params.topk;
  const uint32_t bs = params.block_size;

  PDLWaitPrimary<kUsePDL>();

  int32_t my_id = (lane < topk) ? params.topk_ids[lane] : INT32_MAX;



  uint32_t rank = 0;
  for (uint32_t j = 0; j < topk; ++j) {
    int32_t other = __shfl_sync(0xffffffff, my_id, j);
    if (other < my_id || (other == my_id && j < lane)) {
      rank++;
    }
  }

  if (lane < topk) {
    params.expert_ids[rank] = my_id;


    params.sorted_ids[rank * bs] = static_cast<int32_t>(lane);
  }

  for (uint32_t p = lane; p < topk * bs; p += 32) {
    if (p % bs != 0) {
      params.sorted_ids[p] = static_cast<int32_t>(topk);
    }
  }
  if (lane == 0) {
    params.num_post[0] = static_cast<int32_t>(topk * bs);
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL>
struct AlignSingleTokenKernel {
  static constexpr auto kernel = align_single_token_kernel<kUsePDL>;

  static void
  run(const tvm::ffi::TensorView topk_ids,
      const tvm::ffi::TensorView sorted_ids,
      const tvm::ffi::TensorView expert_ids,
      const tvm::ffi::TensorView num_post,
      int64_t block_size) {
    using namespace host;

    auto One_ = SymbolicSize{"one"};
    auto K_ = SymbolicSize{"topk"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({One_, K_}).with_dtype<int32_t>().with_device(device).verify(topk_ids);

    const auto topk = static_cast<uint32_t>(K_.unwrap());
    RuntimeCheck(One_.unwrap() == 1, "moe_align_single_token requires M == 1");
    RuntimeCheck(topk <= 32, "moe_align_single_token requires topk <= 32");

    const auto params = AlignSingleTokenParams{
        .topk_ids = static_cast<const int32_t*>(topk_ids.data_ptr()),
        .sorted_ids = static_cast<int32_t*>(sorted_ids.data_ptr()),
        .expert_ids = static_cast<int32_t*>(expert_ids.data_ptr()),
        .num_post = static_cast<int32_t*>(num_post.data_ptr()),
        .topk = topk,
        .block_size = static_cast<uint32_t>(block_size),
    };

    LaunchKernel(dim3(1), 32, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}
