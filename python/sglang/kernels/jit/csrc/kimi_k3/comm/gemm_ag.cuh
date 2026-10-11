#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/distributed/communicator.cuh>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/stl.h>

#include <array>
#include <cstdint>
#include <optional>
#include <utility>

namespace sglang {

namespace gemm_ag {

using device::distributed::Counter;

constexpr uint32_t kWorld = 8;
constexpr uint32_t kVecSize = 32 / sizeof(bf16_t);
constexpr uint32_t kSpinBlock = 128;
constexpr uint32_t kSpinVec = 16 / sizeof(bf16_t);



struct ProducerParams {
  uint8_t* ws_mc;
  Counter* counter;
  uint32_t half_bytes;
  uint32_t rank;
};

template <uint32_t K, uint32_t N, uint32_t M, uint32_t N_SPLIT, bool kUsePDL>
__global__ __launch_bounds__(K / kVecSize) void gemm_ag_gemv_kernel(
    const __grid_constant__ ProducerParams params,
    const bf16_t* __restrict__ x,
    const bf16_t* __restrict__ weight) {
  using namespace device;
  using vec_t = AlignedVector<bf16_t, kVecSize>;
  constexpr uint32_t kNLocal = N / kWorld;
  constexpr uint32_t kGemvBlock = K / kVecSize;
  constexpr uint32_t kNumWarps = kGemvBlock / kWarpThreads;
  static_assert(K % kVecSize == 0, "K must be a multiple of the 32B vector width");
  static_assert(kGemvBlock % kWarpThreads == 0, "K / vec_size must fill whole warps");
  static_assert(kGemvBlock <= 1024, "K / vec_size exceeds the maximum block size");
  static_assert(N % kWorld == 0, "N must split evenly over the TP world");
  static_assert(kNLocal % N_SPLIT == 0, "the local column slice must split into whole tiles");
  static_assert(M * N_SPLIT <= kGemvBlock, "output tile must fit one thread each for the final reduce");
  static_assert(N_SPLIT % 2 == 0, "epilogue stores adjacent column pairs");

  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x;


  const bf16_t* weight_tile = weight + (params.rank * kNLocal + bx * N_SPLIT) * K;


  vec_t weight_vec[N_SPLIT];
#pragma unroll
  for (uint32_t n = 0; n < N_SPLIT; ++n) {
    weight_vec[n].load(weight_tile + n * K, tx);
  }

  PDLWaitPrimary<kUsePDL>();




  const uint32_t phase = params.counter[0].get() & 1;

  vec_t input_vec[M];
#pragma unroll
  for (uint32_t m = 0; m < M; ++m) {
    input_vec[m].load(x + m * K, tx);
  }

  __shared__ alignas(16) float s_acc[kNumWarps][M * N_SPLIT];
  const uint32_t warp_id = tx / kWarpThreads;

#pragma unroll
  for (uint32_t m = 0; m < M; ++m) {
#pragma unroll
    for (uint32_t n = 0; n < N_SPLIT; ++n) {
      float acc = 0.0f;
#pragma unroll
      for (uint32_t i = 0; i < kVecSize; ++i) {
        acc = device::math::fma_f32_bf16(input_vec[m][i], weight_vec[n][i], acc);
      }
      s_acc[warp_id][m * N_SPLIT + n] = warp::reduce_sum(acc);
    }
  }
  __syncthreads();
  constexpr uint32_t kNumPairs = M * N_SPLIT / 2;
  if (tx < kNumPairs) {
    auto packed = load_as<float2>(s_acc[0], tx);
#pragma unroll
    for (uint32_t i = 1; i < kNumWarps; ++i) {
      const auto [lo, hi] = load_as<float2>(s_acc[i], tx);
      packed.x += lo;
      packed.y += hi;
    }
    const auto pair = cast<bf16x2_t>(packed);
    auto bits = *reinterpret_cast<const uint32_t*>(&pair);
    if (bits == 0) bits = 0x8000u;
    const uint32_t m = (2 * tx) / N_SPLIT;
    const uint32_t n = (2 * tx) % N_SPLIT;


    const uint32_t elem = (params.rank * M + m) * kNLocal + bx * N_SPLIT + n;
    const auto base = reinterpret_cast<bf16_t*>(params.ws_mc + phase * params.half_bytes);
    const auto dst = reinterpret_cast<uint32_t*>(base + elem);
    ptx::multimem_store_relaxed(dst, bits);
  }
  PDLTriggerSecondary<kUsePDL>();
}



struct ConsumerParams {
  uint8_t* ws_local;
  Counter* counter;
  uint32_t num_counters;
  uint32_t half_bytes;
  const bf16_t* b;
  const bf16_t* c;
  bf16_t* out;
  uint32_t num_rows;
};

template <uint32_t N, bool kHasC, bool kUsePDL>
__global__ void spin_add3_kernel(const __grid_constant__ ConsumerParams params) {
  using namespace device;
  using vec_t = AlignedVector<bf16x2_t, kSpinVec / 2>;
  constexpr uint32_t kNLocal = N / kWorld;
  static_assert(N % kSpinVec == 0, "rows must stay 16B aligned");
  static_assert(kNLocal % kSpinVec == 0, "a vector must never cross a rank block");
  const auto bx = blockIdx.x;
  const auto tx = threadIdx.x;
  const uint32_t tid = bx * kSpinBlock + tx;
  const uint32_t elem = tid * kSpinVec;
  const uint32_t phase = params.counter[bx].get() & 1;

  PDLTriggerSecondary<kUsePDL>();



  if (const auto num_blocks = gridDim.x; bx == num_blocks - 1) {
    [[unlikely]];
    __syncthreads();
    for (uint32_t i = num_blocks - 1 + tx; i < params.num_counters; i += kSpinBlock) {
      params.counter[i].set(phase ^ 1);
    }
    return void();
  }


  if (elem < params.num_rows * N) {
    const auto row = elem / N;
    const auto col = elem % N;


    const auto base = reinterpret_cast<bf16_t*>(params.ws_local + phase * params.half_bytes);
    const auto src = base + ((col / kNLocal) * params.num_rows + row) * kNLocal + col % kNLocal;
    vec_t b_vec, c_vec;
    b_vec.load(params.b + elem);
    if constexpr (kHasC) c_vec.load(params.c + elem);

    uint4 raw;
    do {
      ptx::ld_relaxed_16B(raw, src, 0);
    } while (raw.x == 0 || raw.y == 0 || raw.z == 0 || raw.w == 0);
    const auto& gathered = *reinterpret_cast<const vec_t*>(&raw);
    vec_t out_vec;
#pragma unroll
    for (uint32_t j = 0; j < kSpinVec / 2; ++j) {
      using Trait = DTypeTrait<bf16x2_t>;
      out_vec[j] = Trait::add(gathered[j], b_vec[j]);
      if constexpr (kHasC) out_vec[j] = Trait::add(out_vec[j], c_vec[j]);
    }
    out_vec.store(params.out + elem);
    AlignedVector<uint32_t, 4> zero;
    zero.fill(0);
    zero.store(src);
  }
  __syncthreads();
  if (tx == 0) params.counter[bx].set(phase ^ 1);
}

}
using host::distributed::CommunicatorRef;






template <uint32_t K, uint32_t N, uint32_t kMaxM, bool kUsePDL>
struct GEMMAGKernel {
  using TensorView = tvm::ffi::TensorView;










  static constexpr uint32_t N_SPLIT = 2;
  static constexpr uint32_t kNLocal = N / gemm_ag::kWorld;
  static constexpr uint32_t kGemvBlock = K / gemm_ag::kVecSize;
  static constexpr uint32_t kNumProducerBlocks = kNLocal / N_SPLIT;
  static_assert(kNLocal % N_SPLIT == 0);

  using GemvFn = void (*)(gemm_ag::ProducerParams, const bf16_t*, const bf16_t*);

  template <std::size_t... I>
  static constexpr auto make_table(std::index_sequence<I...>) {
    return std::array<GemvFn, kMaxM + 1>{nullptr, gemm_ag::gemm_ag_gemv_kernel<K, N, I + 1, N_SPLIT, kUsePDL>...};
  }
  static constexpr auto kGemvTable = make_table(std::make_index_sequence<kMaxM>{});

  static void
  run(CommunicatorRef ref, TensorView x, TensorView weight, TensorView b, std::optional<TensorView> c, TensorView out) {
    using namespace host;
    const auto& push = ref.get()->get_push_obj();

    auto M = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, K}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({N, K}).with_dtype<bf16_t>().with_device(device).verify(weight);
    TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).verify(b);
    if (c.has_value()) {
      TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).verify(c.value());
    }
    TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).verify(out);
    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    CHECK_HOST(num_tokens >= 1 && num_tokens <= kMaxM);
    CHECK_HOST(push.world_size == gemm_ag::kWorld) << "the kernel is compiled for TP" << gemm_ag::kWorld;
    CHECK_HOST(push.mc_workspace != nullptr) << "requires a multicast-capable push plane";
    CHECK_HOST(int64_t(num_tokens) * kNLocal * 2 <= push.slot_bytes)
        << "staging slice exceeds the push slot size " << push.slot_bytes;


    CHECK_HOST(push.num_blocks > 0) << "no push blocks available";

    const auto producer_params = gemm_ag::ProducerParams{
        .ws_mc = push.mc_workspace,
        .counter = push.counter,
        .half_bytes = static_cast<uint32_t>(push.slot_bytes * push.world_size),
        .rank = push.rank,
    };
    LaunchKernel(kNumProducerBlocks, kGemvBlock, device.unwrap())
        .enable_pdl(kUsePDL)(
            kGemvTable[num_tokens],
            producer_params,
            static_cast<const bf16_t*>(x.data_ptr()),
            static_cast<const bf16_t*>(weight.data_ptr()));


    const auto consumer_params = gemm_ag::ConsumerParams{
        .ws_local = push.workspaces[push.rank],
        .counter = push.counter,
        .num_counters = push.num_blocks,
        .half_bytes = static_cast<uint32_t>(push.slot_bytes * push.world_size),
        .b = static_cast<const bf16_t*>(b.data_ptr()),
        .c = c.has_value() ? static_cast<const bf16_t*>(c.value().data_ptr()) : nullptr,
        .out = static_cast<bf16_t*>(out.data_ptr()),
        .num_rows = num_tokens,
    };
    const auto num_vecs = num_tokens * N / gemm_ag::kSpinVec;
    const auto num_consumers = host::div_ceil(num_vecs, gemm_ag::kSpinBlock);
    CHECK_HOST(num_consumers + 1 <= push.num_blocks);

    const auto num_consumer_blocks = num_consumers + 1;
    using gemm_ag::spin_add3_kernel;
    const auto kernel = c.has_value() ? spin_add3_kernel<N, 1, kUsePDL> : spin_add3_kernel<N, 0, kUsePDL>;
    host::LaunchKernel(num_consumer_blocks, gemm_ag::kSpinBlock, device.unwrap())
        .enable_pdl(kUsePDL)(kernel, consumer_params);
  }
};

}
