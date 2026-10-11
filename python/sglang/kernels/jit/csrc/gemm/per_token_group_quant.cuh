#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstdint>
#include <cuda_fp8.h>

namespace sglang {

namespace detail {

SGL_DEVICE float silu(const float val) {

#if SGL_ARCH_BLACKWELL_OR_GREATER
  const float half = 0.5f * val;
  return half * (1.0f + __tanhf(half));
#else
  return val * __frcp_rn(1.0f + __expf(-val));
#endif
}

SGL_DEVICE float2 mul2(const float2 a, const float2 b) {


#if SGL_ARCH_BLACKWELL_OR_GREATER
  return __fmul2_rn(a, b);
#else
  return float2{a.x * b.x, a.y * b.y};
#endif
}

template <typename T>
struct WeightTrait {};

template <>
struct WeightTrait<fp8_e4m3_t> {
  using packed2_t = fp8x2_e4m3_t;
  static constexpr float kMaxValue = DTypeTrait<fp8_e4m3_t>::kFloatMax;









  SGL_DEVICE static packed2_t quant(const float2 v) {
    return packed2_t{float2{fminf(v.x, kMaxValue), fminf(v.y, kMaxValue)}};
  }
};

template <>
struct WeightTrait<int8_t> {
  using packed2_t = char2;
  static constexpr float kMaxValue = 127.0f;
  static constexpr float kMinValue = -128.0f;

  SGL_DEVICE static packed2_t quant(const float2 v) {
    return packed2_t{
        static_cast<int8_t>(fminf(fmaxf(v.x, kMinValue), kMaxValue)),
        static_cast<int8_t>(fminf(fmaxf(v.y, kMinValue), kMaxValue))};
  }
};

template <bool kUe8m0>
using scale_t = std::conditional_t<kUe8m0, uint8_t, float>;





struct ScaleStoreArgs {
 public:
  void* __restrict__ base;
  uint32_t expert_stride;
  uint32_t token_stride;
  uint32_t group_stride;
  uint32_t num_groups;

  void check_overflow(uint32_t num_experts, uint32_t num_tokens) const {

    const uint64_t expert_bytes = static_cast<uint64_t>(expert_stride) * num_experts;
    const uint64_t token_bytes = static_cast<uint64_t>(token_stride) * num_tokens;
    const uint64_t group_bytes = static_cast<uint64_t>(group_stride) * num_groups;
    const uint64_t total_bytes = expert_bytes + token_bytes + group_bytes;
    CHECK_HOST(std::max({expert_bytes, token_bytes, group_bytes, total_bytes}) <= UINT32_MAX)
        << "Internal Error: ScaleStoreArgs overflow. Something wild must happened.\n"
        << "More debug info: "
        << " expert_bytes=" << expert_bytes
        << " token_bytes=" << token_bytes
        << " group_bytes=" << group_bytes;
  }

  template <bool kUe8m0, bool kRowMajor, bool kAligned>
  SGL_DEVICE void store(
      const uint32_t expert_idx,
      const uint32_t token_idx,
      const uint32_t group_idx,
      const scale_t<kUe8m0> scale_inv) const {
    static_assert(kAligned || kUe8m0, "Only ue8m0 scales can be unaligned (pad to 4 bytes)");
    using T = scale_t<kUe8m0>;
    if constexpr (kRowMajor || !kUe8m0) {



      const auto token_stride = kRowMajor ? this->token_stride : 1;
      const auto group_stride = kRowMajor ? 1 : this->group_stride;
      const uint64_t offset = expert_idx * expert_stride
                              + token_idx * token_stride
                              + group_idx * group_stride;
      static_cast<T*>(this->base)[offset] = scale_inv;
      if constexpr (!kAligned) this->fill_unaligned(group_idx, offset);
    } else {




      const auto packed_group = group_idx / 4;
      const uint64_t offset = expert_idx * expert_stride
                              + packed_group * group_stride
                              + token_idx * 4 + group_idx % 4;
      static_cast<T*>(this->base)[offset] = scale_inv;
      if constexpr (!kAligned) this->fill_unaligned(group_idx, offset);
    }
  }

 private:
  SGL_DEVICE void fill_unaligned(const uint32_t group_idx, const uint64_t offset) const {




    const auto rem = this->num_groups % 4;
    if (group_idx == this->num_groups - 1) {
      const uint64_t int32_base = offset - (rem - 1);
#pragma unroll
      for (uint32_t b = rem; b < 4; ++b) {
        static_cast<uint8_t*>(this->base)[int32_base + b] = 0;
      }
    }
  }
};

template <typename T2>
struct Vec32B {
 public:
  static_assert(sizeof(T2) == 4, "must be packed fp16/bf16");
  static constexpr uint32_t kVecSize = device::kMaxVecBytes / sizeof(T2);
  static constexpr uint32_t kNumVecs = 32 / device::kMaxVecBytes;
  SGL_DEVICE void load(const void* ptr, const uint32_t lane_id) {
#pragma unroll
    for (uint32_t v = 0; v < kNumVecs; ++v) {
      m_vecs[v].load(ptr, lane_id * kNumVecs + v);
    }
  }
  SGL_DEVICE auto operator[](const uint32_t i) -> T2& {
    return m_vecs[i / kVecSize][i % kVecSize];
  }
  SGL_DEVICE auto operator[](const uint32_t i) const -> T2 {
    return m_vecs[i / kVecSize][i % kVecSize];
  }

 private:
  device::AlignedVector<T2, kVecSize> m_vecs[kNumVecs];
};



struct TensorArgs {
  void* __restrict__ ptr;
  int64_t expert_stride;
  int64_t token_stride;
  template <typename T>
  SGL_DEVICE T* get(const uint32_t expert_idx, const uint32_t token_idx) const {
    const uint64_t offset = expert_idx * this->expert_stride
                            + token_idx * this->token_stride;
    return static_cast<T*>(ptr) + offset;
  }
};

}

struct QuantKernelParams {
  detail::TensorArgs input;
  detail::TensorArgs output;
  detail::ScaleStoreArgs scale;
  uint32_t num_tokens;
  uint32_t hidden_size;
};

struct MaskedQuantKernelParams {
  QuantKernelParams base;



  const int32_t* __restrict__ masked_m;
  uint32_t masked_m_stride;
};



template <
    typename InputType_,
    typename QuantType_,
    uint32_t kGroupSize_,
    bool kUe8m0_,
    bool kRowMajor_,
    bool kAligned_,
    bool kFuseSiluAndMul_>
struct QuantTrait {

  using InputType = InputType_;
  using QuantType = QuantType_;
  static constexpr uint32_t kGroupSize = kGroupSize_;
  static constexpr bool kUe8m0 = kUe8m0_;
  static constexpr bool kRowMajor = kRowMajor_;
  static constexpr bool kAligned = kAligned_;
  static constexpr bool kFuseSiluAndMul = kFuseSiluAndMul_;
  static constexpr uint32_t kBlockSize = 256;
  static constexpr uint32_t kVecSize = 32u / 2;
  static constexpr uint32_t kNumLanes = kGroupSize / kVecSize;
  static_assert(sizeof(InputType) == 2 || sizeof(InputType) == 4, "inputs must be 16-bit (bf16/fp16) or fp32");
  static_assert(sizeof(InputType) == 2 || !kFuseSiluAndMul, "fp32 inputs do not implement the fused silu");
  static_assert(16 <= kGroupSize && kGroupSize <= 256, "supported group sizes are 16..256");
  static_assert(kGroupSize % kVecSize == 0 && 1 <= kNumLanes && kNumLanes <= device::kWarpThreads);
  static_assert(!kUe8m0 || std::is_same_v<QuantType, fp8_e4m3_t>, "ue8m0 scales imply fp8 output");

  SGL_DEVICE static void
  run(const QuantKernelParams& params,
      const uint32_t expert_idx,
      const uint32_t token_idx,
      const uint32_t group_idx,
      const uint32_t lane_id) {
    if constexpr (sizeof(InputType) == 4) {
      run_fp32(params, expert_idx, token_idx, group_idx, lane_id);
    } else {
      run_packed16(params, expert_idx, token_idx, group_idx, lane_id);
    }
  }

  SGL_DEVICE static void run_packed16(
      const QuantKernelParams& params,
      const uint32_t expert_idx,
      const uint32_t token_idx,
      const uint32_t group_idx,
      const uint32_t lane_id) {
    using deepseek_v4::fp8::cast_to_ue8m0;
    using deepseek_v4::fp8::inv_scale_ue8m0;
    using namespace device;
    using T = InputType;
    using T2 = packed_t<T>;
    using Q = QuantType;
    using WTrait = detail::WeightTrait<Q>;
    using Q2 = typename WTrait::packed2_t;
    using in_vec_t = detail::Vec32B<T2>;
    using out_vec_t = AlignedVector<Q2, kVecSize / 2>;
    constexpr float kMaxValue = WTrait::kMaxValue;
    constexpr float kMaxValueInv = 1.f / kMaxValue;

    const T* token_in = params.input.get<const T>(expert_idx, token_idx);
    const uint32_t group_offset = group_idx * kGroupSize;



    in_vec_t in;
    in.load(token_in + group_offset, lane_id);
    if constexpr (kFuseSiluAndMul) {
      in_vec_t up;
      up.load(token_in + group_offset + params.hidden_size, lane_id);
#pragma unroll
      for (uint32_t i = 0; i < kVecSize / 2; ++i) {
        const auto gate = cast<float2>(in[i]);
        const auto act = cast<T2>(float2{detail::silu(gate.x), detail::silu(gate.y)});
        in[i] = __hmul2(act, up[i]);
      }
    }


    T2 local_amax2 = math::abs(in[0]);
#pragma unroll
    for (uint32_t i = 1; i < kVecSize / 2; ++i) {
      local_amax2 = math::max(local_amax2, math::abs(in[i]));
    }
    const auto amax2 = cast<float2>(warp::reduce_max<kNumLanes>(local_amax2));
    const auto amax = math::max(math::max(amax2.x, amax2.y), 1e-10f);
    const float raw_scale = amax * kMaxValueInv;

    out_vec_t out;
    detail::scale_t<kUe8m0> scale_inv;
    if constexpr (kUe8m0) {

      static_assert(std::is_same_v<Q, fp8_e4m3_t>, "ue8m0 scales imply fp8 quantization");
      const auto exp = cast_to_ue8m0(raw_scale);
      scale_inv = static_cast<uint8_t>(exp);
      const float quant_scale = inv_scale_ue8m0(exp);
      const auto scale2 = cast<T2>(float2{quant_scale, quant_scale});




      const auto max_clip = cast<T>(kMaxValue);
      const auto max_clip2 = T2{max_clip, max_clip};
#pragma unroll
      for (uint32_t i = 0; i < kVecSize / 2; ++i) {
        out[i] = static_cast<Q2>(__hmin2(__hmul2(in[i], scale2), max_clip2));
      }
    } else {

      scale_inv = raw_scale;
      const float quant_scale = kMaxValue / amax;
      const float2 quant_scale2 = {quant_scale, quant_scale};
#pragma unroll
      for (uint32_t i = 0; i < kVecSize / 2; ++i) {
        out[i] = WTrait::quant(detail::mul2(cast<float2>(in[i]), quant_scale2));
      }
    }

    out.store(params.output.get<Q>(expert_idx, token_idx) + group_offset, lane_id);
    params.scale.store<kUe8m0, kRowMajor, kAligned>(expert_idx, token_idx, group_idx, scale_inv);
  }

  SGL_DEVICE static void run_fp32(
      const QuantKernelParams& params,
      const uint32_t expert_idx,
      const uint32_t token_idx,
      const uint32_t group_idx,
      const uint32_t lane_id) {
    using deepseek_v4::fp8::cast_to_ue8m0;
    using deepseek_v4::fp8::inv_scale_ue8m0;
    using namespace device;
    using Q = QuantType;
    using WTrait = detail::WeightTrait<Q>;
    using Q2 = typename WTrait::packed2_t;
    constexpr uint32_t kSubVec = kMaxVecBytes / sizeof(fp32_t);
    constexpr uint32_t kNumSubVecs = kVecSize / kSubVec;
    using in_vec_t = AlignedVector<fp32_t, kSubVec>;
    using out_vec_t = AlignedVector<Q2, kVecSize / 2>;
    constexpr float kMaxValue = WTrait::kMaxValue;
    constexpr float kMaxValueInv = 1.f / kMaxValue;

    const fp32_t* token_in = params.input.get<const fp32_t>(expert_idx, token_idx);
    const uint32_t group_offset = group_idx * kGroupSize;

    in_vec_t in_vecs[kNumSubVecs];
#pragma unroll
    for (uint32_t v = 0; v < kNumSubVecs; ++v) {
      in_vecs[v].load(token_in + group_offset, lane_id * kNumSubVecs + v);
    }
    const auto in = [&](const uint32_t i) { return in_vecs[i / kSubVec][i % kSubVec]; };

    float local_amax = fabsf(in(0));
#pragma unroll
    for (uint32_t i = 1; i < kVecSize; ++i) {
      local_amax = math::max(local_amax, fabsf(in(i)));
    }
    const auto amax = math::max(warp::reduce_max<kNumLanes>(local_amax), 1e-10f);
    const float raw_scale = amax * kMaxValueInv;

    out_vec_t out;
    detail::scale_t<kUe8m0> scale_inv;
    float quant_scale;
    if constexpr (kUe8m0) {
      static_assert(std::is_same_v<Q, fp8_e4m3_t>, "ue8m0 scales imply fp8 quantization");
      const auto exp = cast_to_ue8m0(raw_scale);
      scale_inv = static_cast<uint8_t>(exp);
      quant_scale = inv_scale_ue8m0(exp);
    } else {
      scale_inv = raw_scale;
      quant_scale = kMaxValue / amax;
    }
    const float2 quant_scale2 = {quant_scale, quant_scale};
#pragma unroll
    for (uint32_t i = 0; i < kVecSize / 2; ++i) {
      out[i] = WTrait::quant(detail::mul2(float2{in(2 * i), in(2 * i + 1)}, quant_scale2));
    }

    out.store(params.output.get<Q>(expert_idx, token_idx) + group_offset, lane_id);
    params.scale.store<kUe8m0, kRowMajor, kAligned>(expert_idx, token_idx, group_idx, scale_inv);
  }
};




template <typename Trait, bool kUsePDL>
__global__ __launch_bounds__(Trait::kBlockSize) void per_token_group_quant_flat_kernel(
    const __grid_constant__ QuantKernelParams params) {
  using namespace device;
  constexpr uint32_t kNumLanes = Trait::kNumLanes;
  constexpr uint32_t kWorkPerWarp = kWarpThreads / kNumLanes;
  const auto num_groups = params.scale.num_groups;
  const auto global_tid = blockIdx.x * Trait::kBlockSize + threadIdx.x;

  const auto global_warp_id = global_tid / kWarpThreads;
  const auto total_work = params.num_tokens * num_groups;
  if (global_warp_id * kWorkPerWarp >= total_work) return;

  const auto work_id = min(global_tid / kNumLanes, total_work - 1);
  const auto lane_id = threadIdx.x % kNumLanes;
  const auto token_idx = work_id / num_groups;
  const auto group_idx = work_id % num_groups;
  PDLWaitPrimary<kUsePDL>();
  Trait::run(params, 0, token_idx, group_idx, lane_id);
  PDLTriggerSecondary<kUsePDL>();
}





template <typename Trait, bool kUsePDL>
__global__ __launch_bounds__(Trait::kBlockSize) void per_token_group_quant_masked_kernel(
    const __grid_constant__ MaskedQuantKernelParams params) {
  using namespace device;
  constexpr uint32_t kNumLanes = Trait::kNumLanes;
  constexpr uint32_t kWorkPerWarp = kWarpThreads / kNumLanes;
  const auto num_groups = params.base.scale.num_groups;
  const auto expert_idx = blockIdx.y;
  PDLWaitPrimary<kUsePDL>();
  const auto num_expert_tokens = params.masked_m[expert_idx * params.masked_m_stride];
  for (uint32_t global_tid = blockIdx.x * Trait::kBlockSize + threadIdx.x;;
       global_tid += gridDim.x * Trait::kBlockSize) {
    const auto global_warp_id = global_tid / kWarpThreads;
    const auto total_work = num_expert_tokens * num_groups;
    if (global_warp_id * kWorkPerWarp >= total_work) break;
    const auto work_id = min(global_tid / kNumLanes, total_work - 1);
    const auto token_idx = work_id / num_groups;
    const auto group_idx = work_id % num_groups;
    const auto lane_id = threadIdx.x % kNumLanes;
    Trait::run(params.base, expert_idx, token_idx, group_idx, lane_id);
  }
  PDLTriggerSecondary<kUsePDL>();
}









template <typename Trait>
struct QuantHostContext {
  QuantKernelParams params;
  uint32_t num_experts;
  DLDevice device;
};

template <typename Trait, bool kMasked>
QuantHostContext<Trait> build_quant_context(
    const tvm::ffi::TensorView& input,
    const tvm::ffi::TensorView& output_q,
    const tvm::ffi::TensorView& output_s) {
  using namespace host;
  using T = typename Trait::InputType;
  using Q = typename Trait::QuantType;
  using S = std::conditional_t<Trait::kUe8m0, int32_t, float>;
  constexpr int64_t kSiluFactor = Trait::kFuseSiluAndMul ? 2 : 1;

  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  auto E = SymbolicSize{"num_experts"};
  auto N = SymbolicSize{"num_tokens"};
  auto H = SymbolicSize{"hidden_size"};
  auto G = SymbolicSize{"num_scale_groups"};

  if constexpr (kMasked) {
    TensorMatcher({E, N, -1}).with_strides({-1, -1, 1}).with_dtype<T>().with_device(device).verify(input);
    TensorMatcher({E, N, H}).with_strides({-1, -1, 1}).with_dtype<Q>().with_device(device).verify(output_q);
    TensorMatcher({E, N, G}).with_strides({-1, -1, -1}).with_dtype<S>().with_device(device).verify(output_s);
    CHECK_HOST((input.stride(0) * sizeof(T)) % 32 == 0)
        << "input expert stride must keep rows 32B-aligned for the vectorized loads";
  } else {
    TensorMatcher({N, -1}).with_strides({-1, 1}).with_dtype<T>().with_device(device).verify(input);
    TensorMatcher({N, H}).with_strides({-1, 1}).with_dtype<Q>().with_device(device).verify(output_q);
    TensorMatcher({N, G}).with_strides({-1, -1}).with_dtype<S>().with_device(device).verify(output_s);
  }





  CHECK_HOST(reinterpret_cast<uintptr_t>(input.data_ptr()) % 32 == 0)
      << "input base pointer must be 32B-aligned for the vectorized loads";
  CHECK_HOST((input.stride(-2) * sizeof(T)) % 32 == 0)
      << "input token stride must keep rows 32B-aligned for the vectorized loads";

  const uint32_t num_tokens = N.unwrap();
  const uint32_t hidden_size = H.unwrap();
  const uint32_t num_experts = kMasked ? E.unwrap() : 1;
  const uint32_t num_groups = hidden_size / Trait::kGroupSize;
  const uint32_t num_scale_groups = G.unwrap();
  CHECK_HOST(hidden_size % Trait::kGroupSize == 0);
  CHECK_HOST(input.size(-1) == hidden_size * kSiluFactor);
  CHECK_HOST(num_scale_groups == (Trait::kUe8m0 ? div_ceil(num_groups, 4) : num_groups));




  if constexpr (Trait::kUe8m0) {
    CHECK_HOST(Trait::kAligned == (num_groups % 4 == 0));
  }
  auto scale_args = detail::ScaleStoreArgs{
      .base = output_s.data_ptr(),
      .expert_stride = static_cast<uint32_t>(kMasked ? output_s.stride(0) : 0),
      .token_stride = static_cast<uint32_t>(output_s.stride(-2)),
      .group_stride = static_cast<uint32_t>(output_s.stride(-1)),
      .num_groups = static_cast<uint32_t>(num_groups),
  };
  if constexpr (Trait::kRowMajor) {
    CHECK_HOST(scale_args.group_stride == 1);
    if constexpr (Trait::kUe8m0) {
      scale_args.expert_stride *= 4;
      scale_args.token_stride *= 4;
    }
  } else {
    CHECK_HOST(scale_args.token_stride == 1);
    if constexpr (Trait::kUe8m0) {
      scale_args.expert_stride *= 4;
      scale_args.group_stride *= 4;


      scale_args.token_stride = 4;
    }
  }

  scale_args.check_overflow(num_experts, num_tokens);
  const auto input_args = detail::TensorArgs{
      .ptr = input.data_ptr(),
      .expert_stride = kMasked ? input.stride(0) : 0,
      .token_stride = input.stride(-2),
  };
  const auto output_args = detail::TensorArgs{
      .ptr = output_q.data_ptr(),
      .expert_stride = kMasked ? output_q.stride(0) : 0,
      .token_stride = output_q.stride(-2),
  };
  return {
      .params =
          {
              .input = input_args,
              .output = output_args,
              .scale = scale_args,
              .num_tokens = num_tokens,
              .hidden_size = hidden_size,
          },
      .num_experts = num_experts,
      .device = device.unwrap(),
  };
}

template <
    typename InputType,
    typename QuantType,
    uint32_t kGroupSize,
    bool kUe8m0,
    bool kRowMajor,
    bool kAligned,
    bool kFuseSiluAndMul,
    bool kUsePDL>
struct PerTokenGroupQuantFlatKernel {
  using Trait = QuantTrait<InputType, QuantType, kGroupSize, kUe8m0, kRowMajor, kAligned, kFuseSiluAndMul>;

  static void run(tvm::ffi::TensorView input, tvm::ffi::TensorView output_q, tvm::ffi::TensorView output_s) {
    using namespace host;
    const auto ctx = build_quant_context<Trait,             false>(input, output_q, output_s);
    const auto& p = ctx.params;
    const int64_t total_threads = int64_t{p.num_tokens} * p.scale.num_groups * Trait::kNumLanes;
    if (total_threads == 0) return;
    const uint32_t num_blocks = div_ceil(total_threads, int64_t{Trait::kBlockSize});
    LaunchKernel(num_blocks, Trait::kBlockSize, ctx.device)
        .config({.use_pdl = kUsePDL})(per_token_group_quant_flat_kernel<Trait, kUsePDL>, p);
  }
};

template <
    typename InputType,
    typename QuantType,
    uint32_t kGroupSize,
    bool kUe8m0,
    bool kRowMajor,
    bool kAligned,
    bool kFuseSiluAndMul,
    bool kUsePDL>
struct PerTokenGroupQuantMaskedKernel {
  using Trait = QuantTrait<InputType, QuantType, kGroupSize, kUe8m0, kRowMajor, kAligned, kFuseSiluAndMul>;





  static void
  run(tvm::ffi::TensorView input,
      tvm::ffi::TensorView output_q,
      tvm::ffi::TensorView output_s,
      tvm::ffi::TensorView masked_m,
      int32_t expected_m) {
    using namespace host;
    using device::kWarpThreads;
    const auto ctx = build_quant_context<Trait,             true>(input, output_q, output_s);
    auto E = SymbolicSize{"num_experts"};
    E.set_value(ctx.num_experts);
    TensorMatcher({E}).with_dtype<int32_t, int64_t>().with_device(ctx.device).verify(masked_m);


    const uint32_t masked_m_stride = is_type<int64_t>(masked_m.dtype()) ? 2 : 1;
    const auto& p = ctx.params;
    if (p.num_tokens == 0 || ctx.num_experts == 0 || p.scale.num_groups == 0) return;
    const auto params = MaskedQuantKernelParams{p, static_cast<const int32_t*>(masked_m.data_ptr()), masked_m_stride};
    const auto compute_blocks_per_expert = [&](uint32_t num_tokens) -> uint32_t {
      const int64_t total_threads = static_cast<int64_t>(num_tokens) * p.scale.num_groups * Trait::kNumLanes;
      return static_cast<uint32_t>(div_ceil(total_threads, int64_t{Trait::kBlockSize}));
    };
    const auto max_blocks = compute_blocks_per_expert(p.num_tokens);
    constexpr uint32_t kTargetBlocks = 8u * 256;
    const auto target_blocks = div_ceil(kTargetBlocks, ctx.num_experts);
    const uint32_t num_blocks = [&] {
      if (target_blocks >= max_blocks) return max_blocks;
      if (expected_m <= 0) return target_blocks;
      const auto expected = compute_blocks_per_expert(expected_m);
      if (expected > max_blocks) return max_blocks;
      return (expected + target_blocks) / 2;
    }();
    LaunchKernel({num_blocks, ctx.num_experts}, Trait::kBlockSize, ctx.device)
        .config({.use_pdl = kUsePDL})(per_token_group_quant_masked_kernel<Trait, kUsePDL>, params);
  }
};

}
