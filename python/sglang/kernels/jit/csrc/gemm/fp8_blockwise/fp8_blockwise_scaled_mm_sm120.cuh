/* Copyright 2026 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>


#include "cutlass/cutlass.h"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/util/packed_stride.hpp"


namespace sglang {

using namespace host;

#define CUTLASS_CHECK(status)                                                        \
  {                                                                                  \
    cutlass::Status error = status;                                                  \
    RuntimeCheck(error == cutlass::Status::kSuccess, cutlassGetStatusString(error)); \
  }

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

template <
    typename OutType,
    typename MmaTileShape,
    typename PerSmTileShape,
    typename EpilogueTileShape,
    typename ScalesPerTile,
    int TileSizeM_ = 128,
    class ClusterShape = Shape<_1, _1, _1>>
void launch_sm120_fp8_blockwise_scaled_mm(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView scales_a,
    tvm::ffi::TensorView scales_b,
    cudaStream_t stream) {
  using ElementBlockScale = float;


  using ElementA = cutlass::float_e4m3_t;
  using LayoutATag = cutlass::layout::RowMajor;
  constexpr int AlignmentA =
      128 / cutlass::sizeof_bits<ElementA>::value;



  using ElementB = cutlass::float_e4m3_t;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  constexpr int AlignmentB =
      128 / cutlass::sizeof_bits<ElementB>::value;



  using ElementD = OutType;
  using ElementC = void;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  constexpr int AlignmentD =
      128 / cutlass::sizeof_bits<ElementD>::value;

  constexpr int AlignmentC =
      AlignmentD;


  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  static constexpr int ScaleMsPerTile = size<0>(ScalesPerTile{});
  static constexpr int ScaleGranularityM = size<0>(MmaTileShape{}) / ScaleMsPerTile;
  static constexpr int ScaleGranularityN = size<1>(MmaTileShape{}) / size<1>(ScalesPerTile{});
  static constexpr int ScaleGranularityK = size<2>(MmaTileShape{}) / size<2>(ScalesPerTile{});

  using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<
      ScaleGranularityM,
      ScaleGranularityN,
      ScaleGranularityK,
      cute::UMMA::Major::MN,
      cute::UMMA::Major::K>;

  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  constexpr bool kCanUsePingpong = (64 % ScaleGranularityM == 0);

  int m = a.size(0);
  int k = a.size(1);
  int n = b.size(1);

  auto a_ptr = static_cast<ElementA*>(a.data_ptr());
  auto b_ptr = static_cast<ElementB*>(b.data_ptr());
  auto c_ptr = static_cast<ElementD*>(out.data_ptr());

  auto scales_a_ptr = static_cast<ElementBlockScale*>(scales_a.data_ptr());
  auto scales_b_ptr = static_cast<ElementBlockScale*>(scales_b.data_ptr());

  LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
  LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));

  auto run_gemm = [&](auto tag) -> cutlass::Status {
    using GemmKernel = decltype(tag);
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    Gemm gemm_op;

    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideC = typename GemmKernel::StrideD;

    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
    StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));

    typename GemmKernel::MainloopArguments mainloop_args{
        a_ptr, stride_a, b_ptr, stride_b, scales_a_ptr, layout_SFA, scales_b_ptr, layout_SFB};

    typename GemmKernel::EpilogueArguments epilogue_args{{}, c_ptr, stride_c, c_ptr, stride_c};
    epilogue_args.thread.alpha = 1.0f;

    typename Gemm::Arguments args = {
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1},
        mainloop_args,
        epilogue_args,
    };

    auto can_implement = gemm_op.can_implement(args);
    if (can_implement != cutlass::Status::kSuccess) {
      return can_implement;
    }

    size_t workspace_size = gemm_op.get_workspace_size(args);
    auto workspace_tensor = host::ffi::alloc_workspace_tensor(workspace_size, a.device());
    void* workspace = (workspace_size == 0) ? nullptr : workspace_tensor.data_ptr();

    auto init_status = gemm_op.initialize(args, workspace, stream);
    if (init_status != cutlass::Status::kSuccess) {
      return init_status;
    }

    return gemm_op.run(stream);
  };

  using CooperativeCollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      PerSmTileShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementAccumulator,
      ElementC,
      LayoutCTag,
      AlignmentC,
      ElementD,
      LayoutDTag,
      AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CooperativeStageCount = cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
      sizeof(typename CooperativeCollectiveEpilogue::SharedStorage))>;

  using CooperativeCollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA,
      cute::tuple<LayoutATag, LayoutSFA>,
      AlignmentA,
      ElementB,
      cute::tuple<LayoutBTag, LayoutSFB>,
      AlignmentB,
      ElementAccumulator,
      MmaTileShape,
      ClusterShape,
      CooperativeStageCount,
      cutlass::gemm::KernelScheduleSm120Blockwise>::CollectiveOp;

  using CooperativeGemmKernelStreamK = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CooperativeCollectiveMainloop,
      CooperativeCollectiveEpilogue,
      cutlass::gemm::StreamKScheduler>;
  using CooperativeGemmKernelVoid = cutlass::gemm::kernel::
      GemmUniversal<Shape<int, int, int, int>, CooperativeCollectiveMainloop, CooperativeCollectiveEpilogue, void>;

  auto run_cooperative = [&]() -> cutlass::Status {
    static const uint32_t kNumSM = host::runtime::get_sm_count(a.device().device_id);
    constexpr int kTileM = size<0>(MmaTileShape{});
    constexpr int kTileN = size<1>(MmaTileShape{});
    uint64_t tiles = static_cast<uint64_t>((m + kTileM - 1) / kTileM) * ((n + kTileN - 1) / kTileN);
    uint32_t last_wave = static_cast<uint32_t>(tiles % kNumSM);
    if (last_wave == 0) last_wave = kNumSM;
    float waste = 1.0f - static_cast<float>(last_wave) / static_cast<float>(kNumSM);
    return (waste > 0.5f) ? run_gemm(CooperativeGemmKernelStreamK{}) : run_gemm(CooperativeGemmKernelVoid{});
  };

  cutlass::Status status = cutlass::Status::kSuccess;
  if constexpr (kCanUsePingpong) {
    using PingpongMmaTileShape_MNK = Shape<_64, _128, _128>;
    using PingpongCollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag,
        OperatorClass,
        PerSmTileShape,
        ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator,
        ElementAccumulator,
        ElementC,
        LayoutCTag,
        AlignmentC,
        ElementD,
        LayoutDTag,
        AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

    using PingpongStageCount = cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename PingpongCollectiveEpilogue::SharedStorage))>;

    using PingpongCollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag,
        OperatorClass,
        ElementA,
        cute::tuple<LayoutATag, LayoutSFA>,
        AlignmentA,
        ElementB,
        cute::tuple<LayoutBTag, LayoutSFB>,
        AlignmentB,
        ElementAccumulator,
        PingpongMmaTileShape_MNK,
        ClusterShape,
        PingpongStageCount,
        cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120>::CollectiveOp;

    using PingpongGemmKernel = cutlass::gemm::kernel::
        GemmUniversal<Shape<int, int, int, int>, PingpongCollectiveMainloop, PingpongCollectiveEpilogue, void>;

    if (m <= 64) {
      status = run_gemm(PingpongGemmKernel{});
      if (status != cutlass::Status::kSuccess) {
        status = run_cooperative();
      }
    } else {
      status = run_cooperative();
    }
  } else {
    status = run_cooperative();
  }

  CUTLASS_CHECK(status);
}




template <
    typename OutType,
    typename MmaTileShape,
    typename PerSmTileShape,
    typename EpilogueTileShape,
    typename ScalesPerTile,
    class ClusterShape = Shape<_1, _1, _1>>
void launch_sm120_fp8_blockwise_scaled_mm_swapab(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView scales_a,
    tvm::ffi::TensorView scales_b,
    cudaStream_t stream) {
  using ElementBlockScale = float;

  using ElementA = cutlass::float_e4m3_t;
  using LayoutATag = cutlass::layout::RowMajor;
  constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  using ElementB = cutlass::float_e4m3_t;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  using ElementD = OutType;
  using ElementC = void;
  using LayoutCTag = cutlass::layout::ColumnMajor;
  using LayoutDTag = cutlass::layout::ColumnMajor;
  constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  constexpr int AlignmentC = AlignmentD;

  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  static constexpr int ScaleMsPerTile = size<0>(ScalesPerTile{});
  static constexpr int ScaleGranularityM = size<0>(MmaTileShape{}) / ScaleMsPerTile;
  static constexpr int ScaleGranularityN = size<1>(MmaTileShape{}) / size<1>(ScalesPerTile{});
  static constexpr int ScaleGranularityK = size<2>(MmaTileShape{}) / size<2>(ScalesPerTile{});



  using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<
      ScaleGranularityM,
      ScaleGranularityN,
      ScaleGranularityK,
      cute::UMMA::Major::K,
      cute::UMMA::Major::MN>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  int m = a.size(0);
  int k = a.size(1);
  int n = b.size(1);

  auto weight_ptr = static_cast<ElementA*>(b.data_ptr());
  auto act_ptr = static_cast<ElementB*>(a.data_ptr());
  auto c_ptr = static_cast<ElementD*>(out.data_ptr());
  auto weight_scale_ptr = static_cast<ElementBlockScale*>(scales_b.data_ptr());
  auto act_scale_ptr = static_cast<ElementBlockScale*>(scales_a.data_ptr());


  LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(n, m, k, 1));
  LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(n, m, k, 1));

  auto run_gemm = [&](auto tag) -> cutlass::Status {
    using GemmKernel = decltype(tag);
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    Gemm gemm_op;

    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideC = typename GemmKernel::StrideD;

    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(n, k, 1));
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(m, k, 1));
    StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(n, m, 1));

    typename GemmKernel::MainloopArguments mainloop_args{
        weight_ptr, stride_a, act_ptr, stride_b, weight_scale_ptr, layout_SFA, act_scale_ptr, layout_SFB};

    typename GemmKernel::EpilogueArguments epilogue_args{{}, c_ptr, stride_c, c_ptr, stride_c};
    epilogue_args.thread.alpha = 1.0f;

    typename Gemm::Arguments args = {
        cutlass::gemm::GemmUniversalMode::kGemm,
        {n, m, k, 1},
        mainloop_args,
        epilogue_args,
    };

    auto can_implement = gemm_op.can_implement(args);
    if (can_implement != cutlass::Status::kSuccess) {
      return can_implement;
    }

    size_t workspace_size = gemm_op.get_workspace_size(args);
    auto workspace_tensor = host::ffi::alloc_workspace_tensor(workspace_size, a.device());
    void* workspace = (workspace_size == 0) ? nullptr : workspace_tensor.data_ptr();

    auto init_status = gemm_op.initialize(args, workspace, stream);
    if (init_status != cutlass::Status::kSuccess) {
      return init_status;
    }

    return gemm_op.run(stream);
  };

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      PerSmTileShape,
      ClusterShape,
      EpilogueTileShape,
      ElementAccumulator,
      ElementAccumulator,
      ElementC,
      LayoutCTag,
      AlignmentC,
      ElementD,
      LayoutDTag,
      AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using StageCount = cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
      sizeof(typename CollectiveEpilogue::SharedStorage))>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA,
      cute::tuple<LayoutATag, LayoutSFA>,
      AlignmentA,
      ElementB,
      cute::tuple<LayoutBTag, LayoutSFB>,
      AlignmentB,
      ElementAccumulator,
      MmaTileShape,
      ClusterShape,
      StageCount,
      cutlass::gemm::KernelScheduleSm120Blockwise>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;

  CUTLASS_CHECK(run_gemm(GemmKernel{}));
}




template <typename OutType>
void sm120_fp8_blockwise_dispatch_shape(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView scales_a,
    tvm::ffi::TensorView scales_b,
    cudaStream_t stream) {
  const int m = a.size(0);
  using EpilogueTileShape = Shape<_128, _64>;








  if (m <= 128 || (m % 4 != 0)) {
    launch_sm120_fp8_blockwise_scaled_mm_swapab<
        OutType,
        Shape<_128, _32, _128>,
        Shape<_128, _32, _128>,
        cutlass::epilogue::collective::EpilogueTileAuto,
        Shape<_1, _32, _1>>(out, a, b, scales_a, scales_b, stream);
    return;
  }




  if (m <= 256) {
    launch_sm120_fp8_blockwise_scaled_mm_swapab<
        OutType,
        Shape<_128, _64, _128>,
        Shape<_128, _64, _128>,
        Shape<_128, _32>,
        Shape<_1, _64, _1>>(out, a, b, scales_a, scales_b, stream);
    return;
  }

  using MmaTileShape = Shape<_128, _128, _128>;
  using PerSmTileShape = Shape<_128, _128, _128>;
  using ScalesPerTile = Shape<_128, _1, _1>;
  launch_sm120_fp8_blockwise_scaled_mm<OutType, MmaTileShape, PerSmTileShape, EpilogueTileShape, ScalesPerTile>(
      out, a, b, scales_a, scales_b, stream);
}

inline void fp8_blockwise_scaled_mm_sm120(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView mat_a,
    tvm::ffi::TensorView mat_b,
    tvm::ffi::TensorView scales_a,
    tvm::ffi::TensorView scales_b) {
  RuntimeCheck(mat_a.device().device_type == kDLCUDA, "mat_a must be a CUDA tensor");
  RuntimeCheck(mat_b.device().device_type == kDLCUDA, "mat_b must be a CUDA tensor");

  RuntimeCheck(mat_a.dim() == 2, "mat_a must be a 2D tensor");
  RuntimeCheck(mat_b.dim() == 2, "mat_b must be a 2D tensor");
  RuntimeCheck(mat_a.stride(1) == 1, "mat_a must be a row major tensor");
  RuntimeCheck(mat_b.stride(0) == 1, "mat_b must be a column major tensor");
  RuntimeCheck(mat_a.size(1) == mat_b.size(0), "mat_a and mat_b shapes cannot be multiplied");

  RuntimeCheck(
      (mat_a.size(1) * (mat_a.dtype().bits / 8)) % 16 == 0, "mat_a must be multiple of 16 bytes for memory alignment");
  RuntimeCheck(
      (mat_b.size(0) * (mat_b.dtype().bits / 8)) % 16 == 0, "mat_b must be multiple of 16 bytes for memory alignment");
  RuntimeCheck(host::is_type<fp8_e4m3_t>(mat_a.dtype()), "mat_a must be Float8_e4m3fn");
  RuntimeCheck(host::is_type<fp8_e4m3_t>(mat_b.dtype()), "mat_b must be Float8_e4m3fn");

  RuntimeCheck(mat_a.size(0) == scales_a.size(0), "size of scales_a is not matched");
  RuntimeCheck(mat_a.size(1) / 128 == scales_a.size(1), "size of scales_a is not matched");
  RuntimeCheck(mat_b.size(0) / 128 == scales_b.size(0), "size of scales_b is not matched");
  RuntimeCheck(mat_b.size(1) / 128 == scales_b.size(1), "size of scales_b is not matched");
  RuntimeCheck(host::is_type<float>(scales_a.dtype()), "scales_a must be Float32");
  RuntimeCheck(host::is_type<float>(scales_b.dtype()), "scales_b must be Float32");

  RuntimeCheck(
      (out.size(1) * (out.dtype().bits / 8)) % 16 == 0, "out must be multiple of 16 bytes for memory alignment");

  const cudaStream_t stream = LaunchKernel::resolve_device(mat_a.device());

  if (host::is_type<bf16_t>(out.dtype())) {
    sm120_fp8_blockwise_dispatch_shape<cutlass::bfloat16_t>(out, mat_a, mat_b, scales_a, scales_b, stream);
  } else if (host::is_type<fp16_t>(out.dtype())) {
    sm120_fp8_blockwise_dispatch_shape<cutlass::half_t>(out, mat_a, mat_b, scales_a, scales_b, stream);
  } else {
    Panic("out_dtype must be Half or BFloat16");
  }
}

#endif

}
