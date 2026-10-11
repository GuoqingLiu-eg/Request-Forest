
















#pragma once

#include <cub/block/block_reduce.cuh>
#include <cuda/functional>

#include "nv_internal/tensorrt_llm/kernels/quantization_utils.cuh"
#include <cstdint>
#include <cuda_bf16.h>
#include <optional>

namespace flashinfer {
namespace sgl_fused_act_quant {

namespace tk = tensorrt_llm::kernels;



inline __device__ float fused_silu(float x) { return x / (1.0f + expf(-x)); }




template <uint32_t BLOCK_SIZE, tensorrt_llm::QuantizationSFLayout SF_LAYOUT,
          bool DISABLE_FP4_FAST_MATH>
__global__ void fusedActivationQuantKernel(
    int m,
    int innerHalf,

    int innerDim,
    __nv_bfloat16 const *__restrict__ gateUp,

    __nv_bfloat16 const
        *__restrict__ loraDelta,
    __nv_bfloat16 *__restrict__ loraInputOut,

    int32_t const *__restrict__ expandedIdxToPermutedIdx, float globalScaleInv,
    uint8_t *__restrict__ weightOutput,
    uint8_t *__restrict__ scaleOutput,
    float *__restrict__ perTokenScaleOutput) {
  constexpr int SF_VEC_SIZE = 16;
  using InType =
      tk::PackedVec<__nv_bfloat16, SF_VEC_SIZE>;
  using PackedFp4Type = uint64_t;

  int const expandedIdx = blockIdx.x;
  if (expandedIdx >= m)
    return;
  int const permutedIdx = expandedIdxToPermutedIdx[expandedIdx];
  int const num_vecs_per_row = innerHalf / SF_VEC_SIZE;
  int64_t const liBaseRow = (int64_t)expandedIdx * innerHalf;




  if (permutedIdx < 0) {
    if (loraInputOut != nullptr) {
      InType z;
#pragma unroll
      for (int i = 0; i < SF_VEC_SIZE / 2; ++i)
        z.elts[i] = __float2bfloat162_rn(0.0f);
      for (int vecIdx = threadIdx.x; vecIdx < num_vecs_per_row;
           vecIdx += BLOCK_SIZE) {
        *reinterpret_cast<InType *>(
            &loraInputOut[liBaseRow + (int64_t)vecIdx * SF_VEC_SIZE]) = z;
      }
    }
    return;
  }

  int64_t const permBase =
      (int64_t)permutedIdx * innerDim;
  int64_t const expBase =
      (int64_t)expandedIdx * innerDim;
  (void)DISABLE_FP4_FAST_MATH;





  int const vecIdx = threadIdx.x;
  bool const active = vecIdx < num_vecs_per_row;

  InType vec;
  float localAmax = 0.f;
  if (active) {
    int const h0 = vecIdx * SF_VEC_SIZE;
    __nv_bfloat16 const *g =
        gateUp + permBase + (int64_t)2 * h0;
    __nv_bfloat16 const *dlo =
        loraDelta + expBase + h0;
    __nv_bfloat16 const *dhi =
        loraDelta + expBase + innerHalf + h0;
    __nv_bfloat162 amax2 = __float2bfloat162_rn(0.0f);
    union {
      int4 v[4];
      __nv_bfloat16 b[32];
    } gu;
    union {
      int4 v[2];
      __nv_bfloat16 b[16];
    } dl, dh;
    int4 const *gp = reinterpret_cast<int4 const *>(g);
#pragma unroll
    for (int k = 0; k < 4; ++k)
      gu.v[k] = gp[k];
    if (loraDelta != nullptr) {
      int4 const *dlp = reinterpret_cast<int4 const *>(dlo);
      int4 const *dhp = reinterpret_cast<int4 const *>(dhi);
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        dl.v[k] = dlp[k];
        dh.v[k] = dhp[k];
      }
    }
#pragma unroll
    for (int i = 0; i < SF_VEC_SIZE / 2; ++i) {
      int const j0 = 2 * i, j1 = 2 * i + 1;
      float even0 = (float)gu.b[2 * j0], odd0 = (float)gu.b[2 * j0 + 1];
      float even1 = (float)gu.b[2 * j1], odd1 = (float)gu.b[2 * j1 + 1];
      float a0 = odd0, b0 = even0, a1 = odd1, b1 = even1;
      if (loraDelta != nullptr) {
        a0 += (float)dl.b[j0];
        b0 += (float)dh.b[j0];
        a1 += (float)dl.b[j1];
        b1 += (float)dh.b[j1];
      }
      float act0 = fused_silu(a0) * b0;
      float act1 = fused_silu(a1) * b1;
      __nv_bfloat162 e = __float22bfloat162_rn(make_float2(act0, act1));
      vec.elts[i] = e;
      amax2 = __hmax2(amax2, __habs2(e));
    }
    localAmax = (float)__hmax(amax2.x, amax2.y);
    if (loraInputOut != nullptr) {
      *reinterpret_cast<InType *>(&loraInputOut[liBaseRow + h0]) = vec;
    }
  }



  using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  __shared__ float sScale;
  float const globalAmax =
      BlockReduce(tempStorage).Reduce(localAmax, cuda::maximum<>{});
  if (threadIdx.x == 0) {
    float const pts = globalAmax * globalScaleInv;
    perTokenScaleOutput[permutedIdx] = pts;
    sScale = pts;
  }
  __syncthreads();
  float const globalEncodeScale = tk::reciprocal_approximate_ftz(sScale);



  if (active) {
    uint8_t fp8Scale;



    auto fp4Vals =
        tk::cvt_warp_fp16_to_fp4<__nv_bfloat16, SF_VEC_SIZE, SF_VEC_SIZE, false,
                                 false>(vec, globalEncodeScale, &fp8Scale);
    int64_t const vecOffset = (int64_t)permutedIdx * num_vecs_per_row + vecIdx;
    reinterpret_cast<PackedFp4Type *>(weightOutput)[vecOffset] = fp4Vals;



    int64_t sfOffset;
    if constexpr (SF_LAYOUT == tensorrt_llm::QuantizationSFLayout::LINEAR) {
      sfOffset = vecOffset;
    } else if constexpr (SF_LAYOUT ==
                         tensorrt_llm::QuantizationSFLayout::SWIZZLED_128x4) {
      sfOffset = tk::get_sf_out_offset_128x4(             0, permutedIdx,
                                             vecIdx, m, num_vecs_per_row);
    } else {
      sfOffset = tk::get_sf_out_offset_8x4(             0, permutedIdx, vecIdx,
                                           m, num_vecs_per_row);
    }
    scaleOutput[sfOffset] = fp8Scale;
  }
}



inline void launchFusedActivationQuant(
    int m, int innerHalf, int innerDim, __nv_bfloat16 const *gateUp,
    __nv_bfloat16 const *loraDelta, __nv_bfloat16 *loraInputOut,
    int32_t const *expandedIdxToPermutedIdx, float globalScaleInv,
    uint8_t *weightOutput, uint8_t *scaleOutput, float *perTokenScaleOutput,
    tensorrt_llm::QuantizationSFLayout sfLayout, bool disableFp4FastMath,
    cudaStream_t stream) {



  uint32_t const numVecs = static_cast<uint32_t>(innerHalf) / 16;
  auto dispatchBlock = [&](auto blockTag) {
    constexpr uint32_t BLOCK_SIZE = decltype(blockTag)::value;
    dim3 const grid(m), block(BLOCK_SIZE);
    auto launch = [&](auto layoutTag, auto fastMathTag) {
      fusedActivationQuantKernel<BLOCK_SIZE, decltype(layoutTag)::value,
                                 decltype(fastMathTag)::value>
          <<<grid, block, 0, stream>>>(
              m, innerHalf, innerDim, gateUp, loraDelta, loraInputOut,
              expandedIdxToPermutedIdx, globalScaleInv, weightOutput,
              scaleOutput, perTokenScaleOutput);
    };
    auto withFastMath = [&](auto layoutTag) {
      if (disableFp4FastMath) {
        launch(layoutTag, std::integral_constant<bool, true>{});
      } else {
        launch(layoutTag, std::integral_constant<bool, false>{});
      }
    };
    if (sfLayout == tensorrt_llm::QuantizationSFLayout::SWIZZLED_128x4) {
      withFastMath(std::integral_constant<
                   tensorrt_llm::QuantizationSFLayout,
                   tensorrt_llm::QuantizationSFLayout::SWIZZLED_128x4>{});
    } else if (sfLayout == tensorrt_llm::QuantizationSFLayout::LINEAR) {
      withFastMath(
          std::integral_constant<tensorrt_llm::QuantizationSFLayout,
                                 tensorrt_llm::QuantizationSFLayout::LINEAR>{});
    } else {
      withFastMath(std::integral_constant<
                   tensorrt_llm::QuantizationSFLayout,
                   tensorrt_llm::QuantizationSFLayout::SWIZZLED_8x4>{});
    }
  };
  if (numVecs <= 128) {
    dispatchBlock(std::integral_constant<uint32_t, 128>{});
  } else if (numVecs <= 256) {
    dispatchBlock(std::integral_constant<uint32_t, 256>{});
  } else if (numVecs <= 512) {
    dispatchBlock(std::integral_constant<uint32_t, 512>{});
  } else {

    dispatchBlock(std::integral_constant<uint32_t, 1024>{});
  }
}

}
}
