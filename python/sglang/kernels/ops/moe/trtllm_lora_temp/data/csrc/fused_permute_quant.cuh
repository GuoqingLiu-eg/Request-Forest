



















































#pragma once

#include <cub/cub.cuh>

#include "nv_internal/tensorrt_llm/kernels/quantization_utils.cuh"
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <optional>
#include <type_traits>

namespace sgl_fused_permute_quant {

namespace tk = tensorrt_llm::kernels;





template <typename T, uint32_t BLOCK_SIZE,
          tensorrt_llm::QuantizationSFLayout SF_LAYOUT>
__device__ __forceinline__ void
fused_quant_one_row(uint32_t n, T const *input, int readRow, int writeRow,
                    int numRowsSf, float globalScaleInv, uint8_t *weightOutput,
                    uint8_t *scaleOutput, float *perTokenScaleOutput) {
  constexpr int SF_VEC_SIZE = 16;
  constexpr int ELTS_PER_THREAD = 16;
  using InType = tk::PackedVec<T, ELTS_PER_THREAD>;
  using PackedFp4Type =
      std::conditional_t<ELTS_PER_THREAD == 16, uint64_t, uint32_t>;
  uint32_t const num_vecs_per_row = (n + ELTS_PER_THREAD - 1) / ELTS_PER_THREAD;
  uint32_t const num_sf_vecs_per_row = (n + SF_VEC_SIZE - 1) / SF_VEC_SIZE;
  InType const *inBase = reinterpret_cast<InType const *>(input);


  float localAmax = 0.f;
  for (uint32_t vecIdx = threadIdx.x; vecIdx < num_vecs_per_row;
       vecIdx += BLOCK_SIZE) {
    InType vec_in =
        inBase[static_cast<int64_t>(readRow) * num_vecs_per_row + vecIdx];
    std::remove_reference_t<decltype(vec_in.elts[0])> a(0.f, 0.f);
#pragma unroll
    for (int i = 0; i < ELTS_PER_THREAD / 2; ++i) {
      a = __hmax2(a, __habs2(vec_in.elts[i]));
    }
    localAmax = fmaxf(localAmax, static_cast<float>(__hmax(a.x, a.y)));
  }

  using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  float const globalAmax =
      BlockReduce(tempStorage).Reduce(localAmax, cuda::maximum<>{});







  __shared__ float sEncodeScale;
  if (threadIdx.x == 0) {
    float const globalScale = __fdiv_rn(1.0f, globalScaleInv);
    float const rowEncodeScale =
        globalAmax != 0.0f ? fminf(__fdiv_rn(globalScale, globalAmax), FLT_MAX)
                           : FLT_MAX;
    sEncodeScale = rowEncodeScale != 0.0f ? rowEncodeScale : 1.0f;
  }
  __syncthreads();
  float const globalEncodeScale = sEncodeScale;
  float const perTokenScale = __fdiv_rn(1.0f, globalEncodeScale);
  if (threadIdx.x == 0)
    perTokenScaleOutput[writeRow] = perTokenScale;


  for (uint32_t vecIdx = threadIdx.x; vecIdx < num_vecs_per_row;
       vecIdx += BLOCK_SIZE) {
    InType vec_in =
        inBase[static_cast<int64_t>(readRow) * num_vecs_per_row + vecIdx];
    uint8_t fp8Scale;
    auto fp4Vals = tk::cvt_warp_fp16_to_fp4<T, SF_VEC_SIZE, ELTS_PER_THREAD,
                                                         false,
                                                               true>(
        vec_in, globalEncodeScale, &fp8Scale);
    reinterpret_cast<PackedFp4Type *>(
        weightOutput)[static_cast<int64_t>(writeRow) * num_vecs_per_row +
                      vecIdx] = fp4Vals;

    int64_t sfOffset;
    if constexpr (SF_LAYOUT == tensorrt_llm::QuantizationSFLayout::LINEAR) {
      sfOffset = static_cast<int64_t>(writeRow) * num_sf_vecs_per_row + vecIdx;
    } else if constexpr (SF_LAYOUT ==
                         tensorrt_llm::QuantizationSFLayout::SWIZZLED_128x4) {
      sfOffset = tk::get_sf_out_offset_128x4(             0, writeRow, vecIdx,
                                             numRowsSf, num_sf_vecs_per_row);
    } else {
      sfOffset = tk::get_sf_out_offset_8x4(             0, writeRow, vecIdx,
                                           numRowsSf, num_sf_vecs_per_row);
    }
    scaleOutput[sfOffset] = fp8Scale;
  }
}


template <typename T, uint32_t BLOCK_SIZE,
          tensorrt_llm::QuantizationSFLayout SF_LAYOUT>
__global__ void fusedPermuteNvfp4QuantKernel(
    uint32_t numPairs, uint32_t n, uint32_t topK, int numRowsSf, T const *input,
    float globalScaleInv, int32_t const *expandedIdxToPermutedIdx,
    uint8_t *weightOutput, uint8_t *scaleOutput, float *perTokenScaleOutput) {
  uint32_t const expandedIdx = blockIdx.x;
  if (expandedIdx >= numPairs)
    return;
  int const writeRow = expandedIdxToPermutedIdx[expandedIdx];
  if (writeRow < 0)
    return;
  int const readRow = static_cast<int>(expandedIdx / topK);
  fused_quant_one_row<T, BLOCK_SIZE, SF_LAYOUT>(
      n, input, readRow, writeRow, numRowsSf, globalScaleInv, weightOutput,
      scaleOutput, perTokenScaleOutput);
}



template <typename T, uint32_t BLOCK_SIZE,
          tensorrt_llm::QuantizationSFLayout SF_LAYOUT>
__global__ void fusedPermuteNvfp4QuantDedupKernel(
    uint32_t numTokens, uint32_t n, uint32_t topK, int numRowsSf,
    T const *input, float globalScaleInv,
    int32_t const *expandedIdxToPermutedIdx, uint8_t *weightOutput,
    uint8_t *scaleOutput, float *perTokenScaleOutput) {
  constexpr int SF_VEC_SIZE = 16;
  constexpr int ELTS_PER_THREAD = 16;
  using InType = tk::PackedVec<T, ELTS_PER_THREAD>;
  using PackedFp4Type =
      std::conditional_t<ELTS_PER_THREAD == 16, uint64_t, uint32_t>;
  uint32_t const token = blockIdx.x;
  if (token >= numTokens)
    return;
  uint32_t const num_vecs_per_row = (n + ELTS_PER_THREAD - 1) / ELTS_PER_THREAD;
  uint32_t const num_sf_vecs_per_row = (n + SF_VEC_SIZE - 1) / SF_VEC_SIZE;
  InType const *inBase = reinterpret_cast<InType const *>(input);


  float localAmax = 0.f;
  for (uint32_t vecIdx = threadIdx.x; vecIdx < num_vecs_per_row;
       vecIdx += BLOCK_SIZE) {
    InType vec_in =
        inBase[static_cast<int64_t>(token) * num_vecs_per_row + vecIdx];
    std::remove_reference_t<decltype(vec_in.elts[0])> a(0.f, 0.f);
#pragma unroll
    for (int i = 0; i < ELTS_PER_THREAD / 2; ++i) {
      a = __hmax2(a, __habs2(vec_in.elts[i]));
    }
    localAmax = fmaxf(localAmax, static_cast<float>(__hmax(a.x, a.y)));
  }
  using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  float const globalAmax =
      BlockReduce(tempStorage).Reduce(localAmax, cuda::maximum<>{});


  __shared__ float sEncodeScale;
  if (threadIdx.x == 0) {
    float const globalScale = __fdiv_rn(1.0f, globalScaleInv);
    float const rowEncodeScale =
        globalAmax != 0.0f ? fminf(__fdiv_rn(globalScale, globalAmax), FLT_MAX)
                           : FLT_MAX;
    sEncodeScale = rowEncodeScale != 0.0f ? rowEncodeScale : 1.0f;
  }
  __syncthreads();
  float const globalEncodeScale = sEncodeScale;
  float const perTokenScale = __fdiv_rn(1.0f, globalEncodeScale);



  if (threadIdx.x < topK) {
    int const writeRow = expandedIdxToPermutedIdx[token * topK + threadIdx.x];
    if (writeRow >= 0)
      perTokenScaleOutput[writeRow] = perTokenScale;
  }


  for (uint32_t vecIdx = threadIdx.x; vecIdx < num_vecs_per_row;
       vecIdx += BLOCK_SIZE) {
    InType vec_in =
        inBase[static_cast<int64_t>(token) * num_vecs_per_row + vecIdx];
    uint8_t fp8Scale;
    auto fp4Vals = tk::cvt_warp_fp16_to_fp4<T, SF_VEC_SIZE, ELTS_PER_THREAD,
                                                         false,
                                                               true>(
        vec_in, globalEncodeScale, &fp8Scale);
#pragma unroll 1
    for (uint32_t k = 0; k < topK; ++k) {
      int const writeRow = expandedIdxToPermutedIdx[token * topK + k];
      if (writeRow < 0)
        continue;
      reinterpret_cast<PackedFp4Type *>(
          weightOutput)[static_cast<int64_t>(writeRow) * num_vecs_per_row +
                        vecIdx] = fp4Vals;
      int64_t sfOffset;
      if constexpr (SF_LAYOUT == tensorrt_llm::QuantizationSFLayout::LINEAR) {
        sfOffset =
            static_cast<int64_t>(writeRow) * num_sf_vecs_per_row + vecIdx;
      } else if constexpr (SF_LAYOUT ==
                           tensorrt_llm::QuantizationSFLayout::SWIZZLED_128x4) {
        sfOffset = tk::get_sf_out_offset_128x4(             0, writeRow, vecIdx,
                                               numRowsSf, num_sf_vecs_per_row);
      } else {
        sfOffset = tk::get_sf_out_offset_8x4(             0, writeRow, vecIdx,
                                             numRowsSf, num_sf_vecs_per_row);
      }
      scaleOutput[sfOffset] = fp8Scale;
    }
  }
}



template <typename T>
void invokeFusedPermuteNvfp4Quant(uint32_t numTokens, uint32_t topK, uint32_t n,
                                  int numRowsSf, T const *input,
                                  float globalScaleInv,
                                  int32_t const *expandedIdxToPermutedIdx,
                                  uint8_t *weightOutput, uint8_t *scaleOutput,
                                  float *perTokenScaleOutput,
                                  tensorrt_llm::QuantizationSFLayout sfLayout,
                                  bool dedup, cudaStream_t stream) {








  constexpr uint32_t BLOCK_SIZE = 512;
  dim3 const block(BLOCK_SIZE);

  auto dispatch = [&](auto layoutTag) {
    constexpr tensorrt_llm::QuantizationSFLayout LAYOUT =
        decltype(layoutTag)::value;
    if (dedup) {
      dim3 const grid(numTokens);
      fusedPermuteNvfp4QuantDedupKernel<T, BLOCK_SIZE, LAYOUT>
          <<<grid, block, 0, stream>>>(numTokens, n, topK, numRowsSf, input,
                                       globalScaleInv, expandedIdxToPermutedIdx,
                                       weightOutput, scaleOutput,
                                       perTokenScaleOutput);
    } else {
      dim3 const grid(numTokens * topK);
      fusedPermuteNvfp4QuantKernel<T, BLOCK_SIZE, LAYOUT>
          <<<grid, block, 0, stream>>>(numTokens * topK, n, topK, numRowsSf,
                                       input, globalScaleInv,
                                       expandedIdxToPermutedIdx, weightOutput,
                                       scaleOutput, perTokenScaleOutput);
    }
  };

  if (sfLayout == tensorrt_llm::QuantizationSFLayout::SWIZZLED_128x4) {
    dispatch(std::integral_constant<
             tensorrt_llm::QuantizationSFLayout,
             tensorrt_llm::QuantizationSFLayout::SWIZZLED_128x4>{});
  } else {
    dispatch(std::integral_constant<
             tensorrt_llm::QuantizationSFLayout,
             tensorrt_llm::QuantizationSFLayout::SWIZZLED_8x4>{});
  }
}

}
