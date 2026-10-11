#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

struct PlanEntriesParams {

  const int64_t* __restrict__ req_pool_indices;
  const int64_t* __restrict__ prefix_lens;
  const int32_t* __restrict__ req_to_token;
  const int64_t* __restrict__ full_to_swa_lut;
  const int64_t* __restrict__ verify_offsets_scratch;
  const int32_t* __restrict__ verify_enable;
  const int32_t* __restrict__ req_to_verify_expected_tokens;
  const int64_t* __restrict__ req_to_verify_expected_tokens_valid_lens;


  int64_t* __restrict__ out_verify_slot_indices;
  int64_t* __restrict__ out_verify_expected_tokens;
  int64_t* __restrict__ out_verify_expected_positions;
  int64_t* __restrict__ out_verify_prev_slot_indices;

  int32_t bs_padded;
  int64_t verify_capacity;
  int64_t req_to_token_stride0;
  int64_t req_to_verify_expected_tokens_stride0;
  int32_t kv_token_id_vs_position_offset;
  int32_t swa_window_size;
};



SGL_DEVICE int32_t find_req_id(const int64_t* __restrict__ verify_offsets, int32_t bs_padded, int64_t tid) {
  int32_t lo = 0;
  int32_t hi = bs_padded;
  while (hi - lo > 1) {
    const int32_t mid = (lo + hi) >> 1;
    if (verify_offsets[mid] <= tid) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}



SGL_DEVICE int64_t swa_translate(const int64_t* __restrict__ lut, int64_t lut_len, int64_t raw_slot) {
  if (raw_slot < 0) {
    return raw_slot;
  }
  int64_t safe = raw_slot;
  if (lut_len > 0 && safe >= lut_len) {
    safe = lut_len - 1;
  }
  return lut[safe];
}





template <bool HAS_SWA_LUT, bool HAS_VERIFY_EXPECTED_TOKEN_POOL>
__global__ void plan_entries_persistent_kernel(
    const PlanEntriesParams __grid_constant__ params,
    int64_t lut_len
) {
  const int64_t total_verify = params.verify_offsets_scratch[params.bs_padded];
  if (total_verify <= 0) {
    return;
  }

  if (*params.verify_enable == 0) {
    return;
  }



  if (total_verify > params.verify_capacity) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      printf(
          "kv-canary plan_entries: total_verify=%lld exceeds verify_capacity=%lld with "
          "verify_enable=1 (offsets/entries contract broken)\n",
          static_cast<long long>(total_verify),
          static_cast<long long>(params.verify_capacity));
    }
#ifndef USE_ROCM
    __trap();
#else
    __builtin_trap();
#endif
  }

  const int64_t tid_start = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

  const int32_t swa_window = params.swa_window_size;
  const int64_t req_to_token_stride0 = params.req_to_token_stride0;

  for (int64_t tid = tid_start; tid < total_verify; tid += stride) {

    const int32_t req_id = find_req_id(params.verify_offsets_scratch, params.bs_padded, tid);
    const int64_t req_start = params.verify_offsets_scratch[req_id];
    const int64_t entry_idx = tid - req_start;



    const int64_t rp = params.req_pool_indices[req_id];
    const int64_t prefix_len = params.prefix_lens[req_id];
    const int64_t window_start = (swa_window > 0) ? (prefix_len - swa_window > 0 ? prefix_len - swa_window : 0) : 0;
    const int64_t out_position = window_start + entry_idx;


    const int64_t row_base = rp * req_to_token_stride0;
    const int32_t slot_raw = params.req_to_token[row_base + out_position];
    int64_t out_slot;
    if constexpr (HAS_SWA_LUT) {
      out_slot = swa_translate(params.full_to_swa_lut, lut_len, static_cast<int64_t>(slot_raw));
    } else {
      out_slot = static_cast<int64_t>(slot_raw);
    }

    int64_t out_prev_slot;
    if (out_position > 0) {
      const int32_t prev_raw = params.req_to_token[row_base + out_position - 1];
      if constexpr (HAS_SWA_LUT) {
        out_prev_slot = swa_translate(params.full_to_swa_lut, lut_len, static_cast<int64_t>(prev_raw));
      } else {
        out_prev_slot = static_cast<int64_t>(prev_raw);
      }
    } else {
      out_prev_slot = -1;
    }

    int64_t out_expected_input_id = -1;
    if constexpr (HAS_VERIFY_EXPECTED_TOKEN_POOL) {
      const int64_t sot_pos = out_position + static_cast<int64_t>(params.kv_token_id_vs_position_offset);
      const int64_t valid_len = params.req_to_verify_expected_tokens_valid_lens[req_id];
      if (sot_pos >= 0 && sot_pos < valid_len) {
        const int32_t token =
            params.req_to_verify_expected_tokens[rp * params.req_to_verify_expected_tokens_stride0 + sot_pos];
        out_expected_input_id = static_cast<int64_t>(token);
      }
    }


    params.out_verify_slot_indices[tid] = out_slot;
    params.out_verify_expected_tokens[tid] = out_expected_input_id;
    params.out_verify_expected_positions[tid] = out_position;
    params.out_verify_prev_slot_indices[tid] = out_prev_slot;
  }
}

template <bool HAS_SWA_LUT, bool HAS_VERIFY_EXPECTED_TOKEN_POOL>
struct PlanEntriesKernel {
  static constexpr int kBlockSize = 128;
  static constexpr int kBlocksPerSm = 8;

  static auto get_num_sms(DLDevice device) {
    static const auto kNumSM = host::runtime::get_sm_count(device.device_id);
    return kNumSM;
  }

  static void
  run(const tvm::ffi::TensorView req_pool_indices,
      const tvm::ffi::TensorView prefix_lens,
      const tvm::ffi::TensorView req_to_token,
      const tvm::ffi::Optional<tvm::ffi::TensorView> full_to_swa_index_mapping,
      const tvm::ffi::TensorView verify_offsets_scratch,
      const tvm::ffi::TensorView verify_enable,
      const tvm::ffi::Optional<tvm::ffi::TensorView> req_to_verify_expected_tokens,
      const tvm::ffi::Optional<tvm::ffi::TensorView> req_to_verify_expected_tokens_valid_lens,
      const tvm::ffi::TensorView out_verify_slot_indices,
      const tvm::ffi::TensorView out_verify_expected_tokens,
      const tvm::ffi::TensorView out_verify_expected_positions,
      const tvm::ffi::TensorView out_verify_prev_slot_indices,
      int32_t kv_token_id_vs_position_offset,
      int32_t swa_window_size) {
    using namespace host;

    SymbolicSize Nbs = {"bs_padded"};
    SymbolicSize Nscratch = {"verify_offsets_scratch_len"};
    SymbolicSize Ncap = {"verify_capacity"};
    SymbolicSize Nmax_reqs = {"max_reqs"};
    SymbolicSize Nmax_seq_len = {"max_seq_len"};
    SymbolicSize Nlut = {"lut_len"};
    SymbolicSize Npool_rows = {"req_to_verify_expected_tokens_rows"};
    SymbolicSize Npool_cols = {"req_to_verify_expected_tokens_cols"};
    SymbolicDevice device_;
    device_.set_options<kDLGPU>();

    TensorMatcher({Nbs})
        .with_dtype<int64_t>()
        .with_device<kDLGPU>(device_)
        .verify(req_pool_indices)
        .verify(prefix_lens);
    TensorMatcher({Nscratch})
        .with_dtype<int64_t>()
        .with_device<kDLGPU>(device_)
        .verify(verify_offsets_scratch);
    TensorMatcher({1})
        .with_dtype<int32_t>()
        .with_device<kDLGPU>(device_)
        .verify(verify_enable);
    TensorMatcher({Ncap})
        .with_dtype<int64_t>()
        .with_device<kDLGPU>(device_)
        .verify(out_verify_slot_indices)
        .verify(out_verify_expected_tokens)
        .verify(out_verify_expected_positions)
        .verify(out_verify_prev_slot_indices);
    TensorMatcher({Nmax_reqs, Nmax_seq_len})
        .with_dtype<int32_t>()
        .with_device<kDLGPU>(device_)
        .verify(req_to_token);
    RuntimeCheck(
        full_to_swa_index_mapping.has_value() == HAS_SWA_LUT,
        "full_to_swa_index_mapping presence does not match HAS_SWA_LUT specialization");
    RuntimeCheck(
        req_to_verify_expected_tokens.has_value() == HAS_VERIFY_EXPECTED_TOKEN_POOL,
        "req_to_verify_expected_tokens presence does not match HAS_VERIFY_EXPECTED_TOKEN_POOL specialization");
    if constexpr (HAS_VERIFY_EXPECTED_TOKEN_POOL) {
      RuntimeCheck(
          req_to_verify_expected_tokens_valid_lens.has_value(),
          "req_to_verify_expected_tokens_valid_lens must be set when req_to_verify_expected_tokens is set");
    }
    if constexpr (HAS_SWA_LUT) {
      TensorMatcher({Nlut})
          .with_dtype<int64_t>()
          .with_device<kDLGPU>(device_)
          .verify(full_to_swa_index_mapping.value());
    }
    if constexpr (HAS_VERIFY_EXPECTED_TOKEN_POOL) {
      TensorMatcher({Npool_rows, Npool_cols})
          .with_dtype<int32_t>()
          .with_device<kDLGPU>(device_)
          .verify(req_to_verify_expected_tokens.value());
      TensorMatcher({Nbs})
          .with_dtype<int64_t>()
          .with_device<kDLGPU>(device_)
          .verify(req_to_verify_expected_tokens_valid_lens.value());
    }
    RuntimeCheck(Nscratch.unwrap() >= Nbs.unwrap() + 1, "verify_offsets_scratch length must be >= bs_padded + 1");

    const int64_t bs_padded = Nbs.unwrap();
    if (bs_padded <= 0) {
      return;
    }

    const int64_t* lut_ptr = nullptr;
    int64_t lut_len = 0;
    if constexpr (HAS_SWA_LUT) {
      lut_ptr = static_cast<const int64_t*>(full_to_swa_index_mapping.value().data_ptr());
      lut_len = static_cast<int64_t>(Nlut.unwrap());
    }

    const int32_t* expected_token_ids_ptr = nullptr;
    int64_t expected_token_ids_stride0 = 0;
    const int64_t* req_to_verify_expected_tokens_valid_lens_ptr = nullptr;
    if constexpr (HAS_VERIFY_EXPECTED_TOKEN_POOL) {
      expected_token_ids_ptr = static_cast<const int32_t*>(req_to_verify_expected_tokens.value().data_ptr());
      expected_token_ids_stride0 = static_cast<int64_t>(Npool_cols.unwrap());
      req_to_verify_expected_tokens_valid_lens_ptr =
          static_cast<const int64_t*>(req_to_verify_expected_tokens_valid_lens.value().data_ptr());
    }

    const PlanEntriesParams params = PlanEntriesParams{
        .req_pool_indices = static_cast<const int64_t*>(req_pool_indices.data_ptr()),
        .prefix_lens = static_cast<const int64_t*>(prefix_lens.data_ptr()),
        .req_to_token = static_cast<const int32_t*>(req_to_token.data_ptr()),
        .full_to_swa_lut = lut_ptr,
        .verify_offsets_scratch = static_cast<const int64_t*>(verify_offsets_scratch.data_ptr()),
        .verify_enable = static_cast<const int32_t*>(verify_enable.data_ptr()),
        .req_to_verify_expected_tokens = expected_token_ids_ptr,
        .req_to_verify_expected_tokens_valid_lens = req_to_verify_expected_tokens_valid_lens_ptr,
        .out_verify_slot_indices = static_cast<int64_t*>(out_verify_slot_indices.data_ptr()),
        .out_verify_expected_tokens = static_cast<int64_t*>(out_verify_expected_tokens.data_ptr()),
        .out_verify_expected_positions = static_cast<int64_t*>(out_verify_expected_positions.data_ptr()),
        .out_verify_prev_slot_indices = static_cast<int64_t*>(out_verify_prev_slot_indices.data_ptr()),
        .bs_padded = static_cast<int32_t>(bs_padded),
        .verify_capacity = static_cast<int64_t>(Ncap.unwrap()),
        .req_to_token_stride0 = static_cast<int64_t>(Nmax_seq_len.unwrap()),
        .req_to_verify_expected_tokens_stride0 = expected_token_ids_stride0,
        .kv_token_id_vs_position_offset = kv_token_id_vs_position_offset,
        .swa_window_size = swa_window_size,
    };

    const DLDevice device = device_.unwrap();
    const int num_sms = get_num_sms(device);
    const int num_blocks = num_sms * kBlocksPerSm;

    const dim3 grid(num_blocks);
    const dim3 block(kBlockSize);

    LaunchKernel(grid, block, device)(
        plan_entries_persistent_kernel<HAS_SWA_LUT, HAS_VERIFY_EXPECTED_TOKEN_POOL>, params, lut_len);
  }
};

}
