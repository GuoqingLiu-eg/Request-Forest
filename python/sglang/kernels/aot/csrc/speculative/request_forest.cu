#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <tuple>

#ifndef REQUEST_FOREST_PACKED_TREE_METADATA
#define REQUEST_FOREST_PACKED_TREE_METADATA 1
#endif

namespace {

enum class ForestTerminationReason : int8_t {
  kTargetMiss = 0,
  kLeafExhausted = 1,
  kBudgetBoundary = 2,
  kNoCandidates = 3,
  kEos = 4,
};

template <typename cache_index_t>
__global__ void build_tree_attention_metadata_kernel(
    const int32_t* __restrict__ ancestor_row_ptr,
    const int64_t* __restrict__ ancestor_columns,
    const cache_index_t* __restrict__ out_cache_loc,
    int32_t* __restrict__ page_table,
    int32_t* __restrict__ cache_seqlens,
    int32_t* __restrict__ cu_seqlens_q,
    int32_t logical_rows,
    int32_t max_tree_depth) {
  const int32_t row = blockIdx.x;
  if (row >= logical_rows) {
    return;
  }

  const int32_t start = ancestor_row_ptr[row];
  const int32_t length = ancestor_row_ptr[row + 1] - start;
  if (threadIdx.x == 0) {
    cache_seqlens[row] = length;
    cu_seqlens_q[row] = row;
    if (row + 1 == logical_rows) {
      cu_seqlens_q[logical_rows] = logical_rows;
    }
  }

  int32_t* const output_row =
      page_table + static_cast<int64_t>(row) * max_tree_depth;
  for (int32_t depth = threadIdx.x; depth < max_tree_depth;
       depth += blockDim.x) {
    output_row[depth] =
        depth < length
        ? static_cast<int32_t>(
              out_cache_loc[ancestor_columns[start + depth]])
        : 0;
  }
}

// Sub-warp row packing for the measured large, shallow-tree regime. Small row
// counts and depths above 32 retain the original one-block-per-row kernel.
template <typename cache_index_t>
__global__ void build_tree_attention_metadata_packed_kernel(
    const int32_t* __restrict__ ancestor_row_ptr,
    const int64_t* __restrict__ ancestor_columns,
    const cache_index_t* __restrict__ out_cache_loc,
    int32_t* __restrict__ page_table,
    int32_t* __restrict__ cache_seqlens,
    int32_t* __restrict__ cu_seqlens_q,
    int32_t logical_rows,
    int32_t max_tree_depth,
    int32_t row_lane_shift) {
  const int32_t lanes_per_row = 1 << row_lane_shift;
  const int32_t lane = threadIdx.x & (lanes_per_row - 1);
  const int64_t row_index =
      static_cast<int64_t>(blockIdx.x) * (blockDim.x >> row_lane_shift) +
      (threadIdx.x >> row_lane_shift);
  if (row_index >= logical_rows) {
    return;
  }
  const int32_t row = static_cast<int32_t>(row_index);
  const int32_t start = ancestor_row_ptr[row];
  const int32_t length = ancestor_row_ptr[row + 1] - start;
  if (lane == 0) {
    cache_seqlens[row] = length;
    cu_seqlens_q[row] = row;
    if (row + 1 == logical_rows) {
      cu_seqlens_q[logical_rows] = logical_rows;
    }
  }
  if (lane < max_tree_depth) {
    page_table[row_index * max_tree_depth + lane] =
        lane < length
        ? static_cast<int32_t>(out_cache_loc[ancestor_columns[start + lane]])
        : 0;
  }
}

__device__ bool request_forest_is_eos(
    int64_t token_id,
    const int64_t* __restrict__ eos_token_ids,
    int32_t eos_token_count) {
  for (int32_t index = 0; index < eos_token_count; ++index) {
    if (eos_token_ids[index] == token_id) {
      return true;
    }
  }
  return false;
}

__global__ void request_forest_verify_greedy_kernel(
    int32_t* __restrict__ predicts,
    int32_t* __restrict__ output_token_ids,
    int32_t* __restrict__ accept_index,
    int32_t* __restrict__ accepted_draft_lens,
    int32_t* __restrict__ accept_lens,
    int64_t* __restrict__ first_miss_positions,
    int8_t* __restrict__ termination_reasons,
    const int64_t* __restrict__ draft_token,
    const int32_t* __restrict__ first_child,
    const int32_t* __restrict__ next_sibling,
    const int64_t* __restrict__ positions,
    const bool* __restrict__ has_unverified_children,
    const int32_t* __restrict__ request_offsets,
    const int32_t* __restrict__ anchor_rows,
    const int64_t* __restrict__ target_predict,
    const bool* __restrict__ request_ignore_eos,
    const int64_t* __restrict__ eos_token_ids,
    int32_t request_count,
    int32_t max_tree_depth,
    int32_t eos_token_count) {
  const int32_t request = blockIdx.x;
  if (request >= request_count) {
    return;
  }

  const int32_t row_start = request_offsets[request];
  const int32_t row_end = request_offsets[request + 1];
  for (int32_t row = row_start + threadIdx.x; row < row_end;
       row += blockDim.x) {
    predicts[row] = -1;
  }
  const int64_t request_output_start =
      static_cast<int64_t>(request) * max_tree_depth;
  for (int32_t column = threadIdx.x; column < max_tree_depth;
       column += blockDim.x) {
    output_token_ids[request_output_start + column] = -1;
    accept_index[request_output_start + column] = -1;
  }
  if (threadIdx.x == 0) {
    first_miss_positions[request] = -1;
  }
  __syncthreads();

  if (threadIdx.x != 0) {
    return;
  }

  const int32_t anchor = anchor_rows[request];
  int32_t current_row = anchor;
  int32_t accepted_drafts = 0;
  bool accepted_eos = false;
  const bool ignore_eos = request_ignore_eos[request];
  accept_index[request_output_start] = anchor;

  for (int32_t step = 1; step < max_tree_depth; ++step) {
    const int64_t decision = target_predict[current_row];
    int32_t matching_child = first_child[current_row];
    while (matching_child != -1 &&
           draft_token[matching_child] != decision) {
      matching_child = next_sibling[matching_child];
    }
    if (matching_child == -1) {
      break;
    }

    predicts[current_row] = static_cast<int32_t>(decision);
    output_token_ids[request_output_start + accepted_drafts] =
        static_cast<int32_t>(decision);
    ++accepted_drafts;
    accept_index[request_output_start + accepted_drafts] = matching_child;
    current_row = matching_child;
    if (!ignore_eos &&
        request_forest_is_eos(decision, eos_token_ids, eos_token_count)) {
      accepted_eos = true;
      break;
    }
  }

  accepted_draft_lens[request] = accepted_drafts;
  if (accepted_eos) {
    accept_lens[request] = accepted_drafts;
    termination_reasons[request] =
        static_cast<int8_t>(ForestTerminationReason::kEos);
    return;
  }

  const int64_t correction = target_predict[current_row];
  predicts[current_row] = static_cast<int32_t>(correction);
  output_token_ids[request_output_start + accepted_drafts] =
      static_cast<int32_t>(correction);
  accept_lens[request] = accepted_drafts + 1;
  first_miss_positions[request] = positions[current_row] + 1;

  ForestTerminationReason reason;
  if (!ignore_eos &&
      request_forest_is_eos(correction, eos_token_ids, eos_token_count)) {
    reason = ForestTerminationReason::kEos;
  } else if (first_child[current_row] != -1) {
    reason = ForestTerminationReason::kTargetMiss;
  } else if (has_unverified_children[current_row]) {
    reason = ForestTerminationReason::kBudgetBoundary;
  } else if (current_row == anchor) {
    reason = ForestTerminationReason::kNoCandidates;
  } else {
    reason = ForestTerminationReason::kLeafExhausted;
  }
  termination_reasons[request] = static_cast<int8_t>(reason);
}

__device__ __forceinline__ int64_t request_forest_load_index(
    const void* __restrict__ values,
    bool is_int64,
    int64_t index) {
  return is_int64
      ? static_cast<const int64_t*>(values)[index]
      : static_cast<int64_t>(static_cast<const int32_t*>(values)[index]);
}

__device__ __forceinline__ void request_forest_store_index(
    void* __restrict__ values,
    bool is_int64,
    int64_t index,
    int64_t value) {
  if (is_int64) {
    static_cast<int64_t*>(values)[index] = value;
  } else {
    static_cast<int32_t*>(values)[index] = static_cast<int32_t>(value);
  }
}

// Commit is deliberately one block: request forests have at most a few
// thousand physical rows, while avoiding launch-heavy mask/gather/scatter
// chains matters more than saturating the device.  The shared bitset records
// accepted physical rows. Warp ballots compact rejected slots in stable row
// order so allocator behavior remains identical to the former Torch path.
__global__ void request_forest_commit_kv_kernel(
    int64_t* __restrict__ committed_rows,
    void* __restrict__ committed_slots,
    int32_t* __restrict__ commit_offsets,
    void* __restrict__ new_seq_lens,
    void* __restrict__ released_slots,
    const void* __restrict__ accepted_rows,
    const void* __restrict__ commit_lens,
    const void* __restrict__ scratch_slots,
    void* __restrict__ req_to_token,
    const void* __restrict__ req_pool_indices,
    const void* __restrict__ prefix_lens,
    int32_t request_count,
    int32_t accepted_width,
    int32_t physical_rows,
    int32_t committed_count,
    int64_t req_to_token_stride,
    bool accepted_rows_are_int64,
    bool commit_lens_are_int64,
    bool scratch_slots_are_int64,
    bool req_to_token_is_int64,
    bool req_pool_indices_are_int64,
    bool prefix_lens_are_int64,
    bool new_seq_lens_are_int64) {
  extern __shared__ uint32_t kept_rows[];
  __shared__ bool commit_metadata_is_valid;
  const int32_t kept_word_count = (physical_rows + 31) / 32;
  for (int32_t word = threadIdx.x; word < kept_word_count;
       word += blockDim.x) {
    kept_rows[word] = 0;
  }
  if (threadIdx.x == 0) {
    int32_t observed_count = 0;
    commit_metadata_is_valid = true;
    commit_offsets[0] = 0;
    for (int32_t request = 0; request < request_count; ++request) {
      const int64_t length = request_forest_load_index(
          commit_lens, commit_lens_are_int64, request);
      if (length < 0 || length > accepted_width) {
        commit_metadata_is_valid = false;
      } else {
        observed_count += static_cast<int32_t>(length);
      }
      commit_offsets[request + 1] = observed_count;
    }
    commit_metadata_is_valid =
        commit_metadata_is_valid && observed_count == committed_count;
  }
  __syncthreads();
  if (!commit_metadata_is_valid) {
    // committed_count sizes every compact output.  Continuing after an
    // invariant violation could publish or free uninitialized slot ids.
    // Trap the stream instead of returning such outputs to the allocator.
    if (threadIdx.x == 0) {
      asm volatile("trap;");
    }
    return;
  }

  for (int32_t request = threadIdx.x; request < request_count;
       request += blockDim.x) {
    const int32_t output_offset = commit_offsets[request];
    const int32_t commit_len =
        commit_offsets[request + 1] - output_offset;

    const int64_t prefix_len = request_forest_load_index(
        prefix_lens, prefix_lens_are_int64, request);
    request_forest_store_index(
        new_seq_lens,
        new_seq_lens_are_int64,
        request,
        prefix_len + commit_len);
    const int64_t request_pool_row = request_forest_load_index(
        req_pool_indices, req_pool_indices_are_int64, request);

    const int64_t accepted_start =
        static_cast<int64_t>(request) * accepted_width;
    for (int32_t column = 0; column < commit_len; ++column) {
      const int64_t physical_row = request_forest_load_index(
          accepted_rows,
          accepted_rows_are_int64,
          accepted_start + column);
      const int64_t scratch_slot = request_forest_load_index(
          scratch_slots, scratch_slots_are_int64, physical_row);
      const int64_t committed_index = output_offset + column;
      committed_rows[committed_index] = physical_row;
      request_forest_store_index(
          committed_slots,
          scratch_slots_are_int64,
          committed_index,
          scratch_slot);
      request_forest_store_index(
          req_to_token,
          req_to_token_is_int64,
          request_pool_row * req_to_token_stride + prefix_len + column,
          scratch_slot);
      atomicOr(
          kept_rows + (physical_row >> 5),
          uint32_t{1} << (physical_row & 31));
    }
  }
  __syncthreads();

  if (physical_rows <= 32) {
    if (threadIdx.x != 0) {
      return;
    }
    int32_t released_count = 0;
    for (int32_t physical_row = 0; physical_row < physical_rows;
         ++physical_row) {
      const bool kept =
          (kept_rows[physical_row >> 5] &
           (uint32_t{1} << (physical_row & 31))) != 0;
      if (!kept) {
        request_forest_store_index(
            released_slots,
            scratch_slots_are_int64,
            released_count++,
            request_forest_load_index(
                scratch_slots, scratch_slots_are_int64, physical_row));
      }
    }
    return;
  }

  // Each warp emits one ordered group of 32 rows. The eight warp counts
  // provide a block prefix without a second kernel or device/host round trip.
  __shared__ int32_t warp_released_counts[8];
  const int32_t lane = threadIdx.x & 31;
  const int32_t warp = threadIdx.x >> 5;
  int32_t released_base = 0;
  for (int32_t tile = 0; tile < physical_rows; tile += 256) {
    const int32_t physical_row = tile + threadIdx.x;
    const bool released = physical_row < physical_rows &&
        (kept_rows[physical_row >> 5] &
         (uint32_t{1} << (physical_row & 31))) == 0;
    const uint32_t mask = __ballot_sync(0xffffffffu, released);
    if (lane == 0) {
      warp_released_counts[warp] = __popc(mask);
    }
    __syncthreads();
    int32_t warp_prefix = 0;
    int32_t tile_count = 0;
#pragma unroll
    for (int32_t preceding = 0; preceding < 8; ++preceding) {
      const int32_t count = warp_released_counts[preceding];
      if (preceding < warp) {
        warp_prefix += count;
      }
      tile_count += count;
    }
    if (released) {
      const uint32_t lower_lanes = (uint32_t{1} << lane) - 1;
      const int32_t index = released_base + warp_prefix +
          __popc(mask & lower_lanes);
      request_forest_store_index(
          released_slots, scratch_slots_are_int64, index,
          request_forest_load_index(
              scratch_slots, scratch_slots_are_int64, physical_row));
    }
    released_base += tile_count;
    // All warps must consume the counts before the next tile overwrites them.
    __syncthreads();
  }
}

void check_cuda_vector(
    const char* name,
    const at::Tensor& value,
    at::ScalarType dtype,
    const at::Device& device) {
  TORCH_CHECK(value.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(value.device() == device, name, " must share one CUDA device");
  TORCH_CHECK(value.dim() == 1, name, " must be one-dimensional");
  TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(value.scalar_type() == dtype, name, " has the wrong dtype");
}

void check_cuda_index_tensor(
    const char* name,
    const at::Tensor& value,
    const at::Device& device,
    int64_t dimensions) {
  TORCH_CHECK(value.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(value.device() == device, name, " must share one CUDA device");
  TORCH_CHECK(value.dim() == dimensions, name, " has the wrong rank");
  TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(
      value.scalar_type() == at::kInt || value.scalar_type() == at::kLong,
      name,
      " must be int32 or int64");
}

}  // namespace

void request_forest_build_tree_attention(
    const at::Tensor& ancestor_row_ptr,
    const at::Tensor& ancestor_columns,
    const at::Tensor& out_cache_loc,
    at::Tensor page_table,
    at::Tensor cache_seqlens,
    at::Tensor cu_seqlens_q) {
  const at::Device device = ancestor_row_ptr.device();
  check_cuda_vector(
      "ancestor_row_ptr", ancestor_row_ptr, at::kInt, device);
  check_cuda_vector(
      "ancestor_columns", ancestor_columns, at::kLong, device);
  TORCH_CHECK(out_cache_loc.is_cuda(), "out_cache_loc must be a CUDA tensor");
  TORCH_CHECK(
      out_cache_loc.device() == device,
      "out_cache_loc must share one CUDA device");
  TORCH_CHECK(
      out_cache_loc.dim() == 1 && out_cache_loc.is_contiguous(),
      "out_cache_loc must be a contiguous vector");
  TORCH_CHECK(
      out_cache_loc.scalar_type() == at::kInt ||
          out_cache_loc.scalar_type() == at::kLong,
      "out_cache_loc must be int32 or int64");

  TORCH_CHECK(page_table.is_cuda(), "page_table must be a CUDA tensor");
  TORCH_CHECK(
      page_table.device() == device,
      "page_table must share one CUDA device");
  TORCH_CHECK(
      page_table.dim() == 2 && page_table.is_contiguous(),
      "page_table must be a contiguous matrix");
  TORCH_CHECK(
      page_table.scalar_type() == at::kInt,
      "page_table must be int32");

  const int64_t logical_rows = page_table.size(0);
  const int64_t max_tree_depth = page_table.size(1);
  TORCH_CHECK(logical_rows > 0, "page_table needs at least one row");
  TORCH_CHECK(max_tree_depth > 0, "page_table needs positive width");
  TORCH_CHECK(
      logical_rows <= std::numeric_limits<int32_t>::max(),
      "logical row count exceeds int32");
  TORCH_CHECK(
      max_tree_depth <= std::numeric_limits<int32_t>::max(),
      "tree depth exceeds int32");
  TORCH_CHECK(
      ancestor_row_ptr.numel() == logical_rows + 1,
      "ancestor_row_ptr must have logical_rows + 1 entries");
  TORCH_CHECK(
      out_cache_loc.numel() >= logical_rows,
      "out_cache_loc needs one entry per logical row");

  check_cuda_vector(
      "cache_seqlens", cache_seqlens, at::kInt, device);
  check_cuda_vector("cu_seqlens_q", cu_seqlens_q, at::kInt, device);
  TORCH_CHECK(
      cache_seqlens.numel() == logical_rows,
      "cache_seqlens must have logical_rows entries");
  TORCH_CHECK(
      cu_seqlens_q.numel() == logical_rows + 1,
      "cu_seqlens_q must have logical_rows + 1 entries");

  const c10::cuda::OptionalCUDAGuard device_guard(device);
#if REQUEST_FOREST_PACKED_TREE_METADATA
  if (logical_rows >= 512 && max_tree_depth <= 32) {
    int32_t row_lane_shift = 2;
    while ((int32_t{1} << row_lane_shift) < max_tree_depth) {
      ++row_lane_shift;
    }
    constexpr int32_t packed_threads = 128;
    const int32_t rows_per_block = packed_threads >> row_lane_shift;
    const dim3 packed_grid(static_cast<uint32_t>(
        (logical_rows + rows_per_block - 1) / rows_per_block));
    const cudaStream_t packed_stream = at::cuda::getCurrentCUDAStream();
    if (out_cache_loc.scalar_type() == at::kInt) {
      build_tree_attention_metadata_packed_kernel<int32_t>
          <<<packed_grid, packed_threads, 0, packed_stream>>>(
              ancestor_row_ptr.data_ptr<int32_t>(),
              ancestor_columns.data_ptr<int64_t>(),
              out_cache_loc.data_ptr<int32_t>(),
              page_table.data_ptr<int32_t>(),
              cache_seqlens.data_ptr<int32_t>(),
              cu_seqlens_q.data_ptr<int32_t>(),
              static_cast<int32_t>(logical_rows),
              static_cast<int32_t>(max_tree_depth), row_lane_shift);
    } else {
      build_tree_attention_metadata_packed_kernel<int64_t>
          <<<packed_grid, packed_threads, 0, packed_stream>>>(
              ancestor_row_ptr.data_ptr<int32_t>(),
              ancestor_columns.data_ptr<int64_t>(),
              out_cache_loc.data_ptr<int64_t>(),
              page_table.data_ptr<int32_t>(),
              cache_seqlens.data_ptr<int32_t>(),
              cu_seqlens_q.data_ptr<int32_t>(),
              static_cast<int32_t>(logical_rows),
              static_cast<int32_t>(max_tree_depth), row_lane_shift);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
#endif
  int32_t threads = 32;
  while (threads < max_tree_depth && threads < 256) {
    threads *= 2;
  }
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(static_cast<uint32_t>(logical_rows));
  const dim3 block(static_cast<uint32_t>(threads));

  if (out_cache_loc.scalar_type() == at::kInt) {
    build_tree_attention_metadata_kernel<int32_t><<<grid, block, 0, stream>>>(
        ancestor_row_ptr.data_ptr<int32_t>(),
        ancestor_columns.data_ptr<int64_t>(),
        out_cache_loc.data_ptr<int32_t>(),
        page_table.data_ptr<int32_t>(),
        cache_seqlens.data_ptr<int32_t>(),
        cu_seqlens_q.data_ptr<int32_t>(),
        static_cast<int32_t>(logical_rows),
        static_cast<int32_t>(max_tree_depth));
  } else {
    build_tree_attention_metadata_kernel<int64_t><<<grid, block, 0, stream>>>(
        ancestor_row_ptr.data_ptr<int32_t>(),
        ancestor_columns.data_ptr<int64_t>(),
        out_cache_loc.data_ptr<int64_t>(),
        page_table.data_ptr<int32_t>(),
        cache_seqlens.data_ptr<int32_t>(),
        cu_seqlens_q.data_ptr<int32_t>(),
        static_cast<int32_t>(logical_rows),
        static_cast<int32_t>(max_tree_depth));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void request_forest_verify_greedy(
    at::Tensor predicts,
    at::Tensor output_token_ids,
    at::Tensor accept_index,
    at::Tensor accepted_draft_lens,
    at::Tensor accept_lens,
    at::Tensor first_miss_positions,
    at::Tensor termination_reasons,
    const at::Tensor& draft_token,
    const at::Tensor& first_child,
    const at::Tensor& next_sibling,
    const at::Tensor& positions,
    const at::Tensor& has_unverified_children,
    const at::Tensor& request_offsets,
    const at::Tensor& anchor_rows,
    const at::Tensor& target_predict,
    const at::Tensor& request_ignore_eos,
    const at::Tensor& eos_token_ids) {
  const at::Device device = draft_token.device();
  check_cuda_vector("draft_token", draft_token, at::kLong, device);
  check_cuda_vector("first_child", first_child, at::kInt, device);
  check_cuda_vector("next_sibling", next_sibling, at::kInt, device);
  check_cuda_vector("positions", positions, at::kLong, device);
  check_cuda_vector(
      "has_unverified_children", has_unverified_children, at::kBool, device);
  check_cuda_vector("request_offsets", request_offsets, at::kInt, device);
  check_cuda_vector("anchor_rows", anchor_rows, at::kInt, device);
  check_cuda_vector("target_predict", target_predict, at::kLong, device);
  check_cuda_vector(
      "request_ignore_eos", request_ignore_eos, at::kBool, device);
  check_cuda_vector("eos_token_ids", eos_token_ids, at::kLong, device);

  const int64_t logical_rows = draft_token.numel();
  const int64_t request_count = anchor_rows.numel();
  TORCH_CHECK(logical_rows > 0, "draft_token needs at least one row");
  TORCH_CHECK(request_count > 0, "anchor_rows needs at least one request");
  TORCH_CHECK(
      logical_rows <= std::numeric_limits<int32_t>::max(),
      "logical row count exceeds int32");
  TORCH_CHECK(
      request_count <= std::numeric_limits<int32_t>::max(),
      "request count exceeds int32");
  TORCH_CHECK(
      first_child.numel() == logical_rows,
      "first_child must contain logical_rows entries");
  TORCH_CHECK(
      next_sibling.numel() == logical_rows,
      "next_sibling must contain logical_rows entries");
  TORCH_CHECK(
      positions.numel() == logical_rows,
      "positions must contain logical_rows entries");
  TORCH_CHECK(
      has_unverified_children.numel() == logical_rows,
      "has_unverified_children must contain logical_rows entries");
  TORCH_CHECK(
      target_predict.numel() == logical_rows,
      "target_predict must contain logical_rows entries");
  TORCH_CHECK(
      request_offsets.numel() == request_count + 1,
      "request_offsets must have request_count + 1 entries");
  TORCH_CHECK(
      request_ignore_eos.numel() == request_count,
      "request_ignore_eos must have request_count entries");

  check_cuda_vector("predicts", predicts, at::kInt, device);
  check_cuda_vector(
      "accepted_draft_lens", accepted_draft_lens, at::kInt, device);
  check_cuda_vector("accept_lens", accept_lens, at::kInt, device);
  check_cuda_vector(
      "first_miss_positions", first_miss_positions, at::kLong, device);
  check_cuda_vector(
      "termination_reasons", termination_reasons, at::kChar, device);
  TORCH_CHECK(
      predicts.numel() == logical_rows,
      "predicts must contain logical_rows entries");
  TORCH_CHECK(
      accepted_draft_lens.numel() == request_count,
      "accepted_draft_lens must contain request_count entries");
  TORCH_CHECK(
      accept_lens.numel() == request_count,
      "accept_lens must contain request_count entries");
  TORCH_CHECK(
      first_miss_positions.numel() == request_count,
      "first_miss_positions must contain request_count entries");
  TORCH_CHECK(
      termination_reasons.numel() == request_count,
      "termination_reasons must contain request_count entries");

  TORCH_CHECK(
      output_token_ids.is_cuda() && output_token_ids.device() == device &&
          output_token_ids.dim() == 2 && output_token_ids.is_contiguous() &&
          output_token_ids.scalar_type() == at::kInt,
      "output_token_ids must be a contiguous CUDA int32 matrix");
  TORCH_CHECK(
      accept_index.is_cuda() && accept_index.device() == device &&
          accept_index.dim() == 2 && accept_index.is_contiguous() &&
          accept_index.scalar_type() == at::kInt,
      "accept_index must be a contiguous CUDA int32 matrix");
  TORCH_CHECK(
      output_token_ids.sizes() == accept_index.sizes(),
      "output_token_ids and accept_index must have the same shape");
  TORCH_CHECK(
      output_token_ids.size(0) == request_count,
      "output matrices must have request_count rows");
  const int64_t max_tree_depth = output_token_ids.size(1);
  TORCH_CHECK(max_tree_depth > 0, "output matrices need positive width");
  TORCH_CHECK(
      max_tree_depth <= std::numeric_limits<int32_t>::max(),
      "tree depth exceeds int32");
  TORCH_CHECK(
      eos_token_ids.numel() <= std::numeric_limits<int32_t>::max(),
      "EOS token count exceeds int32");

  const c10::cuda::OptionalCUDAGuard device_guard(device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t average_request_rows =
      (logical_rows + request_count - 1) / request_count;
  int32_t threads = 32;
  while (threads < average_request_rows && threads < 128) {
    threads *= 2;
  }
  request_forest_verify_greedy_kernel<<<
      dim3(static_cast<uint32_t>(request_count)),
      dim3(static_cast<uint32_t>(threads)),
      0,
      stream>>>(
      predicts.data_ptr<int32_t>(),
      output_token_ids.data_ptr<int32_t>(),
      accept_index.data_ptr<int32_t>(),
      accepted_draft_lens.data_ptr<int32_t>(),
      accept_lens.data_ptr<int32_t>(),
      first_miss_positions.data_ptr<int64_t>(),
      termination_reasons.data_ptr<int8_t>(),
      draft_token.data_ptr<int64_t>(),
      first_child.data_ptr<int32_t>(),
      next_sibling.data_ptr<int32_t>(),
      positions.data_ptr<int64_t>(),
      has_unverified_children.data_ptr<bool>(),
      request_offsets.data_ptr<int32_t>(),
      anchor_rows.data_ptr<int32_t>(),
      target_predict.data_ptr<int64_t>(),
      request_ignore_eos.data_ptr<bool>(),
      eos_token_ids.data_ptr<int64_t>(),
      static_cast<int32_t>(request_count),
      static_cast<int32_t>(max_tree_depth),
      static_cast<int32_t>(eos_token_ids.numel()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
request_forest_commit_kv(
    const at::Tensor& accepted_rows,
    const at::Tensor& commit_lens,
    const at::Tensor& scratch_slots,
    at::Tensor req_to_token,
    const at::Tensor& req_pool_indices,
    const at::Tensor& prefix_lens,
    int64_t committed_count) {
  const at::Device device = scratch_slots.device();
  check_cuda_index_tensor("accepted_rows", accepted_rows, device, 2);
  check_cuda_index_tensor("commit_lens", commit_lens, device, 1);
  check_cuda_index_tensor("scratch_slots", scratch_slots, device, 1);
  check_cuda_index_tensor("req_pool_indices", req_pool_indices, device, 1);
  check_cuda_index_tensor("prefix_lens", prefix_lens, device, 1);
  check_cuda_index_tensor("req_to_token", req_to_token, device, 2);

  const int64_t request_count = accepted_rows.size(0);
  const int64_t accepted_width = accepted_rows.size(1);
  const int64_t physical_rows = scratch_slots.numel();
  TORCH_CHECK(request_count > 0, "accepted_rows needs at least one request");
  TORCH_CHECK(accepted_width > 0, "accepted_rows needs positive width");
  TORCH_CHECK(physical_rows > 0, "scratch_slots needs at least one row");
  TORCH_CHECK(
      request_count <= std::numeric_limits<int32_t>::max(),
      "request count exceeds int32");
  TORCH_CHECK(
      accepted_width <= std::numeric_limits<int32_t>::max(),
      "accepted width exceeds int32");
  TORCH_CHECK(
      physical_rows <= std::numeric_limits<int32_t>::max(),
      "physical row count exceeds int32");
  TORCH_CHECK(
      commit_lens.numel() == request_count,
      "commit_lens must contain one entry per request");
  TORCH_CHECK(
      req_pool_indices.numel() == request_count,
      "req_pool_indices must contain one entry per request");
  TORCH_CHECK(
      prefix_lens.numel() == request_count,
      "prefix_lens must contain one entry per request");

  // Python stages commit_lens to a reusable pinned buffer on a dedicated D2H
  // stream while Draft KV is prepared.  Keeping the resulting scalar explicit
  // makes this wrapper launch-only: it never synchronizes the current stream.
  TORCH_CHECK(committed_count >= 0, "committed_count must be non-negative");
  TORCH_CHECK(
      committed_count <= physical_rows,
      "committed row count exceeds scratch slot count");
  TORCH_CHECK(
      committed_count <= request_count * accepted_width,
      "committed row count exceeds accepted_rows capacity");

  const c10::cuda::OptionalCUDAGuard device_guard(device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto committed_rows = at::empty(
      {committed_count}, accepted_rows.options().dtype(at::kLong));
  auto committed_slots =
      at::empty({committed_count}, scratch_slots.options());
  auto commit_offsets = at::empty(
      {request_count + 1}, accepted_rows.options().dtype(at::kInt));
  const at::ScalarType new_seq_lens_dtype =
      prefix_lens.scalar_type() == at::kLong ||
          commit_lens.scalar_type() == at::kLong
      ? at::kLong
      : at::kInt;
  auto new_seq_lens = at::empty(
      {request_count}, prefix_lens.options().dtype(new_seq_lens_dtype));
  auto released_slots = at::empty(
      {physical_rows - committed_count}, scratch_slots.options());

  const int64_t kept_word_count = (physical_rows + 31) / 32;
  const int64_t shared_bytes = kept_word_count * sizeof(uint32_t);
  TORCH_CHECK(
      shared_bytes <= 48 * 1024,
      "Request Forest fused commit supports at most 393216 physical rows");
  request_forest_commit_kv_kernel<<<1, 256, shared_bytes, stream>>>(
      committed_rows.data_ptr<int64_t>(),
      committed_slots.data_ptr(),
      commit_offsets.data_ptr<int32_t>(),
      new_seq_lens.data_ptr(),
      released_slots.data_ptr(),
      accepted_rows.data_ptr(),
      commit_lens.data_ptr(),
      scratch_slots.data_ptr(),
      req_to_token.data_ptr(),
      req_pool_indices.data_ptr(),
      prefix_lens.data_ptr(),
      static_cast<int32_t>(request_count),
      static_cast<int32_t>(accepted_width),
      static_cast<int32_t>(physical_rows),
      static_cast<int32_t>(committed_count),
      req_to_token.stride(0),
      accepted_rows.scalar_type() == at::kLong,
      commit_lens.scalar_type() == at::kLong,
      scratch_slots.scalar_type() == at::kLong,
      req_to_token.scalar_type() == at::kLong,
      req_pool_indices.scalar_type() == at::kLong,
      prefix_lens.scalar_type() == at::kLong,
      new_seq_lens_dtype == at::kLong);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {
      committed_rows,
      committed_slots,
      commit_offsets,
      new_seq_lens,
      released_slots};
}
