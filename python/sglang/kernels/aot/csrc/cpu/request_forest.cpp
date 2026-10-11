#include "common.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace {

void check_cpu_contiguous_1d(const at::Tensor& tensor, at::ScalarType dtype, const char* name) {
  TORCH_CHECK(tensor.device().type() == at::kCPU, name, " must be a CPU tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == 1, name, " must be one-dimensional");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an invalid dtype");
}

void check_same_numel(const at::Tensor& tensor, int64_t expected, const char* name) {
  TORCH_CHECK(tensor.numel() == expected, name, " must contain ", expected, " elements, got ", tensor.numel());
}

enum class RequestForestTerminationReason : int64_t {
  TARGET_MISS = 0,
  LEAF_EXHAUSTED = 1,
  BUDGET_BOUNDARY = 2,
  NO_CANDIDATES = 3,
  EOS = 4,
};

struct TraversalSummary {
  int64_t accepted_length = 0;
  int64_t output_length = 0;
  int64_t first_miss_position = -1;
  int64_t correction_token_id = -1;
  int64_t reason = static_cast<int64_t>(RequestForestTerminationReason::TARGET_MISS);
  int64_t visited_row = -1;
};

int64_t find_matching_child(
    const int64_t* token_ids,
    const int64_t* first_child,
    const int64_t* next_sibling,
    int64_t parent_row,
    int64_t token_id,
    int64_t segment_start,
    int64_t segment_end) {
  int64_t child_row = first_child[parent_row];
  int64_t sibling_steps = 0;
  const int64_t max_sibling_steps = segment_end - segment_start;
  while (child_row != -1) {
    TORCH_CHECK(
        child_row >= segment_start && child_row < segment_end,
        "request_forest_commit_cpu: child row ",
        child_row,
        " is outside request segment [",
        segment_start,
        ", ",
        segment_end,
        ")");
    if (token_ids[child_row] == token_id) {
      return child_row;
    }
    child_row = next_sibling[child_row];
    TORCH_CHECK(
        ++sibling_steps <= max_sibling_steps,
        "request_forest_commit_cpu: cycle in next_sibling chain");
  }
  return -1;
}

TraversalSummary traverse_request(
    const int64_t* token_ids,
    const int64_t* first_child,
    const int64_t* next_sibling,
    const int64_t* positions,
    const bool* has_unverified_children,
    const int64_t* packed_to_node,
    const int64_t* target_tokens,
    const bool* target_eos,
    int64_t anchor_row,
    int64_t segment_end,
    int64_t* accepted_output,
    int64_t* token_output) {
  TraversalSummary summary;
  int64_t current_row = anchor_row;
  int64_t traversal_steps = 0;
  const int64_t max_traversal_steps = segment_end - anchor_row;

  while (true) {
    TORCH_CHECK(
        current_row >= anchor_row && current_row < segment_end,
        "request_forest_commit_cpu: traversal row is outside its request segment");
    TORCH_CHECK(
        ++traversal_steps <= max_traversal_steps,
        "request_forest_commit_cpu: cycle in first_child traversal");

    summary.visited_row = current_row;
    const int64_t decision_token = target_tokens[current_row];
    TORCH_CHECK(
        decision_token >= 0,
        "request_forest_commit_cpu: target token must be non-negative at traversed row ",
        current_row);

    const int64_t matching_child = find_matching_child(
        token_ids,
        first_child,
        next_sibling,
        current_row,
        decision_token,
        anchor_row,
        segment_end);

    if (matching_child != -1) {
      const int64_t node_id = packed_to_node[matching_child];
      TORCH_CHECK(
          node_id >= 0,
          "request_forest_commit_cpu: a matching candidate row has no logical node id");
      if (accepted_output != nullptr) {
        accepted_output[summary.accepted_length] = node_id;
      }
      if (token_output != nullptr) {
        token_output[summary.output_length] = decision_token;
      }
      ++summary.accepted_length;
      ++summary.output_length;

      if (target_eos[current_row]) {
        summary.reason = static_cast<int64_t>(RequestForestTerminationReason::EOS);
        return summary;
      }
      current_row = matching_child;
      continue;
    }

    if (token_output != nullptr) {
      token_output[summary.output_length] = decision_token;
    }
    ++summary.output_length;
    summary.first_miss_position = positions[current_row] + 1;
    summary.correction_token_id = decision_token;

    if (target_eos[current_row]) {
      summary.reason = static_cast<int64_t>(RequestForestTerminationReason::EOS);
    } else if (first_child[current_row] == -1) {
      if (has_unverified_children[current_row]) {
        summary.reason = static_cast<int64_t>(RequestForestTerminationReason::BUDGET_BOUNDARY);
      } else if (current_row == anchor_row) {
        summary.reason = static_cast<int64_t>(RequestForestTerminationReason::NO_CANDIDATES);
      } else {
        summary.reason = static_cast<int64_t>(RequestForestTerminationReason::LEAF_EXHAUSTED);
      }
    } else {
      summary.reason = static_cast<int64_t>(RequestForestTerminationReason::TARGET_MISS);
    }
    return summary;
  }
}

}  // namespace

namespace {

void check_compile_input_schema(
    const at::Tensor& node_parent_ids,
    const at::Tensor& node_request_slots,
    const at::Tensor& node_token_ids,
    const at::Tensor& node_depths,
    const at::Tensor& anchor_token_ids,
    const at::Tensor& anchor_positions,
    int64_t physical_rows,
    int64_t padding_token_id,
    const char* operation_name) {
  check_cpu_contiguous_1d(node_parent_ids, at::kLong, "node_parent_ids");
  check_cpu_contiguous_1d(node_request_slots, at::kLong, "node_request_slots");
  check_cpu_contiguous_1d(node_token_ids, at::kLong, "node_token_ids");
  check_cpu_contiguous_1d(node_depths, at::kLong, "node_depths");
  check_cpu_contiguous_1d(anchor_token_ids, at::kLong, "anchor_token_ids");
  check_cpu_contiguous_1d(anchor_positions, at::kLong, "anchor_positions");

  const int64_t node_count = node_parent_ids.numel();
  const int64_t request_count = anchor_token_ids.numel();
  check_same_numel(node_request_slots, node_count, "node_request_slots");
  check_same_numel(node_token_ids, node_count, "node_token_ids");
  check_same_numel(node_depths, node_count, "node_depths");
  check_same_numel(anchor_positions, request_count, "anchor_positions");
  TORCH_CHECK(request_count > 0, operation_name, ": at least one request is required");
  TORCH_CHECK(physical_rows > 0, operation_name, ": physical_rows must be positive");
  TORCH_CHECK(
      padding_token_id >= 0,
      operation_name,
      ": padding_token_id must be non-negative");
}

void validate_compile_metadata(
    const at::Tensor& node_parent_ids,
    const at::Tensor& node_request_slots,
    const at::Tensor& node_token_ids,
    const at::Tensor& node_depths,
    const at::Tensor& anchor_token_ids,
    const at::Tensor& anchor_positions,
    const double* value_ptr,
    const char* operation_name) {
  const int64_t node_count = node_parent_ids.numel();
  const int64_t request_count = anchor_token_ids.numel();
  const int64_t* parent_ptr = node_parent_ids.data_ptr<int64_t>();
  const int64_t* request_ptr = node_request_slots.data_ptr<int64_t>();
  const int64_t* node_token_ptr = node_token_ids.data_ptr<int64_t>();
  const int64_t* node_depth_ptr = node_depths.data_ptr<int64_t>();
  const int64_t* anchor_token_ptr = anchor_token_ids.data_ptr<int64_t>();
  const int64_t* anchor_position_ptr = anchor_positions.data_ptr<int64_t>();

  for (int64_t request_slot = 0; request_slot < request_count; ++request_slot) {
    TORCH_CHECK(
        anchor_token_ptr[request_slot] >= 0,
        operation_name,
        ": anchor token ids must be non-negative");
    TORCH_CHECK(
        anchor_position_ptr[request_slot] >= 0,
        operation_name,
        ": anchor positions must be non-negative");
  }

  for (int64_t node_id = 0; node_id < node_count; ++node_id) {
    const int64_t parent_id = parent_ptr[node_id];
    const int64_t request_slot = request_ptr[node_id];
    TORCH_CHECK(
        request_slot >= 0 && request_slot < request_count,
        operation_name,
        ": invalid request slot at node ",
        node_id);
    TORCH_CHECK(
        node_token_ptr[node_id] >= 0,
        operation_name,
        ": token id must be non-negative at node ",
        node_id);
    TORCH_CHECK(
        parent_id == -1 || (parent_id >= 0 && parent_id < node_id),
        operation_name,
        ": parent of node ",
        node_id,
        " must be -1 or an earlier node");
    if (parent_id == -1) {
      TORCH_CHECK(
          node_depth_ptr[node_id] == 1,
          operation_name,
          ": root candidate depth must be one at node ",
          node_id);
    } else {
      TORCH_CHECK(
          request_ptr[parent_id] == request_slot,
          operation_name,
          ": cross-request parent at node ",
          node_id);
      TORCH_CHECK(
          node_depth_ptr[node_id] == node_depth_ptr[parent_id] + 1,
          operation_name,
          ": invalid depth at node ",
          node_id);
    }

    if (value_ptr != nullptr) {
      TORCH_CHECK(
          std::isfinite(value_ptr[node_id]),
          operation_name,
          ": value for node ",
          node_id,
          " must be finite");
      if (parent_id != -1) {
        TORCH_CHECK(
            value_ptr[node_id] <= value_ptr[parent_id],
            operation_name,
            ": child value exceeds parent value at node ",
            node_id);
      }
    }
  }
}

std::vector<int64_t> rank_positive_nodes(
    const double* value_ptr, int64_t node_count, int64_t capacity) {
  const int64_t limit = std::min<int64_t>(capacity, node_count);
  if (limit == 0) {
    return {};
  }
  std::vector<int64_t> ranked_ids(static_cast<size_t>(node_count));
  std::iota(ranked_ids.begin(), ranked_ids.end(), int64_t{0});
  const auto higher_value_first = [value_ptr](int64_t left, int64_t right) {
    if (value_ptr[left] != value_ptr[right]) {
      return value_ptr[left] > value_ptr[right];
    }
    return left < right;
  };
  if (limit < node_count) {
    // Partition in linear time, then sort only the W winners. This preserves
    // the exact value-descending/node-ID-ascending order of partial_sort.
    std::nth_element(
        ranked_ids.begin(),
        ranked_ids.begin() + limit,
        ranked_ids.end(),
        higher_value_first);
  }
  std::sort(
      ranked_ids.begin(), ranked_ids.begin() + limit, higher_value_first);

  int64_t selected_count = 0;
  while (selected_count < limit && value_ptr[ranked_ids[selected_count]] > 0.0) {
    ++selected_count;
  }
  ranked_ids.resize(static_cast<size_t>(selected_count));
  return ranked_ids;
}

at::Tensor selected_ids_to_tensor(
    const std::vector<int64_t>& selected_ids, const at::TensorOptions& options) {
  at::Tensor output = at::empty({static_cast<int64_t>(selected_ids.size())}, options);
  if (!selected_ids.empty()) {
    std::copy(selected_ids.begin(), selected_ids.end(), output.data_ptr<int64_t>());
  }
  return output;
}

std::vector<at::Tensor> emit_verify_plan_from_selection(
    const at::Tensor& node_parent_ids,
    const at::Tensor& node_request_slots,
    const at::Tensor& node_token_ids,
    const at::Tensor& node_depths,
    const at::Tensor& anchor_token_ids,
    const at::Tensor& anchor_positions,
    const std::vector<uint8_t>& selected,
    int64_t physical_rows,
    int64_t padding_token_id,
    const char* operation_name) {
  const int64_t node_count = node_parent_ids.numel();
  const int64_t request_count = anchor_token_ids.numel();
  TORCH_CHECK(
      selected.size() == static_cast<size_t>(node_count),
      operation_name,
      ": internal selected-mask size mismatch");

  const int64_t* parent_ptr = node_parent_ids.data_ptr<int64_t>();
  const int64_t* request_ptr = node_request_slots.data_ptr<int64_t>();
  const int64_t* node_token_ptr = node_token_ids.data_ptr<int64_t>();
  const int64_t* node_depth_ptr = node_depths.data_ptr<int64_t>();
  const int64_t* anchor_token_ptr = anchor_token_ids.data_ptr<int64_t>();
  const int64_t* anchor_position_ptr = anchor_positions.data_ptr<int64_t>();

  std::vector<int64_t> selected_counts(static_cast<size_t>(request_count), int64_t{0});
  std::vector<uint8_t> has_unselected_root(static_cast<size_t>(request_count), uint8_t{0});
  std::vector<uint8_t> has_unselected_child(static_cast<size_t>(node_count), uint8_t{0});
  int64_t selected_count = 0;
  for (int64_t node_id = 0; node_id < node_count; ++node_id) {
    if (selected[node_id]) {
      ++selected_counts[request_ptr[node_id]];
      ++selected_count;
      continue;
    }
    const int64_t parent_id = parent_ptr[node_id];
    if (parent_id == -1) {
      has_unselected_root[request_ptr[node_id]] = 1;
    } else {
      has_unselected_child[parent_id] = 1;
    }
  }

  const int64_t logical_rows = request_count + selected_count;
  TORCH_CHECK(
      logical_rows <= physical_rows,
      operation_name,
      ": logical rows ",
      logical_rows,
      " exceed physical rows ",
      physical_rows);

  const at::TensorOptions index_options = node_parent_ids.options();
  const at::TensorOptions bool_options = index_options.dtype(at::kBool);

  at::Tensor token_ids = at::full({physical_rows}, padding_token_id, index_options);
  at::Tensor parent_rows = at::full({physical_rows}, -1, index_options);
  at::Tensor first_child = at::full({physical_rows}, -1, index_options);
  at::Tensor next_sibling = at::full({physical_rows}, -1, index_options);
  at::Tensor depths = at::full({physical_rows}, -1, index_options);
  at::Tensor positions = at::zeros({physical_rows}, index_options);
  at::Tensor request_slots = at::full({physical_rows}, -1, index_options);
  at::Tensor valid_mask = at::zeros({physical_rows}, bool_options);
  at::Tensor candidate_mask = at::zeros({physical_rows}, bool_options);
  at::Tensor has_unverified_children = at::zeros({physical_rows}, bool_options);
  at::Tensor packed_to_node = at::full({physical_rows}, -1, index_options);
  at::Tensor node_to_packed = at::full({node_count}, -1, index_options);
  at::Tensor request_offsets = at::empty({request_count + 1}, index_options);
  at::Tensor anchor_rows = at::empty({request_count}, index_options);

  int64_t* token_out = token_ids.data_ptr<int64_t>();
  int64_t* parent_out = parent_rows.data_ptr<int64_t>();
  int64_t* first_child_out = first_child.data_ptr<int64_t>();
  int64_t* next_sibling_out = next_sibling.data_ptr<int64_t>();
  int64_t* depth_out = depths.data_ptr<int64_t>();
  int64_t* position_out = positions.data_ptr<int64_t>();
  int64_t* request_slot_out = request_slots.data_ptr<int64_t>();
  bool* valid_out = valid_mask.data_ptr<bool>();
  bool* candidate_out = candidate_mask.data_ptr<bool>();
  bool* unverified_out = has_unverified_children.data_ptr<bool>();
  int64_t* packed_to_node_out = packed_to_node.data_ptr<int64_t>();
  int64_t* node_to_packed_out = node_to_packed.data_ptr<int64_t>();
  int64_t* request_offsets_out = request_offsets.data_ptr<int64_t>();
  int64_t* anchor_rows_out = anchor_rows.data_ptr<int64_t>();

  request_offsets_out[0] = 0;
  int64_t row_cursor = 0;
  std::vector<int64_t> next_request_row(static_cast<size_t>(request_count), int64_t{0});
  for (int64_t request_slot = 0; request_slot < request_count; ++request_slot) {
    const int64_t anchor_row = row_cursor;
    anchor_rows_out[request_slot] = anchor_row;
    request_offsets_out[request_slot] = anchor_row;
    row_cursor += 1 + selected_counts[request_slot];
    request_offsets_out[request_slot + 1] = row_cursor;
    next_request_row[request_slot] = anchor_row + 1;

    token_out[anchor_row] = anchor_token_ptr[request_slot];
    depth_out[anchor_row] = 0;
    position_out[anchor_row] = anchor_position_ptr[request_slot];
    request_slot_out[anchor_row] = request_slot;
    valid_out[anchor_row] = true;
    unverified_out[anchor_row] = has_unselected_root[request_slot] != 0;
  }
  TORCH_CHECK(
      row_cursor == logical_rows,
      operation_name,
      ": internal row-count mismatch");

  std::vector<int64_t> last_child(static_cast<size_t>(logical_rows), int64_t{-1});
  for (int64_t node_id = 0; node_id < node_count; ++node_id) {
    if (!selected[node_id]) {
      continue;
    }
    const int64_t request_slot = request_ptr[node_id];
    const int64_t packed_row = next_request_row[request_slot]++;
    node_to_packed_out[node_id] = packed_row;

    const int64_t parent_id = parent_ptr[node_id];
    const int64_t parent_row = parent_id == -1 ? anchor_rows_out[request_slot] : node_to_packed_out[parent_id];
    TORCH_CHECK(
        parent_row >= 0,
        operation_name,
        ": prefix closure lost parent of node ",
        node_id);

    token_out[packed_row] = node_token_ptr[node_id];
    parent_out[packed_row] = parent_row;
    depth_out[packed_row] = node_depth_ptr[node_id];
    position_out[packed_row] = anchor_position_ptr[request_slot] + node_depth_ptr[node_id];
    request_slot_out[packed_row] = request_slot;
    valid_out[packed_row] = true;
    candidate_out[packed_row] = true;
    unverified_out[packed_row] = has_unselected_child[node_id] != 0;
    packed_to_node_out[packed_row] = node_id;

    const int64_t previous_child = last_child[parent_row];
    if (previous_child == -1) {
      first_child_out[parent_row] = packed_row;
    } else {
      next_sibling_out[previous_child] = packed_row;
    }
    last_child[parent_row] = packed_row;
  }

  for (int64_t request_slot = 0; request_slot < request_count; ++request_slot) {
    TORCH_CHECK(
        next_request_row[request_slot] == request_offsets_out[request_slot + 1],
        operation_name,
        ": request packing count mismatch");
  }

  at::Tensor ancestor_row_ptr = at::empty({physical_rows + 1}, index_options);
  int64_t* ancestor_ptr_out = ancestor_row_ptr.data_ptr<int64_t>();
  ancestor_ptr_out[0] = 0;
  for (int64_t row = 0; row < logical_rows; ++row) {
    ancestor_ptr_out[row + 1] = ancestor_ptr_out[row] + depth_out[row] + 1;
  }
  const int64_t ancestor_nnz = ancestor_ptr_out[logical_rows];
  for (int64_t row = logical_rows; row < physical_rows; ++row) {
    ancestor_ptr_out[row + 1] = ancestor_nnz;
  }

  at::Tensor ancestor_columns = at::empty({ancestor_nnz}, index_options);
  int64_t* ancestor_columns_out = ancestor_columns.data_ptr<int64_t>();
  at::parallel_for(0, logical_rows, 0, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      const int64_t start = ancestor_ptr_out[row];
      int64_t write_index = ancestor_ptr_out[row + 1] - 1;
      int64_t ancestor = row;
      while (ancestor != -1 && write_index >= start) {
        ancestor_columns_out[write_index--] = ancestor;
        ancestor = parent_out[ancestor];
      }
      TORCH_CHECK(
          write_index == start - 1 && ancestor == -1,
          operation_name,
          ": depth and parent chain disagree at packed row ",
          row);
    }
  });

  at::Tensor logical_rows_scalar = at::scalar_tensor(logical_rows, index_options);
  return {
      token_ids,
      parent_rows,
      first_child,
      next_sibling,
      depths,
      positions,
      request_slots,
      valid_mask,
      candidate_mask,
      has_unverified_children,
      packed_to_node,
      node_to_packed,
      request_offsets,
      anchor_rows,
      ancestor_row_ptr,
      ancestor_columns,
      logical_rows_scalar,
  };
}

struct DFlash2HeapEntry {
  double negative_priority;
  int64_t request_slot;
  int64_t depth;
  int64_t parent_id;
  int64_t token_rank;
  int64_t batch_row;
  int64_t token_id;
  double conditional_probability;
  double rf_path_probability;
  double mix_path_probability;
  // Native lowering keeps at most one representative from a sibling group in
  // the heap when K <= 64.  This payload is deliberately excluded from the
  // comparator so the public stable priority order remains identical to the
  // Python heap implementation.
  int64_t sibling_group;
};

struct DFlash2SiblingGroup {
  int64_t batch_row;
  int64_t depth;
  int64_t predecessor_rank;
  int64_t parent_id;
  double parent_rf_path_probability;
  double parent_mix_path_probability;
  double calibration_scale;
  int64_t raw_argmax_rank;
  uint64_t remaining_mask;
};

struct DFlash2HeapGreater {
  bool operator()(const DFlash2HeapEntry& left, const DFlash2HeapEntry& right) const {
    // This is the exact tuple order used by Python's heapq implementation in
    // DFlash2LatticeLowerer.  The first five fields are unique for normal
    // request slots; the payload fields retain Python behavior even if a
    // caller deliberately supplies duplicate slots.
    return std::tie(
               left.negative_priority,
               left.request_slot,
               left.depth,
               left.parent_id,
               left.token_rank,
               left.batch_row,
               left.token_id,
               left.conditional_probability) >
        std::tie(
               right.negative_priority,
               right.request_slot,
               right.depth,
               right.parent_id,
               right.token_rank,
               right.batch_row,
               right.token_id,
               right.conditional_probability);
  }
};

template <typename scalar_t>
at::Tensor request_forest_vector_to_tensor(
    const std::vector<scalar_t>& values, const at::TensorOptions& options) {
  at::Tensor output = at::empty({static_cast<int64_t>(values.size())}, options);
  if (!values.empty()) {
    std::copy(values.begin(), values.end(), output.data_ptr<scalar_t>());
  }
  return output;
}

}  // namespace

std::vector<at::Tensor> request_forest_dflash2_lower_cpu(
    const at::Tensor& candidate_ids,
    const at::Tensor& probabilities,
    const at::Tensor& request_slots,
    int64_t capacity,
    int64_t start_node_id,
    const std::optional<at::Tensor>& calibration_multipliers,
    int64_t max_nodes_per_request,
    int64_t selection_mode,
    double d2_mix_lambda) {
  constexpr const char* operation_name = "request_forest_dflash2_lower_cpu";
  TORCH_CHECK(candidate_ids.device().type() == at::kCPU, operation_name, ": candidate_ids must be a CPU tensor");
  TORCH_CHECK(candidate_ids.is_contiguous(), operation_name, ": candidate_ids must be contiguous");
  TORCH_CHECK(candidate_ids.dim() == 3, operation_name, ": candidate_ids must have shape [B, E, K]");
  TORCH_CHECK(candidate_ids.scalar_type() == at::kLong, operation_name, ": candidate_ids must have dtype int64");
  TORCH_CHECK(probabilities.device().type() == at::kCPU, operation_name, ": probabilities must be a CPU tensor");
  TORCH_CHECK(probabilities.is_contiguous(), operation_name, ": probabilities must be contiguous");
  TORCH_CHECK(probabilities.dim() == 4, operation_name, ": probabilities must have shape [B, E, K, K]");
  TORCH_CHECK(probabilities.scalar_type() == at::kFloat, operation_name, ": probabilities must have dtype float32");
  check_cpu_contiguous_1d(request_slots, at::kLong, "request_slots");
  TORCH_CHECK(capacity >= 0, operation_name, ": capacity must be non-negative");
  TORCH_CHECK(start_node_id >= 0, operation_name, ": start_node_id must be non-negative");
  TORCH_CHECK(
      max_nodes_per_request == -1 || max_nodes_per_request > 0,
      operation_name,
      ": max_nodes_per_request must be -1 or positive");
  TORCH_CHECK(
      selection_mode >= 0 && selection_mode <= 3,
      operation_name,
      ": selection_mode must be in [0, 3]");
  TORCH_CHECK(
      std::isfinite(d2_mix_lambda) && d2_mix_lambda >= 0.0 &&
          d2_mix_lambda <= 1.0,
      operation_name,
      ": d2_mix_lambda must be finite and in [0, 1]");
  TORCH_CHECK(
      capacity <= std::numeric_limits<int64_t>::max() - start_node_id,
      operation_name,
      ": start_node_id + capacity overflows int64");

  const int64_t batch_size = candidate_ids.size(0);
  const int64_t depth_count = candidate_ids.size(1);
  const int64_t top_k = candidate_ids.size(2);
  const bool chain_mode = selection_mode == 1 || selection_mode == 3;
  const bool equal_mode = selection_mode == 2 || selection_mode == 3;
  const bool mix_enabled = d2_mix_lambda > 0.0 && !chain_mode;
  std::vector<double> depth_discounts;
  if (mix_enabled) {
    depth_discounts.resize(static_cast<size_t>(depth_count + 1), 1.0);
    for (int64_t depth = 1; depth <= depth_count; ++depth) {
      depth_discounts[static_cast<size_t>(depth)] =
          std::pow(static_cast<double>(depth), -d2_mix_lambda);
    }
  }
  TORCH_CHECK(top_k > 0, operation_name, ": K must be positive");
  TORCH_CHECK(
      probabilities.size(0) == batch_size && probabilities.size(1) == depth_count &&
          probabilities.size(2) == top_k && probabilities.size(3) == top_k,
      operation_name,
      ": probabilities must have shape [B, E, K, K] matching candidate_ids");
  check_same_numel(request_slots, batch_size, "request_slots");
  const bool calibration_enabled = calibration_multipliers.has_value();
  const double* calibration_ptr = nullptr;
  int64_t calibration_bin_count = 0;
  std::vector<uint8_t> calibration_depth_active;
  if (calibration_enabled) {
    calibration_depth_active.resize(static_cast<size_t>(depth_count), 0);
    const at::Tensor& table = *calibration_multipliers;
    TORCH_CHECK(
        table.device().type() == at::kCPU && table.is_contiguous() &&
            table.scalar_type() == at::kDouble && table.dim() == 2,
        operation_name,
        ": calibration_multipliers must be contiguous CPU float64 [depth, bin]");
    TORCH_CHECK(
        table.size(0) >= depth_count && table.size(1) > 0,
        operation_name,
        ": calibration_multipliers does not cover the lattice shape");
    calibration_ptr = table.data_ptr<double>();
    calibration_bin_count = table.size(1);
    for (int64_t depth = 0; depth < depth_count; ++depth) {
      for (int64_t bucket = 0; bucket < calibration_bin_count; ++bucket) {
        const double multiplier =
            calibration_ptr[depth * calibration_bin_count + bucket];
        TORCH_CHECK(
            std::isfinite(multiplier) && multiplier >= 0.0,
            operation_name,
            ": calibration multipliers must be finite and non-negative");
        if (multiplier != 1.0) {
          calibration_depth_active[static_cast<size_t>(depth)] = 1;
        }
      }
    }
  }

  const int64_t* candidate_ptr = candidate_ids.data_ptr<int64_t>();
  const float* probability_ptr = probabilities.data_ptr<float>();
  const int64_t* request_slot_ptr = request_slots.data_ptr<int64_t>();
  for (int64_t batch_row = 0; batch_row < batch_size; ++batch_row) {
    TORCH_CHECK(
        request_slot_ptr[batch_row] >= 0,
        operation_name,
        ": request slot must be non-negative at batch row ",
        batch_row);
  }
  const auto calibrated_probability = [&](double probability, int64_t depth) {
    if (!calibration_enabled ||
        calibration_depth_active[static_cast<size_t>(depth - 1)] == 0 ||
        probability == 0.0) {
      return probability;
    }
    const int64_t bucket = std::min(
        static_cast<int64_t>(probability * calibration_bin_count),
        calibration_bin_count - 1);
    const double multiplier =
        calibration_ptr[(depth - 1) * calibration_bin_count + bucket];
    return std::min(1.0, probability * multiplier);
  };

  std::vector<DFlash2HeapEntry> heap;
  const bool use_lazy_siblings = !chain_mode && top_k <= 64;
  if (chain_mode || use_lazy_siblings) {
    heap.reserve(static_cast<size_t>(batch_size + capacity));
  } else {
    const int64_t initial_heap_bound =
        depth_count == 0 ? 0 : candidate_ids.numel() / depth_count;
    const int64_t added_per_node = top_k - 1;
    if (initial_heap_bound >= 0 &&
        (added_per_node == 0 ||
         capacity <=
             (std::numeric_limits<int64_t>::max() - initial_heap_bound) /
                 added_per_node)) {
      heap.reserve(
          static_cast<size_t>(initial_heap_bound + capacity * added_per_node));
    }
  }

  std::vector<DFlash2SiblingGroup> sibling_groups;
  if (use_lazy_siblings) {
    sibling_groups.reserve(static_cast<size_t>(batch_size + capacity));
  }

  const auto make_entry = [&](const DFlash2SiblingGroup& group,
                              int64_t token_rank,
                              int64_t sibling_group) {
    const int64_t token_row_offset =
        (group.batch_row * depth_count + (group.depth - 1)) * top_k;
    const int64_t probability_row_offset =
        ((group.batch_row * depth_count + (group.depth - 1)) * top_k +
         group.predecessor_rank) *
        top_k;
    const int64_t token_id = candidate_ptr[token_row_offset + token_rank];
    const double conditional_probability = static_cast<double>(
        probability_ptr[probability_row_offset + token_rank]);
    const double calibrated =
        calibrated_probability(conditional_probability, group.depth) *
        group.calibration_scale;
    const double rf_path_probability =
        group.parent_rf_path_probability * calibrated;
    double mix_path_probability = rf_path_probability;
    double priority = rf_path_probability;
    if (mix_enabled) {
      const double mixed_conditional =
          (1.0 - d2_mix_lambda) * calibrated +
          (token_rank == group.raw_argmax_rank ? d2_mix_lambda : 0.0);
      mix_path_probability =
          group.parent_mix_path_probability * mixed_conditional;
      priority = mix_path_probability *
          depth_discounts[static_cast<size_t>(group.depth)];
    }
    return DFlash2HeapEntry{
        -priority,
        request_slot_ptr[group.batch_row],
        group.depth,
        group.parent_id,
        token_rank,
        group.batch_row,
        token_id,
        conditional_probability,
        rf_path_probability,
        mix_path_probability,
        sibling_group};
  };

  const auto push_heap_entry = [&](DFlash2HeapEntry entry) {
    heap.push_back(entry);
    std::push_heap(heap.begin(), heap.end(), DFlash2HeapGreater{});
  };

  std::function<void(int64_t)> push_next_sibling;
  push_next_sibling = [&](int64_t group_index) {
    DFlash2SiblingGroup& group = sibling_groups[static_cast<size_t>(group_index)];
    if (group.remaining_mask == 0) {
      return;
    }
    int64_t best_rank = -1;
    DFlash2HeapEntry best{};
    uint64_t mask = group.remaining_mask;
    while (mask != 0) {
      const int64_t token_rank = static_cast<int64_t>(__builtin_ctzll(mask));
      const DFlash2HeapEntry candidate =
          make_entry(group, token_rank, group_index);
      if (best_rank == -1 || DFlash2HeapGreater{}(best, candidate)) {
        best_rank = token_rank;
        best = candidate;
      }
      mask &= mask - 1;
    }
    group.remaining_mask &= ~(uint64_t{1} << best_rank);
    push_heap_entry(best);
  };

  const auto push_children = [&](int64_t batch_row,
                                 int64_t depth,
                                 int64_t predecessor_rank,
                                 int64_t parent_id,
                                 double parent_rf_path_probability,
                                 double parent_mix_path_probability) {
    const int64_t token_row_offset = (batch_row * depth_count + (depth - 1)) * top_k;
    const int64_t probability_row_offset =
        ((batch_row * depth_count + (depth - 1)) * top_k + predecessor_rank) * top_k;
    if (chain_mode) {
      int64_t best_rank = 0;
      double best_probability = -1.0;
      for (int64_t token_rank = 0; token_rank < top_k; ++token_rank) {
        const int64_t token_id = candidate_ptr[token_row_offset + token_rank];
        const double probability = static_cast<double>(
            probability_ptr[probability_row_offset + token_rank]);
        TORCH_CHECK(
            token_id >= 0,
            operation_name,
            ": candidate token IDs must be non-negative");
        TORCH_CHECK(
            std::isfinite(probability) && probability >= 0.0,
            operation_name,
            ": probabilities must be finite and non-negative");
        if (probability > best_probability) {
          best_rank = token_rank;
          best_probability = probability;
        }
      }
      if (best_probability > 0.0) {
        const double path_probability =
            parent_rf_path_probability * best_probability;
        push_heap_entry(DFlash2HeapEntry{
            -path_probability,
            request_slot_ptr[batch_row],
            depth,
            parent_id,
            best_rank,
            batch_row,
            candidate_ptr[token_row_offset + best_rank],
            best_probability,
            path_probability,
            path_probability,
            -1});
      }
      return;
    }
    const bool scan_calibration =
        calibration_enabled &&
        calibration_depth_active[static_cast<size_t>(depth - 1)] != 0;
    double calibrated_mass = 0.0;
    bool calibration_changed = false;
    double calibration_scale = 1.0;
    int64_t raw_argmax_rank = -1;
    if (mix_enabled) {
      double raw_argmax_probability = -1.0;
      for (int64_t token_rank = 0; token_rank < top_k; ++token_rank) {
        const double probability = static_cast<double>(
            probability_ptr[probability_row_offset + token_rank]);
        TORCH_CHECK(
            std::isfinite(probability) && probability >= 0.0,
            operation_name,
            ": probabilities must be finite and non-negative");
        if (probability > raw_argmax_probability) {
          raw_argmax_probability = probability;
          raw_argmax_rank = token_rank;
        }
      }
    }
    if (scan_calibration && !use_lazy_siblings) {
      for (int64_t token_rank = 0; token_rank < top_k; ++token_rank) {
        const double probability = static_cast<double>(
            probability_ptr[probability_row_offset + token_rank]);
        TORCH_CHECK(
            std::isfinite(probability) && probability >= 0.0,
            operation_name,
            ": probabilities must be finite and non-negative");
        const int64_t bucket = std::min(
            static_cast<int64_t>(probability * calibration_bin_count),
            calibration_bin_count - 1);
        calibration_changed =
            calibration_changed ||
            calibration_ptr[
                (depth - 1) * calibration_bin_count + bucket] != 1.0;
        calibrated_mass += calibrated_probability(probability, depth);
      }
      if (calibration_changed && calibrated_mass > 1.0) {
        calibration_scale = 1.0 / calibrated_mass;
      }
    }
    uint64_t positive_mask = 0;
    for (int64_t token_rank = 0; token_rank < top_k; ++token_rank) {
      const int64_t token_id = candidate_ptr[token_row_offset + token_rank];
      const double conditional_probability =
          static_cast<double>(probability_ptr[probability_row_offset + token_rank]);
      TORCH_CHECK(
          token_id >= 0,
          operation_name,
          ": candidate token IDs must be non-negative");
      if (!scan_calibration || use_lazy_siblings) {
        TORCH_CHECK(
            std::isfinite(conditional_probability) &&
                conditional_probability >= 0.0,
            operation_name,
            ": probabilities must be finite and non-negative");
      }
      double calibrated = conditional_probability;
      if (scan_calibration && use_lazy_siblings) {
        const int64_t bucket = std::min(
            static_cast<int64_t>(
                conditional_probability * calibration_bin_count),
            calibration_bin_count - 1);
        calibration_changed =
            calibration_changed ||
            calibration_ptr[(depth - 1) * calibration_bin_count + bucket] !=
                1.0;
        calibrated = std::min(
            1.0,
            conditional_probability *
                calibration_ptr[
                    (depth - 1) * calibration_bin_count + bucket]);
        calibrated_mass += calibrated;
      } else {
        calibrated = calibrated_probability(conditional_probability, depth);
      }
      const double rf_conditional = calibrated * calibration_scale;
      const double rf_path_probability =
          parent_rf_path_probability * rf_conditional;
      double mix_path_probability = rf_path_probability;
      double priority = rf_path_probability;
      if (mix_enabled) {
        const double mixed_conditional =
            (1.0 - d2_mix_lambda) * rf_conditional +
            (token_rank == raw_argmax_rank ? d2_mix_lambda : 0.0);
        mix_path_probability =
            parent_mix_path_probability * mixed_conditional;
        priority = mix_path_probability *
            depth_discounts[static_cast<size_t>(depth)];
      }
      if (priority <= 0.0) {
        continue;
      }
      if (use_lazy_siblings) {
        positive_mask |= uint64_t{1} << token_rank;
      } else {
        push_heap_entry(DFlash2HeapEntry{
            -priority,
            request_slot_ptr[batch_row],
            depth,
            parent_id,
            token_rank,
            batch_row,
            token_id,
            conditional_probability,
            rf_path_probability,
            mix_path_probability,
            -1});
      }
    }
    if (use_lazy_siblings && calibration_changed && calibrated_mass > 1.0) {
      calibration_scale = 1.0 / calibrated_mass;
    }
    if (use_lazy_siblings && positive_mask != 0) {
      const int64_t group_index = static_cast<int64_t>(sibling_groups.size());
      sibling_groups.push_back(DFlash2SiblingGroup{
          batch_row,
          depth,
          predecessor_rank,
          parent_id,
          parent_rf_path_probability,
          parent_mix_path_probability,
          calibration_scale,
          raw_argmax_rank,
          positive_mask});
      push_next_sibling(group_index);
    }
  };

  if (depth_count > 0) {
    for (int64_t batch_row = 0; batch_row < batch_size; ++batch_row) {
      push_children(batch_row, 1, 0, -1, 1.0, 1.0);
    }
  }

  std::vector<int64_t> parent_ids;
  std::vector<int64_t> output_request_slots;
  std::vector<int64_t> output_token_ids;
  std::vector<int64_t> output_depths;
  std::vector<double> path_values;
  std::vector<double> raw_conditionals;
  parent_ids.reserve(static_cast<size_t>(capacity));
  output_request_slots.reserve(static_cast<size_t>(capacity));
  output_token_ids.reserve(static_cast<size_t>(capacity));
  output_depths.reserve(static_cast<size_t>(capacity));
  path_values.reserve(static_cast<size_t>(capacity));
  if (calibration_enabled) {
    raw_conditionals.reserve(static_cast<size_t>(capacity));
  }
  const bool per_request_cap_enabled =
      max_nodes_per_request > 0 || equal_mode;
  std::vector<int64_t> request_limits(
      per_request_cap_enabled ? static_cast<size_t>(batch_size) : 0,
      max_nodes_per_request);
  if (equal_mode && batch_size > 0) {
    const int64_t equal_quota = capacity / batch_size;
    const int64_t extra_quota = capacity % batch_size;
    for (int64_t batch_row = 0; batch_row < batch_size; ++batch_row) {
      const int64_t quota = equal_quota + (batch_row < extra_quota ? 1 : 0);
      request_limits[static_cast<size_t>(batch_row)] =
          max_nodes_per_request > 0
          ? std::min(quota, max_nodes_per_request)
          : quota;
    }
  }
  std::vector<int64_t> selected_per_request(
      per_request_cap_enabled ? static_cast<size_t>(batch_size) : 0, 0);
  // Append frontier marks and deduplicate once at the end. The old set path
  // allocated and performed a tree lookup for every mark on the hot path.
  std::vector<int64_t> capped_root_slots;
  std::vector<int64_t> capped_parent_ids;

  while (!heap.empty() && static_cast<int64_t>(parent_ids.size()) < capacity) {
    std::pop_heap(heap.begin(), heap.end(), DFlash2HeapGreater{});
    const DFlash2HeapEntry entry = heap.back();
    heap.pop_back();
    const int64_t request_limit =
        per_request_cap_enabled
        ? request_limits[static_cast<size_t>(entry.batch_row)]
        : -1;
    const bool request_is_capped =
        request_limit >= 0 &&
        selected_per_request[static_cast<size_t>(entry.batch_row)] >=
            request_limit;
    if (entry.sibling_group >= 0 && !request_is_capped) {
      push_next_sibling(entry.sibling_group);
    }
    if (request_is_capped) {
      if (entry.parent_id == -1) {
        capped_root_slots.push_back(entry.request_slot);
      } else {
        capped_parent_ids.push_back(entry.parent_id);
      }
      continue;
    }
    const int64_t node_id = start_node_id + static_cast<int64_t>(parent_ids.size());
    parent_ids.push_back(entry.parent_id);
    output_request_slots.push_back(entry.request_slot);
    output_token_ids.push_back(entry.token_id);
    output_depths.push_back(entry.depth);
    path_values.push_back(entry.rf_path_probability);
    if (per_request_cap_enabled) {
      ++selected_per_request[static_cast<size_t>(entry.batch_row)];
    }
    if (calibration_enabled) {
      raw_conditionals.push_back(entry.conditional_probability);
    }
    if (entry.depth < depth_count) {
      if (request_limit < 0 ||
          selected_per_request[static_cast<size_t>(entry.batch_row)] <
              request_limit) {
        push_children(
            entry.batch_row,
            entry.depth + 1,
            entry.token_rank,
            node_id,
            entry.rf_path_probability,
            entry.mix_path_probability);
      } else {
        capped_parent_ids.push_back(node_id);
      }
    }
  }

  std::vector<int64_t> truncated_root_slots = std::move(capped_root_slots);
  std::vector<int64_t> truncated_parent_ids = std::move(capped_parent_ids);
  for (const DFlash2HeapEntry& entry : heap) {
    if (entry.parent_id == -1) {
      truncated_root_slots.push_back(entry.request_slot);
    } else {
      truncated_parent_ids.push_back(entry.parent_id);
    }
  }
  std::sort(truncated_root_slots.begin(), truncated_root_slots.end());
  truncated_root_slots.erase(
      std::unique(truncated_root_slots.begin(), truncated_root_slots.end()),
      truncated_root_slots.end());
  std::sort(truncated_parent_ids.begin(), truncated_parent_ids.end());
  truncated_parent_ids.erase(
      std::unique(truncated_parent_ids.begin(), truncated_parent_ids.end()),
      truncated_parent_ids.end());

  const at::TensorOptions index_options = candidate_ids.options();
  const at::TensorOptions value_options = probabilities.options().dtype(at::kDouble);
  std::vector<at::Tensor> outputs{
      request_forest_vector_to_tensor(parent_ids, index_options),
      request_forest_vector_to_tensor(output_request_slots, index_options),
      request_forest_vector_to_tensor(output_token_ids, index_options),
      request_forest_vector_to_tensor(output_depths, index_options),
      request_forest_vector_to_tensor(path_values, value_options),
      request_forest_vector_to_tensor(truncated_root_slots, index_options),
      request_forest_vector_to_tensor(truncated_parent_ids, index_options),
  };
  if (calibration_enabled) {
    outputs.push_back(
        request_forest_vector_to_tensor(raw_conditionals, value_options));
  }
  return outputs;
}

at::Tensor request_forest_select_cpu(const at::Tensor& parent_ids, const at::Tensor& values, int64_t capacity) {
  check_cpu_contiguous_1d(parent_ids, at::kLong, "parent_ids");
  check_cpu_contiguous_1d(values, at::kDouble, "values");
  check_same_numel(values, parent_ids.numel(), "values");
  TORCH_CHECK(capacity >= 0, "request_forest_select_cpu: capacity must be non-negative");

  const int64_t node_count = parent_ids.numel();
  const int64_t* parent_ptr = parent_ids.data_ptr<int64_t>();
  const double* value_ptr = values.data_ptr<double>();
  for (int64_t node_id = 0; node_id < node_count; ++node_id) {
    const int64_t parent_id = parent_ptr[node_id];
    TORCH_CHECK(
        parent_id == -1 || (parent_id >= 0 && parent_id < node_id),
        "request_forest_select_cpu: parent of node ",
        node_id,
        " must be -1 or an earlier node, got ",
        parent_id);
    TORCH_CHECK(
        std::isfinite(value_ptr[node_id]),
        "request_forest_select_cpu: value for node ",
        node_id,
        " must be finite");
    if (parent_id != -1) {
      TORCH_CHECK(
          value_ptr[node_id] <= value_ptr[parent_id],
          "request_forest_select_cpu: child value exceeds parent value at node ",
          node_id);
    }
  }

  return selected_ids_to_tensor(
      rank_positive_nodes(value_ptr, node_count, capacity), parent_ids.options());
}

std::vector<at::Tensor> request_forest_compile_cpu(
    const at::Tensor& node_parent_ids,
    const at::Tensor& node_request_slots,
    const at::Tensor& node_token_ids,
    const at::Tensor& node_depths,
    const at::Tensor& anchor_token_ids,
    const at::Tensor& anchor_positions,
    const at::Tensor& selected_node_ids,
    int64_t physical_rows,
    int64_t padding_token_id) {
  constexpr const char* operation_name = "request_forest_compile_cpu";
  check_compile_input_schema(
      node_parent_ids,
      node_request_slots,
      node_token_ids,
      node_depths,
      anchor_token_ids,
      anchor_positions,
      physical_rows,
      padding_token_id,
      operation_name);
  check_cpu_contiguous_1d(selected_node_ids, at::kLong, "selected_node_ids");
  validate_compile_metadata(
      node_parent_ids,
      node_request_slots,
      node_token_ids,
      node_depths,
      anchor_token_ids,
      anchor_positions,
      nullptr,
      operation_name);

  const int64_t node_count = node_parent_ids.numel();
  const int64_t* parent_ptr = node_parent_ids.data_ptr<int64_t>();
  const int64_t* requested_ptr = selected_node_ids.data_ptr<int64_t>();
  std::vector<uint8_t> selected(static_cast<size_t>(node_count), uint8_t{0});
  std::vector<uint8_t> requested_seen(static_cast<size_t>(node_count), uint8_t{0});
  for (int64_t index = 0; index < selected_node_ids.numel(); ++index) {
    const int64_t requested_id = requested_ptr[index];
    TORCH_CHECK(
        requested_id >= 0 && requested_id < node_count,
        "request_forest_compile_cpu: selected node id is out of range: ",
        requested_id);
    TORCH_CHECK(
        !requested_seen[requested_id],
        "request_forest_compile_cpu: duplicate selected node id ",
        requested_id);
    requested_seen[requested_id] = 1;

    int64_t node_id = requested_id;
    while (node_id != -1 && !selected[node_id]) {
      selected[node_id] = 1;
      node_id = parent_ptr[node_id];
    }
  }

  return emit_verify_plan_from_selection(
      node_parent_ids,
      node_request_slots,
      node_token_ids,
      node_depths,
      anchor_token_ids,
      anchor_positions,
      selected,
      physical_rows,
      padding_token_id,
      operation_name);
}

std::vector<at::Tensor> request_forest_select_compile_cpu(
    const at::Tensor& node_parent_ids,
    const at::Tensor& node_request_slots,
    const at::Tensor& node_token_ids,
    const at::Tensor& node_depths,
    const at::Tensor& node_values,
    const at::Tensor& anchor_token_ids,
    const at::Tensor& anchor_positions,
    int64_t capacity,
    int64_t physical_rows,
    int64_t padding_token_id) {
  constexpr const char* operation_name = "request_forest_select_compile_cpu";
  check_compile_input_schema(
      node_parent_ids,
      node_request_slots,
      node_token_ids,
      node_depths,
      anchor_token_ids,
      anchor_positions,
      physical_rows,
      padding_token_id,
      operation_name);
  check_cpu_contiguous_1d(node_values, at::kDouble, "node_values");
  check_same_numel(node_values, node_parent_ids.numel(), "node_values");
  TORCH_CHECK(capacity >= 0, operation_name, ": capacity must be non-negative");

  const double* value_ptr = node_values.data_ptr<double>();
  validate_compile_metadata(
      node_parent_ids,
      node_request_slots,
      node_token_ids,
      node_depths,
      anchor_token_ids,
      anchor_positions,
      value_ptr,
      operation_name);

  const std::vector<int64_t> ranked_ids =
      rank_positive_nodes(value_ptr, node_parent_ids.numel(), capacity);
  std::vector<uint8_t> selected(
      static_cast<size_t>(node_parent_ids.numel()), uint8_t{0});
  for (const int64_t node_id : ranked_ids) {
    selected[node_id] = 1;
  }

  at::Tensor selected_ids =
      selected_ids_to_tensor(ranked_ids, node_parent_ids.options());
  std::vector<at::Tensor> plan = emit_verify_plan_from_selection(
      node_parent_ids,
      node_request_slots,
      node_token_ids,
      node_depths,
      anchor_token_ids,
      anchor_positions,
      selected,
      physical_rows,
      padding_token_id,
      operation_name);

  std::vector<at::Tensor> outputs;
  outputs.reserve(plan.size() + 1);
  outputs.push_back(std::move(selected_ids));
  for (at::Tensor& tensor : plan) {
    outputs.push_back(std::move(tensor));
  }
  return outputs;
}

std::vector<at::Tensor> request_forest_commit_cpu(
    const at::Tensor& anchor_rows,
    const at::Tensor& token_ids,
    const at::Tensor& first_child,
    const at::Tensor& next_sibling,
    const at::Tensor& positions,
    const at::Tensor& has_unverified_children,
    const at::Tensor& packed_to_node,
    const at::Tensor& target_tokens,
    const at::Tensor& target_eos,
    int64_t logical_rows) {
  check_cpu_contiguous_1d(anchor_rows, at::kLong, "anchor_rows");
  check_cpu_contiguous_1d(token_ids, at::kLong, "token_ids");
  check_cpu_contiguous_1d(first_child, at::kLong, "first_child");
  check_cpu_contiguous_1d(next_sibling, at::kLong, "next_sibling");
  check_cpu_contiguous_1d(positions, at::kLong, "positions");
  check_cpu_contiguous_1d(has_unverified_children, at::kBool, "has_unverified_children");
  check_cpu_contiguous_1d(packed_to_node, at::kLong, "packed_to_node");
  check_cpu_contiguous_1d(target_tokens, at::kLong, "target_tokens");
  check_cpu_contiguous_1d(target_eos, at::kBool, "target_eos");

  const int64_t physical_rows = token_ids.numel();
  const int64_t request_count = anchor_rows.numel();
  check_same_numel(first_child, physical_rows, "first_child");
  check_same_numel(next_sibling, physical_rows, "next_sibling");
  check_same_numel(positions, physical_rows, "positions");
  check_same_numel(has_unverified_children, physical_rows, "has_unverified_children");
  check_same_numel(packed_to_node, physical_rows, "packed_to_node");
  TORCH_CHECK(request_count > 0, "request_forest_commit_cpu: at least one request is required");
  TORCH_CHECK(
      logical_rows >= request_count && logical_rows <= physical_rows,
      "request_forest_commit_cpu: logical_rows is outside the valid range");
  TORCH_CHECK(
      target_tokens.numel() >= logical_rows && target_eos.numel() >= logical_rows,
      "request_forest_commit_cpu: target decision arrays do not cover every logical row");

  const int64_t* anchor_ptr = anchor_rows.data_ptr<int64_t>();
  for (int64_t request_slot = 0; request_slot < request_count; ++request_slot) {
    const int64_t anchor_row = anchor_ptr[request_slot];
    const int64_t next_anchor = request_slot + 1 < request_count ? anchor_ptr[request_slot + 1] : logical_rows;
    TORCH_CHECK(
        anchor_row >= 0 && anchor_row < logical_rows,
        "request_forest_commit_cpu: anchor row is outside the logical plan");
    TORCH_CHECK(
        request_slot == 0 ? anchor_row == 0 : anchor_row > anchor_ptr[request_slot - 1],
        "request_forest_commit_cpu: anchor rows must start at zero and be strictly increasing");
    TORCH_CHECK(
        next_anchor > anchor_row,
        "request_forest_commit_cpu: every request segment must contain its anchor");
  }

  const int64_t* token_ptr = token_ids.data_ptr<int64_t>();
  const int64_t* first_child_ptr = first_child.data_ptr<int64_t>();
  const int64_t* next_sibling_ptr = next_sibling.data_ptr<int64_t>();
  const int64_t* position_ptr = positions.data_ptr<int64_t>();
  const bool* unverified_ptr = has_unverified_children.data_ptr<bool>();
  const int64_t* packed_to_node_ptr = packed_to_node.data_ptr<int64_t>();
  const int64_t* target_token_ptr = target_tokens.data_ptr<int64_t>();
  const bool* target_eos_ptr = target_eos.data_ptr<bool>();

  const at::TensorOptions index_options = token_ids.options();
  at::Tensor accepted_lengths = at::empty({request_count}, index_options);
  at::Tensor first_miss_positions = at::full({request_count}, -1, index_options);
  at::Tensor correction_token_ids = at::full({request_count}, -1, index_options);
  at::Tensor termination_reasons = at::empty({request_count}, index_options);
  at::Tensor visited_rows = at::empty({request_count}, index_options);
  at::Tensor output_lengths = at::empty({request_count}, index_options);

  int64_t* accepted_lengths_ptr = accepted_lengths.data_ptr<int64_t>();
  int64_t* first_miss_ptr = first_miss_positions.data_ptr<int64_t>();
  int64_t* correction_ptr = correction_token_ids.data_ptr<int64_t>();
  int64_t* reason_ptr = termination_reasons.data_ptr<int64_t>();
  int64_t* visited_ptr = visited_rows.data_ptr<int64_t>();
  int64_t* output_lengths_ptr = output_lengths.data_ptr<int64_t>();

  at::parallel_for(0, request_count, 0, [&](int64_t begin, int64_t end) {
    for (int64_t request_slot = begin; request_slot < end; ++request_slot) {
      const int64_t anchor_row = anchor_ptr[request_slot];
      const int64_t segment_end = request_slot + 1 < request_count ? anchor_ptr[request_slot + 1] : logical_rows;
      const TraversalSummary summary = traverse_request(
          token_ptr,
          first_child_ptr,
          next_sibling_ptr,
          position_ptr,
          unverified_ptr,
          packed_to_node_ptr,
          target_token_ptr,
          target_eos_ptr,
          anchor_row,
          segment_end,
          nullptr,
          nullptr);
      accepted_lengths_ptr[request_slot] = summary.accepted_length;
      output_lengths_ptr[request_slot] = summary.output_length;
      first_miss_ptr[request_slot] = summary.first_miss_position;
      correction_ptr[request_slot] = summary.correction_token_id;
      reason_ptr[request_slot] = summary.reason;
      visited_ptr[request_slot] = summary.visited_row;
    }
  });

  at::Tensor accepted_offsets = at::empty({request_count + 1}, index_options);
  at::Tensor output_offsets = at::empty({request_count + 1}, index_options);
  int64_t* accepted_offsets_ptr = accepted_offsets.data_ptr<int64_t>();
  int64_t* output_offsets_ptr = output_offsets.data_ptr<int64_t>();
  accepted_offsets_ptr[0] = 0;
  output_offsets_ptr[0] = 0;
  for (int64_t request_slot = 0; request_slot < request_count; ++request_slot) {
    accepted_offsets_ptr[request_slot + 1] = accepted_offsets_ptr[request_slot] + accepted_lengths_ptr[request_slot];
    output_offsets_ptr[request_slot + 1] = output_offsets_ptr[request_slot] + output_lengths_ptr[request_slot];
  }

  const int64_t total_accepted = accepted_offsets_ptr[request_count];
  const int64_t total_output = output_offsets_ptr[request_count];
  at::Tensor accepted_flat = at::empty({total_accepted}, index_options);
  at::Tensor output_flat = at::empty({total_output}, index_options);
  int64_t* accepted_flat_ptr =
      total_accepted == 0 ? nullptr : accepted_flat.data_ptr<int64_t>();
  int64_t* output_flat_ptr =
      total_output == 0 ? nullptr : output_flat.data_ptr<int64_t>();

  at::parallel_for(0, request_count, 0, [&](int64_t begin, int64_t end) {
    for (int64_t request_slot = begin; request_slot < end; ++request_slot) {
      const int64_t anchor_row = anchor_ptr[request_slot];
      const int64_t segment_end = request_slot + 1 < request_count ? anchor_ptr[request_slot + 1] : logical_rows;
      int64_t* accepted_output = accepted_lengths_ptr[request_slot] == 0
          ? nullptr
          : accepted_flat_ptr + accepted_offsets_ptr[request_slot];
      int64_t* token_output = output_lengths_ptr[request_slot] == 0
          ? nullptr
          : output_flat_ptr + output_offsets_ptr[request_slot];
      const TraversalSummary summary = traverse_request(
          token_ptr,
          first_child_ptr,
          next_sibling_ptr,
          position_ptr,
          unverified_ptr,
          packed_to_node_ptr,
          target_token_ptr,
          target_eos_ptr,
          anchor_row,
          segment_end,
          accepted_output,
          token_output);
      TORCH_CHECK(
          summary.accepted_length == accepted_lengths_ptr[request_slot] &&
              summary.output_length == output_lengths_ptr[request_slot],
          "request_forest_commit_cpu: traversal changed between sizing and write passes");
    }
  });

  return {
      accepted_flat,
      accepted_offsets,
      output_flat,
      output_offsets,
      accepted_lengths,
      first_miss_positions,
      correction_token_ids,
      termination_reasons,
      visited_rows,
  };
}
