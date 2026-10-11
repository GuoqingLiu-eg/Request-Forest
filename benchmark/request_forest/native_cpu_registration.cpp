#include <ATen/ATen.h>
#include <torch/library.h>

#include <cstdint>
#include <optional>
#include <vector>

at::Tensor request_forest_select_cpu(
    const at::Tensor& parent_ids, const at::Tensor& values, int64_t capacity);

std::vector<at::Tensor> request_forest_dflash2_lower_cpu(
    const at::Tensor& candidate_ids,
    const at::Tensor& probabilities,
    const at::Tensor& request_slots,
    int64_t capacity,
    int64_t start_node_id,
    const std::optional<at::Tensor>& calibration_multipliers,
    int64_t max_nodes_per_request,
    int64_t selection_mode,
    double d2_mix_lambda);

std::vector<at::Tensor> request_forest_compile_cpu(
    const at::Tensor& node_parent_ids,
    const at::Tensor& node_request_slots,
    const at::Tensor& node_token_ids,
    const at::Tensor& node_depths,
    const at::Tensor& anchor_token_ids,
    const at::Tensor& anchor_positions,
    const at::Tensor& selected_node_ids,
    int64_t physical_rows,
    int64_t padding_token_id);

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
    int64_t padding_token_id);

std::vector<at::Tensor> request_forest_commit_cpu(
    const at::Tensor& anchor_rows,
    const at::Tensor& token_ids,
    const at::Tensor& first_child,
    const at::Tensor& next_sibling,
    const at::Tensor& positions,
    const at::Tensor& has_unverified_children,
    const at::Tensor& packed_to_node,
    const at::Tensor& target_token_ids,
    const at::Tensor& target_is_eos,
    int64_t logical_rows);

#ifdef REQUEST_FOREST_WITH_EAGLE
void build_tree_kernel_efficient_cpu(
    const at::Tensor& parent_list,
    const at::Tensor& selected_index,
    const at::Tensor& verified_seq_len,
    at::Tensor tree_mask,
    at::Tensor positions,
    at::Tensor retrive_index,
    at::Tensor retrive_next_token,
    at::Tensor retrive_next_sibling,
    int64_t topk,
    int64_t depth,
    int64_t draft_token_num,
    int64_t tree_mask_mode);

namespace {

std::vector<at::Tensor> eagle_build_tree_alloc_cpu(
    const at::Tensor& parent_list,
    const at::Tensor& selected_index,
    const at::Tensor& seq_lens,
    int64_t topk,
    int64_t depth,
    int64_t draft_token_num) {
  TORCH_CHECK(parent_list.dim() == 2, "parent_list must be rank 2");
  TORCH_CHECK(draft_token_num >= 1, "draft_token_num must be positive");

  const int64_t batch_size = parent_list.size(0);
  const auto index_options = parent_list.options();
  auto tree_mask = at::empty(
      {batch_size * draft_token_num * draft_token_num},
      index_options.dtype(at::kBool));
  auto positions = at::empty({batch_size * draft_token_num}, index_options);
  auto retrieve =
      at::full({3, batch_size, draft_token_num}, -1, index_options);
  auto retrieve_index = retrieve.select(0, 0);
  auto retrieve_next_token = retrieve.select(0, 1);
  auto retrieve_next_sibling = retrieve.select(0, 2);

  constexpr int64_t kQlenOnlyTreeMask = 1;
  build_tree_kernel_efficient_cpu(
      parent_list,
      selected_index,
      seq_lens,
      tree_mask,
      positions,
      retrieve_index,
      retrieve_next_token,
      retrieve_next_sibling,
      topk,
      depth,
      draft_token_num,
      kQlenOnlyTreeMask);

  return {
      tree_mask,
      positions,
      retrieve_index,
      retrieve_next_token,
      retrieve_next_sibling};
}

}  // namespace
#endif

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def(
      "request_forest_select_cpu(Tensor parent_ids, Tensor values, int capacity) -> Tensor");
  m.impl(
      "request_forest_select_cpu", at::kCPU, &request_forest_select_cpu);

  m.def(
      "request_forest_dflash2_lower_cpu(Tensor candidate_ids, "
      "Tensor probabilities, Tensor request_slots, int capacity, "
      "int start_node_id, Tensor? calibration_multipliers=None, "
      "int max_nodes_per_request=-1, int selection_mode=0, "
      "float d2_mix_lambda=0.0) -> Tensor[]");
  m.impl(
      "request_forest_dflash2_lower_cpu",
      at::kCPU,
      &request_forest_dflash2_lower_cpu);

  m.def(
      "request_forest_compile_cpu(Tensor node_parent_ids, Tensor node_request_slots, "
      "Tensor node_token_ids, Tensor node_depths, Tensor anchor_token_ids, "
      "Tensor anchor_positions, Tensor selected_node_ids, int physical_rows, "
      "int padding_token_id=0) -> Tensor[]");
  m.impl(
      "request_forest_compile_cpu", at::kCPU, &request_forest_compile_cpu);

  m.def(
      "request_forest_select_compile_cpu(Tensor node_parent_ids, "
      "Tensor node_request_slots, Tensor node_token_ids, Tensor node_depths, "
      "Tensor node_values, Tensor anchor_token_ids, Tensor anchor_positions, "
      "int capacity, int physical_rows, int padding_token_id=0) -> Tensor[]");
  m.impl(
      "request_forest_select_compile_cpu",
      at::kCPU,
      &request_forest_select_compile_cpu);

  m.def(
      "request_forest_commit_cpu(Tensor anchor_rows, Tensor token_ids, "
      "Tensor first_child, Tensor next_sibling, Tensor positions, "
      "Tensor has_unverified_children, Tensor packed_to_node, "
      "Tensor target_token_ids, Tensor target_is_eos, int logical_rows) -> Tensor[]");
  m.impl(
      "request_forest_commit_cpu", at::kCPU, &request_forest_commit_cpu);

#ifdef REQUEST_FOREST_WITH_EAGLE
  m.def(
      "build_tree_kernel_efficient_cpu(Tensor parent_list, Tensor selected_index, "
      "Tensor verified_seq_len, Tensor(a!) tree_mask, Tensor(a!) positions, "
      "Tensor(a!) retrive_index, Tensor(a!) retrive_next_token, "
      "Tensor(a!) retrive_next_sibling, int topk, int depth, "
      "int draft_token_num, int tree_mask_mode) -> ()");
  m.impl(
      "build_tree_kernel_efficient_cpu",
      at::kCPU,
      &build_tree_kernel_efficient_cpu);

  m.def(
      "eagle_build_tree_alloc_cpu(Tensor parent_list, Tensor selected_index, "
      "Tensor seq_lens, int topk, int depth, int draft_token_num) -> Tensor[]");
  m.impl(
      "eagle_build_tree_alloc_cpu", at::kCPU, &eagle_build_tree_alloc_cpu);
#endif
}
