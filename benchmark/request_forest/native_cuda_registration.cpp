#include <ATen/ATen.h>
#include <torch/library.h>

#include <tuple>

void request_forest_build_tree_attention(
    const at::Tensor& ancestor_row_ptr,
    const at::Tensor& ancestor_columns,
    const at::Tensor& out_cache_loc,
    at::Tensor page_table,
    at::Tensor cache_seqlens,
    at::Tensor cu_seqlens_q);

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
    const at::Tensor& eos_token_ids);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
request_forest_commit_kv(
    const at::Tensor& accepted_rows,
    const at::Tensor& commit_lens,
    const at::Tensor& scratch_slots,
    at::Tensor req_to_token,
    const at::Tensor& req_pool_indices,
    const at::Tensor& prefix_lens,
    int64_t committed_count);

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def(
      "request_forest_build_tree_attention(Tensor ancestor_row_ptr, "
      "Tensor ancestor_columns, Tensor out_cache_loc, Tensor(a!) page_table, "
      "Tensor(b!) cache_seqlens, Tensor(c!) cu_seqlens_q) -> ()");
  m.impl(
      "request_forest_build_tree_attention",
      at::kCUDA,
      &request_forest_build_tree_attention);
  m.def(
      "request_forest_verify_greedy(Tensor(a!) predicts, "
      "Tensor(b!) output_token_ids, Tensor(c!) accept_index, "
      "Tensor(d!) accepted_draft_lens, Tensor(e!) accept_lens, "
      "Tensor(f!) first_miss_positions, Tensor(g!) termination_reasons, "
      "Tensor draft_token, Tensor first_child, Tensor next_sibling, "
      "Tensor positions, Tensor has_unverified_children, "
      "Tensor request_offsets, Tensor anchor_rows, Tensor target_predict, "
      "Tensor request_ignore_eos, Tensor eos_token_ids) -> ()");
  m.impl(
      "request_forest_verify_greedy",
      at::kCUDA,
      &request_forest_verify_greedy);
  m.def(
      "request_forest_commit_kv(Tensor accepted_rows, Tensor commit_lens, "
      "Tensor scratch_slots, Tensor(a!) req_to_token, Tensor req_pool_indices, "
      "Tensor prefix_lens, int committed_count) -> "
      "(Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.impl(
      "request_forest_commit_kv",
      at::kCUDA,
      &request_forest_commit_kv);
}
