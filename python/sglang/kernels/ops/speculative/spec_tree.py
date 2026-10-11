# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import triton
import triton.language as tl


@triton.jit
def sgl_build_tree_kernel_efficient_triton(
    parent_list_ptr,
    selected_index_ptr,
    verified_seq_len_ptr,
    seq_len_prefix_sum_ptr,
    tree_mask_ptr,
    positions_ptr,
    retrieve_index_ptr,
    retrieve_next_token_ptr,
    retrieve_next_sibling_ptr,
    topk: tl.constexpr,
    depth: tl.constexpr,
    draft_token_num: tl.constexpr,
    tree_mask_mode: tl.constexpr,
    batch_size: tl.constexpr,
    parent_list_stride: tl.constexpr,
    selected_index_stride: tl.constexpr,
):

    batch_idx = tl.program_id(0)


    seq_len = tl.load(verified_seq_len_ptr + batch_idx)
    seq_len_prefix_sum = tl.load(seq_len_prefix_sum_ptr + batch_idx)


    seq_tree_idx = (
        tl.cast(draft_token_num * draft_token_num * batch_idx, seq_len.dtype)
        + seq_len_prefix_sum * draft_token_num
    )

    positions_offset = batch_idx * draft_token_num
    tl.store(positions_ptr + positions_offset, seq_len)

    retrieve_index_offset = batch_idx * draft_token_num


    for i in range(draft_token_num - 1, 0, -1):
        current_token_idx = retrieve_index_offset + i
        tl.store(
            retrieve_index_ptr + batch_idx * draft_token_num + i,
            current_token_idx,
        )

        parent_tb_idx = (
            tl.load(selected_index_ptr + batch_idx * selected_index_stride + (i - 1))
            // topk
        )
        parent_position = 0
        found = 0

        if parent_tb_idx == 0:
            found = 1
        else:
            parent_token_idx = tl.load(
                parent_list_ptr + batch_idx * parent_list_stride + parent_tb_idx
            )


            for pp in range(draft_token_num - 1):
                if found == 0:
                    sel_idx = tl.load(
                        selected_index_ptr + batch_idx * selected_index_stride + pp
                    )
                    if sel_idx == parent_token_idx:
                        parent_position = pp + 1
                        found = 1

        if found == 1:

            next_tok_addr = (
                retrieve_next_token_ptr + batch_idx * draft_token_num + parent_position
            )
            next_tok = tl.load(next_tok_addr)

            if next_tok == -1:
                tl.store(next_tok_addr, i)
            else:
                tl.store(next_tok_addr, i)
                tl.store(
                    retrieve_next_sibling_ptr + batch_idx * draft_token_num + i,
                    next_tok,
                )

    tl.store(retrieve_index_ptr + batch_idx * draft_token_num, retrieve_index_offset)


    for draft_tokenx in range(draft_token_num):
        if tree_mask_mode == 0:
            token_tree_idx = (
                seq_tree_idx + (seq_len + draft_token_num) * draft_tokenx + seq_len + 1
            )
        else:
            token_tree_idx = (
                draft_token_num * draft_token_num * batch_idx
                + draft_token_num * draft_tokenx
                + 1
            )

        tl.store(tree_mask_ptr + token_tree_idx - 1, 1)
        for i in range(draft_token_num - 1):
            tl.store(tree_mask_ptr + token_tree_idx + i, 0)

        if draft_tokenx > 0:

            cur_position = draft_tokenx - 1
            position = 0
            should_continue = 1

            for _ in range(depth):
                if should_continue:
                    position += 1
                    tl.store(tree_mask_ptr + token_tree_idx + cur_position, 1)

                    parent_tb_idx = (
                        tl.load(
                            selected_index_ptr
                            + batch_idx * selected_index_stride
                            + cur_position
                        )
                        // topk
                    )
                    if parent_tb_idx == 0:
                        should_continue = 0
                    else:
                        parent_token_idx = tl.load(
                            parent_list_ptr
                            + batch_idx * parent_list_stride
                            + parent_tb_idx
                        )


                        found = 0
                        for cp in range(draft_token_num - 1):
                            if found == 0:
                                if (
                                    tl.load(
                                        selected_index_ptr
                                        + batch_idx * selected_index_stride
                                        + cp
                                    )
                                    == parent_token_idx
                                ):
                                    cur_position = cp
                                    found = 1
                        if found == 0:
                            should_continue = 0

            tl.store(
                positions_ptr + batch_idx * draft_token_num + draft_tokenx,
                position + seq_len,
            )


@triton.jit
def verify_tree_greedy_kernel_triton(
    predicts_ptr,
    accept_index_ptr,
    accept_token_num_ptr,
    candidates_ptr,
    retrieve_index_ptr,
    retrieve_next_token_ptr,
    retrieve_next_sibling_ptr,
    target_predict_ptr,
    batch_size: tl.constexpr,
    num_speculative_tokens: tl.constexpr,
    num_draft_tokens: tl.constexpr,
):

    bx = tl.program_id(0)


    last_accept_retrieve_idx = tl.load(retrieve_index_ptr + bx * num_draft_tokens)
    tl.store(accept_index_ptr + bx * num_speculative_tokens, last_accept_retrieve_idx)

    num_accept_tokens = tl.cast(0, last_accept_retrieve_idx.dtype)
    cur_index = tl.cast(0, last_accept_retrieve_idx.dtype)


    should_continue = 1
    for j in range(1, num_speculative_tokens):
        if should_continue:
            cur_index = tl.load(
                retrieve_next_token_ptr + bx * num_draft_tokens + cur_index
            )


            target_row = last_accept_retrieve_idx // num_draft_tokens
            target_col = last_accept_retrieve_idx % num_draft_tokens
            target_token = tl.load(
                target_predict_ptr + target_row * num_draft_tokens + target_col
            )


            found_match = 0
            for _ in range(num_draft_tokens):
                if found_match == 0:

                    is_valid = cur_index != -1


                    safe_cur_index = (
                        cur_index * is_valid
                    )
                    safe_index = bx * num_draft_tokens + safe_cur_index


                    draft_index = tl.load(retrieve_index_ptr + safe_index)
                    draft_token = tl.load(candidates_ptr + safe_index)


                    token_match = is_valid & (draft_token == target_token)


                    tl.store(
                        predicts_ptr + last_accept_retrieve_idx,
                        target_token,
                        mask=token_match,
                    )
                    next_num_accept_tokens = num_accept_tokens + 1
                    tl.store(
                        accept_index_ptr
                        + bx * num_speculative_tokens
                        + next_num_accept_tokens,
                        draft_index,
                        mask=token_match,
                    )

                    num_accept_tokens = num_accept_tokens + token_match
                    last_accept_retrieve_idx = (
                        token_match * draft_index
                        + (~token_match) * last_accept_retrieve_idx
                    )
                    found_match = token_match * 1 + (~is_valid) * (-1)


                    cur_index = tl.load(
                        retrieve_next_sibling_ptr + safe_index,
                        mask=~token_match
                        & is_valid,
                        other=cur_index,
                    )

            if found_match != 1:
                should_continue = 0


    tl.store(accept_token_num_ptr + bx, num_accept_tokens)

    target_row = last_accept_retrieve_idx // num_draft_tokens
    target_col = last_accept_retrieve_idx % num_draft_tokens
    final_target = tl.load(
        target_predict_ptr + target_row * num_draft_tokens + target_col
    )
    tl.store(predicts_ptr + last_accept_retrieve_idx, final_target)
