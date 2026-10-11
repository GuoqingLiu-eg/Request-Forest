import torch


def topk_index_reduce(tensor: torch.Tensor, dim: int) -> torch.Tensor:


    tensor_permuted = torch.movedim(tensor, source=dim, destination=-2)


    combined = tensor_permuted.flatten(start_dim=-2)


    sorted_vals, _ = combined.sort(dim=-1)


    is_new_element = sorted_vals[..., 1:] != sorted_vals[..., :-1]


    first_col_true = torch.ones_like(sorted_vals[..., :1], dtype=torch.bool)
    non_duplicate_mask = torch.cat([first_col_true, is_new_element], dim=-1)


    valid_mask = non_duplicate_mask & (sorted_vals != -1)


    sort_idx = torch.argsort((~valid_mask).int(), dim=-1, stable=True)
    result = torch.gather(sorted_vals, -1, sort_idx)


    valid_count = valid_mask.sum(dim=-1, keepdim=True)
    total_cols = result.size(-1)


    idx_range = torch.arange(total_cols, device=tensor.device)

    final_result = torch.where(idx_range < valid_count, result, -1)

    return final_result
