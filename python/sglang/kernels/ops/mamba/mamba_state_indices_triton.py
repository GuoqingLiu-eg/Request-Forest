import torch
import triton
import triton.language as tl


@triton.jit
def _fused_replay_state_indices_kernel(
    req_pool_indices_ptr,
    mamba_map_ptr,
    out_ptr,
    valid_bs,
    total_bs,
    BS_UPPER: tl.constexpr,
):
    offs = tl.arange(0, BS_UPPER)
    in_range = offs < total_bs
    valid = offs < valid_bs
    req = tl.load(req_pool_indices_ptr + offs, mask=valid, other=0)
    idx = tl.load(mamba_map_ptr + req, mask=valid, other=0)
    out_val = tl.where(valid, idx.to(tl.int32), -1)
    tl.store(out_ptr + offs, out_val, mask=in_range)


    zeros = tl.zeros([BS_UPPER], dtype=req.dtype)
    tl.store(req_pool_indices_ptr + offs, zeros, mask=in_range & (~valid))


def fused_replay_state_indices(
    *,
    req_pool_indices: torch.Tensor,
    mamba_index_mapping: torch.Tensor,
    out_state_indices: torch.Tensor,
    valid_bs: int,
    total_bs: int,
) -> torch.Tensor:

    _fused_replay_state_indices_kernel[(1,)](
        req_pool_indices,
        mamba_index_mapping,
        out_state_indices,
        valid_bs,
        total_bs,
        BS_UPPER=triton.next_power_of_2(total_bs),
    )
    return out_state_indices[:total_bs]
