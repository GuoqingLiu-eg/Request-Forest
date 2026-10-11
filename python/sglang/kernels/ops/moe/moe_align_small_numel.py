import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl


SMALL_NUMEL_LIMIT = 64


@triton.jit
def _moe_align_small_numel_kernel(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_pad_ptr,
    num_experts,
    block_size,
    numel,
    NP: tl.constexpr,
    NB: tl.constexpr,
    USE_GDC: tl.constexpr = False,
):

    if USE_GDC:

        tl.extra.cuda.gdc_wait()

    offs_p = tl.arange(0, NP)
    mask_p = offs_p < numel
    ids = tl.load(topk_ids_ptr + offs_p, mask=mask_p, other=-2)

    bucket = tl.where(mask_p, (ids + 1).to(tl.int32), num_experts)


    same = (bucket[None, :] == bucket[:, None]) & mask_p[None, :] & mask_p[:, None]
    earlier = offs_p[None, :] < offs_p[:, None]
    rank = tl.sum((same & earlier).to(tl.int32), axis=1)
    cnt = tl.sum(same.to(tl.int32), axis=1)
    padded_cnt = ((cnt + block_size - 1) // block_size) * block_size
    is_rep = (rank == 0) & mask_p


    smaller_rep = (bucket[None, :] < bucket[:, None]) & is_rep[None, :]
    excl = tl.sum(smaller_rep.to(tl.int32) * padded_cnt[None, :], axis=1)

    total = tl.sum(tl.where(is_rep, padded_cnt, 0), axis=0)
    tl.store(num_tokens_post_pad_ptr, total.to(tl.int32))


    offs_b = tl.arange(0, NB)
    block_start = offs_b * block_size
    in_range = (
        (block_start[:, None] >= excl[None, :])
        & (block_start[:, None] < (excl + padded_cnt)[None, :])
        & is_rep[None, :]
    )
    eid = tl.sum(in_range.to(tl.int32) * (bucket[None, :] - 1), axis=1)
    tl.store(expert_ids_ptr + offs_b, eid.to(tl.int32), mask=block_start < total)


    n_fill = (total + NP - 1) // NP
    for it in range(n_fill):
        f_offs = it * NP + offs_p
        tl.store(
            sorted_token_ids_ptr + f_offs,
            tl.full([NP], 0, tl.int32) + numel,
            mask=f_offs < total,
        )
    tl.debug_barrier()
    pos = excl + rank
    tl.store(sorted_token_ids_ptr + pos, offs_p.to(tl.int32), mask=mask_p)

    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()


def moe_align_small_numel(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:

    numel = topk_ids.numel()
    pdl_kwargs = {"USE_GDC": True, "launch_pdl": True} if is_arch_support_pdl() else {}
    _moe_align_small_numel_kernel[(1,)](
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        num_experts,
        block_size,
        numel,
        NP=triton.next_power_of_2(max(numel, 2)),
        NB=triton.next_power_of_2(max(expert_ids.numel(), 2)),
        num_warps=4,
        **pdl_kwargs,
    )
