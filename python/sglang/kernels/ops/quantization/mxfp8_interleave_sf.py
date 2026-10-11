import torch
import triton
import triton.language as tl


@triton.jit
def _store_sf_interleaved_kernel(
    sf_in_ptr,
    sf_out_ptr,
    loc_ptr,
    num_tokens,  # type: ignore
    nheads: tl.constexpr,
    page_size: tl.constexpr,
    BLOCK_T: tl.constexpr,
):

    pid = tl.program_id(0)
    tok_start = pid * BLOCK_T
    tok_offsets = tok_start + tl.arange(0, BLOCK_T)
    mask = tok_offsets < num_tokens


    slots = tl.load(loc_ptr + tok_offsets, mask=mask, other=0)
    page_offsets = slots % page_size
    page_idxs = slots // page_size


    interleaved_pos = (page_offsets % 32) * 4 + (page_offsets // 32)


    page_stride: tl.constexpr = nheads * 128

    for h in tl.static_range(nheads):

        in_offsets = tok_offsets * nheads + h
        vals = tl.load(sf_in_ptr + in_offsets, mask=mask, other=0)


        out_offsets = page_idxs * page_stride + h * 128 + interleaved_pos
        tl.store(sf_out_ptr + out_offsets, vals, mask=mask)


def store_sf_interleaved(
    sf_in: torch.Tensor,
    sf_out: torch.Tensor,
    loc: torch.Tensor,
    page_size: int = 128,
):

    assert (
        page_size == 128
    ), f"Interleaved SF layout requires page_size=128, got {page_size}"
    num_tokens, nheads, sf_dim = sf_in.shape
    assert sf_dim == 4, f"Expected sf_dim=4 (hdim=128, sf_vec_size=32), got {sf_dim}"


    sf_in_u8 = sf_in.view(torch.uint8) if sf_in.dtype == torch.float8_e8m0fnu else sf_in
    sf_out_u8 = (
        sf_out.view(torch.uint8) if sf_out.dtype == torch.float8_e8m0fnu else sf_out
    )

    sf_in_u32 = (
        sf_in_u8.reshape(num_tokens, nheads, 4)
        .contiguous()
        .view(torch.int32)
        .reshape(num_tokens, nheads)
    )
    sf_out_u32 = sf_out_u8.reshape(-1, 4).view(torch.int32).reshape(-1)

    BLOCK_T = 128
    grid = ((num_tokens + BLOCK_T - 1) // BLOCK_T,)

    _store_sf_interleaved_kernel[grid](
        sf_in_u32,
        sf_out_u32,
        loc,
        num_tokens,
        nheads=nheads,
        page_size=page_size,
        BLOCK_T=BLOCK_T,
    )
