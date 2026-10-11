import torch
import triton
import triton.language as tl


@triton.jit
def _pack_topk_kernel(ids_ptr, w_ptr, out_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    ids = tl.load(ids_ptr + offs, mask=mask)
    w = tl.load(w_ptr + offs, mask=mask)
    wb = w.to(tl.bfloat16)
    wbits = wb.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    packed = (ids << 16) | wbits
    tl.store(out_ptr + offs, packed, mask=mask)


def fused_pack_topk(topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:

    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(torch.int32)
    topk_ids = topk_ids.contiguous()
    if topk_weights.dtype != torch.float32:
        topk_weights = topk_weights.to(torch.float32)
    topk_weights = topk_weights.contiguous()
    out = torch.empty_like(topk_ids, dtype=torch.int32)
    numel = out.numel()
    if numel == 0:
        return out
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _pack_topk_kernel[grid](topk_ids, topk_weights, out, numel, BLOCK=BLOCK)
    return out
