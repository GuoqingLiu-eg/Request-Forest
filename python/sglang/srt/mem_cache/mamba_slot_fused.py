from __future__ import annotations

from typing import List, NamedTuple

import torch
import triton
import triton.language as tl

_BLOCK = 1024


class ConvSlotDescriptor(NamedTuple):
    ptr: torch.Tensor
    feat: torch.Tensor
    layer_stride: torch.Tensor
    slot_stride: torch.Tensor
    num_layers: int
    max_feat_blocks: int


@triton.jit
def _fused_slot_clear_kernel(
    ptr_arr,
    feat_arr,
    layer_stride_arr,
    slot_stride_arr,
    index_arr,
    MAX_FEAT_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    iid = tl.program_id(0)
    tid = tl.program_id(1)
    lid = tl.program_id(2)
    base_addr = tl.load(ptr_arr + tid)
    feat = tl.load(feat_arr + tid)
    slot = tl.load(index_arr + iid)
    base = base_addr.to(tl.pointer_type(tl.bfloat16))
    row = (
        base
        + lid * tl.load(layer_stride_arr + tid)
        + slot * tl.load(slot_stride_arr + tid)
    )
    zeros = tl.zeros((BLOCK,), dtype=tl.bfloat16)
    for fb in tl.static_range(MAX_FEAT_BLOCKS):
        cols = fb * BLOCK + tl.arange(0, BLOCK)
        tl.store(row + cols, zeros, mask=cols < feat)


@triton.jit
def _fused_slot_copy_kernel(
    ptr_arr,
    feat_arr,
    layer_stride_arr,
    slot_stride_arr,
    src_arr,
    dst_arr,
    MAX_FEAT_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    iid = tl.program_id(0)
    tid = tl.program_id(1)
    lid = tl.program_id(2)
    base_addr = tl.load(ptr_arr + tid)
    feat = tl.load(feat_arr + tid)
    layer_off = lid * tl.load(layer_stride_arr + tid)
    slot_stride = tl.load(slot_stride_arr + tid)
    base = base_addr.to(tl.pointer_type(tl.bfloat16))
    src_row = base + layer_off + tl.load(src_arr + iid) * slot_stride
    dst_row = base + layer_off + tl.load(dst_arr + iid) * slot_stride
    for fb in tl.static_range(MAX_FEAT_BLOCKS):
        cols = fb * BLOCK + tl.arange(0, BLOCK)
        mask = cols < feat
        tl.store(dst_row + cols, tl.load(src_row + cols, mask=mask), mask=mask)


def build_conv_slot_descriptor(tensors: List[torch.Tensor]) -> ConvSlotDescriptor:

    t0 = tensors[0]
    num_layers = t0.shape[0]
    device = t0.device
    ptr, feat, layer_stride, slot_stride = [], [], [], []
    max_feat = 0
    for t in tensors:
        assert t.dtype == torch.bfloat16, "fused slot ops assume bf16 conv state"
        assert t.shape[0] == num_layers, "conv tensors must share num_layers"
        assert t.device == device
        assert t[0, 0].is_contiguous(), "per-slot feature block must be contiguous"
        ptr.append(t.data_ptr())
        feat.append(t[0, 0].numel())
        layer_stride.append(t.stride(0))
        slot_stride.append(t.stride(1))
        max_feat = max(max_feat, t[0, 0].numel())
    to_i64 = lambda xs: torch.tensor(xs, dtype=torch.int64, device=device)
    return ConvSlotDescriptor(
        ptr=to_i64(ptr),
        feat=to_i64(feat),
        layer_stride=to_i64(layer_stride),
        slot_stride=to_i64(slot_stride),
        num_layers=num_layers,
        max_feat_blocks=triton.cdiv(max_feat, _BLOCK),
    )


def fused_clear_conv_slots(desc: ConvSlotDescriptor, indices: torch.Tensor):

    if desc.ptr.numel() == 0 or indices.numel() == 0:
        return
    index_arr = indices.to(torch.int64)

    grid = (index_arr.numel(), desc.ptr.numel(), desc.num_layers)
    _fused_slot_clear_kernel[grid](
        desc.ptr,
        desc.feat,
        desc.layer_stride,
        desc.slot_stride,
        index_arr,
        MAX_FEAT_BLOCKS=desc.max_feat_blocks,
        BLOCK=_BLOCK,
    )


def fused_copy_conv_slots(
    desc: ConvSlotDescriptor, src_indices: torch.Tensor, dst_indices: torch.Tensor
):

    if desc.ptr.numel() == 0 or src_indices.numel() == 0:
        return
    src_arr = src_indices.to(torch.int64)
    dst_arr = dst_indices.to(torch.int64)
    grid = (src_arr.numel(), desc.ptr.numel(), desc.num_layers)
    _fused_slot_copy_kernel[grid](
        desc.ptr,
        desc.feat,
        desc.layer_stride,
        desc.slot_stride,
        src_arr,
        dst_arr,
        MAX_FEAT_BLOCKS=desc.max_feat_blocks,
        BLOCK=_BLOCK,
    )
