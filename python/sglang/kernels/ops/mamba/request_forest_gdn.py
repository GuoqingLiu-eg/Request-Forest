import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["requests", "source_width"])
def _stage_rf_gdn_graph_kernel(
    offsets, source_p2d, source_child, source_sibling, source_parent,
    query_offsets, packed_to_dense, dense_to_packed, child, sibling, parent,
    requests, source_width, SLOTS: tl.constexpr,
    CAP: tl.constexpr, ROWS: tl.constexpr,
    HAS_PARENT: tl.constexpr, BLOCK_ROWS: tl.constexpr,
    BLOCK_DENSE: tl.constexpr, BLOCK_OFFSETS: tl.constexpr,
):
    logical_rows = tl.load(offsets + requests)
    row = tl.arange(0, BLOCK_ROWS)
    source_index = tl.load(
        source_p2d + row, mask=(row < ROWS) & (row < logical_rows), other=0
    )
    mapped = source_index // source_width * CAP + source_index % source_width
    tl.store(packed_to_dense + row, tl.where(row < logical_rows, mapped, -1), row < ROWS)

    flat = tl.arange(0, BLOCK_DENSE)
    request, local = flat // CAP, flat % CAP
    start = tl.load(offsets + request, request < requests, other=0)
    end = tl.load(offsets + request + 1, request < requests, other=0)
    valid = (request < requests) & (local < end - start) & (local < source_width)
    source = request * source_width + local
    next_child = tl.load(source_child + source, valid, other=-1)
    next_sibling = tl.load(source_sibling + source, valid, other=-1)
    if HAS_PARENT:
        parent_row = tl.load(source_parent + source, valid, other=0)
    else:
        parent_row = tl.full((BLOCK_DENSE,), 0, tl.int32)
    tl.store(child + flat, next_child, flat < SLOTS * CAP)
    tl.store(sibling + flat, next_sibling, flat < SLOTS * CAP)
    tl.store(parent + flat, parent_row, flat < SLOTS * CAP)
    tl.store(dense_to_packed + flat, tl.where(valid, start + local, -1), flat < SLOTS * CAP)

    index = tl.arange(0, BLOCK_OFFSETS)
    end_offset = tl.load(offsets + tl.minimum(index, requests), index <= SLOTS, other=0)
    tl.store(query_offsets + index, end_offset, index <= SLOTS)


def stage_rf_gdn_graph_metadata(spec, metadata) -> None:

    offsets = spec.request_offsets
    requests = offsets.numel() - 1
    width = spec.retrieve_next_token.shape[1]
    capacity = metadata.gdn_graph_capacity
    slots = metadata.gdn_dense_to_packed.shape[0]
    rows = metadata.gdn_packed_to_dense.numel()
    source_parent = getattr(spec, "gdn_retrieve_parent_token", None)
    if offsets.device.type == "cuda":
        _stage_rf_gdn_graph_kernel[(1,)](
            offsets, spec.gdn_packed_to_dense, spec.retrieve_next_token,
            spec.retrieve_next_sibling, source_parent,
            metadata.query_start_loc, metadata.gdn_packed_to_dense,
            metadata.gdn_dense_to_packed, metadata.retrieve_next_token,
            metadata.retrieve_next_sibling, metadata.retrieve_parent_token,
            requests=requests, source_width=width, SLOTS=slots, CAP=capacity,
            ROWS=rows, HAS_PARENT=source_parent is not None,
            BLOCK_ROWS=triton.next_power_of_2(rows),
            BLOCK_DENSE=triton.next_power_of_2(slots * capacity),
            BLOCK_OFFSETS=triton.next_power_of_2(slots + 1), num_warps=4,
        )
        return


    metadata.query_start_loc.fill_(offsets[-1])
    metadata.query_start_loc[: requests + 1].copy_(offsets)
    metadata.gdn_packed_to_dense.fill_(-1)
    logical_rows = int(offsets[-1])
    live_p2d = spec.gdn_packed_to_dense[:logical_rows]
    metadata.gdn_packed_to_dense[:logical_rows].copy_(
        live_p2d // width * capacity + live_p2d % width
    )
    local = torch.arange(capacity, device=offsets.device)
    lengths = offsets[1:] - offsets[:-1]
    valid = local[None, :] < lengths[:, None]
    metadata.gdn_dense_to_packed.fill_(-1)
    metadata.gdn_dense_to_packed[:requests].copy_(
        torch.where(valid, offsets[:-1, None] + local[None, :], -1)
    )
    for source, destination, fill in (
        (spec.retrieve_next_token, metadata.retrieve_next_token, -1),
        (spec.retrieve_next_sibling, metadata.retrieve_next_sibling, -1),
        (source_parent, metadata.retrieve_parent_token, 0),
    ):
        destination.fill_(fill)
        if source is not None:
            destination[:requests, :width].copy_(
                torch.where(valid[:, :width], source, fill)
            )


@triton.jit(do_not_specialize=["INDEX_ROWS"])
def _gather_rf_gdn_rows_kernel(
    source, indices, destination,
    FEATURES: tl.constexpr, ROWS: tl.constexpr, INDEX_ROWS,
    SOURCE_ROWS_PER_GROUP: tl.constexpr, SOURCE_GROUP_STRIDE: tl.constexpr,
    SOURCE_ROW_STRIDE: tl.constexpr, SOURCE_FEATURE_STRIDE: tl.constexpr,
    MASK_ONLY: tl.constexpr, BLOCK: tl.constexpr,
):
    item = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, feature = item // FEATURES, item % FEATURES
    index = tl.load(indices + row, (row < ROWS) & (row < INDEX_ROWS), other=-1)
    if MASK_ONLY:
        tl.store(destination + item, 0, (item < ROWS * FEATURES) & (index < 0))
    else:
        value = tl.load(
            source
            + index // SOURCE_ROWS_PER_GROUP * SOURCE_GROUP_STRIDE
            + index % SOURCE_ROWS_PER_GROUP * SOURCE_ROW_STRIDE
            + feature * SOURCE_FEATURE_STRIDE,
            (item < ROWS * FEATURES) & (index >= 0), other=0,
        )
        tl.store(destination + item, value, item < ROWS * FEATURES)


def _gather_rf_gdn_rows(source, indices, out):
    flat_indices = indices.view(-1)
    features = source.shape[-1]
    rows = out.numel() // features
    if source.device.type == "cuda":
        if source.ndim == 3:
            group_rows = source.shape[1]
            group_stride, row_stride, feature_stride = source.stride()
        else:
            group_rows = 1
            group_stride, feature_stride = source.stride()
            row_stride = 0
        _gather_rf_gdn_rows_kernel[(triton.cdiv(out.numel(), 256),)](
            source, flat_indices, out, FEATURES=features, ROWS=rows,
            INDEX_ROWS=flat_indices.numel(), SOURCE_ROWS_PER_GROUP=group_rows,
            SOURCE_GROUP_STRIDE=group_stride, SOURCE_ROW_STRIDE=row_stride,
            SOURCE_FEATURE_STRIDE=feature_stride, MASK_ONLY=False, BLOCK=256,
        )
    else:
        flat_out = out.view(rows, features)
        count = flat_indices.numel()
        torch.index_select(
            source.reshape(-1, features), 0, flat_indices.clamp_min(0),
            out=flat_out[:count],
        )
        flat_out[:count].masked_fill_(flat_indices[:, None] < 0, 0)
        flat_out[count:].zero_()
    return out


def pack_rf_gdn_graph_rows(packed, dense_to_packed, out):

    return _gather_rf_gdn_rows(packed, dense_to_packed, out)


def unpack_rf_gdn_graph_rows(dense, packed_to_dense, out):

    return _gather_rf_gdn_rows(dense, packed_to_dense, out)


def zero_rf_gdn_graph_padding_(output, packed_to_dense, *, rows=None):

    if rows is None:
        rows = packed_to_dense.numel()
    features = output.numel() // rows
    if output.device.type == "cuda":
        _gather_rf_gdn_rows_kernel[(triton.cdiv(output.numel(), 256),)](
            output, packed_to_dense, output, FEATURES=features, ROWS=rows,
            INDEX_ROWS=packed_to_dense.numel(),
            SOURCE_ROWS_PER_GROUP=1, SOURCE_GROUP_STRIDE=features,
            SOURCE_ROW_STRIDE=features, SOURCE_FEATURE_STRIDE=1,
            MASK_ONLY=True, BLOCK=256,
        )
    else:
        count = packed_to_dense.numel()
        output.view(rows, features)[:count].masked_fill_(packed_to_dense[:, None] < 0, 0)
        output.view(rows, features)[count:].zero_()
    return output
