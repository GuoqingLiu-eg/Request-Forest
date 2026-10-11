import triton
import triton.language as tl


Q_MODE_NONE = 0
Q_MODE_CUMSUM = 1
Q_MODE_STRIDED = 2


@triton.jit
def update_trtllm_mha_graph_metadata_kernel(

    req_pool_indices_ptr,
    seq_lens_ptr,
    req_to_token_ptr,
    swa_mapping_ptr,
    out_cache_loc_ptr,
    qlens_ptr,

    cache_seqlens_ptr,
    cu_seqlens_k_ptr,
    cu_seqlens_q_ptr,
    page_table_ptr,
    swa_page_table_ptr,
    swa_out_cache_loc_ptr,

    bs,
    seqlen_offset,
    max_seq_pages,
    q_stride,
    num_out_tokens,
    swa_out_len,
    req_to_token_stride,
    page_table_stride,
    swa_page_table_stride,

    PAGE_SIZE: tl.constexpr,
    HAS_SWA: tl.constexpr,
    HAS_SWA_OUT: tl.constexpr,
    Q_MODE: tl.constexpr,
    PAGE_BLOCK: tl.constexpr,
    BS_BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    if pid < bs:

        req_pool_index = tl.load(req_pool_indices_ptr + pid).to(tl.int64)
        seqlen = (tl.load(seq_lens_ptr + pid) + seqlen_offset).to(tl.int32)
        tl.store(cache_seqlens_ptr + pid, seqlen)

        row_in = req_to_token_ptr + req_pool_index * req_to_token_stride
        row_out = page_table_ptr + pid.to(tl.int64) * page_table_stride
        if HAS_SWA:
            swa_row_out = swa_page_table_ptr + pid.to(tl.int64) * swa_page_table_stride


        num_live_pages = tl.minimum(tl.cdiv(seqlen, PAGE_SIZE), max_seq_pages)
        for i in range(tl.cdiv(num_live_pages, PAGE_BLOCK)):
            page_idx = i * PAGE_BLOCK + tl.arange(0, PAGE_BLOCK)
            mask = page_idx < num_live_pages
            token = tl.load(
                row_in + page_idx.to(tl.int64) * PAGE_SIZE, mask=mask, other=0
            )
            tl.store(row_out + page_idx, token // PAGE_SIZE, mask=mask)
            if HAS_SWA:
                token64 = token.to(tl.int64)


                swa_token = tl.load(
                    swa_mapping_ptr + token64, mask=mask & (token64 >= 0), other=-1
                )
                swa_page = tl.where(swa_token < 0, -1, swa_token // PAGE_SIZE)
                tl.store(swa_row_out + page_idx, swa_page.to(tl.int32), mask=mask)
    elif pid == bs:

        offs = tl.arange(0, BS_BLOCK)
        mask = offs < bs
        seqlens = (tl.load(seq_lens_ptr + offs, mask=mask, other=0)).to(tl.int32)
        seqlens = tl.where(mask, seqlens + seqlen_offset, 0)
        tl.store(cu_seqlens_k_ptr + 1 + offs, tl.cumsum(seqlens, axis=0), mask=mask)
        if Q_MODE == 1:
            qlens = tl.load(qlens_ptr + offs, mask=mask, other=0).to(tl.int32)
            qlens = tl.where(mask, qlens, 0)
            tl.store(cu_seqlens_q_ptr + 1 + offs, tl.cumsum(qlens, axis=0), mask=mask)
        if Q_MODE == 2:
            tl.store(
                cu_seqlens_q_ptr + 1 + offs,
                ((offs + 1) * q_stride).to(tl.int32),
                mask=mask,
            )
    else:

        if HAS_SWA_OUT:
            out_idx = (pid - bs - 1) * PAGE_BLOCK + tl.arange(0, PAGE_BLOCK)
            in_range = out_idx < swa_out_len
            is_real = in_range & (out_idx < num_out_tokens)
            loc = tl.load(out_cache_loc_ptr + out_idx, mask=is_real, other=0)
            translated = tl.load(
                swa_mapping_ptr + loc, mask=is_real & (loc >= 0), other=0
            )
            translated = tl.where(is_real & (loc < 0), -1, translated)
            tl.store(swa_out_cache_loc_ptr + out_idx, translated, mask=in_range)


def update_trtllm_mha_graph_metadata(
    *,
    req_pool_indices,
    seq_lens,
    req_to_token,
    cache_seqlens,
    cu_seqlens_k,
    page_table,
    bs: int,
    seqlen_offset: int,
    max_seq_pages: int,
    page_size: int,
    swa_mapping=None,
    swa_page_table=None,
    out_cache_loc=None,
    swa_out_cache_loc=None,
    cu_seqlens_q=None,
    qlens=None,
    q_stride: int = 0,
    q_mode: int = Q_MODE_NONE,
):

    if bs == 0:
        return


    PAGE_BLOCK = 512
    has_swa = swa_page_table is not None
    has_swa_out = swa_out_cache_loc is not None

    swa_out_len = swa_out_cache_loc.shape[0] if has_swa_out else 0
    if has_swa_out and out_cache_loc is not None:
        num_out_tokens = min(swa_out_len, out_cache_loc.shape[0])
    else:
        num_out_tokens = 0
    if num_out_tokens == 0:

        out_cache_loc = swa_out_cache_loc

    grid_extra = triton.cdiv(swa_out_len, PAGE_BLOCK) if has_swa_out else 0
    grid = (bs + 1 + grid_extra,)

    update_trtllm_mha_graph_metadata_kernel[grid](
        req_pool_indices,
        seq_lens,
        req_to_token,
        swa_mapping,
        out_cache_loc,
        qlens,
        cache_seqlens,
        cu_seqlens_k,
        cu_seqlens_q,
        page_table,
        swa_page_table,
        swa_out_cache_loc,
        bs,
        seqlen_offset,
        max_seq_pages,
        q_stride,
        num_out_tokens,
        swa_out_len,
        req_to_token.stride(0),
        page_table.stride(0),
        swa_page_table.stride(0) if has_swa else 0,
        PAGE_SIZE=page_size,
        HAS_SWA=has_swa,
        HAS_SWA_OUT=has_swa_out,
        Q_MODE=q_mode,
        PAGE_BLOCK=PAGE_BLOCK,
        BS_BLOCK=triton.next_power_of_2(bs),
    )
