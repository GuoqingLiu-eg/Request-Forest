import triton
import triton.language as tl


@triton.jit
def compress_k_complete_kernel_new(
    key_cache_ptr,
    token_table_ptr,
    cu_new_k_token_nums_ptr,
    history_compress_k_token_nums_ptr,
    compressed_k_table_ptr,
    cu_total_compress_k_token_nums_ptr,
    full_compressed_k_ptr,
    batch_size,
    max_chunks_per_seq,
    token_table_cols,
    compressed_k_table_cols,
    head_num_k: tl.constexpr,
    head_dim: tl.constexpr,
    kernel_size: tl.constexpr,
    kernel_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    max_grid_chunks: tl.constexpr,
):

    batch_idx = tl.program_id(0)
    grid_chunk_idx = tl.program_id(1)
    head_idx = tl.program_id(2)


    chunk_stride = max_grid_chunks

    if batch_idx >= batch_size or head_idx >= head_num_k:
        return


    history_compress = tl.load(history_compress_k_token_nums_ptr + batch_idx)


    cu_new_k_start = tl.load(cu_new_k_token_nums_ptr + batch_idx)
    cu_new_k_end = tl.load(cu_new_k_token_nums_ptr + batch_idx + 1)
    new_k_count = cu_new_k_end - cu_new_k_start
    new_chunks_in_seq = tl.where(
        new_k_count >= kernel_size, (new_k_count - kernel_size) // kernel_stride + 1, 0
    )


    history_chunks_in_seq = history_compress
    total_chunks_in_seq = history_chunks_in_seq + new_chunks_in_seq

    output_start = tl.load(cu_total_compress_k_token_nums_ptr + batch_idx)


    chunk_in_seq = grid_chunk_idx

    while chunk_in_seq < total_chunks_in_seq:

        is_history_chunk = chunk_in_seq < history_chunks_in_seq

        if is_history_chunk:


            history_chunk_idx = chunk_in_seq

            global_full_idx = output_start + history_chunk_idx


            full_compressed_idx = tl.load(
                compressed_k_table_ptr
                + batch_idx * compressed_k_table_cols
                + history_chunk_idx
            ).to(tl.int32)

            head_offset = (
                full_compressed_idx * head_num_k * head_dim + head_idx * head_dim
            )
            x = tl.load(
                key_cache_ptr + head_offset + tl.arange(0, BLOCK_SIZE),
                mask=tl.arange(0, BLOCK_SIZE) < head_dim,
                other=0.0,
            )
            out_offset = global_full_idx * head_num_k * head_dim + head_idx * head_dim
            tl.store(
                full_compressed_k_ptr + out_offset + tl.arange(0, BLOCK_SIZE),
                x,
                mask=tl.arange(0, BLOCK_SIZE) < head_dim,
            )

        else:


            new_chunk_idx = chunk_in_seq - history_chunks_in_seq


            y = (new_chunk_idx + history_compress) * kernel_stride


            if y < token_table_cols:


                compressed_table_y = new_chunk_idx + history_compress

                if compressed_table_y < compressed_k_table_cols:

                    new_compressed_k_indices = tl.load(
                        compressed_k_table_ptr
                        + batch_idx * compressed_k_table_cols
                        + compressed_table_y
                    ).to(tl.int32)


                    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

                    for token_offset in range(kernel_size):

                        token_y = (
                            new_chunk_idx * kernel_stride + token_offset
                        ) + history_compress * kernel_stride


                        if token_y < token_table_cols:
                            token_k_indices = tl.load(
                                token_table_ptr + batch_idx * token_table_cols + token_y
                            ).to(tl.int32)
                        else:
                            token_k_indices = 0


                        key_base_offset = (
                            token_k_indices * head_num_k * head_dim
                            + head_idx * head_dim
                        )


                        x = tl.load(
                            key_cache_ptr + key_base_offset + tl.arange(0, BLOCK_SIZE),
                            mask=tl.arange(0, BLOCK_SIZE) < head_dim,
                            other=0.0,
                        ).to(tl.float32)

                        acc += x


                    acc = acc / kernel_size

                    head_offset = (
                        new_compressed_k_indices * head_num_k * head_dim
                        + head_idx * head_dim
                    )
                    tl.store(
                        key_cache_ptr + head_offset + tl.arange(0, BLOCK_SIZE),
                        acc,
                        mask=tl.arange(0, BLOCK_SIZE) < head_dim,
                    )

                    global_full_idx = output_start + history_compress + new_chunk_idx
                    out_offset = (
                        global_full_idx * head_num_k * head_dim + head_idx * head_dim
                    )
                    tl.store(
                        full_compressed_k_ptr + out_offset + tl.arange(0, BLOCK_SIZE),
                        acc,
                        mask=tl.arange(0, BLOCK_SIZE) < head_dim,
                    )


        chunk_in_seq += chunk_stride
