from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

logger = logging.getLogger(__name__)


@cache_once
def _jit_fused_metadata_copy_module(
    forward_mode: int, has_real_page_table: bool, has_flashmla: bool
):

    args = make_cpp_args(forward_mode, has_real_page_table, has_flashmla)
    try:
        return load_jit(
            "fused_metadata_copy",
            *args,
            cuda_files=["elementwise/fused_metadata_copy.cuh"],
            cuda_wrappers=[
                (
                    "fused_metadata_copy",
                    f"FusedMetadataCopyKernel<{args}>::run",
                )
            ],
        )
    except Exception as e:
        logger.error(
            f"Failed to compile JIT fused metadata copy kernel "
            f"(forward_mode={forward_mode}, has_real_page_table={has_real_page_table}, "
            f"has_flashmla={has_flashmla}): {e}"
        )
        raise


@cache_once
def _jit_fused_metadata_copy_multi_module(
    has_real_page_table: bool, has_flashmla: bool
):

    args = make_cpp_args(has_real_page_table, has_flashmla)
    try:
        return load_jit(
            "fused_metadata_copy_multi",
            *args,
            cuda_files=["elementwise/fused_metadata_copy.cuh"],
            cuda_wrappers=[
                (
                    "fused_metadata_copy_multi",
                    f"FusedMetadataCopyMultiKernel<{args}>::run",
                )
            ],
        )
    except Exception as e:
        logger.error(
            f"Failed to compile JIT fused metadata copy multi kernel "
            f"(has_real_page_table={has_real_page_table}, has_flashmla={has_flashmla}): {e}"
        )
        raise


def fused_metadata_copy_cuda(
    cache_seqlens_src: torch.Tensor,
    cu_seqlens_k_src: torch.Tensor,
    page_indices_src: torch.Tensor,
    dsa_cache_seqlens_src: torch.Tensor,
    seqlens_expanded_src: Optional[torch.Tensor],
    dsa_cu_seqlens_k_src: torch.Tensor,
    real_page_table_src: Optional[torch.Tensor],
    flashmla_num_splits_src: Optional[torch.Tensor],
    flashmla_metadata_src: Optional[torch.Tensor],
    cache_seqlens_dst: torch.Tensor,
    cu_seqlens_k_dst: torch.Tensor,
    page_table_1_dst: torch.Tensor,
    dsa_cache_seqlens_dst: torch.Tensor,
    seqlens_expanded_dst: Optional[torch.Tensor],
    dsa_cu_seqlens_k_dst: torch.Tensor,
    real_page_table_dst: Optional[torch.Tensor],
    flashmla_num_splits_dst: Optional[torch.Tensor],
    flashmla_metadata_dst: Optional[torch.Tensor],
    forward_mode: int,
    bs: int,
    max_len: int,
    max_seqlen_k: int,
    seqlens_expanded_size: int,
) -> None:


    has_real_page_table = real_page_table_src is not None
    has_flashmla = flashmla_num_splits_src is not None


    module = _jit_fused_metadata_copy_module(
        forward_mode, has_real_page_table, has_flashmla
    )


    cache_seqlens_src = cache_seqlens_src.contiguous()
    cu_seqlens_k_src = cu_seqlens_k_src.contiguous()
    page_indices_src = page_indices_src.contiguous()
    dsa_cache_seqlens_src = dsa_cache_seqlens_src.contiguous()
    if seqlens_expanded_src is not None:
        seqlens_expanded_src = seqlens_expanded_src.contiguous()
    dsa_cu_seqlens_k_src = dsa_cu_seqlens_k_src.contiguous()


    module.fused_metadata_copy(
        cache_seqlens_src,
        cu_seqlens_k_src,
        page_indices_src,
        dsa_cache_seqlens_src,
        seqlens_expanded_src,
        dsa_cu_seqlens_k_src,
        real_page_table_src,
        flashmla_num_splits_src,
        flashmla_metadata_src,
        cache_seqlens_dst,
        cu_seqlens_k_dst,
        page_table_1_dst,
        dsa_cache_seqlens_dst,
        seqlens_expanded_dst,
        dsa_cu_seqlens_k_dst,
        real_page_table_dst,
        flashmla_num_splits_dst,
        flashmla_metadata_dst,
        bs,
        max_len,
        max_seqlen_k,
        seqlens_expanded_size,
    )


def fused_metadata_copy_multi_cuda(
    cache_seqlens_src: torch.Tensor,
    cu_seqlens_k_src: torch.Tensor,
    page_indices_src: torch.Tensor,
    dsa_cache_seqlens_src: torch.Tensor,
    dsa_cu_seqlens_k_src: torch.Tensor,
    real_page_table_src: Optional[torch.Tensor],
    flashmla_num_splits_src: Optional[torch.Tensor],
    flashmla_metadata_src: Optional[torch.Tensor],
    cache_seqlens_dst0: torch.Tensor,
    cu_seqlens_k_dst0: torch.Tensor,
    page_table_1_dst0: torch.Tensor,
    dsa_cache_seqlens_dst0: torch.Tensor,
    dsa_cu_seqlens_k_dst0: torch.Tensor,
    real_page_table_dst0: Optional[torch.Tensor],
    flashmla_num_splits_dst0: Optional[torch.Tensor],
    flashmla_metadata_dst0: Optional[torch.Tensor],
    cache_seqlens_dst1: torch.Tensor,
    cu_seqlens_k_dst1: torch.Tensor,
    page_table_1_dst1: torch.Tensor,
    dsa_cache_seqlens_dst1: torch.Tensor,
    dsa_cu_seqlens_k_dst1: torch.Tensor,
    real_page_table_dst1: Optional[torch.Tensor],
    flashmla_num_splits_dst1: Optional[torch.Tensor],
    flashmla_metadata_dst1: Optional[torch.Tensor],
    cache_seqlens_dst2: torch.Tensor,
    cu_seqlens_k_dst2: torch.Tensor,
    page_table_1_dst2: torch.Tensor,
    dsa_cache_seqlens_dst2: torch.Tensor,
    dsa_cu_seqlens_k_dst2: torch.Tensor,
    real_page_table_dst2: Optional[torch.Tensor],
    flashmla_num_splits_dst2: Optional[torch.Tensor],
    flashmla_metadata_dst2: Optional[torch.Tensor],
    bs: int,
    max_len: int,
    seqlens_expanded_size: int,
) -> None:


    has_real_page_table = real_page_table_src is not None
    has_flashmla = flashmla_num_splits_src is not None


    module = _jit_fused_metadata_copy_multi_module(has_real_page_table, has_flashmla)


    cache_seqlens_src = cache_seqlens_src.contiguous()
    cu_seqlens_k_src = cu_seqlens_k_src.contiguous()
    page_indices_src = page_indices_src.contiguous()
    dsa_cache_seqlens_src = dsa_cache_seqlens_src.contiguous()
    dsa_cu_seqlens_k_src = dsa_cu_seqlens_k_src.contiguous()


    module.fused_metadata_copy_multi(
        cache_seqlens_src,
        cu_seqlens_k_src,
        page_indices_src,
        dsa_cache_seqlens_src,
        dsa_cu_seqlens_k_src,
        real_page_table_src,
        flashmla_num_splits_src,
        flashmla_metadata_src,
        cache_seqlens_dst0,
        cu_seqlens_k_dst0,
        page_table_1_dst0,
        dsa_cache_seqlens_dst0,
        dsa_cu_seqlens_k_dst0,
        real_page_table_dst0,
        flashmla_num_splits_dst0,
        flashmla_metadata_dst0,
        cache_seqlens_dst1,
        cu_seqlens_k_dst1,
        page_table_1_dst1,
        dsa_cache_seqlens_dst1,
        dsa_cu_seqlens_k_dst1,
        real_page_table_dst1,
        flashmla_num_splits_dst1,
        flashmla_metadata_dst1,
        cache_seqlens_dst2,
        cu_seqlens_k_dst2,
        page_table_1_dst2,
        dsa_cache_seqlens_dst2,
        dsa_cu_seqlens_k_dst2,
        real_page_table_dst2,
        flashmla_num_splits_dst2,
        flashmla_metadata_dst2,
        bs,
        max_len,
        seqlens_expanded_size,
    )
