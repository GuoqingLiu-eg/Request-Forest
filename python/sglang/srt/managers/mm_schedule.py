from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch

from sglang.srt.managers.schedule_batch import MultimodalDataItem
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult, MultiModalStaticCache
from sglang.srt.multimodal.evs import EVSEmbeddingResult
from sglang.srt.runtime_context import get_parallel, get_schedule
from sglang.srt.utils import is_hip, is_npu, is_xpu
from sglang.srt.utils.async_probe import maybe_assert_sum
from sglang.utils import logger

_is_hip = is_hip()
_is_npu = is_npu()
_is_xpu = is_xpu()

embedding_cache: Optional[MultiModalStaticCache] = None


def init_mm_embedding_cache(max_size: int = 0):
    global embedding_cache
    embedding_cache = MultiModalStaticCache(max_size)


def get_embedding_chunk(
    embedding: torch.Tensor,
    extend_prefix_len: int,
    extend_seq_len: int,
    items_offset: List[Tuple[int, int]],
) -> Tuple[torch.Tensor, int, int]:

    start_index, end_index = 0, 0
    extend_start_index = extend_prefix_len
    extend_end_index = extend_prefix_len + extend_seq_len - 1

    for start, end in items_offset:
        if extend_start_index >= start and extend_start_index <= end:
            start_index += extend_start_index - start
        elif extend_start_index > end:
            start_index += end - start + 1

        if extend_end_index >= start and extend_end_index <= end:
            end_index += extend_end_index - start + 1
        elif extend_end_index > end:
            end_index += end - start + 1

    embedding = embedding.reshape(-1, embedding.shape[-1])
    embedding_chunk = embedding[start_index:end_index]
    return embedding_chunk, start_index, end_index


def _get_precomputed_embedding(
    items: List[MultimodalDataItem],
    items_size: List[int],
    prefix_length: List[int],
    extend_length: List[int],
    items_offset_list: List[List[Tuple[int, int]]],
) -> Optional[torch.Tensor]:

    precomputed_embeddings = []
    max_iterations = min(len(items_size) - 1, len(prefix_length))

    for i in range(max_iterations):
        if items_size[i] == items_size[i + 1]:
            continue

        items_per_req = items[items_size[i] : items_size[i + 1]]
        extend_len = extend_length[i] if i < len(extend_length) else 0
        items_offset = items_offset_list[i]

        if any(item.precomputed_embeddings is None for item in items_per_req):
            chunk = None
        else:
            req_embeddings = torch.concat(
                [item.precomputed_embeddings for item in items_per_req]
            )
            chunk, _, _ = get_embedding_chunk(
                embedding=req_embeddings,
                extend_prefix_len=prefix_length[i],
                extend_seq_len=extend_len,
                items_offset=items_offset,
            )

        if chunk is None and len(items_per_req) > 1:
            return None
        precomputed_embeddings.append(chunk)

    if any(feature is not None for feature in precomputed_embeddings):
        if not all(feature is not None for feature in precomputed_embeddings):
            raise NotImplementedError(
                "MM inputs where only some items are precomputed."
            )


        target_device = next(
            (t.device for t in precomputed_embeddings if t.is_cuda),
            precomputed_embeddings[0].device,
        )
        precomputed_embeddings = [
            t if t.device == target_device else t.to(target_device, non_blocking=True)
            for t in precomputed_embeddings
        ]
        result = torch.concat(precomputed_embeddings)

        result = result.reshape(-1, result.shape[-1])
        return result
    return None


DataEmbeddingFunc = Callable[
    [List[MultimodalDataItem]],
    torch.Tensor | List[torch.Tensor] | EVSEmbeddingResult,
]


def _flatten_embedding_result(
    embedding: torch.Tensor | List[torch.Tensor],
) -> torch.Tensor:

    if isinstance(embedding, list):
        if not embedding:
            raise ValueError(
                "DataEmbeddingFunc returned an empty per-item list; expected "
                "one entry per input item"
            )
        flat = [e.reshape(-1, e.shape[-1]) for e in embedding]
        return flat[0] if len(flat) == 1 else torch.cat(flat, dim=0)
    return embedding


def _can_skip_pre_embed_feature_move(data_embedding_func: DataEmbeddingFunc) -> bool:

    owner = getattr(data_embedding_func, "__self__", None)
    if owner is None:
        return False
    if getattr(data_embedding_func, "__name__", None) not in (
        "get_image_feature",
        "get_video_feature",
    ):
        return False
    return owner.__class__.__name__ in {
        "Qwen3VLForConditionalGeneration",
        "Qwen3VLMoeForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        "KimiK25ForConditionalGeneration",
        "KimiK3ForConditionalGeneration",
    }


def _move_items_to_device(
    items: List[MultimodalDataItem], device: torch.device
) -> None:

    for item in items:
        if isinstance(item.feature, torch.Tensor) and item.feature.device != device:
            item.feature = item.feature.to(device, non_blocking=True)


def _acknowledge_deferred_cuda_ipc_cache_hits(
    items: List[MultimodalDataItem],
) -> None:

    parallel = get_parallel()
    if parallel.attn_tp_rank != 0:
        return


    consumer_count = max(parallel.tp_size, 1)
    for item in items:
        item.acknowledge_deferred_cuda_ipc_feature(consumer_count)


def _get_chunked_embedding_full(
    data_embedding_func: DataEmbeddingFunc,
    embedding_items_per_req: List[MultimodalDataItem],
    items_offset: List[Tuple[int, int]],
    extend_prefix_len: int,
    extend_seq_len: int,
    input_ids: torch.Tensor,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:

    item_hashes = [item.hash for item in embedding_items_per_req]
    embedding_items_hash = MultiModalStaticCache.combine_hashes(item_hashes)
    embedding_per_req = embedding_cache.get(item_hashes)

    if embedding_per_req is None:
        if not _can_skip_pre_embed_feature_move(data_embedding_func):
            _move_items_to_device(embedding_items_per_req, device)
        embedding = data_embedding_func(embedding_items_per_req)
        if isinstance(embedding, list):


            embedding = _flatten_embedding_result(embedding)
        embedding_per_req = (
            EmbeddingResult(embedding=embedding)
            if isinstance(embedding, torch.Tensor)
            else embedding
        )
        embedding_cache.set(embedding_items_hash, embedding_per_req)
    else:
        _acknowledge_deferred_cuda_ipc_cache_hits(embedding_items_per_req)

    if isinstance(embedding_per_req, EVSEmbeddingResult):
        item = embedding_items_per_req[0]
        input_ids, items_offset = (
            embedding_per_req.redistribute_pruned_frames_placeholders(
                input_ids,
                items_offset,
                item=item,
                extend_prefix_len=extend_prefix_len,
                extend_seq_len=extend_seq_len,
            )
        )

    embedding_per_req_chunk, _, _ = get_embedding_chunk(
        embedding=embedding_per_req.embedding,
        extend_prefix_len=extend_prefix_len,
        extend_seq_len=extend_seq_len,
        items_offset=items_offset,
    )
    return embedding_per_req_chunk, input_ids


@dataclass
class PerImageRequestInfo:


    req_idx: int
    items: List[MultimodalDataItem]
    items_offset: List[Tuple[int, int]]
    extend_prefix_len: int
    extend_seq_len: int
    overlapping: List[Tuple[int, MultimodalDataItem, int, int]] = field(
        default_factory=list
    )


def _batch_encode_per_image_misses(
    data_embedding_func: DataEmbeddingFunc,
    per_image_requests: List[PerImageRequestInfo],
    device: torch.device,
) -> Dict[int, torch.Tensor]:

    unique_misses: Dict[int, Tuple[MultimodalDataItem, int]] = {}
    hash_to_embedding: Dict[int, torch.Tensor] = {}


    for req_info in per_image_requests:
        chunk_start = req_info.extend_prefix_len
        chunk_end = chunk_start + req_info.extend_seq_len
        overlapping = []
        if req_info.extend_seq_len > 0:
            for idx, (item, (start, end)) in enumerate(
                zip(req_info.items, req_info.items_offset)
            ):
                if end >= chunk_start and start < chunk_end:
                    overlapping.append((idx, item, start, end))
        req_info.overlapping = overlapping

        for _idx, item, start, end in overlapping:
            if item.hash in hash_to_embedding:
                continue
            cached = embedding_cache.get_single(item.hash)
            if cached is not None:
                hash_to_embedding[item.hash] = cached.embedding
            elif item.hash not in unique_misses:
                token_count = end - start + 1
                unique_misses[item.hash] = (item, token_count)


    if unique_misses:
        ordered_hashes = list(unique_misses.keys())
        miss_items = [unique_misses[h][0] for h in ordered_hashes]
        token_counts = [unique_misses[h][1] for h in ordered_hashes]

        if not _can_skip_pre_embed_feature_move(data_embedding_func):
            _move_items_to_device(miss_items, device)
        all_miss_embedding = data_embedding_func(miss_items)

        if isinstance(all_miss_embedding, list):


            assert len(all_miss_embedding) == len(miss_items), (
                f"per-item embedding count {len(all_miss_embedding)} != "
                f"cache-miss item count {len(miss_items)}"
            )
            split_embeddings = [
                emb.reshape(-1, emb.shape[-1]) for emb in all_miss_embedding
            ]
        else:
            all_miss_embedding = all_miss_embedding.reshape(
                -1, all_miss_embedding.shape[-1]
            )
            split_embeddings = torch.split(all_miss_embedding, token_counts, dim=0)
        for h, emb in zip(ordered_hashes, split_embeddings):
            embedding_cache.set(h, EmbeddingResult(embedding=emb))

            hash_to_embedding[h] = emb

    return hash_to_embedding


def _get_chunked_embedding_by_item(
    data_embedding_func: DataEmbeddingFunc,
    embedding_items_per_req: List[MultimodalDataItem],
    items_offset: List[Tuple[int, int]],
    extend_prefix_len: int,
    extend_seq_len: int,
    device: torch.device,
) -> Optional[torch.Tensor]:

    chunk_start = extend_prefix_len
    chunk_end = extend_prefix_len + extend_seq_len

    if extend_seq_len <= 0:
        return None

    overlapping = []
    for idx, (item, (start, end)) in enumerate(
        zip(embedding_items_per_req, items_offset)
    ):
        if end >= chunk_start and start < chunk_end:
            overlapping.append((idx, item, start, end))

    if not overlapping:
        return None

    cached_embeddings = {}
    miss_items = []
    for idx, item, start, end in overlapping:
        cached = embedding_cache.get_single(item.hash)
        if cached is not None:
            cached_embeddings[idx] = cached.embedding
            _acknowledge_deferred_cuda_ipc_cache_hits([item])
        else:
            miss_items.append((idx, item, start, end))

    if miss_items:
        miss_item_list = [item for _, item, _, _ in miss_items]
        if not _can_skip_pre_embed_feature_move(data_embedding_func):
            _move_items_to_device(miss_item_list, device)
        all_miss_embedding = data_embedding_func(miss_item_list)

        if isinstance(all_miss_embedding, list):


            assert len(all_miss_embedding) == len(miss_items), (
                f"per-item embedding count {len(all_miss_embedding)} != "
                f"cache-miss item count {len(miss_items)}"
            )
            split_embeddings = [
                emb.reshape(-1, emb.shape[-1]) for emb in all_miss_embedding
            ]
        else:
            all_miss_embedding = all_miss_embedding.reshape(
                -1, all_miss_embedding.shape[-1]
            )

            token_counts = [end - start + 1 for _, _, start, end in miss_items]
            split_embeddings = torch.split(all_miss_embedding, token_counts, dim=0)

        for (idx, item, _, _), emb in zip(miss_items, split_embeddings):
            cached_embeddings[idx] = emb
            embedding_cache.set(item.hash, EmbeddingResult(embedding=emb))

    chunk_slices = []
    for idx, _, start, end in overlapping:
        emb = cached_embeddings[idx]
        overlap_start = max(start, chunk_start)
        overlap_end = min(end, chunk_end - 1)
        local_start = overlap_start - start
        local_end = overlap_end - start + 1
        chunk_slices.append(emb[local_start:local_end])

    return torch.cat(chunk_slices, dim=0)


def _assemble_per_image_chunk(
    overlapping: List[Tuple[int, MultimodalDataItem, int, int]],
    hash_to_embedding: Dict[int, torch.Tensor],
    extend_prefix_len: int,
    extend_seq_len: int,
) -> Optional[torch.Tensor]:

    if not overlapping:
        return None

    chunk_start = extend_prefix_len
    chunk_end = extend_prefix_len + extend_seq_len

    chunk_slices = []
    for _idx, item, start, end in overlapping:
        emb = hash_to_embedding[item.hash]
        overlap_start = max(start, chunk_start)
        overlap_end = min(end, chunk_end - 1)
        local_start = overlap_start - start
        local_end = overlap_end - start + 1
        chunk_slices.append(emb[local_start:local_end])

    return torch.cat(chunk_slices, dim=0)


def _get_chunked_prefill_embedding(
    data_embedding_func: DataEmbeddingFunc,
    embedding_items: List[MultimodalDataItem],
    items_size: List[int],
    prefix_length: List[int],
    extend_length: List[int],
    items_offset_list: List[List[Tuple[int, int]]],
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor]:

    device = input_ids.device

    max_iterations = min(len(items_size) - 1, len(prefix_length))


    per_image_requests = []
    full_path_requests = []
    all_chunks: List[Tuple[int, torch.Tensor]] = []

    for i in range(max_iterations):
        if items_size[i] == items_size[i + 1]:
            continue
        embedding_items_per_req = embedding_items[items_size[i] : items_size[i + 1]]
        items_offset = items_offset_list[i]
        assert items_offset is not None, items_offset

        extend_prefix_len = prefix_length[i]
        extend_seq_len = extend_length[i] if i < len(extend_length) else 0
        if extend_seq_len <= 0:
            continue


        if all(offset_end < prefix_length[i] for _, offset_end in items_offset):
            continue

        req_info = PerImageRequestInfo(
            req_idx=i,
            items=embedding_items_per_req,
            items_offset=items_offset,
            extend_prefix_len=extend_prefix_len,
            extend_seq_len=extend_seq_len,
        )

        is_per_image = all(len(item.offsets) == 1 for item in embedding_items_per_req)
        if is_per_image:
            if _is_hip or _is_npu or _is_xpu:


                chunk = _get_chunked_embedding_by_item(
                    data_embedding_func,
                    embedding_items_per_req,
                    items_offset,
                    extend_prefix_len,
                    extend_seq_len,
                    device,
                )
                if chunk is not None:
                    all_chunks.append((i, chunk))
            else:
                per_image_requests.append(req_info)
        else:
            full_path_requests.append(req_info)


    hash_to_embedding: Dict[int, torch.Tensor] = {}
    if per_image_requests:
        hash_to_embedding = _batch_encode_per_image_misses(
            data_embedding_func, per_image_requests, device
        )


    for req_info in per_image_requests:
        chunk = _assemble_per_image_chunk(
            req_info.overlapping,
            hash_to_embedding,
            req_info.extend_prefix_len,
            req_info.extend_seq_len,
        )
        if chunk is not None:
            all_chunks.append((req_info.req_idx, chunk))

    for req_info in full_path_requests:
        chunk_embedding, input_ids = _get_chunked_embedding_full(
            data_embedding_func,
            req_info.items,
            req_info.items_offset,
            req_info.extend_prefix_len,
            req_info.extend_seq_len,
            input_ids,
            device,
        )
        if chunk_embedding is not None:
            all_chunks.append((req_info.req_idx, chunk_embedding))


    all_chunks.sort(key=lambda x: x[0])
    embedding_list = [chunk for _, chunk in all_chunks]

    if len(embedding_list) == 0:
        return None, input_ids
    return torch.concat(embedding_list, dim=0), input_ids


def _get_multimodal_mask(
    input_ids: torch.Tensor, placeholder_tensor: torch.Tensor
) -> torch.Tensor:
    return torch.isin(input_ids, placeholder_tensor).unsqueeze(-1)


def _count_mm_tokens_in_extend(
    prefix_length: List[int],
    extend_length: List[int],
    items_offset_list: List[List[Tuple[int, int]]],
) -> int:

    num_mm_tokens = 0
    for i, (extend_start, items_offset) in enumerate(
        zip(prefix_length, items_offset_list)
    ):
        extend_end = extend_start + (extend_length[i] if i < len(extend_length) else 0)
        for item_start, item_end in items_offset:
            overlap_start = max(item_start, extend_start)
            overlap_end = min(item_end + 1, extend_end)
            num_mm_tokens += max(overlap_end - overlap_start, 0)

    return num_mm_tokens


def _adjust_embedding_length(
    embedding: torch.Tensor,
    num_mm_tokens_in_input_ids: int,
    logger,
) -> torch.Tensor:
    num_mm_tokens_in_embedding = embedding.shape[0]
    if num_mm_tokens_in_input_ids != num_mm_tokens_in_embedding:
        logger.warning(
            f"Number of tokens in multimodal embedding does not match those in the input text. "
            f"Got {num_mm_tokens_in_input_ids} tokens in the text but {num_mm_tokens_in_embedding} "
            f"tokens from multimodal embeddings."
        )
        if num_mm_tokens_in_input_ids < num_mm_tokens_in_embedding:
            chunked_prefill_size = get_schedule().chunked_prefill_size
            if chunked_prefill_size != -1:
                logger.warning(
                    "You may want to avoid this issue by raising `chunked_prefill_size`, or disabling chunked prefill"
                )

            if embedding.dim() == 2:
                embedding = embedding[-num_mm_tokens_in_input_ids:, :]
            else:
                num_multimodal = num_mm_tokens_in_input_ids // embedding.shape[0]
                embedding = embedding[-num_multimodal:, :]
        else:
            raise RuntimeError(
                f"Insufficient multimodal embedding length: {num_mm_tokens_in_input_ids=} vs {num_mm_tokens_in_embedding=}. This is an internal error"
            )
    return embedding


def get_embedding_and_mask(
    data_embedding_func: DataEmbeddingFunc,
    embedding_items: List[MultimodalDataItem],
    placeholder_tensor: torch.Tensor,
    input_ids: torch.Tensor,
    items_size: List[int],
    prefix_length: List[int],
    extend_length: List[int],
    items_offset_list: List[List[Tuple[int, int]]],
) -> Tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:

    original_input_ids = input_ids
    num_mm_tokens_in_input_ids = _count_mm_tokens_in_extend(
        prefix_length,
        extend_length,
        items_offset_list,
    )


    embedding = _get_precomputed_embedding(
        embedding_items, items_size, prefix_length, extend_length, items_offset_list
    )
    if embedding is None:
        embedding, input_ids = _get_chunked_prefill_embedding(
            data_embedding_func,
            embedding_items,
            items_size,
            prefix_length,
            extend_length,
            items_offset_list,
            input_ids,
        )
        if embedding is None:
            return None, None, input_ids

    if _is_npu:
        torch.npu.current_stream().synchronize()
    special_multimodal_mask = _get_multimodal_mask(input_ids, placeholder_tensor)

    if input_ids is not original_input_ids:

        num_mm_tokens_in_input_ids = special_multimodal_mask.sum().item()
    else:
        maybe_assert_sum(
            special_multimodal_mask,
            num_mm_tokens_in_input_ids,
            "MM placeholder count derived from offsets does not match input_ids",
        )
    embedding = _adjust_embedding_length(embedding, num_mm_tokens_in_input_ids, logger)
    return embedding, special_multimodal_mask, input_ids
