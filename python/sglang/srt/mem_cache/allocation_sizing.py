from __future__ import annotations

from sglang.srt.runtime_context import (
    get_parallel,
    get_schedule,
    get_spec,
    max_speculative_num_draft_tokens,
)


def get_alloc_page_size() -> int:


    return get_schedule().page_size * get_parallel().attn_dcp_size


def get_alloc_len_per_decode() -> int:

    spec = get_spec()
    if spec.speculative_algorithm is None:
        return 1


    spec_steps = spec.speculative_num_steps or 1
    spec_topk = spec.speculative_eagle_topk or 1
    spec_tokens = max_speculative_num_draft_tokens()
    page_size = get_alloc_page_size()

    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    spec_algo = SpeculativeAlgorithm.from_string(spec.speculative_algorithm)
    if page_size == 1 or spec_topk == 1 or not spec_algo.has_draft_kv():
        return max(spec_steps * spec_topk, spec_tokens)
    else:


        num_new_pages_per_topk = (
            (page_size - 1) + spec_steps + page_size - 1
        ) // page_size
        return max(num_new_pages_per_topk * page_size * spec_topk, spec_tokens)


def get_alloc_reserve_per_decode() -> int:

    return 2 * get_alloc_len_per_decode()


def page_aligned_decode_alloc_lens(
    reqs,
    *,
    reserve: int,
    page_size: int,
):

    cur_kv_lens = [0] * len(reqs)
    nxt_kv_lens = [0] * len(reqs)
    num_needed_tokens = 0
    for i, r in enumerate(reqs):
        cur = r.kv.kv_allocated_len
        nxt = max(
            cur,
            (r.kv_committed_len + reserve + page_size - 1) // page_size * page_size,
        )
        cur_kv_lens[i] = cur
        nxt_kv_lens[i] = nxt
        num_needed_tokens += nxt - cur
    return cur_kv_lens, nxt_kv_lens, num_needed_tokens


def get_req_to_token_extra_context_len() -> int:


    extra = 4 + (max_speculative_num_draft_tokens() or 0)
    if get_spec().speculative_algorithm == "REQUEST_FOREST":
        from sglang.srt.speculative.request_forest.adaptive_config import (
            adaptive_rf_enabled,
        )

        if adaptive_rf_enabled():


            extra = max(extra, 2 * max_speculative_num_draft_tokens())
    page_size = get_alloc_page_size()
    if get_spec().speculative_algorithm is not None and page_size > 1:


        extra = max(extra, get_alloc_reserve_per_decode() + page_size - 1)
    return extra
