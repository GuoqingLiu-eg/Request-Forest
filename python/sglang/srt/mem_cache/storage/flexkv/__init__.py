from __future__ import annotations

import logging

from sglang.srt.mem_cache.registry import register_radix_cache_backend

logger = logging.getLogger(__name__)


def _flexkv_factory(ctx):

    from sglang.srt.distributed.parallel_state import (
        get_attn_cp_group,
        get_attn_tp_group,
        get_pp_group,
    )
    from sglang.srt.mem_cache.storage.flexkv.flexkv_radix_cache import (
        FlexKVRadixCache,
    )

    server_args = ctx.server_args


    try:
        pp_group = get_pp_group()
    except (RuntimeError, AssertionError):
        pp_group = None
    try:
        attn_tp_group = get_attn_tp_group()
    except (RuntimeError, AssertionError):
        attn_tp_group = ctx.tp_group
    try:
        attn_cp_group = get_attn_cp_group()
    except (RuntimeError, AssertionError):
        attn_cp_group = None


    pp_rank = pp_group.rank_in_group if pp_group is not None else 0
    attn_cp_rank = attn_cp_group.rank_in_group if attn_cp_group is not None else 0

    return FlexKVRadixCache(
        params=ctx.params,
        model_config=ctx.model_config,
        server_args=server_args,
        tp_rank=ctx.tp_rank,
        tp_size=ctx.tp_size,


        dp_rank=None,
        pp_rank=pp_rank,
        attn_cp_rank=attn_cp_rank,
        tp_group=ctx.tp_group,
        pp_group=pp_group,
        attn_tp_group=attn_tp_group,
        attn_cp_group=attn_cp_group,
    )


try:
    register_radix_cache_backend("flexkv", _flexkv_factory)
except ValueError as exc:


    logger.debug("flexkv backend already registered: %s", exc)
