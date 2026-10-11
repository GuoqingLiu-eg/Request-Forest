# Copyright (c) 2026, SGLang Team.


from functools import lru_cache


@lru_cache(maxsize=None)
def get_forward_host(arch: int):

    if arch // 10 == 12:
        from sglang.kernels.ops.attention.fa4_sm120.runtime import (
            sm120_forward_host,
        )

        return sm120_forward_host
    return None


def resolve_runtime_policy(
    *,
    device_capability: tuple[int, int],
    deterministic: bool,
) -> tuple[int, int, bool]:

    arch = device_capability[0] * 10 + device_capability[1]
    uses_arch_decode_policy = get_forward_host(arch) is not None
    no_splitkv = device_capability < (9, 0) or uses_arch_decode_policy
    num_splits = 1 if deterministic or no_splitkv else 0
    decode_num_splits = (
        0 if uses_arch_decode_policy and not deterministic else num_splits
    )
    return num_splits, decode_num_splits, uses_arch_decode_policy


@lru_cache(maxsize=None)
def get_forward_arch(device) -> int | None:

    import torch

    major, minor = torch.cuda.get_device_capability(device)
    arch = major * 10 + minor
    return arch if get_forward_host(arch) is not None else None


def try_cached_paged_decode(*, arch: int, **kwargs):

    host = get_forward_host(arch)
    return None if host is None else host.try_paged_decode(arch=arch, **kwargs)


def try_cached_varlen(*, arch: int, **kwargs):

    host = get_forward_host(arch)
    return None if host is None else host.try_varlen(arch=arch, **kwargs)
