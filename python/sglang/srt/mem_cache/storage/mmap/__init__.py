# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project


from .mmap_allocator import alloc_mmap, alloc_shm

__all__ = [
    "alloc_mmap",
    "alloc_shm",
]
