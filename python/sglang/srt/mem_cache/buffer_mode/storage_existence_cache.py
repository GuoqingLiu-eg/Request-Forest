from __future__ import annotations

from collections import OrderedDict
from typing import Container, Iterable, Sequence


HICACHE_EXISTENCE_CACHE_MAX_ENTRIES = 128 * 1024


class StorageExistenceCache:
    def __init__(self, max_entries: int = HICACHE_EXISTENCE_CACHE_MAX_ENTRIES):
        self.max_entries = max_entries
        self._entries: OrderedDict[tuple[str, str], None] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, pool: str, hashes: Iterable[str]) -> None:
        entries = self._entries
        for h in hashes:
            entries[(pool, h)] = None
            entries.move_to_end((pool, h))
        while len(entries) > self.max_entries:
            entries.popitem(last=False)

    def contains(self, pool: str, page_hash: str) -> bool:
        entries = self._entries
        if (pool, page_hash) not in entries:
            return False
        entries.move_to_end((pool, page_hash))
        return True

    def contains_all(self, pool: str, hashes: Iterable[str]) -> bool:
        return all(self.contains(pool, h) for h in hashes)

    def covers_all(
        self,
        pool: str,
        hashes: Iterable[str],
        extra_cover: Container[str] = frozenset(),
    ) -> bool:

        return all(self.contains(pool, h) or h in extra_cover for h in hashes)

    def invalidate_beyond(
        self, pool: str, hashes: Sequence[str], keep_pages: int
    ) -> None:

        for h in hashes[keep_pages:]:
            self._entries.pop((pool, h), None)

    def clear(self) -> None:
        self._entries.clear()
