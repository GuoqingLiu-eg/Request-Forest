import hashlib

BUCKET_HEX_CHARS = 2
_BUCKET_MASK = (1 << (4 * BUCKET_HEX_CHARS)) - 1


def stable_key_hash(key: str) -> int:

    return int.from_bytes(
        hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(), "big"
    )


def route_key(key: str, num_disks: int) -> tuple[int, str]:

    if num_disks <= 0:
        raise ValueError("num_disks must be positive")
    key_hash = stable_key_hash(key)
    return (
        (key_hash >> 16) % num_disks,
        f"{key_hash & _BUCKET_MASK:0{BUCKET_HEX_CHARS}x}",
    )


def route_disk(key: str, num_disks: int) -> int:

    return route_key(key, num_disks)[0]
