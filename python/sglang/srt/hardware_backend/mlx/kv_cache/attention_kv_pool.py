import logging

import mlx.core as mx

logger = logging.getLogger(__name__)


class MlxAttentionKVPool:


    def __init__(
        self,
        pool_size: int,
        num_layers: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: mx.Dtype = mx.float16,
    ):
        self.pool_size = pool_size
        self.num_layers = num_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype


        self.k_buffer: list[mx.array] = [
            mx.zeros((pool_size, n_kv_heads, head_dim), dtype=dtype)
            for _ in range(num_layers)
        ]
        self.v_buffer: list[mx.array] = [
            mx.zeros((pool_size, n_kv_heads, head_dim), dtype=dtype)
            for _ in range(num_layers)
        ]

        mem_mb = (pool_size * n_kv_heads * head_dim * 2 * num_layers * dtype.size) / (
            1024 * 1024
        )
        logger.info(
            f"MlxAttentionKVPool: {pool_size} slots x {num_layers} layers "
            f"x {n_kv_heads} heads x {head_dim} dim, "
            f"dtype={dtype}, ~{mem_mb:.1f} MB"
        )

    def set_kv(self, layer_id: int, slots: mx.array, k: mx.array, v: mx.array) -> None:

        self.k_buffer[layer_id][slots] = k
        self.v_buffer[layer_id][slots] = v

    def get_kv(self, layer_id: int, slots: mx.array) -> tuple[mx.array, mx.array]:

        return self.k_buffer[layer_id][slots], self.v_buffer[layer_id][slots]

    def get_kv_all_layers(self, slots: mx.array) -> tuple[mx.array, mx.array]:

        k_all = mx.stack([self.k_buffer[i][slots] for i in range(self.num_layers)])
        v_all = mx.stack([self.v_buffer[i][slots] for i in range(self.num_layers)])
        return k_all, v_all

    def set_kv_all_layers(
        self, slots: mx.array, k_all: mx.array, v_all: mx.array
    ) -> None:

        for i in range(self.num_layers):
            self.set_kv(i, slots, k_all[i], v_all[i])

    def all_buffers(self) -> list[mx.array]:

        return self.k_buffer + self.v_buffer

    def clear(self) -> None:

        shape = (self.pool_size, self.n_kv_heads, self.head_dim)
        for i in range(self.num_layers):
            self.k_buffer[i] = mx.zeros(shape, dtype=self.dtype)
            self.v_buffer[i] = mx.zeros(shape, dtype=self.dtype)
