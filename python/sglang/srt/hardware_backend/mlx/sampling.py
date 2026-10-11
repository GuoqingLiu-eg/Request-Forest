from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)


DEFAULT_SAMPLING_SEED = 42


MAX_BOUNDED_TOP_K = 1024

_warned_ignored_penalties = False


@dataclass(frozen=True)
class MlxSamplingParams:


    temperature: float
    top_k: int
    top_p: float
    min_p: float
    seed: int | None

    @classmethod
    def from_req(
        cls, req: Any, deterministic_seeding: bool = False
    ) -> MlxSamplingParams:
        sp = req.sampling_params
        global _warned_ignored_penalties
        if not _warned_ignored_penalties and (
            sp.frequency_penalty != 0.0
            or sp.presence_penalty != 0.0
            or sp.repetition_penalty != 1.0
        ):
            _warned_ignored_penalties = True
            logger.warning(
                "MLX sampling ignores frequency/presence/repetition penalties; "
                "a request specified them. (Warning logged once.)"
            )


        seed = None
        if deterministic_seeding:
            seed = (
                sp.sampling_seed
                if sp.sampling_seed is not None
                else DEFAULT_SAMPLING_SEED
            )
        return cls(
            temperature=sp.temperature,
            top_k=sp.top_k,
            top_p=sp.top_p,
            min_p=sp.min_p,
            seed=seed,
        )

    @property
    def is_greedy(self) -> bool:
        return self.top_k == 1


GREEDY_PARAMS = MlxSamplingParams(
    temperature=1.0, top_k=1, top_p=1.0, min_p=0.0, seed=None
)


@dataclass(frozen=True)
class MlxLogprobSpec:


    top_ks: tuple[int, ...]
    token_ids: tuple[tuple[int, ...] | None, ...]


@dataclass
class MlxLazyLogprobs:


    chosen: mx.array
    top_val: mx.array | None
    top_idx: mx.array | None
    token_ids_val: list[mx.array | None]
    spec: MlxLogprobSpec


@dataclass
class MlxStepLogprobs:


    chosen: list[float]
    top_val: list[list[float]]
    top_idx: list[list[int]]
    token_ids_val: list[list[float]]
    token_ids_idx: list[list[int]]


def lazy_logprob_arrays(lazy_logprobs: MlxLazyLogprobs | None) -> list[mx.array]:

    if lazy_logprobs is None:
        return []
    arrays = [lazy_logprobs.chosen]
    if lazy_logprobs.top_val is not None:
        arrays += [lazy_logprobs.top_val, lazy_logprobs.top_idx]
    arrays += [a for a in lazy_logprobs.token_ids_val if a is not None]
    return arrays


def all_greedy(params: list[MlxSamplingParams]) -> bool:
    return all(p.is_greedy for p in params)


def sanitize_logits(logits: mx.array) -> mx.array:

    return mx.clip(mx.where(mx.isnan(logits), -1e30, logits), -1e30, 1e30)


def scale_by_temperature(
    last_logits: mx.array, params: list[MlxSamplingParams]
) -> mx.array:

    temps = mx.array([p.temperature for p in params], dtype=mx.float32)[:, None]
    return last_logits.astype(mx.float32) / temps


def compute_logprobs(
    last_logits: mx.array,
    params: list[MlxSamplingParams],
    tokens: mx.array,
    spec: MlxLogprobSpec,
    scaled: mx.array | None = None,
) -> MlxLazyLogprobs:

    if scaled is None:
        scaled = scale_by_temperature(last_logits, params)
    logp = scaled - mx.logsumexp(scaled, axis=-1, keepdims=True)

    chosen = mx.take_along_axis(logp, tokens[:, None], axis=-1).squeeze(-1)

    max_k = max(spec.top_ks) if spec.top_ks else 0
    if max_k > 0:
        top_idx = mx.argsort(-logp, axis=-1)[:, :max_k]
        top_val = mx.take_along_axis(logp, top_idx, axis=-1)
    else:
        top_idx = None
        top_val = None

    token_ids_val: list[mx.array | None] = [
        logp[row, mx.array(ids)] if ids else None
        for row, ids in enumerate(spec.token_ids)
    ]
    return MlxLazyLogprobs(
        chosen=chosen,
        top_val=top_val,
        top_idx=top_idx,
        token_ids_val=token_ids_val,
        spec=spec,
    )


def sample_tokens(
    last_logits: mx.array,
    params: list[MlxSamplingParams],
    positions: list[int],
    key: mx.array,
    scaled: mx.array | None = None,
) -> mx.array:

    batch_size, vocab_size = last_logits.shape
    logits32 = last_logits.astype(mx.float32)
    if scaled is None:
        scaled = scale_by_temperature(logits32, params)

    filtering = any(
        not p.is_greedy and (p.top_k < vocab_size or p.top_p < 1.0 or p.min_p > 0.0)
        for p in params
    )


    candidates: mx.array | None = None
    if filtering:
        probs = mx.softmax(scaled, axis=-1)
        width = _candidate_width(params, vocab_size)
        sorted_idx = mx.argsort(-probs, axis=-1)[:, :width]
        p_sort = mx.take_along_axis(probs, sorted_idx, axis=-1)
        ranks = mx.arange(width, dtype=mx.int32)[None, :]


        top_ks = mx.array([min(p.top_k, width) for p in params], dtype=mx.int32)[
            :, None
        ]
        top_ps = mx.array([p.top_p for p in params], dtype=mx.float32)[:, None]
        min_ps = mx.array([p.min_p for p in params], dtype=mx.float32)[:, None]
        cum = mx.cumsum(p_sort, axis=-1)
        masked_out = (
            (ranks >= top_ks)
            | ((cum - p_sort) > top_ps)
            | (p_sort < p_sort[:, :1] * min_ps)
        )
        w_sort = mx.where(masked_out, 0.0, p_sort)
        if width < vocab_size:


            log_weights = mx.log(w_sort)
            candidates = sorted_idx
        else:


            weights = mx.put_along_axis(
                mx.zeros_like(probs), sorted_idx, w_sort, axis=-1
            )
            log_weights = mx.log(weights)
    else:


        log_weights = scaled

    noise = _gumbel_noise(
        params=params,
        positions=positions,
        shape=log_weights.shape,
        key=key,
        columns=candidates,
    )


    sampled = mx.argmax(log_weights + noise, axis=-1)
    if candidates is not None:
        sampled = mx.take_along_axis(candidates, sampled[:, None], axis=-1).squeeze(-1)

    greedy = [p.is_greedy for p in params]
    if not any(greedy):
        return sampled


    return mx.where(mx.array(greedy), mx.argmax(logits32, axis=-1), sampled)


def _candidate_width(params: list[MlxSamplingParams], vocab_size: int) -> int:

    largest_top_k = max(p.top_k for p in params)
    if largest_top_k > MAX_BOUNDED_TOP_K or largest_top_k >= vocab_size:
        return vocab_size
    return largest_top_k


def _gumbel_noise(
    params: list[MlxSamplingParams],
    positions: list[int],
    shape: tuple[int, int],
    key: mx.array,
    columns: mx.array | None = None,
) -> mx.array:

    seeded_rows = [p.seed is not None for p in params]
    if not any(seeded_rows):
        return mx.random.gumbel(shape=shape, key=key)

    hashed = _murmur_hash32(
        seeds=[p.seed if p.seed is not None else 0 for p in params],
        positions=positions,
        vocab_size=shape[1],
        columns=columns,
    )
    u = hashed.astype(mx.float32) / float(0xFFFFFFFF)


    u = mx.clip(u, 2.0**-32, 1.0 - 2.0**-24)
    hash_noise = -mx.log(-mx.log(u))

    if all(seeded_rows):
        return hash_noise
    random_noise = mx.random.gumbel(shape=shape, key=key)
    return mx.where(mx.array(seeded_rows)[:, None], hash_noise, random_noise)


def _murmur3_mix_py(h: int, k: int) -> int:

    k = (k * 0xCC9E2D51) & 0xFFFFFFFF
    k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
    k = (k * 0x1B873593) & 0xFFFFFFFF
    h ^= k
    h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
    h = (h * 5 + 0xE6546B64) & 0xFFFFFFFF
    return h


def _murmur_hash32(
    seeds: list[int],
    positions: list[int],
    vocab_size: int,
    columns: mx.array | None = None,
) -> mx.array:

    row_states = []
    for seed, pos in zip(seeds, positions):
        seed &= 0xFFFFFFFFFFFFFFFF
        h = _murmur3_mix_py(0, seed & 0xFFFFFFFF)
        h = _murmur3_mix_py(h, (seed >> 32) & 0xFFFFFFFF)
        h = _murmur3_mix_py(h, pos & 0xFFFFFFFF)
        row_states.append(h)

    h = mx.array(row_states, dtype=mx.uint32)[:, None]
    if columns is None:
        k = mx.arange(vocab_size, dtype=mx.uint32)[None, :]
    else:
        k = columns.astype(mx.uint32)


    k = k * mx.array(0xCC9E2D51, dtype=mx.uint32)
    k = (k << 15) | (k >> 17)
    k = k * mx.array(0x1B873593, dtype=mx.uint32)
    h = h ^ k
    h = (h << 13) | (h >> 19)
    h = h * mx.array(5, dtype=mx.uint32) + mx.array(0xE6546B64, dtype=mx.uint32)


    h = h ^ mx.array(16, dtype=mx.uint32)
    h = h ^ (h >> 16)
    h = h * mx.array(0x85EBCA6B, dtype=mx.uint32)
    h = h ^ (h >> 13)
    h = h * mx.array(0xC2B2AE35, dtype=mx.uint32)
    h = h ^ (h >> 16)
    return h
