from __future__ import annotations

import io
import math
from typing import List, Tuple

import numpy as np
import soundfile as sf

from sglang.srt.utils import load_audio


SPLIT_SEARCH_WINDOW_S = 1.0


MIN_ENERGY_WINDOW_SIZE = 1600


def find_split_point(wav: np.ndarray, start_idx: int, end_idx: int) -> int:

    segment = wav[start_idx:end_idx]
    min_energy = math.inf
    quietest_idx = start_idx
    for i in range(0, len(segment) - MIN_ENERGY_WINDOW_SIZE, MIN_ENERGY_WINDOW_SIZE):
        window = segment[i : i + MIN_ENERGY_WINDOW_SIZE]
        energy = (window**2).mean() ** 0.5
        if energy < min_energy:
            quietest_idx = i + start_idx
            min_energy = energy
    return quietest_idx


def split_audio_energy_aware(
    audio_data: bytes,
    max_clip_s: float,
    sample_rate: int = 16000,
) -> Tuple[List[bytes], List[float]]:

    if not audio_data:
        raise ValueError("audio_data is empty")
    audio = load_audio(audio_data, sr=sample_rate, mono=True)
    chunk_size = int(sample_rate * max_clip_s)
    search_size = int(sample_rate * SPLIT_SEARCH_WINDOW_S)
    total = audio.shape[-1]

    raw_chunks: List[np.ndarray] = []
    offsets_s: List[float] = []
    i = 0
    while i < total:
        offsets_s.append(i / sample_rate)
        if i + chunk_size >= total:
            raw_chunks.append(audio[i:])
            break
        search_start = i + chunk_size - search_size
        search_end = min(i + chunk_size, total)
        split_point = find_split_point(audio, search_start, search_end)
        raw_chunks.append(audio[i:split_point])
        i = split_point

    chunks: List[bytes] = []
    for chunk in raw_chunks:
        buf = io.BytesIO()
        sf.write(buf, chunk, sample_rate, format="WAV")
        chunks.append(buf.getvalue())
    return chunks, offsets_s
