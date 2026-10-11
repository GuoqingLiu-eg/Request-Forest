from __future__ import annotations

import logging
import re
from typing import List, Optional

from transformers.models.whisper.tokenization_whisper import LANGUAGES

from sglang.srt.entrypoints.openai.protocol import (
    TranscriptionRequest,
    TranscriptionSegment,
    TranscriptionUsage,
    TranscriptionVerboseResponse,
)
from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    TranscriptionAdapter,
    register_transcription_adapter,
)

logger = logging.getLogger(__name__)


FUSED_AUTODETECT_FLAG = "_detect_language"


WHISPER_LANG_TOKEN_CODES: frozenset[str] = frozenset(LANGUAGES.keys())


_LANG_ALT = "|".join(re.escape(c) for c in sorted(WHISPER_LANG_TOKEN_CODES))
_LANG_PREFIX = r"<\|(" + _LANG_ALT + r")\|>"
WHISPER_AUTODETECT_REGEX = (
    _LANG_PREFIX + r"<\|transcribe\|>" + r"<\|notimestamps\|>" + r"[\s\S]*"
)
WHISPER_AUTODETECT_TS_REGEX = (
    _LANG_PREFIX + r"<\|transcribe\|>" + r"<\|0\.00\|>" + r"[\s\S]*"
)


_FUSED_PREFIX_RE_NOTS = re.compile(
    r"^" + _LANG_PREFIX + r"<\|transcribe\|><\|notimestamps\|>"
)
_FUSED_PREFIX_RE_TS = re.compile(r"^" + _LANG_PREFIX + r"<\|transcribe\|>")


_WHISPER_CONTROL_TOKENS = frozenset(
    {
        "endoftext",
        "startoftranscript",
        "startofprev",
        "startoflm",
        "translate",
        "transcribe",
        "notimestamps",
        "nospeech",
    }
)


_SPECIAL_TOKEN_RE = re.compile(
    r"<\|(?:"
    + "|".join(sorted(WHISPER_LANG_TOKEN_CODES | _WHISPER_CONTROL_TOKENS))
    + r"|\d+\.\d{2}"
    + r")\|>"
)


@register_transcription_adapter("Whisper")
class WhisperAdapter(TranscriptionAdapter):
    TIMESTAMP_BASE_TOKEN_ID = 50365
    TIMESTAMP_BASE_OFFSET = 0.02

    @property
    def max_audio_clip_s(self) -> Optional[float]:


        return 30.0

    def build_sampling_params(self, request: TranscriptionRequest) -> dict:
        params: dict = {
            "temperature": request.temperature,
            "max_new_tokens": 448,
            "language": request.language,
        }
        if request.timestamp_granularities:
            params["timestamp_granularities"] = request.timestamp_granularities
        return params


    @property
    def supports_language_detection(self) -> bool:
        return True

    def build_fused_autodetect_params(self, request: TranscriptionRequest) -> dict:

        ts_variant = bool(request.timestamp_granularities)
        params: dict = {
            "temperature": request.temperature,


            "max_new_tokens": 447,
            "regex": (
                WHISPER_AUTODETECT_TS_REGEX if ts_variant else WHISPER_AUTODETECT_REGEX
            ),
            "skip_special_tokens": False,


            "spaces_between_special_tokens": False,
            FUSED_AUTODETECT_FLAG: True,
        }
        if ts_variant:
            params["timestamp_granularities"] = request.timestamp_granularities
        return params

    @staticmethod
    def parse_fused_output(
        text: str, *, ts_variant: bool = False, strip: bool = True
    ) -> tuple[Optional[str], Optional[str]]:

        pattern = _FUSED_PREFIX_RE_TS if ts_variant else _FUSED_PREFIX_RE_NOTS
        m = pattern.match(text)
        if not m:
            return None, None
        transcription = text[m.end() :]


        transcription = _SPECIAL_TOKEN_RE.sub("", transcription)
        return m.group(1), transcription.strip() if strip else transcription

    @staticmethod
    def strip_special_tokens(text: str) -> str:

        return _SPECIAL_TOKEN_RE.sub("", text)


    def build_verbose_response(
        self,
        request: TranscriptionRequest,
        text: str,
        ret: dict,
        tokenizer,
        usage: TranscriptionUsage,
    ) -> TranscriptionVerboseResponse:
        output_ids = ret.get("output_ids", [])
        parsed_text, segments = self._parse_segments(output_ids, tokenizer)
        return TranscriptionVerboseResponse(


            language=request.language,
            duration=round(request.audio_duration_s, 2),
            text=parsed_text or text,
            segments=segments,
            usage=usage,
        )

    def build_verbose_response_chunked(
        self,
        request: TranscriptionRequest,
        text: str,
        rets: List[dict],
        chunk_offsets_s: List[float],
        tokenizer,
        usage: TranscriptionUsage,
    ) -> TranscriptionVerboseResponse:
        segments: list[TranscriptionSegment] = []
        for ret, offset_s in zip(rets, chunk_offsets_s):
            _, part_segments = self._parse_segments(
                ret.get("output_ids", []),
                tokenizer,
                time_offset_s=offset_s,
                seg_id_start=len(segments),
            )
            segments.extend(part_segments)
        return TranscriptionVerboseResponse(
            language=request.language,
            duration=round(request.audio_duration_s, 2),


            text=text,
            segments=segments,
            usage=usage,
        )

    @staticmethod
    def _parse_segments(
        output_ids: List[int],
        tokenizer,
        time_offset_s: float = 0.0,
        seg_id_start: int = 0,
    ) -> tuple[str, List[TranscriptionSegment]]:

        eos_token_id = getattr(tokenizer, "eos_token_id", 50257)
        ts_base = WhisperAdapter.TIMESTAMP_BASE_TOKEN_ID
        ts_step = WhisperAdapter.TIMESTAMP_BASE_OFFSET

        segments: list[TranscriptionSegment] = []
        full_text_parts: list[str] = []
        current_text_tokens: list[int] = []

        current_start = time_offset_s
        seg_id = seg_id_start

        for token_id in output_ids:
            if token_id >= ts_base:
                timestamp = (token_id - ts_base) * ts_step + time_offset_s

                if current_text_tokens:
                    seg_text = tokenizer.decode(
                        current_text_tokens, skip_special_tokens=True
                    ).strip()
                    if seg_text:
                        segments.append(
                            TranscriptionSegment(
                                id=seg_id,
                                start=round(current_start, 2),
                                end=round(timestamp, 2),
                                text=seg_text,
                            )
                        )
                        full_text_parts.append(seg_text)
                        seg_id += 1
                    current_text_tokens = []

                current_start = timestamp

            elif token_id == eos_token_id:
                continue
            else:
                current_text_tokens.append(token_id)

        if current_text_tokens:
            seg_text = tokenizer.decode(
                current_text_tokens, skip_special_tokens=True
            ).strip()
            if seg_text:
                segments.append(
                    TranscriptionSegment(
                        id=seg_id,
                        start=round(current_start, 2),
                        end=round(current_start, 2),
                        text=seg_text,
                    )
                )
                full_text_parts.append(seg_text)

        return " ".join(full_text_parts), segments
