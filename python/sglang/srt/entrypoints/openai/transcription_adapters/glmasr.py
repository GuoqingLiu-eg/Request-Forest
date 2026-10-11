from __future__ import annotations

from sglang.srt.entrypoints.openai.protocol import (
    TranscriptionRequest,
    TranscriptionUsage,
    TranscriptionVerboseResponse,
)
from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    TranscriptionAdapter,
    register_transcription_adapter,
)


_MAX_NEW_TOKENS_PER_SECOND = 15

_DEFAULT_MAX_NEW_TOKENS = 448


@register_transcription_adapter("GlmAsr")
class GlmAsrAdapter(TranscriptionAdapter):


    _ASSISTANT_PREFIXES = (
        "The spoken content of the audio is",
        "The transcription of the audio is",
        "The content of the input audio is",
    )

    def build_sampling_params(self, request: TranscriptionRequest) -> dict:


        duration_s = request.audio_duration_s or 0.0
        return {
            "temperature": request.temperature,
            "max_new_tokens": max(
                _DEFAULT_MAX_NEW_TOKENS,
                int(duration_s * _MAX_NEW_TOKENS_PER_SECOND),
            ),
        }

    def postprocess_text(self, text: str) -> str:


        stripped = text.strip()
        for prefix in self._ASSISTANT_PREFIXES:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix) :].strip()
                break
        if stripped.endswith("."):
            stripped = stripped[:-1].strip()
        if (
            len(stripped) >= 2
            and stripped[0] == stripped[-1]
            and stripped[0]
            in {
                "'",
                '"',
            }
        ):
            stripped = stripped[1:-1].strip()
        return stripped

    def build_verbose_response(
        self,
        request: TranscriptionRequest,
        text: str,
        ret: dict,
        tokenizer,
        usage: TranscriptionUsage,
    ) -> TranscriptionVerboseResponse:
        return TranscriptionVerboseResponse(
            language=None,
            duration=round(request.audio_duration_s, 2),
            text=text,
            segments=[],
            usage=usage,
        )
