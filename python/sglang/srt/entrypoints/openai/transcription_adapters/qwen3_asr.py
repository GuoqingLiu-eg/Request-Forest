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
from sglang.srt.multimodal.processors.qwen3_asr import DEFAULT_ASR_PROMPT


@register_transcription_adapter("Qwen3ASR")
class Qwen3ASRAdapter(TranscriptionAdapter):
    ASR_TEXT_TAG = "<asr_text>"

    @property
    def supports_chunked_streaming(self) -> bool:
        return True

    @property
    def chunked_streaming_config(self) -> dict:


        return {
            "chunk_size_sec": 2.0,
            "unfixed_chunk_num": 2,
            "unfixed_token_num": 5,
        }

    @property
    def prompt_template(self) -> str:
        return DEFAULT_ASR_PROMPT

    def build_sampling_params(self, request: TranscriptionRequest) -> dict:
        temperature = request.temperature
        if temperature == 0.0:
            temperature = 0.01
        return {
            "temperature": temperature,
            "max_new_tokens": 256,
        }

    def postprocess_text(self, text: str) -> str:


        if self.ASR_TEXT_TAG in text:
            return text.split(self.ASR_TEXT_TAG, 1)[-1]
        return text

    def build_verbose_response(
        self,
        request: TranscriptionRequest,
        text: str,
        ret: dict,
        tokenizer,
        usage: TranscriptionUsage,
    ) -> TranscriptionVerboseResponse:

        return TranscriptionVerboseResponse(
            language=request.language or "auto",
            duration=round(request.audio_duration_s, 2),
            text=text,
            segments=[],
            usage=usage,
        )
