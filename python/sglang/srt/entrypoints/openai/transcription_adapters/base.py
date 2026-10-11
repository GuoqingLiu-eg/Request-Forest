from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from sglang.srt.entrypoints.openai.protocol import (
    TranscriptionRequest,
    TranscriptionUsage,
    TranscriptionVerboseResponse,
)


class TranscriptionAdapter(ABC):


    @abstractmethod
    def build_sampling_params(self, request: TranscriptionRequest) -> dict:
        pass

    @property
    def supports_language_detection(self) -> bool:

        return False


    def build_fused_autodetect_params(self, request) -> dict:

        raise NotImplementedError

    @staticmethod
    def parse_fused_output(
        text: str, *, ts_variant: bool = False, strip: bool = True
    ) -> tuple[Optional[str], Optional[str]]:

        raise NotImplementedError

    @staticmethod
    def strip_special_tokens(text: str) -> str:

        return text

    @property
    def max_audio_clip_s(self) -> Optional[float]:

        return None

    @property
    def supports_chunked_streaming(self) -> bool:

        return False

    @property
    def model_sample_rate(self) -> int:

        return 16000

    @property
    def prompt_template(self) -> str:

        return ""

    @property
    def chunked_streaming_config(self) -> dict:

        return {}

    def postprocess_text(self, text: str) -> str:

        return text

    @abstractmethod
    def build_verbose_response(
        self,
        request: TranscriptionRequest,
        text: str,
        ret: dict,
        tokenizer,
        usage: TranscriptionUsage,
    ) -> TranscriptionVerboseResponse:
        pass

    def build_verbose_response_chunked(
        self,
        request: TranscriptionRequest,
        text: str,
        rets: List[dict],
        chunk_offsets_s: List[float],
        tokenizer,
        usage: TranscriptionUsage,
    ) -> TranscriptionVerboseResponse:

        return TranscriptionVerboseResponse(
            language=request.language,
            duration=round(request.audio_duration_s, 2),
            text=text,
            segments=[],
            usage=usage,
        )


_ADAPTER_REGISTRY: dict[str, type[TranscriptionAdapter]] = {}
_DEFAULT_ADAPTER_KEY = "Whisper"


def register_transcription_adapter(
    key: str,
) -> callable:


    def decorator(cls: type[TranscriptionAdapter]) -> type[TranscriptionAdapter]:
        _ADAPTER_REGISTRY[key] = cls
        return cls

    return decorator


def resolve_adapter(architectures: List[str]) -> TranscriptionAdapter:

    for arch in architectures or []:
        for key, adapter_cls in _ADAPTER_REGISTRY.items():
            if key in arch:
                return adapter_cls()
    default_cls = _ADAPTER_REGISTRY.get(_DEFAULT_ADAPTER_KEY)
    if default_cls is None:
        raise RuntimeError(
            "No transcription adapters registered. "
            "Make sure 'transcription_adapters' package is importable."
        )
    return default_cls()
