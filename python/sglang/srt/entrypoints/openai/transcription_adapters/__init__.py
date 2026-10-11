from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    TranscriptionAdapter,
    register_transcription_adapter,
    resolve_adapter,
)


from sglang.srt.entrypoints.openai.transcription_adapters.glmasr import (
    GlmAsrAdapter,
)
from sglang.srt.entrypoints.openai.transcription_adapters.granite_speech import (
    GraniteSpeechAdapter,
)
from sglang.srt.entrypoints.openai.transcription_adapters.mimo_v2_asr import (
    MiMoV2ASRAdapter,
)
from sglang.srt.entrypoints.openai.transcription_adapters.qwen2_audio import (
    Qwen2AudioAdapter,
)
from sglang.srt.entrypoints.openai.transcription_adapters.qwen3_asr import (
    Qwen3ASRAdapter,
)
from sglang.srt.entrypoints.openai.transcription_adapters.whisper import (
    WhisperAdapter,
)

__all__ = [
    "TranscriptionAdapter",
    "register_transcription_adapter",
    "resolve_adapter",
    "WhisperAdapter",
    "Qwen3ASRAdapter",
    "MiMoV2ASRAdapter",
    "GraniteSpeechAdapter",
    "Qwen2AudioAdapter",
    "GlmAsrAdapter",
]
