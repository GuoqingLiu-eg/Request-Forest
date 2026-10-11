import logging
from typing import List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.core_types import StreamingParseResult
from sglang.srt.function_call.qwen25_detector import Qwen25Detector

logger = logging.getLogger(__name__)


class TrinityDetector(Qwen25Detector):


    def _strip_think_tags(self, text: str) -> str:

        return text.replace("<think>", "").replace("</think>", "")

    def has_tool_call(self, text: str) -> bool:

        return super().has_tool_call(self._strip_think_tags(text))

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:

        return super().detect_and_parse(self._strip_think_tags(text), tools)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:

        return super().parse_streaming_increment(
            self._strip_think_tags(new_text), tools
        )
