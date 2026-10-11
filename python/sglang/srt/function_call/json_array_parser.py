from typing import List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import StreamingParseResult


class JsonArrayParser(BaseFormatDetector):


    def __init__(self):
        super().__init__()

        self.bot_token = "["
        self.eot_token = "]"
        self.tool_call_separator = ","

    def has_tool_call(self, text: str) -> bool:

        return "[" in text or "{" in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:

        raise NotImplementedError(
            "Detect and parse not supported for JSON schema constraints."
        )

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:

        return super().parse_streaming_increment(new_text, tools)

    def structure_info(self) -> callable:

        raise NotImplementedError("structure_info not used for JSON schema constraints")
