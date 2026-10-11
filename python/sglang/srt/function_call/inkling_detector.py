import json
import logging
from collections.abc import Mapping
from typing import List, Optional

from xgrammar import StructuralTag

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.parser.inkling_tokenizer import (
    CONTENT_INVOKE_TOOL_JSON,
    CONTENT_INVOKE_TOOL_TEXT,
    END_MESSAGE,
    INKLING_CONTROL_TOKENS,
    INKLING_SPECIAL_TOKEN_IDS,
    MESSAGE_MODEL,
)

logger = logging.getLogger(__name__)


def _reject_nonfinite_number(value: str) -> float:

    raise ValueError(f"{value} is not a valid JSON number")


class InklingDetector(BaseFormatDetector):


    def __init__(self):
        super().__init__()
        self.bot_token = CONTENT_INVOKE_TOOL_JSON
        self.eot_token = END_MESSAGE


        self._stream_call_index = 0
        self._raw_passthrough = False

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text or CONTENT_INVOKE_TOOL_TEXT in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        if not self.has_tool_call(text):
            return StreamingParseResult(normal_text=self._clean_normal_text(text))

        parsed = self._parse_canonical(text, tools)
        if parsed is not None:
            normal_text, calls = parsed
            return StreamingParseResult(normal_text=normal_text, calls=calls)


        recovered = self._recover_last_json_call(text, tools)
        if recovered is not None:
            return StreamingParseResult(normal_text="", calls=[recovered])
        return StreamingParseResult(normal_text=self._clean_normal_text(text))

    def _parse_canonical(
        self, text: str, tools: List[Tool]
    ) -> tuple[str, list[ToolCallItem]] | None:

        calls: list[ToolCallItem] = []
        normal_parts: list[str] = []
        pos = 0
        while pos < len(text):
            marker_pos, marker_token, is_json = self._next_tool_marker(text, pos)
            if marker_pos is None:
                normal_parts.append(text[pos:])
                break

            prefix, _ = self._split_trailing_tool_header(text[pos:marker_pos])
            normal_parts.append(prefix)

            body_start = marker_pos + len(marker_token)
            eot = text.find(self.eot_token, body_start)
            if eot == -1:
                return None
            body = text[body_start:eot]

            if is_json:
                call = self._canonical_json_call(body, tools, len(calls))
                if call is None:
                    return None
            else:
                call = self._text_tool_call(body, len(calls))
            calls.append(call)
            pos = eot + len(self.eot_token)

        return self._clean_normal_text("".join(normal_parts)), calls

    def _next_tool_marker(self, text: str, start: int) -> tuple[int | None, str, bool]:

        json_pos = text.find(self.bot_token, start)
        text_pos = text.find(CONTENT_INVOKE_TOOL_TEXT, start)
        if json_pos == -1 and text_pos == -1:
            return None, "", True
        if text_pos == -1 or (json_pos != -1 and json_pos <= text_pos):
            return json_pos, self.bot_token, True
        return text_pos, CONTENT_INVOKE_TOOL_TEXT, False

    def _canonical_json_call(
        self, body: str, tools: List[Tool], call_index: int
    ) -> ToolCallItem | None:
        try:
            payload = json.loads(body.strip(), parse_constant=_reject_nonfinite_number)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(payload, Mapping):
            return None
        return self._tool_call_item(payload, tools, call_index)

    def _recover_last_json_call(
        self, text: str, tools: List[Tool]
    ) -> ToolCallItem | None:

        last = text.rfind(self.bot_token)
        if last == -1:
            return None
        body_start = last + len(self.bot_token)
        eot = text.find(self.eot_token, body_start)
        candidate = text[body_start:eot] if eot != -1 else text[body_start:]
        candidate = self._clean_normal_text(candidate).strip()
        try:
            payload = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(payload, Mapping) or not payload.get("name"):
            return None
        return self._tool_call_item(payload, tools, 0)

    def _text_tool_call(self, body: str, call_index: int) -> ToolCallItem:

        return ToolCallItem(
            tool_index=call_index,
            name="",
            parameters=json.dumps({"text": self._clean_normal_text(body)}),
        )

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        self._buffer += new_text
        if self._raw_passthrough:
            out = self._clean_normal_text(self._buffer)
            self._buffer = ""
            return StreamingParseResult(normal_text=out)

        normal_parts: list[str] = []
        calls: list[ToolCallItem] = []
        while self._buffer:
            marker_pos, marker_token, is_json = self._next_tool_marker(self._buffer, 0)
            if marker_pos is None:
                safe, hold = self._split_safe_text(self._buffer)
                normal_parts.append(self._clean_normal_text(safe))
                self._buffer = hold
                break

            prefix, _ = self._split_trailing_tool_header(self._buffer[:marker_pos])
            body_start = marker_pos + len(marker_token)
            eot = self._buffer.find(self.eot_token, body_start)
            if eot == -1:


                normal_parts.append(self._clean_normal_text(prefix))
                self._buffer = self._buffer[marker_pos:]
                break

            body = self._buffer[body_start:eot]
            if is_json:
                call = self._canonical_json_call(body, tools, self._stream_call_index)
                if call is None:
                    return self._stream_recover_or_fallback(tools)
            else:
                call = self._text_tool_call(body, self._stream_call_index)
            normal_parts.append(self._clean_normal_text(prefix))
            calls.append(call)
            self._stream_call_index += 1
            self._buffer = self._buffer[eot + len(self.eot_token) :]

        return StreamingParseResult(normal_text="".join(normal_parts), calls=calls)

    def _stream_recover_or_fallback(self, tools: List[Tool]) -> StreamingParseResult:


        self._raw_passthrough = True
        recovered = self._recover_last_json_call(self._buffer, tools)
        buffered = self._buffer
        self._buffer = ""
        if recovered is not None:
            recovered.tool_index = self._stream_call_index
            self._stream_call_index += 1
            return StreamingParseResult(calls=[recovered])
        return StreamingParseResult(normal_text=self._clean_normal_text(buffered))

    def _split_safe_text(self, text: str) -> tuple[str, str]:

        header_start = self._pending_tool_header_start(text)
        if header_start is not None:
            return text[:header_start], text[header_start:]
        partial_len = max(
            (
                self._ends_with_partial_token(text, token)
                for token in INKLING_CONTROL_TOKENS
            ),
            default=0,
        )
        if partial_len:
            return text[:-partial_len], text[-partial_len:]
        return text, ""

    def structure_info(self) -> _GetInfoFunc:
        def info(name: str) -> StructureInfo:
            trigger = f"{MESSAGE_MODEL}{name}{self.bot_token}"
            return StructureInfo(
                begin=f'{trigger}{{"name":"{name}","args":',
                end=f"}}{self.eot_token}",
                trigger=trigger,
            )

        return info

    def get_auto_tool_call_structural_tag(
        self,
        tools: Optional[List[Tool]] = None,
        thinking_mode: bool = False,
        parallel_tool_calls: bool = True,
    ) -> StructuralTag:

        del tools, thinking_mode, parallel_tool_calls
        return StructuralTag.model_validate(
            {
                "type": "structural_tag",
                "format": {
                    "type": "token_triggered_tags",
                    "trigger_tokens": [
                        INKLING_SPECIAL_TOKEN_IDS[CONTENT_INVOKE_TOOL_JSON]
                    ],
                    "tags": [
                        {
                            "type": "tag",
                            "begin": {
                                "type": "token",
                                "token": INKLING_SPECIAL_TOKEN_IDS[
                                    CONTENT_INVOKE_TOOL_JSON
                                ],
                            },
                            "content": {
                                "type": "json_schema",
                                "json_schema": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "args": {"type": "object"},
                                    },
                                    "required": ["name", "args"],
                                    "additionalProperties": False,
                                },
                            },
                            "end": {
                                "type": "token",
                                "token": INKLING_SPECIAL_TOKEN_IDS[END_MESSAGE],
                            },
                        }
                    ],
                },
            }
        )

    def _tool_call_item(
        self,
        payload: Mapping[str, object],
        tools: List[Tool],
        call_index: int,
    ) -> ToolCallItem | None:
        name = payload.get("name")
        args = payload.get("args")
        if not isinstance(name, str) or not isinstance(args, Mapping):
            logger.warning("Invalid Inkling tool call payload: %s", payload)
            return None

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)
        if name not in self._tool_indices:


            logger.warning("Surfacing Inkling call to undeclared tool: %s", name)

        return ToolCallItem(
            tool_index=call_index,
            name=name,
            parameters=json.dumps(args, ensure_ascii=False),
        )

    def _split_trailing_tool_header(self, text: str) -> tuple[str, str | None]:
        message_pos = self._pending_tool_header_start(text)
        if message_pos is None:
            return text, None
        header = text[message_pos + len(MESSAGE_MODEL) :]
        return text[:message_pos], header.strip() or None

    def _pending_tool_header_start(self, text: str) -> int | None:

        message_pos = text.rfind(MESSAGE_MODEL)
        if message_pos < 0:
            return None
        header = text[message_pos + len(MESSAGE_MODEL) :]
        if any(token in header for token in INKLING_CONTROL_TOKENS):
            return None
        return message_pos

    def _clean_normal_text(self, text: str) -> str:
        for token in INKLING_CONTROL_TOKENS:
            text = text.replace(token, "")
        return text
