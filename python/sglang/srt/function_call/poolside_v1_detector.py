import json
import re
from enum import Enum, auto
from typing import Any, List, Optional

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import safe_literal_eval


class _ParseState(Enum):


    OUTSIDE = auto()
    READING_NAME = auto()
    READING_KEY = auto()
    READING_VALUE = auto()
    DRAINING = auto()


class PoolsideV1Detector(BaseFormatDetector):


    tool_call_start_token = "<tool_call>"
    tool_call_end_token = "</tool_call>"
    arg_key_start = "<arg_key>"
    arg_key_end = "</arg_key>"
    arg_value_start = "<arg_value>"
    arg_value_end = "</arg_value>"

    tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


    arg_pair_regex = re.compile(
        r"<arg_key>([^<]*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
        re.DOTALL,
    )

    _partial_tag_prefixes = (
        tool_call_start_token,
        tool_call_end_token,
        arg_key_start,
        arg_key_end,
        arg_value_start,
        arg_value_end,
    )

    def __init__(self):
        super().__init__()
        self.parsed_pos: int = 0
        self._state: _ParseState = _ParseState.OUTSIDE
        self.current_func_name: Optional[str] = None
        self.current_pending_key: Optional[str] = None
        self.json_started: bool = False


    def _reset_call_state(self) -> None:

        self.current_func_name = None
        self.current_pending_key = None
        self.json_started = False

    def _consume_arg_key(self, slice_: str) -> bool:

        end = slice_.find(self.arg_key_end)
        if end == -1:
            return False
        self.current_pending_key = slice_[len(self.arg_key_start) : end].strip()
        self.parsed_pos += end + len(self.arg_key_end)
        return True

    def _close_current_call(self, calls: List[ToolCallItem]) -> None:

        fragment = "}" if self.json_started else "{}"
        calls.append(
            ToolCallItem(
                tool_index=self.current_tool_id,
                parameters=fragment,
            )
        )
        self.streamed_args_for_tool[self.current_tool_id] += fragment
        self.parsed_pos += len(self.tool_call_end_token)
        self._state = _ParseState.OUTSIDE
        self._reset_call_state()

    def has_tool_call(self, text: str) -> bool:
        return self.tool_call_start_token in text

    @staticmethod
    def _get_param_schema(
        func_name: Optional[str], tools: Optional[List[Tool]]
    ) -> dict:
        if not tools or not func_name:
            return {}
        for tool in tools:
            try:
                if (
                    tool.type == "function"
                    and tool.function.name == func_name
                    and isinstance(tool.function.parameters, dict)
                ):
                    return tool.function.parameters.get("properties", {})
            except AttributeError:
                continue
        return {}

    _STRING_TYPES = frozenset({"string", "str", "text", "enum"})

    @staticmethod
    def _convert_param_value(raw: str, schema: dict, key: str) -> Any:

        spec = schema.get(key) if isinstance(schema, dict) else None
        param_type = str(spec.get("type", "")).lower() if isinstance(spec, dict) else ""
        if param_type in PoolsideV1Detector._STRING_TYPES:
            return raw

        decoders = (json.loads,) if not param_type else (json.loads, safe_literal_eval)
        for decoder in decoders:
            try:
                result = decoder(raw)


                json.dumps(result)
                return result
            except (ValueError, SyntaxError, TypeError):
                continue
        return raw

    def _find_name_boundary(self, text: str) -> int:

        hits = (
            text.find("\n"),
            text.find(self.arg_key_start),
            text.find(self.tool_call_end_token),
        )
        positive = [h for h in hits if h != -1]
        return min(positive) if positive else -1

    def _is_partial_tag(self, slice_: str) -> bool:

        return any(
            tag.startswith(slice_) and tag != slice_
            for tag in self._partial_tag_prefixes
        )


    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        if self.tool_call_start_token not in text:
            return StreamingParseResult(normal_text=text)

        tool_indices = self._get_tool_indices(tools)
        first_idx = text.find(self.tool_call_start_token)
        normal_text = text[:first_idx] if first_idx > 0 else ""

        calls: List[ToolCallItem] = []
        for body in self.tool_call_regex.findall(text):


            boundary = self._find_name_boundary(body)
            name = (body if boundary == -1 else body[:boundary]).strip()
            if not name or name not in tool_indices:
                continue

            schema = self._get_param_schema(name, tools)
            args: dict = {}
            for raw_key, raw_val in self.arg_pair_regex.findall(body):
                key = raw_key.strip()


                val = raw_val.removeprefix("\n").removesuffix("\n")
                args[key] = self._convert_param_value(val, schema, key)

            calls.append(
                ToolCallItem(
                    tool_index=tool_indices[name],
                    name=name,
                    parameters=json.dumps(args, ensure_ascii=False),
                )
            )

        return StreamingParseResult(normal_text=normal_text, calls=calls)


    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        self._buffer += new_text
        if not self._buffer:
            return StreamingParseResult()

        tool_indices = self._get_tool_indices(tools)
        calls: List[ToolCallItem] = []
        normal_text_chunks: List[str] = []


        while True:
            slice_ = self._buffer[self.parsed_pos :]
            if not slice_:
                break
            state = self._state

            if state is _ParseState.OUTSIDE:
                if slice_.startswith(self.tool_call_start_token):
                    self.parsed_pos += len(self.tool_call_start_token)
                    self._state = _ParseState.READING_NAME
                    self._reset_call_state()
                    continue
                if slice_.startswith("<"):
                    if self._is_partial_tag(slice_):
                        break
                    normal_text_chunks.append("<")
                    self.parsed_pos += 1
                    continue
                next_lt = slice_.find("<")
                segment = slice_ if next_lt == -1 else slice_[:next_lt]
                normal_text_chunks.append(segment)
                self.parsed_pos += len(segment)
                continue

            if state is _ParseState.READING_NAME:
                boundary = self._find_name_boundary(slice_)
                if boundary == -1:
                    break
                name = slice_[:boundary].strip()


                consume = boundary
                if boundary < len(slice_) and slice_[boundary : boundary + 1] == "\n":
                    consume += 1
                self.parsed_pos += consume

                if name and name in tool_indices:
                    self.current_tool_id += 1
                    while len(self.streamed_args_for_tool) <= self.current_tool_id:
                        self.streamed_args_for_tool.append("")
                    self.current_func_name = name


                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=name,
                            parameters="",
                        )
                    )
                    self._state = _ParseState.READING_KEY
                else:


                    self._state = _ParseState.DRAINING
                continue

            if state is _ParseState.READING_KEY:
                if slice_.startswith(self.tool_call_end_token):
                    self._close_current_call(calls)
                    continue
                if slice_.startswith(self.arg_key_start):
                    if not self._consume_arg_key(slice_):
                        break
                    self._state = _ParseState.READING_VALUE
                    continue
                if slice_.startswith("<"):
                    if self._is_partial_tag(slice_):
                        break


                    self.parsed_pos += 1
                    continue

                next_lt = slice_.find("<")
                self.parsed_pos += len(slice_) if next_lt == -1 else next_lt
                continue

            if state is _ParseState.READING_VALUE:


                if slice_.startswith(self.tool_call_end_token):
                    self._close_current_call(calls)
                    continue


                if slice_.startswith(self.arg_key_start):
                    if not self._consume_arg_key(slice_):
                        break
                    continue
                if slice_.startswith(self.arg_value_start):
                    end = slice_.find(self.arg_value_end)
                    if end == -1:
                        break
                    raw = (
                        slice_[len(self.arg_value_start) : end]
                        .removeprefix("\n")
                        .removesuffix("\n")
                    )


                    schema = self._get_param_schema(self.current_func_name, tools)
                    converted = self._convert_param_value(
                        raw, schema, self.current_pending_key
                    )
                    kv = (
                        f"{json.dumps(self.current_pending_key)}: "
                        f"{json.dumps(converted, ensure_ascii=False)}"
                    )
                    fragment = "{" + kv if not self.json_started else ", " + kv
                    self.json_started = True
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            parameters=fragment,
                        )
                    )
                    self.streamed_args_for_tool[self.current_tool_id] += fragment
                    self.current_pending_key = None
                    self.parsed_pos += end + len(self.arg_value_end)
                    self._state = _ParseState.READING_KEY
                    continue
                if slice_.startswith("<"):
                    if self._is_partial_tag(slice_):
                        break
                    self.parsed_pos += 1
                    continue
                next_lt = slice_.find("<")
                self.parsed_pos += len(slice_) if next_lt == -1 else next_lt
                continue

            if state is _ParseState.DRAINING:
                end_idx = slice_.find(self.tool_call_end_token)
                if end_idx != -1:
                    self.parsed_pos += end_idx + len(self.tool_call_end_token)
                    self._state = _ParseState.OUTSIDE
                    continue


                holdback = self._ends_with_partial_token(
                    slice_, self.tool_call_end_token
                )
                self.parsed_pos += len(slice_) - holdback
                break

        if self.parsed_pos > 0:
            self._buffer = self._buffer[self.parsed_pos :]
            self.parsed_pos = 0

        return StreamingParseResult(
            calls=calls,
            normal_text="".join(normal_text_chunks),
        )


    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin=f"<tool_call>{name}\n",
            end="</tool_call>",
            trigger="<tool_call>",
        )
