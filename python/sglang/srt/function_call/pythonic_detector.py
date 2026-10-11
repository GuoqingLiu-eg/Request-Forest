import ast
import json
import logging
import re
from typing import List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import safe_ast_parse

logger = logging.getLogger(__name__)


class PythonicDetector(BaseFormatDetector):


    def __init__(self):
        super().__init__()
        self.tool_call_regex = re.compile(
            r"\[([a-zA-Z]+\w*\(([a-zA-Z]+\w*=.*,\s*)*([a-zA-Z]+\w*=.*\s)?\),\s*)*([a-zA-Z]+\w*\(([a-zA-Z]+\w*=.*,\s*)*([a-zA-Z]+\w*=.*\s*)?\)\s*)+\]",
            re.DOTALL,
        )

    @staticmethod
    def _text_strip(text: str) -> str:


        text = text.replace("<|python_start|>", "")
        text = text.replace("<|python_end|>", "")
        return text

    def has_tool_call(self, text: str) -> bool:
        return bool(self.tool_call_regex.search(self._text_strip(text.strip())))

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:

        text = text.strip()


        text = self._text_strip(text)

        match = self.tool_call_regex.search(text)
        if match is None:
            return StreamingParseResult(normal_text=text, calls=[])


        tool_call_start = match.start()
        tool_call_end = match.end()

        normal_text_before = text[:tool_call_start] if tool_call_start > 0 else ""
        tool_call_text = text[tool_call_start:tool_call_end]
        normal_text_after = text[tool_call_end:] if tool_call_end < len(text) else ""


        normal_text = normal_text_before + normal_text_after

        try:
            module = safe_ast_parse(tool_call_text)
            parsed = getattr(module.body[0], "value", None)
            if not (
                isinstance(parsed, ast.List)
                and all(isinstance(e, ast.Call) for e in parsed.elts)
            ):
                return StreamingParseResult(normal_text=normal_text, calls=[])

            calls = []
            tool_indices = self._get_tool_indices(tools)
            for call_index, call in enumerate(parsed.elts):
                if not isinstance(call.func, ast.Name):
                    continue
                function_name = call.func.id

                if function_name not in tool_indices:
                    logger.warning(
                        f"Model attempted to call undefined function: {function_name}"
                    )
                    if not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get():
                        continue


                try:
                    arguments = {}
                    for keyword in call.keywords:
                        arguments[keyword.arg] = self._get_parameter_value(
                            keyword.value
                        )


                    parameters = json.dumps(
                        arguments, ensure_ascii=False, allow_nan=False
                    )
                except (ValueError, TypeError) as e:
                    logger.warning(f"Skipping tool call {function_name}: {e}")
                    continue
                calls.append(
                    ToolCallItem(
                        tool_index=call_index,
                        name=function_name,
                        parameters=parameters,
                    )
                )

            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception:
            logger.exception("Error in pythonic tool call parsing.")
            return StreamingParseResult(normal_text=normal_text, calls=[])

    def _find_matching_bracket(self, buffer: str, start: int) -> int:

        bracket_count = 0
        for i in range(start, len(buffer)):
            if buffer[i] == "[":
                bracket_count += 1
            elif buffer[i] == "]":
                bracket_count -= 1
                if bracket_count == 0:
                    return i
        return -1

    def _strip_and_split_buffer(self, buffer: str) -> tuple[str, str]:


        special_tokens = ["<|python_start|>", "<|python_end|>"]

        for token in special_tokens:
            partial_length = self._ends_with_partial_token(buffer, token)
            if partial_length > 0:

                safe_text = buffer[:-partial_length]
                held_back = buffer[-partial_length:]

                safe_text = self._text_strip(safe_text)
                return safe_text, held_back


        safe_text = self._text_strip(buffer)
        return safe_text, ""

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:

        self._buffer += new_text


        stripped_buffer, held_back = self._strip_and_split_buffer(self._buffer)

        start = stripped_buffer.find("[")

        if start == -1:

            self._buffer = held_back
            return StreamingParseResult(normal_text=stripped_buffer)

        normal_text = stripped_buffer[:start] if start > 0 else ""

        end = self._find_matching_bracket(stripped_buffer, start)
        if end != -1:

            call_text = stripped_buffer[start : end + 1]
            result = self.detect_and_parse(call_text, tools)


            remaining_text = stripped_buffer[end + 1 :] + held_back
            self._buffer = remaining_text


            if normal_text:
                result.normal_text = normal_text + (result.normal_text or "")

            return result


        self._buffer = stripped_buffer[start:] + held_back

        if normal_text:
            return StreamingParseResult(normal_text=normal_text)


        return StreamingParseResult(normal_text="")

    def _get_parameter_value(self, val):
        if isinstance(val, ast.Constant):
            if val.value is None or isinstance(val.value, (str, int, float)):
                return val.value


            raise ValueError(
                f"Constant has no JSON representation: {type(val.value).__name__}"
            )
        elif isinstance(val, ast.Dict):
            return {
                k.value: self._get_parameter_value(v)
                for k, v in zip(val.keys, val.values)
            }
        elif isinstance(val, ast.List):
            return [self._get_parameter_value(v) for v in val.elts]
        else:
            raise ValueError("Tool call arguments must be literals")

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError
