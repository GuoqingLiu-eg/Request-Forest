import json
import logging
import re
from typing import List, Literal, Optional, Union

from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
from sglang.srt.function_call.base_format_detector import (
    BaseFormatDetector,
    StructuralTag,
    get_model_structural_tag,
)
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)

_KIMI_K2_SPECIAL_TOKENS = [
    "<|tool_calls_section_begin|>",
    "<|tool_calls_section_end|>",
    "<|tool_call_begin|>",
    "<|tool_call_end|>",
    "<|tool_call_argument_begin|>",
]

_KIMI_NON_STRICT_ARGUMENTS_SCHEMA = {"type": "object"}


def _strip_special_tokens(text: str) -> str:

    for token in _KIMI_K2_SPECIAL_TOKENS:
        text = text.replace(token, "")
    return text


class KimiK2Detector(BaseFormatDetector):


    def __init__(self):
        super().__init__()

        self.bot_token: str = "<|tool_calls_section_begin|>"
        self.eot_token: str = "<|tool_calls_section_end|>"

        self.tool_call_start_token: str = "<|tool_call_begin|>"
        self.tool_call_end_token: str = "<|tool_call_end|>"
        self.tool_call_argument_begin_token: str = "<|tool_call_argument_begin|>"


        self.tool_call_regex = re.compile(
            r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[^\s<|]+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>\{.*?\})\s*<\|tool_call_end\|>",
            re.DOTALL,
        )

        self.stream_tool_call_portion_regex = re.compile(
            r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[^\s<|]+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>\{.*)",
            re.DOTALL,
        )

        self._last_arguments = ""
        self._current_stream_function_name: str | None = None


        self.tool_call_id_regex = re.compile(
            r"^(?:functions\.)?(?P<name>[\w.\-]+):(?P<index>\d+)$"
        )

        self.tool_call_id_counter_regex = re.compile(r"^\d+$")

    def _parse_tool_call_id(
        self, function_id: str, tools: List[Tool], function_args: str = None
    ):

        m = self.tool_call_id_regex.match(function_id)
        if m:
            return m.group("name"), int(m.group("index"))

        if self.tool_call_id_counter_regex.match(function_id):
            call_index = int(function_id)
            name = self._infer_tool_name(tools, function_args)
            if name:
                return name, call_index
            return None, call_index

        logger.warning("Unexpected tool_call_id format: %s", function_id)
        return None, 0

    def _infer_tool_name(self, tools: List[Tool], function_args: str = None):

        if not tools:
            return None
        if len(tools) == 1:
            return tools[0].function.name

        if not function_args:
            logger.debug(
                "No function_args for tool name inference with %d tools", len(tools)
            )
            return None

        try:
            arg_keys = set(json.loads(function_args).keys())
        except (json.JSONDecodeError, TypeError):
            logger.debug(
                "Could not parse function_args for tool name inference "
                "(may be partial JSON in streaming)"
            )
            return None


        best_name = None
        best_score = -1
        for tool in tools:
            params = tool.function.parameters or {}
            props = set(params.get("properties", {}).keys())
            if not props:
                continue
            overlap = len(arg_keys & props)
            extra = len(arg_keys - props)
            score = overlap - extra
            if score > best_score:
                best_score = score
                best_name = tool.function.name

        return best_name

    def has_tool_call(self, text: str) -> bool:

        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:

        if self.bot_token not in text:
            return StreamingParseResult(normal_text=text, calls=[])
        try:
            function_call_tuples = self.tool_call_regex.findall(text)

            logger.debug("function_call_tuples: %s", function_call_tuples)

            tool_calls = []


            local_tool_index = 0
            for match in function_call_tuples:
                function_id, function_args = match
                function_name, _ = self._parse_tool_call_id(
                    function_id, tools, function_args
                )
                if function_name is None:
                    continue

                logger.debug(f"function_name {function_name}")

                tool_calls.append(
                    ToolCallItem(
                        tool_index=local_tool_index,
                        name=function_name,
                        parameters=function_args,
                    )
                )
                local_tool_index += 1

            content = text[: text.find(self.bot_token)]
            return StreamingParseResult(normal_text=content, calls=tool_calls)

        except Exception as e:
            logger.error("Error in detect_and_parse: %s", e, exc_info=True)
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:

        self._buffer += new_text


        if (
            self._current_stream_function_name is None
            and self.bot_token not in self._buffer
            and self.tool_call_start_token not in self._buffer
        ):
            emit, hold = self._split_pending_start(self._buffer)
            self._buffer = hold
            return StreamingParseResult(normal_text=_strip_special_tokens(emit))

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        normal_text_parts: list[str] = []
        calls: list[ToolCallItem] = []

        try:
            while True:
                buffer = self._buffer


                begin_idx = self._locate_tool_call_start(buffer, normal_text_parts)
                if begin_idx is None:
                    break
                buffer = self._buffer


                arg_begin_idx = buffer.find(self.tool_call_argument_begin_token)
                next_begin = buffer.find(
                    self.tool_call_start_token, len(self.tool_call_start_token)
                )
                if next_begin != -1 and (
                    arg_begin_idx == -1 or next_begin < arg_begin_idx
                ):
                    logger.warning(
                        "Kimi-K2 tool_call_begin without preceding tool_call_end; "
                        "discarding incomplete section."
                    )
                    self._buffer = buffer[next_begin:]
                    self._reset_inflight_call_state()
                    continue

                if arg_begin_idx == -1:

                    break

                id_start = len(self.tool_call_start_token)
                function_id = buffer[id_start:arg_begin_idx].strip()
                args_start = arg_begin_idx + len(self.tool_call_argument_begin_token)
                end_idx = buffer.find(self.tool_call_end_token)


                name_just_resolved = False
                if self._current_stream_function_name is None:
                    args_for_inference = (
                        buffer[args_start:end_idx]
                        if end_idx != -1
                        else buffer[args_start:]
                    )
                    resolved = self._resolve_function_name(
                        function_id, tools, args_for_inference
                    )
                    if resolved is None:
                        if end_idx == -1:

                            break
                        logger.warning(
                            "Kimi-K2 unrecognized tool_call_id %r; skipping section.",
                            function_id,
                        )
                        self._buffer = buffer[end_idx + len(self.tool_call_end_token) :]
                        self._reset_inflight_call_state()
                        continue
                    name = resolved
                    self._current_stream_function_name = name
                    name_just_resolved = True


                    if self.current_tool_id == -1:
                        self.current_tool_id = 0
                        self.prev_tool_call_arr = []
                        self.streamed_args_for_tool = [""]
                    while len(self.prev_tool_call_arr) <= self.current_tool_id:
                        self.prev_tool_call_arr.append({})
                    while len(self.streamed_args_for_tool) <= self.current_tool_id:
                        self.streamed_args_for_tool.append("")
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": name,
                        "arguments": {},
                    }
                    self.current_tool_name_sent = True


                if end_idx != -1:
                    args_full = buffer[args_start:end_idx]
                else:
                    args_full = buffer[args_start:]
                argument_diff = args_full[len(self._last_arguments) :]
                if argument_diff or name_just_resolved:
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=(
                                self._current_stream_function_name
                                if name_just_resolved
                                else None
                            ),
                            parameters=argument_diff,
                        )
                    )
                    if argument_diff:
                        self._last_arguments += argument_diff
                        self.streamed_args_for_tool[
                            self.current_tool_id
                        ] += argument_diff

                if end_idx == -1:

                    break


                self._buffer = buffer[end_idx + len(self.tool_call_end_token) :]
                self.current_tool_id += 1
                self._reset_inflight_call_state()

            return StreamingParseResult(
                normal_text="".join(normal_text_parts), calls=calls
            )

        except Exception as e:
            logger.error("Error in parse_streaming_increment: %s", e, exc_info=True)

            self._buffer = ""
            self._reset_inflight_call_state()
            return StreamingParseResult(
                normal_text="".join(normal_text_parts), calls=calls
            )

    def _reset_inflight_call_state(self) -> None:

        self._last_arguments = ""
        self.current_tool_name_sent = False
        self._current_stream_function_name = None

    def _locate_tool_call_start(
        self, buffer: str, normal_text_parts: list
    ) -> int | None:

        begin_idx = buffer.find(self.tool_call_start_token)
        if begin_idx == -1:
            emit, hold = self._split_pending_start(buffer)
            if emit:
                normal_text_parts.append(_strip_special_tokens(emit))
            self._buffer = hold
            return None

        if begin_idx > 0:
            normal_text_parts.append(_strip_special_tokens(buffer[:begin_idx]))
            self._buffer = buffer[begin_idx:]
        return 0

    def _split_pending_start(self, text: str) -> tuple[str, str]:

        candidates = (self.bot_token, self.tool_call_start_token)
        max_tail = max(len(t) for t in candidates) - 1
        for n in range(min(len(text), max_tail), 1, -1):
            tail = text[-n:]
            if any(t.startswith(tail) for t in candidates):
                return text[:-n], tail
        return text, ""

    def _resolve_function_name(
        self, function_id: str, tools: List[Tool], function_args: str
    ) -> Optional[str]:

        if not function_id:
            return self._infer_tool_name(tools, function_args)

        m = self.tool_call_id_regex.match(function_id)
        if m:
            return m.group("name")

        if self.tool_call_id_counter_regex.match(function_id):
            return self._infer_tool_name(tools, function_args)

        return None

    def structure_info(self) -> _GetInfoFunc:


        def get_info(name: str) -> StructureInfo:
            return StructureInfo(
                begin=f"<|tool_calls_section_begin|><|tool_call_begin|>functions.{name}:0<|tool_call_argument_begin|>",
                end="<|tool_call_end|><|tool_calls_section_end|>",
                trigger="<|tool_calls_section_begin|>",
            )

        return get_info

    def get_structural_tag(
        self,
        tools: Union[List[Tool], None] = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required"]] = "auto",
        thinking_mode: bool = False,
        parallel_tool_calls: bool = True,
    ) -> Optional[StructuralTag]:
        if not (
            tools and (tool_choice == "required" or isinstance(tool_choice, ToolChoice))
        ):
            return super().get_structural_tag(
                tools=tools,
                tool_choice=tool_choice,
                thinking_mode=thinking_mode,
                parallel_tool_calls=parallel_tool_calls,
            )
        if get_model_structural_tag is None:
            return None

        converted_tools = []
        for tool in tools:
            converted_tool = tool.model_dump()
            function = converted_tool["function"]
            if not function.get("strict", False):


                function["strict"] = True
                function["parameters"] = _KIMI_NON_STRICT_ARGUMENTS_SCHEMA
            converted_tools.append(converted_tool)

        converted_tool_choice = (
            tool_choice.model_dump()
            if isinstance(tool_choice, ToolChoice)
            else tool_choice
        )
        return get_model_structural_tag(
            model="kimi",
            tools=converted_tools,
            tool_choice=converted_tool_choice,
            reasoning=thinking_mode,
        )

    def get_structural_tag_name(self) -> str:
        return "kimi"
