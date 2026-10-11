import inspect
import re
from typing import Dict, List, Optional, Tuple, Type

from sglang.srt.entrypoints.openai.encoding_dsv4 import dsml_token as dsv4_dsml_token
from sglang.srt.entrypoints.openai.encoding_dsv4 import eos_token as dsv4_eos_token
from sglang.srt.entrypoints.openai.encoding_dsv4 import (
    thinking_end_token as dsv4_thinking_end_token,
)
from sglang.srt.entrypoints.openai.encoding_dsv4 import (
    thinking_start_token as dsv4_thinking_start_token,
)
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.function_call.hunyuan_detector import resolve_hunyuan_tokens
from sglang.srt.function_call.kimik3_format import (
    MESSAGE_CLOSE,
    RESPONSE_CLOSE,
    RESPONSE_OPEN,
    THINK_CLOSE,
    THINK_OPEN,
    TOOLS_OPEN,
    partial_suffix_len,
    strip_partial_marker_suffix,
    strip_response_wrappers,
)
from sglang.srt.function_call.muse_glimmer_format import (
    EOM,
    EOT,
    MAX_CHANNEL_MARKER,
    MESSAGE,
    RECIPIENT_RE,
    START,
    could_start_header,
    has_atem_markers,
    partial_marker_len,
)
from sglang.srt.parser.harmony_parser import HarmonyParser
from sglang.srt.parser.inkling_tokenizer import (
    CONTENT_INVOKE_TOOL_JSON,
    CONTENT_INVOKE_TOOL_TEXT,
    CONTENT_MODEL_END_SAMPLING,
    CONTENT_TEXT,
    CONTENT_THINKING,
    END_MESSAGE,
    INKLING_CONTROL_TOKENS,
    MESSAGE_MODEL,
)


class StreamingParseResult:


    def __init__(
        self,
        normal_text: Optional[str] = None,
        reasoning_text: Optional[str] = None,
    ):
        self.normal_text = normal_text or ""
        self.reasoning_text = reasoning_text or ""


class BaseReasoningFormatDetector:


    def __init__(
        self,
        think_start_token: str,
        think_end_token: str,
        think_excluded_tokens: Optional[List[str]] = None,
        force_reasoning: bool = False,
        stream_reasoning: bool = True,
        tool_start_token: Optional[str] = None,
        continue_final_message: bool = False,
        previous_content: str = "",
        thinks_internally: bool = False,
        reasoning_default: str = "always",
        force_nonempty_content: bool = False,
    ):
        self.think_start_token = think_start_token
        self.think_end_token = think_end_token
        self.think_excluded_tokens = think_excluded_tokens
        self.tool_start_token = tool_start_token
        self.force_reasoning = force_reasoning
        self._in_reasoning = force_reasoning
        self.stream_reasoning = stream_reasoning
        self.thinks_internally = thinks_internally
        self.reasoning_default = reasoning_default

        self._buffer = ""
        self.stripped_think_start = False
        self.think_start_self_label = ""

        self._force_nonempty_content = force_nonempty_content
        self._accumulated_reasoning = ""

        self.continue_final_message = continue_final_message
        if self.continue_final_message:
            self.previous_content = previous_content
            self.previous_count = len(previous_content)
        else:
            self.previous_content = ""
            self.previous_count = 0

        if self.think_start_token in self.previous_content:
            self._in_reasoning = True
        if self.think_end_token in self.previous_content:
            self._in_reasoning = False

    def _maybe_apply_force_nonempty_content(
        self, ret: StreamingParseResult
    ) -> StreamingParseResult:
        if self._force_nonempty_content and not ret.normal_text:
            ret.normal_text, ret.reasoning_text = ret.reasoning_text, ret.normal_text
        return ret

    def detect_and_parse(self, text: str) -> StreamingParseResult:

        return self._maybe_apply_force_nonempty_content(
            self._detect_and_parse_impl(text)
        )

    def _detect_and_parse_impl(self, text: str) -> StreamingParseResult:
        in_reasoning = self._in_reasoning or self.think_start_token in text

        if not in_reasoning:
            return StreamingParseResult(normal_text=text)


        think_start_text = self.think_start_token + self.think_start_self_label
        processed_text = text
        while processed_text.startswith(think_start_text):
            processed_text = processed_text[len(think_start_text) :]

        if (
            self.think_end_token not in processed_text
            and self.think_end_token not in self.previous_content
        ):

            if (
                in_reasoning
                and self.tool_start_token is not None
                and self.tool_start_token in processed_text
            ):

                tool_idx = processed_text.find(self.tool_start_token)
                reasoning_text = processed_text[:tool_idx]

                normal_text = processed_text[tool_idx:]
                return StreamingParseResult(
                    normal_text=normal_text, reasoning_text=reasoning_text
                )

            return StreamingParseResult(reasoning_text=processed_text)


        if self.think_end_token in processed_text:
            splits = processed_text.split(self.think_end_token, maxsplit=1)
            reasoning_text = splits[0]
            normal_text = splits[1]

            return StreamingParseResult(
                normal_text=normal_text, reasoning_text=reasoning_text
            )
        else:

            return StreamingParseResult(normal_text=processed_text)

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:

        ret = self._parse_streaming_increment_impl(new_text)
        if self._force_nonempty_content:
            if self._in_reasoning:
                self._accumulated_reasoning += ret.reasoning_text
            else:
                self._accumulated_reasoning = ""
        return ret

    def _parse_streaming_increment_impl(self, new_text: str) -> StreamingParseResult:
        self._buffer += new_text
        current_text = self._buffer

        think_start_text = self.think_start_token + self.think_start_self_label


        tokens_to_check = [think_start_text, self.think_end_token]
        if self.tool_start_token:
            tokens_to_check.append(self.tool_start_token)
        if any(
            token.startswith(current_text) and token != current_text
            for token in tokens_to_check
        ):
            return StreamingParseResult()


        if not self.stripped_think_start and think_start_text in current_text:
            current_text = current_text.replace(think_start_text, "", 1)

            self._buffer = current_text
            self.stripped_think_start = True
            self._in_reasoning = True


        if self._in_reasoning and self.think_end_token in current_text:
            end_idx = current_text.find(self.think_end_token)

            reasoning_text = current_text[:end_idx]

            self._buffer = ""
            self._in_reasoning = False
            normal_text = current_text[end_idx + len(self.think_end_token) :]

            return StreamingParseResult(
                normal_text=normal_text, reasoning_text=reasoning_text
            )


        if self._in_reasoning:


            if self.tool_start_token and self.tool_start_token in current_text:
                tool_idx = current_text.find(self.tool_start_token)
                reasoning_text = current_text[:tool_idx]

                normal_text = current_text[tool_idx:]
                self._buffer = ""
                self._in_reasoning = False
                return StreamingParseResult(
                    normal_text=normal_text, reasoning_text=reasoning_text
                )
            if self.stream_reasoning:

                holdback_tokens = [self.think_end_token]
                if self.tool_start_token:
                    holdback_tokens.append(self.tool_start_token)
                if not self.stripped_think_start:

                    holdback_tokens.append(think_start_text)
                holdback = max(
                    self._ends_with_partial_token(current_text, token)
                    for token in holdback_tokens
                )
                self._buffer = current_text[len(current_text) - holdback :]
                return StreamingParseResult(
                    reasoning_text=current_text[: len(current_text) - holdback]
                )
            else:
                return StreamingParseResult()


        if not self._in_reasoning:
            self._buffer = ""
            return StreamingParseResult(normal_text=current_text)

        return StreamingParseResult()

    def _strip_leading_think_start(self, text: str) -> str:
        think_start_text = self.think_start_token + self.think_start_self_label
        if text.startswith(think_start_text):
            return text[len(think_start_text) :]
        return text

    @staticmethod
    def _ends_with_partial_token(buffer: str, token: str) -> int:

        for i in range(min(len(buffer), len(token) - 1), 0, -1):
            if token.startswith(buffer[-i:]):
                return i
        return 0

    def finish(self) -> StreamingParseResult:

        if not self._in_reasoning:


            leftover = self._buffer
            self._buffer = ""
            return StreamingParseResult(normal_text=leftover)


        buffer = self._strip_leading_think_start(self._buffer)
        self._buffer = ""

        if self._force_nonempty_content:
            normal_text = self._accumulated_reasoning + buffer
            self._accumulated_reasoning = ""
            if normal_text:
                return StreamingParseResult(normal_text=normal_text)
            return StreamingParseResult()

        if buffer:
            return StreamingParseResult(reasoning_text=buffer)

        return StreamingParseResult()


class DeepSeekR1Detector(BaseReasoningFormatDetector):


    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = True,
        continue_final_message: bool = False,
        previous_content: str = "",
        force_nonempty_content: bool = False,
    ):

        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=True,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            force_nonempty_content=force_nonempty_content,
        )


class Qwen3Detector(BaseReasoningFormatDetector):


    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
        force_nonempty_content: bool = False,
    ):
        think_excluded_tokens = [
            "<tool_call>",
            "</tool_call>",
            "<|im_end|>",
            "<|endoftext|>",
        ]
        super().__init__(
            "<think>",
            "</think>",
            think_excluded_tokens=think_excluded_tokens,
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,


            tool_start_token="<tool_call>",
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            thinks_internally=True,
            reasoning_default="enable_thinking",
            force_nonempty_content=force_nonempty_content,
        )


class KimiDetector(BaseReasoningFormatDetector):


    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
        force_nonempty_content: bool = False,
    ):
        super().__init__(
            "◁think▷",
            "◁/think▷",
            force_reasoning=False,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            force_nonempty_content=force_nonempty_content,
        )


class KimiK2Detector(BaseReasoningFormatDetector):


    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
        force_nonempty_content: bool = False,
    ):
        think_excluded_tokens = [
            "<think>",
            "<|tool_calls_section_begin|>",
            "<|tool_call_begin|>",
            "<|tool_call_argument_begin|>",
            "<|tool_call_section_end|>",
            "<|tool_call_end|>",
            "[EOS]",
            "<|im_end|>",
            "<|end_header_id|>",
            "[EOT]",
        ]
        super().__init__(
            "<think>",
            "</think>",
            think_excluded_tokens=think_excluded_tokens,
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            tool_start_token="<|tool_calls_section_begin|>",
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            reasoning_default="thinking",
            force_nonempty_content=force_nonempty_content,
        )


class KimiK3Detector(BaseReasoningFormatDetector):


    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = True,
        continue_final_message: bool = False,
        previous_content: str = "",
    ):


        think_excluded_tokens = [
            "response",
            "message",
            "<|end_of_msg|>",
            "[EOS]",
            "[EOT]",
        ]
        super().__init__(
            THINK_OPEN,
            THINK_CLOSE,
            think_excluded_tokens=think_excluded_tokens,
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            tool_start_token=TOOLS_OPEN,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            reasoning_default="thinking",
        )
        self._reasoning_done = False
        self._tools_passthrough = False

    def _clean_content(self, text: str) -> str:
        tools_idx = text.find(TOOLS_OPEN)
        if tools_idx != -1:
            return strip_response_wrappers(text[:tools_idx]) + text[tools_idx:]
        return strip_response_wrappers(text)

    def _next_channel_idx(self, text: str, start: int = 0) -> int:
        found = [
            idx
            for token in (RESPONSE_OPEN, self.tool_start_token)
            if (idx := text.find(token, start)) != -1
        ]
        return min(found) if found else -1

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        in_reasoning = self._in_reasoning or self.think_start_token in text
        if not in_reasoning and self.think_end_token not in text:
            return StreamingParseResult(normal_text=self._clean_content(text))

        open_idx = text.find(self.think_start_token)
        start = open_idx + len(self.think_start_token) if open_idx != -1 else 0
        close_idx = text.find(self.think_end_token, start)
        tools_idx = text.find(self.tool_start_token, start)
        if close_idx != -1 and tools_idx != -1 and tools_idx < close_idx:
            return StreamingParseResult(
                reasoning_text=strip_partial_marker_suffix(text[start:tools_idx]),
                normal_text=self._clean_content(text[tools_idx:]),
            )
        if close_idx == -1:
            channel_idx = self._next_channel_idx(text, start)
            if channel_idx != -1:
                return StreamingParseResult(
                    reasoning_text=strip_partial_marker_suffix(text[start:channel_idx]),
                    normal_text=self._clean_content(text[channel_idx:]),
                )
            return StreamingParseResult(
                reasoning_text=strip_partial_marker_suffix(text[start:])
            )

        reasoning_text = text[start:close_idx]
        rest = text[close_idx + len(self.think_end_token) :]
        return StreamingParseResult(
            reasoning_text=reasoning_text, normal_text=self._clean_content(rest)
        )

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        self._buffer += new_text

        if not self._in_reasoning and not self._reasoning_done:
            open_idx = self._buffer.find(self.think_start_token)
            if open_idx != -1:
                self._buffer = self._buffer[open_idx + len(self.think_start_token) :]
                self._in_reasoning = True
                self.stripped_think_start = True
            elif self.think_start_token.startswith(self._buffer):
                return StreamingParseResult()
            else:
                self._reasoning_done = True

        if self._in_reasoning:
            buf = self._buffer
            if not self.stripped_think_start:
                open_idx = buf.find(self.think_start_token)
                if open_idx != -1:
                    buf = buf[open_idx + len(self.think_start_token) :]
                    self._buffer = buf
                    self.stripped_think_start = True

            close_idx = buf.find(self.think_end_token)
            tools_idx = buf.find(self.tool_start_token)
            if close_idx != -1 and not (tools_idx != -1 and tools_idx < close_idx):
                reasoning_text = buf[:close_idx]
                self._buffer = buf[close_idx + len(self.think_end_token) :]
                self._in_reasoning = False
                self._reasoning_done = True
                return StreamingParseResult(
                    reasoning_text=reasoning_text or None,
                    normal_text=self._drain_content() or None,
                )

            channel_idx = self._next_channel_idx(buf)
            if channel_idx != -1:
                reasoning_text = strip_partial_marker_suffix(buf[:channel_idx])
                self._buffer = buf[channel_idx:]
                self._in_reasoning = False
                self._reasoning_done = True
                self._tools_passthrough = buf.startswith(
                    self.tool_start_token, channel_idx
                )
                return StreamingParseResult(
                    reasoning_text=reasoning_text or None,
                    normal_text=self._drain_content() or None,
                )

            if not self.stream_reasoning:
                return StreamingParseResult()
            markers = [self.think_end_token, self.tool_start_token, RESPONSE_OPEN]
            if not self.stripped_think_start:
                markers.append(self.think_start_token)
            holdback = partial_suffix_len(buf, markers)
            emit = buf[: len(buf) - holdback] if holdback else buf
            emit = strip_partial_marker_suffix(emit)
            self._buffer = buf[len(emit) :]
            return StreamingParseResult(reasoning_text=emit)

        return StreamingParseResult(normal_text=self._drain_content())

    def _drain_content(self) -> str:
        buf = self._buffer
        if not buf:
            return ""
        if self._tools_passthrough:
            self._buffer = ""
            return buf

        tools_idx = buf.find(TOOLS_OPEN)
        if tools_idx != -1:
            head = buf[:tools_idx]
            for marker in (RESPONSE_OPEN, RESPONSE_CLOSE, MESSAGE_CLOSE):
                head = head.replace(marker, "")
            self._tools_passthrough = True
            self._buffer = ""
            return head + buf[tools_idx:]

        holdback = partial_suffix_len(
            buf, [RESPONSE_OPEN, RESPONSE_CLOSE, MESSAGE_CLOSE, TOOLS_OPEN]
        )
        emit = buf[: len(buf) - holdback] if holdback else buf
        self._buffer = buf[len(emit) :]
        for marker in (RESPONSE_OPEN, RESPONSE_CLOSE, MESSAGE_CLOSE):
            emit = emit.replace(marker, "")
        return emit


class Glm45Detector(BaseReasoningFormatDetector):


    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        force_nonempty_content: bool = False,
    ):
        think_excluded_tokens = [
            "<tool_call>",
            "</tool_call>",
            "<eop>",
            "<|user|>",
            "<|endoftext|>",
        ]
        super().__init__(
            "<think>",
            "</think>",
            think_excluded_tokens=think_excluded_tokens,
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            tool_start_token="<tool_call>",
            thinks_internally=True,
            reasoning_default="enable_thinking",
            force_nonempty_content=force_nonempty_content,
        )


class GptOssDetector(BaseReasoningFormatDetector):


    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = True,
        continue_final_message: bool = False,
        previous_content: str = "",
        force_nonempty_content: bool = False,
    ):
        super().__init__(
            "<|channel|>analysis<|message|>",
            "<|end|>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            force_nonempty_content=force_nonempty_content,
        )
        self.parser = HarmonyParser()

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        events = self.parser.parse(text)

        events += self.parser.parse("")

        reasoning_text = "".join(
            [e.content for e in events if e.event_type == "reasoning"]
        )
        normal_parts = []
        for e in events:
            if e.event_type == "normal":
                normal_parts.append(e.content)
            elif e.event_type == "tool_call":

                normal_parts.append(e.raw_text if e.raw_text else e.content)
        normal_text = "".join(normal_parts)


        return self._maybe_apply_force_nonempty_content(
            StreamingParseResult(
                normal_text=normal_text,
                reasoning_text=reasoning_text,
            )
        )

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        events = self.parser.parse(new_text)

        reasoning_text = "".join(
            [e.content for e in events if e.event_type == "reasoning"]
        )
        normal_parts = []
        for e in events:
            if e.event_type == "normal":
                normal_parts.append(e.content)
            elif e.event_type == "tool_call":

                normal_parts.append(e.raw_text if e.raw_text else e.content)
        normal_text = "".join(normal_parts)

        return StreamingParseResult(
            normal_text=normal_text,
            reasoning_text=reasoning_text,
        )


class MiniMaxAppendThinkDetector(BaseReasoningFormatDetector):


    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
        force_nonempty_content: bool = False,
    ):

        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            force_nonempty_content=force_nonempty_content,
        )
        self.is_first_chunk = False

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        if not self.is_first_chunk:
            self.is_first_chunk = True
            new_text = self.think_start_token + new_text
        return StreamingParseResult(normal_text=new_text)

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        return StreamingParseResult(normal_text=self.think_start_token + text)


class Nemotron3Detector(BaseReasoningFormatDetector):


    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
        force_nonempty_content: bool = False,
    ):
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            tool_start_token="<tool_call>",
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            reasoning_default="enable_thinking",
            force_nonempty_content=force_nonempty_content,
        )


class MiniMaxM3Detector(BaseReasoningFormatDetector):


    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
        force_nonempty_content: bool = False,
    ):
        super().__init__(
            "<mm:think>",
            "</mm:think>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
        )
        self._lead_buffer = ""
        self._checked_leading_close = False
        self._force_nonempty_content = force_nonempty_content

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        if not self._in_reasoning and text.lstrip().startswith(self.think_end_token):
            text = text.lstrip()[len(self.think_end_token) :]
        ret = super().detect_and_parse(text)
        if self._force_nonempty_content and not ret.normal_text:
            ret.normal_text, ret.reasoning_text = ret.reasoning_text, ret.normal_text
        return ret

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:

        if not self._checked_leading_close and not self._in_reasoning:
            self._lead_buffer += new_text
            stripped = self._lead_buffer.lstrip()
            if not stripped:
                return StreamingParseResult()
            self._checked_leading_close = True
            if stripped.startswith(self.think_end_token):
                new_text = stripped[len(self.think_end_token) :]
            else:
                new_text = self._lead_buffer
            self._lead_buffer = ""
            if not new_text:
                return StreamingParseResult()
        return super().parse_streaming_increment(new_text)


class MistralDetector(BaseReasoningFormatDetector):


    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
        force_nonempty_content: bool = False,
    ):
        super().__init__(
            "[THINK]",
            "[/THINK]",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            reasoning_default="mistral",
            force_nonempty_content=force_nonempty_content,
        )


class HunyuanDetector(BaseReasoningFormatDetector):


    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
        tokenizer=None,
        force_nonempty_content: bool = False,
    ):
        t = resolve_hunyuan_tokens(tokenizer)
        think_open = t["think"]
        think_close = (
            "</" + think_open[1:] if think_open.startswith("<") else think_open
        )
        super().__init__(
            think_open,
            think_close,
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            tool_start_token=t["tool_calls"],
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            force_nonempty_content=force_nonempty_content,
        )


class Gemma4Detector(BaseReasoningFormatDetector):


    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
        force_nonempty_content: bool = False,
    ):
        super().__init__(
            "<|channel>",
            "<channel|>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            reasoning_default="explicit_enable_thinking",
            force_nonempty_content=force_nonempty_content,
        )
        self.think_start_self_label = "thought\n"


_INKLING_CONTENT_KINDS = {
    CONTENT_THINKING: "reasoning",
    CONTENT_TEXT: "content",
}
_INKLING_END_TOKENS = {
    CONTENT_MODEL_END_SAMPLING,
    END_MESSAGE,
}
_INKLING_CONTROL_TOKENS = INKLING_CONTROL_TOKENS
_INKLING_CONTROL_RE = re.compile(
    "|".join(re.escape(t) for t in sorted(_INKLING_CONTROL_TOKENS))
)


class InklingDetector(BaseReasoningFormatDetector):


    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
        force_nonempty_content: bool = False,
    ):
        del force_nonempty_content
        super().__init__(
            CONTENT_THINKING,
            END_MESSAGE,
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            thinks_internally=False,
            reasoning_default="always",
        )

        self._kind: str | None = None
        self._pending_header = ""
        self._pending_reasoning = ""

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        self._buffer = ""
        self._kind = None
        self._pending_header = ""
        self._pending_reasoning = ""
        ret = self._parse_blocks(text)
        if self._kind == "reasoning" and not self.stream_reasoning:
            ret.reasoning_text += self._pending_reasoning
        self._kind = None
        self._pending_header = ""
        self._pending_reasoning = ""
        return ret

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        text = self._buffer + new_text
        partial_len = self._partial_control_length(text)
        if partial_len:
            self._buffer = text[-partial_len:]
            text = text[:-partial_len]
        else:
            self._buffer = ""
        return self._parse_blocks(text)

    def finish(self) -> StreamingParseResult:


        reasoning_text = ""
        if self._kind == "reasoning" and not self.stream_reasoning:
            reasoning_text = self._pending_reasoning
        self._buffer = ""
        self._pending_reasoning = ""
        self._pending_header = ""
        self._kind = None
        return StreamingParseResult(reasoning_text=reasoning_text)

    @staticmethod
    def _partial_control_length(text: str) -> int:
        max_token_len = max(map(len, _INKLING_CONTROL_TOKENS))
        for length in range(min(len(text), max_token_len - 1), 0, -1):
            suffix = text[-length:]
            if any(
                len(suffix) < len(token) and token.startswith(suffix)
                for token in _INKLING_CONTROL_TOKENS
            ):
                return length
        return 0

    def _parse_blocks(self, text: str) -> StreamingParseResult:
        reasoning: list[str] = []
        content: list[str] = []
        saw_control = False
        pos = 0

        def emit(text: str) -> None:
            if self._kind == "reasoning":
                if self.stream_reasoning:
                    reasoning.append(text)
                else:
                    self._pending_reasoning += text
            elif self._kind == "content":
                content.append(text)
            elif self._kind == "tool":
                content.append(text)
            elif self._kind == "header":
                self._pending_header += text
            elif text:


                content.append(text)

        def flush_reasoning() -> None:
            if self._kind == "reasoning" and not self.stream_reasoning:
                reasoning.append(self._pending_reasoning)
                self._pending_reasoning = ""

        for match in _INKLING_CONTROL_RE.finditer(text):
            saw_control = True
            emit(text[pos : match.start()])

            token = match.group(0)
            pos = match.end()
            if token == MESSAGE_MODEL:
                if self._kind in (None, "header"):
                    flush_reasoning()
                    self._pending_header = ""
                    self._kind = "header"
                else:


                    emit(token)
            elif token in (CONTENT_INVOKE_TOOL_JSON, CONTENT_INVOKE_TOOL_TEXT):


                flush_reasoning()
                if self._kind == "header":
                    content.extend((MESSAGE_MODEL, self._pending_header, token))
                    self._pending_header = ""
                else:
                    content.append(token)
                self._kind = "tool"
            elif self._kind == "tool":
                content.append(token)
                if token in _INKLING_END_TOKENS:
                    self._kind = None
            elif token in _INKLING_CONTENT_KINDS:
                flush_reasoning()
                self._pending_header = ""
                self._kind = _INKLING_CONTENT_KINDS[token]
            elif token in _INKLING_END_TOKENS:
                flush_reasoning()
                self._pending_header = ""
                self._kind = None

        tail = text[pos:]
        if saw_control or self._kind is not None:
            emit(tail)
        else:
            content.append(text)

        return StreamingParseResult(
            normal_text="".join(content),
            reasoning_text="".join(reasoning),
        )


class _DeepSeekV3Detector(Qwen3Detector):


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reasoning_default = "explicit_thinking"


class DeepSeekV4Detector(BaseReasoningFormatDetector):
    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
        force_nonempty_content: bool = False,
    ):
        super().__init__(
            dsv4_thinking_start_token,
            dsv4_thinking_end_token,
            think_excluded_tokens=[dsv4_eos_token, dsv4_dsml_token],

            tool_start_token=f"<{dsv4_dsml_token}",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            thinks_internally=True,
            reasoning_default="explicit_thinking",
            force_nonempty_content=force_nonempty_content,
        )


class _MimoDetector(Qwen3Detector):


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reasoning_default = "explicit_enable_thinking"


class _PoolsideV1Detector(Qwen3Detector):


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reasoning_default = "explicit_enable_thinking"


class Apertus2509Detector(BaseReasoningFormatDetector):


    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
        force_nonempty_content: bool = False,
    ):
        super().__init__(
            "<|inner_prefix|>",
            "<|inner_suffix|>",
            force_reasoning=False,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            force_nonempty_content=force_nonempty_content,
        )
        self._force_reasoning = force_reasoning
        self._tool_start_token = "<|tools_prefix|>["
        self._tool_end_token = "<|tools_suffix|>"
        self._reasoning_acc: str = ""
        self._in_inner_tool: bool = False

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        blocks = self.detect_and_parse_block_sequence(text)
        reasoning_parts = [t for k, t in blocks if k == "reasoning"]
        text_parts = [t for k, t in blocks if k == "text"]
        ret = StreamingParseResult(
            normal_text="".join(text_parts),
            reasoning_text="".join(reasoning_parts),
        )
        return self._maybe_apply_force_nonempty_content(ret)

    def detect_and_parse_block_sequence(self, text: str) -> list[tuple[str, str]]:

        start_tok = self.think_start_token
        end_tok = self.think_end_token
        blocks: list[tuple[str, str]] = []
        cursor = 0


        if self._in_reasoning:
            if (e := text.find(end_tok, cursor)) == -1:
                blocks.extend(self._split_inner_reasoning(text[cursor:]))
                blocks.append(("text", ""))
                return blocks
            blocks.extend(self._split_inner_reasoning(text[cursor:e]))
            cursor = e + len(end_tok)

        while True:
            if (s := text.find(start_tok, cursor)) == -1:

                blocks.append(("text", text[cursor:]))
                break
            if s > cursor:
                blocks.append(("text", text[cursor:s]))

            cursor = s + len(start_tok)
            if (e := text.find(end_tok, cursor)) == -1:
                blocks.extend(self._split_inner_reasoning(text[cursor:]))
                blocks.append(("text", ""))
                break
            blocks.extend(self._split_inner_reasoning(text[cursor:e]))
            cursor = e + len(end_tok)

        last_idx = len(blocks) - 1
        blocks = [
            (k, t)
            for i, (k, t) in enumerate(blocks)
            if not (k == "text" and t == "" and i != last_idx)
        ]

        return blocks

    def _split_inner_reasoning(self, inner_text: str) -> list[tuple[str, str]]:

        tool_start = self._tool_start_token
        tool_end = self._tool_end_token
        out: list[tuple[str, str]] = []
        cursor = 0

        while True:
            if (s := inner_text.find(tool_start, cursor)) == -1:
                if (tail := inner_text[cursor:]) != "":
                    out.append(("reasoning", tail))
                break
            if s > cursor:
                out.append(("reasoning", inner_text[cursor:s]))

            if (e := inner_text.find(tool_end, s)) == -1:
                out.append(("text", inner_text[s:]))
                break

            out.append(("text", inner_text[s : e + len(tool_end)]))
            cursor = e + len(tool_end)

        return out

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        self._buffer += new_text

        out_reasoning = ""
        out_normal = ""

        start_tok = self.think_start_token
        end_tok = self.think_end_token
        tool_start = self._tool_start_token
        tool_end = self._tool_end_token

        while True:
            if not self._in_reasoning:
                if (s := self._buffer.find(start_tok)) == -1:
                    if partial := self._ends_with_partial_token(
                        self._buffer, start_tok
                    ):
                        out_normal += self._buffer[:-partial]
                        self._buffer = self._buffer[-partial:]
                    else:
                        out_normal += self._buffer
                        self._buffer = ""
                    return StreamingParseResult(
                        normal_text=out_normal, reasoning_text=out_reasoning
                    )

                out_normal += self._buffer[:s]
                self._buffer = self._buffer[s + len(start_tok) :]
                self._in_reasoning = True
                self._reasoning_acc = ""
                self._in_inner_tool = False
                continue

            if self._in_inner_tool:
                if (end_pos := self._buffer.find(tool_end)) == -1:
                    if (
                        hold := self._ends_with_partial_token(self._buffer, tool_end)
                    ) != 0:
                        out_normal += self._buffer[:-hold]
                        self._buffer = self._buffer[-hold:]
                    else:
                        out_normal += self._buffer
                        self._buffer = ""
                    return StreamingParseResult(
                        normal_text=out_normal, reasoning_text=out_reasoning
                    )

                out_normal += self._buffer[: end_pos + len(tool_end)]
                self._buffer = self._buffer[end_pos + len(tool_end) :]
                self._in_inner_tool = False
                continue

            pos_tool = self._buffer.find(tool_start)
            pos_end = self._buffer.find(end_tok)

            if pos_tool == -1 and pos_end == -1:
                if self.stream_reasoning:
                    if (
                        hold := max(
                            self._ends_with_partial_token(self._buffer, end_tok),
                            self._ends_with_partial_token(self._buffer, tool_start),
                        )
                    ) != 0:
                        out_reasoning += self._buffer[:-hold]
                        self._buffer = self._buffer[-hold:]
                    else:
                        out_reasoning += self._buffer
                        self._buffer = ""
                return StreamingParseResult(
                    normal_text=out_normal, reasoning_text=out_reasoning
                )

            next_pos = min(p for p in [pos_tool, pos_end] if p != -1)

            if pos_end != -1 and pos_end == next_pos:
                reasoning_chunk = self._buffer[:pos_end]
                if self.stream_reasoning:
                    out_reasoning += reasoning_chunk
                else:
                    self._reasoning_acc += reasoning_chunk
                    out_reasoning += self._reasoning_acc
                    self._reasoning_acc = ""
                self._buffer = self._buffer[pos_end + len(end_tok) :]
                self._in_reasoning = False
                continue

            reasoning_chunk = self._buffer[:pos_tool]
            if self.stream_reasoning:
                out_reasoning += reasoning_chunk
            else:
                self._reasoning_acc += reasoning_chunk
            self._buffer = self._buffer[pos_tool:]
            self._in_inner_tool = True
            continue


class CohereCommand4Detector(BaseReasoningFormatDetector):


    TEXT_START_TOKEN = "<|START_TEXT|>"
    TEXT_END_TOKEN = "<|END_TEXT|>"


    ACTION_START_TOKEN = "<|START_ACTION|>"

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = True,
        continue_final_message: bool = False,
        previous_content: str = "",
        force_nonempty_content: bool = False,
    ):


        super().__init__(
            think_start_token="<|START_THINKING|>",
            think_end_token="<|END_THINKING|>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            force_nonempty_content=force_nonempty_content,
        )


        self._reasoning_done = False
        self._saw_text_start = False
        self._saw_text_end = False
        self._in_action_mode = False

    @classmethod
    def _strip_text_markers(cls, raw: str) -> str:

        if not raw:
            return ""
        s = raw.find(cls.TEXT_START_TOKEN)
        if s == -1:
            if cls.ACTION_START_TOKEN in raw:
                return raw
            return ""
        s += len(cls.TEXT_START_TOKEN)
        tail = raw[s:]
        e = tail.find(cls.TEXT_END_TOKEN)
        if e == -1:
            return tail
        return tail[:e]

    def detect_and_parse(self, text: str) -> StreamingParseResult:


        end_think_idx = text.find(self.think_end_token)
        text_start_idx = text.find(self.TEXT_START_TOKEN)
        action_start_idx = text.find(self.ACTION_START_TOKEN)
        if end_think_idx != -1:
            reasoning = text[:end_think_idx]
            rest = text[end_think_idx + len(self.think_end_token) :]
        elif text_start_idx != -1:
            reasoning = text[:text_start_idx]
            rest = text[text_start_idx:]
        elif action_start_idx != -1:


            reasoning = text[:action_start_idx]
            rest = text[action_start_idx:]
        else:
            reasoning = text
            rest = ""


        think_start_text = self.think_start_token + self.think_start_self_label
        if reasoning.startswith(think_start_text):
            reasoning = reasoning[len(think_start_text) :]

        return self._maybe_apply_force_nonempty_content(
            StreamingParseResult(
                normal_text=self._strip_text_markers(rest),
                reasoning_text=reasoning,
            )
        )

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:

        self._buffer += new_text
        buf = self._buffer

        if not self._reasoning_done:


            markers = (
                (self.think_end_token, "think_end"),
                (self.TEXT_START_TOKEN, "text"),
                (self.ACTION_START_TOKEN, "action"),
            )
            first_pos = None
            first_marker = None
            first_kind = None
            for marker_text, kind in markers:
                p = buf.find(marker_text)
                if p != -1 and (first_pos is None or p < first_pos):
                    first_pos, first_marker, first_kind = p, marker_text, kind
            if first_pos is None:


                if not self.stream_reasoning:
                    return StreamingParseResult()
                max_keep = max(len(m) for m, _ in markers) - 1
                if len(buf) > max_keep:
                    head = buf[:-max_keep]
                    self._buffer = buf[-max_keep:]
                    return StreamingParseResult(reasoning_text=head)
                return StreamingParseResult()

            reasoning_chunk = buf[:first_pos]
            if first_kind == "think_end":
                self._buffer = buf[first_pos + len(first_marker) :]
            else:


                self._buffer = buf[first_pos:]
            self._reasoning_done = True
            if reasoning_chunk:
                return StreamingParseResult(reasoning_text=reasoning_chunk)
            buf = self._buffer


        if self._in_action_mode:
            if not buf:
                return StreamingParseResult()
            self._buffer = ""
            return StreamingParseResult(normal_text=buf)

        if not self._saw_text_start:
            s_text = buf.find(self.TEXT_START_TOKEN)
            s_action = buf.find(self.ACTION_START_TOKEN)
            picks = [
                (p, k) for p, k in ((s_text, "text"), (s_action, "action")) if p != -1
            ]
            if not picks:
                max_keep = (
                    max(len(self.TEXT_START_TOKEN), len(self.ACTION_START_TOKEN)) - 1
                )
                if len(buf) > max_keep:
                    self._buffer = buf[-max_keep:]
                return StreamingParseResult()
            picks.sort()
            first_pos, first_kind = picks[0]
            if first_kind == "action":
                self._in_action_mode = True
                out_normal = buf[first_pos:]
                self._buffer = ""
                return StreamingParseResult(normal_text=out_normal)


            self._buffer = buf[first_pos + len(self.TEXT_START_TOKEN) :]
            self._saw_text_start = True
            buf = self._buffer

        if self._saw_text_start and not self._saw_text_end:
            e = buf.find(self.TEXT_END_TOKEN)
            if e == -1:

                keep = len(self.TEXT_END_TOKEN) - 1
                if len(buf) > keep:
                    out_normal = buf[:-keep]
                    self._buffer = buf[-keep:]
                    return StreamingParseResult(normal_text=out_normal)
                return StreamingParseResult()
            out_normal = buf[:e]
            self._buffer = buf[e + len(self.TEXT_END_TOKEN) :]
            self._saw_text_end = True
            return StreamingParseResult(normal_text=out_normal)

        return StreamingParseResult()

    def finish(self) -> StreamingParseResult:


        buffer = self._buffer
        self._buffer = ""
        if not self._reasoning_done:
            ret = StreamingParseResult(
                reasoning_text=self._strip_leading_think_start(buffer)
            )
        elif self._saw_text_start and not self._saw_text_end:
            ret = StreamingParseResult(normal_text=buffer)
        else:
            return StreamingParseResult()
        return self._maybe_apply_force_nonempty_content(ret)


class MuseGlimmerDetector(BaseReasoningFormatDetector):


    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
        force_nonempty_content: bool = False,
        tool_call_parser_active: bool = False,
    ):
        super().__init__(
            " to=self" + MESSAGE,
            EOM,
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            force_nonempty_content=force_nonempty_content,
        )
        self._recipient: Optional[str] = None
        self._in_body = False
        self._at_stream_start = True
        self._pending_reasoning = ""
        self._tool_call_parser_active = tool_call_parser_active
        self._saw_reasoning_block = False

    def _sink(self, recipient: Optional[str]) -> str:
        return "reasoning" if recipient == "self" else "normal"

    def _consume(self, flush: bool, preserve_channels: bool = False) -> Tuple[str, str]:

        reasoning_parts: List[str] = []
        normal_parts: List[str] = []

        while self._buffer:
            if not self._in_body:


                if not (self._at_stream_start and could_start_header(self._buffer)):
                    ws = len(self._buffer) - len(self._buffer.lstrip())
                    head = self._buffer[ws : ws + len(START)]
                    if not START.startswith(head):
                        self._in_body = True
                        self._recipient = None
                        self._at_stream_start = False
                        continue
                    if ws:
                        normal_parts.append(self._buffer[:ws])
                        self._buffer = self._buffer[ws:]
                    if len(head) < len(START):
                        break

                idx = self._buffer.find(MESSAGE)
                if idx == -1:
                    if flush:
                        normal_parts.append(self._buffer)
                        self._buffer = ""
                    break
                self._at_stream_start = False
                header = self._buffer[:idx]
                m = RECIPIENT_RE.search(header)
                self._recipient = m.group(1) if m else "user"
                self._buffer = self._buffer[idx + len(MESSAGE) :]
                self._in_body = True
                if self._sink(self._recipient) == "reasoning":
                    if self._saw_reasoning_block:
                        reasoning_parts.append("\n")
                    self._saw_reasoning_block = True
                elif self._recipient != "user" or preserve_channels:

                    normal_parts.append(header + MESSAGE)
                continue

            end_idx, end_tok = -1, ""
            for tok in (EOM, EOT):
                i = self._buffer.find(tok)
                if i != -1 and (end_idx == -1 or i < end_idx):
                    end_idx, end_tok = i, tok

            if end_idx != -1:
                body = self._buffer[:end_idx]
                self._buffer = self._buffer[end_idx + len(end_tok) :]
                self._in_body = False
                if self._sink(self._recipient) == "reasoning":
                    reasoning_parts.append(body)
                else:
                    normal_parts.append(body)
                    if self._recipient != "user" or preserve_channels:
                        normal_parts.append(end_tok)
                self._recipient = None
                continue


            if flush:
                body, self._buffer = self._buffer, ""
            else:
                keep = partial_marker_len(
                    self._buffer, (EOM, EOT, START), MAX_CHANNEL_MARKER
                )
                if keep == len(self._buffer):
                    break
                body = self._buffer[: len(self._buffer) - keep]
                self._buffer = self._buffer[len(self._buffer) - keep :]
            if not body:
                break
            if self._sink(self._recipient) == "reasoning":
                reasoning_parts.append(body)
            else:
                normal_parts.append(body)

        return "".join(reasoning_parts), "".join(normal_parts)

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        self._buffer += text
        raw = self._buffer
        reasoning, normal = self._consume(flush=True)
        if self._tool_call_parser_active and has_atem_markers(normal):
            self._buffer = raw
            self._recipient = None
            self._in_body = False
            self._at_stream_start = True
            self._saw_reasoning_block = False
            reasoning, normal = self._consume(flush=True, preserve_channels=True)
        return self._maybe_apply_force_nonempty_content(
            StreamingParseResult(normal_text=normal, reasoning_text=reasoning)
        )

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        self._buffer += new_text
        reasoning, normal = self._consume(
            flush=False, preserve_channels=self._tool_call_parser_active
        )
        if not self.stream_reasoning:
            self._pending_reasoning += reasoning
            reasoning = ""
            if not self._in_body and self._pending_reasoning:
                reasoning, self._pending_reasoning = self._pending_reasoning, ""
        if self._force_nonempty_content:


            self._accumulated_reasoning += reasoning
            if normal:
                self._accumulated_reasoning = ""
        return StreamingParseResult(normal_text=normal, reasoning_text=reasoning)

    def finish(self) -> StreamingParseResult:
        reasoning, normal = self._consume(
            flush=True, preserve_channels=self._tool_call_parser_active
        )
        if self._pending_reasoning:
            reasoning = self._pending_reasoning + reasoning
            self._pending_reasoning = ""
        if self._force_nonempty_content:
            promoted = self._accumulated_reasoning + reasoning
            self._accumulated_reasoning = ""
            if not normal and promoted:
                return StreamingParseResult(normal_text=promoted)
        return StreamingParseResult(normal_text=normal, reasoning_text=reasoning)


class ReasoningParser:


    DetectorMap: Dict[str, Type[BaseReasoningFormatDetector]] = {
        "apertus2509": Apertus2509Detector,
        "deepseek-r1": DeepSeekR1Detector,
        "deepseek-v3": _DeepSeekV3Detector,
        "deepseek-v4": DeepSeekV4Detector,
        "dots": Qwen3Detector,
        "glm45": Glm45Detector,
        "hunyuan": HunyuanDetector,
        "gpt-oss": GptOssDetector,
        "kimi": KimiDetector,
        "kimi_k2": KimiK2Detector,
        "kimi_k3": KimiK3Detector,
        "mimo": _MimoDetector,
        "muse": MuseGlimmerDetector,
        "poolside_v1": _PoolsideV1Detector,
        "qwen3": Qwen3Detector,
        "qwen3-thinking": Qwen3Detector,
        "minimax": Qwen3Detector,
        "minimax-append-think": MiniMaxAppendThinkDetector,
        "minimax-m3": MiniMaxM3Detector,
        "step3": DeepSeekR1Detector,
        "step3p5": DeepSeekR1Detector,
        "mistral": MistralDetector,
        "nemotron_3": Nemotron3Detector,
        "interns1": Qwen3Detector,
        "gemma4": Gemma4Detector,
        "inkling": InklingDetector,
        "cohere_command4": CohereCommand4Detector,
    }

    def __init__(
        self,
        model_type: Optional[str] = None,
        stream_reasoning: bool = True,
        force_reasoning: Optional[bool] = None,
        request: ChatCompletionRequest = None,
        tokenizer=None,
        tool_call_parser_active: bool = False,
    ):
        if not model_type:
            raise ValueError("Model type must be specified")

        detector_class = self.DetectorMap.get(model_type.lower())
        if not detector_class:
            raise ValueError(f"Unsupported model type: {model_type}")

        chat_template_kwargs = getattr(request, "chat_template_kwargs", None) or {}


        if model_type.lower() in {
            "qwen3-thinking",
            "gpt-oss",
            "minimax",
        }:
            force_reasoning = True


        if model_type.lower() == "minimax-m3" and force_reasoning is None:
            force_reasoning = chat_template_kwargs.get("thinking_mode") == "enabled"


        kwargs = {"stream_reasoning": stream_reasoning}
        if force_reasoning is not None:
            kwargs["force_reasoning"] = force_reasoning

        if (
            request is not None
            and isinstance(request, ChatCompletionRequest)
            and request.continue_final_message
            and request.messages[-1].role == "assistant"
        ):
            kwargs["continue_final_message"] = True
            kwargs["previous_content"] = request.messages[-1].content

        if chat_template_kwargs.get("force_nonempty_content") is True:
            kwargs["force_nonempty_content"] = True

        if tokenizer is not None:
            sig = inspect.signature(detector_class)
            if "tokenizer" in sig.parameters:
                kwargs["tokenizer"] = tokenizer

        if tool_call_parser_active:
            sig = inspect.signature(detector_class)
            if "tool_call_parser_active" in sig.parameters:
                kwargs["tool_call_parser_active"] = True

        self.detector = detector_class(**kwargs)

    def parse_non_stream(self, full_text: str) -> Tuple[Optional[str], Optional[str]]:

        ret = self.detector.detect_and_parse(full_text)
        return ret.reasoning_text, ret.normal_text

    def parse_non_stream_blocks(self, full_text: str) -> list[dict]:

        if hasattr(self.detector, "detect_and_parse_block_sequence"):
            seq = self.detector.detect_and_parse_block_sequence(full_text)
            return [{"type": k, "text": t} for k, t in seq]

        ret = self.detector.detect_and_parse(full_text)
        blocks: list[dict] = []
        if ret.reasoning_text:
            blocks.append({"type": "reasoning", "text": ret.reasoning_text})
        blocks.append({"type": "text", "text": ret.normal_text or ""})
        return blocks

    def parse_stream_chunk(
        self, chunk_text: str
    ) -> Tuple[Optional[str], Optional[str]]:

        ret = self.detector.parse_streaming_increment(chunk_text)
        return ret.reasoning_text, ret.normal_text

    def parse_stream_end(self) -> Tuple[Optional[str], Optional[str]]:

        ret = self.detector.finish()
        return ret.reasoning_text, ret.normal_text
