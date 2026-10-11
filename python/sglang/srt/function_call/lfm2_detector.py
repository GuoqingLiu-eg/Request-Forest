import ast
import json
import keyword as _python_keyword
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import safe_ast_parse

logger = logging.getLogger(__name__)


_PYTHONIC_NAME_LITERALS = {
    "True": True,
    "False": False,
    "None": None,
    "true": True,
    "false": False,
    "null": None,
}

_QUOTE_FOLLOWERS = {",", ")", "]", "}", ":"}
_RESERVED_KW_SUFFIX = "_pyreservedkw_"


def _rename_reserved_kwargs(text: str) -> Tuple[str, bool]:

    out: List[str] = []
    quote: Optional[str] = None
    changed = False
    last_sig = ""
    index, length = 0, len(text)
    while index < length:
        char = text[index]
        if quote is not None:
            out.append(char)
            if char == "\\" and index + 1 < length:
                out.append(text[index + 1])
                index += 2
                continue
            if char == quote:
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            out.append(char)
            last_sig = char
            index += 1
            continue
        if char.isalpha() or char == "_":
            end = index
            while end < length and (text[end].isalnum() or text[end] == "_"):
                end += 1
            name = text[index:end]
            look = end
            while look < length and text[look].isspace():
                look += 1
            if (
                _python_keyword.iskeyword(name)
                and look < length
                and text[look] == "="
                and (look + 1 >= length or text[look + 1] != "=")
                and last_sig in {"(", ","}
            ):
                out.append(name + _RESERVED_KW_SUFFIX)
                changed = True
            else:
                out.append(name)
            last_sig = name[-1]
            index = end
            continue
        out.append(char)
        if not char.isspace():
            last_sig = char
        index += 1
    return "".join(out), changed


def _restore_reserved_kwarg_names(arguments: dict) -> dict:

    restored = {}
    for key, value in arguments.items():
        if (
            isinstance(key, str)
            and key.endswith(_RESERVED_KW_SUFFIX)
            and _python_keyword.iskeyword(key[: -len(_RESERVED_KW_SUFFIX)])
        ):
            restored[key[: -len(_RESERVED_KW_SUFFIX)]] = value
        else:
            restored[key] = value
    return restored


def _is_escaped(text: str, index: int) -> bool:

    backslashes = 0
    j = index - 1
    while j >= 0 and text[j] == "\\":
        backslashes += 1
        j -= 1
    return backslashes % 2 == 1


def _escape_nested_quotes_in_strings(text: str) -> Tuple[str, bool]:


    def unescaped_quotes(start: int, quote: str) -> List[int]:
        positions = []
        j = start
        while j < len(text):
            if text[j] == "\\":
                j += 2
                continue
            if text[j] == quote:
                positions.append(j)
            j += 1
        return positions

    def is_closer(pos: int) -> bool:
        k = pos + 1
        while k < len(text) and text[k].isspace():
            k += 1
        return k < len(text) and text[k] in _QUOTE_FOLLOWERS


    expected_calls = len(_split_top_level_calls(text, respect_strings=False))
    prefix: List[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char not in {"'", '"'}:
            prefix.append(char)
            index += 1
            continue
        quotes = unescaped_quotes(index + 1, char)
        if not quotes:
            return text, False
        if is_closer(quotes[0]):
            prefix.append(text[index : quotes[0] + 1])
            index = quotes[0] + 1
            continue
        winners = []
        for close in (j for j in quotes if is_closer(j)):
            interior: List[str] = []
            for j in range(index + 1, close):
                if text[j] == char and not _is_escaped(text, j):
                    interior.append("\\")
                interior.append(text[j])
            candidate = "".join(
                ["".join(prefix), char, "".join(interior), char, text[close + 1 :]]
            )
            try:
                module = safe_ast_parse(_escape_ctrl_chars_in_strings(candidate))
            except (SyntaxError, ValueError):
                continue
            if expected_calls > 1 and _top_level_call_count(module) < expected_calls:
                continue
            winners.append(candidate)
        if len(winners) == 1:
            return winners[0], True
        return text, False
    return text, False


def _escape_ctrl_chars_in_strings(text: str) -> str:

    out: List[str] = []
    quote: Optional[str] = None
    index, length = 0, len(text)
    while index < length:
        char = text[index]
        if quote is None:
            if char in {"'", '"'}:
                quote = char
            out.append(char)
        elif char == "\\" and index + 1 < length:
            out.append(char)
            out.append(text[index + 1])
            index += 2
            continue
        elif char == quote:
            quote = None
            out.append(char)
        elif char == "\n":
            out.append("\\n")
        elif char == "\r":
            out.append("\\r")
        elif char == "\t":
            out.append("\\t")
        elif char == "\x00":
            out.append("\\x00")
        else:
            out.append(char)
        index += 1
    return "".join(out)


def _normalize_leading_zero_ints(text: str) -> str:

    out: List[str] = []
    quote: Optional[str] = None
    index, length = 0, len(text)
    while index < length:
        char = text[index]
        if quote is not None:
            out.append(char)
            if char == "\\" and index + 1 < length:
                out.append(text[index + 1])
                index += 2
                continue
            if char == quote:
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            out.append(char)
            index += 1
            continue
        if char.isalpha() or char == "_":
            end = index
            while end < length and (text[end].isalnum() or text[end] == "_"):
                end += 1
            out.append(text[index:end])
            index = end
            continue
        if char.isdigit():
            end = index
            while end < length and (text[end].isdigit() or text[end] == "_"):
                end += 1
            token = text[index:end]
            digits = token.replace("_", "")
            follower = text[end] if end < length else ""
            preceded_by_dot = index > 0 and text[index - 1] == "."
            if (
                digits[0] == "0"
                and digits.strip("0")
                and not preceded_by_dot
                and follower not in {".", "e", "E", "j", "J"}
            ):
                out.append(str(int(digits)))
            else:
                out.append(token)
            index = end
            continue
        out.append(char)
        index += 1
    return "".join(out)


def _recovery_candidates(content: str) -> List[Tuple[str, bool]]:

    escaped = _escape_ctrl_chars_in_strings(_normalize_leading_zero_ints(content))
    candidates: List[Tuple[str, bool]] = [(escaped, False)]
    requoted, requote_changed = _escape_nested_quotes_in_strings(escaped)
    if requote_changed:
        candidates.append((_escape_ctrl_chars_in_strings(requoted), False))
    for text, _ in list(candidates):
        renamed, kw_renamed = _rename_reserved_kwargs(text)
        if kw_renamed:
            candidates.append((renamed, True))


            requoted_after, requote_after_changed = _escape_nested_quotes_in_strings(
                renamed
            )
            if requote_after_changed:
                candidates.append((_escape_ctrl_chars_in_strings(requoted_after), True))
    return candidates


def _split_top_level_calls(text: str, *, respect_strings: bool = True) -> List[str]:

    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    segments: List[str] = []
    start = 0
    depth = 0
    quote: Optional[str] = None
    index = 0
    while index < len(text):
        char = text[index]
        if respect_strings and quote is not None:
            if char == "\\":
                index += 2
                continue
            if char == quote:
                quote = None
            index += 1
            continue
        if respect_strings and char in {"'", '"'}:
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            segments.append(text[start:index])
            start = index + 1
        index += 1
    segments.append(text[start:])
    return [segment.strip() for segment in segments if segment.strip()]


def _top_level_call_count(module: ast.Module) -> int:

    if not module.body:
        return 0
    value = getattr(module.body[0], "value", None)
    if isinstance(value, ast.List):
        return sum(1 for element in value.elts if isinstance(element, ast.Call))
    return 1 if isinstance(value, ast.Call) else 0


def _salvage_calls_from_unparsable_block(text: str) -> List[Tuple[ast.Call, bool]]:

    best: List[Tuple[ast.Call, bool]] = []
    for respect_strings in (True, False):
        segments = _split_top_level_calls(text, respect_strings=respect_strings)
        if len(segments) < 2:
            continue
        calls: List[Tuple[ast.Call, bool]] = []
        for segment in segments:
            for candidate, kw_renamed in [(segment, False)] + _recovery_candidates(
                segment
            ):
                try:
                    module = safe_ast_parse(candidate)
                except (SyntaxError, ValueError):
                    continue
                parsed = getattr(module.body[0], "value", None) if module.body else None
                if isinstance(parsed, ast.Call):
                    calls.append((parsed, kw_renamed))
                break
        if len(calls) > len(best):
            best = calls
    return best


class Lfm2Detector(BaseFormatDetector):


    def __init__(self):

        super().__init__()
        self.bot_token = "<|tool_call_start|>"
        self.eot_token = "<|tool_call_end|>"
        self.tool_call_separator = ""

    def has_tool_call(self, text: str) -> bool:

        return self.bot_token in text

    def _get_parameter_value(self, val: ast.AST) -> Any:

        if isinstance(val, ast.Constant):
            if val.value is None or isinstance(val.value, (str, int, float)):
                return val.value


            raise ValueError(
                f"Constant has no JSON representation: {type(val.value).__name__}"
            )
        elif isinstance(val, ast.Dict):
            return {
                self._get_parameter_value(k): self._get_parameter_value(v)
                for k, v in zip(val.keys, val.values)
                if k is not None
            }
        elif isinstance(val, ast.List):
            return [self._get_parameter_value(v) for v in val.elts]
        elif isinstance(val, ast.Tuple):
            return tuple(self._get_parameter_value(v) for v in val.elts)
        elif isinstance(val, ast.Set):

            return [self._get_parameter_value(v) for v in val.elts]
        elif isinstance(val, ast.JoinedStr) and all(
            isinstance(part, ast.Constant) for part in val.values
        ):


            return "".join(str(part.value) for part in val.values)
        elif isinstance(val, ast.Name):


            try:
                return _PYTHONIC_NAME_LITERALS[val.id]
            except KeyError:
                raise ValueError(f"Unsupported name reference: {val.id}") from None
        elif isinstance(val, ast.UnaryOp) and isinstance(val.op, (ast.USub, ast.UAdd)):

            inner = self._get_parameter_value(val.operand)
            if isinstance(inner, (int, float)) and not isinstance(inner, bool):
                return -inner if isinstance(val.op, ast.USub) else inner
            raise ValueError(f"Cannot apply sign to non-numeric value: {inner}")
        else:
            raise ValueError(
                f"Tool call arguments must be literals, got: {type(val).__name__}"
            )

    def _get_function_name(self, func: ast.AST) -> Optional[str]:

        parts: List[str] = []
        while isinstance(func, ast.Attribute):
            parts.append(func.attr)
            func = func.value

        if not isinstance(func, ast.Name):
            return None

        parts.append(func.id)
        return ".".join(reversed(parts))

    def _parse_pythonic_call(
        self,
        call: ast.Call,
        call_index: int,
        tool_indices: Dict[str, int],
        *,
        restore_reserved_kwarg: bool = False,
    ) -> Optional[ToolCallItem]:

        function_name = self._get_function_name(call.func)
        if function_name is None:
            logger.warning(
                f"Tool call function must be a name or dotted name, got: {type(call.func).__name__}"
            )
            return None


        if function_name not in tool_indices:
            logger.warning(
                f"Model attempted to call undefined function: {function_name}"
            )
            if not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get():
                return None

        if call.args:


            logger.warning(f"Tool call {function_name} has positional arguments")
            return None


        arguments = {}
        for keyword in call.keywords:
            if keyword.arg is None:


                try:
                    unpacked = self._get_parameter_value(keyword.value)
                except ValueError as e:
                    logger.warning(f"Failed to parse **-unpacked arguments: {e}")
                    return None
                if not isinstance(unpacked, dict):
                    logger.warning("**-unpacked arguments must be a dict literal")
                    return None
                arguments.update(unpacked)
                continue
            try:
                arguments[keyword.arg] = self._get_parameter_value(keyword.value)
            except ValueError as e:
                logger.warning(f"Failed to parse argument {keyword.arg}: {e}")
                return None

        if restore_reserved_kwarg:


            arguments = _restore_reserved_kwarg_names(arguments)

        try:


            parameters = json.dumps(arguments, ensure_ascii=False, allow_nan=False)
        except (ValueError, TypeError) as e:
            logger.warning(f"Arguments of {function_name} are not valid JSON: {e}")
            return None

        return ToolCallItem(
            tool_index=call_index,
            name=function_name,
            parameters=parameters,
        )

    def _parse_pythonic_content(
        self, content: str, tools: List[Tool]
    ) -> Tuple[List[ToolCallItem], str]:

        content = content.strip()
        tool_indices = self._get_tool_indices(tools)

        try:
            kw_renamed = False
            try:
                module = safe_ast_parse(content)
            except (SyntaxError, ValueError):


                for candidate, kw_renamed in _recovery_candidates(content):
                    try:
                        module = safe_ast_parse(candidate)
                        break
                    except (SyntaxError, ValueError):
                        continue
                else:


                    salvaged = _salvage_calls_from_unparsable_block(content)
                    if not salvaged:
                        raise
                    calls = []
                    for call_index, (call, segment_kw_renamed) in enumerate(salvaged):
                        item = self._parse_pythonic_call(
                            call,
                            call_index,
                            tool_indices,
                            restore_reserved_kwarg=segment_kw_renamed,
                        )
                        if item is not None:
                            calls.append(item)
                    return calls, ""
            parsed = getattr(module.body[0], "value", None) if module.body else None

            if parsed is None:
                return [], "Empty or invalid Python expression"


            if isinstance(parsed, ast.List):
                call_nodes = parsed.elts
            elif isinstance(parsed, ast.Call):
                call_nodes = [parsed]
            else:
                return (
                    [],
                    f"Expected function call or list, got: {type(parsed).__name__}",
                )


            if not all(isinstance(e, ast.Call) for e in call_nodes):
                return [], "Not all elements in list are function calls"

            calls = []
            for call_index, call in enumerate(call_nodes):
                item = self._parse_pythonic_call(
                    call,
                    call_index,
                    tool_indices,
                    restore_reserved_kwarg=kw_renamed,
                )
                if item is not None:
                    calls.append(item)

            return calls, ""

        except (SyntaxError, ValueError) as e:
            return [], f"Python syntax error: {e}"
        except Exception as e:
            logger.exception("Unexpected error in pythonic tool call parsing")
            return [], f"Unexpected error: {e}"

    def _parse_json_content(
        self, content: str, tools: List[Tool]
    ) -> Tuple[List[ToolCallItem], str]:

        content = content.strip()

        try:
            parsed = json.loads(content)


            calls = self.parse_base_json(parsed, tools)
            return calls, ""

        except json.JSONDecodeError as e:
            return [], f"JSON parse error: {e}"

    def _parse_tool_calls_content(
        self, content: str, tools: List[Tool]
    ) -> List[ToolCallItem]:

        content = content.strip()


        if content.startswith("[{") or content.startswith("{"):
            calls, error = self._parse_json_content(content, tools)
            if calls:
                return calls

            if error:
                logger.debug(f"JSON parsing failed: {error}, trying Pythonic format")


        calls, error = self._parse_pythonic_content(content, tools)
        if calls:
            return calls

        if error:
            logger.warning(f"Failed to parse tool calls: {error}")

        return []

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:

        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text

        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])


        pattern = rf"{re.escape(self.bot_token)}(.*?){re.escape(self.eot_token)}"
        match_result_list = re.findall(pattern, text, re.DOTALL)

        calls = []
        for match_result in match_result_list:
            parsed_calls = self._parse_tool_calls_content(match_result, tools)
            calls.extend(parsed_calls)

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def _strip_special_tokens(self, text: str) -> str:

        return text.replace(self.bot_token, "").replace(self.eot_token, "")

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:

        self._buffer += new_text


        partial_bot = self._ends_with_partial_token(self._buffer, self.bot_token)
        partial_eot = self._ends_with_partial_token(self._buffer, self.eot_token)


        bot_pos = self._buffer.find(self.bot_token)

        if bot_pos == -1:

            if partial_bot:

                safe_text = self._buffer[:-partial_bot]
                self._buffer = self._buffer[-partial_bot:]
                return StreamingParseResult(normal_text=safe_text)
            else:

                normal_text = self._strip_special_tokens(self._buffer)
                self._buffer = ""
                return StreamingParseResult(normal_text=normal_text)


        normal_text_before = self._buffer[:bot_pos] if bot_pos > 0 else ""


        eot_pos = self._buffer.find(self.eot_token, bot_pos + len(self.bot_token))

        if eot_pos == -1:

            if partial_eot:


                if normal_text_before:
                    self._buffer = self._buffer[bot_pos:]
                    return StreamingParseResult(normal_text=normal_text_before)

                return StreamingParseResult(normal_text="")


            if normal_text_before:
                self._buffer = self._buffer[bot_pos:]
                return StreamingParseResult(normal_text=normal_text_before)


            return StreamingParseResult(normal_text="")


        tool_call_block = self._buffer[bot_pos : eot_pos + len(self.eot_token)]
        remaining = self._buffer[eot_pos + len(self.eot_token) :]


        result = self.detect_and_parse(tool_call_block, tools)


        self._buffer = remaining


        if normal_text_before:
            result.normal_text = normal_text_before + (result.normal_text or "")

        return result

    def supports_structural_tag(self) -> bool:

        return False

    def structure_info(self) -> _GetInfoFunc:

        return lambda name: StructureInfo(
            begin="<|tool_call_start|>[" + name + "(",
            end=")]<|tool_call_end|>",
            trigger="<|tool_call_start|>",
        )
