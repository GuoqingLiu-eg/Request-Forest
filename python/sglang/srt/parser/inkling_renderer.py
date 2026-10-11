from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from sglang.srt.parser.inkling_tokenizer import (
    AUDIO_END,
    AUDIO_TOKEN_ID,
    CONTENT_AUDIO_INPUT,
    CONTENT_IMAGE,
    CONTENT_INVOKE_TOOL_JSON,
    CONTENT_MODEL_END_SAMPLING,
    CONTENT_TEXT,
    CONTENT_THINKING,
    CONTENT_XML,
    END_MESSAGE,
    IMAGE_TOKEN_ID,
    MESSAGE_MODEL,
    ROLE_MESSAGE_TOKENS,
)


class InklingTextTokenizer(Protocol):
    def encode_text(self, text: str) -> list[int]: ...

    def encode_special(self, token: str) -> int: ...


_IMAGE_PART_TYPES = frozenset({"image", "input_image", "image_url"})
_AUDIO_PART_TYPES = frozenset({"audio", "input_audio", "audio_url"})
INKLING_DEFAULT_REASONING_EFFORT = 0.9


def render_inkling_messages(
    messages: Sequence[Mapping[str, Any]],
    tokenizer: InklingTextTokenizer,
    *,
    add_generation_prompt: bool = False,
    tools: Sequence[Mapping[str, Any]] | None = None,
    reasoning_effort: float | None = None,
) -> list[int]:

    input_ids: list[int] = []
    tool_call_id_to_name: dict[str, str] = {}

    if tools:
        _append_message(
            input_ids,
            tokenizer,
            "system",
            "xml",
            _tool_declare_json(tools),
            author_name="tool_declare",
        )


    message_list = [
        {**message, "role": "system"} if message.get("role") == "developer" else message
        for message in messages
    ]
    leading_system_count = 0
    for message in message_list:
        if message.get("role") != "system":
            break
        leading_system_count += 1

    def append_effort() -> None:
        effort = (
            INKLING_DEFAULT_REASONING_EFFORT
            if reasoning_effort is None
            else reasoning_effort
        )
        _append_message(
            input_ids,
            tokenizer,
            "system",
            "text",
            f"Thinking effort level: {_format_reasoning_effort(effort)}",
        )

    for message_index, message in enumerate(message_list):
        if message_index == leading_system_count:
            append_effort()
        role = _expect_role(message)
        if role == "tool":
            tool_name = str(
                message.get("name")
                or tool_call_id_to_name.get(message.get("tool_call_id") or "", "")
            )


            tool_parts = list(_iter_render_parts(message.get("content", "")))
            if not tool_parts:
                tool_parts = [("text", "")]
            for kind, text in tool_parts:
                if kind == "thinking":
                    raise ValueError("Inkling thinking parts require role='assistant'")
                _append_message(
                    input_ids,
                    tokenizer,
                    "tool",
                    kind,
                    text,
                    author_name=tool_name,
                )
            continue

        parts = list(_iter_render_parts(message.get("content", "")))
        turn_start = len(input_ids)
        if role == "assistant":
            reasoning_content = message.get("reasoning_content")
            if reasoning_content:
                if not isinstance(reasoning_content, str):
                    raise TypeError(
                        "assistant reasoning_content must be a string for Inkling rendering"
                    )
                if any(kind == "thinking" for kind, _ in parts):
                    raise ValueError(
                        "assistant message cannot mix reasoning_content with ordered thinking parts"
                    )
                _append_message(
                    input_ids,
                    tokenizer,
                    "assistant",
                    "thinking",
                    reasoning_content,
                )

        for kind, text in parts:
            if kind == "thinking" and role != "assistant":
                raise ValueError("Inkling thinking parts require role='assistant'")
            _append_message(input_ids, tokenizer, role, kind, text)

        if role == "assistant":
            for tool_call in message.get("tool_calls") or []:
                name, args = _tool_call_name_and_args(tool_call)
                tool_call_id = _as_mapping(tool_call).get("id")
                if tool_call_id:
                    tool_call_id_to_name[str(tool_call_id)] = name
                _append_message(
                    input_ids,
                    tokenizer,
                    "assistant",
                    "invoke_tool_json",
                    _tool_call_json(name, args),
                    author_name=name,
                )
            if len(input_ids) > turn_start:


                input_ids.append(tokenizer.encode_special(CONTENT_MODEL_END_SAMPLING))

    if leading_system_count == len(message_list):
        append_effort()

    if add_generation_prompt:
        input_ids.append(tokenizer.encode_special(MESSAGE_MODEL))
    return input_ids


def _append_message(
    input_ids: list[int],
    tokenizer: InklingTextTokenizer,
    role: str,
    kind: str,
    text: str,
    *,
    author_name: str | None = None,
) -> None:
    input_ids.append(tokenizer.encode_special(ROLE_MESSAGE_TOKENS[role]))
    if author_name:
        input_ids.extend(tokenizer.encode_text(author_name))

    if kind == "text":
        input_ids.append(tokenizer.encode_special(CONTENT_TEXT))
        input_ids.extend(tokenizer.encode_text(text))
    elif kind == "image":
        input_ids.append(tokenizer.encode_special(CONTENT_IMAGE))
        input_ids.append(IMAGE_TOKEN_ID)
    elif kind == "audio":
        input_ids.append(tokenizer.encode_special(CONTENT_AUDIO_INPUT))
        input_ids.append(AUDIO_TOKEN_ID)
        input_ids.append(tokenizer.encode_special(AUDIO_END))
    elif kind == "thinking":
        input_ids.append(tokenizer.encode_special(CONTENT_THINKING))
        input_ids.extend(tokenizer.encode_text(text))
    elif kind == "xml":
        input_ids.append(tokenizer.encode_special(CONTENT_XML))
        input_ids.extend(tokenizer.encode_text(text))
    elif kind == "invoke_tool_json":
        input_ids.append(tokenizer.encode_special(CONTENT_INVOKE_TOOL_JSON))
        input_ids.extend(tokenizer.encode_text(text))
    else:
        raise ValueError(f"unsupported Inkling render part kind: {kind!r}")

    input_ids.append(tokenizer.encode_special(END_MESSAGE))


def _iter_render_parts(content: Any):

    if content is None:
        return
    if isinstance(content, str):
        if content:
            yield ("text", content)
        return
    if not isinstance(content, Sequence) or isinstance(content, (bytes, bytearray)):
        raise TypeError("message content must be a string or a sequence of parts")
    for part in content:
        if isinstance(part, str):
            yield ("text", part)
            continue
        if not isinstance(part, Mapping):
            raise TypeError(f"content part must be mapping, got {type(part).__name__}")
        ptype = part.get("type")
        if ptype in (None, "text", "input_text"):
            text = part.get("text", "")
            yield ("text", text if isinstance(text, str) else "")
        elif ptype in ("thinking", "reasoning"):
            text = part.get("thinking")
            if text is None:
                text = part.get("text", "")
            if not isinstance(text, str):
                raise TypeError("Inkling thinking part payload must be a string")
            yield ("thinking", text)
        elif ptype in _IMAGE_PART_TYPES:
            yield ("image", "")
        elif ptype in _AUDIO_PART_TYPES:
            yield ("audio", "")
        else:
            raise ValueError(f"unsupported content part type: {ptype!r}")


def _format_reasoning_effort(reasoning_effort: float) -> str:
    if isinstance(reasoning_effort, bool) or not isinstance(
        reasoning_effort, (int, float)
    ):
        raise TypeError("Inkling reasoning_effort must be a number")
    value = float(reasoning_effort)
    if not math.isfinite(value) or not 0.0 <= value <= 0.99:
        raise ValueError("Inkling reasoning_effort must be finite and in [0.0, 0.99]")
    return f"{round(value, 2):g}"


def _expect_role(message: Mapping[str, Any]) -> str:
    role = message.get("role")
    if role not in ROLE_MESSAGE_TOKENS:
        raise ValueError(
            f"unsupported Inkling message role {role!r}; expected one of {sorted(ROLE_MESSAGE_TOKENS)}"
        )
    return str(role)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    raise TypeError(f"expected mapping, got {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _sort_json(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _sort_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sort_json(value[key]) for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_sort_json(item) for item in value]
    return value


def _tool_declare_json(tools: Sequence[Mapping[str, Any]]) -> str:
    tool_specs = []
    for tool_value in tools:
        tool = _as_mapping(tool_value)
        function = _as_mapping(tool.get("function", {}))
        tool_specs.append(
            {
                "description": function.get("description") or "",
                "name": function["name"],
                "parameters": function.get("parameters") or {},
                "type": tool.get("type", "function"),
            }
        )
    return _canonical_json(tool_specs)


def _tool_call_name_and_args(tool_call_value: Any) -> tuple[str, Mapping[str, Any]]:
    tool_call = _as_mapping(tool_call_value)
    function = _as_mapping(tool_call.get("function", {}))
    name = function.get("name")
    if not isinstance(name, str):
        raise TypeError("tool call function name must be a string")

    raw_args = function.get("arguments") or {}
    if isinstance(raw_args, str):
        args = json.loads(raw_args) if raw_args else {}
    else:
        args = raw_args
    if not isinstance(args, Mapping):
        raise TypeError("tool call function arguments must decode to an object")
    return name, args


def _tool_call_json(name: str, args: Mapping[str, Any]) -> str:
    name_json = json.dumps(name, ensure_ascii=False, allow_nan=False)
    return f'{{"name":{name_json},"args":{_canonical_json(args)}}}'
