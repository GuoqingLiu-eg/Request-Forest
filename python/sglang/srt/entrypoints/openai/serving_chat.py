from __future__ import annotations

import copy
import json
import logging
import math
import time
import uuid
from enum import Enum
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional, Union


class ThinkingMode(str, Enum):


    CHAT = "chat"
    THINKING = "thinking"


import jinja2
import orjson
from fastapi import Request

try:
    from mistral_common.exceptions import MistralCommonException

    _MISTRAL_COMMON_ERRORS: tuple[type[BaseException], ...] = (MistralCommonException,)
except ImportError:
    _MISTRAL_COMMON_ERRORS = ()

_CHAT_TEMPLATE_CLIENT_ERRORS: tuple[type[BaseException], ...] = (
    jinja2.TemplateError,
    TypeError,
) + _MISTRAL_COMMON_ERRORS
from fastapi.responses import ORJSONResponse, StreamingResponse
from jsonschema import Draft202012Validator, SchemaError

from sglang.srt.entrypoints.openai import chat_encoding, encoding_dsv4, encoding_dsv32
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionMessageContentTextPart,
    ChatCompletionMessageContentVideoPart,
    ChatCompletionMessageGenericParam,
    ChatCompletionMessageUserParam,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatCompletionTokenLogprob,
    ChatMessage,
    ChoiceLogprobs,
    DeltaMessage,
    ErrorResponse,
    FunctionResponse,
    LogProbs,
    MessageProcessingResult,
    PromptTokensDetails,
    ResponseParserProtocol,
    SglExt,
    Tool,
    ToolCall,
    ToolCallProcessingResult,
    ToolChoice,
    TopLogprob,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.entrypoints.openai.sse_utils import build_sse_content
from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor
from sglang.srt.entrypoints.openai.utils import (
    cached_tokens_details_from_dict,
    process_cached_tokens_details_from_ret,
    process_hidden_states_for_response,
    process_hidden_states_from_ret,
    process_routed_experts_from_ret,
    process_spec_tokens_details_from_ret,
    should_include_usage,
    spec_tokens_details_from_meta_info,
    to_openai_style_logprobs,
)
from sglang.srt.entrypoints.request_headers import apply_header_overrides
from sglang.srt.environ import envs
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.json_array_parser import JsonArrayParser
from sglang.srt.function_call.utils import (
    get_json_schema_constraint,
    normalize_json_schema_types,
)
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.parser.conversation import generate_chat_conv
from sglang.srt.parser.jinja_template_utils import process_content_for_template_format
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.utils.weight_versions import build_endpoint_weight_version_metadata

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager
    from sglang.srt.parser.template_manager import TemplateManager

logger = logging.getLogger(__name__)

_MEDIA_CONTENT_PART_TYPES = frozenset({"image_url", "video_url", "audio_url"})


def normalize_tool_content(role: str, content):

    if role != "tool" or not isinstance(content, list):
        return content
    parts = content
    is_openai_text_parts = all(
        (isinstance(p, dict) and p.get("type") == "text") or isinstance(p, str)
        for p in parts
    )
    if is_openai_text_parts:
        text_parts = [p.get("text", "") if isinstance(p, dict) else p for p in parts]
        return " ".join(text_parts)
    return content


def parse_tool_call_arguments(arguments: str) -> Dict[str, Any]:

    try:
        parsed_arguments = orjson.loads(arguments)
    except orjson.JSONDecodeError as exc:
        raise ValueError(
            "Assistant tool call function.arguments must be valid JSON."
        ) from exc

    if not isinstance(parsed_arguments, dict):
        raise ValueError(
            "Assistant tool call function.arguments must be a JSON object."
        )

    return parsed_arguments


def normalize_assistant_tool_call_arguments(
    message: Dict[str, Any], *, strict: bool = True
) -> None:

    if message.get("role") != "assistant" or not isinstance(
        message.get("tool_calls"), list
    ):
        return

    for item in message["tool_calls"]:
        function = item.get("function") if isinstance(item, dict) else None
        if not isinstance(function, dict):
            continue
        if "arguments" in function and isinstance(function["arguments"], str):
            try:
                function["arguments"] = parse_tool_call_arguments(function["arguments"])
            except ValueError:
                if strict:
                    raise


def _extract_max_dynamic_patch(request: ChatCompletionRequest):
    img_vals = []
    vid_vals = []
    for msg in request.messages or []:
        content = getattr(msg, "content", None)
        if not isinstance(content, list):
            continue
        for part in content:

            if getattr(part, "type", None) == "image_url":
                iu = getattr(part, "image_url", None)
                mdp = getattr(iu, "max_dynamic_patch", None) if iu else None
                if mdp is not None:
                    img_vals.append(int(mdp))
            elif getattr(part, "type", None) == "video_url":
                vu = getattr(part, "video_url", None)
                mdp = getattr(vu, "max_dynamic_patch", None) if vu else None
                if mdp is not None:
                    vid_vals.append(int(mdp))


    img_max_dynamic_patch = min(img_vals) if img_vals else None
    vid_max_dynamic_patch = min(vid_vals) if vid_vals else None
    return img_max_dynamic_patch, vid_max_dynamic_patch


KIMI_K3_IMAGE_PLACEHOLDER = "<|kimi_image_placeholder|>"
KIMI_K3_IMAGE_PLACEHOLDER_ESCAPED = "<| kimi_image_placeholder |>"


def neutralize_kimi_k3_image_placeholder(text: str) -> str:
    return text.replace(KIMI_K3_IMAGE_PLACEHOLDER, KIMI_K3_IMAGE_PLACEHOLDER_ESCAPED)


def neutralize_kimi_k3_image_placeholder_value(value: Any) -> Any:
    if isinstance(value, str):
        return neutralize_kimi_k3_image_placeholder(value)
    if isinstance(value, list):
        return [neutralize_kimi_k3_image_placeholder_value(item) for item in value]
    if isinstance(value, dict):
        return {
            key: neutralize_kimi_k3_image_placeholder_value(item)
            for key, item in value.items()
        }
    return value


def _extract_video_question(request: ChatCompletionRequest) -> Optional[str]:

    for message in reversed(request.messages or []):
        if not isinstance(message, ChatCompletionMessageUserParam):
            continue
        content = message.content
        if not isinstance(content, list):
            continue
        has_video = any(
            isinstance(part, ChatCompletionMessageContentVideoPart) for part in content
        )
        if not has_video:
            continue
        return "".join(
            part.text
            for part in content
            if isinstance(part, ChatCompletionMessageContentTextPart)
        )
    return None


def _build_video_config(request: ChatCompletionRequest) -> Optional[Dict[str, Any]]:

    config = dict(request.video_config or {})
    question = _extract_video_question(request)
    if question is not None:


        config["_question"] = question
    return config or None


class OpenAIServingChat(OpenAIServingBase):


    _default_sampling_params_logged = False
    _KIMI_K3_GENERATION_STUB_TOKENS = 3

    def __init__(
        self,
        tokenizer_manager: TokenizerManager,
        template_manager: TemplateManager,
    ):
        super().__init__(tokenizer_manager)
        self.template_manager = template_manager
        self.tool_call_parser = self.tokenizer_manager.config_value("tool_call_parser")
        self.reasoning_parser = self.tokenizer_manager.config_value("reasoning_parser")
        self.default_chat_template_kwargs = (
            self.tokenizer_manager.server_args.default_chat_template_kwargs or {}
        )
        self._reasoning_detector = None
        if self.reasoning_parser:
            try:
                rp = ReasoningParser(
                    model_type=self.reasoning_parser,
                    stream_reasoning=True,
                    tokenizer=self.tokenizer_manager.tokenizer,
                )
                self._reasoning_detector = rp.detector
            except ValueError as e:
                logger.warning(
                    "Failed to initialize reasoning detector for parser '%s': %s",
                    self.reasoning_parser,
                    e,
                )


        self.default_sampling_params = (
            self.tokenizer_manager.model_config.get_default_sampling_params()
        )
        if (
            self.default_sampling_params
            and not OpenAIServingChat._default_sampling_params_logged
        ):
            logger.info(
                f"Using default chat sampling params from model generation config: {self.default_sampling_params}",
            )
            OpenAIServingChat._default_sampling_params_logged = True


        self.is_gpt_oss = (
            hasattr(self.tokenizer_manager.model_config, "hf_config")
            and hasattr(self.tokenizer_manager.model_config.hf_config, "model_type")
            and self.tokenizer_manager.model_config.hf_config.model_type == "gpt_oss"
        )
        self.is_gemma4 = (
            hasattr(self.tokenizer_manager.model_config, "hf_config")
            and hasattr(self.tokenizer_manager.model_config.hf_config, "model_type")
            and self.tokenizer_manager.model_config.hf_config.model_type
            in ("gemma4", "gemma4_unified")
        )


        self.chat_encoding_spec = self._resolve_chat_encoding_spec()
        self._dsv4_reasoning_effort_profile = (
            chat_encoding.resolve_dsv4_reasoning_effort_profile(
                model_path=self.tokenizer_manager.model_path,
                revision=self.tokenizer_manager.server_args.revision,
                override=self.tokenizer_manager.model_config.hf_config.to_dict().get(
                    chat_encoding.DSV4_REASONING_EFFORT_PROFILE_OVERRIDE
                ),
            )
            if self.chat_encoding_spec == "dsv4"
            else None
        )


        self._inkling_default_reasoning_effort: Optional[float] = (
            self._get_inkling_default_reasoning_effort()
            if self.chat_encoding_spec == "inkling"
            else None
        )


        self._response_parser: Optional[ResponseParserProtocol] = None


        try:
            self._tokenizer_auto_adds_specials = (
                len(self.tokenizer_manager.tokenizer.encode("")) > 0
            )
        except Exception:
            self._tokenizer_auto_adds_specials = True

    def _handle_last_assistant_message(
        self,
        messages: List[Dict[str, Any]],
        request: ChatCompletionRequest,
    ) -> tuple[List[Dict[str, Any]], Optional[str]]:

        assistant_prefix = None
        if messages and messages[-1].get("role") == "assistant":
            last_content = messages[-1].get("content")

            if isinstance(last_content, str):
                if request.continue_final_message:

                    assistant_prefix = last_content
                    messages = messages[:-1]
                else:

                    messages[-1] = {"role": "user", "content": last_content}
        return messages, assistant_prefix

    def _append_assistant_prefix_to_prompt_ids(
        self, prompt_ids: List[int], assistant_prefix: str
    ) -> List[int]:

        encoded = self.tokenizer_manager.tokenizer.encode(assistant_prefix)
        if encoded and encoded[0] == self.tokenizer_manager.tokenizer.bos_token_id:
            encoded = encoded[1:]
        return prompt_ids + encoded

    def _resolve_chat_encoding_spec(self) -> Optional[str]:

        return chat_encoding.resolve_chat_encoding_spec(
            hf_config=self.tokenizer_manager.model_config.hf_config,
            tokenizer=self.tokenizer_manager.tokenizer,
            tool_call_parser=self.tool_call_parser,
        )

    def _request_id_prefix(self) -> str:
        return "chatcmpl-"

    def _effective_tools(self, request: ChatCompletionRequest) -> List[Tool]:
        tools = list(request.tools or [])
        for message in request.messages:
            if (
                isinstance(message, ChatCompletionMessageGenericParam)
                and message.role in ("system", "developer")
                and message.tools
            ):
                tools.extend(message.tools)
        return tools

    def _prepare_kimi_k3_messages(
        self,
        messages: List[Dict[str, Any]],
        request: ChatCompletionRequest,
    ) -> tuple[List[Dict[str, Any]], int, Optional[str]]:
        image_count = 0
        for index, message in enumerate(messages):
            content = message.get("content")
            if isinstance(content, list):
                parts = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    part_type = part.get("type")
                    if part_type in ("text", "input_text"):
                        parts.append(
                            {
                                "type": "text",
                                "text": neutralize_kimi_k3_image_placeholder(
                                    part["text"]
                                ),
                            }
                        )
                    elif part_type in ("image_url", "input_image"):
                        image = part.get("image_url") or {}
                        if isinstance(image, str):
                            image = {"url": image, "detail": part.get("detail")}
                        parts.append({"type": "image_url", "image_url": image})
                        image_count += 1
                message["content"] = parts
            elif isinstance(content, str):
                message["content"] = neutralize_kimi_k3_image_placeholder(content)
            elif content is None:
                message["content"] = ""

            if message.get("role") == "assistant":
                for key in ("reasoning_content", "reasoning"):
                    if key in message:
                        message[key] = neutralize_kimi_k3_image_placeholder_value(
                            message[key]
                        )
                for tool_call in message.get("tool_calls") or []:
                    function = (
                        tool_call.get("function")
                        if isinstance(tool_call, dict)
                        else None
                    )
                    if isinstance(function, dict) and "arguments" in function:
                        function["arguments"] = (
                            neutralize_kimi_k3_image_placeholder_value(
                                function["arguments"]
                            )
                        )

            source = request.messages[index]
            if (
                isinstance(source, ChatCompletionMessageGenericParam)
                and source.role in ("system", "developer")
                and source.tools
            ):
                message["tools"] = [
                    tool.model_dump(exclude_unset=True, by_alias=True)
                    for tool in source.tools
                ]
            if message.get("role") == "developer":
                message["role"] = "system"

        assistant_prefix = None
        if request.continue_final_message:
            messages, assistant_prefix = self._handle_last_assistant_message(
                messages, request
            )
        return messages, image_count, assistant_prefix

    def _encode_messages(
        self,
        messages: List[Dict[str, Any]],
        request: ChatCompletionRequest,
        thinking_mode: ThinkingMode,
        tools: Optional[List[Dict]] = None,
    ) -> Optional[List[int]]:

        if self.chat_encoding_spec == "inkling":


            from sglang.srt.parser.inkling_renderer import render_inkling_messages
            from sglang.srt.parser.inkling_tokenizer import (
                CONTENT_TEXT,
                MESSAGE_MODEL,
                InklingTokenizer,
            )

            inkling_tokenizer = InklingTokenizer(
                tokenizer=self.tokenizer_manager.tokenizer
            )
            reasoning_effort = self._parse_inkling_reasoning_effort(
                request.reasoning_effort
            )
            if reasoning_effort is None:
                reasoning_effort = self._inkling_default_reasoning_effort
            assistant_prefix = self._pop_inkling_assistant_prefix(messages, request)
            prompt_ids = render_inkling_messages(
                messages,
                inkling_tokenizer,
                add_generation_prompt=False,
                tools=tools,
                reasoning_effort=reasoning_effort,
            )
            if assistant_prefix is not None:


                prompt_ids += [
                    inkling_tokenizer.encode_special(MESSAGE_MODEL),
                    inkling_tokenizer.encode_special(CONTENT_TEXT),
                    *inkling_tokenizer.encode_text(assistant_prefix),
                ]
            return prompt_ids
        if self.chat_encoding_spec == "kimi_k3":
            messages, image_count, assistant_prefix = self._prepare_kimi_k3_messages(
                messages, request
            )
            template_kwargs = dict(request.chat_template_kwargs or {})
            template_kwargs.pop("tokenize", None)
            template_kwargs.pop("return_dict", None)
            template_kwargs.pop("image_prompts", None)
            if image_count:
                template_kwargs["image_prompts"] = ["<|media_pad|>"] * image_count

            if (
                request.reasoning_effort in ("low", "high", "max")
                and "thinking_effort" not in template_kwargs
            ):
                template_kwargs["thinking_effort"] = request.reasoning_effort
            elif request.reasoning_effort not in (
                None,
                "none",
                "low",
                "high",
                "max",
            ):
                logger.warning(
                    "Kimi K3 does not support reasoning_effort=%r; using the "
                    "encoder default.",
                    request.reasoning_effort,
                )

            effective_tools = self._effective_tools(request)
            if (
                effective_tools
                and isinstance(request.tool_choice, str)
                and request.tool_choice in ("required", "none")
            ):
                template_kwargs.setdefault("tool_choice", request.tool_choice)
            if request.response_format is not None:
                template_kwargs.setdefault(
                    "response_format",
                    request.response_format.model_dump(
                        exclude_unset=True, by_alias=True
                    ),
                )

            request_tools = (
                [
                    tool.model_dump(exclude_unset=True, by_alias=True)
                    for tool in request.tools
                ]
                if request.tools
                else None
            )
            prompt_ids = self.tokenizer_manager.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                tools=request_tools,
                return_dict=False,
                **template_kwargs,
            )
            if assistant_prefix:
                prompt_ids = self._append_assistant_prefix_to_prompt_ids(
                    prompt_ids, assistant_prefix
                )
            return prompt_ids
        return None

    @staticmethod
    def _pop_inkling_assistant_prefix(
        messages: List[Dict[str, Any]],
        request: ChatCompletionRequest,
    ) -> Optional[str]:

        if not request.continue_final_message or not messages:
            return None
        last = messages[-1]
        if (
            last.get("role") != "assistant"
            or not isinstance(last.get("content"), str)
            or last.get("tool_calls")
            or last.get("reasoning_content")
        ):
            return None
        messages.pop()
        return last["content"]

    @staticmethod
    def _parse_inkling_reasoning_effort(
        value: Optional[Union[str, float]],
    ) -> Optional[float]:

        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("Inkling reasoning_effort must not be a boolean")
        if isinstance(value, (int, float)):
            parsed = float(value)
            if not math.isfinite(parsed) or not 0.0 <= parsed <= 0.99:
                raise ValueError("Inkling reasoning_effort must be in [0.0, 0.99]")
            return parsed
        _EFFORT_MAP = {
            "none": 0.0,
            "minimal": 0.1,
            "low": 0.2,
            "medium": 0.7,
            "high": 0.9,
            "xhigh": 0.99,
            "max": 0.99,
        }
        if value in _EFFORT_MAP:
            return _EFFORT_MAP[value]
        try:
            parsed = float(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid Inkling reasoning_effort: {value!r}") from exc
        if not math.isfinite(parsed) or not 0.0 <= parsed <= 0.99:
            raise ValueError("Inkling reasoning_effort must be in [0.0, 0.99]")
        return parsed

    @staticmethod
    def _get_inkling_default_reasoning_effort() -> float:

        from sglang.srt.environ import envs

        val = envs.SGLANG_INKLING_DEFAULT_REASONING_EFFORT.get()
        if not val:
            return 0.9
        try:
            parsed = float(val)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "SGLANG_INKLING_DEFAULT_REASONING_EFFORT must be numeric"
            ) from exc
        if not math.isfinite(parsed) or not 0.0 <= parsed <= 0.99:
            raise ValueError(
                "SGLANG_INKLING_DEFAULT_REASONING_EFFORT must be in [0.0, 0.99]"
            )
        return parsed

    def _decode_response(self, ret_item: Dict[str, Any]) -> Union[str, ErrorResponse]:

        return ret_item["text"]

    def _get_parsed_response_fields(
        self,
        reasoning_text: Optional[str],
        tool_calls: Optional[List[Dict]],
    ) -> tuple[Optional[str], Optional[List[Dict]]]:

        return reasoning_text, tool_calls

    def _continuous_usage_cached_details(
        self, content: Dict[str, Any]
    ) -> Optional[PromptTokensDetails]:
        if not self.tokenizer_manager.server_args.enable_cache_report:
            return None
        return UsageProcessor._details_if_cached(
            content["meta_info"].get("cached_tokens", 0)
        )

    def _reported_prompt_tokens(self, meta_info: Dict[str, Any]) -> int:
        prompt_tokens = meta_info.get("prompt_tokens", 0)
        if self.chat_encoding_spec == "kimi_k3":


            prompt_tokens = max(0, prompt_tokens - self._KIMI_K3_GENERATION_STUB_TOKENS)
        return prompt_tokens

    async def _generate_stream_content(
        self,
        content: Dict[str, Any],
        index: int,
        request: ChatCompletionRequest,
        stream_offsets: Dict[int, int],
        reasoning_parser_dict: Dict,
        parser_dict: Dict,
        has_tool_calls: Dict[int, bool],
        choice_logprobs: Optional[Dict],
        finish_reason_type: Optional[str],
        continuous_usage_stats: bool,
        prompt_tokens: Dict[int, int],
        reasoning_tokens: Dict[int, int],
        completion_tokens: Dict[int, int],
    ) -> AsyncGenerator[str, None]:

        offset = stream_offsets.get(index, 0)
        if self.tokenizer_manager.server_args.incremental_streaming_output:
            delta = content["text"]
        else:
            delta = content["text"][offset:]
            stream_offsets[index] = len(content["text"])


        remaining_logprobs = choice_logprobs


        if self.reasoning_parser and request.separate_reasoning:
            reasoning_text, delta = self._process_reasoning_stream(
                index,
                delta,
                reasoning_parser_dict,
                content,
                request,
                finish_reason_type,
            )
            if reasoning_text:
                usage = None
                if continuous_usage_stats:
                    usage = UsageProcessor.calculate_token_usage(
                        prompt_tokens=prompt_tokens.get(index, 0),
                        reasoning_tokens=reasoning_tokens.get(index, 0),
                        completion_tokens=completion_tokens.get(index, 0),
                        cached_tokens=self._continuous_usage_cached_details(content),
                    ).model_dump()

                yield build_sse_content(
                    chunk_id=content["meta_info"]["id"],
                    created=int(time.time()),
                    model=request.model,
                    index=index,
                    reasoning_content=reasoning_text,
                    logprobs=remaining_logprobs,
                    usage=usage,
                )
                remaining_logprobs = None


        if self._tool_call_parsing_active(request):
            async for chunk in self._process_tool_call_stream(
                index,
                delta,
                parser_dict,
                content,
                request,
                has_tool_calls,
                continuous_usage_stats,
                flush=finish_reason_type is not None and finish_reason_type != "abort",
            ):
                if chunk:
                    yield chunk


            if finish_reason_type is not None and index in parser_dict:
                parser = parser_dict[index]
                remaining_chunk = self._check_for_unstreamed_tool_args(
                    parser, content, request, index
                )
                if remaining_chunk:
                    yield remaining_chunk

        else:

            if delta:
                usage = None
                if continuous_usage_stats:
                    usage = UsageProcessor.calculate_token_usage(
                        prompt_tokens=prompt_tokens.get(index, 0),
                        reasoning_tokens=reasoning_tokens.get(index, 0),
                        completion_tokens=completion_tokens.get(index, 0),
                        cached_tokens=self._continuous_usage_cached_details(content),
                    ).model_dump()

                yield build_sse_content(
                    chunk_id=content["meta_info"]["id"],
                    created=int(time.time()),
                    model=request.model,
                    index=index,
                    content=delta,
                    logprobs=remaining_logprobs,
                    usage=usage,
                )
                remaining_logprobs = None


        if remaining_logprobs is not None and (
            self.reasoning_parser or self.tool_call_parser
        ):
            usage = None
            if continuous_usage_stats:
                usage = UsageProcessor.calculate_token_usage(
                    prompt_tokens=prompt_tokens.get(index, 0),
                    reasoning_tokens=reasoning_tokens.get(index, 0),
                    completion_tokens=completion_tokens.get(index, 0),
                    cached_tokens=self._continuous_usage_cached_details(content),
                ).model_dump()

            yield build_sse_content(
                chunk_id=content["meta_info"]["id"],
                created=int(time.time()),
                model=request.model,
                index=index,
                logprobs=remaining_logprobs,
                usage=usage,
            )

    def _tool_call_parsing_active(self, request: ChatCompletionRequest) -> bool:

        return bool(
            request.tool_choice != "none"
            and self._effective_tools(request)
            and self.tool_call_parser
        )

    def _validate_request(self, request: ChatCompletionRequest) -> Optional[str]:

        if not request.messages:
            return "Messages cannot be empty."

        if request.return_sampling_mask and not request.return_meta_info:
            return "return_sampling_mask requires return_meta_info=true."

        media_error = self._validate_media_content(request)
        if media_error:
            return media_error

        effective_tools = self._effective_tools(request)
        has_message_tools = any(
            isinstance(message, ChatCompletionMessageGenericParam)
            and message.role in ("system", "developer")
            and message.tools
            for message in request.messages
        )
        if (
            isinstance(request.tool_choice, str)
            and request.tool_choice.lower() == "required"
            and not effective_tools
        ):
            return "Tools cannot be empty if tool choice is set to required."

        if request.tool_choice is not None and not isinstance(request.tool_choice, str):
            if not effective_tools:
                return "Tools cannot be empty if tool choice is set to a specific tool."
            tool_name = request.tool_choice.function.name
            tool_exists = any(
                tool.function.name == tool_name for tool in effective_tools
            )
            if not tool_exists:
                return f"Tool '{tool_name}' not found in tools list."

        if has_message_tools:
            names = [tool.function.name for tool in effective_tools]
            if len(names) != len(set(names)):
                return "Tool names must be unique across request and message tools."


        for i, tool in enumerate(effective_tools):
            if tool.function.parameters is None:
                continue
            try:


                normalize_json_schema_types(tool.function.parameters)
                Draft202012Validator.check_schema(tool.function.parameters)
            except SchemaError as e:
                return f"Tool {i} function has invalid 'parameters' schema: {str(e)}"
            except RecursionError:
                return (
                    f"Tool {i} function 'parameters' schema is too deeply nested "
                    "or contains a cycle."
                )

        max_output_tokens = request.max_completion_tokens or request.max_tokens
        server_context_length = self.tokenizer_manager.server_args.context_length
        if (
            max_output_tokens
            and server_context_length
            and max_output_tokens > server_context_length
        ) and not self.tokenizer_manager.server_args.allow_auto_truncate:
            return (
                f"max_completion_tokens is too large: {max_output_tokens}."
                f"This model supports at most {server_context_length} completion tokens."
            )

        if request.response_format and request.response_format.type == "json_schema":
            schema = getattr(request.response_format.json_schema, "schema_", None)
            if schema is None:
                return "schema_ is required for json_schema response format request."

        return None

    def _validate_media_content(self, request: ChatCompletionRequest) -> Optional[str]:
        if self.tokenizer_manager.model_config.is_multimodal:
            return None

        media_type = next(
            (
                part.type
                for message in request.messages
                if isinstance(message.content, list)
                for part in message.content
                if part.type in _MEDIA_CONTENT_PART_TYPES
            ),
            None,
        )
        if media_type is None:
            return None

        return (
            "Model only supports text input; "
            f"received unsupported content type '{media_type}'."
        )

    def _convert_to_internal_request(
        self,
        request: ChatCompletionRequest,
        raw_request: Request = None,
    ) -> tuple[GenerateReqInput, ChatCompletionRequest]:
        reasoning_effort = (
            request.chat_template_kwargs.pop("reasoning_effort", None)
            if request.chat_template_kwargs
            else None
        )
        if self.is_gpt_oss and reasoning_effort == "none":
            raise ValueError(
                f"Harmony does not support reasoning effort {reasoning_effort}"
            )

        if reasoning_effort is not None:
            request.reasoning_effort = reasoning_effort

        if request.stream:
            if request.return_prompt_token_ids:
                raise ValueError(
                    "return_prompt_token_ids is not supported with streaming. "
                    "Please set stream=false when using return_prompt_token_ids=true."
                )
            if request.return_token_ids:
                raise ValueError(
                    "return_token_ids is not supported with streaming on "
                    "/v1/chat/completions. Please set stream=false when using "
                    "return_token_ids=true."
                )
            if request.return_meta_info:
                raise ValueError(
                    "return_meta_info is not supported with streaming. "
                    "Please set stream=false when using return_meta_info=true."
                )

        is_multimodal = self.tokenizer_manager.model_config.is_multimodal


        processed_messages = self._process_messages(request, is_multimodal)

        sampling_params = request.to_sampling_params(
            stop=processed_messages.stop,
            model_generation_config=self.default_sampling_params,
            tool_call_constraint=processed_messages.tool_call_constraint,
            renderer_handles_response_format=self.chat_encoding_spec == "kimi_k3",
        )


        if request.input_ids is not None:
            prompt_kwargs = {"input_ids": processed_messages.prompt_ids}
        elif is_multimodal and self.chat_encoding_spec == "kimi_k3":
            prompt_kwargs = {"input_ids": processed_messages.prompt_ids}
        elif is_multimodal:


            if (
                self.chat_encoding_spec == "inkling"
                and isinstance(processed_messages.prompt_ids, list)
                and processed_messages.prompt_ids
            ):
                prompt_kwargs = {"input_ids": processed_messages.prompt_ids}
            else:
                prompt_kwargs = {"text": processed_messages.prompt}
        else:
            if isinstance(processed_messages.prompt_ids, str):
                prompt_kwargs = {"text": processed_messages.prompt_ids}
            else:
                prompt_kwargs = {"input_ids": processed_messages.prompt_ids}


        custom_labels = self.extract_custom_labels(raw_request)


        effective_routed_dp_rank = self.extract_routed_dp_rank_from_header(
            raw_request, request.routed_dp_rank
        )


        lora_path = self._resolve_lora_path(request.model, request.lora_path)
        img_max_dynamic_patch, vid_max_dynamic_patch = _extract_max_dynamic_patch(
            request
        )
        adapted_request = GenerateReqInput(
            **prompt_kwargs,
            image_data=processed_messages.image_data,
            video_data=processed_messages.video_data,
            audio_data=processed_messages.audio_data,
            sampling_params=sampling_params,
            return_logprob=request.logprobs,
            logprob_start_len=-1,
            top_logprobs_num=request.top_logprobs or 0,
            return_sampling_mask=request.return_sampling_mask,
            stream=request.stream,
            return_text_in_logprobs=True,
            modalities=processed_messages.modalities,
            lora_path=lora_path,
            bootstrap_host=request.bootstrap_host,
            bootstrap_port=request.bootstrap_port,
            bootstrap_room=request.bootstrap_room,
            routed_dp_rank=effective_routed_dp_rank,
            disagg_prefill_dp_rank=request.disagg_prefill_dp_rank,
            return_hidden_states=request.return_hidden_states,
            return_routed_experts=request.return_routed_experts,
            routed_experts_start_len=request.routed_experts_start_len,
            rid=request.rid,
            session_id=request.session_id,
            extra_key=request.extra_key,
            cache_salt=request.cache_salt,
            require_reasoning=processed_messages.require_reasoning,
            priority=request.priority,
            routing_key=self.extract_routing_key(raw_request),
            custom_labels=custom_labels,
            custom_logit_processor=request.custom_logit_processor,
            images_config=getattr(request, "images_config", None),
            video_config=_build_video_config(request),
            image_max_dynamic_patch=img_max_dynamic_patch,
            video_max_dynamic_patch=vid_max_dynamic_patch,
            max_dynamic_patch=getattr(request, "max_dynamic_patch", None),
            use_audio_in_video=getattr(request, "use_audio_in_video", False),
            return_prompt_token_ids=request.return_prompt_token_ids
            or request.return_token_ids,
        )
        if (
            raw_request is not None
            and envs.SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES.get()
        ):
            apply_header_overrides(adapted_request, raw_request.headers)

        return adapted_request, request

    def _process_messages(
        self, request: ChatCompletionRequest, is_multimodal: bool
    ) -> MessageProcessingResult:

        if self.default_chat_template_kwargs:
            ctk = dict(request.chat_template_kwargs or {})
            for k, v in self.default_chat_template_kwargs.items():
                ctk.setdefault(k, v)
            request.chat_template_kwargs = ctk
            effort = ctk.get("reasoning_effort")
            if effort is not None and request.reasoning_effort is None:
                request.reasoning_effort = effort


        if self.is_gpt_oss or self.is_gemma4:
            request.skip_special_tokens = False

        self._patch_reasoning_skip_special_tokens(request)

        thinking_mode = self._get_reasoning_from_request(request)


        xgrammar_reasoning = thinking_mode and (self.reasoning_parser is None)
        tool_call_constraint = None


        tools = None
        tool_call_stop = None
        required_parsed_natively = False
        effective_tools = self._effective_tools(request)
        if effective_tools and request.tool_choice != "none":
            request.skip_special_tokens = False
            if not isinstance(request.tool_choice, str):
                tools = [
                    item.model_dump()
                    for item in request.tools or []
                    if item.function.name == request.tool_choice.function.name
                ] or None
            elif request.tools:
                tools = [item.model_dump() for item in request.tools]
            if self.tool_call_parser:
                parser = FunctionCallParser(
                    effective_tools,
                    self.tool_call_parser,
                    tokenizer=self.tokenizer_manager.tokenizer,
                )
                tool_call_constraint = parser.get_structure_constraint(
                    request.tool_choice,
                    parallel_tool_calls=request.parallel_tool_calls,
                    thinking_mode=xgrammar_reasoning,
                )
                required_parsed_natively = parser.detector.parses_required_natively()
                if self.chat_encoding_spec == "kimi_k3":
                    tool_call_stop = parser.detector.eot_token
            if (
                tool_call_constraint is None
                and not required_parsed_natively
                and not (
                    self.chat_encoding_spec == "kimi_k3"
                    and self.tool_call_parser == "kimi_k3"
                )
                and (
                    request.tool_choice == "required"
                    or isinstance(request.tool_choice, ToolChoice)
                )
            ):
                json_schema = get_json_schema_constraint(
                    effective_tools,
                    request.tool_choice,
                    parallel_tool_calls=request.parallel_tool_calls,
                )
                tool_call_constraint = ("json_schema", json_schema)


        if request.input_ids is not None:
            result = MessageProcessingResult(
                prompt="",
                prompt_ids=request.input_ids,
                image_data=None,
                audio_data=None,
                video_data=None,
                modalities=[],
                stop=request.stop or [],
            )
        elif self.template_manager.chat_template_name is None:
            result = self._apply_jinja_template(request, tools, is_multimodal)
        else:
            result = self._apply_conversation_template(request, is_multimodal)

        if tool_call_stop is not None:
            if isinstance(result.stop, str):
                result.stop = [result.stop]
            elif result.stop is None:
                result.stop = []
            else:
                result.stop = list(result.stop)
            if tool_call_stop not in result.stop:
                result.stop.append(tool_call_stop)

        result.tool_call_constraint = tool_call_constraint
        result.require_reasoning = thinking_mode
        result.skip_special_tokens = request.skip_special_tokens
        return result

    def _apply_jinja_template(
        self,
        request: ChatCompletionRequest,
        tools: Optional[List[Dict]],
        is_multimodal: bool,
    ) -> MessageProcessingResult:

        prompt = ""
        prompt_ids = []
        openai_compatible_messages = []
        image_data = []
        video_data = []
        audio_data = []
        modalities = []

        template_content_format = self.template_manager.jinja_template_content_format


        thinking_requested = (request.chat_template_kwargs or {}).get(
            "thinking", envs.SGLANG_DEFAULT_THINKING.get()
        )
        thinking_mode = (
            ThinkingMode.THINKING if thinking_requested else ThinkingMode.CHAT
        )
        messages = [msg.model_dump() for msg in request.messages]
        for message in messages:
            normalize_assistant_tool_call_arguments(
                message, strict=self.chat_encoding_spec != "kimi_k3"
            )

        prompt_ids = self._encode_messages(
            copy.deepcopy(messages),
            request,
            thinking_mode,
            tools=tools,
        )

        if prompt_ids is not None:
            if self.chat_encoding_spec in ("inkling", "kimi_k3"):
                for message in request.messages:
                    msg_dict = message.model_dump()
                    if msg_dict.get("content") is None:
                        msg_dict["content"] = ""
                    process_content_for_template_format(
                        msg_dict,
                        "openai",
                        image_data,
                        video_data,
                        audio_data,
                        modalities,
                    )
        elif self.chat_encoding_spec is not None:

            messages = copy.deepcopy(messages)


            for i, msg in enumerate(messages):
                if isinstance(msg.get("content"), list):
                    messages[i] = process_content_for_template_format(
                        msg, "string", [], [], [], []
                    )

            for msg in messages:
                if msg.get("content") is None:
                    msg["content"] = ""
                processed_msg = process_content_for_template_format(
                    msg,
                    template_content_format,
                    image_data,
                    video_data,
                    audio_data,
                    modalities,
                    use_dpsk_v32_encoding=self.chat_encoding_spec == "dsv32",
                )
                msg.update(processed_msg)


            messages, assistant_prefix = self._handle_last_assistant_message(
                messages, request
            )

            if messages[0]["role"] != "system":

                messages.insert(0, {"role": "system", "content": ""})
            if request.tools:
                messages[0]["tools"] = [tool.model_dump() for tool in request.tools]


            if self.chat_encoding_spec == "dsv4":
                effort_source = request.reasoning_effort
                if effort_source is None:
                    env_val = envs.SGLANG_DSV4_REASONING_EFFORT.get()
                    if env_val:
                        effort_source = env_val
                reasoning_effort_profile = self._dsv4_reasoning_effort_profile
                assert reasoning_effort_profile is not None
                accepted_efforts = encoding_dsv4.REASONING_EFFORT_PROFILES[
                    reasoning_effort_profile
                ]
                v4_reasoning_effort = (
                    effort_source if effort_source in accepted_efforts else None
                )
                if request.task is not None:
                    encoding_dsv4.attach_task_to_last_user_message(
                        messages, request.task
                    )
                real_input = encoding_dsv4.encode_messages(
                    messages,
                    thinking_mode=thinking_mode,
                    reasoning_effort=v4_reasoning_effort,
                    reasoning_effort_profile=reasoning_effort_profile,
                )
                prompt_ids = self.tokenizer_manager.tokenizer.encode(real_input)
            else:
                real_input = encoding_dsv32.encode_messages(
                    messages, thinking_mode=thinking_mode
                )
                prompt_ids = self.tokenizer_manager.tokenizer.encode(real_input)


            if assistant_prefix:
                prompt_ids = self._append_assistant_prefix_to_prompt_ids(
                    prompt_ids, assistant_prefix
                )
        else:
            for msg_dict in copy.deepcopy(messages):
                if msg_dict.get("content") is None:
                    msg_dict["content"] = ""


                processed_msg = process_content_for_template_format(
                    msg_dict,
                    template_content_format,
                    image_data,
                    video_data,
                    audio_data,
                    modalities,
                )

                processed_msg["content"] = normalize_tool_content(
                    processed_msg["role"], processed_msg.get("content")
                )

                openai_compatible_messages.append(processed_msg)


            openai_compatible_messages, assistant_prefix = (
                self._handle_last_assistant_message(openai_compatible_messages, request)
            )

            extra_template_kwargs = {}
            if request.reasoning_effort is not None:
                extra_template_kwargs["reasoning_effort"] = request.reasoning_effort
            if request.chat_template_kwargs:
                extra_template_kwargs.update(request.chat_template_kwargs)

            rc = self.template_manager.reasoning_config
            if rc is not None and rc.effort_kwarg is not None:
                if request.reasoning_effort == "low":
                    extra_template_kwargs.setdefault(rc.effort_kwarg, True)
                elif request.reasoning_effort in ("medium", "high", "max"):
                    logger.warning(
                        "Model '%s' supports only 'low' reasoning effort; "
                        "requested '%s' treated as default thinking",
                        self.tokenizer_manager.served_model_name,
                        request.reasoning_effort,
                    )


            encode_kwargs = (
                {"add_special_tokens": False}
                if self._tokenizer_auto_adds_specials
                else {}
            )
            try:
                rendered_prompt = self.tokenizer_manager.tokenizer.apply_chat_template(
                    openai_compatible_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    tools=tools,
                    return_dict=False,
                    **extra_template_kwargs,
                )
                prompt_ids = self.tokenizer_manager.tokenizer.encode(
                    rendered_prompt, **encode_kwargs
                )
            except Exception:


                tools = (
                    [t["function"] if "function" in t else t for t in tools]
                    if tools
                    else None
                )
                try:
                    rendered_prompt = (
                        self.tokenizer_manager.tokenizer.apply_chat_template(
                            openai_compatible_messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            tools=tools,
                            return_dict=False,
                            **extra_template_kwargs,
                        )
                    )
                    prompt_ids = self.tokenizer_manager.tokenizer.encode(
                        rendered_prompt, **encode_kwargs
                    )
                except _CHAT_TEMPLATE_CLIENT_ERRORS as template_error:


                    raise ValueError(str(template_error)) from template_error


            if assistant_prefix:
                prompt_ids = self._append_assistant_prefix_to_prompt_ids(
                    prompt_ids, assistant_prefix
                )

            if is_multimodal:
                prompt = self.tokenizer_manager.tokenizer.decode(prompt_ids)

        stop = request.stop
        image_data = image_data if image_data else None
        audio_data = audio_data if audio_data else None
        video_data = video_data if video_data else None
        modalities = modalities if modalities else []
        return MessageProcessingResult(
            prompt=prompt,
            prompt_ids=prompt_ids,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            modalities=modalities,
            stop=stop,
        )

    def _apply_conversation_template(
        self,
        request: ChatCompletionRequest,
        is_multimodal: bool,
    ) -> MessageProcessingResult:

        prompt = ""
        prompt_ids = []
        conv = generate_chat_conv(request, self.template_manager.chat_template_name)


        if (
            request.continue_final_message
            and request.messages
            and request.messages[-1].role == "assistant"
        ):

            if conv.messages and conv.messages[-1][1] is None:
                conv.messages.pop()

            prompt = conv.get_prompt()

            if isinstance(conv.stop_str, list):
                for stop_token in conv.stop_str:
                    if prompt.endswith(stop_token):
                        prompt = prompt[: -len(stop_token)]
            elif isinstance(conv.stop_str, str) and prompt.endswith(conv.stop_str):
                prompt = prompt[: -len(conv.stop_str)]
            if conv.sep and prompt.endswith(conv.sep):
                prompt = prompt[: -len(conv.sep)]
            if getattr(conv, "sep2", None) and prompt.endswith(conv.sep2):
                prompt = prompt[: -len(conv.sep2)]
        else:
            prompt = conv.get_prompt()
            if self._get_reasoning_from_request(request) and (
                self._reasoning_detector is None
                or not self._reasoning_detector.thinks_internally
            ):

                prompt += "<think>"

        image_data = conv.image_data if conv.image_data else None
        video_data = conv.video_data if conv.video_data else None
        audio_data = conv.audio_data if conv.audio_data else None
        modalities = conv.modalities if conv.modalities else []
        stop = copy.copy(conv.stop_str or [] if not request.ignore_eos else [])

        if request.stop:
            if isinstance(request.stop, str):
                stop.append(request.stop)
            else:
                stop.extend(request.stop)

        if not is_multimodal:
            prompt_ids = self.tokenizer_manager.tokenizer.encode(prompt)

        return MessageProcessingResult(
            prompt=prompt,
            prompt_ids=prompt_ids,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            modalities=modalities,
            stop=stop,
        )

    async def _handle_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: ChatCompletionRequest,
        raw_request: Request,
    ) -> Union[StreamingResponse, ErrorResponse]:

        generator = self._generate_chat_stream(adapted_request, request, raw_request)


        try:
            first_chunk = await generator.__anext__()
        except ValueError as e:
            return self.create_error_response(str(e))

        async def prepend_first_chunk():
            yield first_chunk
            async for chunk in generator:
                yield chunk

        return StreamingResponse(
            prepend_first_chunk(),
            media_type="text/event-stream",
            background=self.tokenizer_manager.create_abort_task(adapted_request),
        )

    async def _generate_chat_stream(
        self,
        adapted_request: GenerateReqInput,
        request: ChatCompletionRequest,
        raw_request: Request,
    ) -> AsyncGenerator[str, None]:


        parser_dict = {}
        reasoning_parser_dict = {}


        is_firsts = {}
        stream_offsets = {}
        n_prev_tokens = {}
        has_tool_calls = {}
        finish_reasons = {}


        prompt_tokens = {}
        reasoning_tokens = {}
        completion_tokens = {}
        cached_tokens = {}
        hidden_states = {}
        routed_experts = {}
        cached_tokens_details = {}
        spec_tokens_details = {}
        image_tokens = {}
        audio_tokens = {}
        video_tokens = {}

        stream_started = False
        try:
            include_usage, continuous_usage_stats = should_include_usage(
                request.stream_options,
                self.tokenizer_manager.server_args.stream_response_default_include_usage,
            )

            async for content in self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            ):
                index = content.get("index", 0)

                prompt_tokens[index] = self._reported_prompt_tokens(
                    content["meta_info"]
                )
                completion_tokens[index] = content["meta_info"].get(
                    "completion_tokens", 0
                )
                reasoning_tokens[index] = content["meta_info"].get(
                    "reasoning_tokens", 0
                )
                cached_tokens[index] = content["meta_info"].get("cached_tokens", 0)
                hidden_states[index] = content["meta_info"].get("hidden_states", None)
                routed_experts[index] = content["meta_info"].get("routed_experts", None)
                cached_tokens_details[index] = content["meta_info"].get(
                    "cached_tokens_details", None
                )
                if request.return_spec_tokens_details:
                    spec_tokens_details[index] = spec_tokens_details_from_meta_info(
                        content["meta_info"]
                    )
                image_tokens[index] = content["meta_info"].get("image_tokens", 0)
                audio_tokens[index] = content["meta_info"].get("audio_tokens", 0)
                video_tokens[index] = content["meta_info"].get("video_tokens", 0)


                choice_logprobs = None
                if request.logprobs:
                    n_prev_token = n_prev_tokens.get(index, 0)
                    total_output_logprobs = content["meta_info"][
                        "output_token_logprobs_length"
                    ]
                    if n_prev_token < total_output_logprobs:
                        choice_logprobs = self._process_streaming_logprobs(
                            content, n_prev_token, total_output_logprobs
                        ).model_dump()
                    n_prev_tokens[index] = total_output_logprobs

                finish_reason = content["meta_info"].get("finish_reason", None)
                finish_reason_type = finish_reason["type"] if finish_reason else None


                if finish_reason_type:


                    if finish_reason_type == "abort" and isinstance(
                        finish_reason.get("status_code"), HTTPStatus
                    ):
                        code = finish_reason["status_code"]
                        error = self.create_streaming_error_response(
                            finish_reason.get("message", "Generation aborted."),
                            code.name,
                            code.value,
                        )
                        yield f"data: {error}\n\n"
                        break
                    finish_reasons[index] = finish_reason


                if is_firsts.get(index, True):
                    is_firsts[index] = False
                    yield build_sse_content(
                        chunk_id=content["meta_info"]["id"],
                        created=int(time.time()),
                        model=request.model,
                        index=index,
                        role="assistant",
                        content="",
                    )
                    stream_started = True


                async for chunk in self._generate_stream_content(
                    content=content,
                    index=index,
                    request=request,
                    stream_offsets=stream_offsets,
                    reasoning_parser_dict=reasoning_parser_dict,
                    parser_dict=parser_dict,
                    has_tool_calls=has_tool_calls,
                    choice_logprobs=choice_logprobs,
                    finish_reason_type=finish_reason_type,
                    continuous_usage_stats=continuous_usage_stats,
                    prompt_tokens=prompt_tokens,
                    reasoning_tokens=reasoning_tokens,
                    completion_tokens=completion_tokens,
                ):
                    yield chunk


            for idx, finish_reason_data in finish_reasons.items():
                finish_reason_type = finish_reason_data["type"]


                final_finish_reason = finish_reason_type
                if has_tool_calls.get(idx, False) and finish_reason_type == "stop":
                    final_finish_reason = "tool_calls"

                matched_stop = finish_reason_data.get("matched")
                yield build_sse_content(
                    chunk_id=content["meta_info"]["id"],
                    created=int(time.time()),
                    model=request.model,
                    index=idx,
                    finish_reason=final_finish_reason,
                    matched_stop=matched_stop,
                )


            if request.return_hidden_states and hidden_states:
                for index, choice_hidden_states in hidden_states.items():
                    if choice_hidden_states:
                        response_hidden_states = process_hidden_states_for_response(
                            choice_hidden_states, request.return_hidden_states
                        )
                        hidden_states_chunk = ChatCompletionStreamResponse(
                            id=content["meta_info"]["id"],
                            created=int(time.time()),
                            choices=[
                                ChatCompletionResponseStreamChoice(
                                    index=index,
                                    delta=DeltaMessage(
                                        hidden_states=response_hidden_states
                                    ),
                                    finish_reason=None,
                                )
                            ],
                            model=request.model,
                        )
                        yield f"data: {hidden_states_chunk.model_dump_json()}\n\n"

            sglext_routed = None
            if request.return_routed_experts and routed_experts:
                sglext_routed = next(
                    (v for v in routed_experts.values() if v is not None), None
                )

            sglext_cached_tokens_details = None
            if request.return_cached_tokens_details and cached_tokens_details:
                first_details = next(
                    (v for v in cached_tokens_details.values() if v is not None), None
                )
                if first_details is not None:
                    sglext_cached_tokens_details = cached_tokens_details_from_dict(
                        first_details
                    )

            sglext_spec_tokens_details = None
            if request.return_spec_tokens_details and spec_tokens_details:
                spec_details = [
                    spec_tokens_details[index]
                    for index in sorted(spec_tokens_details)
                    if spec_tokens_details[index] is not None
                ]
                if spec_details:
                    sglext_spec_tokens_details = (
                        spec_details if request.n > 1 else spec_details[0]
                    )

            if any(
                obj is not None
                for obj in [
                    sglext_routed,
                    sglext_cached_tokens_details,
                    sglext_spec_tokens_details,
                ]
            ):
                sglext_chunk = ChatCompletionStreamResponse(
                    id=content["meta_info"]["id"],
                    created=int(time.time()),
                    choices=[],
                    model=request.model,
                    sglext=SglExt(
                        routed_experts=sglext_routed,
                        cached_tokens_details=sglext_cached_tokens_details,
                        spec_tokens_details=sglext_spec_tokens_details,
                    ),
                )
                yield f"data: {sglext_chunk.model_dump_json()}\n\n"


            if include_usage:


                total_image_tokens = sum(
                    tok for idx, tok in image_tokens.items() if idx % request.n == 0
                )
                total_audio_tokens = sum(
                    tok for idx, tok in audio_tokens.items() if idx % request.n == 0
                )
                total_video_tokens = sum(
                    tok for idx, tok in video_tokens.items() if idx % request.n == 0
                )
                usage = UsageProcessor.calculate_streaming_usage(
                    prompt_tokens,
                    reasoning_tokens,
                    completion_tokens,
                    cached_tokens=cached_tokens,
                    n_choices=request.n,
                    enable_cache_report=self.tokenizer_manager.server_args.enable_cache_report,
                    image_tokens=total_image_tokens,
                    audio_tokens=total_audio_tokens,
                    video_tokens=total_video_tokens,
                )
                usage_chunk = ChatCompletionStreamResponse(
                    id=content["meta_info"]["id"],
                    created=int(time.time()),
                    choices=[],
                    model=request.model,
                    usage=usage,
                )
                yield f"data: {usage_chunk.model_dump_json()}\n\n"

        except ValueError as e:
            if not stream_started:
                raise
            error = self.create_streaming_error_response(str(e))
            yield f"data: {error}\n\n"

        yield "data: [DONE]\n\n"

    async def _handle_non_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: ChatCompletionRequest,
        raw_request: Request,
    ) -> Union[ChatCompletionResponse, ErrorResponse, ORJSONResponse]:

        try:
            ret = await self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            ).__anext__()
        except ValueError as e:
            return self.create_error_response(str(e))

        if not isinstance(ret, list):
            ret = [ret]

        response = self._build_chat_response(
            request,
            ret,
            int(time.time()),
        )

        return response

    def _build_chat_response(
        self,
        request: ChatCompletionRequest,
        ret: List[Dict[str, Any]],
        created: int,
    ) -> Union[ChatCompletionResponse, ORJSONResponse]:

        if self.chat_encoding_spec == "kimi_k3":
            ret = [
                {
                    **item,
                    "meta_info": {
                        **item["meta_info"],
                        "prompt_tokens": self._reported_prompt_tokens(
                            item["meta_info"]
                        ),
                    },
                }
                for item in ret
            ]

        choices = []


        first_ret = ret[0]
        routed_experts = (
            None
            if request.return_meta_info
            else process_routed_experts_from_ret(first_ret, request)
        )
        cached_tokens_details = process_cached_tokens_details_from_ret(
            first_ret, request
        )
        spec_details = [
            detail
            for detail in (
                process_spec_tokens_details_from_ret(item, request) for item in ret
            )
            if detail is not None
        ]
        spec_tokens_details = (
            spec_details
            if request.n > 1
            else (spec_details[0] if spec_details else None)
        )
        response_sglext = None
        if routed_experts or cached_tokens_details or spec_tokens_details:
            response_sglext = SglExt(
                routed_experts=routed_experts,
                cached_tokens_details=cached_tokens_details,
                spec_tokens_details=spec_tokens_details,
            )

        for idx, ret_item in enumerate(ret):

            choice_logprobs = None
            if request.logprobs:
                choice_logprobs = self._process_response_logprobs(ret_item)


            hidden_states = process_hidden_states_from_ret(ret_item, request)

            finish_reason = ret_item["meta_info"]["finish_reason"]

            text = self._decode_response(ret_item)
            if isinstance(text, ErrorResponse):
                return ORJSONResponse(content=text.model_dump(), status_code=text.code)


            reasoning_text = None
            if self.reasoning_parser and request.separate_reasoning:
                force_reasoning = (
                    self.template_manager.force_reasoning
                    or self._get_reasoning_from_request(request)
                )
                try:
                    parser = ReasoningParser(
                        model_type=self.reasoning_parser,
                        stream_reasoning=False,
                        force_reasoning=force_reasoning,
                        request=request,
                        tokenizer=self.tokenizer_manager.tokenizer,
                        tool_call_parser_active=self._tool_call_parsing_active(request),
                    )
                    reasoning_text, text = parser.parse_non_stream(text)
                except Exception as e:
                    logger.error(f"Reasoning parsing error: {e}")
                    return self.create_error_response(
                        "Failed to parse reasoning content",
                        err_type="InternalServerError",
                        status_code=500,
                    )


            tool_calls = None
            effective_tools = self._effective_tools(request)
            if self._tool_call_parsing_active(request):
                history_tool_calls_cnt = self._get_history_tool_calls_cnt(request)
                tool_calls, text, finish_reason = self._process_tool_calls(
                    text,
                    effective_tools,
                    finish_reason,
                    request.tool_choice,
                    history_tool_calls_cnt,
                )


            choice_prompt_token_ids = (
                ret_item.get("prompt_token_ids")
                if request.return_prompt_token_ids or request.return_token_ids
                else None
            )
            choice_token_ids = (
                ret_item["output_ids"] if request.return_token_ids else None
            )

            choice_meta_info = (
                ret_item["meta_info"] if request.return_meta_info else None
            )

            reasoning_text, tool_calls = self._get_parsed_response_fields(
                reasoning_text, tool_calls
            )

            choice_data = ChatCompletionResponseChoice(
                index=idx,
                message=ChatMessage(
                    role="assistant",
                    content=text if text else "",
                    tool_calls=tool_calls,
                    reasoning_content=reasoning_text if reasoning_text else None,
                ),
                logprobs=choice_logprobs,
                finish_reason=finish_reason["type"] if finish_reason else None,
                matched_stop=(
                    finish_reason["matched"]
                    if finish_reason and "matched" in finish_reason
                    else None
                ),
                hidden_states=hidden_states,
                prompt_token_ids=choice_prompt_token_ids,
                response_token_ids=choice_token_ids,
                meta_info=choice_meta_info,
            )
            choices.append(choice_data)


        image_tokens = sum(
            ret[i]["meta_info"].get("image_tokens", 0)
            for i in range(0, len(ret), request.n)
        )
        audio_tokens = sum(
            ret[i]["meta_info"].get("audio_tokens", 0)
            for i in range(0, len(ret), request.n)
        )
        video_tokens = sum(
            ret[i]["meta_info"].get("video_tokens", 0)
            for i in range(0, len(ret), request.n)
        )
        usage = UsageProcessor.calculate_response_usage(
            ret,
            n_choices=request.n,
            enable_cache_report=self.tokenizer_manager.server_args.enable_cache_report,
            image_tokens=image_tokens,
            audio_tokens=audio_tokens,
            video_tokens=video_tokens,
        )

        return ChatCompletionResponse(
            id=ret[0]["meta_info"]["id"],
            created=created,
            model=request.model,
            choices=choices,
            usage=usage,
            metadata=build_endpoint_weight_version_metadata(ret[0]["meta_info"]),
            sglext=response_sglext,
        )

    def _process_logprobs_tokens(
        self, logprobs: LogProbs, use_token_index: bool = False
    ) -> List[ChatCompletionTokenLogprob]:

        token_logprobs = []

        for token_idx, (token, logprob) in enumerate(
            zip(logprobs.tokens, logprobs.token_logprobs)
        ):
            token_bytes = list(token.encode("utf-8"))
            top_logprobs = []
            if logprobs.top_logprobs:


                top_logprobs_idx = token_idx if use_token_index else 0
                for top_token, top_logprob in logprobs.top_logprobs[
                    top_logprobs_idx
                ].items():
                    top_token_bytes = list(top_token.encode("utf-8"))
                    top_logprobs.append(
                        TopLogprob(
                            token=top_token,
                            bytes=top_token_bytes,
                            logprob=top_logprob,
                        )
                    )
            token_logprobs.append(
                ChatCompletionTokenLogprob(
                    token=token,
                    bytes=token_bytes,
                    logprob=logprob,
                    top_logprobs=top_logprobs,
                )
            )

        return token_logprobs

    def _process_response_logprobs(self, ret_item: Dict[str, Any]) -> ChoiceLogprobs:

        logprobs = to_openai_style_logprobs(
            output_token_logprobs=ret_item["meta_info"]["output_token_logprobs"],
            output_top_logprobs=ret_item["meta_info"].get("output_top_logprobs", None),
        )

        token_logprobs = self._process_logprobs_tokens(logprobs, use_token_index=True)
        return ChoiceLogprobs(content=token_logprobs)

    def _process_tool_call_id(
        self,
        call_item: ToolCallItem,
        history_tool_calls_cnt: int,
    ) -> str:

        if self.tool_call_parser == "kimi_k3":
            return f"{call_item.name}:{history_tool_calls_cnt + call_item.tool_index}"
        if self.tool_call_parser != "kimi_k2":

            tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
            return tool_call_id
        tool_call_id = (
            f"functions.{call_item.name}:"
            f"{history_tool_calls_cnt + call_item.tool_index}"
        )
        logger.debug(
            f"Process tool call idx, parser: {self.tool_call_parser}, tool_call_id: {tool_call_id}, history_cnt: {history_tool_calls_cnt}"
        )
        return tool_call_id

    def _process_tool_calls(
        self,
        text: str,
        tools: List[Any],
        finish_reason: Dict[str, Any],
        tool_choice: Optional[Union[str, ToolChoice]] = None,
        history_tool_calls_cnt: int = 0,
    ) -> ToolCallProcessingResult:


        is_required = tool_choice == "required" or isinstance(tool_choice, ToolChoice)


        if self.tool_call_parser:
            parser = FunctionCallParser(
                tools, self.tool_call_parser, tokenizer=self.tokenizer_manager.tokenizer
            )
            detector_owns_format = (
                parser.detector.supports_structural_tag()
                or parser.detector.parses_required_natively()
            )
            should_try_parser = not is_required or detector_owns_format
            if should_try_parser and parser.has_tool_call(text):
                try:
                    text, call_info_list = parser.parse_non_stream(text)
                    if not call_info_list:
                        logger.warning(
                            "Tool call marker present but no complete call parsed "
                            "from %s output; dropping the incomplete call",
                            self.tool_call_parser,
                        )
                        logger.debug(
                            "Unparsed tool call output (%d chars): %r",
                            len(text),
                            text[:2000],
                        )
                        return ToolCallProcessingResult(None, text, finish_reason)

                    tool_calls = []
                    for call_info in call_info_list:
                        tool_id = self._process_tool_call_id(
                            call_info, history_tool_calls_cnt
                        )
                        tool_calls.append(
                            ToolCall(
                                id=tool_id,
                                index=getattr(call_info, "tool_index", None),
                                function=FunctionResponse(
                                    name=call_info.name,
                                    arguments=call_info.parameters,
                                ),
                            )
                        )
                    if finish_reason["type"] == "stop":
                        finish_reason["type"] = "tool_calls"
                        finish_reason["matched"] = None
                    return ToolCallProcessingResult(tool_calls, text, finish_reason)
                except Exception as e:
                    logger.error(f"Tool call parsing error: {e}")
                    return ToolCallProcessingResult(None, text, finish_reason)

            if is_required and detector_owns_format:
                logger.warning(
                    "Required tool call missing from %s output (%d chars)",
                    self.tool_call_parser,
                    len(text),
                )
                logger.debug("Unparsed required tool call output: %r", text[:2000])
                return ToolCallProcessingResult(None, text, finish_reason)


        if is_required:
            original_finish_type = finish_reason["type"]
            if finish_reason["type"] == "stop":
                finish_reason["type"] = "tool_calls"
                finish_reason["matched"] = None
            try:
                tool_call_data = orjson.loads(text)
                if isinstance(tool_call_data, dict):
                    tool_call_data = [tool_call_data]
                if not isinstance(tool_call_data, list):
                    raise ValueError(
                        "expected a JSON array of tool calls, got "
                        f"{type(tool_call_data).__name__}"
                    )
                if not all(
                    isinstance(tool, dict) and "name" in tool for tool in tool_call_data
                ):
                    raise ValueError(
                        "every tool call must be a JSON object with a 'name'"
                    )
                tool_calls = []
                for i, tool in enumerate(tool_call_data):
                    parameters = json.dumps(
                        tool.get("parameters", {}), ensure_ascii=False
                    )
                    call_info = ToolCallItem(
                        tool_index=i,
                        name=tool["name"],
                        parameters=parameters,
                    )
                    tool_id = self._process_tool_call_id(
                        call_info, history_tool_calls_cnt
                    )
                    tool_calls.append(
                        ToolCall(
                            id=tool_id,
                            index=i,
                            function=FunctionResponse(
                                name=tool["name"],
                                arguments=parameters,
                            ),
                        )
                    )
                return ToolCallProcessingResult(tool_calls, "", finish_reason)
            except Exception as e:
                logger.error(f"Tool call parsing error: {e}")
                logger.debug("Unparsed required tool call output: %r", text[:2000])
                finish_reason["type"] = original_finish_type
                return ToolCallProcessingResult(None, text, finish_reason)

        return ToolCallProcessingResult(None, text, finish_reason)

    def _process_streaming_logprobs(
        self,
        content: Dict[str, Any],
        n_prev_token: int,
        total_output_logprobs: int,
    ) -> ChoiceLogprobs:

        output_token_logprobs = content["meta_info"]["output_token_logprobs"]
        output_top_logprobs = content["meta_info"].get("output_top_logprobs", [])
        if not self.tokenizer_manager.server_args.incremental_streaming_output:
            output_token_logprobs = output_token_logprobs[
                n_prev_token:total_output_logprobs
            ]
            output_top_logprobs = output_top_logprobs[
                n_prev_token:total_output_logprobs
            ]
        logprobs = to_openai_style_logprobs(
            output_token_logprobs=output_token_logprobs,
            output_top_logprobs=output_top_logprobs,
        )

        token_logprobs = self._process_logprobs_tokens(logprobs, use_token_index=False)
        return ChoiceLogprobs(content=token_logprobs)

    def _process_reasoning_stream(
        self,
        index: int,
        delta: str,
        reasoning_parser_dict: Dict[int, ReasoningParser],
        content: Dict[str, Any],
        request: ChatCompletionRequest,
        finish_reason_type: Optional[str] = None,
    ) -> tuple[Optional[str], str]:

        if index not in reasoning_parser_dict:
            is_force_reasoning = (
                self.template_manager.force_reasoning
                or self._get_reasoning_from_request(request)
            )
            reasoning_parser_dict[index] = ReasoningParser(
                self.reasoning_parser,
                request.stream_reasoning,
                is_force_reasoning,
                request,
                tokenizer=self.tokenizer_manager.tokenizer,
                tool_call_parser_active=self._tool_call_parsing_active(request),
            )
        reasoning_parser = reasoning_parser_dict[index]
        reasoning_text, normal_text = reasoning_parser.parse_stream_chunk(delta)
        if finish_reason_type is not None and finish_reason_type != "abort":
            end_reasoning_text, end_normal_text = reasoning_parser.parse_stream_end()
            if end_reasoning_text:
                reasoning_text = (reasoning_text or "") + end_reasoning_text
            if end_normal_text:
                normal_text = (normal_text or "") + end_normal_text
        return reasoning_text, normal_text

    def _get_history_tool_calls_cnt(self, request: ChatCompletionRequest) -> int:

        messages = getattr(request, "messages", [])
        idx = 0
        for msg in messages:
            if msg.role == "assistant":
                tool_calls = getattr(msg, "tool_calls", None)
                idx += len(list(tool_calls)) if tool_calls is not None else 0
        return idx

    def _patch_reasoning_skip_special_tokens(
        self, request: ChatCompletionRequest
    ) -> None:

        if self.reasoning_parser == "apertus2509":
            request.skip_special_tokens = False
        if self.reasoning_parser == "kimi_k3" or self.chat_encoding_spec == "kimi_k3":
            request.skip_special_tokens = False

        if (
            self.reasoning_parser in ["mistral"]
            and request.reasoning_effort is not None
            and request.reasoning_effort != "none"
        ):
            request.skip_special_tokens = False
        elif self.reasoning_parser == "inkling":
            request.skip_special_tokens = False
        elif self.reasoning_parser == "muse":
            request.skip_special_tokens = False

    def supports_native_reasoning_history(self) -> bool:

        return chat_encoding.spec_owns_reasoning_history(self.chat_encoding_spec)

    def wrap_reasoning_history(self, reasoning_text: str) -> str:

        if self._reasoning_detector is None:
            raise ValueError(
                "Cannot rewrap thinking history: no reasoning detector is "
                "configured for this model"
            )
        d = self._reasoning_detector
        return (
            f"{d.think_start_token}{d.think_start_self_label}"
            f"{reasoning_text}\n{d.think_end_token}"
        )

    def _reasoning_default_mode(self) -> Optional[str]:
        if self._reasoning_detector is None:
            return None
        return self._reasoning_detector.reasoning_default

    def _get_reasoning_toggle_param(self) -> Optional[str]:

        config = self.template_manager.reasoning_config
        if config is not None:
            return config.toggle_param

        mode = self._reasoning_default_mode()
        if mode in ("thinking", "enable_thinking"):
            return mode
        if mode in ("explicit_thinking", "explicit_enable_thinking"):
            return mode.replace("explicit_", "")
        return None

    def apply_reasoning_enabled(
        self, request: ChatCompletionRequest, enabled: bool
    ) -> None:

        if not self.reasoning_parser:
            if enabled:
                raise ValueError(
                    "Anthropic thinking is not supported for models without "
                    "a reasoning parser"
                )
            return

        if self.reasoning_parser == "hunyuan":
            request.reasoning_effort = "medium" if enabled else "no_think"
            return

        if self.reasoning_parser == "inkling":

            if not enabled:
                request.reasoning_effort = "none"
                return

        config = self.template_manager.reasoning_config
        is_mistral = (config is not None and config.special_case == "mistral") or (
            config is None and self._reasoning_default_mode() == "mistral"
        )
        if is_mistral:
            request.reasoning_effort = "medium" if enabled else "none"
            return

        is_always_on = (config is not None and config.special_case == "always") or (
            config is None and self._reasoning_default_mode() == "always"
        )
        if is_always_on:
            if not enabled:
                raise ValueError(
                    f"Reasoning parser '{self.reasoning_parser}' is always-on "
                    f"and cannot be disabled via Anthropic thinking"
                )
            return

        toggle_param = self._get_reasoning_toggle_param()


        config = self.template_manager.reasoning_config
        read_side_supported = toggle_param is not None and (
            config is None or config.default_enabled is not None
        )
        if not read_side_supported:
            if not enabled:
                return
            raise ValueError(
                f"Anthropic thinking is not supported for reasoning parser "
                f"'{self.reasoning_parser}'"
            )

        chat_template_kwargs = dict(request.chat_template_kwargs or {})
        chat_template_kwargs[toggle_param] = enabled
        request.chat_template_kwargs = chat_template_kwargs

    def _get_reasoning_from_request(self, request: ChatCompletionRequest) -> bool:

        if not self.reasoning_parser:
            return False

        if self.reasoning_parser == "minimax-m3":


            return (request.chat_template_kwargs or {}).get(
                "thinking_mode"
            ) == "enabled"

        if self.reasoning_parser == "hunyuan":


            return request.reasoning_effort not in (None, "none", "no_think")

        config = self.template_manager.reasoning_config
        if config is None:


            mode = (
                self._reasoning_detector.reasoning_default
                if self._reasoning_detector is not None
                else None
            )
            if mode is None:
                return False
            if mode == "always":
                return True
            if mode == "mistral":
                return (
                    request.reasoning_effort is not None
                    and request.reasoning_effort != "none"
                )
            if mode in ("thinking", "enable_thinking"):
                return (
                    not request.chat_template_kwargs
                    or request.chat_template_kwargs.get(mode) is not False
                )
            if mode in ("explicit_thinking", "explicit_enable_thinking"):
                toggle = mode.replace("explicit_", "")
                return (
                    request.chat_template_kwargs is not None
                    and request.chat_template_kwargs.get(toggle) is True
                )
            logger.warning(
                "Unknown reasoning_default mode '%s', defaulting to reasoning disabled",
                mode,
            )
            return False

        if config.special_case == "always":
            return True

        if config.special_case == "mistral":
            return (
                request.reasoning_effort is not None
                and request.reasoning_effort != "none"
            )

        if config.toggle_param is None or config.default_enabled is None:
            return False

        if config.default_enabled:
            return (
                not request.chat_template_kwargs
                or request.chat_template_kwargs.get(config.toggle_param) is not False
            )
        return (
            request.chat_template_kwargs is not None
            and request.chat_template_kwargs.get(config.toggle_param) is True
        )

    async def _process_tool_call_stream(
        self,
        index: int,
        delta: str,
        parser_dict: Dict[int, FunctionCallParser],
        content: Dict[str, Any],
        request: ChatCompletionRequest,
        has_tool_calls: Dict[int, bool],
        continuous_usage_stats: bool = False,
        flush: bool = False,
    ):

        effective_tools = self._effective_tools(request)
        if index not in parser_dict:
            is_required = request.tool_choice == "required" or isinstance(
                request.tool_choice, ToolChoice
            )


            if is_required:
                use_native_parser = False
                if self.tool_call_parser:
                    probe = FunctionCallParser(
                        tools=effective_tools,
                        tool_call_parser=self.tool_call_parser,
                        tokenizer=self.tokenizer_manager.tokenizer,
                    )
                    use_native_parser = (
                        probe.detector.supports_structural_tag()
                        or probe.detector.parses_required_natively()
                    )
                if use_native_parser:
                    parser_dict[index] = probe
                else:
                    parser_dict[index] = JsonArrayParser()
            else:
                parser_dict[index] = FunctionCallParser(
                    tools=effective_tools,
                    tool_call_parser=self.tool_call_parser,
                    tokenizer=self.tokenizer_manager.tokenizer,
                )

        parser = parser_dict[index]


        if isinstance(parser, JsonArrayParser):
            result = parser.parse_streaming_increment(delta, effective_tools)
            normal_text, calls = result.normal_text, result.calls
        else:
            normal_text, calls = parser.parse_stream_chunk(delta)
            if flush:
                end_text, end_calls = parser.parse_stream_end()
                normal_text = (normal_text or "") + end_text
                calls = list(calls) + end_calls


        if normal_text:
            choice_data = ChatCompletionResponseStreamChoice(
                index=index,
                delta=DeltaMessage(content=normal_text),
                finish_reason=None,
            )
            chunk = ChatCompletionStreamResponse(
                id=content["meta_info"]["id"],
                created=int(time.time()),
                choices=[choice_data],
                model=request.model,
            )


            if continuous_usage_stats:
                prompt_tokens = self._reported_prompt_tokens(content["meta_info"])
                completion_tokens = content["meta_info"].get("completion_tokens", 0)
                reasoning_tokens = content["meta_info"].get("reasoning_tokens", 0)
                chunk.usage = UsageProcessor.calculate_token_usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    reasoning_tokens=reasoning_tokens,
                    cached_tokens=self._continuous_usage_cached_details(content),
                )

            yield f"data: {chunk.model_dump_json()}\n\n"


        history_tool_calls_cnt = self._get_history_tool_calls_cnt(request)
        for call_item in calls:

            has_tool_calls[index] = True


            if call_item.name:

                tool_call_id = self._process_tool_call_id(
                    call_item, history_tool_calls_cnt
                )
                function_name = call_item.name
            else:

                tool_call_id = None
                function_name = None

            tool_call = ToolCall(
                id=tool_call_id,
                index=call_item.tool_index,
                function=FunctionResponse(
                    name=function_name,
                    arguments=call_item.parameters,
                ),
            )

            choice_data = ChatCompletionResponseStreamChoice(
                index=index,
                delta=DeltaMessage(tool_calls=[tool_call]),
                finish_reason=None,
            )
            chunk = ChatCompletionStreamResponse(
                id=content["meta_info"]["id"],
                created=int(time.time()),
                choices=[choice_data],
                model=request.model,
            )


            if continuous_usage_stats:
                prompt_tokens = self._reported_prompt_tokens(content["meta_info"])
                completion_tokens = content["meta_info"].get("completion_tokens", 0)
                reasoning_tokens = content["meta_info"].get("reasoning_tokens", 0)
                chunk.usage = UsageProcessor.calculate_token_usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    reasoning_tokens=reasoning_tokens,
                    cached_tokens=self._continuous_usage_cached_details(content),
                )

            yield f"data: {chunk.model_dump_json()}\n\n"

    def _check_for_unstreamed_tool_args(
        self,
        parser: Union[FunctionCallParser, JsonArrayParser],
        content: Dict[str, Any],
        request: ChatCompletionRequest,
        index: int,
    ) -> Optional[str]:


        detector = parser.detector if hasattr(parser, "detector") else parser


        if (
            not hasattr(detector, "prev_tool_call_arr")
            or not detector.prev_tool_call_arr
        ):
            return None

        if (
            not hasattr(detector, "streamed_args_for_tool")
            or not detector.streamed_args_for_tool
        ):
            return None


        tool_index = len(detector.prev_tool_call_arr) - 1
        if tool_index < 0 or tool_index >= len(detector.streamed_args_for_tool):
            return None


        expected_args = detector.prev_tool_call_arr[tool_index].get("arguments", {})
        if isinstance(expected_args, str):
            expected_call = expected_args
        else:
            expected_call = json.dumps(expected_args, ensure_ascii=False)
        actual_call = detector.streamed_args_for_tool[tool_index]


        remaining_call = (
            expected_call[len(actual_call) :]
            if expected_call.startswith(actual_call)
            else ""
        )

        if remaining_call:

            tool_call = ToolCall(
                id=None,
                index=tool_index,
                function=FunctionResponse(
                    name=None,
                    arguments=remaining_call,
                ),
            )

            choice_data = ChatCompletionResponseStreamChoice(
                index=index,
                delta=DeltaMessage(tool_calls=[tool_call]),
                finish_reason=None,
            )

            chunk = ChatCompletionStreamResponse(
                id=content["meta_info"]["id"],
                created=int(time.time()),
                choices=[choice_data],
                model=request.model,
            )

            return f"data: {chunk.model_dump_json()}\n\n"

        return None
