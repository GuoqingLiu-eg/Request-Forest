# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from __future__ import annotations

import asyncio
import io
import logging
import math
import time
import uuid
from typing import TYPE_CHECKING, AsyncGenerator, List, Optional, Union

from fastapi import Request, WebSocket
from fastapi.responses import ORJSONResponse, Response, StreamingResponse

from sglang.srt.entrypoints.openai.audio_chunking import split_audio_energy_aware
from sglang.srt.entrypoints.openai.protocol import (
    DeltaMessage,
    ErrorResponse,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionStreamChoice,
    TranscriptionStreamResponse,
    TranscriptionUsage,
    TranscriptionVerboseResponse,
)
from sglang.srt.entrypoints.openai.realtime import (
    handle_realtime_transcription,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.entrypoints.openai.streaming_asr import (
    StreamingASRState,
    needs_space,
    process_asr_chunk,
    split_audio_chunks,
)
from sglang.srt.entrypoints.openai.transcription_adapters import resolve_adapter
from sglang.srt.managers.io_struct import GenerateReqInput

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


class OpenAIServingTranscription(OpenAIServingBase):


    def __init__(self, tokenizer_manager: TokenizerManager):
        super().__init__(tokenizer_manager)
        model_config = tokenizer_manager.model_config
        self._adapter = resolve_adapter(
            getattr(model_config.hf_config, "architectures", [])
        )


        self._session_semaphore = asyncio.Semaphore(
            tokenizer_manager.server_args.asr_max_concurrent_sessions
        )

    def _request_id_prefix(self) -> str:
        return "trsc-"

    def _validate_request(self, request: TranscriptionRequest) -> Optional[str]:


        return None

    def _convert_to_internal_request(
        self,
        request: TranscriptionRequest,
        raw_request: Request = None,
    ) -> tuple[GenerateReqInput, TranscriptionRequest]:

        if getattr(request, "_fused_autodetect", False):
            sampling_params = self._adapter.build_fused_autodetect_params(request)
        else:
            sampling_params = self._adapter.build_sampling_params(request)
        adapted_request = GenerateReqInput(
            text="",
            audio_data=request.audio_data,
            sampling_params=sampling_params,
            stream=request.stream,
            modalities=["audio"],
            routing_key=self.extract_routing_key(raw_request),
        )

        return adapted_request, request

    @staticmethod
    def _get_audio_duration(audio_data: bytes) -> float:

        try:
            import soundfile as sf

            info = sf.info(io.BytesIO(audio_data))
            return info.duration
        except Exception:


            try:
                from sglang.srt.utils import load_audio

                audio = load_audio(audio_data, sr=16000, mono=True)
                return audio.shape[-1] / 16000.0
            except Exception as e:
                logger.warning(f"Could not calculate audio duration: {e}")
                return 0.0

    async def create_transcription(
        self,
        audio_data: bytes,
        model: str,
        language: Optional[str],
        response_format: str,
        temperature: float,
        stream: bool,
        raw_request: Request,
        timestamp_granularities: Optional[List[str]] = None,
    ) -> Union[
        TranscriptionResponse,
        TranscriptionVerboseResponse,
        StreamingResponse,
        Response,
        ORJSONResponse,
    ]:


        audio_duration_s = await asyncio.to_thread(self._get_audio_duration, audio_data)


        audio_chunks: Optional[List[bytes]] = None
        chunk_offsets_s: Optional[List[float]] = None
        max_clip_s = self._adapter.max_audio_clip_s
        if max_clip_s is not None and audio_duration_s > max_clip_s:
            split_error = (
                f"Failed to split audio longer than {max_clip_s:g} seconds into "
                "supported chunks."
            )
            try:


                audio_chunks, chunk_offsets_s = await asyncio.to_thread(
                    split_audio_energy_aware, audio_data, max_clip_s
                )
            except Exception as e:
                logger.warning(
                    "Failed to split %.1fs audio into chunks of <=%ss: %s",
                    audio_duration_s,
                    max_clip_s,
                    e,
                )
                return self.create_error_response(split_error)
            else:
                if (
                    not audio_chunks
                    or len(audio_chunks) <= 1
                    or chunk_offsets_s is None
                    or len(chunk_offsets_s) != len(audio_chunks)
                ):
                    logger.error(
                        "Audio splitter returned an invalid result for %.1fs audio: "
                        "%d chunks and %s offsets",
                        audio_duration_s,
                        len(audio_chunks or []),
                        (
                            "no"
                            if chunk_offsets_s is None
                            else str(len(chunk_offsets_s))
                        ),
                    )
                    return self.create_error_response(split_error)
                logger.info(
                    "Split %.1fs audio into %d chunks of <=%ss for transcription",
                    audio_duration_s,
                    len(audio_chunks),
                    max_clip_s,
                )


        use_fused = language is None and self._adapter.supports_language_detection


        request = TranscriptionRequest(
            audio_data=audio_data,
            model=model,
            language=language,
            response_format=response_format,
            temperature=temperature,
            timestamp_granularities=timestamp_granularities,
            stream=stream,
            audio_duration_s=audio_duration_s,
        )
        if use_fused:
            request._fused_autodetect = True


            request._fused_ts_variant = bool(timestamp_granularities)
        if audio_chunks is not None and len(audio_chunks) > 1:
            request._audio_chunks = audio_chunks
            request._chunk_offsets_s = chunk_offsets_s


        return await self.handle_request(request, raw_request)

    async def _handle_non_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: TranscriptionRequest,
        raw_request: Request,
    ) -> Union[
        TranscriptionResponse,
        TranscriptionVerboseResponse,
        ErrorResponse,
        ORJSONResponse,
        Response,
    ]:

        if getattr(request, "_audio_chunks", None):
            return await self._handle_chunked_non_streaming_request(
                adapted_request, request, raw_request
            )

        try:
            ret = await self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            ).__anext__()
        except ValueError as e:
            return self.create_error_response(str(e))

        text = self._finalize_text(request, ret.get("text", ""))

        usage = TranscriptionUsage(seconds=int(math.ceil(request.audio_duration_s)))


        if request.response_format == "text":
            return Response(content=text, media_type="text/plain")

        if request.response_format == "verbose_json":
            tokenizer = self.tokenizer_manager.tokenizer
            return self._adapter.build_verbose_response(
                request, text, ret, tokenizer, usage
            )


        return TranscriptionResponse(text=text, usage=usage)

    def _finalize_text(
        self, request: TranscriptionRequest, raw_text: str, strip: bool = True
    ) -> str:

        text = self._adapter.postprocess_text(raw_text)
        if not getattr(request, "_fused_autodetect", False):
            return text
        lang, visible = self._adapter.parse_fused_output(
            text,
            ts_variant=getattr(request, "_fused_ts_variant", False),
            strip=strip,
        )
        if visible is None:
            logger.warning(
                "Fused auto-detect parse failed on non-streaming response; "
                "falling back to raw-text scrub."
            )
            return self._adapter.strip_special_tokens(text)
        if lang is not None and visible.strip() and request.language is None:
            request.language = lang
            logger.info("Auto-detected language: '%s'", lang)
        return visible

    def _build_chunk_request(
        self,
        adapted_request: GenerateReqInput,
        chunk_audio: bytes,
        stream: bool,
    ) -> GenerateReqInput:

        sampling_params = adapted_request.sampling_params
        assert isinstance(sampling_params, dict)
        chunk_request = GenerateReqInput(
            text="",
            audio_data=chunk_audio,
            sampling_params=dict(sampling_params),
            stream=stream,
            modalities=["audio"],
            routing_key=adapted_request.routing_key,
        )
        chunk_request.received_time = adapted_request.received_time
        return chunk_request

    def _abort_chunk_requests(self, chunk_requests: List[GenerateReqInput]) -> None:


        def abort_assigned_rids():


            for chunk_request in chunk_requests:
                if isinstance(chunk_request.rid, str):
                    self.tokenizer_manager.abort_request(chunk_request.rid)

        abort_assigned_rids()

        async def abort_after_dispatch_window():
            await asyncio.sleep(2)
            abort_assigned_rids()

        asyncio.create_task(abort_after_dispatch_window())

    async def _handle_chunked_non_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: TranscriptionRequest,
        raw_request: Request,
    ) -> Union[
        TranscriptionResponse,
        TranscriptionVerboseResponse,
        ErrorResponse,
        ORJSONResponse,
        Response,
    ]:

        chunk_requests = [
            self._build_chunk_request(adapted_request, chunk_audio, stream=False)
            for chunk_audio in request._audio_chunks
        ]
        rets = []
        try:
            for chunk_request in chunk_requests:
                ret = await self.tokenizer_manager.generate_request(
                    chunk_request, raw_request
                ).__anext__()
                rets.append(ret)
        except BaseException as e:


            self._abort_chunk_requests(chunk_requests)
            if isinstance(e, ValueError):
                return self.create_error_response(str(e))
            raise

        fused = getattr(request, "_fused_autodetect", False)


        texts = [
            self._finalize_text(request, ret.get("text", ""), strip=False)
            for ret in rets
        ]
        text = "".join(texts)
        if fused:
            text = text.strip()

        usage = TranscriptionUsage(seconds=int(math.ceil(request.audio_duration_s)))

        if request.response_format == "text":
            return Response(content=text, media_type="text/plain")

        if request.response_format == "verbose_json":
            return self._adapter.build_verbose_response_chunked(
                request,
                text,
                rets,
                request._chunk_offsets_s,
                self.tokenizer_manager.tokenizer,
                usage,
            )

        return TranscriptionResponse(text=text, usage=usage)

    async def _handle_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: TranscriptionRequest,
        raw_request: Request,
    ) -> StreamingResponse:

        if self._adapter.supports_chunked_streaming:


            return StreamingResponse(
                self._generate_chunked_asr_stream(
                    adapted_request, request, raw_request
                ),
                media_type="text/event-stream",
            )
        if getattr(request, "_audio_chunks", None):


            return StreamingResponse(
                self._generate_long_audio_stream(adapted_request, request, raw_request),
                media_type="text/event-stream",
            )
        return StreamingResponse(
            self._generate_transcription_stream(adapted_request, request, raw_request),
            media_type="text/event-stream",
            background=self.tokenizer_manager.create_abort_task(adapted_request),
        )

    async def _generate_transcription_stream(
        self,
        adapted_request: GenerateReqInput,
        request: TranscriptionRequest,
        raw_request: Request,
    ) -> AsyncGenerator[str, None]:

        created_time = int(time.time())
        request_id = f"{self._request_id_prefix()}{uuid.uuid4().hex}"
        model = request.model
        visible_buffer = ""

        fused_mode = getattr(request, "_fused_autodetect", False)
        ts_variant = getattr(request, "_fused_ts_variant", False)


        incremental = getattr(
            self.tokenizer_manager.server_args,
            "incremental_streaming_output",
            False,
        )
        cumulative_text = ""

        try:
            async for content in self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            ):
                finish_reason = content["meta_info"]["finish_reason"]
                finish_reason_type = finish_reason["type"] if finish_reason else None

                chunk_text = content.get("text", "")
                if incremental:
                    cumulative_text += chunk_text
                else:
                    cumulative_text = chunk_text

                if fused_mode:
                    lang, visible = self._adapter.parse_fused_output(
                        cumulative_text, ts_variant=ts_variant
                    )
                    if visible is None:


                        if not finish_reason_type:
                            continue


                        logger.warning(
                            "Fused auto-detect stream finished before prefix "
                            "was parseable; returning detection-failed error."
                        )
                        error = self.create_streaming_error_response(
                            "language auto-detect failed: forced-prefix sentinel "
                            "was not produced before stream end"
                        )
                        yield f"data: {error}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    if (
                        lang is not None
                        and visible.strip()
                        and request.language is None
                    ):
                        request.language = lang
                        logger.info("Auto-detected language: '%s'", lang)
                else:
                    visible = cumulative_text

                delta = visible[len(visible_buffer) :]
                visible_buffer = visible


                if delta:
                    choice_data = TranscriptionStreamChoice(
                        delta=DeltaMessage(content=delta),
                        finish_reason=None,
                    )
                    chunk = TranscriptionStreamResponse(
                        id=request_id,
                        created=created_time,
                        model=model,
                        choices=[choice_data],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"


                if finish_reason_type:
                    choice_data = TranscriptionStreamChoice(
                        delta=DeltaMessage(),
                        finish_reason=finish_reason_type,
                    )
                    chunk = TranscriptionStreamResponse(
                        id=request_id,
                        created=created_time,
                        model=model,
                        choices=[choice_data],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"

        except ValueError as e:
            error = self.create_streaming_error_response(str(e))
            yield f"data: {error}\n\n"

        yield "data: [DONE]\n\n"

    async def _generate_long_audio_stream(
        self,
        adapted_request: GenerateReqInput,
        request: TranscriptionRequest,
        raw_request: Request,
    ) -> AsyncGenerator[str, None]:

        created_time = int(time.time())
        request_id = f"{self._request_id_prefix()}{uuid.uuid4().hex}"
        model = request.model
        fused_mode = getattr(request, "_fused_autodetect", False)
        ts_variant = getattr(request, "_fused_ts_variant", False)
        incremental = getattr(
            self.tokenizer_manager.server_args,
            "incremental_streaming_output",
            False,
        )

        def _frame(delta: Optional[str], finish_reason: Optional[str] = None) -> str:
            chunk = TranscriptionStreamResponse(
                id=request_id,
                created=created_time,
                model=model,
                choices=[
                    TranscriptionStreamChoice(
                        delta=DeltaMessage(content=delta) if delta else DeltaMessage(),
                        finish_reason=finish_reason,
                    )
                ],
            )
            return f"data: {chunk.model_dump_json()}\n\n"

        finish_reason_type = None
        in_flight: Optional[GenerateReqInput] = None
        emitted_text = False
        try:
            for chunk_audio in request._audio_chunks:
                if await raw_request.is_disconnected():
                    break
                in_flight = self._build_chunk_request(
                    adapted_request, chunk_audio, stream=True
                )
                cumulative_text = ""
                visible_buffer = ""
                chunk_finish_reason = None
                strip_chunk = not emitted_text
                async for content in self.tokenizer_manager.generate_request(
                    in_flight, raw_request
                ):
                    finish_reason = content["meta_info"]["finish_reason"]
                    chunk_finish_reason = (
                        finish_reason["type"] if finish_reason else None
                    )

                    chunk_text = content.get("text", "")
                    if incremental:
                        cumulative_text += chunk_text
                    else:
                        cumulative_text = chunk_text

                    if fused_mode:


                        lang, visible = self._adapter.parse_fused_output(
                            cumulative_text,
                            ts_variant=ts_variant,
                            strip=strip_chunk,
                        )
                        if visible is None:
                            if not chunk_finish_reason:
                                continue
                            logger.warning(
                                "Fused auto-detect stream finished before prefix "
                                "was parseable; returning detection-failed error."
                            )
                            error = self.create_streaming_error_response(
                                "language auto-detect failed: forced-prefix "
                                "sentinel was not produced before stream end"
                            )
                            yield f"data: {error}\n\n"
                            yield "data: [DONE]\n\n"
                            return
                        if (
                            lang is not None
                            and visible.strip()
                            and request.language is None
                        ):
                            request.language = lang
                            logger.info("Auto-detected language: '%s'", lang)
                    else:
                        visible = cumulative_text

                    delta = visible[len(visible_buffer) :]
                    visible_buffer = visible
                    if delta:
                        yield _frame(delta)
                        emitted_text = True

                in_flight = None
                if finish_reason_type in (None, "stop") and chunk_finish_reason:
                    finish_reason_type = chunk_finish_reason

            yield _frame(None, finish_reason=finish_reason_type or "stop")
        except ValueError as e:
            error = self.create_streaming_error_response(str(e))
            yield f"data: {error}\n\n"
        finally:


            if in_flight is not None and isinstance(in_flight.rid, str):
                self.tokenizer_manager.abort_request(in_flight.rid)

        yield "data: [DONE]\n\n"

    async def _generate_chunked_asr_stream(
        self,
        adapted_request: GenerateReqInput,
        request: TranscriptionRequest,
        raw_request: Request,
    ) -> AsyncGenerator[str, None]:

        created_time = int(time.time())
        request_id = f"{self._request_id_prefix()}{uuid.uuid4().hex}"
        model = request.model
        state = StreamingASRState(**self._adapter.chunked_streaming_config)


        last_char = ""

        try:
            chunks = split_audio_chunks(request.audio_data, state.chunk_size_sec)

            for i, chunk_audio in enumerate(chunks):
                if await raw_request.is_disconnected():
                    logger.info("[streaming_asr] client disconnected, stopping")
                    break
                is_last = i == len(chunks) - 1

                delta = await process_asr_chunk(
                    tokenizer_manager=self.tokenizer_manager,
                    adapter=self._adapter,
                    state=state,
                    audio_data=chunk_audio,
                    sampling_params=adapted_request.sampling_params,
                    is_last=is_last,
                    raw_request=raw_request,
                    routing_key=self.extract_routing_key(raw_request),
                )

                if delta:
                    for word in delta.split(" "):
                        if not word:
                            continue
                        content = f" {word}" if needs_space(last_char, word) else word
                        last_char = content[-1]
                        chunk_resp = TranscriptionStreamResponse(
                            id=request_id,
                            created=created_time,
                            model=model,
                            choices=[
                                TranscriptionStreamChoice(
                                    delta=DeltaMessage(content=content),
                                    finish_reason=None,
                                )
                            ],
                        )
                        yield f"data: {chunk_resp.model_dump_json()}\n\n"


            chunk_resp = TranscriptionStreamResponse(
                id=request_id,
                created=created_time,
                model=model,
                choices=[
                    TranscriptionStreamChoice(
                        delta=DeltaMessage(),
                        finish_reason="stop",
                    )
                ],
            )
            yield f"data: {chunk_resp.model_dump_json()}\n\n"

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("[streaming_asr] unrecoverable error")
            error = self.create_streaming_error_response(str(e))
            yield f"data: {error}\n\n"

        yield "data: [DONE]\n\n"

    async def handle_websocket(self, websocket: WebSocket) -> None:
        await handle_realtime_transcription(
            websocket,
            tokenizer_manager=self.tokenizer_manager,
            adapter=self._adapter,
            server_args=self.tokenizer_manager.server_args,
            session_semaphore=self._session_semaphore,
        )
