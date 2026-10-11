import logging
from typing import Union

import torch
from fastapi import Request
from fastapi.responses import ORJSONResponse

from sglang.srt.entrypoints.openai.protocol import (
    ErrorResponse,
    ScoringRequest,
    ScoringResponse,
    UsageInfo,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase

logger = logging.getLogger(__name__)


class OpenAIServingScore(OpenAIServingBase):


    def _request_id_prefix(self) -> str:
        return "score-"

    def _convert_to_internal_request(
        self,
        request: ScoringRequest,
        raw_request: Request = None,
    ) -> tuple[ScoringRequest, ScoringRequest]:


        return request, request

    async def _handle_non_streaming_request(
        self,
        adapted_request: ScoringRequest,
        request: ScoringRequest,
        raw_request: Request,
    ) -> Union[ScoringResponse, ErrorResponse]:

        try:

            query_embed_overrides = (
                [
                    torch.tensor(v, dtype=torch.float32)
                    for v in request.query_embed_overrides
                ]
                if request.query_embed_overrides is not None
                else None
            )

            item_embed_overrides = (
                [
                    (
                        [torch.tensor(v, dtype=torch.float32) for v in per_item]
                        if per_item is not None
                        else None
                    )
                    for per_item in request.item_embed_overrides
                ]
                if request.item_embed_overrides is not None
                else None
            )

            result = await self.tokenizer_manager.score_request(
                query=request.query,
                items=request.items,
                label_token_ids=request.label_token_ids,
                apply_softmax=request.apply_softmax,
                item_first=request.item_first,
                embed_override_token_id=request.embed_override_token_id,
                query_embed_overrides=query_embed_overrides,
                item_embed_overrides=item_embed_overrides,
                request=raw_request,
                return_pooled_hidden_states=request.return_pooled_hidden_states,
            )

            phs_as_lists = None
            if result.pooled_hidden_states is not None:
                phs_as_lists = [
                    t.tolist() if t is not None else None
                    for t in result.pooled_hidden_states
                ]

            response = ScoringResponse(
                scores=result.scores,
                pooled_hidden_states=phs_as_lists,
                model=request.model,
                usage=UsageInfo(
                    prompt_tokens=result.prompt_tokens,
                    total_tokens=result.prompt_tokens,
                ),
            )
            return ORJSONResponse(content=response.model_dump())

        except ValueError as e:
            return self.create_error_response(str(e))
