from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


def validate_request_forest_request(req: Req) -> Optional[str]:


    sampling_params = req.sampling_params
    if sampling_params.top_k > 1:
        return "Request Forest currently supports greedy decoding only."
    if sampling_params.frequency_penalty != 0.0:
        return "Request Forest does not yet support frequency_penalty."
    if sampling_params.presence_penalty != 0.0:
        return "Request Forest does not yet support presence_penalty."
    if sampling_params.repetition_penalty != 1.0:
        return "Request Forest does not yet support repetition_penalty."
    if sampling_params.min_new_tokens != 0:
        return "Request Forest does not yet support min_new_tokens."
    if sampling_params.logit_bias:
        return "Request Forest does not yet support logit_bias."
    if sampling_params.stop_token_ids:
        return "Request Forest does not yet support request-specific stop_token_ids."
    if req.custom_logit_processor:
        return "Request Forest does not yet support a custom logit processor."
    if req.grammar is not None or any(
        getattr(sampling_params, field, None)
        for field in ("json_schema", "regex", "ebnf", "structural_tag")
    ):
        return "Request Forest does not yet support grammar constraints."
    if req.return_logprob:
        return "Request Forest does not yet support logprob output."
    if req.return_hidden_states:
        return "Request Forest does not yet support hidden states output."
    if req.return_sampling_mask:
        return "Request Forest does not yet support sampling mask output."
    return None


__all__ = ["validate_request_forest_request"]
