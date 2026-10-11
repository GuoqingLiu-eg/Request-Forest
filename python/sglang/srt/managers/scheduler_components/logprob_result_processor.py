from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import (
    List,
    Tuple,
)

import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.io_struct import build_flat_input_top_logprobs_arrays
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.runtime_context import get_exec
from sglang.srt.server_args import MIS_DELIMITER_TOKEN_ID

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerLogprobResultProcessor:
    model_config: ModelConfig

    def _process_input_token_logprobs(
        self, req: Req, input_token_logprobs: List
    ) -> None:

        is_multi_item_scoring = self._is_multi_item_scoring(req)


        if is_multi_item_scoring:

            req.logprob.input_token_logprobs_val = input_token_logprobs
        else:

            req.logprob.input_token_logprobs_val = [None] + input_token_logprobs[:-1]


        if is_multi_item_scoring:


            delimiter_count = len(req.multi_item_delimiter_indices)
            input_token_logprobs_idx = [MIS_DELIMITER_TOKEN_ID] * delimiter_count
        else:

            input_token_logprobs_idx = req.origin_input_ids[req.logprob_start_len :]


        req.logprob.input_token_logprobs_idx = [
            x if x < self.model_config.vocab_size - 1 else 0
            for x in input_token_logprobs_idx
        ]

    def _process_input_top_logprobs(self, req: Req) -> None:

        if req.logprob.top_logprobs_num <= 0:
            return

        is_multi_item_scoring = self._is_multi_item_scoring(req)


        req.logprob.input_top_logprobs_val = [] if is_multi_item_scoring else [None]
        req.logprob.input_top_logprobs_idx = [] if is_multi_item_scoring else [None]


        for val, idx in zip(
            req.temp_input_top_logprobs_val,
            req.temp_input_top_logprobs_idx,
            strict=True,
        ):
            req.logprob.input_top_logprobs_val.extend(val)
            req.logprob.input_top_logprobs_idx.extend(idx)


        if not is_multi_item_scoring:
            req.logprob.input_top_logprobs_val.pop()
            req.logprob.input_top_logprobs_idx.pop()


        req.temp_input_top_logprobs_idx = None
        req.temp_input_top_logprobs_val = None

    def _flatten_input_top_logprobs(self, req: Req) -> None:

        if req.logprob.top_logprobs_num <= 0:
            return
        try:
            (
                req.logprob.input_top_logprobs_val_flat,
                req.logprob.input_top_logprobs_idx_flat,
                req.logprob.input_top_logprobs_flat_null_prefix,
            ) = build_flat_input_top_logprobs_arrays(
                req.logprob.input_top_logprobs_val,
                req.logprob.input_top_logprobs_idx,
                req.logprob.top_logprobs_num,
            )
        except ValueError as e:


            logger.warning(
                "Falling back to nested input top logprobs for rid=%s: %s",
                req.rid,
                e,
            )
            return
        req.logprob.input_top_logprobs_val = []
        req.logprob.input_top_logprobs_idx = []

    def _process_input_token_ids_logprobs(self, req: Req) -> None:

        if req.logprob.token_ids_logprob is None:
            return

        is_multi_item_scoring = self._is_multi_item_scoring(req)


        req.logprob.input_token_ids_logprobs_val = (
            [] if is_multi_item_scoring else [None]
        )
        req.logprob.input_token_ids_logprobs_idx = (
            [] if is_multi_item_scoring else [None]
        )


        for val, idx in zip(
            req.temp_input_token_ids_logprobs_val,
            req.temp_input_token_ids_logprobs_idx,
            strict=True,
        ):
            val_list = val.tolist() if isinstance(val, torch.Tensor) else val
            req.logprob.input_token_ids_logprobs_val.extend(
                val_list if isinstance(val_list, list) else [val_list]
            )
            req.logprob.input_token_ids_logprobs_idx.extend(idx)


        if not is_multi_item_scoring:
            req.logprob.input_token_ids_logprobs_val.pop()
            req.logprob.input_token_ids_logprobs_idx.pop()


        req.temp_input_token_ids_logprobs_idx = None
        req.temp_input_token_ids_logprobs_val = None

    def _calculate_relevant_tokens_len(self, req: Req) -> int:

        is_multi_item_scoring = self._is_multi_item_scoring(req)

        if is_multi_item_scoring:
            return len(req.multi_item_delimiter_indices)
        else:
            return len(req.origin_input_ids[req.logprob_start_len :])

    def calculate_num_input_logprobs(
        self,
        req: Req,
        extend_input_len: int,
        extend_logprob_start_len: int,
    ) -> int:

        is_multi_item_scoring = self._is_multi_item_scoring(req)

        if is_multi_item_scoring:

            return sum(
                1
                for idx in req.multi_item_delimiter_indices
                if extend_logprob_start_len <= idx < extend_input_len
            )
        else:

            return extend_input_len - extend_logprob_start_len

    def _is_multi_item_scoring(self, req: Req) -> bool:

        return (
            get_exec().features.enable_mis
            and req.is_prefill_only
            and req.multi_item_delimiter_indices is not None
        )

    def add_input_logprob_return_values(
        self,
        i: int,
        req: Req,
        output: LogitsProcessorOutput,
        logprob_pt: int,
        num_input_logprobs: int,
        last_prefill_chunk: bool,
    ):

        assert output.input_token_logprobs is not None
        if req.input_token_logprobs is None:
            req.input_token_logprobs = []
        if req.temp_input_top_logprobs_val is None:
            req.temp_input_top_logprobs_val = []
        if req.temp_input_top_logprobs_idx is None:
            req.temp_input_top_logprobs_idx = []
        if req.temp_input_token_ids_logprobs_val is None:
            req.temp_input_token_ids_logprobs_val = []
        if req.temp_input_token_ids_logprobs_idx is None:
            req.temp_input_token_ids_logprobs_idx = []

        if req.logprob.input_token_logprobs_val is not None:


            if req.logprob.top_logprobs_num > 0:
                assert req.logprob.input_token_logprobs_val is not None
            return


        assert isinstance(output.input_token_logprobs, tuple)
        input_token_logprobs: Tuple[int] = output.input_token_logprobs
        input_token_logprobs = input_token_logprobs[
            logprob_pt : logprob_pt + num_input_logprobs
        ]
        req.input_token_logprobs.extend(input_token_logprobs)

        if req.logprob.top_logprobs_num > 0:
            req.temp_input_top_logprobs_val.append(output.input_top_logprobs_val[i])
            req.temp_input_top_logprobs_idx.append(output.input_top_logprobs_idx[i])

        if req.logprob.token_ids_logprob is not None:
            req.temp_input_token_ids_logprobs_val.append(
                output.input_token_ids_logprobs_val[i]
            )
            req.temp_input_token_ids_logprobs_idx.append(
                output.input_token_ids_logprobs_idx[i]
            )

        if last_prefill_chunk:
            input_token_logprobs = req.input_token_logprobs
            req.input_token_logprobs = None
            assert req.logprob.input_token_logprobs_val is None
            assert req.logprob.input_token_logprobs_idx is None
            assert req.logprob.input_top_logprobs_val is None
            assert req.logprob.input_top_logprobs_idx is None


            self._process_input_token_logprobs(req, input_token_logprobs)
            self._process_input_top_logprobs(req)

            self._process_input_token_ids_logprobs(req)

            if req.return_logprob:
                relevant_tokens_len = self._calculate_relevant_tokens_len(req)
                assert len(req.logprob.input_token_logprobs_val) == relevant_tokens_len
                assert len(req.logprob.input_token_logprobs_idx) == relevant_tokens_len
                if req.logprob.top_logprobs_num > 0:
                    assert (
                        len(req.logprob.input_top_logprobs_val) == relevant_tokens_len
                    )
                    assert (
                        len(req.logprob.input_top_logprobs_idx) == relevant_tokens_len
                    )
                if req.logprob.token_ids_logprob is not None:
                    assert (
                        len(req.logprob.input_token_ids_logprobs_val)
                        == relevant_tokens_len
                    )
                    assert (
                        len(req.logprob.input_token_ids_logprobs_idx)
                        == relevant_tokens_len
                    )


            if req.return_flat_raw_top_logprobs:
                self._flatten_input_top_logprobs(req)

    def add_logprob_return_values(
        self,
        i: int,
        req: Req,
        pt: int,
        next_token_ids: List[int],
        num_input_logprobs: int,
        output: LogitsProcessorOutput,
    ):

        if output.next_token_logprobs is not None:
            req.logprob.output_token_logprobs_val.append(output.next_token_logprobs[i])
            req.logprob.output_token_logprobs_idx.append(next_token_ids[i])


        if num_input_logprobs > 0:
            self.add_input_logprob_return_values(
                i,
                req,
                output,
                pt,
                num_input_logprobs,
                last_prefill_chunk=True,
            )
        else:
            self._initialize_empty_logprob_containers(req)

        if (
            req.logprob.top_logprobs_num > 0
            and output.next_token_top_logprobs_val is not None
        ):


            req.logprob.output_top_logprobs_val.append(
                output.next_token_top_logprobs_val[i]
            )
            req.logprob.output_top_logprobs_idx.append(
                output.next_token_top_logprobs_idx[i]
            )

        if (
            req.logprob.token_ids_logprob is not None
            and output.next_token_token_ids_logprobs_val is not None
        ):

            logprobs_val = output.next_token_token_ids_logprobs_val[i]
            if isinstance(logprobs_val, torch.Tensor):
                logprobs_val = logprobs_val.tolist()
            req.logprob.output_token_ids_logprobs_val.append(logprobs_val)
            req.logprob.output_token_ids_logprobs_idx.append(
                output.next_token_token_ids_logprobs_idx[i]
            )

        return num_input_logprobs

    def _initialize_empty_logprob_containers(self, req: Req) -> None:

        if req.logprob.input_token_logprobs_val is None:
            req.logprob.input_token_logprobs_val = []
        if req.logprob.input_token_logprobs_idx is None:
            req.logprob.input_token_logprobs_idx = []
        if req.logprob.input_top_logprobs_val is None:
            req.logprob.input_top_logprobs_val = []
        if req.logprob.input_top_logprobs_idx is None:
            req.logprob.input_top_logprobs_idx = []
        if req.logprob.input_token_ids_logprobs_val is None:
            req.logprob.input_token_ids_logprobs_val = []
        if req.logprob.input_token_ids_logprobs_idx is None:
            req.logprob.input_token_ids_logprobs_idx = []
