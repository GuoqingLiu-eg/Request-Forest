from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    List,
    Optional,
)

import torch
import zmq

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import (
    BatchEmbeddingOutput,
    BatchTokenIDOutput,
    CachedTokensDetails,
    wrap_as_pickle,
)
from sglang.srt.managers.schedule_batch import (
    BaseFinishReason,
    Req,
)
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.runtime_context import get_observability, get_serving
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils.weight_versions import compute_weight_version_spans

if TYPE_CHECKING:
    from sglang.srt.managers.rust_server import RustServer


logger = logging.getLogger(__name__)


DEFAULT_FORCE_STREAM_INTERVAL = envs.SGLANG_FORCE_STREAM_INTERVAL.get()


@dataclass(kw_only=True, slots=True)
class SchedulerOutputStreamer:
    has_additional_customized_info: ClassVar[bool] = False

    send_to_detokenizer: zmq.Socket
    tree_cache: BasePrefixCache
    ps: ParallelState
    server_args: ServerArgs
    is_generation: bool
    spec_algorithm: SpeculativeAlgorithm
    disaggregation_mode: DisaggregationMode
    enable_hicache_storage: Callable[[], bool]


    rust_server: Optional[RustServer] = None
    _test_stream_output_count: int = 0

    def __post_init__(self) -> None:
        if self.has_additional_customized_info and self.rust_server is not None:
            raise ValueError(
                "additional customized generation output is not supported by "
                "Rust egress"
            )

    def _get_storage_backend_type(self) -> str:

        storage_backend_type = "none"
        cache_controller = getattr(self.tree_cache, "cache_controller", None)
        if cache_controller and hasattr(cache_controller, "storage_backend"):
            storage_backend = cache_controller.storage_backend
            if storage_backend is not None:
                storage_backend_type = type(storage_backend).__name__
        return storage_backend_type

    def get_cached_tokens_details(self, req: Req) -> Optional[CachedTokensDetails]:

        if (
            req.cached_tokens_device > 0
            or req.cached_tokens_host > 0
            or req.cached_tokens_storage > 0
        ):
            details = {
                "device": req.cached_tokens_device,
                "host": req.cached_tokens_host,
            }


            if req.cached_tokens_storage > 0 or self.enable_hicache_storage():
                details["storage"] = req.cached_tokens_storage
            if self.enable_hicache_storage():
                details["storage_backend"] = self._get_storage_backend_type()
            return details

        if req.cached_tokens > 0:
            return {
                "device": req.cached_tokens,
                "host": 0,
            }

        return None

    def stream_output(
        self,
        reqs: List[Req],
        return_logprob: bool,
        skip_req: Optional[Req] = None,
    ):

        if self.is_generation:
            self._stream_output_generation(reqs, return_logprob, skip_req)
        else:
            self._stream_output_embedding(reqs)

        if envs.SGLANG_TEST_CRASH_AFTER_STREAM_OUTPUTS.get() > 0:
            self._trigger_crash_for_tests(
                envs.SGLANG_TEST_CRASH_AFTER_STREAM_OUTPUTS.get()
            )

    def _trigger_crash_for_tests(self, crash_threshold: int):


        self._test_stream_output_count += 1
        if self._test_stream_output_count >= crash_threshold:
            raise RuntimeError(
                f"Test crash after stream_output called {self._test_stream_output_count} times"
            )

    def _stream_output_generation(
        self,
        reqs: List[Req],
        return_logprob: bool,
        skip_req: Optional[Req] = None,
        is_idle_batch: bool = False,
    ):
        return_hidden_states = any(
            req.return_hidden_states for req in reqs if req is not skip_req
        )
        return_routed_experts = any(
            req.return_routed_experts for req in reqs if req is not skip_req
        )
        return_indexer_topk = any(
            req.return_indexer_topk for req in reqs if req is not skip_req
        )
        return_sampling_mask = any(
            req.return_sampling_mask for req in reqs if req is not skip_req
        )

        acc = _GenerationStreamAccumulator(
            return_logprob=return_logprob,
            return_hidden_states=return_hidden_states,
            return_routed_experts=return_routed_experts,
            return_indexer_topk=return_indexer_topk,
            return_sampling_mask=return_sampling_mask,
            spec_algorithm=self.spec_algorithm,
            disaggregation_mode=self.disaggregation_mode,
            default_stream_interval=get_serving().stream_interval,
            default_force_stream_interval=DEFAULT_FORCE_STREAM_INTERVAL,
            get_cached_tokens_details=self.get_cached_tokens_details,
            rust_server_mode=self.rust_server is not None,
            current_weight_version=get_serving().weight_version,
        )
        for req in reqs:
            if req is skip_req:
                continue
            if req.finished() and req.finished_output:


                continue

            acc.accept(req=req)
            self._maybe_log_time_stats(req=req)

        if (
            self.has_additional_customized_info
            and self.should_build_additional_customized_info()
        ):
            additional_customized_info = self.build_additional_customized_info(
                acc.output_reqs
            )
            for key, values in additional_customized_info.items():
                if key in acc.customized_info:
                    raise ValueError(f"duplicate customized_info key: {key}")
                if len(values) != len(acc.output_reqs):
                    raise ValueError(
                        f"customized_info key {key!r} returned {len(values)} values "
                        f"for {len(acc.output_reqs)} requests"
                    )
                acc.customized_info[key] = values


        payload = acc.to_payload(
            dp_rank=self.ps.dp_rank,
            is_idle_batch=is_idle_batch,
        )
        if payload is not None:
            if self.rust_server is not None:
                self.rust_server.push_generation(payload)
            else:
                self.send_to_detokenizer.send_output(payload)

    def build_additional_customized_info(self, reqs: List[Req]) -> dict[str, list]:

        return {}

    def should_build_additional_customized_info(self) -> bool:

        return True

    def _maybe_log_time_stats(self, *, req: Req) -> None:
        if (
            req.finished()
            and self.ps.attn_tp_rank == 0
            and get_observability().enable_request_time_stats_logging
        ):
            req.log_time_stats()

    def _stream_output_embedding(self, reqs: List[Req]):
        rids = []
        http_worker_ipcs = []
        finished_reasons: List[BaseFinishReason] = []

        embeddings = []
        prompt_tokens = []
        cached_tokens = []
        cached_tokens_details = []
        time_stats = []
        retraction_counts = []
        phs_list = []
        has_phs = False
        for req in reqs:
            if req.finished():
                rids.append(req.rid)
                http_worker_ipcs.append(req.http_worker_ipc)
                finished_reasons.append(req.finished_reason.to_json())
                embeddings.append(req.embedding)
                prompt_tokens.append(len(req.origin_input_ids))
                cached_tokens.append(req.cached_tokens)


                cached_tokens_details.append(self.get_cached_tokens_details(req))
                time_stats.append(req.time_stats)
                retraction_counts.append(req.retraction_count)

                phs = req.pooled_hidden_state
                phs_list.append(phs)
                if phs is not None:
                    has_phs = True


        stacked_phs = None
        if has_phs:
            all_have_phs = all(t is not None for t in phs_list)
            if all_have_phs:
                if len(phs_list) > 1 and all(
                    t.shape == phs_list[0].shape for t in phs_list
                ):

                    stacked_phs = [torch.stack(phs_list)]
                else:

                    stacked_phs = phs_list
            else:

                stacked_phs = phs_list

        self.send_to_detokenizer.send_output(
            BatchEmbeddingOutput(
                rids=rids,
                http_worker_ipcs=http_worker_ipcs,
                time_stats=wrap_as_pickle(time_stats),
                finished_reasons=finished_reasons,
                embeddings=embeddings,
                prompt_tokens=prompt_tokens,
                cached_tokens=cached_tokens,
                cached_tokens_details=cached_tokens_details,
                placeholder_tokens_idx=None,
                placeholder_tokens_val=None,
                retraction_counts=retraction_counts,
                pooled_hidden_states=stacked_phs,
            )
        )


@dataclass(slots=True, kw_only=True)
class _GenerationStreamAccumulator:
    return_logprob: bool
    return_hidden_states: bool
    return_routed_experts: bool
    return_indexer_topk: bool
    return_sampling_mask: bool = False
    spec_algorithm: Any
    disaggregation_mode: DisaggregationMode
    default_stream_interval: int
    default_force_stream_interval: int
    get_cached_tokens_details: Callable[[Req], Optional[CachedTokensDetails]]
    current_weight_version: Optional[str]
    rids: list = field(default_factory=list)
    output_reqs: list[Req] = field(default_factory=list)
    http_worker_ipcs: list = field(default_factory=list)
    finished_reasons: list = field(default_factory=list)
    decoded_texts: list = field(default_factory=list)
    decode_ids_list: list = field(default_factory=list)
    read_offsets: list = field(default_factory=list)
    output_ids: list = field(default_factory=list)
    skip_special_tokens: list = field(default_factory=list)
    spaces_between_special_tokens: list = field(default_factory=list)
    no_stop_trim: list = field(default_factory=list)
    prompt_tokens: list = field(default_factory=list)
    reasoning_tokens: list = field(default_factory=list)
    completion_tokens: list = field(default_factory=list)
    cached_tokens: list = field(default_factory=list)
    cached_tokens_details: list = field(
        default_factory=list
    )
    image_tokens: list = field(default_factory=list)
    audio_tokens: list = field(default_factory=list)
    video_tokens: list = field(default_factory=list)
    spec_verify_ct: list = field(default_factory=list)
    spec_num_correct_drafts: list = field(default_factory=list)
    spec_num_proposed_drafts: list = field(default_factory=list)
    spec_num_block_accept_tokens: list = field(default_factory=list)
    spec_num_cap_tokens: list = field(default_factory=list)
    spec_correct_drafts_histogram: list = field(default_factory=list)
    spec_cap_lens_histogram: list = field(default_factory=list)
    retraction_counts: list = field(default_factory=list)
    weight_versions: list = field(default_factory=list)
    output_hidden_states: Optional[list] = None
    routed_experts: Optional[list] = None
    indexer_topk: Optional[list] = None
    customized_info: dict = field(default_factory=dict)
    time_stats: list = field(default_factory=list)
    input_token_logprobs_val: Optional[list] = None
    input_token_logprobs_idx: Optional[list] = None
    output_token_logprobs_val: Optional[list] = None
    output_token_logprobs_idx: Optional[list] = None
    input_top_logprobs_val: Optional[list] = None
    input_top_logprobs_idx: Optional[list] = None


    input_top_logprobs_val_flat: Optional[list] = None
    input_top_logprobs_idx_flat: Optional[list] = None
    input_top_logprobs_flat_null_prefix: Optional[list] = None
    has_input_top_logprobs_flat: bool = False
    output_top_logprobs_val: Optional[list] = None
    output_top_logprobs_idx: Optional[list] = None
    input_token_ids_logprobs_val: Optional[list] = None
    input_token_ids_logprobs_idx: Optional[list] = None
    output_token_ids_logprobs_val: Optional[list] = None
    output_token_ids_logprobs_idx: Optional[list] = None
    output_token_sampling_mask: Optional[list] = None
    output_token_sampling_logprobs: Optional[list] = None


    rust_server_mode: bool = False

    def __post_init__(self) -> None:
        if self.return_hidden_states:
            self.output_hidden_states = []
        if self.return_routed_experts:
            self.routed_experts = []
        if self.return_indexer_topk:
            self.indexer_topk = []

        if self.return_logprob:
            self.input_token_logprobs_val = []
            self.input_token_logprobs_idx = []
            self.output_token_logprobs_val = []
            self.output_token_logprobs_idx = []
            self.input_top_logprobs_val = []
            self.input_top_logprobs_idx = []
            self.input_top_logprobs_val_flat = []
            self.input_top_logprobs_idx_flat = []
            self.input_top_logprobs_flat_null_prefix = []
            self.output_top_logprobs_val = []
            self.output_top_logprobs_idx = []
            self.input_token_ids_logprobs_val = []
            self.input_token_ids_logprobs_idx = []
            self.output_token_ids_logprobs_val = []
            self.output_token_ids_logprobs_idx = []
        if self.return_sampling_mask:
            self.output_token_sampling_mask = []
            self.output_token_sampling_logprobs = []

    def accept(self, *, req: Req) -> None:
        if req.finished():
            assert not req.finished_output
            req.finished_output = True
            if req.finished_len is None:
                req.finished_len = len(req.output_ids)
            should_output = True
        else:
            if req.stream:
                stream_interval = (
                    req.sampling_params.stream_interval or self.default_stream_interval
                )


                should_output = (
                    len(req.output_ids) % stream_interval == 1
                    if stream_interval > 1
                    else len(req.output_ids) % stream_interval == 0
                )

                if should_output:

                    should_output &= not req.check_match_stop_str_prefix()
            else:
                should_output = (
                    len(req.output_ids) % self.default_force_stream_interval == 0
                )

        if not should_output:
            return

        send_token_offset = req.send_token_offset
        send_output_token_logprobs_offset = req.send_output_token_logprobs_offset
        self.rids.append(req.rid)
        self.output_reqs.append(req)
        self.finished_reasons.append(
            req.finished_reason.to_json() if req.finished_reason else None
        )

        output_ids_ = req.output_ids_through_stop
        self.output_ids.append(output_ids_[send_token_offset:])
        req.send_token_offset = len(output_ids_)
        self.prompt_tokens.append(len(req.origin_input_ids))

        if not self.rust_server_mode:


            self.http_worker_ipcs.append(req.http_worker_ipc)
            self.decoded_texts.append(req.decoded_text)
            decode_ids, read_offset = req.init_incremental_detokenize()
            self.decode_ids_list.append(decode_ids[req.send_decode_id_offset :])
            req.send_decode_id_offset = len(decode_ids)
            self.read_offsets.append(read_offset)
            self.skip_special_tokens.append(req.sampling_params.skip_special_tokens)
            self.spaces_between_special_tokens.append(
                req.sampling_params.spaces_between_special_tokens
            )
            self.no_stop_trim.append(req.sampling_params.no_stop_trim)
            self.reasoning_tokens.append(req.reasoning_tokens)
            self.completion_tokens.append(len(output_ids_))
            self.cached_tokens.append(req.cached_tokens)


            self.cached_tokens_details.append(self.get_cached_tokens_details(req))


        if req.mm_image_tokens or req.mm_audio_tokens or req.mm_video_tokens:
            image_t = req.mm_image_tokens
            audio_t = req.mm_audio_tokens
            video_t = req.mm_video_tokens
        elif req.multimodal_inputs:
            image_t, audio_t, video_t = req.multimodal_inputs.compute_mm_token_counts()
        else:
            image_t = audio_t = video_t = 0
        self.image_tokens.append(image_t)
        self.audio_tokens.append(audio_t)
        self.video_tokens.append(video_t)

        self.retraction_counts.append(req.retraction_count)
        if req.finished():
            self.weight_versions.append(
                compute_weight_version_spans(
                    req.weight_version_events,
                    current_version=self.current_weight_version,
                    num_output_tokens=len(output_ids_),
                )
            )
        else:
            self.weight_versions.append(None)

        self.time_stats.append(req.time_stats)

        if not self.spec_algorithm.is_none():
            self.spec_verify_ct.append(req.spec_verify_ct)
            self.spec_num_correct_drafts.append(req.spec_num_correct_drafts)
            self.spec_num_proposed_drafts.append(req.spec_num_proposed_drafts)
            self.spec_num_block_accept_tokens.append(req.spec_num_block_accept_tokens)
            self.spec_num_cap_tokens.append(req.spec_num_cap_tokens)
            self.spec_correct_drafts_histogram.append(req.spec_correct_drafts_histogram)
            self.spec_cap_lens_histogram.append(req.spec_cap_lens_histogram)

        if self.return_logprob:
            if (
                req.return_logprob
                and not req.input_logprob_sent

                and self.disaggregation_mode != DisaggregationMode.DECODE

                and req.logprob.input_token_logprobs_val is not None
            ):
                self.input_token_logprobs_val.append(
                    req.logprob.input_token_logprobs_val
                )
                self.input_token_logprobs_idx.append(
                    req.logprob.input_token_logprobs_idx
                )
                self.input_top_logprobs_val.append(req.logprob.input_top_logprobs_val)
                self.input_top_logprobs_idx.append(req.logprob.input_top_logprobs_idx)
                self.input_top_logprobs_val_flat.append(
                    req.logprob.input_top_logprobs_val_flat
                )
                self.input_top_logprobs_idx_flat.append(
                    req.logprob.input_top_logprobs_idx_flat
                )
                self.input_top_logprobs_flat_null_prefix.append(
                    req.logprob.input_top_logprobs_flat_null_prefix
                )
                if req.logprob.input_top_logprobs_val_flat is not None:
                    self.has_input_top_logprobs_flat = True
                self.input_token_ids_logprobs_val.append(
                    req.logprob.input_token_ids_logprobs_val
                )
                self.input_token_ids_logprobs_idx.append(
                    req.logprob.input_token_ids_logprobs_idx
                )
                req.input_logprob_sent = True
            else:
                self.input_token_logprobs_val.append([])
                self.input_token_logprobs_idx.append([])
                self.input_top_logprobs_val.append([])
                self.input_top_logprobs_idx.append([])
                self.input_top_logprobs_val_flat.append(None)
                self.input_top_logprobs_idx_flat.append(None)
                self.input_top_logprobs_flat_null_prefix.append(None)
                self.input_token_ids_logprobs_val.append([])
                self.input_token_ids_logprobs_idx.append([])

            if req.return_logprob:
                logprob_end = (
                    len(output_ids_) if req.is_retracted else max(len(output_ids_), 1)
                )
                self.output_token_logprobs_val.append(
                    req.logprob.output_token_logprobs_val[
                        send_output_token_logprobs_offset:logprob_end
                    ]
                )
                self.output_token_logprobs_idx.append(
                    req.logprob.output_token_logprobs_idx[
                        send_output_token_logprobs_offset:logprob_end
                    ]
                )
                self.output_top_logprobs_val.append(
                    req.logprob.output_top_logprobs_val[
                        send_output_token_logprobs_offset:logprob_end
                    ]
                )
                self.output_top_logprobs_idx.append(
                    req.logprob.output_top_logprobs_idx[
                        send_output_token_logprobs_offset:logprob_end
                    ]
                )
                self.output_token_ids_logprobs_val.append(
                    req.logprob.output_token_ids_logprobs_val[
                        send_output_token_logprobs_offset:logprob_end
                    ]
                )
                self.output_token_ids_logprobs_idx.append(
                    req.logprob.output_token_ids_logprobs_idx[
                        send_output_token_logprobs_offset:logprob_end
                    ]
                )
                req.send_output_token_logprobs_offset = logprob_end
            else:
                self.output_token_logprobs_val.append([])
                self.output_token_logprobs_idx.append([])
                self.output_top_logprobs_val.append([])
                self.output_top_logprobs_idx.append([])
                self.output_token_ids_logprobs_val.append([])
                self.output_token_ids_logprobs_idx.append([])

        if self.return_sampling_mask:
            if req.return_sampling_mask:
                send_output_sampling_mask_offset = req.send_output_sampling_mask_offset
                sampling_mask_end = len(req.output_token_sampling_mask)
                self.output_token_sampling_mask.append(
                    req.output_token_sampling_mask[
                        send_output_sampling_mask_offset:sampling_mask_end
                    ]
                )
                self.output_token_sampling_logprobs.append(
                    req.output_token_sampling_logprobs[
                        send_output_sampling_mask_offset:sampling_mask_end
                    ]
                )
                req.send_output_sampling_mask_offset = sampling_mask_end
            else:
                self.output_token_sampling_mask.append([])
                self.output_token_sampling_logprobs.append([])

        if self.return_hidden_states:
            if req.return_hidden_states:
                if req.return_hidden_states == "last":


                    self.output_hidden_states.append(
                        req.hidden_states[-1] if req.hidden_states else None
                    )
                else:


                    hs = req.hidden_states
                    if req.finished_len is not None:
                        hs = hs[: req.finished_len]
                    self.output_hidden_states.append(hs)
            else:
                self.output_hidden_states.append(None)
        if self.return_routed_experts:
            self.routed_experts.append(
                req.routed_experts if req.return_routed_experts else None
            )
        if self.return_indexer_topk:
            self.indexer_topk.append(
                req.indexer_topk if req.return_indexer_topk else None
            )

        current_output_len = len(self.output_ids[-1])
        if req.customized_info is not None:
            for key, req_values in req.customized_info.items():
                if key not in self.customized_info:
                    self.customized_info[key] = [
                        [None] * len(prev_output_ids)
                        for prev_output_ids in self.output_ids[:-1]
                    ]
                self.customized_info[key].append(
                    [None] * current_output_len
                    if req_values is None
                    else req_values[send_token_offset : len(output_ids_)]
                )

        for per_request_values in self.customized_info.values():
            if len(per_request_values) < len(self.output_ids):
                per_request_values.append([None] * current_output_len)

    def to_payload(
        self, *, dp_rank: int, is_idle_batch: bool
    ) -> Optional[BatchTokenIDOutput]:
        if not (self.rids or is_idle_batch):
            return None
        dp_ranks = [dp_rank] * len(self.rids) if self.rids else None
        return BatchTokenIDOutput(
            rids=self.rids,
            http_worker_ipcs=self.http_worker_ipcs,
            spec_verify_ct=self.spec_verify_ct,
            spec_num_correct_drafts=self.spec_num_correct_drafts,
            spec_num_proposed_drafts=self.spec_num_proposed_drafts,
            spec_num_block_accept_tokens=self.spec_num_block_accept_tokens,
            spec_num_cap_tokens=self.spec_num_cap_tokens,
            spec_correct_drafts_histogram=self.spec_correct_drafts_histogram,
            spec_cap_lens_histogram=self.spec_cap_lens_histogram,
            time_stats=wrap_as_pickle(self.time_stats),
            finished_reasons=self.finished_reasons,
            decoded_texts=self.decoded_texts,
            decode_ids=self.decode_ids_list,
            read_offsets=self.read_offsets,
            output_ids=self.output_ids,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=self.spaces_between_special_tokens,
            no_stop_trim=self.no_stop_trim,
            prompt_tokens=self.prompt_tokens,
            reasoning_tokens=self.reasoning_tokens,
            completion_tokens=self.completion_tokens,
            cached_tokens=self.cached_tokens,
            cached_tokens_details=self.cached_tokens_details,
            image_tokens=self.image_tokens,
            audio_tokens=self.audio_tokens,
            video_tokens=self.video_tokens,
            input_token_logprobs_val=self.input_token_logprobs_val,
            input_token_logprobs_idx=self.input_token_logprobs_idx,
            output_token_logprobs_val=self.output_token_logprobs_val,
            output_token_logprobs_idx=self.output_token_logprobs_idx,
            input_top_logprobs_val=self.input_top_logprobs_val,
            input_top_logprobs_idx=self.input_top_logprobs_idx,


            input_top_logprobs_val_flat=(
                self.input_top_logprobs_val_flat
                if self.has_input_top_logprobs_flat
                else None
            ),
            input_top_logprobs_idx_flat=(
                self.input_top_logprobs_idx_flat
                if self.has_input_top_logprobs_flat
                else None
            ),
            input_top_logprobs_flat_null_prefix=(
                self.input_top_logprobs_flat_null_prefix
                if self.has_input_top_logprobs_flat
                else None
            ),
            output_top_logprobs_val=self.output_top_logprobs_val,
            output_top_logprobs_idx=self.output_top_logprobs_idx,
            input_token_ids_logprobs_val=self.input_token_ids_logprobs_val,
            input_token_ids_logprobs_idx=self.input_token_ids_logprobs_idx,
            output_token_ids_logprobs_val=self.output_token_ids_logprobs_val,
            output_token_ids_logprobs_idx=self.output_token_ids_logprobs_idx,
            output_token_entropy_val=None,
            output_token_sampling_mask=self.output_token_sampling_mask,
            output_token_sampling_logprobs=self.output_token_sampling_logprobs,
            output_hidden_states=self.output_hidden_states,
            routed_experts=self.routed_experts,
            indexer_topk=self.indexer_topk,
            customized_info=(
                wrap_as_pickle(self.customized_info) if self.customized_info else None
            ),
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
            retraction_counts=self.retraction_counts,
            weight_versions=(
                self.weight_versions if any(self.weight_versions) else None
            ),
            dp_ranks=dp_ranks,
        )
