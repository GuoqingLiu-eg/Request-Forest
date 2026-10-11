import logging
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import torch

from sglang.srt.hardware_backend.mlx.model_runner import (
    MlxPendingDecode,
    MlxPendingExtend,
    MlxPendingPrefill,
)
from sglang.srt.hardware_backend.mlx.sampling import (
    MlxLogprobSpec,
    MlxStepLogprobs,
    lazy_logprob_arrays,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    PPProxyTensors,
)
from sglang.srt.runtime_context import (
    get_device,
    get_exec,
    get_memory,
    get_model,
    get_schedule,
)

logger = logging.getLogger(__name__)


@dataclass
class MlxLaunch:


    lazy_tokens: Optional[mx.array]
    prefills: list[MlxPendingPrefill]
    extends: list[MlxPendingExtend]
    decode: Optional[MlxPendingDecode]
    mode: str


class MlxTpModelWorker(TpModelWorker):


    def _init_model_runner(self):

        from sglang.srt.hardware_backend.mlx.model_runner_stub import (
            MlxModelRunnerStub,
        )

        MlxModelRunnerStub.validate_startup_weight_load_mode(self.server_args)

        from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

        logger.info("Initializing MlxModelRunner for end-to-end MLX inference")
        init_kwargs = dict(
            model_path=get_model().model_path,
            trust_remote_code=get_model().trust_remote_code,
            disable_radix_cache=get_memory().disable_radix_cache,
            mem_fraction_static=get_schedule().mem_fraction_static,
            quantization=get_model().quantization,
            revision=get_model().revision,
            enable_sampling=get_device().mlx_enable_sampling,
            sampling_rng_seed=get_device().random_seed,
            deterministic_seeding=(
                get_exec().deterministic.enable_deterministic_inference
            ),
        )
        if get_schedule().max_total_tokens is not None:
            init_kwargs["pool_size"] = get_schedule().max_total_tokens
        self._mlx_runner = MlxModelRunner(**init_kwargs)

        self._model_runner = MlxModelRunnerStub(
            model_config=self.model_config,
            mem_fraction_static=get_schedule().mem_fraction_static,
            gpu_id=self.gpu_id,
            ps=self.ps,
            nccl_port=self.nccl_port,
            server_args=self.server_args,
            is_draft_worker=self.is_draft_worker,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            memory_pool_config=self.memory_pool_config,
            mlx_pool_size=self._mlx_runner.pool_size,
        )

        self._mlx_active_rids: set[str] = set()
        self._mlx_pool_initialized = False

    def get_pad_input_ids_func(self):

        return None

    def _ensure_mlx_pool_initialized(self):

        if not self._mlx_pool_initialized:
            self._mlx_runner.init_cache_pools(self._model_runner.req_to_token_pool)
            self._mlx_pool_initialized = True

    def forward_batch_generation(
        self,
        batch: Optional[ScheduleBatch],
        forward_batch: Optional[ForwardBatch] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        is_verify: bool = False,
        skip_attn_backend_init: Optional[bool] = None,
        *,
        capture_hidden_mode: Optional[CaptureHiddenMode] = None,
    ) -> GenerationBatchResult:

        if batch is not None:
            self._ensure_mlx_pool_initialized()
            return self._forward_batch_generation_mlx(batch)


        return super().forward_batch_generation(
            batch,
            forward_batch,
            pp_proxy_tensors,
            is_verify,
            skip_attn_backend_init,
            capture_hidden_mode=capture_hidden_mode,
        )

    def _cleanup_stale_rids(self, forward_mode, current_rids: set[str]) -> None:

        if forward_mode.is_decode():
            stale_rids = self._mlx_active_rids - current_rids
            for rid in stale_rids:
                self._mlx_runner.remove_request(rid)
            self._mlx_active_rids = current_rids
        else:
            self._mlx_active_rids |= current_rids

    def prepare_for_kv_cache_release(self, req) -> None:

        if self._mlx_runner.has_request(req.rid):
            self._mlx_runner.store_auxiliary_state_for_request(req.rid)


            req.mamba_last_track_seqlen = None

    def _route_extend_request(self, rid: str, decoding_rids: set[str]) -> str:

        if not self._mlx_runner.has_request(rid):
            return "prefill"
        if rid in decoding_rids:
            return "decode"
        return "continuation"

    @staticmethod
    def _chunk_needs_logits(req) -> bool:

        if req.extend_range is None:
            return True
        return req.extend_range.end >= len(req.full_untruncated_fill_ids)

    @staticmethod
    def _sampling_active(batch: ScheduleBatch) -> bool:
        return get_device().mlx_enable_sampling and batch.sampling_info is not None

    def _build_logit_edit_rows(
        self, batch: ScheduleBatch
    ) -> dict[str, mx.array] | None:

        if not self._sampling_active(batch):
            return None
        sinfo = batch.sampling_info


        sinfo.grammars = (
            [req.grammar for req in batch.reqs] if batch.has_grammar else None
        )
        has_grammar = bool(sinfo.grammars)
        if not has_grammar and sinfo.logit_bias is None:
            return None
        if not has_grammar:


            combined = sinfo.logit_bias.to(device="cpu", dtype=torch.float32)
        else:
            combined = torch.zeros(
                len(batch.reqs), sinfo.vocab_size, dtype=torch.float32
            )
            sinfo.update_regex_vocab_mask()
            if sinfo.grammar_mask is not None:
                grammar_mask = sinfo.grammar_mask
                grammar_mask.grammar.apply_vocab_mask(
                    logits=combined,
                    vocab_mask=grammar_mask.vocab_mask.to("cpu"),
                )


                sinfo.grammar_mask = None
            if sinfo.logit_bias is not None:
                combined += sinfo.logit_bias.to("cpu")
        rows = mx.array(combined.numpy())
        return {req.rid: rows[i] for i, req in enumerate(batch.reqs)}

    def _logprob_rows(
        self, batch: ScheduleBatch
    ) -> dict[str, tuple[int, tuple[int, ...] | None]] | None:

        if not self._sampling_active(batch) or not batch.return_logprob:
            return None
        tops = batch.top_logprobs_nums or [0] * len(batch.reqs)
        tids = batch.token_ids_logprobs or [None] * len(batch.reqs)
        rows = {}
        for req, top_k, token_ids in zip(batch.reqs, tops, tids):
            if req.return_logprob:
                rows[req.rid] = (
                    int(top_k or 0),
                    tuple(token_ids) if token_ids else None,
                )
        return rows or None

    @staticmethod
    def _logprob_spec_for(
        rows: dict[str, tuple[int, tuple[int, ...] | None]] | None,
        rids: list[str],
    ) -> MlxLogprobSpec | None:
        if rows is None or not any(rid in rows for rid in rids):
            return None
        return MlxLogprobSpec(
            top_ks=tuple(rows.get(rid, (0, None))[0] for rid in rids),
            token_ids=tuple(rows.get(rid, (0, None))[1] for rid in rids),
        )

    def _custom_logits_hook(self, batch: ScheduleBatch):

        if not (
            self._sampling_active(batch)
            and batch.sampling_info.has_custom_logit_processor
        ):
            return None
        sinfo = batch.sampling_info

        def hook(logits_np):
            from sglang.srt.layers.sampler import apply_custom_logit_processor


            logits_t = torch.from_numpy(logits_np)
            apply_custom_logit_processor(logits_t, sinfo)
            return logits_np

        return hook

    @staticmethod
    def _assemble_logprob_output(step_rows: dict[str, tuple], reqs: list):

        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        chosen, top_val, top_idx, tid_val, tid_idx = [], [], [], [], []
        for req in reqs:
            row = step_rows.get(req.rid)
            if row is None:
                row = (0.0, [], [], [], [])
            chosen.append(row[0])
            top_val.append(torch.tensor(row[1], dtype=torch.float32))
            top_idx.append(torch.tensor(row[2], dtype=torch.long))
            tid_val.append(torch.tensor(row[3], dtype=torch.float32))
            tid_idx.append(list(row[4]))
        return LogitsProcessorOutput(
            next_token_logits=None,
            next_token_logprobs=torch.tensor(chosen, dtype=torch.float32),
            next_token_top_logprobs_val=top_val,
            next_token_top_logprobs_idx=top_idx,
            next_token_token_ids_logprobs_val=tid_val,
            next_token_token_ids_logprobs_idx=tid_idx,
        )

    @staticmethod
    def _step_logprob_rows(
        step: Optional[MlxStepLogprobs], rids: list[str]
    ) -> dict[str, tuple]:

        if step is None:
            return {}
        return {
            rid: (
                step.chosen[i],
                step.top_val[i],
                step.top_idx[i],
                step.token_ids_val[i],
                step.token_ids_idx[i],
            )
            for i, rid in enumerate(rids)
        }

    def _collect_step_logprobs(
        self,
        step_rows: dict[str, tuple],
        lazy_logprobs,
        rids: list[str],
    ) -> None:

        step = self._mlx_runner.collect_logprobs(lazy_logprobs)
        step_rows.update(self._step_logprob_rows(step, rids))

    def _forward_batch_generation_mlx(
        self, batch: ScheduleBatch
    ) -> GenerationBatchResult:

        launch = self.async_forward_batch_generation_mlx(batch)
        return self.finalize_mlx_result(launch, batch.reqs)

    @staticmethod
    def _stacked_edit_rows(
        edit_rows: dict[str, mx.array] | None, req_ids: list[str]
    ) -> Optional[mx.array]:

        if not edit_rows:
            return None
        return mx.stack([edit_rows[rid] for rid in req_ids])

    def async_forward_batch_generation_mlx(self, batch: ScheduleBatch) -> MlxLaunch:

        self._ensure_mlx_pool_initialized()

        forward_mode = batch.forward_mode
        reqs = batch.reqs

        if forward_mode.is_idle():
            return MlxLaunch(
                lazy_tokens=None, prefills=[], extends=[], decode=None, mode="idle"
            )

        self._cleanup_stale_rids(forward_mode, {req.rid for req in reqs})

        if forward_mode.is_decode():
            req_ids = [req.rid for req in reqs]
            pending_decode = self._mlx_runner.decode_batch_start(
                req_ids,
                edit_rows=self._stacked_edit_rows(
                    self._build_logit_edit_rows(batch), req_ids
                ),
                logprob_spec=self._logprob_spec_for(self._logprob_rows(batch), req_ids),
                logits_hook=self._custom_logits_hook(batch),
            )
            mx.async_eval(
                pending_decode.lazy_tokens,
                *lazy_logprob_arrays(pending_decode.lazy_logprobs),
            )
            return MlxLaunch(
                lazy_tokens=pending_decode.lazy_tokens,
                prefills=[],
                extends=[],
                decode=pending_decode,
                mode="decode",
            )

        if forward_mode.is_extend():


            self._mlx_runner.flush_all_decode_kv()
            return self._async_extend_batch(batch)

        raise ValueError(
            f"MLX async runner does not support forward mode: {forward_mode}"
        )

    def _async_extend_batch(self, batch: ScheduleBatch) -> MlxLaunch:

        reqs = batch.reqs
        input_ids_cpu = batch.input_ids.cpu().tolist()
        out_cache_loc_cpu = batch.out_cache_loc.cpu().tolist()
        extend_seq_lens = batch.extend_lens
        edit_rows = self._build_logit_edit_rows(batch)
        logprob_rows = self._logprob_rows(batch)

        offset = 0
        slot_offset = 0
        pending_prefills: list[MlxPendingPrefill] = []
        pending_extends: list[MlxPendingExtend] = []
        mixed_decode_rids: list[str] = []


        decoding_rids = {r.rid for r in (batch.decoding_reqs or [])}

        for i, req in enumerate(reqs):
            seq_len = extend_seq_lens[i]
            req_token_ids = input_ids_cpu[offset : offset + seq_len]
            req_new_slots = out_cache_loc_cpu[slot_offset : slot_offset + seq_len]
            offset += seq_len
            slot_offset += seq_len

            route = self._route_extend_request(req.rid, decoding_rids)
            if route == "continuation":
                pending_extends.append(
                    self._mlx_runner.extend_start(
                        req_id=req.rid,
                        new_token_ids=req_token_ids,
                        new_slot_ids=req_new_slots,
                        needs_logits=self._chunk_needs_logits(req),
                        logit_edit_row=edit_rows[req.rid] if edit_rows else None,
                        logprob_spec=self._logprob_spec_for(logprob_rows, [req.rid]),
                    )
                )
            elif route == "decode":
                mixed_decode_rids.append(req.rid)
            else:
                prefix_slot_ids = req.prefix_indices.tolist()
                full_token_ids = list(req.get_fill_ids())
                pending_prefills.append(
                    self._mlx_runner.prefill_start(
                        req_id=req.rid,
                        new_token_ids=req_token_ids,
                        full_token_ids=full_token_ids,
                        prefix_slot_ids=prefix_slot_ids,
                        new_slot_ids=req_new_slots,
                        req_pool_idx=req.req_pool_idx,
                        req=req,
                        needs_logits=self._chunk_needs_logits(req),
                        logit_edit_row=edit_rows[req.rid] if edit_rows else None,
                        logprob_spec=self._logprob_spec_for(logprob_rows, [req.rid]),
                    )
                )

        pending_mixed_decode: Optional[MlxPendingDecode] = None
        if mixed_decode_rids:
            pending_mixed_decode = self._mlx_runner.decode_batch_start(
                mixed_decode_rids,
                edit_rows=self._stacked_edit_rows(edit_rows, mixed_decode_rids),
                logprob_spec=self._logprob_spec_for(logprob_rows, mixed_decode_rids),
            )


        prefill_ext_tokens: list[mx.array] = [p.lazy_token for p in pending_prefills]
        prefill_ext_tokens.extend(e.lazy_token for e in pending_extends)

        async_args: list[mx.array] = []
        if prefill_ext_tokens:
            lazy_stacked = mx.stack(prefill_ext_tokens, axis=0)
            async_args.append(lazy_stacked)
        else:
            lazy_stacked = None

        for pending in (*pending_prefills, *pending_extends):
            async_args.extend(self._mlx_runner.cache_state_arrays([pending.cache]))
            async_args.extend(lazy_logprob_arrays(pending.lazy_logprobs))
        if pending_mixed_decode is not None:
            async_args.append(pending_mixed_decode.lazy_tokens)
            async_args.extend(lazy_logprob_arrays(pending_mixed_decode.lazy_logprobs))
            async_args.extend(
                self._mlx_runner.cache_state_arrays(pending_mixed_decode.caches)
            )

        if async_args:
            mx.async_eval(*async_args)

        return MlxLaunch(
            lazy_tokens=lazy_stacked,
            prefills=pending_prefills,
            extends=pending_extends,
            decode=pending_mixed_decode,
            mode="extend",
        )

    def async_chained_decode_mlx(self, prev_pending: MlxPendingDecode) -> MlxLaunch:

        pending = self._mlx_runner.decode_batch_start_chained(prev_pending)
        mx.async_eval(pending.lazy_tokens)
        return MlxLaunch(
            lazy_tokens=pending.lazy_tokens,
            prefills=[],
            extends=[],
            decode=pending,
            mode="decode",
        )

    def finalize_mlx_result(
        self, launch: MlxLaunch, reqs: list
    ) -> GenerationBatchResult:

        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        decode = launch.decode
        if launch.mode == "idle":
            return GenerationBatchResult(
                logits_output=LogitsProcessorOutput(next_token_logits=None),
                can_run_cuda_graph=False,
            )

        step_logprob_rows: dict[str, tuple] = {}

        if launch.mode == "decode":
            assert decode is not None
            next_tokens_list = self._mlx_runner.decode_batch_finalize(decode)
            self._collect_step_logprobs(
                step_logprob_rows, decode.lazy_logprobs, decode.req_ids
            )

        elif launch.mode == "extend":
            prefill_map: dict[str, int] = {}
            for pending_p in launch.prefills:
                prefill_map[pending_p.req_id] = self._mlx_runner.prefill_finalize(
                    pending_p
                )
                self._collect_step_logprobs(
                    step_logprob_rows, pending_p.lazy_logprobs, [pending_p.req_id]
                )

            extend_map: dict[str, int] = {}
            for pending_e in launch.extends:
                extend_map[pending_e.req_id] = self._mlx_runner.extend_finalize(
                    pending_e
                )
                self._collect_step_logprobs(
                    step_logprob_rows, pending_e.lazy_logprobs, [pending_e.req_id]
                )

            decode_map: dict[str, int] = {}
            if decode is not None:
                mixed_tokens = self._mlx_runner.decode_batch_finalize(decode)
                decode_map = {
                    rid: tok for rid, tok in zip(decode.req_ids, mixed_tokens)
                }
                self._collect_step_logprobs(
                    step_logprob_rows, decode.lazy_logprobs, decode.req_ids
                )

            next_tokens_list = []
            for req in reqs:
                if req.rid in decode_map:
                    next_tokens_list.append(decode_map[req.rid])
                elif req.rid in extend_map:
                    next_tokens_list.append(extend_map[req.rid])
                else:
                    next_tokens_list.append(prefill_map[req.rid])

        else:
            raise ValueError(f"Unknown MLX async mode: {launch.mode}")

        next_token_ids = torch.tensor(next_tokens_list, dtype=torch.long, device="cpu")
        logits_output = (
            self._assemble_logprob_output(step_logprob_rows, reqs)
            if step_logprob_rows
            else LogitsProcessorOutput(next_token_logits=None)
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            can_run_cuda_graph=False,
        )
