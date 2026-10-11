from __future__ import annotations

import importlib
import json
import logging
import os
from array import array
from itertools import chain
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Optional, Tuple

import msgspec

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.srt.managers.utils import (
    MsgpackDecodeError,
    compute_num_reserved_tokens,
    msgpack_decode_explained,
)
from sglang.srt.runtime_context import (
    get_mm,
    get_observability,
    get_parallel,
    get_serving,
)
from sglang.srt.utils.flatten import (
    FlatPairColumns,
    NestedRowColumns,
    RaggedPairColumns,
)
from sglang.version import __version__

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.io_struct import BatchTokenIDOutput
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.rust_extensions._server import MmSpec, Server, ServerArgs
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class NativeMmSpec(msgspec.Struct, frozen=True, kw_only=True):


    family: str
    feature_shm: bool
    image_token_id: int
    patch_size: int
    merge_size: int
    temporal_patch_size: int
    min_pixels: int
    max_pixels: int
    image_mean: Tuple[float, ...]
    image_std: Tuple[float, ...]


    resample: str
    vision_start_token_id: Optional[int]
    vision_end_token_id: Optional[int]
    video_token_id: Optional[int]


    DRAIN_ONLY = ("vision_start_token_id", "vision_end_token_id", "video_token_id")

    @property
    def feature_dim(self) -> int:
        return 3 * self.temporal_patch_size * self.patch_size * self.patch_size

    def rust_json(self) -> str:

        fields = (f for f in self.__struct_fields__ if f not in self.DRAIN_ONLY)
        return msgspec.json.encode({f: getattr(self, f) for f in fields}).decode()


class NativeMmFamily(msgspec.Struct, frozen=True, kw_only=True):


    name: str


    mm_processor: str

    model_types: FrozenSet[str]


    image_processors: Dict[str, str]

    def serves(self, mm_processor_cls: Any, model_type: Optional[str]) -> bool:
        module_name, _, class_name = self.mm_processor.partition(":")
        cls = getattr(importlib.import_module(module_name), class_name)
        return mm_processor_cls is cls and model_type in self.model_types


NATIVE_MM_FAMILIES: Tuple[NativeMmFamily, ...] = (
    NativeMmFamily(
        name="qwen_vl",
        mm_processor="sglang.srt.multimodal.processors.qwen_vl:QwenVLImageProcessor",
        model_types=frozenset(
            (
                "qwen2_vl",
                "qwen2_5_vl",
                "qwen3_vl",
                "qwen3_vl_moe",
                "qwen3_5",
                "qwen3_5_moe",
            )
        ),
        image_processors={
            "Qwen2VLImageProcessor": "aten_u8",
            "Qwen2VLImageProcessorFast": "aten_u8",
            "Qwen2VLImageProcessorPil": "pil",
        },
    ),
)


def native_mm_family_for(
    mm_processor_cls: Any, model_type: Optional[str]
) -> Optional[NativeMmFamily]:

    return next(
        (f for f in NATIVE_MM_FAMILIES if f.serves(mm_processor_cls, model_type)), None
    )


class NativeMmHost:


    AUTO_MM_WORKERS = 8

    def __init__(
        self,
        *,
        server_args: ServerArgs,
        model_config: ModelConfig,
        processor: Any = None,
    ):


        from sglang.srt.managers.multimodal_processor import import_processors
        from sglang.srt.managers.tokenizer_manager import get_processor_wrapper

        self.server_args = server_args
        self.model_config = model_config

        self.mm_workers = server_args.mm_processor_worker_num or self.AUTO_MM_WORKERS


        import_processors("sglang.srt.multimodal.processors")
        if mm_process_pkg := envs.SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE.get():
            import_processors(mm_process_pkg, overwrite=True)
        self._processor = processor or get_processor_wrapper(self.server_args)

    def resolve_native_spec(self) -> Optional[NativeMmSpec]:

        from sglang.srt.managers.multimodal_processor import get_mm_processor_cls

        hf_config = self.model_config.hf_config
        mm_processor_cls = get_mm_processor_cls(
            hf_config, self.server_args, model_config=self.model_config
        )
        family = native_mm_family_for(
            mm_processor_cls, getattr(hf_config, "model_type", None)
        )
        if family is None:
            return None
        ip = getattr(self._processor, "image_processor", None)
        resample = family.image_processors.get(type(ip).__name__)
        if resample is None:
            return None


        stages = ("do_resize", "do_rescale", "do_normalize")
        if not all(getattr(ip, stage, True) for stage in stages):
            return None
        if getattr(ip, "rescale_factor", None) != 1 / 255:
            return None


        image_overrides = dict((get_mm().mm_process_config or {}).get("image", {}))
        if not set(image_overrides) <= {"min_pixels", "max_pixels"}:
            return None

        size = getattr(ip, "size", None) or {}
        min_pixels = image_overrides.get(
            "min_pixels", getattr(ip, "min_pixels", None) or size.get("shortest_edge")
        )
        max_pixels = image_overrides.get(
            "max_pixels", getattr(ip, "max_pixels", None) or size.get("longest_edge")
        )
        try:
            spec = NativeMmSpec(
                family=family.name,
                feature_shm=self._use_feature_shm(),
                image_token_id=hf_config.image_token_id,
                patch_size=ip.patch_size,
                merge_size=ip.merge_size,
                temporal_patch_size=ip.temporal_patch_size,
                min_pixels=int(min_pixels),
                max_pixels=int(max_pixels),
                image_mean=tuple(float(x) for x in ip.image_mean),
                image_std=tuple(float(x) for x in ip.image_std),
                resample=resample,
                vision_start_token_id=getattr(hf_config, "vision_start_token_id", None),
                vision_end_token_id=getattr(hf_config, "vision_end_token_id", None),
                video_token_id=getattr(hf_config, "video_token_id", None),
            )
        except (AttributeError, TypeError):
            return None
        logger.info("rust server: native MM pipeline enabled (family=%s)", family.name)
        return spec

    def _use_feature_shm(self) -> bool:

        from sglang.srt.multimodal.transport import (
            determine_tensor_transport_mode,
        )

        return (
            get_parallel().config.tp_size > 1
            and determine_tensor_transport_mode() != "default"
            and not self.server_args.skip_tokenizer_init
        )

    @staticmethod
    def build_native_mm(spec: NativeMmSpec, entry):

        import torch

        from sglang.srt.managers.mm_utils import ShmPointerMMData
        from sglang.srt.managers.schedule_batch import (
            Modality,
            MultimodalDataItem,
            MultimodalProcessorOutput,
        )

        shm_names = entry.shm_names
        if shm_names is None:
            features = torch.from_numpy(entry.features.reshape(-1, spec.feature_dim))
        items = []
        row = 0
        for index, ((t, h, w), item_hash, offset) in enumerate(
            zip(entry.grids, entry.hashes, entry.offsets)
        ):
            n = t * h * w
            if shm_names is None:
                feature = features[row : row + n]
            else:


                feature = ShmPointerMMData.__new__(ShmPointerMMData)
                feature.__setstate__(
                    {
                        "shm_name": shm_names[index],
                        "shape": (n, spec.feature_dim),
                        "dtype": torch.float32,
                        "precomputed_hash": item_hash,
                    }
                )
            items.append(
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    feature=feature,
                    hash=item_hash,
                    offsets=[tuple(offset)],
                    model_specific_data={
                        "image_grid_thw": torch.tensor([[t, h, w]], dtype=torch.long)
                    },
                )
            )
            row += n
        if envs.SGLANG_MM_PRECOMPUTE_HASH.get():
            for item in items:
                item.set_pad_value()
        return MultimodalProcessorOutput(
            mm_items=items,
            im_token_id=spec.image_token_id,
            im_start_id=spec.vision_start_token_id,
            im_end_id=spec.vision_end_token_id,
            video_token_id=spec.video_token_id,
            mrope_positions=torch.from_numpy(entry.mrope.reshape(3, -1)),
            mrope_position_delta=torch.tensor([[entry.mrope_delta]], dtype=torch.long),
        )


class RustServer:


    def __init__(
        self,
        server: Server,
        mm_spec: Optional[NativeMmSpec] = None,
        max_per_poll: int = 256,
    ):
        self.server = server
        self.mm_spec = mm_spec
        self._max_per_poll = max_per_poll

    @classmethod
    def launch(cls, scheduler: Scheduler) -> RustServer:

        from sglang.srt.rust_extensions import load_rust_extension

        Server = load_rust_extension("sglang.srt.rust_extensions._server").Server


        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        server_args = scheduler.server_args


        if get_serving().preferred_sampling_params:
            raise ValueError(
                "SGLANG_RUST_SERVER does not yet apply --preferred-sampling-params "
                "(the Python TokenizerManager merges it into every request; the rust "
                "ingress has no equivalent). Launch without SGLANG_RUST_SERVER, or "
                "drop --preferred-sampling-params and send those values per request."
            )
        http_addr = f"{get_serving().host}:{server_args.port}"


        dp_rank = scheduler.ps.attn_dp_rank if scheduler.ps.dp_size > 1 else None
        if dp_rank is not None:
            http_addr = f"{get_serving().host}:{server_args.port + dp_rank}"

        launch_cores, server_cores = cls._partition_cores(
            mm_workers=(
                (server_args.mm_processor_worker_num or NativeMmHost.AUTO_MM_WORKERS)
                if scheduler.model_config.is_multimodal
                else 0
            )
        )

        server = Server(
            cls._build_server_args(scheduler),

            cores=server_cores,
            http_addr=http_addr,
        )


        mm_spec = None
        if scheduler.model_config.is_multimodal:


            if server_cores is not None:
                try:
                    os.sched_setaffinity(0, set(server_cores))
                except OSError as e:
                    logger.warning(
                        "rust server: cannot confine mm threads to server cores: %s", e
                    )
            mm_host = NativeMmHost(
                server_args=server_args,
                model_config=scheduler.model_config,
                processor=scheduler.processor,
            )
            mm_spec = mm_host.resolve_native_spec()
            if mm_spec is None:
                supported = sorted(
                    set(chain.from_iterable(f.model_types for f in NATIVE_MM_FAMILIES))
                )
                raise RuntimeError(
                    "SGLANG_RUST_SERVER=1: no native Rust MM pipeline for "
                    f"model_type={scheduler.model_config.hf_config.model_type!r} "
                    f"(supported: {', '.join(supported)}; "
                    "images only). Unset SGLANG_RUST_SERVER to serve this model."
                )
            server.start_mm_workers(cls._build_mm_spec(mm_spec), mm_host.mm_workers)


        if launch_cores is not None:
            try:

                os.sched_setaffinity(0, set(launch_cores))
            except OSError as e:
                logger.warning("rust server: cannot pin scheduler launch thread: %s", e)


        dp_note = (
            "" if dp_rank is None else f" (DP rank {dp_rank}/{scheduler.ps.dp_size})"
        )
        logger.info(
            "SGLANG_RUST_SERVER enabled, Rust server listen on %s%s",
            http_addr,
            dp_note,
        )

        return cls(server, mm_spec=mm_spec)

    def wait_request(self, timeout_ms: int) -> None:

        self.server.wait_request(timeout_ms)

    def drain(self, max_recv: int) -> List[Any]:

        limit = max_recv if max_recv > 0 else self._max_per_poll
        batch = self.server.recv_requests(limit)

        headers, data, lengths = batch.headers, batch.data, batch.lengths
        if not headers:
            return []

        ids_view = memoryview(data)
        out = []
        pos = 0
        for header, n in zip(headers, lengths):
            nbytes = n * 8
            try:
                obj = msgpack_decode_explained(header)
            except MsgpackDecodeError as e:

                logger.warning(
                    "rust ingress: dropping undecodable request %s: %s", e.rid, e.reason
                )
                if e.rid is not None:
                    self.server.push_error(e.rid, f"invalid request: {e.reason}")
                pos += nbytes
                continue
            if n:
                ids = array("q")
                ids.frombytes(ids_view[pos : pos + nbytes])
                obj.input_ids = ids
                pos += nbytes
            if self.mm_spec is not None and isinstance(obj, TokenizedGenerateReqInput):


                native = self.server.take_mm(obj.rid)
                if native is not None:
                    obj.mm_inputs = NativeMmHost.build_native_mm(self.mm_spec, native)
            out.append(obj)
        return out

    def push_control_output(self, recv_req, output) -> None:


        assert (
            recv_req.rid is not None
        ), f"control response without rid: {type(output).__name__}"


        payload = (
            msgspec.structs.asdict(output)
            if isinstance(output, msgspec.Struct)
            else output
        )


        encoded = msgspec.msgpack.encode(payload, enc_hook=str)

        self.server.push_control_result(recv_req.rid, encoded)

    def push_generation(self, payload: BatchTokenIDOutput) -> None:

        output_ids = payload.output_ids or []
        prompt_tokens = payload.prompt_tokens or []


        has_extra = bool(
            payload.output_token_logprobs_val
            or payload.input_token_logprobs_val
            or payload.output_top_logprobs_val
            or payload.input_top_logprobs_val
            or payload.output_token_ids_logprobs_val
            or payload.input_token_ids_logprobs_val
            or payload.output_hidden_states
        )


        rids = payload.rids
        finish_reasons = payload.finished_reasons
        tok_lens = list(map(len, output_ids))
        flat_ids = array("i", chain.from_iterable(output_ids))


        header_cols = [rids, finish_reasons, prompt_tokens, tok_lens]
        data_cols = [flat_ids.tobytes()]

        if has_extra:


            batch_size = len(rids)
            extras = (
                FlatPairColumns(
                    "output_token_logprobs",
                    payload.output_token_logprobs_val or [],
                    payload.output_token_logprobs_idx or [],
                ),
                FlatPairColumns(
                    "input_token_logprobs",
                    payload.input_token_logprobs_val or [],
                    payload.input_token_logprobs_idx or [],
                    first_none_to_nan=True,
                ),
                RaggedPairColumns(
                    "output_top_logprobs",
                    payload.output_top_logprobs_val or [],
                    payload.output_top_logprobs_idx or [],
                ),
                RaggedPairColumns(
                    "input_top_logprobs",
                    payload.input_top_logprobs_val or [],
                    payload.input_top_logprobs_idx or [],
                ),
                RaggedPairColumns(
                    "output_token_ids_logprobs",
                    payload.output_token_ids_logprobs_val or [],
                    payload.output_token_ids_logprobs_idx or [],
                ),
                RaggedPairColumns(
                    "input_token_ids_logprobs",
                    payload.input_token_ids_logprobs_val or [],
                    payload.input_token_ids_logprobs_idx or [],
                ),
                NestedRowColumns(
                    "output_hidden_states", payload.output_hidden_states or []
                ),
            )


            active = []
            for extra in extras:
                populated = False
                for name, col in extra.columns():
                    assert len(col) in (
                        0,
                        batch_size,
                    ), f"extras column {name}: {len(col)} entries for a batch of {batch_size}"
                    populated |= len(col) > 0
                if populated:
                    active.append(extra)


            for extra in active:
                accept = extra.accept
                for i in range(batch_size):
                    accept(i)

            for extra in extras:
                header_cols += extra.header_cols()
                data_cols += extra.data_cols()

        header = msgspec.msgpack.encode(header_cols)


        if not self.server.push_decode_result_batch(header, data_cols):
            logger.warning(
                "Rust egress closed; dropped batch of %d requests during shutdown",
                len(rids),
            )

    @staticmethod
    def _build_mm_spec(spec: NativeMmSpec) -> MmSpec:

        from sglang.srt.rust_extensions import load_rust_extension

        ext = load_rust_extension("sglang.srt.rust_extensions._server")
        family = {"qwen_vl": ext.MmFamily.QwenVl}[spec.family]
        resample = {"aten_u8": ext.MmResample.AtenU8, "pil": ext.MmResample.Pil}[
            spec.resample
        ]
        return ext.MmSpec(
            family=family,
            feature_shm=spec.feature_shm,
            image_token_id=spec.image_token_id,
            patch_size=spec.patch_size,
            merge_size=spec.merge_size,
            temporal_patch_size=spec.temporal_patch_size,
            min_pixels=spec.min_pixels,
            max_pixels=spec.max_pixels,
            image_mean=spec.image_mean,
            image_std=spec.image_std,
            resample=resample,
        )

    @staticmethod
    def _build_server_args(scheduler: Scheduler) -> ServerArgs:

        from sglang.srt.rust_extensions import load_rust_extension

        ext = load_rust_extension("sglang.srt.rust_extensions._server")

        sa = resolving_view(scheduler.server_args)
        mc = scheduler.model_config
        disaggregation_mode = {
            "null": ext.DisaggregationMode.Null,
            "prefill": ext.DisaggregationMode.Prefill,
            "decode": ext.DisaggregationMode.Decode,
        }[sa.disaggregation_mode]
        return ext.ServerArgs(
            model_path=sa.model_path,
            served_model_name=sa.served_model_name,
            tokenizer_path=sa.tokenizer_path,
            revision=sa.revision,
            load_format=sa.load_format,
            weight_version=sa.weight_version,
            host=get_serving().host,
            port=sa.port,
            log_level=get_observability().log_level,
            log_level_http=sa.log_level_http,
            chat_template=sa.chat_template,
            tool_call_parser=sa.tool_call_parser,
            reasoning_parser=sa.reasoning_parser,
            stream_response_default_include_usage=sa.stream_response_default_include_usage,
            tokenizer_worker_num=sa.tokenizer_worker_num,
            detokenizer_worker_num=sa.detokenizer_worker_num,
            skip_tokenizer_init=sa.skip_tokenizer_init,
            incremental_streaming_output=sa.incremental_streaming_output,
            disaggregation_mode=disaggregation_mode,
            model_config=ext.ModelConfig(
                context_len=mc.context_len,
                vocab_size=mc.vocab_size,
                is_multimodal=mc.is_multimodal,


                default_sampling_params=ext.DefaultSamplingParams(
                    **mc.get_default_sampling_params()
                ),
            ),


            preferred_sampling_params=(
                json.dumps(sa.preferred_sampling_params)
                if sa.preferred_sampling_params is not None
                else None
            ),
            allow_auto_truncate=sa.allow_auto_truncate,
            enable_return_hidden_states=sa.enable_return_hidden_states,


            num_reserved_tokens=compute_num_reserved_tokens(),


            version=__version__,
            max_total_num_tokens=scheduler.max_total_num_tokens,
        )

    @staticmethod
    def _partition_cores(
        mm_workers: int = 0,
    ) -> Tuple[Optional[List[int]], Optional[List[int]]]:

        if not hasattr(os, "sched_getaffinity"):
            return None, None
        try:
            allowed = sorted(os.sched_getaffinity(0))
        except OSError as e:
            logger.warning("rust server: cannot read cpu affinity: %s", e)
            return None, None


        if len(allowed) < 4:
            logger.info(
                "rust server: only %d cores allowed; running pools unpinned",
                len(allowed),
            )
            return None, None


        reserve = min(2, len(allowed) // 4)
        launch_cores = allowed[:reserve]


        pool_budget = max(8, mm_workers + 4)
        server_cores = allowed[reserve : reserve + pool_budget]
        logger.info(
            "rust server cores=%s, scheduler launch cores=%s",
            server_cores,
            launch_cores,
        )
        return launch_cores, server_cores
