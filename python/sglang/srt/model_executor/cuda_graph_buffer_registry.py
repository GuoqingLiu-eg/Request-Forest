# Copyright 2023-2026 SGLang Team
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

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import torch

from sglang.srt.model_executor.input_buffers import (
    INDEX_SEMANTIC_BUFFERS,
    share_input_buffer,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


_has_foreach_copy = hasattr(torch, "_foreach_copy_")


def _grouped_foreach_copy_(dsts: List[torch.Tensor], srcs: List[torch.Tensor]) -> None:


    def _foreach_copy(
        group_dsts: List[torch.Tensor], group_srcs: List[torch.Tensor]
    ) -> None:
        if _has_foreach_copy:
            torch._foreach_copy_(group_dsts, group_srcs)
        else:
            for dst, src in zip(group_dsts, group_srcs):
                dst.copy_(src)

    groups: Dict[Tuple[torch.dtype, torch.dtype], Tuple[List, List]] = {}
    for dst, src in zip(dsts, srcs):
        key = (dst.dtype, src.dtype)
        if key not in groups:
            groups[key] = ([], [])
        groups[key][0].append(dst)
        groups[key][1].append(src)
    for group_dsts, group_srcs in groups.values():
        _foreach_copy(group_dsts, group_srcs)


class PaddingPolicy(Enum):


    KEEP_PAD = "keep_pad"
    FILL_SENTINEL = "fill_sentinel"
    ZERO = "zero"
    FOREACH_COPY = "foreach_copy"
    FILL_ONCE = "fill_once"


@dataclass
class FillContext:


    raw_bs: int
    padded_bs: int
    raw_num_tokens: int
    padded_num_tokens: int


    pp_proxy_tensors: Optional[Any] = None


@dataclass
class GraphSlot:


    name: str
    shape_fn: Callable[[int, int], Tuple[int, ...]]
    dtype: torch.dtype
    axis: str = "tokens"
    device: Optional[torch.device] = None
    padding_policy: PaddingPolicy = PaddingPolicy.FOREACH_COPY
    pad_value: Optional[Any] = None
    enabled: bool = True
    copy_from_fb: bool = True
    post_fill: Optional[Callable[[torch.Tensor, ForwardBatch, FillContext], None]] = (
        None
    )
    slice_fn: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None
    source_fn: Optional[
        Callable[[ForwardBatch, FillContext], Optional[torch.Tensor]]
    ] = None


    buffer: Optional[torch.Tensor] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.axis not in ("bs", "tokens", "none"):
            raise ValueError(
                f"GraphSlot {self.name!r}: axis must be one of "
                f"'bs'/'tokens'/'none', got {self.axis!r}"
            )

    def _padded_n(self, padded_bs: int, padded_num_tokens: int) -> int:
        if self.axis == "bs":
            return padded_bs
        if self.axis == "tokens":
            return padded_num_tokens

        return self.buffer.shape[0] if self.buffer is not None else 0

    def _raw_n(self, raw_bs: int, raw_num_tokens: int) -> int:
        if self.axis == "bs":
            return raw_bs
        if self.axis == "tokens":
            return raw_num_tokens
        return self.buffer.shape[0] if self.buffer is not None else 0

    def slice_for(self, padded_bs: int, padded_num_tokens: int) -> torch.Tensor:

        if self.buffer is None:
            raise RuntimeError(f"GraphSlot {self.name!r}: buffer not allocated")
        if self.slice_fn is not None:
            return self.slice_fn(
                self.buffer, self._padded_n(padded_bs, padded_num_tokens)
            )
        if self.axis == "none":
            return self.buffer
        return self.buffer[: self._padded_n(padded_bs, padded_num_tokens)]

    def reset_padding(self, raw_n: int, padded_n: int) -> None:

        if self.buffer is None or raw_n >= padded_n:
            return
        if self.padding_policy in (
            PaddingPolicy.KEEP_PAD,
            PaddingPolicy.FOREACH_COPY,
            PaddingPolicy.FILL_ONCE,
        ):
            return


        if self.slice_fn is not None:


            padded_slice = self.slice_fn(self.buffer, padded_n)
            tail = (
                padded_slice[..., raw_n:padded_n]
                if padded_slice.dim() > 1
                else padded_slice[raw_n:padded_n]
            )
        else:
            tail = self.buffer[raw_n:padded_n]
        if self.padding_policy == PaddingPolicy.FILL_SENTINEL:
            if self.pad_value is None:
                raise RuntimeError(
                    f"GraphSlot {self.name!r}: FILL_SENTINEL requires pad_value"
                )
            tail.fill_(self.pad_value)
        elif self.padding_policy == PaddingPolicy.ZERO:
            tail.zero_()


class CudaGraphBufferRegistry:


    def __init__(
        self,
        *,
        device: torch.device,
        max_bs: int,
        max_num_tokens: int,
        share_pool: bool = False,
    ) -> None:
        self.device = device
        self.max_bs = max_bs
        self.max_num_tokens = max_num_tokens


        self.share_pool = share_pool
        self._slots: Dict[str, GraphSlot] = {}


    def register_slot(
        self, slot: GraphSlot, bind: Optional[torch.Tensor] = None
    ) -> GraphSlot:

        if slot.name in self._slots:
            raise ValueError(
                f"GraphSlot {slot.name!r} already registered; "
                "use enable()/disable() to gate per-iter."
            )
        if not slot.enabled:


            self._slots[slot.name] = slot
            return slot
        shape = slot.shape_fn(self.max_bs, self.max_num_tokens)
        device = slot.device if slot.device is not None else self.device
        if bind is not None:
            expected = tuple(shape)
            if tuple(bind.shape) != expected:
                raise ValueError(
                    f"bind tensor for slot {slot.name!r} has shape "
                    f"{tuple(bind.shape)}, expected {expected}."
                )
            if bind.dtype != slot.dtype:
                raise ValueError(
                    f"bind tensor for slot {slot.name!r} has dtype {bind.dtype}, "
                    f"expected {slot.dtype}."
                )
            slot.buffer = bind
            self._slots[slot.name] = slot
            return slot
        buffer = torch.zeros(shape, dtype=slot.dtype, device=device)
        if self.share_pool:


            buffer = share_input_buffer(slot.name, buffer)
        if (
            slot.padding_policy
            in (PaddingPolicy.FILL_SENTINEL, PaddingPolicy.FILL_ONCE)
            and slot.pad_value is not None
        ):
            buffer.fill_(slot.pad_value)
        slot.buffer = buffer
        self._slots[slot.name] = slot
        return slot

    def has_slot(self, name: str) -> bool:
        return name in self._slots and self._slots[name].enabled

    def get_slot(self, name: str) -> GraphSlot:
        return self._slots[name]

    def slot_names(self) -> List[str]:
        return [name for name, s in self._slots.items() if s.enabled]


    def fill_from(
        self,
        forward_batch: ForwardBatch,
        *,
        raw_bs: int,
        padded_bs: int,
        raw_num_tokens: int,
        padded_num_tokens: int,
        pp_proxy_tensors: Optional[Any] = None,
    ) -> None:

        ctx = FillContext(
            raw_bs=raw_bs,
            padded_bs=padded_bs,
            raw_num_tokens=raw_num_tokens,
            padded_num_tokens=padded_num_tokens,
            pp_proxy_tensors=pp_proxy_tensors,
        )


        for slot in self._slots.values():
            if not slot.enabled or slot.buffer is None:
                continue
            raw_n = slot._raw_n(raw_bs, raw_num_tokens)
            padded_n = slot._padded_n(padded_bs, padded_num_tokens)
            slot.reset_padding(raw_n, padded_n)


        gpu_dsts: List[torch.Tensor] = []
        gpu_srcs: List[torch.Tensor] = []
        cpu_dsts: List[torch.Tensor] = []
        cpu_srcs: List[torch.Tensor] = []
        for slot in self._slots.values():
            if not slot.enabled or slot.buffer is None or not slot.copy_from_fb:
                continue
            if slot.source_fn is not None:


                src = slot.source_fn(forward_batch, ctx)
                if src is None:
                    continue
                dst = slot.buffer[: src.shape[0]]
            else:
                src = getattr(forward_batch, slot.name, None)
                if src is None:
                    continue
                if not isinstance(src, torch.Tensor):


                    continue
                raw_n = slot._raw_n(raw_bs, raw_num_tokens)
                if slot.slice_fn is not None:
                    dst = slot.slice_fn(slot.buffer, raw_n)
                elif slot.axis == "none":
                    dst = slot.buffer
                else:
                    dst = slot.buffer[:raw_n]


            if dst.device.type == "cpu":
                cpu_dsts.append(dst)
                cpu_srcs.append(src)
            else:
                gpu_dsts.append(dst)
                gpu_srcs.append(src)
        if gpu_dsts:
            _grouped_foreach_copy_(gpu_dsts, gpu_srcs)
        for dst, src in zip(cpu_dsts, cpu_srcs):
            dst.copy_(src)


        for slot in self._slots.values():
            if not slot.enabled or slot.buffer is None or slot.post_fill is None:
                continue
            slot.post_fill(slot.buffer, forward_batch, ctx)

    def extract_buffer(
        self,
        *,
        padded_bs: int,
        padded_num_tokens: int,
        forward_batch_template: ForwardBatch,
    ) -> ForwardBatch:

        import dataclasses

        replace_kwargs: Dict[str, Any] = {"batch_size": padded_bs}
        for slot in self._slots.values():
            if not slot.enabled or slot.buffer is None:
                continue


            if "." in slot.name:
                continue
            is_computed = slot.post_fill is not None or not slot.copy_from_fb
            if (
                not is_computed
                and slot.source_fn is None
                and getattr(forward_batch_template, slot.name, None) is None
            ):

                continue
            replace_kwargs[slot.name] = slot.slice_for(padded_bs, padded_num_tokens)
        return dataclasses.replace(forward_batch_template, **replace_kwargs)


def build_decode_registry(
    *,
    device: torch.device,
    max_bs: int,
    max_num_token: int,
    seq_len_fill_value: int,
    cache_loc_dtype: torch.dtype,
    enable_mamba_track: bool = False,
    is_encoder_decoder: bool = False,
    encoder_len_fill_value: int = 0,
    encoder_lens_dtype: torch.dtype = torch.int32,
    enable_num_token_non_padded: bool = False,
    require_gathered_buffer: bool = False,
    enable_prefill_cp: bool = False,
    require_mlp_tp_gather: bool = False,
    dp_size: int = 1,
    register_global_num_tokens: bool = True,
    share_pool: bool = True,
    source: Optional[Any] = None,
) -> CudaGraphBufferRegistry:

    reg = CudaGraphBufferRegistry(
        device=device,
        max_bs=max_bs,
        max_num_tokens=max_num_token,
        share_pool=share_pool,
    )

    def _tokens(_bs: int, mt: int) -> Tuple[int, ...]:
        return (mt,)

    def _bs(bs: int, _mt: int) -> Tuple[int, ...]:
        return (bs,)

    slots = [
        GraphSlot("input_ids", _tokens, torch.int64, axis="tokens"),
        GraphSlot(
            "positions",
            _tokens,
            torch.int64,
            axis="tokens",
            padding_policy=PaddingPolicy.ZERO,
        ),
        GraphSlot(
            "out_cache_loc",
            _tokens,
            cache_loc_dtype,
            axis="tokens",
            padding_policy=PaddingPolicy.ZERO,
        ),
        GraphSlot(
            "req_pool_indices",
            _bs,
            torch.int64,
            axis="bs",
            padding_policy=PaddingPolicy.ZERO,
        ),
        GraphSlot(
            "seq_lens",
            _bs,
            torch.int64,
            axis="bs",
            padding_policy=PaddingPolicy.FILL_SENTINEL,
            pad_value=seq_len_fill_value,
        ),
        GraphSlot(
            "seq_lens_cpu",
            _bs,
            torch.int64,
            axis="bs",
            device=torch.device("cpu"),
            padding_policy=PaddingPolicy.FILL_SENTINEL,
            pad_value=seq_len_fill_value,
        ),
        GraphSlot(
            "mrope_positions",
            lambda _bs2, mt: (3, mt),
            torch.int64,
            axis="tokens",
            slice_fn=lambda buf, n: buf[:, :n],
            padding_policy=PaddingPolicy.ZERO,
        ),
    ]
    if enable_mamba_track:
        slots.append(
            GraphSlot(
                "mamba_track_indices",
                _bs,
                torch.int64,
                axis="bs",
                padding_policy=PaddingPolicy.ZERO,
            )
        )
        slots.append(
            GraphSlot(
                "mamba_track_mask",
                _bs,
                torch.bool,
                axis="bs",
                padding_policy=PaddingPolicy.ZERO,
            )
        )
    if is_encoder_decoder:


        slots.append(
            GraphSlot(
                "encoder_lens",
                _bs,
                encoder_lens_dtype,
                axis="bs",
                padding_policy=PaddingPolicy.FILL_ONCE,
                pad_value=encoder_len_fill_value,
            )
        )
    if enable_num_token_non_padded:
        from sglang.srt.model_executor.forward_batch_info import (
            compute_local_num_token_non_padded,
        )

        def _num_token_non_padded_post_fill(buf, fb, ctx):


            if require_gathered_buffer and not enable_prefill_cp:
                buf.copy_(
                    compute_local_num_token_non_padded(
                        global_num_token_non_padded=fb.num_token_non_padded,
                        num_tokens_per_dp=ctx.padded_num_tokens,
                    )
                )

        slots.append(
            GraphSlot(
                "num_token_non_padded",
                lambda _bs, _mt: (1,),
                torch.int32,
                axis="none",
                post_fill=_num_token_non_padded_post_fill,
            )
        )


    if register_global_num_tokens:

        def _global_num_tokens_post_fill(buf, fb, ctx):

            if require_gathered_buffer:
                buf.fill_(ctx.padded_num_tokens)

        _global_shape = (
            (lambda _bs, _mt: (dp_size,))
            if require_mlp_tp_gather
            else (lambda _bs, _mt: (1,))
        )
        for _global_name in (
            "global_num_tokens_gpu",
            "global_num_tokens_for_logprob_gpu",
        ):
            slots.append(
                GraphSlot(
                    _global_name,
                    _global_shape,
                    torch.int32,
                    axis="none",
                    copy_from_fb=False,
                    post_fill=_global_num_tokens_post_fill,
                )
            )

    for slot in slots:
        bind = None
        if source is not None:
            bind = getattr(source, slot.name, None)
            if bind is None:
                raise ValueError(
                    f"source is missing buffer {slot.name!r} required by the "
                    "decode registry; cannot adopt."
                )
        reg.register_slot(slot, bind=bind)


    if source is not None:
        ngram = getattr(source, "ngram_embedding_info", None)
        if ngram is not None:

            def _ngram_source(attr):
                def _fn(fb, _ctx):
                    info = getattr(fb, "ngram_embedding_info", None)
                    return None if info is None else getattr(info, attr)

                return _fn

            for _attr in ("column_starts", "req_lens"):
                backing = getattr(ngram, _attr)
                reg.register_slot(
                    GraphSlot(
                        name=f"ngram_embedding_info.{_attr}",
                        shape_fn=lambda _bs, _mt, _s=tuple(backing.shape): _s,
                        dtype=backing.dtype,
                        axis="none",
                        padding_policy=PaddingPolicy.KEEP_PAD,
                        source_fn=_ngram_source(_attr),
                    ),
                    bind=backing,
                )


        pp = getattr(source, "pp_proxy_tensors", None)
        if pp is not None:

            def _pp_source(key):
                def _fn(_fb, ctx):
                    ppx = ctx.pp_proxy_tensors
                    return None if ppx is None else ppx.tensors[key]

                return _fn

            for _key, _backing in pp.items():
                reg.register_slot(
                    GraphSlot(
                        name=f"pp_proxy_tensors.{_key}",
                        shape_fn=lambda _bs, _mt, _s=tuple(_backing.shape): _s,
                        dtype=_backing.dtype,
                        axis="none",
                        padding_policy=PaddingPolicy.KEEP_PAD,
                        source_fn=_pp_source(_key),
                    ),
                    bind=_backing,
                )


        for _cname in ("rids_int", "bootstrap_room_ids_int"):
            canary = getattr(source, _cname, None)
            if canary is not None:
                reg.register_slot(
                    GraphSlot(
                        name=_cname,
                        shape_fn=lambda _bs, _mt, _s=tuple(canary.shape): _s,
                        dtype=canary.dtype,
                        axis="bs",
                    ),
                    bind=canary,
                )


    registered = set(reg.slot_names())
    zero_slots = {
        name
        for name in registered
        if reg.get_slot(name).padding_policy is PaddingPolicy.ZERO
    }
    expected_zero = INDEX_SEMANTIC_BUFFERS & registered
    assert zero_slots == expected_zero, (
        "ZERO-policy slots and INDEX_SEMANTIC_BUFFERS disagree: "
        f"zero_but_unlisted={sorted(zero_slots - expected_zero)}, "
        f"listed_but_not_zero={sorted(expected_zero - zero_slots)}"
    )

    return reg


def build_prefill_registry(
    *,
    device: torch.device,
    max_bs: int,
    max_num_token: int,
    cache_loc_dtype: torch.dtype,
    is_multimodal: bool = False,
    hidden_size: int = 0,
    embed_dtype: Optional[torch.dtype] = None,
    enable_mamba_track: bool = False,
    enable_num_token_non_padded: bool = False,
    require_gathered_buffer: bool = False,
    enable_prefill_cp: bool = False,
    register_input_embeds: bool = True,
    share_pool: bool = True,
    source: Optional[Any] = None,
) -> CudaGraphBufferRegistry:

    reg = CudaGraphBufferRegistry(
        device=device,
        max_bs=max_bs,
        max_num_tokens=max_num_token,
        share_pool=share_pool,
    )

    def _tokens(_bs: int, mt: int) -> Tuple[int, ...]:
        return (mt,)

    def _bs(bs: int, _mt: int) -> Tuple[int, ...]:
        return (bs,)

    slots = [
        GraphSlot(
            "input_ids",
            _tokens,
            torch.int64,
            axis="tokens",
            padding_policy=PaddingPolicy.ZERO,
        ),
        GraphSlot(
            "positions",
            _tokens,
            torch.int64,
            axis="tokens",
            padding_policy=PaddingPolicy.ZERO,
        ),
        GraphSlot(
            "out_cache_loc",
            _tokens,
            cache_loc_dtype,
            axis="tokens",
            padding_policy=PaddingPolicy.ZERO,
        ),
    ]
    if is_multimodal:
        slots.append(
            GraphSlot(
                "mrope_positions",
                lambda _bs2, mt: (3, mt),
                torch.int64,
                axis="tokens",
                padding_policy=PaddingPolicy.ZERO,
                slice_fn=lambda buf, n: buf[:, :n],
            )
        )
        if register_input_embeds:
            slots.append(
                GraphSlot(
                    "input_embeds",
                    lambda _bs2, mt: (mt, hidden_size),
                    embed_dtype,
                    axis="tokens",
                    padding_policy=PaddingPolicy.ZERO,
                    copy_from_fb=False,
                )
            )
    if enable_mamba_track:
        slots.append(GraphSlot("mamba_track_indices", _bs, torch.int64, axis="bs"))
        slots.append(GraphSlot("mamba_track_mask", _bs, torch.bool, axis="bs"))
        slots.append(GraphSlot("mamba_track_seqlens", _bs, torch.int32, axis="bs"))
    if enable_num_token_non_padded:
        from sglang.srt.model_executor.forward_batch_info import (
            compute_local_num_token_non_padded_cpu,
        )

        def _prefill_num_token_non_padded_post_fill(buf, fb, ctx):


            if require_gathered_buffer and not enable_prefill_cp:
                buf.fill_(
                    compute_local_num_token_non_padded_cpu(
                        global_num_token_non_padded=fb.num_token_non_padded_cpu,
                        num_tokens_per_dp=ctx.padded_num_tokens,
                    )
                )

        slots.append(
            GraphSlot(
                "num_token_non_padded",
                lambda _bs2, _mt: (1,),
                torch.int32,
                axis="none",
                post_fill=_prefill_num_token_non_padded_post_fill,
            )
        )

    for slot in slots:
        bind = None
        if source is not None:
            bind = getattr(source, slot.name, None)
            if bind is None:
                raise ValueError(
                    f"source is missing buffer {slot.name!r} required by the "
                    "prefill registry; cannot adopt."
                )
        reg.register_slot(slot, bind=bind)
    return reg


def build_eager_registry(
    *,
    device: torch.device,
    max_bs: int,
    max_num_token: int,
    cache_loc_dtype: torch.dtype,
    enable_mamba_track: bool = False,
    is_encoder_decoder: bool = False,
    encoder_len_fill_value: int = 0,
    encoder_lens_dtype: torch.dtype = torch.int32,
    dp_size: int = 1,
) -> CudaGraphBufferRegistry:

    return build_decode_registry(
        device=device,
        max_bs=max_bs,
        max_num_token=max_num_token,
        seq_len_fill_value=0,
        cache_loc_dtype=cache_loc_dtype,
        enable_mamba_track=enable_mamba_track,
        is_encoder_decoder=is_encoder_decoder,
        encoder_len_fill_value=encoder_len_fill_value,
        encoder_lens_dtype=encoder_lens_dtype,
        enable_num_token_non_padded=False,
        register_global_num_tokens=False,
        require_gathered_buffer=False,
        require_mlp_tp_gather=False,
        dp_size=dp_size,
        share_pool=True,
        source=None,
    )
