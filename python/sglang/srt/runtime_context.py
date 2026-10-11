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

import dataclasses
import functools
import logging
import math
import os
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def _ps():
    from sglang.srt.distributed import parallel_state

    return parallel_state


def _dp():
    from sglang.srt.layers import dp_attention

    return dp_attention


_PARALLEL_FIELDS = frozenset(
    {
        "world_size",
        "world_rank",
        "tp_size",
        "tp_rank",
        "pp_size",
        "pp_rank",
        "moe_ep_size",
        "moe_ep_rank",
        "moe_dp_size",
        "moe_dp_rank",
        "moe_tp_size",
        "moe_tp_rank",
        "attn_tp_size",
        "attn_tp_rank",
        "attn_cp_size",
        "attn_cp_rank",
        "dcp_enabled",
        "dcp_size",
        "dcp_rank",
        "attn_dcp_size",
        "attn_dcp_rank",
        "attn_dp_size",
        "attn_dp_rank",
        "world_group",
        "tp_group",
        "pp_group",
        "moe_ep_group",
        "moe_dp_group",
        "moe_tp_group",
        "attn_tp_group",
        "attn_cp_group",
        "dcp_group",
    }
)


class ParallelContext:


    __slots__ = ("_overrides", "_config")

    def __init__(self):
        self._overrides = {}
        self._config = None

    @property
    def config(self) -> _ConfigBag:

        config = self._config
        if config is None:
            raise ValueError("config namespace 'parallel' not published")
        return config

    def __getattr__(self, name):


        if name.startswith("_"):


            raise AttributeError(name)
        config = self._config
        if config is not None and name in config._fields:
            raise AttributeError(
                f"{name!r} is a parallel config leaf, not live topology; read it "
                f"as get_parallel().config.{name}"
            )
        raise AttributeError(f"ParallelContext has no {name!r}")

    def _v(self, name, getter):
        overrides = self._overrides
        return overrides[name] if name in overrides else getter()

    @contextmanager
    def override(self, **kwargs):

        unknown = set(kwargs) - _PARALLEL_FIELDS
        if unknown:
            raise ValueError(f"unknown parallel field(s): {sorted(unknown)}")
        saved = dict(self._overrides)
        self._overrides.update(kwargs)
        try:
            yield self
        finally:
            self._overrides = saved

    @property
    def world_size(self) -> int:
        return self._v("world_size", _ps().get_world_size)

    @property
    def world_rank(self) -> int:
        return self._v("world_rank", _ps().get_world_rank)

    @property
    def tp_size(self) -> int:
        return self._v("tp_size", _ps().get_tensor_model_parallel_world_size)

    @property
    def tp_rank(self) -> int:
        return self._v("tp_rank", _ps().get_tensor_model_parallel_rank)

    @property
    def pp_size(self) -> int:
        return self._v("pp_size", _ps().get_pipeline_model_parallel_world_size)

    @property
    def pp_rank(self) -> int:
        return self._v("pp_rank", _ps().get_pipeline_model_parallel_rank)

    @property
    def moe_ep_size(self) -> int:
        return self._v("moe_ep_size", _ps().get_moe_expert_parallel_world_size)

    @property
    def moe_ep_rank(self) -> int:
        return self._v("moe_ep_rank", _ps().get_moe_expert_parallel_rank)

    @property
    def moe_dp_size(self) -> int:
        return self._v("moe_dp_size", _ps().get_moe_data_parallel_world_size)

    @property
    def moe_dp_rank(self) -> int:
        return self._v("moe_dp_rank", _ps().get_moe_data_parallel_rank)

    @property
    def moe_tp_size(self) -> int:
        return self._v("moe_tp_size", _ps().get_moe_tensor_parallel_world_size)

    @property
    def moe_tp_rank(self) -> int:
        return self._v("moe_tp_rank", _ps().get_moe_tensor_parallel_rank)

    @property
    def attn_tp_size(self) -> int:
        return self._v("attn_tp_size", _ps().get_attn_tensor_model_parallel_world_size)

    @property
    def attn_tp_rank(self) -> int:
        return self._v("attn_tp_rank", _ps().get_attn_tensor_model_parallel_rank)

    @property
    def attn_cp_size(self) -> int:
        return self._v("attn_cp_size", _ps().get_attn_context_model_parallel_world_size)

    @property
    def attn_cp_rank(self) -> int:
        return self._v("attn_cp_rank", _ps().get_attn_context_model_parallel_rank)

    @property
    def dcp_size(self) -> int:
        return self._v("dcp_size", _ps().get_dcp_world_size)

    @property
    def dcp_rank(self) -> int:
        return self._v("dcp_rank", _ps().get_dcp_rank)

    @property
    def dcp_enabled(self) -> bool:
        def getter():
            if _ps().get_dcp_group_no_assert() is None:
                return False
            return self.dcp_size > 1

        return self._v("dcp_enabled", getter)

    @property
    def attn_dcp_size(self) -> int:
        return self._v(
            "attn_dcp_size", lambda: self.dcp_size if self.dcp_enabled else 1
        )

    @property
    def attn_dcp_rank(self) -> int:
        return self._v(
            "attn_dcp_rank", lambda: self.dcp_rank if self.dcp_enabled else 0
        )

    @property
    def attn_dp_size(self) -> int:
        return self._v("attn_dp_size", _dp().get_attention_dp_size)

    @property
    def attn_dp_rank(self) -> int:
        return self._v("attn_dp_rank", _dp().get_attention_dp_rank)

    @property
    def world_group(self) -> Any:
        return self._v("world_group", _ps().get_world_group)

    @property
    def tp_group(self) -> Any:
        return self._v("tp_group", _ps().get_tp_group)

    @property
    def pp_group(self) -> Any:
        return self._v("pp_group", _ps().get_pp_group)

    @property
    def moe_ep_group(self) -> Any:
        return self._v("moe_ep_group", _ps().get_moe_ep_group)

    @property
    def moe_dp_group(self) -> Any:
        return self._v("moe_dp_group", _ps().get_moe_dp_group)

    @property
    def moe_tp_group(self) -> Any:
        return self._v("moe_tp_group", _ps().get_moe_tp_group)

    @property
    def attn_tp_group(self) -> Any:
        return self._v("attn_tp_group", _ps().get_attn_tp_group)

    @property
    def attn_cp_group(self) -> Any:
        return self._v("attn_cp_group", _ps().get_attn_cp_group)

    @property
    def dcp_group(self) -> Any:
        return self._v("dcp_group", _ps().get_dcp_group)


class _FlagGroupBase:


    def __setattr__(self, name: str, value: Any) -> None:
        if name not in type(self).__dataclass_fields__:
            raise AttributeError(
                f"{type(self).__name__} has no flag '{name}' (leaves are "
                "declared as dataclass fields; check for typos)"
            )
        object.__setattr__(self, name, value)

    @contextmanager
    def override(self, **kwargs):

        fields = type(self).__dataclass_fields__
        unknown = set(kwargs) - set(fields)
        if unknown:
            raise ValueError(
                f"unknown flag(s) for {type(self).__name__}: {sorted(unknown)}"
            )
        saved = {name: getattr(self, name) for name in kwargs}
        for name, value in kwargs.items():
            object.__setattr__(self, name, value)
        try:
            yield self
        finally:
            for name, value in saved.items():
                object.__setattr__(self, name, value)


@dataclasses.dataclass
class CaptureFlags(_FlagGroupBase):


    enable_torch_compile: bool = False


    disable_dispose_tensor: bool = False


@dataclasses.dataclass
class MoeFlags(_FlagGroupBase):


    a2a_backend: Any = None
    runner_backend: Any = None
    speculative_runner_backend: Any = None
    speculative_a2a_backend: Any = None
    deepep_mode: Any = None
    deepep_config: str | None = None
    tbo_enabled: bool | None = None
    sbo_enabled: bool | None = None
    tbo_token_distribution_threshold: float | None = None
    disable_fp4_allgather: bool | None = None
    quantization: str | None = None


    disable_shared_experts_fusion: bool | None = None
    speculative_disable_shared_experts_fusion: bool | None = None


    in_speculative_scope: bool = False


@dataclasses.dataclass
class DpFlags(_FlagGroupBase):


    enabled: bool = False
    use_world_group_for_gather: bool = False
    joiner_skip_all_gather: bool = False


    max_len_with_idle: bool = False


    buffer_hidden_size: Any = None
    buffer_dtype: Any = None
    buffer_device: Any = None


@dataclasses.dataclass
class Flags(_FlagGroupBase):


    capture: CaptureFlags = dataclasses.field(default_factory=CaptureFlags)
    moe: MoeFlags = dataclasses.field(default_factory=MoeFlags)
    dp: DpFlags = dataclasses.field(default_factory=DpFlags)


@dataclasses.dataclass
class Resources(_FlagGroupBase):


    graph_memory_pool: Any = None


    expert_distribution_recorder: Any = None
    expert_location_metadata: Any = None

    lplb_solvers: dict = dataclasses.field(default_factory=dict)

    streams: dict = dataclasses.field(default_factory=dict)


    buffers: dict = dataclasses.field(default_factory=dict)


    tbo_event_pool: dict = dataclasses.field(default_factory=dict)

    indexer_capturer: Any = None
    experts_capturer: Any = None

    tcp_store: Any = None

    trace_level: Any = None


class ForwardFlags:


    _DEFAULTS = {
        "multi_stream": False,
        "moe_output_buffer": None,


        "attn_input_scattered": False,
        "attn_inputs": None,


        "is_extend_in_batch": False,


        "fuse_mlp_allreduce": False,
        "mlp_reduce_scatter": False,
        "flashinfer_trtllm_bypass": False,
    }


    _GRAPH_VISIBLE = frozenset(
        {
            "attn_input_scattered",
            "attn_inputs",
            "is_extend_in_batch",
            "fuse_mlp_allreduce",
            "mlp_reduce_scatter",
            "flashinfer_trtllm_bypass",
        }
    )

    __slots__ = ("_vars", "_plain")

    def __init__(self):
        import contextvars

        object.__setattr__(
            self,
            "_plain",
            {
                name: default
                for name, default in self._DEFAULTS.items()
                if name in self._GRAPH_VISIBLE
            },
        )
        object.__setattr__(
            self,
            "_vars",
            {
                name: contextvars.ContextVar(f"forward.{name}", default=default)
                for name, default in self._DEFAULTS.items()
                if name not in self._GRAPH_VISIBLE
            },
        )

    def __getattr__(self, name: str) -> Any:
        plain = self._plain
        if name in plain:
            return plain[name]
        try:
            return self._vars[name].get()
        except KeyError:
            raise AttributeError(
                f"ForwardFlags has no flag '{name}' (flags are declared in "
                "ForwardFlags._DEFAULTS; check for typos)"
            ) from None

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            "ForwardFlags is written through scoped(**kw) (or the legacy "
            "set() shim), never by attribute assignment"
        )

    def set(self, name: str, value: Any) -> None:

        if name in self._plain:
            self._plain[name] = value
        else:
            self._vars[name].set(value)

    @contextmanager
    def scoped(self, **kwargs):

        unknown = set(kwargs) - set(self._DEFAULTS)
        if unknown:
            raise ValueError(f"unknown forward flag(s): {sorted(unknown)}")
        plain_saved = [
            (name, self._plain[name]) for name in kwargs if name in self._plain
        ]
        tokens = []
        for name, value in kwargs.items():
            if name in self._plain:
                self._plain[name] = value
            else:
                tokens.append((self._vars[name], self._vars[name].set(value)))
        try:
            yield self
        finally:
            for var, token in reversed(tokens):
                var.reset(token)
            for name, value in reversed(plain_saved):
                self._plain[name] = value


class _ConfigBag:


    def __init__(self, path: str):
        object.__setattr__(self, "_path", path)
        object.__setattr__(self, "_fields", {})
        object.__setattr__(self, "_subs", {})

    def __getattr__(self, name: str) -> Any:


        fields = object.__getattribute__(self, "_fields")
        if name in fields:
            return fields[name]
        subs = object.__getattribute__(self, "_subs")
        if name in subs:
            return subs[name]
        path = object.__getattribute__(self, "_path")
        raise AttributeError(f"config namespace {path!r} has no leaf/subgroup {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            f"config namespace {self._path!r} is read-only; write via "
            "get_context().override(source, ...) or the scoped .override(**kw)"
        )

    def _set(self, name: str, value: Any) -> None:

        object.__getattribute__(self, "_fields")[name] = value
        object.__setattr__(self, name, value)

    def _set_sub(self, name: str, sub: _ConfigBag) -> None:

        object.__getattribute__(self, "_subs")[name] = sub
        object.__setattr__(self, name, sub)

    def __contains__(self, name: str) -> bool:
        return name in object.__getattribute__(self, "_fields")

    @contextmanager
    def override(self, **kwargs):

        fields = object.__getattribute__(self, "_fields")
        unknown = set(kwargs) - set(fields)
        if unknown:
            path = object.__getattribute__(self, "_path")
            raise ValueError(f"unknown config leaf for {path!r}: {sorted(unknown)}")
        saved = {name: fields[name] for name in kwargs}
        for name, value in kwargs.items():
            self._set(name, value)
        try:
            yield self
        finally:
            for name, value in saved.items():
                self._set(name, value)


def _build_config_bags(server_args: Any) -> dict:

    from sglang.srt.arg_groups.arg_utils import namespace_of
    from sglang.srt.arg_groups.overrides import resolution_result

    _MISSING = object()
    tops: dict = {}
    for field, path in namespace_of(type(server_args)).items():
        value = resolution_result(server_args, field, _MISSING)
        if value is _MISSING:


            raise AttributeError(
                f"config field {field!r} is declared NS({path!r}) but absent from "
                f"the published {type(server_args).__name__}; cannot project its bag leaf"
            )
        parts = path.split(".")
        bag = tops.get(parts[0])
        if bag is None:
            bag = tops[parts[0]] = _ConfigBag(parts[0])
        for depth in range(1, len(parts)):
            name = parts[depth]
            if name in object.__getattribute__(bag, "_fields"):
                raise ValueError(
                    f"config namespace collision: {'.'.join(parts[: depth + 1])!r} "
                    "is declared as both a leaf and a subgroup"
                )
            subs = object.__getattribute__(bag, "_subs")
            child = subs.get(name)
            if child is None:
                child = _ConfigBag(".".join(parts[: depth + 1]))
                bag._set_sub(name, child)
            bag = child
        if field in object.__getattribute__(bag, "_subs"):
            raise ValueError(
                f"config namespace collision: leaf {field!r} under {path!r} "
                "clashes with a subgroup of the same name"
            )
        bag._set(field, value)
    return tops


class RuntimeContext:


    __slots__ = (
        "parallel",
        "_server_args",
        "_config_bags",
        "_overrides_log",
        "_publish_role",
        "flags",
        "resources",
        "forward",
    )

    def __init__(self, parallel: ParallelContext):
        self.parallel = parallel
        self._server_args: ServerArgs | None = None
        self._config_bags: dict | None = None
        self._overrides_log: list = []
        self._publish_role: str | None = None
        self.flags = Flags()
        self.resources = Resources()
        self.forward = ForwardFlags()

    def get_stream(self, name: str) -> Any:

        stream = self.resources.streams.get(name)
        if stream is None:
            import torch

            from sglang.srt.arg_groups.overrides import resolution_result

            device = (
                resolution_result(self._server_args, "device")
                if self._server_args
                else "cuda"
            )
            stream = torch.get_device_module(device).Stream()
            self.resources.streams[name] = stream
        return stream

    def set_stream(self, name: str, stream: Any) -> Any:

        self.resources.streams[name] = stream
        return stream

    def get_buffer(self, name: str, factory: Any) -> Any:

        buf = self.resources.buffers.get(name)
        if buf is None:
            buf = factory()
            self.resources.buffers[name] = buf
        return buf

    @property
    def server_args(self) -> ServerArgs:

        server_args = self._server_args
        if server_args is None:

            raise ValueError("Global server args is not set yet!")
        return server_args

    def set_server_args(self, server_args: ServerArgs) -> None:


        self.flags.capture.enable_torch_compile = getattr(
            server_args, "enable_torch_compile", False
        )
        self._server_args = server_args


        _adaptive_draft_token_bound.cache_clear()


        self._config_bags = _build_config_bags(server_args)


        self.parallel._config = self._config_bags.get("parallel")

        self._overrides_log = []
        self._publish_role = None

    def config_bag(self, name: str) -> _ConfigBag:

        bags = self._config_bags
        if not bags or name not in bags:
            raise ValueError(f"config namespace {name!r} not published")
        if _ROLE_NS_MODE != "off":
            self._check_role_namespace(name)
        return bags[name]

    def is_config_namespace_published(self, name: str) -> bool:

        bags = self._config_bags
        return bags is not None and name in bags

    def _check_role_namespace(self, name: str) -> None:


        role = self._publish_role
        if _ROLE_NS_MODE == "record":
            if not _is_compiling():
                _record_namespace_read(role, name)
        elif _ROLE_NS_MODE == "enforce" and role is not None:
            if role not in ROLE_NAMESPACE_SETS:
                raise ValueError(
                    f"publish role {role!r} has no ROLE_NAMESPACE_SETS entry; "
                    "declare its namespace set (None for the full tree)."
                )
            allowed = ROLE_NAMESPACE_SETS[role]
            if allowed is not None and name not in allowed:
                raise ValueError(
                    f"config namespace {name!r} is outside the declared set "
                    f"for publish role {role!r} ({sorted(allowed)}). If this "
                    "read is legitimate for the process type, extend "
                    "ROLE_NAMESPACE_SETS; if not, the read belongs in a "
                    "different process or behind a per-instance boundary."
                )

    def override(self, source: str, **fields) -> None:

        if not fields:
            return
        bags = self._config_bags
        if bags is None:
            raise ValueError("config not published; cannot override")
        from sglang.srt.arg_groups.arg_utils import namespace_of

        nsmap = namespace_of(type(self._server_args))
        targets = []
        for name, value in fields.items():
            path = nsmap.get(name)
            if path is None:
                raise ValueError(
                    f"override: unknown config field {name!r} (no NS namespace) — "
                    "not a resolved config leaf"
                )
            parts = path.split(".")
            bag = bags.get(parts[0])
            if bag is None:
                raise ValueError(f"override: namespace {parts[0]!r} not published")
            for seg in parts[1:]:
                bag = object.__getattribute__(bag, "_subs").get(seg)
                if bag is None:
                    raise ValueError(
                        f"override: subgroup {seg!r} missing under {path!r}"
                    )
            if name not in bag:
                raise ValueError(f"override: field {name!r} not projected on {path!r}")
            targets.append((bag, name, value))
        for bag, name, value in targets:
            bag._set(name, value)
        self._overrides_log.append((source, dict(fields)))

    def config_leaf(self, name: str):

        bags = self._config_bags
        if bags is None:
            raise ValueError("config not published; cannot read a config leaf")
        from sglang.srt.arg_groups.arg_utils import namespace_of

        path = namespace_of(type(self._server_args)).get(name)
        if path is None:
            raise ValueError(f"{name!r} is not a config leaf (no NS namespace)")
        parts = path.split(".")
        bag = self.config_bag(parts[0])
        for seg in parts[1:]:
            bag = object.__getattribute__(bag, "_subs").get(seg)
            if bag is None:
                raise ValueError(f"subgroup {seg!r} missing under {path!r}")
        return getattr(bag, name)

    def overrides_log(self) -> list:

        return [(source, dict(fields)) for source, fields in self._overrides_log]

    def resolved_server_args_dict(self, base: dict | None = None) -> dict:

        d = self.server_args.resolved_dict() if base is None else dict(base)
        for _source, fields in self._overrides_log:
            d.update(fields)
        return d

    def override_server_args(self, **fields) -> _ServerArgsOverride:

        return _ServerArgsOverride(self, fields)


class _ServerArgsOverride:


    __slots__ = (
        "_context",
        "_fields",
        "_prev_server_args",
        "_prev_bags",
        "_prev_overrides_log",
        "_prev_publish_role",
        "_prev_parallel_config",
        "_prev_capture",
        "_installed",
    )

    def __init__(self, context: RuntimeContext, fields: dict):
        self._context = context
        self._fields = fields
        self._installed = False

    def install(self) -> ServerArgs:

        from sglang.srt.server_args import ServerArgs

        assert not self._installed, "override_server_args already installed"
        ctx = self._context
        self._prev_server_args = ctx._server_args
        self._prev_bags = ctx._config_bags
        self._prev_overrides_log = ctx._overrides_log
        self._prev_publish_role = ctx._publish_role
        self._prev_parallel_config = ctx.parallel._config
        self._prev_capture = ctx.flags.capture.enable_torch_compile
        from sglang.srt.arg_groups.overrides import (
            _apply_fields,
            declare_late_resolution,
        )

        server_args = ServerArgs(model_path="dummy")
        server_args.resolve_once()


        unknown = {name for name in self._fields if not name.startswith("_")} - set(
            type(server_args).__dataclass_fields__
        )
        if unknown:
            raise ValueError(
                f"override_server_args: unknown ServerArgs field(s): {sorted(unknown)}"
            )


        declared = {
            name: value for name, value in self._fields.items() if name[0] != "_"
        }
        if declared:
            declare_late_resolution(server_args, "override_server_args", **declared)


        _apply_fields(server_args, self._fields)
        ctx.set_server_args(server_args)
        self._installed = True
        return server_args

    def restore(self) -> None:

        if not self._installed:
            return
        self._installed = False
        ctx = self._context
        ctx._server_args = self._prev_server_args
        ctx._config_bags = self._prev_bags
        ctx._overrides_log = self._prev_overrides_log
        ctx._publish_role = self._prev_publish_role
        ctx.parallel._config = self._prev_parallel_config
        ctx.flags.capture.enable_torch_compile = self._prev_capture
        self._prev_server_args = None
        self._prev_bags = None
        self._prev_overrides_log = None
        self._prev_parallel_config = None

    def __enter__(self) -> ServerArgs:
        return self.install()

    def __exit__(self, *exc) -> None:
        self.restore()


_PARALLEL = ParallelContext()
_CONTEXT = RuntimeContext(parallel=_PARALLEL)


def get_context() -> RuntimeContext:
    return _CONTEXT


def get_parallel() -> ParallelContext:
    return _PARALLEL


def get_server_args() -> ServerArgs:
    return _CONTEXT.server_args


def get_flags() -> Flags:
    return _CONTEXT.flags


def get_resources() -> Resources:
    return _CONTEXT.resources


def get_forward() -> ForwardFlags:
    return _CONTEXT.forward


def get_device() -> _ConfigBag:
    return _CONTEXT.config_bag("device")


def get_model() -> _ConfigBag:
    return _CONTEXT.config_bag("model")


def get_exec() -> _ConfigBag:
    return _CONTEXT.config_bag("exec")


def get_schedule() -> _ConfigBag:
    return _CONTEXT.config_bag("schedule")


def get_memory() -> _ConfigBag:
    return _CONTEXT.config_bag("memory")


def get_spec() -> _ConfigBag:
    return _CONTEXT.config_bag("spec")


def get_lora() -> _ConfigBag:
    return _CONTEXT.config_bag("lora")


def get_mm() -> _ConfigBag:
    return _CONTEXT.config_bag("mm")


def get_disagg() -> _ConfigBag:
    return _CONTEXT.config_bag("disagg")


def get_serving() -> _ConfigBag:
    return _CONTEXT.config_bag("serving")


def get_observability() -> _ConfigBag:
    return _CONTEXT.config_bag("observability")


ROLE_NAMESPACE_SETS: dict[str, frozenset[str] | None] = {

    "scheduler": None,
    "test": None,


    "dp_controller": frozenset({"exec", "parallel", "device", "disagg"}),


    "tokenizer": None,

    "detokenizer": None,
    "encoder": None,
    "expert_backup": None,
    "weight_cache_daemon": None,


    "diffusion_gpu_worker": None,
}


def _validated_role_ns_mode(value: str) -> str:
    mode = value.strip().lower()
    if mode not in ("off", "record", "enforce"):
        raise ValueError(
            f"SGLANG_ROLE_NAMESPACES={value!r} is not one of off / record / "
            "enforce — refusing to guess (a typo here would silently disable "
            "enforcement)."
        )
    return mode


def _role_ns_mode_from_env() -> str:


    from sglang.srt.environ import envs

    return _validated_role_ns_mode(envs.SGLANG_ROLE_NAMESPACES.get())


_ROLE_NS_MODE = _role_ns_mode_from_env()
_RECORDED_NS_READS: set[tuple[str | None, str]] = set()
_RECORD_DUMP_REGISTERED = False


def _is_compiling() -> bool:


    torch = sys.modules.get("torch")
    return torch is not None and torch.compiler.is_compiling()


def _ensure_record_dump_registered() -> None:
    global _RECORD_DUMP_REGISTERED
    if not _RECORD_DUMP_REGISTERED:
        _RECORD_DUMP_REGISTERED = True
        import atexit

        atexit.register(_dump_recorded_namespace_reads)


def _append_role_ns_out(role: str | None, name: str) -> None:


    from sglang.srt.environ import envs

    out = envs.SGLANG_ROLE_NAMESPACES_OUT.get()
    if not out:
        return
    try:
        with open(out, "a") as f:
            f.write(f"{role} {name}\n")
    except OSError as e:

        print(
            f"[role-namespaces] pid={os.getpid()} failed to append "
            f"({role}, {name}) to {out!r}: {e}",
            file=sys.stderr,
            flush=True,
        )


def _record_namespace_read(role: str | None, name: str) -> None:
    if (role, name) in _RECORDED_NS_READS:
        return
    _RECORDED_NS_READS.add((role, name))
    _append_role_ns_out(role, name)
    _ensure_record_dump_registered()


def _dump_recorded_namespace_reads() -> None:

    by_role: dict = {}
    own_role = _CONTEXT._publish_role
    if own_role is not None:
        by_role.setdefault(own_role, set())
    for role, name in _RECORDED_NS_READS:
        if name == "-":
            by_role.setdefault(role, set())
            continue
        by_role.setdefault(role, set()).add(name)
    for role in sorted(by_role, key=str):
        print(
            f"[role-namespaces] pid={os.getpid()} role={role} "
            f"read={','.join(sorted(by_role[role]))}",
            file=sys.stderr,
            flush=True,
        )


def publish(server_args, *, role: str, hf_config: Any = None) -> RuntimeContext:

    if _ROLE_NS_MODE == "enforce" and role not in ROLE_NAMESPACE_SETS:

        raise ValueError(
            f"publish role {role!r} has no ROLE_NAMESPACE_SETS entry; declare "
            "its namespace set (None for the full tree)."
        )
    server_args.resolve_once()
    discarded = _CONTEXT.overrides_log()
    _CONTEXT.set_server_args(server_args)
    if discarded:
        logger.warning(
            "publish(role=%s) re-projected the config bags and dropped %d "
            "override(s) taken since the last publish: %s",
            role,
            len(discarded),
            ", ".join(
                f"{source}({', '.join(sorted(fields))})" for source, fields in discarded
            ),
        )
    _CONTEXT._publish_role = role
    if _ROLE_NS_MODE == "record":


        _record_namespace_read(role, "-")
        print(
            f"[role-namespaces] pid={os.getpid()} role={role} recording; note: "
            "reads inside torch.compile-traced code are not observed — audit "
            "with compilation disabled before restricting a role.",
            file=sys.stderr,
            flush=True,
        )
    return _CONTEXT


def assert_published(server_args, *, role: str) -> RuntimeContext:

    if _CONTEXT._server_args is server_args and _CONTEXT._publish_role == role:
        return _CONTEXT
    if _CONTEXT._server_args is None:
        detail = "nothing is published in this process"
    elif _CONTEXT._server_args is not server_args:
        detail = (
            "a different record is published "
            f"(role={_CONTEXT._publish_role!r}); this constructor was handed "
            "one the process never published"
        )
    else:
        detail = (
            f"this record is published under role "
            f"{_CONTEXT._publish_role!r}, not {role!r}"
        )
    raise RuntimeError(
        f"config not published for role {role!r}: {detail}. The process entry "
        "publishes -- add publish(server_args, role=...) there rather than "
        "publishing from a constructor."
    )


def publish_role() -> str | None:

    return _CONTEXT._publish_role


def get_stream(name: str) -> Any:
    return _CONTEXT.get_stream(name)


def set_stream(name: str, stream: Any) -> Any:
    return _CONTEXT.set_stream(name, stream)


def get_buffer(name: str, factory: Any) -> Any:
    return _CONTEXT.get_buffer(name, factory)


_GLOBAL_DWDP_MANAGER: Any = None


def get_global_dwdp_manager() -> Any:
    return _GLOBAL_DWDP_MANAGER


def set_global_dwdp_manager(manager: Any) -> None:
    global _GLOBAL_DWDP_MANAGER
    _GLOBAL_DWDP_MANAGER = manager


def _group_leaves(group: _FlagGroupBase) -> dict[str, Any]:

    leaves: dict[str, Any] = {}
    for name in type(group).__dataclass_fields__:
        value = getattr(group, name)
        if isinstance(value, _FlagGroupBase):
            leaves[name] = _group_leaves(value)
        elif isinstance(value, (dict, list)):
            leaves[name] = type(value)(value)
        else:
            leaves[name] = value
    return leaves


def _restore_leaves(group: _FlagGroupBase, leaves: dict[str, Any]) -> None:
    for name, value in leaves.items():
        current = getattr(group, name)
        if isinstance(current, _FlagGroupBase):
            _restore_leaves(current, value)
        elif isinstance(current, dict):
            current.clear()
            current.update(value)
        elif isinstance(current, list):
            current[:] = value
        else:
            setattr(group, name, value)


def snapshot_context() -> dict[str, Any]:

    state: dict[str, Any] = {}
    for name in RuntimeContext.__slots__:
        if name == "parallel":
            continue
        value = getattr(_CONTEXT, name)
        if isinstance(value, _FlagGroupBase):
            state[name] = (value, _group_leaves(value))
        elif isinstance(value, list):
            state[name] = list(value)
        else:
            state[name] = value
    state["__parallel__"] = {
        name: getattr(_CONTEXT.parallel, name)
        for name in type(_CONTEXT.parallel).__slots__
    }
    state["__dwdp__"] = get_global_dwdp_manager()
    return state


def restore_context(state: dict[str, Any]) -> None:

    for name in RuntimeContext.__slots__:
        if name == "parallel":
            continue
        value = state[name]
        if isinstance(value, tuple) and isinstance(value[0], _FlagGroupBase):
            group, leaves = value
            setattr(_CONTEXT, name, group)
            _restore_leaves(group, leaves)
        else:
            setattr(_CONTEXT, name, value)
    for name, value in state["__parallel__"].items():
        setattr(_CONTEXT.parallel, name, value)
    _adaptive_draft_token_bound.cache_clear()
    set_global_dwdp_manager(state["__dwdp__"])


def reset_context() -> None:

    _CONTEXT._server_args = None
    _CONTEXT._config_bags = None
    _adaptive_draft_token_bound.cache_clear()
    _CONTEXT._overrides_log = []
    _CONTEXT._publish_role = None
    _CONTEXT.parallel._config = None
    _CONTEXT.flags = Flags()
    _CONTEXT.resources = Resources()
    _CONTEXT.forward = ForwardFlags()
    set_global_dwdp_manager(None)


def mamba_extra_buffer_enabled() -> bool:

    return (
        get_memory().disable_radix_cache is False
        and get_exec().mamba.mamba_radix_cache_strategy
        in ("extra_buffer", "extra_buffer_lazy")
    )


def mamba_extra_buffer_lazy_enabled() -> bool:

    return (
        get_memory().disable_radix_cache is False
        and get_exec().mamba.mamba_radix_cache_strategy == "extra_buffer_lazy"
    )


def remote_instance_transfer_engine_enabled(load_format: str | None = None) -> bool:

    from sglang.srt.arg_groups.overrides import remote_instance_transfer_engine_of

    return remote_instance_transfer_engine_of(get_model(), load_format)


def max_prefill_buffer_tokens() -> int:

    import math

    schedule = get_schedule()
    chunked = (
        schedule.chunked_prefill_size
        if schedule.chunked_prefill_size and schedule.chunked_prefill_size > 0
        else 0
    )
    tokens = chunked
    if (
        schedule.enable_dynamic_chunking
        and get_parallel().config.pp_size > 1
        and chunked
    ):
        tokens = max(
            tokens, schedule.max_prefill_tokens or 0, math.ceil(chunked * 1.25)
        )
    return tokens


def pre_capture_activation_reserve_mb(gpu_mem: float | None) -> float:

    schedule = get_schedule()
    if get_disagg().disaggregation_mode == "decode":
        running_requests = (
            schedule.max_running_requests
            or get_exec().graph.cuda_graph_config.decode.max_bs
            or 1
        )
        activation_tokens = max(
            running_requests * (get_spec().speculative_num_draft_tokens or 1), 2048
        )
    elif schedule.chunked_prefill_size > 0:
        activation_tokens = max(schedule.chunked_prefill_size, 2048)
    else:
        activation_tokens = max(schedule.max_prefill_tokens, 2048)
    parallel = get_parallel().config
    reserved_mem = (
        512 + activation_tokens * 1.5 + parallel.tp_size * parallel.pp_size / 8 * 1024
    )
    if gpu_mem is not None and gpu_mem > 60 * 1024:
        reserved_mem = max(reserved_mem, 10 * 1024)
    return reserved_mem


def mamba_cache_chunk_size() -> int:

    return get_server_args().mamba_cache_chunk_size


def mamba_checkpoint_grid(tree_page: int) -> int:

    return math.lcm(mamba_cache_chunk_size(), tree_page)


def mamba_track_grid(tree_page: int) -> int:

    return math.lcm(
        mamba_checkpoint_grid(tree_page), get_exec().mamba.mamba_track_interval
    )


def max_speculative_num_draft_tokens() -> int | None:

    spec = get_spec()
    if spec.speculative_num_draft_tokens is None:
        return None
    if not spec.speculative_adaptive:
        return spec.speculative_num_draft_tokens


    return _adaptive_draft_token_bound(spec.speculative_adaptive_config)


@functools.lru_cache(maxsize=8)
def _adaptive_draft_token_bound(cfg_path: str | None) -> int:
    from sglang.srt.speculative.adaptive_spec_params import (
        resolve_candidate_steps_from_config,
    )

    candidate_steps = resolve_candidate_steps_from_config(cfg_path=cfg_path)


    return max(candidate_steps) + 1


def uses_mla_backend() -> bool:

    return get_server_args().use_mla_backend()


def attention_backends() -> tuple:

    from sglang.srt.arg_groups.overrides import attention_backends_of


    return attention_backends_of(get_exec().kernel)


def process_model_config():

    return get_server_args().get_model_config()


def cutedsl_moe_max_num_tokens() -> int:

    from sglang.srt.model_executor.cuda_graph_config import Backend

    spec = get_spec()
    num_tokens_per_req = (
        (spec.speculative_num_draft_tokens or 1) if spec.speculative_algorithm else 1
    )
    prefill_tokens = get_schedule().max_prefill_tokens
    cg_config = get_exec().graph.cuda_graph_config
    if cg_config is not None and cg_config.prefill.backend == Backend.TC_PIECEWISE:
        prefill_tokens = max(prefill_tokens, cg_config.prefill.max_bs or 0)
    decode_max_bs = (cg_config.decode.max_bs if cg_config is not None else 0) or 0
    return max(prefill_tokens, decode_max_bs * num_tokens_per_req)


def is_ep_joiner() -> bool:

    return get_exec().moe.ep_join_mode in ("scale", "recover")


def is_ep_scale_joiner() -> bool:

    return get_exec().moe.ep_join_mode == "scale"
