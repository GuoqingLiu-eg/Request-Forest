# Copyright 2023-2024 SGLang Team
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
import types
from typing import (
    Annotated,
    Any,
    Callable,
    List,
    Literal,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

A = Annotated


@dataclasses.dataclass(frozen=True)
class Arg:


    help: str = ""
    choices: Optional[list] = None
    aliases: Optional[List[str]] = None
    cli_name: Optional[str] = None
    type_parser: Optional[Callable] = None
    nargs: Optional[str] = None
    required: Optional[bool] = None
    action: Optional[Any] = None
    action_kwargs: Optional[dict] = None
    const: Optional[Any] = None


    no_cli: bool = False


    resolvable: bool = False


@dataclasses.dataclass(frozen=True)
class NS:


    path: str


@functools.lru_cache(maxsize=None)
def namespace_of(cls) -> dict:

    if not dataclasses.is_dataclass(cls):
        return {}
    hints = get_type_hints(cls, include_extras=True)
    out = {}
    for field in dataclasses.fields(cls):
        tp = hints.get(field.name, field.type)
        if get_origin(tp) is Annotated:
            for a in get_args(tp)[1:]:
                if isinstance(a, NS):
                    out[field.name] = a.path
                    break
    return out


@functools.lru_cache(maxsize=None)
def field_names(cls) -> frozenset:

    if not dataclasses.is_dataclass(cls):
        return frozenset()
    return frozenset(field.name for field in dataclasses.fields(cls))


@functools.lru_cache(maxsize=None)
def resolvable_fields(cls) -> frozenset:

    if not dataclasses.is_dataclass(cls):
        return frozenset()
    hints = get_type_hints(cls, include_extras=True)
    names = set()
    for field in dataclasses.fields(cls):
        _, arg = _unwrap_annotated(hints.get(field.name, field.type))
        if arg is not None and arg.resolvable:
            names.add(field.name)
    return frozenset(names)


_MISSING = dataclasses.MISSING


def _unwrap_annotated(tp):

    origin = get_origin(tp)
    if origin is Annotated:
        args = get_args(tp)
        inner = args[0]
        for a in args[1:]:
            if isinstance(a, Arg):
                return inner, a
            if isinstance(a, str):
                return inner, Arg(help=a)
        return inner, None
    return tp, None


def _unwrap_optional(tp):

    origin = get_origin(tp)
    is_union = origin is Union or (
        hasattr(types, "UnionType") and origin is types.UnionType
    )
    if is_union:
        args = get_args(tp)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0], True
    return tp, False


def _unwrap_literal(tp):

    origin = get_origin(tp)
    if origin is Literal:
        return list(get_args(tp))
    return None


def _infer_type_func(tp):

    if tp is str:
        return str
    if tp is int:
        return int
    if tp is float:
        return float
    return str


def _field_default(field):

    if field.default is not _MISSING:
        return field.default
    if field.default_factory is not _MISSING:
        return field.default_factory()
    return _MISSING


def _field_to_cli_name(name: str) -> str:

    return "--" + name.replace("_", "-")


def add_cli_args_from_dataclass(parser, cls, *, fields: Optional[List[str]] = None):

    hints = get_type_hints(cls, include_extras=True)

    for field in dataclasses.fields(cls):
        if fields is not None and field.name not in fields:
            continue

        hint = hints.get(field.name)
        if hint is None:
            continue

        raw_type, arg_meta = _unwrap_annotated(hint)
        if arg_meta is None:
            continue
        if arg_meta.no_cli:
            continue

        cli_name = arg_meta.cli_name or _field_to_cli_name(field.name)
        names = [cli_name] + (arg_meta.aliases or [])
        default = _field_default(field)


        auto_dest = cli_name.lstrip("-").replace("-", "_")
        dest_kwarg = {"dest": field.name} if field.name != auto_dest else {}


        if arg_meta.action is not None:
            kwargs = {
                "action": arg_meta.action,
                "help": arg_meta.help,
                **dest_kwarg,
            }
            if default is not _MISSING:
                kwargs["default"] = default
            if arg_meta.action_kwargs:
                kwargs.update(arg_meta.action_kwargs)
            parser.add_argument(*names, **kwargs)
            continue


        inner_type, is_optional = _unwrap_optional(raw_type)


        literal_vals = _unwrap_literal(inner_type)
        if literal_vals is not None:
            choices = arg_meta.choices or literal_vals

            val_type = type(literal_vals[0]) if literal_vals else str
            type_func = arg_meta.type_parser or _infer_type_func(val_type)
            kwargs = dict(
                type=type_func, choices=choices, help=arg_meta.help, **dest_kwarg
            )
            if default is not _MISSING:
                kwargs["default"] = default
            if arg_meta.const is not None:
                kwargs["const"] = arg_meta.const
            parser.add_argument(*names, **kwargs)
            continue


        origin = get_origin(inner_type)
        if (origin is list or origin is List) and arg_meta.type_parser is None:
            elem_args = get_args(inner_type)
            elem_type = elem_args[0] if elem_args else str
            type_func = _infer_type_func(elem_type)
            nargs = arg_meta.nargs or "+"
            kwargs = dict(
                type=type_func,
                nargs=nargs,
                help=arg_meta.help,
                **dest_kwarg,
            )
            if arg_meta.choices:
                kwargs["choices"] = arg_meta.choices
            if default is not _MISSING:
                kwargs["default"] = default
            if arg_meta.const is not None:
                kwargs["const"] = arg_meta.const
            parser.add_argument(*names, **kwargs)
            continue


        if inner_type is bool:
            kwargs = dict(action="store_true", help=arg_meta.help, **dest_kwarg)
            if default is not _MISSING:
                kwargs["default"] = default
            parser.add_argument(*names, **kwargs)
            continue


        type_func = arg_meta.type_parser or _infer_type_func(inner_type)
        kwargs = dict(type=type_func, help=arg_meta.help, **dest_kwarg)
        if arg_meta.choices:
            kwargs["choices"] = arg_meta.choices
        if arg_meta.nargs:
            kwargs["nargs"] = arg_meta.nargs
        if default is not _MISSING:
            kwargs["default"] = default
        if arg_meta.const is not None:
            kwargs["const"] = arg_meta.const
        if (
            arg_meta.required is True
            or (arg_meta.required is None and default is _MISSING)
        ) and any(name.startswith("-") for name in names):
            kwargs["required"] = True
        parser.add_argument(*names, **kwargs)
