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


from typing import Any, Callable, Dict, Optional

from transformers import PretrainedConfig

from sglang.srt.configs.muse_glimmer import MuseGlimmerConfig

GGUF_NATIVE_CONFIG_BUILDERS: Dict[str, Callable[[str], PretrainedConfig]] = {
    "muse-glimmer": MuseGlimmerConfig.from_gguf,
}


def read_gguf_architecture(gguf_path: str) -> Optional[str]:

    try:
        from gguf import GGUFReader

        reader = GGUFReader(gguf_path)
        field = reader.fields.get("general.architecture")
        if field is None:
            return None
        value = field.contents()
        return value if isinstance(value, str) else None
    except Exception:
        return None


def has_native_gguf_support(gguf_path: str) -> bool:
    return read_gguf_architecture(gguf_path) in GGUF_NATIVE_CONFIG_BUILDERS


def build_gguf_config(gguf_path: str) -> PretrainedConfig:
    arch = read_gguf_architecture(gguf_path)
    return GGUF_NATIVE_CONFIG_BUILDERS[arch](gguf_path)


_GPT4O_SPLIT_REGEX = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

_PRE_TOKENIZER_REGEX = {

    "llama-bpe": (
        r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
        r"[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|"
        r"\s*[\r\n]+|\s+(?!\S)|\s+"
    ),
    "gpt-4o": _GPT4O_SPLIT_REGEX,
    "llama4": _GPT4O_SPLIT_REGEX,
}

_GGML_TOKEN_TYPE_CONTROL = 3


def build_gguf_generation_config(gguf_path: str):

    from gguf import GGUFReader
    from transformers import GenerationConfig

    reader = GGUFReader(gguf_path)
    meta = {key: field.contents() for key, field in reader.fields.items()}

    stop_ids = []
    for key in ("tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id"):
        if key in meta:
            value = int(meta[key])
            if value not in stop_ids:
                stop_ids.append(value)
    if not stop_ids:
        return None

    fields: Dict[str, Any] = {
        "eos_token_id": stop_ids if len(stop_ids) > 1 else stop_ids[0]
    }
    if "tokenizer.ggml.bos_token_id" in meta:
        fields["bos_token_id"] = int(meta["tokenizer.ggml.bos_token_id"])
    if "tokenizer.ggml.padding_token_id" in meta:
        fields["pad_token_id"] = int(meta["tokenizer.ggml.padding_token_id"])
    return GenerationConfig(**fields)


def build_gguf_tokenizer(gguf_path: str, **kwargs: Any):

    import json

    from gguf import GGUFReader
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    reader = GGUFReader(gguf_path)
    meta = {key: field.contents() for key, field in reader.fields.items()}

    tokens = list(meta["tokenizer.ggml.tokens"])
    token_types = [int(t) for t in meta["tokenizer.ggml.token_type"]]
    merges = [tuple(m.split(" ", 1)) for m in meta["tokenizer.ggml.merges"]]

    pre_name = meta.get("tokenizer.ggml.pre")
    if pre_name not in _PRE_TOKENIZER_REGEX:
        raise ValueError(
            f"No pre-tokenizer regex known for tokenizer.ggml.pre={pre_name!r}; "
            f"known: {sorted(_PRE_TOKENIZER_REGEX)}"
        )

    control_ids = [
        i for i, t in enumerate(token_types) if t == _GGML_TOKEN_TYPE_CONTROL
    ]
    control = set(control_ids)
    vocab = {tok: i for i, tok in enumerate(tokens) if i not in control}

    def token_of(key):
        idx = meta.get(f"tokenizer.ggml.{key}")
        return None if idx is None else tokens[int(idx)]

    bos = token_of("bos_token_id")

    spec = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": i,
                "content": tokens[i],
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for i in control_ids
        ],
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": _PRE_TOKENIZER_REGEX[pre_name]},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": False,
                },
            ],
        },
        "post_processor": None,
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": True,
            "vocab": vocab,
            "merges": [list(m) for m in merges],
        },
    }

    if meta.get("tokenizer.ggml.add_bos_token") and bos is not None:
        bos_id = int(meta["tokenizer.ggml.bos_token_id"])
        spec["post_processor"] = {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": bos, "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
            ],
            "pair": [
                {"SpecialToken": {"id": bos, "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"Sequence": {"id": "B", "type_id": 0}},
            ],
            "special_tokens": {
                bos: {"id": bos, "ids": [bos_id], "tokens": [bos]},
            },
        }

    backend = Tokenizer.from_str(json.dumps(spec))

    named = {
        bos,
        token_of("eos_token_id"),
        token_of("padding_token_id"),
        token_of("unknown_token_id"),
    }
    additional = [tokens[i] for i in control_ids if tokens[i] not in named]

    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token=bos,
        eos_token=token_of("eos_token_id"),
        unk_token=token_of("unknown_token_id"),
        pad_token=token_of("padding_token_id"),
        additional_special_tokens=additional,
        chat_template=meta.get("tokenizer.chat_template"),
        **kwargs,
    )
