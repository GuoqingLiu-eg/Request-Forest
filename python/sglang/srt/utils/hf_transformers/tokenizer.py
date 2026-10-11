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


import json
import logging
import warnings
from pathlib import Path
from typing import Optional, Union

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from sglang.srt.connector import create_remote_connector
from sglang.srt.utils import is_remote_url, logger
from sglang.srt.utils.patch_tokenizer import patch_tokenizer

from ..hf_transformers_patches import _ensure_gguf_version
from .common import (
    _resolve_local_or_cached_file,
    attach_additional_stop_token_ids,
    check_gguf_file,
    gguf_sidecar_dir,
    resolve_runai_obj_uri,
)
from .gguf_native import build_gguf_tokenizer, has_native_gguf_support
from .mistral_utils import (
    _MISTRAL_TOKENIZER_REDIRECTS,
    is_bare_tekken_checkpoint,
    patch_mistral_common_tokenizer,
    retry_without_mistral_common_kwargs,
)


_FAST_LLAMA_TOKENIZER = "hf-internal-testing/llama-tokenizer"


_TOKENIZERS_BACKEND = "TokenizersBackend"


def _load_tokenizer_by_declared_class(tokenizer_name, *args, **kwargs):

    import transformers

    try:
        revision = kwargs.get("revision") or kwargs.get("tokenizer_revision")
        config_file = _resolve_local_or_cached_file(
            tokenizer_name, "tokenizer_config.json", revision
        )
        with open(config_file) as f:
            tok_config = json.load(f)
        tok_class_name = tok_config.get("tokenizer_class")
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as e:
        logger.debug(
            "Failed to read tokenizer_config.json for %s: %s", tokenizer_name, e
        )
        return None

    if not tok_class_name:
        return None


    if tok_class_name in ("PreTrainedTokenizer", "PreTrainedTokenizerBase"):
        return None

    tok_cls = getattr(transformers, tok_class_name, None)
    if tok_cls is None and kwargs.get("trust_remote_code"):

        try:
            auto_map = tok_config.get("auto_map", {})
            auto_tok_ref = auto_map.get("AutoTokenizer")
            if isinstance(auto_tok_ref, (list, tuple)):
                auto_tok_ref = auto_tok_ref[0]
            if auto_tok_ref:
                from transformers.dynamic_module_utils import (
                    get_class_from_dynamic_module,
                )

                tok_cls = get_class_from_dynamic_module(
                    auto_tok_ref,
                    tokenizer_name,
                    code_revision=revision,
                )
        except (OSError, ImportError, ValueError, RuntimeError) as e:
            logger.debug("Dynamic module lookup for %s failed: %s", tok_class_name, e)
    if tok_cls is None:
        return None

    logger.debug(
        "Loading tokenizer for %s directly as %s (bypassing AutoTokenizer)",
        tokenizer_name,
        tok_class_name,
    )
    try:
        return tok_cls.from_pretrained(tokenizer_name, *args, **kwargs)
    except (OSError, ValueError, TypeError, ImportError) as e:
        logger.warning(
            "Direct load as %s failed for %s: %s. "
            "Falling back to AutoTokenizer result.",
            tok_class_name,
            tokenizer_name,
            e,
        )
        return None


class TokenizerWarningsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Calling super().encode with" not in record.getMessage()


_tokenizer_warnings_filter = TokenizerWarningsFilter()


def _install_tokenizer_warnings_filter(tokenizer):
    logging.getLogger(tokenizer.__class__.__module__).addFilter(
        _tokenizer_warnings_filter
    )


def _resolve_tokenizer_name(tokenizer_name, kwargs):

    tokenizer_name = _MISTRAL_TOKENIZER_REDIRECTS.get(tokenizer_name, tokenizer_name)

    if check_gguf_file(tokenizer_name):
        _ensure_gguf_version()
        if gguf_sidecar_dir(tokenizer_name, "tokenizer_config.json") is None:
            kwargs["gguf_file"] = tokenizer_name
        tokenizer_name = Path(tokenizer_name).parent

    tokenizer_name = resolve_runai_obj_uri(tokenizer_name)

    if is_remote_url(tokenizer_name):


        client = create_remote_connector(tokenizer_name)
        client.pull_files(ignore_pattern=["*.pt", "*.safetensors", "*.bin"])
        tokenizer_name = client.get_local_dir()

    return tokenizer_name


def _auto_tokenizer_from_pretrained(tokenizer_name, *args, **common_kwargs):

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, *args, **common_kwargs
        )
        return tokenizer
    except TypeError as e:
        err_msg = (
            "Failed to load the tokenizer. If you are using a LLaMA V1 model "
            f"consider using '{_FAST_LLAMA_TOKENIZER}' instead of the "
            "original tokenizer."
        )
        raise RuntimeError(err_msg) from e
    except ValueError as e:


        if "are not supported by" in str(e) and "MistralCommon" in str(e):
            return retry_without_mistral_common_kwargs(
                tokenizer_name, *args, **common_kwargs
            )


        if not common_kwargs.get("trust_remote_code") and (
            "does not exist or is not currently imported." in str(e)
            or "requires you to execute the tokenizer file" in str(e)
        ):
            err_msg = (
                "Failed to load the tokenizer. If the tokenizer is a custom "
                "tokenizer not yet available in the HuggingFace transformers "
                "library, consider setting `trust_remote_code=True` in LLM "
                "or using the `--trust-remote-code` flag in the CLI."
            )
            raise RuntimeError(err_msg) from e
        raise


def _resolve_tokenizers_backend(tokenizer_name, *args, **common_kwargs):

    logger.debug(
        "Tokenizer loaded as generic TokenizersBackend for %s, "
        "retrying with use_fast=False",
        tokenizer_name,
    )
    common_kwargs = {**common_kwargs, "use_fast": False}
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, *args, **common_kwargs
        )
    except (ValueError, TypeError, OSError, ImportError, RuntimeError) as e:
        raise RuntimeError(
            f"Retry with use_fast=False for {tokenizer_name} also failed "
            f"(initial load returned TokenizersBackend): {e}"
        ) from e

    if type(tokenizer).__name__ == _TOKENIZERS_BACKEND:
        tokenizer = (
            _load_tokenizer_by_declared_class(tokenizer_name, *args, **common_kwargs)
            or tokenizer
        )

    if type(tokenizer).__name__ == _TOKENIZERS_BACKEND:
        if common_kwargs.get("trust_remote_code"):
            logger.warning(
                "Tokenizer for %s is still TokenizersBackend after retries "
                "with --trust-remote-code. Model-specific tokenizer attributes "
                "may be missing.",
                tokenizer_name,
            )
        else:
            logger.debug(
                "Tokenizer for %s loaded as generic TokenizersBackend. "
                "Set --trust-remote-code to load the model-specific tokenizer.",
                tokenizer_name,
            )

    return tokenizer


def _fix_v5_tokenizer_components(tokenizer, model_name_or_path, revision=None):

    backend = getattr(tokenizer, "_tokenizer", None)
    if backend is None:
        return

    try:
        from tokenizers import Tokenizer as RawTokenizer

        tok_file = _resolve_local_or_cached_file(
            model_name_or_path, "tokenizer.json", revision
        )
        raw = RawTokenizer.from_file(tok_file)
    except FileNotFoundError:
        return
    except (OSError, ValueError, RuntimeError) as e:
        logger.warning(
            "_fix_v5_tokenizer_components: unexpected error loading tokenizer.json "
            "for %s, v5 component fix will not be applied: %s",
            model_name_or_path,
            e,
        )
        return

    raw_pre = type(raw.pre_tokenizer).__name__ if raw.pre_tokenizer else None
    loaded_pre = type(backend.pre_tokenizer).__name__ if backend.pre_tokenizer else None

    if raw_pre and loaded_pre and raw_pre != loaded_pre:
        logger.info(
            "Fixing v5 tokenizer component mismatch for %s: "
            "pre_tokenizer %s -> %s, decoder %s -> %s",
            model_name_or_path,
            loaded_pre,
            raw_pre,
            type(backend.decoder).__name__ if backend.decoder else None,
            type(raw.decoder).__name__ if raw.decoder else None,
        )
        backend.pre_tokenizer = raw.pre_tokenizer
        backend.decoder = raw.decoder


def _fix_v5_add_bos_eos_token(tokenizer, model_name_or_path, revision=None):


    _V4_CLASSES_WITH_BOS_EOS_FLAGS = frozenset(
        {
            "LlamaTokenizer",
            "LlamaTokenizerFast",
            "CodeLlamaTokenizer",
            "CodeLlamaTokenizerFast",
            "GemmaTokenizer",
            "GemmaTokenizerFast",
            "CohereTokenizerFast",
        }
    )

    try:
        config_file = _resolve_local_or_cached_file(
            model_name_or_path, "tokenizer_config.json", revision
        )
        with open(config_file) as f:
            config = json.load(f)
    except FileNotFoundError:
        return
    except (OSError, json.JSONDecodeError, ValueError) as e:
        logger.warning(
            "_fix_v5_add_bos_eos_token: failed to read tokenizer_config.json "
            "for %s, BOS/EOS token restoration will not be applied: %s",
            model_name_or_path,
            e,
        )
        return

    tokenizer_class = config.get("tokenizer_class", "")
    if tokenizer_class not in _V4_CLASSES_WITH_BOS_EOS_FLAGS:
        logger.debug(
            "_fix_v5_add_bos_eos_token: skipping %s (tokenizer_class=%s "
            "did not support add_bos/eos_token in v4)",
            model_name_or_path,
            tokenizer_class,
        )
        return


    _V4_DEFAULTS = {"add_bos_token": True, "add_eos_token": False}

    changed = False
    for attr in ("add_bos_token", "add_eos_token"):
        config_val = config.get(attr)
        if config_val is None:

            config_val = _V4_DEFAULTS.get(attr, False)


        if attr == "add_eos_token" and isinstance(tokenizer, PreTrainedTokenizerFast):
            config_val = _V4_DEFAULTS["add_eos_token"]
        current_val = getattr(tokenizer, attr, None)
        if current_val != config_val:
            logger.info(
                "Restoring %s=%s for %s (was %s after v5 loading)",
                attr,
                config_val,
                model_name_or_path,
                current_val,
            )


            setattr(tokenizer, f"_{attr}", config_val)
            changed = True


    if changed and hasattr(tokenizer, "update_post_processor"):
        tokenizer.update_post_processor()


def _fix_special_tokens_pattern(tokenizer):

    pattern = getattr(tokenizer, "special_tokens_pattern", None)
    if pattern == "cls_sep" and (
        tokenizer.cls_token_id is None or tokenizer.sep_token_id is None
    ):
        tokenizer.special_tokens_pattern = "none"


def _apply_post_load_fixes(tokenizer, tokenizer_name, revision):

    _install_tokenizer_warnings_filter(tokenizer)
    _fix_v5_tokenizer_components(tokenizer, tokenizer_name, revision)
    _fix_v5_add_bos_eos_token(tokenizer, tokenizer_name, revision)

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        warnings.warn(
            "Using a slow tokenizer. This might cause a significant "
            "slowdown. Consider using a fast tokenizer instead."
        )

    patch_mistral_common_tokenizer(tokenizer)
    _fix_special_tokens_pattern(tokenizer)
    attach_additional_stop_token_ids(tokenizer)
    return patch_tokenizer(tokenizer)


_fastokens_patched = False


def _ensure_fastokens_patched():

    global _fastokens_patched
    if _fastokens_patched:
        return
    try:
        import fastokens
    except ImportError:
        raise ImportError(
            "The fastokens package is required when --tokenizer-backend=fastokens. "
            "Install it with: pip install 'sglang[fastokens]'"
        ) from None

    fastokens.patch_transformers()
    _fastokens_patched = True
    logger.info("fastokens backend enabled - transformers patched successfully")


def get_tokenizer(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    tokenizer_revision: Optional[str] = None,
    tokenizer_backend: str = "huggingface",
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:


    if tokenizer_name.endswith(".json"):
        from sglang.srt.tokenizer.tiktoken_tokenizer import TiktokenTokenizer

        return TiktokenTokenizer(tokenizer_name)

    if tokenizer_backend == "fastokens":
        _ensure_fastokens_patched()

    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False
    elif tokenizer_mode == "auto":


        if "use_fast" not in kwargs:
            kwargs["use_fast"] = True

    if (
        check_gguf_file(tokenizer_name)
        and gguf_sidecar_dir(tokenizer_name, "tokenizer_config.json") is None
        and has_native_gguf_support(tokenizer_name)
    ):
        _ensure_gguf_version()
        tokenizer = build_gguf_tokenizer(tokenizer_name)
        _fix_special_tokens_pattern(tokenizer)
        attach_additional_stop_token_ids(tokenizer)
        return patch_tokenizer(tokenizer)

    tokenizer_name = _resolve_tokenizer_name(tokenizer_name, kwargs)

    common_kwargs = dict(
        trust_remote_code=trust_remote_code,
        tokenizer_revision=tokenizer_revision,
        clean_up_tokenization_spaces=False,
        **kwargs,
    )

    try:
        if is_bare_tekken_checkpoint(tokenizer_name, tokenizer_revision):
            from transformers.tokenization_mistral_common import (
                MistralCommonTokenizer,
            )

            logger.info(
                "Detected bare-tekken checkpoint %s (tekken.json, no "
                "tokenizer.json); loading via mistral-common MistralCommonTokenizer, "
                "ignoring tokenizer_backend=%r.",
                tokenizer_name,
                tokenizer_backend,
            )

            tokenizer = MistralCommonTokenizer.from_pretrained(
                tokenizer_name, revision=tokenizer_revision
            )
        else:
            tokenizer = _auto_tokenizer_from_pretrained(
                tokenizer_name, *args, **common_kwargs
            )


            if (
                type(tokenizer).__name__ == _TOKENIZERS_BACKEND
                and tokenizer_backend != "fastokens"
            ):
                tokenizer = _resolve_tokenizers_backend(
                    tokenizer_name, *args, **common_kwargs
                )

        return _apply_post_load_fixes(tokenizer, tokenizer_name, tokenizer_revision)
    except Exception as e:
        if tokenizer_backend == "fastokens":
            raise RuntimeError(
                f"fastokens failed to load tokenizer for {tokenizer_name!r}. "
                f"This model's tokenizer may not be supported by fastokens — "
                f"see https://github.com/crusoecloud/fastokens. "
                f"Re-run without --tokenizer-backend=fastokens to use the default backend."
            ) from e
        raise


def _fix_added_tokens_encoding(tokenizer):


    def _is_special_token_attr(val):
        return (
            isinstance(val, str)
            and val.startswith("<")
            and val.endswith(">")
            and len(val) <= 20
        )

    candidates = {}
    for attr in dir(tokenizer):
        if attr.startswith("_"):
            continue
        try:
            val = getattr(tokenizer, attr)
        except (AttributeError, TypeError, ValueError):
            continue
        if not _is_special_token_attr(val):
            continue
        token_id = tokenizer.convert_tokens_to_ids(val)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            candidates[val] = token_id

    if not candidates:
        return

    def _encodes_correctly(token_str, expected_id):
        try:
            ids = tokenizer.encode(token_str, add_special_tokens=False)
            return len(ids) == 1 and ids[0] == expected_id
        except (ValueError, OverflowError, RuntimeError) as e:
            logger.debug("Token %s encode check failed: %s", token_str, e)
            return False

    broken = [
        tok for tok, eid in candidates.items() if not _encodes_correctly(tok, eid)
    ]

    if not broken:
        return

    from transformers import AddedToken

    tokens_to_add = [AddedToken(tok, special=True, normalized=False) for tok in broken]
    tokenizer.add_tokens(tokens_to_add, special_tokens=True)
    logger.info(
        "Re-registered %d special tokens for correct v5 encoding: %s",
        len(broken),
        broken[:10],
    )
