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


import inspect

from sglang.srt.utils import logger

_applied = False


def apply_all():

    global _applied
    if _applied:
        return
    try:
        import transformers
    except ImportError:
        _applied = True
        return
    _applied = True


    _patch_flash_attn_availability()
    _patch_rope_parameters_validation()
    _patch_removed_symbols()
    _patch_image_processor_kwargs()
    _patch_image_process_cuda_tensor()
    _patch_nemotron_h_pattern()


    _ensure_clean_up_tokenization_compat()
    _ensure_is_torch_fx_available_compat()


    patch_is_base_mistral_in_ci()

    logger.debug("transformers compatibility patches applied")


def normalize_rope_scaling_compat(config) -> None:


    def _patch(cfg):
        rs = getattr(cfg, "rope_scaling", None)
        if isinstance(rs, dict) and "rope_type" in rs and "type" not in rs:
            rs["type"] = rs["rope_type"]

        for attr in (
            "text_config",
            "llm_config",
            "language_config",
            "vision_config",
            "thinker_config",
        ):
            sub = getattr(cfg, attr, None)
            if sub is not None:
                _patch(sub)

    _patch(config)


def _ensure_gguf_version():

    try:
        import gguf

        if not hasattr(gguf, "__version__"):
            import importlib.metadata

            try:
                gguf.__version__ = importlib.metadata.version("gguf")
            except importlib.metadata.PackageNotFoundError:
                gguf.__version__ = "0.0.0"
            except (ValueError, OSError, TypeError) as e:
                logger.warning(
                    "Failed to determine gguf package version: %s. "
                    "Falling back to '0.0.0'.",
                    e,
                )
                gguf.__version__ = "0.0.0"
    except ImportError:
        pass


def _patch_rope_parameters_validation():

    from transformers import PretrainedConfig


    if hasattr(PretrainedConfig, "standardize_rope_params"):
        _orig_standardize = PretrainedConfig.standardize_rope_params

        def _safe_standardize(self):
            if not hasattr(self, "max_position_embeddings"):
                return
            return _orig_standardize(self)

        PretrainedConfig.standardize_rope_params = _safe_standardize


def _patch_flash_attn_availability():

    try:
        import flash_attn as _fa

        if not hasattr(_fa, "flash_attn_func"):
            import transformers.utils as _u
            import transformers.utils.import_utils as _ui

            _ui.is_flash_attn_2_available = lambda: False
            _u.is_flash_attn_2_available = lambda: False
    except ImportError:
        pass


def _patch_removed_symbols():


    try:
        import logging


        _torchao_logger = logging.getLogger("torchao")
        _prev_level = _torchao_logger.level
        _torchao_logger.setLevel(logging.ERROR)
        try:
            from transformers.models.llama import modeling_llama
        finally:
            _torchao_logger.setLevel(_prev_level)

        if not hasattr(modeling_llama, "LlamaFlashAttention2"):
            if hasattr(modeling_llama, "LlamaAttention"):
                modeling_llama.LlamaFlashAttention2 = modeling_llama.LlamaAttention
    except ImportError:
        logger.warning(
            "Could not import transformers.models.llama.modeling_llama; "
            "LlamaFlashAttention2 compat patch not applied."
        )


    try:
        import transformers.utils as _u

        if not hasattr(_u, "is_flash_attn_greater_or_equal_2_10"):
            if hasattr(_u, "is_flash_attn_greater_or_equal"):
                _u.is_flash_attn_greater_or_equal_2_10 = (
                    lambda: _u.is_flash_attn_greater_or_equal("2.10.0")
                )
            else:
                _u.is_flash_attn_greater_or_equal_2_10 = lambda: False
    except ImportError:
        logger.warning(
            "Could not import transformers.utils; "
            "is_flash_attn_greater_or_equal_2_10 compat patch not applied."
        )


def _patch_image_processor_kwargs():

    try:
        from transformers.image_processing_utils import BaseImageProcessor

        original = BaseImageProcessor.__call__
        accepted_kwargs_cache = {}
        warned_unsupported_kwargs = set()

        def safe_call(self, images, *args, **kwargs):
            processor_type = type(self)
            accepted_kwargs = accepted_kwargs_cache.get(processor_type)
            if accepted_kwargs is None and processor_type not in accepted_kwargs_cache:
                sig = inspect.signature(self.preprocess)
                params = sig.parameters
                if any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
                ):
                    accepted_kwargs = None
                else:
                    accepted_kwargs = frozenset(params)
                accepted_kwargs_cache[processor_type] = accepted_kwargs

            if accepted_kwargs is None:
                return original(self, images, *args, **kwargs)

            dropped = frozenset(kwargs) - accepted_kwargs
            if dropped:
                warning_key = (processor_type, dropped)
                if warning_key not in warned_unsupported_kwargs:
                    logger.warning(
                        "Image processor %s.preprocess() does not accept %s; "
                        "filtering them before preprocessing. Update the model's image "
                        "processor to accept **kwargs.",
                        processor_type.__name__,
                        sorted(dropped),
                    )
                    warned_unsupported_kwargs.add(warning_key)
                kwargs = {k: v for k, v in kwargs.items() if k in accepted_kwargs}
            return original(self, images, *args, **kwargs)

        BaseImageProcessor.__call__ = safe_call
    except ImportError:
        logger.debug(
            "_patch_image_processor_kwargs: BaseImageProcessor not importable, patch skipped"
        )


def _patch_image_process_cuda_tensor():

    try:
        import torch
        import transformers.image_processing_backends as ipb

        for cls_name in ("PilBackend", "PilImageProcessingMixin"):
            cls = getattr(ipb, cls_name, None)
            if cls is None or not hasattr(cls, "process_image"):
                continue
            original = cls.process_image

            def patched_process_image(
                self, image, *args, _orig=original, _Tensor=torch.Tensor, **kwargs
            ):
                if isinstance(image, _Tensor) and image.is_cuda:
                    image = image.cpu()
                return _orig(self, image, *args, **kwargs)

            cls.process_image = patched_process_image
    except ImportError:
        logger.debug(
            "_patch_image_process_cuda_tensor: required modules not importable, patch skipped"
        )


def _patch_nemotron_h_pattern():

    try:
        from transformers.models.nemotron_h.configuration_nemotron_h import (
            NemotronHConfig,
        )

        @staticmethod
        def _pattern_to_list(pattern: str) -> list:
            pattern_mapping = {
                "M": "mamba",
                "E": "moe",
                "*": "attention",
            }
            return [
                pattern_mapping[char] for char in pattern if char in pattern_mapping
            ]

        NemotronHConfig._pattern_to_list = _pattern_to_list
    except ImportError:
        logger.debug(
            "_patch_nemotron_h_pattern: NemotronHConfig not importable, patch skipped"
        )


def _ensure_clean_up_tokenization_compat() -> None:

    from transformers import PreTrainedTokenizerBase

    if hasattr(PreTrainedTokenizerBase, "clean_up_tokenization"):
        return

    @staticmethod
    def clean_up_tokenization(out_string: str) -> str:
        out_string = (
            out_string.replace(" .", ".")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ,", ",")
            .replace(" ' ", "'")
            .replace(" n't", "n't")
            .replace(" 'm", "'m")
            .replace(" 's", "'s")
            .replace(" 've", "'ve")
            .replace(" 're", "'re")
        )
        return out_string

    PreTrainedTokenizerBase.clean_up_tokenization = clean_up_tokenization


def _ensure_is_torch_fx_available_compat() -> None:

    import transformers.utils.import_utils as _import_utils

    if hasattr(_import_utils, "is_torch_fx_available"):
        return

    _import_utils.is_torch_fx_available = lambda: True


_is_base_mistral_patched = False


def patch_is_base_mistral_in_ci():

    global _is_base_mistral_patched
    if _is_base_mistral_patched:
        return

    from sglang.srt.environ import envs

    if not envs.SGLANG_IS_IN_CI.get():
        return

    from transformers import PreTrainedTokenizerFast

    if hasattr(PreTrainedTokenizerFast, "_patch_mistral_regex"):

        @classmethod
        def _noop_patch_mistral_regex(cls, tokenizer, *args, **kwargs):
            return tokenizer

        PreTrainedTokenizerFast._patch_mistral_regex = _noop_patch_mistral_regex
        logger.info("CI: patched _patch_mistral_regex to skip HF API calls")

    _is_base_mistral_patched = True
