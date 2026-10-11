from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from transformers import PretrainedConfig

logger = logging.getLogger(__name__)


class ModelConfigParserBase(ABC):
    @abstractmethod
    def parse(
        self,
        model: str | Path,
        trust_remote_code: bool,
        revision: Optional[str] = None,
        **kwargs,
    ) -> PretrainedConfig:
        raise NotImplementedError


_MODEL_CONFIG_PARSER_REGISTRY: dict[str, type[ModelConfigParserBase]] = {}


def register_model_config_parser(name: str):


    def _wrapper(cls):
        if not issubclass(cls, ModelConfigParserBase):
            raise ValueError("Model-config parser must subclass ModelConfigParserBase.")
        if name in _MODEL_CONFIG_PARSER_REGISTRY:
            logger.warning(
                "Model-config parser %r already registered; overwriting with %s",
                name,
                cls,
            )
        _MODEL_CONFIG_PARSER_REGISTRY[name] = cls
        logger.debug("Registered model-config parser %r -> %s", name, cls.__name__)
        return cls

    return _wrapper


def get_model_config_parser(name: str) -> ModelConfigParserBase:

    if name not in _MODEL_CONFIG_PARSER_REGISTRY:
        raise ValueError(
            f"Unknown model-config parser {name!r}. "
            f"Registered: {sorted(_MODEL_CONFIG_PARSER_REGISTRY)}"
        )
    return _MODEL_CONFIG_PARSER_REGISTRY[name]()
