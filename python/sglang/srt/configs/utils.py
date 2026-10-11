from typing import Type

from transformers import (
    AutoImageProcessor,
    AutoProcessor,
    BaseImageProcessor,
    PretrainedConfig,
    ProcessorMixin,
)


def register_image_processor(
    config: Type[PretrainedConfig], image_processor: Type[BaseImageProcessor]
):

    AutoImageProcessor.register(
        config, slow_image_processor_class=image_processor, exist_ok=True
    )


def register_processor(config: Type[PretrainedConfig], processor: Type[ProcessorMixin]):

    AutoProcessor.register(config, processor, exist_ok=True)
