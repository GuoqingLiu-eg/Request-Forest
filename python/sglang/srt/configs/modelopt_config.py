from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelOptConfig:


    quant: Optional[str] = None
    checkpoint_restore_path: Optional[str] = None
    checkpoint_save_path: Optional[str] = None
    export_path: Optional[str] = None
    quantize_and_serve: bool = False

    def __post_init__(self):


        pass
