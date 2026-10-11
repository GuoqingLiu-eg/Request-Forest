from typing import Literal, TypeAlias

QUANT_CFG_CHOICES = {
    "fp8": "FP8_DEFAULT_CFG",
    "int4_awq": "INT4_AWQ_CFG",
    "w4a8_awq": "W4A8_AWQ_BETA_CFG",
    "nvfp4": "NVFP4_DEFAULT_CFG",
    "nvfp4_awq": "NVFP4_AWQ_LITE_CFG",
}


ModelOptQuantMethod: TypeAlias = Literal[
    "modelopt_fp8",
    "modelopt_fp4",
    "mxfp8",
]


_MODELOPT_QUANT_ALGO_TO_METHOD: dict[str, ModelOptQuantMethod] = {
    "FP8": "modelopt_fp8",
    "MXFP8": "mxfp8",
    "FP4": "modelopt_fp4",
    "NVFP4": "modelopt_fp4",
    "NVFP4_AWQ": "modelopt_fp4",
    "W4A16_NVFP4": "modelopt_fp4",
}


def canonicalize_modelopt_quant_algo(
    quant_algo: object,
) -> ModelOptQuantMethod | None:


    if not isinstance(quant_algo, str):
        return None
    return _MODELOPT_QUANT_ALGO_TO_METHOD.get(quant_algo.upper())
