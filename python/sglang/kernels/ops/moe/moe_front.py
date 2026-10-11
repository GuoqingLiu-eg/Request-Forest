from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

import json
import os

NUM_EXPERTS = 896
TOPK = 16


MERGED_FRONT_MAX_TOKENS = 1024

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs", "moe_front")


DEFAULT_EPILOGUE_CONFIG = {"block_size": 224, "cast_vec": 8, "cast_first": False}

_tables = {}


def _table(kind: str, device_name: str):
    path = os.path.join(
        _CONFIG_DIR,
        f"{kind},E={NUM_EXPERTS},topk={TOPK},device_name={device_name}.json",
    )
    if path not in _tables:
        table = None
        if os.path.exists(path):
            with open(path) as f:
                table = {int(k): v for k, v in json.load(f)["configs"].items()}
        _tables[path] = table
    return _tables[path]


def _device_name(device) -> str:
    return torch.cuda.get_device_name(device).replace(" ", "_").replace("/", "_")


def get_config(kind: str, num_tokens: int, device, default: dict) -> dict:

    table = _table(kind, _device_name(device))
    if not table:
        return dict(default)
    pick = min(table)
    for k in sorted(table):
        if k <= num_tokens:
            pick = k
        else:
            break
    return dict(table[pick])


def get_front_strategy(num_tokens: int, device) -> str:

    table = _table("strategy", _device_name(device))
    if table is not None and num_tokens in table:
        return str(table[num_tokens])
    return "merged_fp32" if num_tokens <= MERGED_FRONT_MAX_TOKENS else "unfused"


@cache_once
def _jit_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "moe_front",
        *args,
        cuda_files=["moe/route_radix.cuh"],
        cuda_wrappers=[
            ("front_epilogue", f"FusedFrontEpilogueKernel<{args}>::run"),
        ],


        extra_cuda_cflags=["-O3"],
    )


@cache_once
def available() -> bool:
    import logging

    try:
        _jit_module()
        return True
    except Exception as e:
        logging.getLogger(__name__).warning(
            f"Failed to load the JIT MoE front kernels: {e}"
        )
        return False


def fused_front_covered(
    hidden_states: torch.Tensor,
    merged_weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    topk: int,
    latent: int,
) -> bool:

    return (
        hidden_states.dim() == 2
        and merged_weight.dim() == 2
        and hidden_states.dtype == torch.bfloat16
        and merged_weight.dtype == torch.bfloat16
        and bias is not None
        and bias.dtype == torch.float32
        and bias.numel() == NUM_EXPERTS
        and int(topk) == TOPK
        and merged_weight.shape[0] == NUM_EXPERTS + latent
        and merged_weight.shape[1] == hidden_states.shape[1]
        and latent % 4 == 0
        and 0 < hidden_states.shape[0] <= MERGED_FRONT_MAX_TOKENS
        and hidden_states.stride(1) == 1
        and merged_weight.stride(1) == 1
    )


def fused_front(
    hidden_states: torch.Tensor,
    merged_weight: torch.Tensor,
    correction_bias: torch.Tensor,
    latent: int,
    topk: int = TOPK,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    apply_routed_scaling_factor_on_output: bool = False,
    config: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    M = hidden_states.shape[0]
    device = hidden_states.device
    if config is None:
        config = get_config("epilogue", M, device, DEFAULT_EPILOGUE_CONFIG)


    merged = torch.mm(hidden_states, merged_weight.t(), out_dtype=torch.float32)

    weights = torch.empty((M, topk), dtype=torch.float32, device=device)
    ids = torch.empty((M, topk), dtype=torch.int32, device=device)
    routed = torch.empty((M, latent), dtype=torch.bfloat16, device=device)

    _jit_module().front_epilogue(
        merged,
        correction_bias,
        weights,
        ids,
        routed,
        topk,
        float(routed_scaling_factor if routed_scaling_factor is not None else 1.0),
        bool(renormalize),
        bool(apply_routed_scaling_factor_on_output),
        int(config["block_size"]),
        int(config["cast_vec"]),
        bool(config["cast_first"]),
    )
    return weights, ids, routed
